from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.config import Settings
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import Document, DocumentChunk, LLMInvocation
from app.db.session import create_database_engine, create_session_factory
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.ports import ProviderAdapterError
from app.llm.qwen_adapters import create_qwen_adapters
from tests.evals.live_baseline import probe_clean_git_head, verified_live_retrieval_candidate
from tests.evals.live_retrieval import (
    normalized_live_retrieval_report,
    run_live_retrieval_benchmark,
)
from tests.evals.live_retrieval_contracts import load_live_retrieval_dataset

pytestmark = pytest.mark.integration


class _RetryFirstRetrievalEmbedding:
    provider = "fake"
    model = "text-embedding-v4"

    def __init__(self) -> None:
        self.delegate = FakeEmbeddingModel()
        self.retrieval_attempts = 0

    async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
        if metadata.get("graph_node") == "document_retrieval":
            self.retrieval_attempts += 1
            if self.retrieval_attempts == 1:
                raise ProviderAdapterError(category="rate_limited", retryable=True)
        return await self.delegate.embed(texts, metadata, attempt=attempt)


class _PermanentRetrievalFailure(_RetryFirstRetrievalEmbedding):
    async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
        if metadata.get("graph_node") == "document_retrieval":
            self.retrieval_attempts += 1
            raise ProviderAdapterError(category="provider_rejected", retryable=False)
        return await self.delegate.embed(texts, metadata, attempt=attempt)


class _SearchSpyRepository:
    def __init__(self, delegate: SqlAlchemyDocumentRepository) -> None:
        self.delegate = delegate
        self.search_calls = 0

    async def find_complete(self, **kwargs):  # type: ignore[no-untyped-def]
        return await self.delegate.find_complete(**kwargs)

    async def persist(self, representation):  # type: ignore[no-untyped-def]
        return await self.delegate.persist(representation)

    async def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.search_calls += 1
        return await self.delegate.search(**kwargs)


async def _no_sleep(_seconds: float) -> None:
    return None


async def _table_count(sessions, table: str) -> int:  # type: ignore[no-untyped-def]
    async with sessions() as session:
        return int(await session.scalar(text(f'SELECT count(*) FROM "{table}"')) or 0)


def _all_field_names(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            field for nested in value.values() for field in _all_field_names(nested)
        }
    if isinstance(value, list):
        return {field for nested in value for field in _all_field_names(nested)}
    return set()


async def test_offline_live_runner_production_chain_retry_isolation_and_sanitization(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    adapter = _RetryFirstRetrievalEmbedding()
    spy = _SearchSpyRepository(SqlAlchemyDocumentRepository(sessions))
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=adapter,
        provider="fake",
        random_source=lambda: 0,
        sleeper=_no_sleep,
    )
    dataset = load_live_retrieval_dataset()
    try:
        report = await run_live_retrieval_benchmark(
            sessions,
            factory=factory,
            repository=spy,
            subject_namespace="gate-11-3-offline",
            confirm_disposable_database=True,
        )
        serialized = normalized_live_retrieval_report(report).decode()
        payload = json.loads(serialized)

        assert report.complete is True
        assert report.candidate_pool_size == 24
        assert len(report.cases) == 20
        assert spy.search_calls == 20
        assert adapter.retrieval_attempts == 21
        assert report.provider_attempt_count == 29
        assert report.aggregate.workspace_leakage_count == 0
        assert report.aggregate.allowlist_leakage_count == 0
        assert all(case.returned_context_bytes <= 4000 for case in report.cases)
        assert all(
            ref.document_alias not in {"same_workspace_decoy", "foreign_workspace_canary"}
            for case in report.cases
            for ref in case.ranked_retrieved_chunks
        )

        async with sessions() as session:
            document_count = await session.scalar(select(func.count()).select_from(Document))
            chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
            invocations = (
                await session.execute(
                    select(
                        LLMInvocation.status,
                        LLMInvocation.error_category,
                        LLMInvocation.invocation_kind,
                        LLMInvocation.graph_node,
                        LLMInvocation.run_id,
                    ).order_by(LLMInvocation.created_at, LLMInvocation.id)
                )
            ).all()
        assert document_count == 8
        assert chunk_count == 26
        assert len(invocations) == 29
        assert invocations[8].status == "failed"
        assert invocations[8].error_category == "rate_limited"
        assert invocations[9].status == "succeeded"
        assert all(row.invocation_kind == "embedding" and row.run_id is None for row in invocations)
        assert {row.graph_node for row in invocations} == {
            "document_ingestion",
            "document_retrieval",
        }

        forbidden_fields = {
            "query",
            "workspace_id",
            "actor_user_id",
            "document_id",
            "chunk_id",
            "database_url",
            "hostname",
            "port",
            "embedding",
        }
        assert _all_field_names(payload).isdisjoint(forbidden_fields)
        assert not re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            serialized,
        )
        assert all(query.query not in serialized for query in dataset.queries)

        for table in (
            "conversations",
            "messages",
            "runs",
            "run_jobs",
            "run_events",
            "action_intents",
            "approval_requests",
            "approval_decisions",
            "tool_invocations",
            "mock_submissions",
        ):
            assert await _table_count(sessions, table) == 0
        async with sessions() as session:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                if await session.scalar(text("SELECT to_regclass(:table)"), {"table": table}):
                    assert await session.scalar(text(f'SELECT count(*) FROM "{table}"')) == 0
    finally:
        await engine.dispose()


async def test_permanent_provider_failure_preserves_attempt_and_returns_partial_report(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    adapter = _PermanentRetrievalFailure()
    spy = _SearchSpyRepository(SqlAlchemyDocumentRepository(sessions))
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=adapter,
        provider="fake",
    )
    try:
        report = await run_live_retrieval_benchmark(
            sessions,
            factory=factory,
            repository=spy,
            subject_namespace="gate-11-3-permanent-failure",
            confirm_disposable_database=True,
        )
        assert report.complete is report.passed is False
        assert report.error_category == "provider_failure"
        assert report.cases == ()
        assert report.candidate_pool_size == 24
        assert report.provider_attempt_count == 9
        assert adapter.retrieval_attempts == 1
        assert spy.search_calls == 0
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(LLMInvocation.status, LLMInvocation.error_category).order_by(
                        LLMInvocation.created_at, LLMInvocation.id
                    )
                )
            ).all()
        assert rows[-1] == ("failed", "provider_rejected")
        assert sum(row.status == "succeeded" for row in rows) == 8
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    os.environ.get("PF_RUN_GATE11_LIVE_RETRIEVAL") != "1",
    reason="paid Gate 11.3 Qwen retrieval benchmark requires explicit operator opt-in",
)
async def test_real_qwen_embedding_retrieval_benchmark(
    migrated_database_url: str,
) -> None:
    settings = Settings(
        _env_file=None,
        database_url=SecretStr(migrated_database_url),
    )
    assert settings.llm_mode == "qwen"
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    bundle = create_qwen_adapters(
        api_key=settings.qwen_api_key,
        workspace_id=settings.qwen_workspace_id,
    )
    try:
        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=bundle.chat,
            embedding_adapter=bundle.embedding,
            provider="qwen",
        )
        report = await run_live_retrieval_benchmark(
            sessions,
            factory=factory,
            subject_namespace="gate-11-3-real-qwen",
            confirm_disposable_database=True,
        )
        print(normalized_live_retrieval_report(report).decode(), end="")
        assert report.complete is True
        assert report.passed is True
    finally:
        await bundle.aclose()
        await engine.dispose()


@pytest.mark.skipif(
    os.environ.get("PF_RUN_GATE11_LIVE_RETRIEVAL_ACCEPTED") != "1",
    reason="accepted Gate 11.4 retrieval requires explicit operator opt-in and clean commit",
)
async def test_accepted_qwen_embedding_retrieval_benchmark(
    migrated_database_url: str,
) -> None:
    report_path_value = os.environ.get("PF_GATE11_ACCEPTED_RETRIEVAL_REPORT")
    assert report_path_value
    report_path = Path(report_path_value)
    assert report_path.parent.is_dir() and not report_path.exists()
    start_commit_sha = probe_clean_git_head()
    settings = Settings(_env_file=None, database_url=SecretStr(migrated_database_url))
    assert settings.llm_mode == "qwen"
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    bundle = create_qwen_adapters(
        api_key=settings.qwen_api_key,
        workspace_id=settings.qwen_workspace_id,
    )
    try:
        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=bundle.chat,
            embedding_adapter=bundle.embedding,
            provider="qwen",
        )
        report = await run_live_retrieval_benchmark(
            sessions,
            factory=factory,
            subject_namespace="gate-11-4-accepted-qwen",
            confirm_disposable_database=True,
            accepted=True,
        )
        assert report.exploratory is False
        assert report.complete is True
        assert report.passed is True
        candidate = verified_live_retrieval_candidate(
            report=report,
            start_commit_sha=start_commit_sha,
            end_commit_sha=probe_clean_git_head(),
        )
        with report_path.open("xb") as report_file:
            report_file.write((candidate.model_dump_json(indent=2) + "\n").encode())
    finally:
        await bundle.aclose()
        await engine.dispose()
