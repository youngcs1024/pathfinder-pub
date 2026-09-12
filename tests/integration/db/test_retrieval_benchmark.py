from __future__ import annotations

import json
import re
from hashlib import sha256

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.db.models import Document, DocumentChunk, LLMInvocation
from app.db.session import create_database_engine, create_session_factory
from app.retrieval.chunking import (
    CHUNKING_VERSION,
    NORMALIZATION_VERSION,
    normalize_and_chunk_batch,
)
from app.retrieval.documents import EMBEDDING_PROFILE, RETRIEVAL_TOP_K
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.evals.retrieval_benchmark import (
    RetrievalBenchmarkConfigurationError,
    load_retrieval_dataset,
    normalized_retrieval_report,
    run_retrieval_benchmark,
)
from tests.evals.retrieval_regression import (
    DEFAULT_RETRIEVAL_POLICY_PATH,
    run_retrieval_regression,
)

pytestmark = pytest.mark.integration


async def _table_count(sessions, table: str) -> int:  # type: ignore[no-untyped-def]
    async with sessions() as session:
        return int(await session.scalar(text(f'SELECT count(*) FROM "{table}"')) or 0)


async def test_real_db_benchmark_pipeline_filters_accounting_and_determinism(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    dataset = load_retrieval_dataset()
    expected_chunk_count = 8
    try:
        first = await run_retrieval_benchmark(
            sessions,
            dataset=dataset,
            subject_namespace="gate-9-5-first",
            confirm_disposable_database=True,
        )
        second = await run_retrieval_benchmark(
            sessions,
            dataset=dataset,
            subject_namespace="gate-9-5-second",
            confirm_disposable_database=True,
        )
        first_bytes = normalized_retrieval_report(first)
        second_bytes = normalized_retrieval_report(second)

        assert first_bytes == second_bytes
        assert first.passed is True
        assert first.evidence_scope == "deterministic_fake_embedding_regression"
        assert first.semantic_quality_claim is False
        assert first.normalization_version == NORMALIZATION_VERSION
        assert first.chunking_version == CHUNKING_VERSION
        assert first.embedding_profile == EMBEDDING_PROFILE
        assert first.embedding_dimension == 1536
        assert first.retrieval_top_k == RETRIEVAL_TOP_K == 5
        assert len(first.cases) == 10
        assert (
            sum(
                len(
                    normalize_and_chunk_batch(
                        ValidatedIngestionBatch(
                            sources=(
                                ValidatedIngestionSource(
                                    source_name=document.source_name,
                                    source_type=document.source_type,
                                    title=document.title,
                                    raw_text=document.raw_text,
                                    character_count=len(document.raw_text),
                                ),
                            )
                        )
                    )
                    .sources[0]
                    .chunks
                )
                for document in dataset.documents
                if document.alias in dataset.cases[0].allowed_document_aliases
            )
            > 5
        )
        assert all(len(case.ranked_retrieved_chunks) == 5 for case in first.cases)
        assert all(len(case.ranked_retrieved_chunks) <= 5 for case in first.cases)
        assert (
            next(case for case in first.cases if case.case_id == "python_backend_stack").recall_at_5
            == 0
        )
        assert all(case.workspace_leakage_count == 0 for case in first.cases)
        assert all(case.allowlist_leakage_count == 0 for case in first.cases)
        assert first.aggregate.workspace_leakage_count == 0
        assert first.aggregate.allowlist_leakage_count == 0
        foreign_case = next(
            case for case in first.cases if case.case_id == "foreign_allowlist_canary"
        )
        assert all(
            reference.document_alias != "foreign_workspace_canary"
            for reference in foreign_case.ranked_retrieved_chunks
        )
        assert all(
            reference.document_alias != "same_workspace_decoy"
            for case in first.cases
            for reference in case.ranked_retrieved_chunks
        )

        serialized = first_bytes.decode()
        payload = json.loads(serialized)
        assert not re.search(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            serialized,
        )
        forbidden_fields = {
            "generated_at",
            "created_at",
            "document_id",
            "chunk_id",
            "workspace_id",
            "actor_user_id",
            "database_url",
            "hostname",
            "path",
            "port",
        }

        def field_names(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {
                    name for nested in value.values() for name in field_names(nested)
                }
            if isinstance(value, list):
                return {name for nested in value for name in field_names(nested)}
            return set()

        assert field_names(payload).isdisjoint(forbidden_fields)

        async with sessions() as session:
            document_count = await session.scalar(select(func.count()).select_from(Document))
            chunk_count = await session.scalar(select(func.count()).select_from(DocumentChunk))
            invocations = (
                await session.execute(
                    select(
                        LLMInvocation.workspace_id,
                        LLMInvocation.actor_user_id,
                        LLMInvocation.status,
                        LLMInvocation.invocation_kind,
                        LLMInvocation.graph_node,
                        LLMInvocation.run_id,
                    ).order_by(LLMInvocation.created_at)
                )
            ).all()
            profiles = (
                await session.execute(
                    select(Document.embedding_model, DocumentChunk.embedding_model)
                    .join(
                        DocumentChunk,
                        (DocumentChunk.workspace_id == Document.workspace_id)
                        & (DocumentChunk.document_id == Document.id),
                    )
                    .distinct()
                )
            ).all()

        assert document_count == 10
        assert chunk_count == expected_chunk_count * 2
        assert len(invocations) == 30
        assert all(row.status == "succeeded" for row in invocations)
        assert all(row.invocation_kind == "embedding" for row in invocations)
        assert all(row.run_id is None for row in invocations)
        assert {row.graph_node for row in invocations} == {
            "document_ingestion",
            "document_retrieval",
        }
        assert all(
            row.workspace_id is not None and row.actor_user_id is not None for row in invocations
        )
        invocation_scope_counts: dict[tuple[object, object], int] = {}
        for row in invocations:
            key = (row.workspace_id, row.actor_user_id)
            invocation_scope_counts[key] = invocation_scope_counts.get(key, 0) + 1
        assert sorted(invocation_scope_counts.values()) == [1, 1, 14, 14]
        assert profiles == [(EMBEDDING_PROFILE, EMBEDDING_PROFILE)]

        for table in (
            "runs",
            "run_jobs",
            "run_events",
            "action_intents",
            "approval_requests",
            "approval_decisions",
            "tool_invocations",
        ):
            assert await _table_count(sessions, table) == 0
        async with sessions() as session:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                relation = await session.scalar(
                    text("SELECT to_regclass(:table)"), {"table": table}
                )
                if relation is not None:
                    assert await session.scalar(text(f'SELECT count(*) FROM "{table}"')) == 0

        policy_before = DEFAULT_RETRIEVAL_POLICY_PATH.read_bytes()
        regression, regression_exit_code = run_retrieval_regression(current_report=first)
        assert regression_exit_code == 0
        assert regression.passed is True
        assert DEFAULT_RETRIEVAL_POLICY_PATH.read_bytes() == policy_before

        print(
            "RETRIEVAL BENCHMARK DETERMINISM "
            f"runs=2 bytes={len(first_bytes)} sha256={sha256(first_bytes).hexdigest()}"
        )
        print(first_bytes.decode(), end="")
        print(regression.model_dump_json(indent=2))
    finally:
        await engine.dispose()


async def test_nonexistent_relevant_ordinal_fails_closed_after_real_ingestion(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    dataset = load_retrieval_dataset()
    bad_ref = dataset.cases[0].relevant_chunks[0].model_copy(update={"ordinal": 99})
    bad_case = dataset.cases[0].model_copy(update={"relevant_chunks": (bad_ref,)})
    invalid_runtime_dataset = dataset.model_copy(update={"cases": (bad_case, *dataset.cases[1:])})
    try:
        with pytest.raises(RetrievalBenchmarkConfigurationError, match="ordinal"):
            await run_retrieval_benchmark(
                sessions,
                dataset=invalid_runtime_dataset,
                subject_namespace="gate-9-4-bad-ordinal",
                confirm_disposable_database=True,
            )
        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Document)) == 5
            assert await session.scalar(select(func.count()).select_from(DocumentChunk)) == 8
            rows = (
                await session.execute(select(LLMInvocation.status, LLMInvocation.graph_node))
            ).all()
        assert rows == [("succeeded", "document_ingestion")] * 5
        for table in (
            "runs",
            "run_jobs",
            "run_events",
            "action_intents",
            "approval_requests",
            "approval_decisions",
            "tool_invocations",
        ):
            assert await _table_count(sessions, table) == 0
        async with sessions() as session:
            for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
                relation = await session.scalar(
                    text("SELECT to_regclass(:table)"), {"table": table}
                )
                if relation is not None:
                    assert await session.scalar(text(f'SELECT count(*) FROM "{table}"')) == 0
    finally:
        await engine.dispose()
