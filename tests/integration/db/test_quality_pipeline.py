"""E4.5 real PostgreSQL and production services with offline controlled adapters."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import func, select

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import Document, LLMInvocation, Run
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory
from app.domain.provisioning import ProvisioningService
from app.llm.fake import FakeEmbeddingModel
from app.llm.ports import ModelUsage, ProviderAdapterError
from app.retrieval.documents import DocumentRetrievalInvariantError
from tests.evals import quality_run
from tests.evals.quality_contracts import QualityRetrievalReportV1
from tests.evals.quality_retrieval_support import fake_factory, retrieval_inputs
from tests.evals.quality_run import QualityRetrievalError, run_quality_retrieval

pytestmark = pytest.mark.integration


@pytest.fixture
async def sessions(migrated_database_url):
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


class ControlledEmbedding(FakeEmbeddingModel):
    def __init__(self, fault=None):
        self.fault = fault
        self.calls = []
        self.queries = 0
        self.ingestions = 0

    async def embed(self, texts, metadata, *, attempt):
        self.calls.append((tuple(texts), dict(metadata)))
        if metadata["graph_node"] == "document_retrieval":
            self.queries += 1
            if self.queries == 1:
                if self.fault == "retry":
                    raise ProviderAdapterError(category="rate_limited", retryable=True)
                if self.fault == "provider":
                    raise ProviderAdapterError(category="provider_rejected", retryable=False)
                if self.fault == "cancel":
                    raise asyncio.CancelledError
                if self.fault == "body_error":
                    raise ValueError("private_output_canary")
        else:
            self.ingestions += 1
            if self.fault == "ingestion" and self.ingestions == 2:
                raise ProviderAdapterError(category="provider_rejected", retryable=False)
        result = await super().embed(texts, metadata, attempt=attempt)
        if self.fault == "dimension" and self.queries:
            return result.model_copy(update={"vectors": ((1.0, 2.0),)})
        if self.fault == "tokens" and self.queries:
            return result.model_copy(update={"usage": ModelUsage(input_tokens=100_001)})
        return result


class RepositoryProbe:
    def __init__(self, sessions, fault=None):
        self.delegate = SqlAlchemyDocumentRepository(sessions)
        self.sessions = sessions
        self.fault = fault
        self.representations = []
        self.searches = 0
        self.foreign_id = None

    async def find_complete(self, **kwargs):
        return await self.delegate.find_complete(**kwargs)

    async def persist(self, representation):
        self.representations.append(representation)
        if self.fault == "partial" and len(self.representations) == 2:
            representation = replace(representation, chunks=representation.chunks[:-1])
        return await self.delegate.persist(representation)

    async def search(self, **kwargs):
        self.searches += 1
        assert len(kwargs["allowed_document_ids"]) == 1
        if self.fault == "profile":
            return await self.delegate.search(**{**kwargs, "embedding_model": "wrong-profile"})
        if self.fault == "filters" and self.foreign_id is None:
            foreign = await ProvisioningService(
                SqlAlchemyProvisioningStore(self.sessions)
            ).provision_personal_workspace("quality-foreign-control")
            representation = self.representations[0]
            self.foreign_id = await self.delegate.persist(
                replace(
                    representation,
                    identity=replace(representation.identity, workspace_id=foreign.workspace_id),
                    created_by_user_id=foreign.user_id,
                )
            )
            # Same vectors in another workspace must be excluded by SQL even if explicitly listed.
            widened = {
                **kwargs,
                "allowed_document_ids": (*kwargs["allowed_document_ids"], self.foreign_id),
            }
            filtered = await self.delegate.search(**widened)
            assert all(hit.document_id != self.foreign_id for hit in filtered)
            with pytest.raises(DocumentRetrievalInvariantError):
                await self.delegate.search(**{**kwargs, "embedding_model": "wrong-profile"})
        hits = await self.delegate.search(**kwargs)
        if self.fault == "scope":
            return (replace(hits[0], document_id=uuid4()),)
        if self.fault == "text":
            return (replace(hits[0], text="private_output_canary"),)
        return hits


async def test_real_pipeline_filters_maps_accounts_and_projects_without_gold(sessions, tmp_path):
    adapter = ControlledEmbedding()
    repository = RepositoryProbe(sessions, "filters")
    inputs = retrieval_inputs(tmp_path / "run", factory=fake_factory(adapter))
    report = await run_quality_retrieval(sessions, repository=repository, **inputs)
    assert report.measurement_complete and report.evidence_valid
    assert not report.semantic_quality_claim
    assert report.executed == 24 and report.failed == report.not_run == 0
    assert report.scope_leakage_count == 0
    assert [r.chunk_count for r in report.representations] == [8, 8, 7]
    assert report.ingestion_usage.provider_attempts == 3
    assert report.total_usage.provider_attempts == 3 + repository.searches
    assert report.total_usage.cost.total_cost_cny is None
    assert report.total_usage.cost.unknown_cost_attempts == report.total_usage.provider_attempts
    assert report.total_usage.accounting_complete
    assert repository.searches == 22
    cases = {case.case_id: case for case in inputs["dataset"].cases}
    for result in report.cases:
        case = cases[result.observation.case_id]
        assert all(ref.source_alias == case.resume_alias for ref in result.refs)
        assert result.metrics.returned_context_bytes <= 4000
        if case.resume_alias is None or case.scope_expectation == "reject":
            assert result.refs == ()
            assert result.observation.provider_attempts == 0
    query_payloads = [
        texts for texts, meta in adapter.calls if meta["graph_node"] == "document_retrieval"
    ]
    assert query_payloads == [
        (c.query,)
        for c in cases.values()
        if c.resume_alias is not None and c.scope_expectation == "allowed"
    ]
    assert all(set(meta) == {"graph_node"} for _, meta in adapter.calls)
    # DB is authoritative; unrelated control representations do not enter experiment counts.
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 4
        assert await session.scalar(select(func.count()).select_from(Run)) == 0
        attempts = (await session.scalars(select(LLMInvocation))).all()
        assert len(attempts) == report.total_usage.provider_attempts
        assert all(a.run_id is None and a.provider == "fake" for a in attempts)
        identities = [str(a.workspace_id) for a in attempts] + [str(a.id) for a in attempts]
    serialized = (inputs["output_dir"] / "report.json").read_text()
    assert QualityRetrievalReportV1.model_validate_json(serialized) == report
    assert all(case.query not in serialized for case in cases.values())
    assert all(identity not in serialized for identity in identities)
    assert "private_output_canary" not in serialized
    assert len(list(inputs["output_dir"].glob("case-*.json"))) == 24
    # A tampered summary or fake semantic claim must fail artifact validation.
    for update in ({"semantic_quality_claim": True}, {"executed": 0}, {"scope_leakage_count": 1}):
        with pytest.raises(ValidationError):
            QualityRetrievalReportV1.model_validate_json(
                json.dumps({**report.model_dump(mode="json"), **update})
            )


async def test_default_fake_ignores_live_environment_and_empty_scope_skips_provider(
    sessions, tmp_path, monkeypatch
):
    monkeypatch.setenv("PF_LLM_MODE", "qwen")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "private_output_canary")
    inputs = retrieval_inputs(tmp_path / "fake", selected=["scope_missing", "scope_rejected"])
    inputs.pop("factory")
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.measurement_complete
    assert report.total_usage.provider_attempts == report.ingestion_usage.provider_attempts == 3
    assert all(c.observation.provider_attempts == 0 for c in report.cases)
    assert report.manifest.llm_mode == "fake"


@pytest.mark.parametrize("fault", ["retry", "provider", "body_error"])
async def test_query_failure_retained_and_next_slot_runs_without_reroll(
    sessions, tmp_path, fault, capsys
):
    adapter = ControlledEmbedding(fault)
    inputs = retrieval_inputs(
        tmp_path / "run", factory=fake_factory(adapter), selected=["mixed_alpha", "mixed_beta"]
    )
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.measurement_complete
    assert report.stop_reason is None
    assert report.cases[1].observation.status == "succeeded"
    assert report.cases[0].observation.status == ("succeeded" if fault == "retry" else "failed")
    assert report.cases[0].observation.provider_attempts == (2 if fault == "retry" else 1)
    assert report.failed == (0 if fault == "retry" else 1)
    assert report.full_coverage_cases.unassessed_count == report.failed
    assert "private_output_canary" not in (inputs["output_dir"] / "report.json").read_text()
    captured = capsys.readouterr()
    assert "private_output_canary" not in captured.out + captured.err


@pytest.mark.parametrize("fault, category", [("dimension", "integrity"), ("tokens", "budget")])
async def test_invalid_vectors_and_token_stop_preserve_pending_slots(
    sessions, tmp_path, fault, category
):
    inputs = retrieval_inputs(
        tmp_path / "run",
        factory=fake_factory(ControlledEmbedding(fault)),
        selected=["mixed_alpha", "mixed_beta"],
    )
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.stop_reason == category
    assert report.cases[0].observation.status == "failed"
    assert report.cases[1].observation.status == "not_run"
    assert report.failed == report.not_run == 1
    assert not report.measurement_complete


@pytest.mark.parametrize("fault", ["ingestion", "partial"])
async def test_partial_ingestion_never_claims_complete_representation(sessions, tmp_path, fault):
    adapter = ControlledEmbedding(fault)
    inputs = retrieval_inputs(
        tmp_path / "run", factory=fake_factory(adapter), selected=["mixed_alpha"]
    )
    report = await run_quality_retrieval(
        sessions, repository=RepositoryProbe(sessions, fault), **inputs
    )
    assert not report.representation_complete
    assert not report.measurement_complete
    assert report.not_run == 1 and report.executed == 0
    assert len(report.representations) == 1
    assert adapter.queries == 0
    assert report.ingestion_usage.provider_attempts == 2
    assert (inputs["output_dir"] / "report.json").exists()
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == (
            1 if fault == "ingestion" else 2
        )


@pytest.mark.parametrize(
    "fault, category", [("scope", "safety"), ("text", "integrity"), ("profile", "integrity")]
)
async def test_repository_failures_stop_and_do_not_publish_untrusted_refs(
    sessions, tmp_path, fault, category
):
    inputs = retrieval_inputs(tmp_path / "run", selected=["mixed_alpha", "mixed_beta"])
    report = await run_quality_retrieval(
        sessions, repository=RepositoryProbe(sessions, fault), **inputs
    )
    assert report.stop_reason == category
    assert not report.evidence_valid
    assert report.scope_leakage_count == int(fault == "scope")
    assert report.cases[0].refs == ()
    assert report.not_run == 1
    assert "private_output_canary" not in (inputs["output_dir"] / "report.json").read_text()


@pytest.mark.parametrize("phase", ["prepare", "finalize"])
async def test_accounting_write_failure_stops_without_fabricating_success(
    sessions, tmp_path, monkeypatch, phase
):
    class BrokenRecorder(SqlAlchemyInvocationRecorder):
        async def prepare(self, attempt):
            if phase == "prepare" and attempt.graph_node == "document_retrieval":
                raise RuntimeError("private_output_canary")
            await super().prepare(attempt)

        async def finalize(self, attempt, outcome):
            if phase == "finalize" and attempt.graph_node == "document_retrieval":
                raise RuntimeError("private_output_canary")
            await super().finalize(attempt, outcome)

    monkeypatch.setattr(quality_run, "SqlAlchemyInvocationRecorder", BrokenRecorder)
    adapter = ControlledEmbedding()
    inputs = retrieval_inputs(
        tmp_path / "run", factory=fake_factory(adapter), selected=["mixed_alpha", "mixed_beta"]
    )
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.stop_reason == "integrity"
    assert not report.total_usage.accounting_complete
    assert not report.evidence_valid
    assert report.not_run == 1
    assert adapter.queries == int(phase == "finalize")
    assert report.cases[0].observation.output_digest is None


async def test_attempt_budget_blocks_before_next_provider_attempt(sessions, tmp_path):
    adapter = ControlledEmbedding()
    inputs = retrieval_inputs(
        tmp_path / "run",
        factory=fake_factory(adapter),
        selected=["mixed_alpha", "mixed_beta"],
        provider_attempt_cap=4,
    )
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.stop_reason == "budget"
    assert report.total_usage.provider_attempts == 4
    assert adapter.queries == 1
    assert report.cases[0].observation.status == "succeeded"
    assert report.cases[1].observation.failure_type == "budget"
    assert not report.measurement_complete


async def test_cancel_is_propagated_after_partial_report_is_created(sessions, tmp_path):
    inputs = retrieval_inputs(
        tmp_path / "run",
        factory=fake_factory(ControlledEmbedding("cancel")),
        selected=["mixed_alpha", "mixed_beta"],
    )
    with pytest.raises(asyncio.CancelledError):
        await run_quality_retrieval(sessions, **inputs)
    report = QualityRetrievalReportV1.model_validate_json(
        (inputs["output_dir"] / "report.json").read_bytes()
    )
    assert report.stop_reason == "cancelled"
    assert report.cases[0].observation.failure_type == "cancelled"
    assert report.not_run == 1
    assert report.total_usage.provider_attempts == 4


async def test_duplicate_output_directory_fails_before_new_invocations(sessions, tmp_path):
    inputs = retrieval_inputs(tmp_path / "run", selected=["scope_missing"])
    await run_quality_retrieval(sessions, **inputs)
    async with sessions() as session:
        before = await session.scalar(select(func.count()).select_from(LLMInvocation))
    with pytest.raises(QualityRetrievalError):
        await run_quality_retrieval(sessions, **inputs)
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(LLMInvocation)) == before


@pytest.mark.parametrize("retry", [False, True])
async def test_unknown_cost_reserve_and_retry_share_the_same_admission_cap(
    sessions, tmp_path, retry
):
    adapter = ControlledEmbedding("retry" if retry else None)
    inputs = retrieval_inputs(
        tmp_path / "run",
        factory=fake_factory(adapter),
        selected=["mixed_alpha", "mixed_beta"],
        cost_admission_budget_cny="0.004",
    )
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.stop_reason == "budget"
    assert report.total_usage.provider_attempts == 4
    assert report.total_usage.cost.unknown_cost_attempts == 4
    assert report.total_usage.cost.total_cost_cny is None
    assert adapter.queries == 1
    if retry:
        assert report.cases[0].observation.failure_type == "budget"
        assert report.cases[1].observation.status == "not_run"


async def test_explicit_live_path_uses_injected_offline_qwen_shaped_adapters(sessions, tmp_path):
    from app.llm.fake import FakeChatModel

    class QwenChat(FakeChatModel):
        provider = "qwen"

    class QwenEmbedding(ControlledEmbedding):
        provider = "qwen"

        async def embed(self, texts, metadata, *, attempt):
            result = await super().embed(texts, metadata, attempt=attempt)
            return result.model_copy(
                update={"provider": "qwen", "usage": ModelUsage(input_tokens=2)}
            )

    # This is an offline gate test, not a live measurement. No SDK or provider HTTP is used.
    adapter = QwenEmbedding()
    factory = replace(
        fake_factory(), provider="qwen", chat_adapter=QwenChat(), embedding_adapter=adapter
    )
    inputs = retrieval_inputs(tmp_path / "run", factory=factory, selected=["mixed_alpha"])
    inputs.update(provider_mode="qwen", confirm_live=True)
    report = await run_quality_retrieval(sessions, **inputs)
    assert report.measurement_complete and report.evidence_valid
    assert report.total_usage.provider_attempts == 4
    assert report.total_usage.cost.unknown_cost_attempts == 0
    assert report.total_usage.cost.total_cost_cny > 0
    assert report.meets_quality_target is None


async def test_used_experiment_namespace_cannot_reroll_after_ingestion_failure(sessions, tmp_path):
    adapter = ControlledEmbedding("ingestion")
    inputs = retrieval_inputs(
        tmp_path / "failed", factory=fake_factory(adapter), selected=["mixed_alpha"]
    )
    failed = await run_quality_retrieval(sessions, **inputs)
    assert not failed.representation_complete
    before = len(adapter.calls)
    inputs["output_dir"] = tmp_path / "reroll"
    with pytest.raises(QualityRetrievalError):
        await run_quality_retrieval(sessions, **inputs)
    assert not inputs["output_dir"].exists()
    assert len(adapter.calls) == before


async def test_non_disposable_database_is_rejected_before_mutation(postgres_url, tmp_path):
    engine = create_database_engine(SecretStr(postgres_url))
    try:
        inputs = retrieval_inputs(tmp_path / "out", selected=["mixed_alpha"])
        with pytest.raises(QualityRetrievalError):
            await run_quality_retrieval(create_session_factory(engine), **inputs)
        assert not inputs["output_dir"].exists()
    finally:
        await engine.dispose()
