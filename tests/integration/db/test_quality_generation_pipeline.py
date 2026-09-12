"""E4.6 production graph/Registry/DB integration with offline model adapters."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import func, select

from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Document,
    LLMInvocation,
    Run,
    RunJob,
    ToolInvocation,
)
from app.db.session import create_database_engine, create_session_factory
from app.llm.fake import FakeEmbeddingModel
from app.llm.ports import ModelToolCall, ModelUsage, ProviderAdapterError
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from tests.evals.live_chat import SECRET_CANARY
from tests.evals.quality_contracts import QualityGenerationReportV1
from tests.evals.quality_dataset import quality_digest
from tests.evals.quality_generation import generation_configuration_digest
from tests.evals.quality_generation_fixtures import generation_factory, generation_inputs
from tests.evals.quality_generation_support import QualityGenerationError
from tests.evals.quality_run import run_quality_generation

pytestmark = pytest.mark.integration


@pytest.fixture
async def sessions(migrated_database_url):
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture
def paths(tmp_path):
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    return tmp_path / "public", private_root


class ObservedEmbedding(FakeEmbeddingModel):
    def __init__(self, fault=None):
        self.fault = fault
        self.queries = 0
        self.ingestions = 0

    async def embed(self, texts, metadata, *, attempt):
        if metadata["graph_node"] == "document_retrieval":
            self.queries += 1
        else:
            self.ingestions += 1
            if self.fault == "partial_ingestion" and self.ingestions == 2:
                raise ProviderAdapterError(category="provider_rejected", retryable=False)
        result = await super().embed(texts, metadata, attempt=attempt)
        if self.fault == "token_usage":
            result = result.model_copy(update={"usage": ModelUsage(input_tokens=2)})
        return result


class ControlledChat(DeterministicResearchFakeChatAdapter):
    def __init__(self, fault=None, *, both_tools=False):
        self.fault, self.both_tools = fault, both_tools
        self.calls = []
        self.failed = False

    async def invoke(self, messages, tools, metadata, *, attempt):
        self.calls.append((tuple(messages), tuple(tools), dict(metadata)))
        if not self.failed:
            self.failed = True
            if self.fault == "provider":
                raise ProviderAdapterError(category="provider_rejected", retryable=False)
            if self.fault == "retry":
                raise ProviderAdapterError(category="rate_limited", retryable=True)
            if self.fault == "cancel":
                raise asyncio.CancelledError
            if self.fault == "body_error":
                raise ValueError("private-provider-error-canary")
            if self.fault == "secret":
                return self._result(content=SECRET_CANARY)
            if self.fault == "bad_output":
                return self._result(content="not valid model json")
        result = await super().invoke(messages, tools, metadata, attempt=attempt)
        if self.both_tools and metadata["graph_node"] == "research_agent" and result.tool_calls:
            existing = result.tool_calls[0]
            if existing.name == "retrieve_documents":
                return result.model_copy(
                    update={
                        "tool_calls": (
                            *result.tool_calls,
                            ModelToolCall(
                                call_id="web-" + existing.call_id,
                                name="search_web",
                                arguments={"query": "model rewritten query", "max_results": 8},
                            ),
                        )
                    }
                )
        return result


async def test_generation_persists_calls_and_private_evidence_without_actions(
    sessions, paths, monkeypatch
):
    # A fake run must not discover live providers even when provider/trace keys are present.
    for key in ("PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "PF_LANGFUSE_SECRET_KEY"):
        monkeypatch.setenv(key, "unusable-provider-secret-canary")

    async def no_http(*args, **kwargs):
        pytest.fail("fake generation attempted outbound HTTP")

    monkeypatch.setattr("httpx.AsyncClient.send", no_http)
    chat, embedding = ControlledChat(both_tools=True), ObservedEmbedding()
    args = generation_inputs(
        *paths,
        selected=["mixed_alpha", "scope_missing", "scope_rejected"],
        factory=generation_factory(chat, embedding),
    )
    report = await run_quality_generation(sessions, **args)
    assert report.measurement_complete and report.evidence_valid
    assert not report.annotation_complete and not report.semantic_quality_claim
    assert report.meets_quality_target is None
    assert [c.observation.status for c in report.cases] == ["succeeded", "succeeded", "failed"]
    assert report.cases[-1].scope_rejected
    assert report.cases[-1].observation.provider_attempts == 0
    assert embedding.queries == 1
    private = paths[1] / report.start.manifest.experiment_id
    for entry in report.private_files:
        data = (private / entry.name).read_bytes()
        assert quality_digest(data) == entry.digest
    first = json.loads((private / "case-0000.json").read_text())
    assert {t["tool_name"] for t in first["tools"]} == {"search_web", "retrieve_documents"}
    assert all(t["delivered"] for t in first["tools"])
    output = json.loads((private / "output-0000.json").read_text())
    assert output["output"]["application_draft"] is not None
    assert set(first["input"]) == {"mode", "query"}
    model_payloads = "\n".join(m.model_dump_json() for call in chat.calls for m in call[0])
    for forbidden in (
        "required_unit_ids",
        "expected_behavior",
        "rubric_version",
        "scope_expectation",
    ):
        assert forbidden not in model_payloads
    async with sessions() as session:
        invocations = (await session.scalars(select(LLMInvocation))).all()
        assert len(invocations) == report.total_usage.provider_attempts
        assert all(i.run_id for i in invocations if i.invocation_kind == "chat")
        assert await session.scalar(select(func.count()).select_from(ActionIntent)) == 0
        assert await session.scalar(select(func.count()).select_from(ApprovalRequest)) == 0
        assert await session.scalar(select(func.count()).select_from(Document)) == 3
        assert set(await session.scalars(select(RunJob.status))) == {"done"}
        assert set(await session.scalars(select(Run.status))) == {"completed"}
        assert set(await session.scalars(select(ToolInvocation.status))) == {"succeeded"}
    public = "\n".join(p.read_text() for p in paths[0].glob("*.json"))
    for secret in (
        args["dataset"].cases[0].query,
        "unusable-provider-secret-canary",
        "My synthetic background",
        str(paths[1]),
    ):
        assert secret not in public
    assert QualityGenerationReportV1.model_validate_json(report.model_dump_json()) == report
    # Digest tampering cannot silently retarget an E4.7 annotation.
    tampered = report.model_dump(mode="json")
    tampered["cases"][0]["observation"]["output_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        QualityGenerationReportV1.model_validate_json(json.dumps(tampered))


@pytest.mark.parametrize("fault", ["provider", "bad_output", "body_error", "retry"])
async def test_business_failures_continue_fixed_order_and_retry_is_accounted(
    sessions, paths, fault, capsys
):
    chat = ControlledChat(fault)
    args = generation_inputs(
        *paths, selected=["mixed_alpha", "mixed_beta"], factory=generation_factory(chat)
    )
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason is None and report.not_run == 0
    assert report.cases[-1].observation.status == "succeeded"
    assert report.cases[0].observation.status == ("succeeded" if fault == "retry" else "failed")
    if fault == "retry":
        assert (
            report.cases[0].observation.provider_attempts
            > report.cases[0].observation.model_calls + 1
        )
    else:
        assert report.failed == 1
    public = "\n".join(p.read_text() for p in paths[0].glob("*.json"))
    captured = capsys.readouterr()
    assert "private-provider-error-canary" not in public + captured.out + captured.err
    assert "not valid model json" not in public


@pytest.mark.parametrize("fault", ["cancel", "secret"])
async def test_cancel_and_safety_stop_preserve_remaining_slots(sessions, paths, fault):
    args = generation_inputs(
        *paths,
        selected=["mixed_alpha", "mixed_beta"],
        factory=generation_factory(ControlledChat(fault)),
    )
    if fault == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run_quality_generation(sessions, **args)
        report = QualityGenerationReportV1.model_validate_json(
            (paths[0] / "report.json").read_bytes()
        )
    else:
        report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == ("cancelled" if fault == "cancel" else "safety")
    assert report.cases[0].observation.status == "failed"
    assert report.cases[1].observation.status == "not_run"
    assert not report.measurement_complete
    if fault == "secret":
        assert not report.evidence_valid and report.cases[0].observation.safety.secret_leak
    for root in paths:
        for path in root.rglob("*.json"):
            assert SECRET_CANARY not in path.read_text()
    async with sessions() as session:
        assert set(await session.scalars(select(RunJob.status))) == {"done"}


@pytest.mark.parametrize(
    "field,value",
    [("provider_attempt_cap", 1), ("input_token_cap", 1), ("cost_admission_budget_cny", "0.001")],
)
async def test_global_admission_caps_stop_without_reroll(sessions, paths, field, value):
    # Default fake usage is zero; this adapter supplies measured tokens to cross the cap.
    factory = generation_factory(embedding=ObservedEmbedding("token_usage"))
    args = generation_inputs(
        *paths, selected=["mixed_alpha", "mixed_beta"], factory=factory, **{field: value}
    )
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == "budget" and report.not_run == 2
    if field == "cost_admission_budget_cny":
        assert report.total_usage.cost.unknown_cost_attempts == 1
        assert report.total_usage.cost.total_cost_cny is None
    if field == "input_token_cap":
        assert report.total_usage.input_tokens == 2
        assert report.total_usage.provider_attempts == 1
    assert len(report.cases) == 2
    with pytest.raises(QualityGenerationError, match="generation_preflight_failed"):
        await run_quality_generation(sessions, **args)
    assert (paths[0] / "report.json").exists()


@pytest.mark.parametrize("where", ["prepare", "finalize", "tool"])
async def test_accounting_failures_are_global_integrity_stops(sessions, paths, monkeypatch, where):
    async def fail(*args, **kwargs):
        raise RuntimeError("private-accounting-error-canary")

    if where == "tool":
        monkeypatch.setattr(
            "app.db.tool_invocations.SqlAlchemyToolInvocationRecorder.succeed", fail
        )
    else:
        monkeypatch.setattr(f"app.db.llm_invocations.SqlAlchemyInvocationRecorder.{where}", fail)
    args = generation_inputs(*paths, selected=["mixed_alpha", "mixed_beta"])
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == "integrity" and not report.evidence_valid
    assert report.cases[-1].observation.status == "not_run"
    assert "private-accounting-error-canary" not in (paths[0] / "report.json").read_text()


async def test_partial_ingestion_never_runs_generation(sessions, paths):
    chat = ControlledChat()
    args = generation_inputs(
        *paths,
        selected=["mixed_alpha"],
        factory=generation_factory(chat, ObservedEmbedding("partial_ingestion")),
    )
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == "provider"
    assert not report.representation_complete and report.not_run == 1
    assert len(report.representations) == 1 and not chat.calls
    assert report.total_usage.cost.unknown_cost_attempts > 0
    assert report.total_usage.cost.total_cost_cny is None


async def test_scope_leak_from_repository_stops_before_delivering_evidence(sessions, paths):
    from app.db.documents import SqlAlchemyDocumentRepository

    class LeakingRepository:
        def __init__(self):
            self.delegate = SqlAlchemyDocumentRepository(sessions)

        async def find_complete(self, **kwargs):
            return await self.delegate.find_complete(**kwargs)

        async def persist(self, representation):
            return await self.delegate.persist(representation)

        async def search(self, **kwargs):
            hits = await self.delegate.search(**kwargs)
            return tuple(replace(h, document_id=uuid4()) for h in hits)

    args = generation_inputs(*paths, selected=["mixed_alpha", "mixed_beta"])
    args["repository"] = LeakingRepository()
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == "safety"
    assert report.cases[0].observation.safety.unauthorized_access > 0
    assert report.cases[-1].observation.status == "not_run"
    assert not report.evidence_valid


async def test_explicit_live_gate_uses_offline_qwen_shaped_adapters(sessions, paths, monkeypatch):
    async def no_http(*args, **kwargs):
        pytest.fail("offline Qwen-shaped adapter attempted outbound HTTP")

    monkeypatch.setattr("httpx.AsyncClient.send", no_http)

    class QwenChat(ControlledChat):
        provider = "qwen"

        async def invoke(self, messages, tools, metadata, *, attempt):
            result = await super().invoke(messages, tools, metadata, attempt=attempt)
            return result.model_copy(update={"provider": self.provider})

    class QwenEmbedding(ObservedEmbedding):
        provider = "qwen"

        async def embed(self, texts, metadata, *, attempt):
            result = await super().embed(texts, metadata, attempt=attempt)
            return result.model_copy(update={"provider": self.provider})

    factory = replace(
        generation_factory(),
        provider="qwen",
        chat_adapter=QwenChat(),
        embedding_adapter=QwenEmbedding(),
    )
    args = generation_inputs(*paths, selected=["mixed_alpha"], factory=factory)
    args.update(provider_mode="qwen", confirm_live=False)
    with pytest.raises(QualityGenerationError, match="generation_preflight_failed"):
        await run_quality_generation(sessions, **args)
    assert not paths[0].exists()
    args["confirm_live"] = True
    report = await run_quality_generation(sessions, **args)
    assert report.measurement_complete
    assert report.cases[0].observation.status == "succeeded"
    assert report.start.manifest.document_mode == "real_embedding_db"
    assert report.start.manifest.measurement_scope == "generation"
    assert report.start.manifest.configuration_digest == generation_configuration_digest(
        args["policy"], factory
    )
    assert not report.semantic_quality_claim and not report.annotation_complete


async def test_generation_attempt_stop_is_inside_active_case_not_hidden_in_ingestion(
    sessions, paths
):
    # Pilot ingestion is three embedding attempts; permit only the first plan attempt next.
    args = generation_inputs(*paths, selected=["mixed_alpha", "mixed_beta"], provider_attempt_cap=4)
    report = await run_quality_generation(sessions, **args)
    assert report.stop_reason == "budget" and report.representation_complete
    assert report.total_usage.provider_attempts == 4
    assert report.ingestion_usage.provider_attempts == 3
    assert report.cases[0].observation.failure_type == "budget"
    assert report.cases[0].observation.model_calls == 1
    assert report.cases[1].observation.status == "not_run"


async def test_private_output_write_failure_stops_and_preserves_prior_evidence(
    sessions, paths, monkeypatch
):
    from tests.evals.quality_generation_support import GenerationArtifacts

    original = GenerationArtifacts.write

    def fail_output(self, name, payload, *, private=False):
        if name.startswith("output-"):
            raise QualityGenerationError("artifact_publication_failed")
        return original(self, name, payload, private=private)

    monkeypatch.setattr(GenerationArtifacts, "write", fail_output)
    report = await run_quality_generation(
        sessions, **generation_inputs(*paths, selected=["mixed_alpha", "mixed_beta"])
    )
    assert report.stop_reason == "integrity" and not report.evidence_valid
    assert report.cases[0].observation.output_digest is None
    assert report.cases[1].observation.status == "not_run"
    private = paths[1] / report.start.manifest.experiment_id
    assert (private / "sources.json").exists()
    assert (paths[0] / "report.json").exists()


async def test_new_experiment_keeps_prior_failure_without_overwriting_it(sessions, paths):
    first = generation_inputs(
        *paths, selected=["mixed_alpha"], factory=generation_factory(ControlledChat("provider"))
    )
    failed = await run_quality_generation(sessions, **first)
    old_report = (paths[0] / "report.json").read_bytes()
    second = generation_inputs(
        paths[0].parent / "public2",
        paths[1],
        selected=["mixed_alpha"],
        experiment_id="new_experiment",
    )
    succeeded = await run_quality_generation(sessions, **second)
    assert failed.failed == 1 and succeeded.failed == 0
    assert (paths[0] / "report.json").read_bytes() == old_report
    assert len(tuple(paths[1].iterdir())) == 2
