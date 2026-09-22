"""Owned PostgreSQL generation, not a worker/checkpoint or live comparison acceptance."""

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text, update

from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    LLMInvocation,
    Run,
    RunJob,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.errors import DomainInvariantError
from app.domain.runs import CURRENT_GRAPH_VERSION, DEFAULT_RUN_LIMITS, RunMode
from app.domain.tenancy import TenantContext
from app.llm.ports import ProviderAdapterError
from app.worker.langgraph_executor import LangGraphRunExecutor
from tests.evals.live_chat import SECRET_CANARY
from tests.evals.quality_e7a_contracts import E7AResearchOutputV1, EvidenceContext
from tests.evals.quality_experiment_database import (
    GRAPH_VERSION,
    ExperimentDatabaseError,
    OwnedExperimentDatabase,
    require_owned_database,
)
from tests.evals.quality_generation_fixtures import (
    E7AFakeChatAdapter,
    generation_factory,
    generation_inputs,
)
from tests.evals.quality_generation_support import (
    ExperimentGenerationReportV1,
    QualityGenerationError,
)
from tests.evals.quality_run import (
    finish_quality_generation,
    prepare_quality_generation,
    run_quality_generation,
    run_quality_generation_slot,
)
from tests.legacy_runtime import SqlAlchemyRunStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def owned():
    manager = OwnedExperimentDatabase()
    async with manager as handle:
        yield handle
    assert not manager.cleanup_failed


def args_for(
    tmp_path, name="candidate", *, outcome="sufficient", selected=None, on_call=None, **changes
):
    root = tmp_path / "private"
    root.mkdir(mode=0o700, exist_ok=True)
    return generation_inputs(
        tmp_path / name,
        root,
        selected=selected or ["mixed_alpha"],
        arm="candidate",
        factory=generation_factory(E7AFakeChatAdapter(outcome, on_call)),
        experiment_id=name,
        **changes,
    )


def context_from(fixture, raw):
    output = E7AResearchOutputV1.model_validate_json(json.dumps(raw), strict=True)
    return EvidenceContext(
        fixture.request, output.sources, output.evidence, fixture.resume_document_id
    )


@pytest.mark.parametrize("outcome", ["sufficient", "partial", "insufficient", "conflicting"])
async def test_owned_candidate_persists_four_outcomes_and_real_accounting(owned, tmp_path, outcome):
    owner = require_owned_database(owned)
    args = args_for(tmp_path, outcome=outcome)
    state = await prepare_quality_generation(owned_database=owned, **args)
    result = await run_quality_generation_slot(state, 0)
    report = await finish_quality_generation(state)
    assert result.observation.status == "succeeded"
    assert report.measurement_complete and report.evidence_valid
    assert type(report) is ExperimentGenerationReportV1
    assert report.start.arm == "candidate" and not report.start.production_schema
    assert (
        report.start.assessment_prompt_digest
        and report.start.schema_digest == owner.identity["schema_digest"]
    )
    assert not report.semantic_quality_claim and not report.annotation_complete
    fixture = state.runs[0]
    async with owner._sessions() as session:
        row = await session.get(Run, fixture.run_id)
        raw = row.result_json
        assert row.graph_version == GRAPH_VERSION and row.status == "completed"
        assert row.limits_json == dict(DEFAULT_RUN_LIMITS)
        assert await session.scalar(text("SELECT to_regclass('public.checkpoints')")) is None
        assert raw["assessment"]["outcome"] == outcome
        assert (raw["application_draft"] is not None) == (outcome == "sufficient")
        assert await session.scalar(select(func.count()).select_from(ActionIntent)) == 0
        assert await session.scalar(select(func.count()).select_from(ApprovalRequest)) == 0
        assert set(await session.scalars(select(RunJob.status))) == {"done"}
        calls = list(await session.scalars(select(LLMInvocation)))
        assert len(calls) == report.total_usage.provider_attempts
        chat = [c for c in calls if c.invocation_kind == "chat"]
        assert all(c.run_id == fixture.run_id for c in chat)
        assert (
            len(chat) == result.observation.model_calls == state.usage_owners[0].usage.model_calls
        )
        assert {c.graph_node for c in chat} >= {
            "plan",
            "research_agent",
            "evidence_assessment",
            "write_report",
        }
        assert all(c.prompt_version for c in chat)
        tools = list(await session.scalars(select(ToolInvocation)))
        assert len(tools) == result.observation.tool_calls
        assert all(t.status == "succeeded" for t in tools)
    context = context_from(fixture, raw)
    output = await owner.read_result(state.tenant, fixture, context=context)
    private = json.loads((tmp_path / "private/candidate/output-0000.json").read_text())
    assert private["output"] == output.model_dump(mode="json") == raw
    # Terminal re-entry does not rewrite results or manufacture an action.
    await owner.finish_run(state.tenant, fixture, output, None, context=context)
    with pytest.raises(DomainInvariantError):
        await SqlAlchemyRunStore(owner._sessions).get_run(
            tenant=state.tenant, run_id=fixture.run_id
        )
    with pytest.raises(DomainInvariantError):
        await SqlAlchemyRunStore(owner._sessions).create_run(
            tenant=state.tenant,
            mode=RunMode.APPLICATION,
            request=fixture.request,
            resume_document_id=fixture.resume_document_id,
            limits=dict(DEFAULT_RUN_LIMITS),
            graph_version=GRAPH_VERSION,
        )
    # The real executor rejects the version before touching any injected dependency.
    executor = object.__new__(LangGraphRunExecutor)
    rejected = await executor._execute(fixture.run_id, state.tenant, GRAPH_VERSION)
    assert rejected.status.value == "failed"
    for payload in tmp_path.joinpath("candidate").glob("*.json"):
        assert SECRET_CANARY not in payload.read_text()
        assert fixture.request.query not in payload.read_text()


async def test_both_arms_share_fixture_slots_and_baseline_reader(owned, tmp_path):
    owner = require_owned_database(owned)
    candidate_args = args_for(tmp_path, selected=["mixed_alpha", "mixed_beta"])
    baseline_args = generation_inputs(
        tmp_path / "baseline",
        tmp_path / "private",
        selected=["mixed_alpha", "mixed_beta"],
        experiment_id="baseline",
    )
    candidate = await prepare_quality_generation(owned_database=owned, **candidate_args)
    baseline = await prepare_quality_generation(owned_database=owned, **baseline_args)
    assert candidate.start.schema_digest == baseline.start.schema_digest
    for state in (baseline, candidate):
        await run_quality_generation_slot(state, 0)
        with pytest.raises(QualityGenerationError, match="generation_slot_unavailable"):
            await run_quality_generation_slot(state, 0)
    token = object()
    owner.claim_slot(token)
    with pytest.raises(ExperimentDatabaseError, match="resource_busy"):
        await run_quality_generation_slot(candidate, 1)
    owner.release_slot(token)
    for state in (candidate, baseline):
        await run_quality_generation_slot(state, 1)
        report = await finish_quality_generation(state)
        assert report.measurement_complete
    old = await SqlAlchemyRunStore(owner._sessions).get_run(
        tenant=baseline.tenant, run_id=baseline.runs[0].run_id
    )
    assert old.graph_version == CURRENT_GRAPH_VERSION and old.result is not None
    assert owner.identity["schema_digest"] == baseline.start.schema_digest


async def test_tenant_reader_and_schema_drift_fail_closed(owned, tmp_path):
    owner = require_owned_database(owned)
    state = await prepare_quality_generation(owned_database=owned, **args_for(tmp_path))
    await run_quality_generation_slot(state, 0)
    fixture = state.runs[0]
    async with owner._sessions() as session:
        raw = (await session.get(Run, fixture.run_id)).result_json
    context = context_from(fixture, raw)
    foreign = TenantContext(uuid4(), state.tenant.actor_user_id, state.tenant.role)
    with pytest.raises(ExperimentDatabaseError, match="access_denied"):
        await owner.read_result(foreign, fixture, context=context)
    async with owner._sessions.begin() as session:
        await session.execute(
            update(Run)
            .where(Run.id == fixture.run_id)
            .values(result_json={**raw, "evidence_sufficient": False})
        )
    with pytest.raises(ExperimentDatabaseError, match="invalid_result"):
        await owner.read_result(state.tenant, fixture, context=context)
    async with owner._sessions.begin() as session:
        await session.execute(
            update(Run)
            .where(Run.id == fixture.run_id)
            .values(result_json=raw, graph_version=CURRENT_GRAPH_VERSION)
        )
    with pytest.raises(ExperimentDatabaseError, match="invalid_run"):
        await owner.read_result(state.tenant, fixture, context=context)
    async with owner._sessions.begin() as session:
        await session.execute(
            text("ALTER TABLE runs ADD CONSTRAINT e7a_test_drift CHECK (next_event_seq > 0)")
        )
    with pytest.raises(ExperimentDatabaseError, match="schema_drift"):
        await owner.verify()


@pytest.mark.parametrize("fault", ["write", "read", "model_accounting", "tool_accounting"])
async def test_persistence_and_accounting_failures_stop_without_success(
    owned, tmp_path, monkeypatch, fault
):
    owner = require_owned_database(owned)
    state = await prepare_quality_generation(
        owned_database=owned, **args_for(tmp_path, selected=["mixed_alpha", "mixed_beta"])
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("private-database-body-canary")

    target, method = {
        "write": (owner, "finish_run"),
        "read": (owner, "read_result"),
        "model_accounting": (SqlAlchemyInvocationRecorder, "prepare"),
        "tool_accounting": (SqlAlchemyToolInvocationRecorder, "succeed"),
    }[fault]
    monkeypatch.setattr(target, method, fail)
    result = await run_quality_generation_slot(state, 0)
    report = await finish_quality_generation(state)
    assert result.observation.status == "failed"
    assert report.stop_reason == "integrity" and not report.evidence_valid
    assert report.not_run == 1
    assert not (tmp_path / "private/candidate/output-0000.json").exists()
    assert all(
        "private-database-body-canary" not in p.read_text()
        for p in (tmp_path / "candidate").glob("*.json")
    )


async def test_provider_retry_and_fresh_factory_views_account_once_per_logical_call(
    owned, tmp_path
):
    attempts = 0

    async def retry(messages, tools, metadata, attempt):
        nonlocal attempts
        if metadata["graph_node"] == "evidence_assessment":
            attempts += 1
            if attempts == 1:
                raise ProviderAdapterError(category="provider_timeout", retryable=True)

    state = await prepare_quality_generation(
        owned_database=owned, **args_for(tmp_path, on_call=retry)
    )
    result = await run_quality_generation_slot(state, 0)
    report = await finish_quality_generation(state)
    assert report.measurement_complete and attempts == 2
    assert result.observation.model_calls == state.usage_owners[0].usage.model_calls
    assert result.observation.provider_attempts > result.observation.model_calls


async def test_revocation_before_next_call_cancels_owned_run_and_prevents_read(owned, tmp_path):
    owner = require_owned_database(owned)
    state = None

    async def revoke(messages, tools, metadata, attempt):
        if metadata["graph_node"] == "plan":
            async with owner._sessions.begin() as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(WorkspaceMembership.workspace_id == state.tenant.workspace_id)
                    .values(revoked_at=datetime.now(UTC))
                )

    state = await prepare_quality_generation(
        owned_database=owned, **args_for(tmp_path, on_call=revoke)
    )
    result = await run_quality_generation_slot(state, 0)
    assert result.observation.failure_type == "cancelled"
    with pytest.raises(asyncio.CancelledError):
        await finish_quality_generation(state)
    async with owner._sessions() as session:
        assert (await session.get(Run, state.runs[0].run_id)).status == "cancelled"
    with pytest.raises(ExperimentDatabaseError, match="access_denied"):
        await owner.read_result(state.tenant, state.runs[0])


async def test_budget_stops_during_ingestion_and_retains_unexecuted_slots(owned, tmp_path):
    report = await run_quality_generation(
        owned_database=owned,
        **args_for(tmp_path, selected=["mixed_alpha", "mixed_beta"], provider_attempt_cap=1),
    )
    assert report.stop_reason == "budget" and report.not_run == 2
    assert report.total_usage.provider_attempts == 1 and not report.measurement_complete


async def test_cancellation_during_result_commit_retains_slot_and_accounting(
    owned, tmp_path, monkeypatch
):
    owner = require_owned_database(owned)
    state = await prepare_quality_generation(owned_database=owned, **args_for(tmp_path))
    original = owner.finish_run
    attempted = False

    async def interrupt(*args, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            raise asyncio.CancelledError
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, "finish_run", interrupt)
    with pytest.raises(asyncio.CancelledError):
        await run_quality_generation_slot(state, 0)
    with pytest.raises(asyncio.CancelledError):
        await finish_quality_generation(state)
    report = ExperimentGenerationReportV1.model_validate_json(
        (tmp_path / "candidate/report.json").read_bytes()
    )
    assert report.failed == 1 and report.not_run == 0 and report.stop_reason == "cancelled"
    assert report.cases[0].observation.provider_attempts > 0
    async with owner._sessions() as session:
        assert (await session.get(Run, state.runs[0].run_id)).status == "cancelled"
