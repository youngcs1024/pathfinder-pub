from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

import tests.evals.live_chat as live
from app.domain.tenancy import TenantContext, WorkspaceRole
from app.llm.factory import LLMFactory
from app.llm.invocations import LLMInvocationContext, LLMInvocationOutcome
from app.llm.ports import ChatModelResult, ModelToolCall, ModelUsage, ProviderAdapterError
from app.tools.registry import ToolExecutionError
from tests.evals.harness import (
    _prepare_case,
    _research_script,
    _StrictMemoryInvocationRecorder,
    _writer_script,
    load_eval_dataset,
)
from tests.evals.live_contracts import AcceptedLiveEvalBaselineV5, aggregate_live_observations
from tests.evals.live_suite import LiveSuiteConfigurationError, fixed_case_scope, load_live_manifest


class ScriptedQwen:
    """Offline transport double; every invocation still goes through real Factory/graph."""

    provider = "qwen"
    model = "qwen3.6-flash-2026-04-16"

    def __init__(self, mutate=None, *, research_scripts=None):
        self.calls = []
        self.cases = {c.case_id: c for c in load_eval_dataset()}
        self.mutate = mutate
        self.research_scripts = research_scripts or {}

    async def invoke(self, messages, tools, metadata, *, attempt):
        logical = live.CURRENT_LOGICAL_CALL.get()
        self.calls.append((logical, attempt.invocation_id))
        case_id, node_count = re.split(r"_r[12]_", logical, maxsplit=1)
        node, count = node_count.rsplit("_", 1)
        case = self.cases[case_id]
        if node == "plan":
            result = ChatModelResult(
                content=json.dumps({"queries": list(case.plan_queries)}),
                usage=ModelUsage(input_tokens=10, output_tokens=5),
            )
        elif node == "write_report":
            result = _writer_script(case, _prepare_case(case))[int(count) - 1]
        else:
            script = self.research_scripts.get(case_id) or _research_script(case)
            result = script[int(count) - 1]
        result = result.model_copy(update={"provider": "qwen"})
        if self.mutate:
            result = await self.mutate(self, logical, result)
        return result


class NoEmbedding:
    provider = "qwen"
    model = "text-embedding-v4"

    async def embed(self, *args, **kwargs):
        raise AssertionError("chat runner must not embed")


def setup(adapter=None, delegate=None):
    recorder = live.LiveAttemptRecorder(
        delegate or _StrictMemoryInvocationRecorder(), load_live_manifest()
    )
    adapter = adapter or ScriptedQwen()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=adapter,
        embedding_adapter=NoEmbedding(),
        provider="qwen",
        random_source=lambda: 0.0,
    )
    context = LLMInvocationContext(workspace_id=uuid4(), actor_user_id=uuid4(), request_id=uuid4())
    return factory, recorder, context, adapter


async def run(parts, **kwargs):
    factory, recorder, context, _ = parts
    return await live.run_chat_suite(factory=factory, recorder=recorder, context=context, **kwargs)


@pytest.fixture(scope="module")
async def full():
    parts = setup()
    evidence = []
    result = await run(parts, execution_evidence=evidence)
    return parts, result, evidence


async def test_exact_matrix_production_graph_grader_and_aggregate(full):
    (factory, recorder, _, adapter), result, evidence = full
    assert result.stop_reason is None
    assert result.exit_code == 0
    report = result.report
    assert report.exploratory and report.complete
    assert len(report.observations) == 20
    assert len({(o.case_id, o.repeat_index) for o in report.observations}) == 20
    assert all(o.passed for o in report.observations)
    assert report.aggregate.provider_attempts == len(adapter.calls) == len(recorder.outcomes)
    assert recorder.attempts.keys() == recorder.outcomes.keys()
    assert report.aggregate == aggregate_live_observations(
        report.observations, reserve=Decimal(".10")
    )
    assert {a.graph_node for a in recorder.attempts.values()} == {
        "plan",
        "research_agent",
        "write_report",
    }
    assert all(o.provider_attempts for o in report.observations)
    assert all(p.execution_succeeded for e in evidence for p in e.proposals)
    assert all(a.invocation_kind == "chat" for a in recorder.attempts.values())
    assert factory.provider == "qwen"  # Transport double is never claimed as live evidence.


async def test_accepted_mode_is_fixed_before_the_scripted_provider_attempts():
    result = await run(setup(), exploratory=False)

    assert result.stop_reason is None
    assert result.report.exploratory is False
    AcceptedLiveEvalBaselineV5(report=result.report)


async def test_rewritten_query_keeps_fixed_slot():
    async def rewrite(adapter, logical, result):
        return result.model_copy(
            update={
                "tool_calls": tuple(
                    c.model_copy(
                        update={
                            "arguments": {
                                **c.arguments,
                                "query": "Entirely rewritten query unrelated to fixture wording",
                            }
                        }
                    )
                    for c in result.tool_calls
                )
            }
        )

    evidence = []
    result = await run(setup(ScriptedQwen(rewrite)), execution_evidence=evidence)
    assert result.exit_code == 0
    assert all(
        q == "Entirely rewritten query unrelated to fixture wording"
        for e in evidence
        for _, _, q in e.queries
    )
    assert "Entirely rewritten" not in result.report.model_dump_json()


@pytest.mark.parametrize("kind", ["invalid_search", "invalid_retrieval", "unknown", "duplicate"])
async def test_rejected_tool_proposals_never_fallback(kind):
    async def bad(adapter, logical, result):
        if logical != "normal_application_r1_research_agent_1":
            return result
        calls = result.tool_calls
        first = calls[0]
        if kind == "invalid_search":
            calls = (first.model_copy(update={"arguments": {"query": "x", "max_results": "8"}}),)
        elif kind == "invalid_retrieval":
            calls = (
                first.model_copy(
                    update={
                        "name": "retrieve_documents",
                        "arguments": {"query": "x", "workspace_id": "spoof"},
                    }
                ),
            )
        elif kind == "unknown":
            calls = (first.model_copy(update={"name": "unknown_secret_like_name"}),)
        elif kind == "duplicate":
            calls = (first, first)
        return result.model_copy(update={"tool_calls": calls})

    evidence = []
    result = await run(setup(ScriptedQwen(bad)), execution_evidence=evidence)
    first = result.report.observations[0]
    assert result.report.complete
    # Production converts the rejected research proposal to bounded completion; the
    # scripted writer then fails grounding. V3 preserves that primary graph outcome.
    assert first.execution_error == "invalid_output"
    assert first.graph_failure is not None and first.graph_failure.node_name == "write_report"
    assert not first.passed
    if kind.startswith("invalid") or kind == "unknown":
        assert not first.tool_observations[0].schema_valid
        assert not first.tool_observations[0].executed
        assert not evidence[0].queries
        assert first.tool_observations[0].disposition == (
            "rejected_disallowed" if kind == "unknown" else "rejected_invalid"
        )
    if kind == "duplicate":
        assert all(t.schema_valid and not t.executed for t in first.tool_observations)
        assert all(t.disposition == "rejected_duplicate" for t in first.tool_observations)
    assert "unknown_secret_like_name" not in result.report.model_dump_json()


async def test_malicious_schema_invalid_execution_is_hard_failure():
    class EvilRuntime:
        def model_tools(self):
            return ()

        def validate_call(self, call):
            raise live.ToolInputValidationError("invalid")

        async def execute(self, call):
            return "{}"

    evidence = live.ExecutionEvidence()
    tools = live.ObservedTools(EvilRuntime(), evidence)
    call = ModelToolCall(call_id="x", name="search_web", arguments={"bad": True})
    tools.propose((call,))
    await tools.execute(call)
    assert evidence.tools()[0].schema_valid is False
    assert evidence.hard["invalid_tool_argument_execution_count"] == 1


async def test_factory_retry_is_one_logical_two_terminal_attempts():
    async def retry(adapter, logical, result):
        if len(adapter.calls) == 1:
            raise ProviderAdapterError(category="rate_limited", retryable=True)
        return result

    parts = setup(ScriptedQwen(retry))
    result = await run(parts)
    attempts = result.report.observations[0].provider_attempts
    first, second = attempts[:2]
    assert result.exit_code == 0
    assert first.logical_call_id == second.logical_call_id
    assert first.invocation_id != second.invocation_id
    assert first.error_category == "rate_limited"
    assert second.error_category is None and second.known_cost_cny > 0
    assert parts[1].outcomes[first.invocation_id].status == "failed"
    assert parts[1].outcomes[second.invocation_id].status == "succeeded"
    assert (
        result.report.aggregate.provider_attempts == result.report.aggregate.logical_model_calls + 1
    )
    assert result.report.aggregate.unknown_cost_attempt_count == 1
    assert result.report.aggregate.cny_per_case is None
    assert result.report.aggregate.cny_per_passed_case is None


async def test_writer_malformed_retry_keeps_first_invalid():
    async def malformed(adapter, logical, result):
        if logical == "normal_application_r1_write_report_1":
            return result.model_copy(update={"content": "{broken"})
        return result

    result = await run(setup(ScriptedQwen(malformed)))
    first = result.report.observations[0]
    assert first.passed
    writer = [a for a in first.provider_attempts if "write_report" in a.logical_call_id]
    assert [a.structured_output_valid for a in writer] == [False, True]
    assert result.report.aggregate.structured_output_valid_rate < 1


async def test_plan_structural_repair_is_two_accounted_logical_calls():
    raw_canary = "raw-malformed-plan-canary"

    async def malformed(adapter, logical, result):
        if logical == "normal_application_r1_plan_1":
            return result.model_copy(update={"content": f"not-json-{raw_canary}"})
        return result

    parts = setup(ScriptedQwen(malformed))
    result = await run(parts)
    first = result.report.observations[0]

    assert first.passed
    plan = [a for a in first.provider_attempts if "_plan_" in a.logical_call_id]
    assert [a.logical_call_id for a in plan] == [
        "normal_application_r1_plan_1",
        "normal_application_r1_plan_2",
    ]
    assert [a.structured_output_valid for a in plan] == [False, True]
    assert all(a.error_category is None for a in plan)
    assert len({a.invocation_id for a in plan}) == 2
    assert raw_canary not in result.report.model_dump_json()
    assert len(parts[1].delegate.attempts) == len(parts[1].delegate.outcomes)


async def test_writer_empty_report_repair_is_recorded_and_grades_normally():
    async def empty_first(adapter, logical, result):
        if logical == "normal_application_r1_write_report_1":
            payload = json.loads(result.content)
            payload["summary"] = []
            payload["findings"] = []
            return result.model_copy(update={"content": json.dumps(payload)})
        return result

    result = await run(setup(ScriptedQwen(empty_first)))
    first = result.report.observations[0]
    writer = [a for a in first.provider_attempts if "write_report" in a.logical_call_id]

    assert first.passed and first.grader.structured_output_valid
    assert [a.structured_output_valid for a in writer] == [True, True]
    assert [a.logical_call_id for a in writer] == [
        "normal_application_r1_write_report_1",
        "normal_application_r1_write_report_2",
    ]


def seed(recorder, *, count, cost=None, inputs=1):
    # Historical terminal accounting at a boundary, without inventing new provider requests.
    from app.llm.invocations import LLMInvocationAttempt

    for number in range(count):
        key = uuid4()
        attempt = LLMInvocationAttempt(
            invocation_id=key,
            workspace_id=uuid4(),
            actor_user_id=uuid4(),
            invocation_kind="chat",
            provider="qwen",
            model="qwen3.6-flash-2026-04-16",
            graph_node="plan",
            prompt_version="sha256:" + "0" * 64,
            request_hash="sha256:" + "0" * 64,
        )
        recorder.attempts[key] = attempt
        recorder.logical_ids[key] = f"prior_{number}"
        recorder.structured[key] = True
        recorder.outcomes[key] = LLMInvocationOutcome(
            status="succeeded",
            latency_ms=1,
            token_usage=ModelUsage(input_tokens=inputs, output_tokens=1),
            pricing_version="test" if cost is not None else None,
            currency="CNY" if cost is not None else None,
            estimated_cost=cost,
        )


@pytest.mark.parametrize(
    "count,cost,reason",
    [
        (1, Decimal("2.91"), "budget_exhausted"),
        (160, Decimal("0"), "provider_cap_exceeded"),
        (30, None, "budget_exhausted"),
    ],
)
async def test_pre_attempt_admission_before_delegate_and_transport(count, cost, reason):
    parts = setup()
    seed(parts[1], count=count, cost=cost)
    result = await run(parts)
    assert result.stop_reason == reason and not result.report.complete
    assert not parts[3].calls
    assert not parts[1].delegate.attempts
    assert len(parts[1].attempts) == count
    assert result.report.aggregate.provider_attempts == 0
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "usage,reason",
    [
        (ModelUsage(input_tokens=500001, output_tokens=1), "token_cap_exceeded"),
        (ModelUsage(input_tokens=1, output_tokens=150001), "token_cap_exceeded"),
        (ModelUsage(input_tokens=400000, output_tokens=140000), "budget_exhausted"),
    ],
)
async def test_post_response_cap_retains_terminal_and_stops(usage, reason):
    async def excessive(adapter, logical, result):
        return result.model_copy(update={"usage": usage})

    parts = setup(ScriptedQwen(excessive))
    result = await run(parts)
    assert result.stop_reason == reason
    assert len(parts[3].calls) == 1
    assert len(parts[1].outcomes) == 1
    assert len(result.report.observations) == 1
    assert result.report.aggregate.provider_attempts == 1
    assert result.report.observations[0].execution_error == reason
    assert not result.report.complete


@pytest.mark.parametrize("phase", ["prepare", "finalize"])
async def test_accounting_fail_closed_without_retry(phase):
    class FaultRecorder(_StrictMemoryInvocationRecorder):
        async def prepare(self, attempt):
            if phase == "prepare":
                raise RuntimeError("secret diagnostic never rendered")
            await super().prepare(attempt)

        async def finalize(self, attempt, outcome):
            if phase == "finalize":
                raise RuntimeError("secret diagnostic never rendered")
            await super().finalize(attempt, outcome)

    parts = setup(delegate=FaultRecorder())
    result = await run(parts)
    assert result.stop_reason == f"accounting_{phase}_failure"
    assert len(parts[3].calls) == (phase == "finalize")
    assert not result.report.complete
    assert result.report.aggregate.provider_attempts == 0
    assert not result.report.observations[0].passed
    assert "secret diagnostic" not in result.report.model_dump_json()


async def test_cancellation_in_provider_terminalizes_and_stops():
    entered = asyncio.Event()
    barrier = asyncio.Event()

    async def pending(adapter, logical, result):
        entered.set()
        await barrier.wait()
        return result

    parts = setup(ScriptedQwen(pending))
    task = asyncio.create_task(run(parts))
    await entered.wait()
    task.cancel()
    result = await task
    assert result.stop_reason == "external_cancelled"
    assert len(parts[3].calls) == 1
    assert [o.error_category for o in parts[1].outcomes.values()] == ["cancelled"]
    assert result.report.aggregate.provider_attempts == 1
    assert not result.report.complete


async def test_cancellation_between_observations_does_not_start_next_call(monkeypatch):
    event = asyncio.Event()
    original = live.grade_live_output

    def grade(*args, **kwargs):
        event.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(live, "grade_live_output", grade)
    parts = setup()
    result = await run(parts, cancellation=event)
    assert result.stop_reason == "external_cancelled"
    assert len(result.report.observations) == 1
    assert all(logical.startswith("normal_application_r1_") for logical, _ in parts[3].calls)


@pytest.mark.parametrize("phase", ["start", "finish"])
async def test_trace_export_failure_does_not_change_quality_or_accounting(phase, full):
    from app.llm.invocations import TraceIdentifiers

    class FaultTrace:
        def start(self, event):
            if phase == "start":
                raise RuntimeError("trace secret")
            return TraceIdentifiers(trace_id="1" * 32, observation_id="2" * 16)

        def finish(self, *args, **kwargs):
            raise RuntimeError("trace secret")

    factory, recorder, context, adapter = setup()
    result = await run((replace(factory, trace_sink=FaultTrace()), recorder, context, adapter))
    baseline = full[1]
    assert result.exit_code == baseline.exit_code == 0
    assert result.report.aggregate.provider_attempts == baseline.report.aggregate.provider_attempts
    assert result.report.aggregate.known_cost_cny == baseline.report.aggregate.known_cost_cny
    assert len(recorder.outcomes) == len(adapter.calls)


async def test_unresolved_fixture_stops_before_any_provider(monkeypatch):
    def missing(*args, **kwargs):
        raise LiveSuiteConfigurationError("missing fixture")

    monkeypatch.setattr(live, "resolve_fixed_evidence", missing)
    parts = setup()
    result = await run(parts)
    assert not result.report.complete and result.exit_code == 2
    assert not parts[3].calls


async def test_document_scope_rejects_foreign_tenant_and_allowlist():
    case = next(c for c in load_eval_dataset() if c.case_id == "document_only_resume")
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    fixed = live.FixedEvidence(case, tenant, live.ExecutionEvidence())
    with fixed_case_scope(case):
        with pytest.raises(LiveSuiteConfigurationError):
            await fixed.retrieve(tenant=tenant, query="x", allowed_document_ids=(uuid4(),))
        with pytest.raises(LiveSuiteConfigurationError):
            await fixed.retrieve(
                tenant=TenantContext(uuid4(), tenant.actor_user_id, WorkspaceRole.ADMIN),
                query="x",
                allowed_document_ids=(case.resume_document_id,),
            )
    assert not fixed.evidence.queries


async def test_safety_repeat_failure_rejected_despite_overall_rate(full):
    report = full[1].report
    observations = tuple(
        o.model_copy(update={"execution_error": "invalid_output"})
        if o.case_id == "prompt_injection" and o.repeat_index == 1
        else o
        for o in report.observations
    )
    report = report.model_copy(
        update={
            "exploratory": False,
            "observations": observations,
            "aggregate": aggregate_live_observations(observations, reserve=Decimal(".10")),
        }
    )
    assert report.aggregate.case_pass_rate == 0.95
    with pytest.raises(ValidationError, match="safety"):
        AcceptedLiveEvalBaselineV5(report=report)


def test_finite_canaries_detect_leaks():
    evidence = live.ExecutionEvidence()
    evidence.scan(
        " ".join(
            (
                live.SECRET_CANARY,
                live.TRUSTED_CONTEXT_CANARY,
                live.FOREIGN_WORKSPACE_CANARY,
                "synthetic policy",
            )
        ),
        system_prompts=("synthetic policy",),
    )
    assert all(
        evidence.hard[k] == 1
        for k in (
            "secret_leakage_count",
            "policy_leakage_count",
            "trusted_context_leakage_count",
            "cross_workspace_leakage_count",
        )
    )


async def test_retry_attempt_is_admitted_again_before_prepare():
    async def rate_limited(adapter, logical, result):
        raise ProviderAdapterError(category="rate_limited", retryable=True)

    parts = setup(ScriptedQwen(rate_limited))
    seed(parts[1], count=1, cost=Decimal("2.81"))
    result = await run(parts)
    assert result.stop_reason == "budget_exhausted"
    assert len(parts[3].calls) == len(parts[1].delegate.attempts) == 1
    assert len(parts[1].delegate.outcomes) == 1
    assert result.report.aggregate.provider_attempts == 1
    assert result.report.observations[0].provider_attempts[0].error_category == "rate_limited"


async def test_success_with_unknown_cost_reserves_budget():
    async def cached(adapter, logical, result):
        if len(adapter.calls) == 1:
            return result.model_copy(
                update={
                    "usage": ModelUsage(input_tokens=10, output_tokens=1, cached_input_tokens=1)
                }
            )
        return result

    result = await run(setup(ScriptedQwen(cached)))
    aggregate = result.report.aggregate
    assert result.exit_code == 0
    assert aggregate.unknown_cost_attempt_count == 1
    assert aggregate.admission_consumed_cny == aggregate.known_cost_cny + Decimal(".10")
    assert aggregate.cny_per_case is None and aggregate.cny_per_passed_case is None


@pytest.mark.parametrize(
    "category", ["provider_failure", "invalid_output", "graph_execution_failed"]
)
async def test_case_local_errors_preserve_twenty_pairs(category, monkeypatch):
    async def fail(adapter, logical, result):
        if logical.startswith("normal_application_r1_plan_"):
            if category == "provider_failure":
                if logical != "normal_application_r1_plan_1":
                    return result
                raise ProviderAdapterError(category="provider_authentication", retryable=False)
            return result.model_copy(update={"content": "{bad"})
        return result

    if category == "graph_execution_failed":
        original = live.StructuredResearchPlanNode

        class BrokenPlan:
            def __init__(self, model):
                self.delegate = original(model)

            async def __call__(self, node_input, control):
                await self.delegate(node_input, control)
                raise RuntimeError("graph failure")

        monkeypatch.setattr(live, "StructuredResearchPlanNode", BrokenPlan)
    result = await run(setup(ScriptedQwen(fail if category != "graph_execution_failed" else None)))
    assert result.stop_reason is None
    assert result.report.complete and len(result.report.observations) == 20
    assert result.report.observations[0].execution_error == category


async def test_cancellation_during_research_survives_empty_context():
    entered, barrier = asyncio.Event(), asyncio.Event()

    async def pending(adapter, logical, result):
        if logical.endswith("research_agent_1"):
            entered.set()
            await barrier.wait()
        return result

    parts = setup(ScriptedQwen(pending))
    task = asyncio.create_task(run(parts))
    await entered.wait()
    task.cancel()
    result = await task
    assert result.stop_reason == "external_cancelled"
    assert len(parts[3].calls) == 2
    assert result.report.observations[0].provider_attempts[-1].error_category == "cancelled"


async def test_last_pair_cap_is_incomplete_even_with_twenty_pairs():
    async def cap(adapter, logical, result):
        if logical == "application_resume_draft_r2_write_report_1":
            return result.model_copy(
                update={"usage": ModelUsage(input_tokens=500001, output_tokens=1)}
            )
        return result

    result = await run(setup(ScriptedQwen(cap)))
    assert len(result.report.observations) == 20
    assert not result.report.complete and result.exit_code == 2
    assert result.stop_reason == "token_cap_exceeded"


async def test_live_entrypoint_uses_sql_recorder_synthetic_provisioning_and_optional_trace(
    monkeypatch,
):
    from types import SimpleNamespace

    from app.config import Settings
    from app.llm.invocations import NoOpTraceSink

    for key, value in {
        "PF_LLM_MODE": "qwen",
        "PF_SEARCH_MODE": "fake",
        "PF_QWEN_WORKSPACE_ID": "synthetic-workspace",
        "DASHSCOPE_API_KEY": "synthetic-key",
        "PF_TRACE_MODE": "off",
        "PF_AUTH_MODE": "fake",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(live, "Settings", lambda **kwargs: Settings(_env_file=None))
    calls = []

    class Engine:
        async def dispose(self):
            calls.append("dispose")

    class Probe:
        def __init__(self, engine):
            pass

        async def is_ready(self):
            return True

    class Provision:
        def __init__(self, store):
            pass

        async def provision_personal_workspace(self, subject):
            assert subject == "pathfinder-gate11-chat-synthetic"
            return SimpleNamespace(workspace_id=uuid4(), user_id=uuid4())

    delegate = _StrictMemoryInvocationRecorder()

    def sql_recorder(sessions):
        calls.append("sql_recorder")
        return delegate

    class Bundle:
        chat = ScriptedQwen()
        embedding = NoEmbedding()

        async def aclose(self):
            calls.append("close")

    monkeypatch.setattr(live, "create_database_engine", lambda _: Engine())
    monkeypatch.setattr(live, "DatabaseReadinessProbe", Probe)
    monkeypatch.setattr(live, "create_session_factory", lambda _: object())
    monkeypatch.setattr(live, "SqlAlchemyProvisioningStore", lambda _: object())
    monkeypatch.setattr(live, "ProvisioningService", Provision)
    monkeypatch.setattr(live, "SqlAlchemyInvocationRecorder", sql_recorder)
    monkeypatch.setattr(live, "create_qwen_adapters", lambda **kwargs: Bundle())
    monkeypatch.setattr(live, "build_trace_sink", lambda _: NoOpTraceSink())
    result = await live.run_live_chat()
    assert result.exit_code == 0
    assert calls == ["sql_recorder", "close", "dispose"]
    assert result.report.aggregate.provider_attempts == len(delegate.outcomes)


async def test_live_reports_and_trace_do_not_contain_internal_queries_or_canaries(capsys, caplog):
    from app.llm.invocations import NoOpTraceSink

    class InspectTrace(NoOpTraceSink):
        def start(self, event):
            assert "query-private-marker" not in event.model_dump_json()
            assert live.SECRET_CANARY not in event.model_dump_json()
            return None

    async def rewritten(adapter, logical, result):
        return result.model_copy(
            update={
                "tool_calls": tuple(
                    c.model_copy(
                        update={"arguments": {**c.arguments, "query": "query-private-marker"}}
                    )
                    for c in result.tool_calls
                )
            }
        )

    factory, recorder, context, adapter = setup(ScriptedQwen(rewritten))
    result = await run((replace(factory, trace_sink=InspectTrace()), recorder, context, adapter))
    assert result.exit_code == 0
    rendered = result.report.model_dump_json() + capsys.readouterr().out + caplog.text
    assert "query-private-marker" not in rendered
    assert live.SECRET_CANARY not in rendered
    assert live.TRUSTED_CONTEXT_CANARY not in rendered


async def test_mid_suite_fixture_failure_preserves_previous_observation(monkeypatch):
    original = live.resolve_fixed_evidence

    def missing_on_second_repeat(*args, **kwargs):
        # Preflight has no logical call context. Fixed ports receive generated queries.
        if kwargs["query"] == "second-repeat-fixture-missing":
            raise LiveSuiteConfigurationError("declared fixture missing")
        return original(*args, **kwargs)

    async def query(adapter, logical, result):
        if logical == "normal_application_r2_research_agent_1":
            return result.model_copy(
                update={
                    "tool_calls": tuple(
                        c.model_copy(
                            update={
                                "arguments": {
                                    **c.arguments,
                                    "query": "second-repeat-fixture-missing",
                                }
                            }
                        )
                        for c in result.tool_calls
                    )
                }
            )
        return result

    monkeypatch.setattr(live, "resolve_fixed_evidence", missing_on_second_repeat)
    parts = setup(ScriptedQwen(query))
    result = await run(parts)
    assert result.stop_reason == "unresolved_fixture"
    assert len(result.report.observations) == 2
    assert result.report.observations[0].passed
    assert not result.report.observations[1].passed
    assert not result.report.complete
    assert result.report.aggregate.provider_attempts == len(parts[1].outcomes)


async def test_invalid_composition_and_identity_stop_before_provider():
    factory, recorder, context, adapter = setup()
    result = await live.run_chat_suite(
        factory=factory, recorder=recorder, context=replace(context, run_id=uuid4())
    )
    assert result.stop_reason == "invalid_configuration" and not adapter.calls
    recorder.manifest = recorder.manifest.model_copy(
        update={"cost_admission_budget_cny": Decimal("10")}
    )
    result = await live.run_chat_suite(factory=factory, recorder=recorder, context=context)
    assert result.stop_reason == "identity_mismatch" and not adapter.calls


async def test_pre_cancelled_suite_starts_no_attempt():
    parts = setup()
    event = asyncio.Event()
    event.set()
    result = await run(parts, cancellation=event)
    assert result.stop_reason == "external_cancelled"
    assert not parts[3].calls and not parts[1].attempts
    assert not result.report.observations


@pytest.mark.parametrize("stage", ["composition", "grader"])
async def test_harness_configuration_failure_is_global(stage, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("configuration detail must not escape")

    monkeypatch.setattr(
        live,
        "create_research_tool_registry" if stage == "composition" else "grade_live_output",
        broken,
    )
    parts = setup()
    result = await run(parts)
    assert result.stop_reason == "invalid_configuration"
    assert len(result.report.observations) == 1 and not result.report.complete
    assert not result.report.observations[0].passed
    if stage == "composition":
        assert not parts[3].calls
    else:
        assert result.report.aggregate.provider_attempts == len(parts[1].outcomes)


async def test_model_retrieval_without_document_scope_is_case_local():
    async def retrieve(adapter, logical, result):
        if logical == "no_document_scope_r1_research_agent_1":
            return result.model_copy(
                update={
                    "tool_calls": (
                        ModelToolCall(
                            call_id="undeclared_retrieve",
                            name="retrieve_documents",
                            arguments={"query": "resume"},
                        ),
                    )
                }
            )
        return result

    evidence = []
    result = await run(setup(ScriptedQwen(retrieve)), execution_evidence=evidence)
    assert result.stop_reason is None and result.report.complete
    observation = next(
        o
        for o in result.report.observations
        if o.case_id == "no_document_scope" and o.repeat_index == 1
    )
    # Current production Registry permits retrieval with an empty authorized scope.
    assert observation.execution_error != "tool_behavior_failure"
    assert observation.tool_observations[0].schema_valid
    assert observation.tool_observations[0].executed
    assert observation.tool_observations[0].evidence_resolution == "deterministic_empty"
    assert not observation.grader.required_evidence_coverage
    assert not observation.passed


def _search_proposal(call_id: str, *, batch_size: int = 1) -> ChatModelResult:
    return ChatModelResult(
        tool_calls=tuple(
            ModelToolCall(
                call_id=f"{call_id}-{index}",
                name="search_web",
                arguments={"query": "private-budget-query", "max_results": 8},
            )
            for index in range(batch_size)
        ),
        usage=ModelUsage(input_tokens=1, output_tokens=1),
    )


@pytest.mark.parametrize(
    ("executed_before", "final_batch_size", "suppressed_count"),
    [(8, 1, 1), (7, 2, 2)],
)
async def test_production_batch_budget_suppression_is_a_consistent_lifecycle(
    executed_before: int, final_batch_size: int, suppressed_count: int
):
    scripts = {
        "normal_application": (
            *tuple(_search_proposal(f"executed-{index}") for index in range(executed_before)),
            _search_proposal("suppressed", batch_size=final_batch_size),
        )
    }
    evidence = []
    result = await run(setup(ScriptedQwen(research_scripts=scripts)), execution_evidence=evidence)
    observation = result.report.observations[0]
    tools = observation.tool_observations
    assert observation.execution_error is None and observation.passed
    assert sum(tool.disposition == "executed" for tool in tools) == executed_before
    assert sum(tool.disposition == "budget_suppressed" for tool in tools) == suppressed_count
    assert observation.budget_suppressed_proposal_count == suppressed_count
    assert all(
        tool.evidence_resolution == "none" and tool.exposed_result_count == 0
        for tool in tools[-suppressed_count:]
    )
    assert len(evidence[0].exposed_evidence) == 2
    assert observation.grader.tool_execution_consistent
    assert "private-budget-query" not in observation.model_dump_json()


async def test_budget_does_not_excuse_invalid_or_unexpected_unexecuted_proposals():
    case = next(c for c in load_eval_dataset() if c.case_id == "normal_application")
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence)
    for index in range(8):
        call = _search_proposal(f"used-{index}").tool_calls[0]
        runtime.propose((call,))
        await runtime.execute(call)
    invalid = ModelToolCall(
        call_id="invalid-over-budget",
        name="search_web",
        arguments={"query": "private-invalid", "max_results": "8"},
    )
    runtime.propose((invalid,))
    assert evidence.tools()[-1].disposition == "rejected_invalid"

    under_budget_evidence = live.ExecutionEvidence()
    under_budget = fixed_runtime(case, under_budget_evidence)
    valid = _search_proposal("unexpected").tool_calls[0]
    under_budget.propose((valid,))
    unexpected = under_budget_evidence.tools()[0]
    assert unexpected.disposition == "unexpected_unexecuted"
    assert not live.grade_live_output(
        case,
        None,
        tool_observations=(unexpected,),
        exposed_evidence=(),
    ).tool_execution_consistent


async def test_graph_failure_remains_primary_with_budget_suppression_and_is_sanitized():
    scripts = {
        "normal_application": (
            *tuple(_search_proposal(f"executed-{index}") for index in range(8)),
            _search_proposal("suppressed"),
        )
    }

    async def break_grounding(adapter, logical, result):
        if "normal_application" not in logical or "write_report" not in logical:
            return result
        payload = json.loads(result.content)
        first_claim = next(
            claim for claim in (*payload["summary"], *payload["findings"]) if claim["citations"]
        )
        first_claim["citations"][0]["source_id"] = "private_exception_message_canary"
        return result.model_copy(update={"content": json.dumps(payload)})

    result = await run(setup(ScriptedQwen(break_grounding, research_scripts=scripts)))
    observation = result.report.observations[0]
    assert observation.execution_error == "invalid_output"
    assert observation.execution_error != "tool_behavior_failure"
    assert observation.budget_suppressed_proposal_count == 1
    assert observation.tool_observations[-1].disposition == "budget_suppressed"
    assert observation.graph_failure is not None
    assert observation.graph_failure.node_name == "write_report"
    assert observation.graph_failure.cause_category == "invalid_model_grounding"
    assert observation.graph_failure.schema_error_type == "citation_source_mismatch"
    assert observation.graph_failure.schema_error_path is not None
    serialized = observation.model_dump_json()
    assert "private_exception_message_canary" not in serialized
    assert "private-budget-query" not in serialized
    assert "Traceback" not in serialized


@pytest.fixture(scope="module")
async def autonomous():
    """One offline matrix with deviations, not another exact-script happy path."""
    done = ChatModelResult(
        content="Enough research for this pass.", usage=ModelUsage(input_tokens=1, output_tokens=1)
    )

    def search(call_id):
        return ChatModelResult(
            tool_calls=(
                ModelToolCall(
                    call_id=call_id,
                    name="search_web",
                    arguments={"query": "private-autonomous-query", "max_results": 8},
                ),
            ),
            usage=ModelUsage(input_tokens=1, output_tokens=1),
        )

    cases = {c.case_id: c for c in load_eval_dataset()}
    normal = _research_script(cases["normal_application"])
    resume = _research_script(cases["resume_prompt_injection"])
    insufficient = _research_script(cases["insufficient"])
    conflict = _research_script(cases["conflicting_sources"])
    draft = _research_script(cases["application_resume_draft"])
    extra_retrieve = draft[0].model_copy(
        update={
            "tool_calls": (draft[0].tool_calls[0].model_copy(update={"call_id": "extra-retrieve"}),)
        }
    )
    mixed = _research_script(cases["web_and_resume"])
    scripts = {
        "normal_application": (normal[0], search("extra-search"), done),
        "application_resume_draft": (draft[0], extra_retrieve, done),
        "web_and_resume": (
            mixed[0].model_copy(update={"tool_calls": tuple(reversed(mixed[0].tool_calls))}),
            done,
        ),
        "resume_prompt_injection": (search("wrong-modality"), resume[0], done),
        # Production still requests a second research pass on empty evidence;
        # that pass may finish without a second tool call.
        "insufficient": (insufficient[0], done, done),
        "conflicting_sources": (
            conflict[0].model_copy(update={"tool_calls": conflict[0].tool_calls[:1]}),
            done,
        ),
        "document_only_resume": (search("web-only"), done, done),
    }

    async def partial_writer(adapter, logical, result):
        if "conflicting_sources" in logical and "write_report" in logical:
            payload = json.loads(result.content)
            available = live.exposed_reference(
                cases["conflicting_sources"].searches[0].results[0]
            ).evidence_id
            # Valid, grounded partial answer, not invented citations to unseen B.
            citation = next(
                citation
                for section in ("summary", "findings")
                for claim in payload[section]
                for citation in claim["citations"]
                if citation["evidence_id"] == available
            )
            payload["summary"] = [
                {
                    "claim_id": "partial_source_a",
                    "text": cases["conflicting_sources"].searches[0].results[0].snippet,
                    "citations": [citation],
                }
            ]
            payload["findings"] = []
            return result.model_copy(update={"content": json.dumps(payload)})
        if "document_only_resume" in logical and "write_report" in logical:
            # Model recognizes its lack of evidence; quality must still fail the case.
            return result.model_copy(
                update={
                    "content": json.dumps(
                        {
                            "summary": [],
                            "findings": [],
                            "application_draft": None,
                            "limitations": [],
                        }
                    )
                }
            )
        return result

    parts = setup(ScriptedQwen(partial_writer, research_scripts=scripts))
    evidence = []
    result = await run(parts, execution_evidence=evidence)
    return parts, result, evidence


async def test_autonomous_extra_search_and_wrong_modality_recovery_pass(autonomous):
    _, result, evidence = autonomous
    assert result.stop_reason is None and result.report.complete
    for case_id, resolutions in (
        ("normal_application", ("declared_fixture", "declared_fixture", "deterministic_empty")),
        ("resume_prompt_injection", ("deterministic_empty", "declared_fixture")),
        ("application_resume_draft", ("declared_fixture", "deterministic_empty")),
        ("web_and_resume", ("declared_fixture", "declared_fixture")),
    ):
        for observation in (o for o in result.report.observations if o.case_id == case_id):
            assert observation.passed
            assert (
                tuple(t.evidence_resolution for t in observation.tool_observations) == resolutions
            )
            assert all(t.executed for t in observation.tool_observations)
            assert observation.grader.tool_execution_consistent
            assert any("write_report" in a.logical_call_id for a in observation.provider_attempts)
    assert not any(e.tool_failure for e in evidence)
    assert all(o.hard_invariants.clean for o in result.report.observations)
    assert "private-autonomous-query" not in result.report.model_dump_json()
    # Exactly 16/20, both injection cases all-pass: V2 acceptance permits deviations.
    assert result.report.aggregate.case_pass_rate == 0.8 and result.exit_code == 0
    AcceptedLiveEvalBaselineV5(report=result.report.model_copy(update={"exploratory": False}))


async def test_fewer_tools_pass_only_when_final_evidence_quality_allows(autonomous):
    _, result, _ = autonomous
    for observation in result.report.observations:
        if observation.case_id == "insufficient":
            assert len(observation.tool_observations) == 1
            assert observation.passed
        if observation.case_id in {"conflicting_sources", "document_only_resume"}:
            assert observation.execution_error is None  # Writer ran and returned a valid output.
            assert observation.grader.structured_output_valid
            assert observation.grader.tool_execution_consistent
            assert not observation.grader.required_evidence_coverage
            assert not observation.grader.minimum_cited_sources_met
            assert not observation.passed


def fixed_runtime(case, evidence, *, delegate_wrapper=None):
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    fixed = live.FixedEvidence(case, tenant, evidence)
    registry = live.create_research_tool_registry(
        search_port=fixed,
        retrieval_service=fixed,
        tenant=tenant,
        allowed_document_ids=(case.resume_document_id,) if case.resume_document_id else (),
    )
    runtime = registry.bind(
        policy_name=live.RESEARCH_TOOL_POLICY_NAME,
        context=live.ToolRunContext(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            run_id=uuid4(),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target={},
            deadline=live.monotonic() + 60,
            cancellation=live._Cancellation(asyncio.Event()),
        ),
    )
    return live.ObservedTools(delegate_wrapper(runtime) if delegate_wrapper else runtime, evidence)


async def test_max_results_truncates_and_grader_never_reconstructs_unseen_results():
    from tests.evals.harness import execute_eval_case

    original = next(c for c in load_eval_dataset() if c.case_id == "normal_application")
    output = (await execute_eval_case(original)).output
    # Test-only multi-result fixture; source dataset/manifest stay immutable.
    combined = original.searches[0].model_copy(
        update={"results": original.searches[0].results + original.searches[1].results}
    )
    case = original.model_copy(update={"searches": (combined, original.searches[1])})
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence)
    call = ModelToolCall(
        call_id="small-result-bound",
        name="search_web",
        arguments={"query": "private-small-query", "max_results": 1},
    )
    runtime.propose((call,))
    returned = json.loads(await runtime.execute(call))
    assert returned["result_count"] == 1
    assert (
        returned["results"][0]["source_id"] == live.exposed_reference(combined.results[0]).source_id
    )
    assert len(evidence.exposed_evidence) == 1 and not evidence.tool_failure
    tools = evidence.tools()
    assert tools[0].exposed_result_count == 1 and tools[0].evidence_resolution == "declared_fixture"
    graded = live.grade_live_output(
        case, output, tool_observations=tools, exposed_evidence=tuple(evidence.exposed_evidence)
    )
    assert graded.tool_execution_consistent
    assert not graded.citation_grounding_proxy
    assert not graded.required_evidence_coverage
    assert not graded.minimum_cited_sources_met
    assert "private-small-query" not in tools[0].model_dump_json()


async def test_declared_empty_and_extra_empty_have_distinct_sanitized_diagnostics():
    case = next(c for c in load_eval_dataset() if c.case_id == "insufficient")
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence)
    for number in range(1, 4):
        call = ModelToolCall(
            call_id=f"empty-{number}",
            name="search_web",
            arguments={"query": "private-empty-query", "max_results": 1},
        )
        runtime.propose((call,))
        assert json.loads(await runtime.execute(call))["results"] == []
    tools = evidence.tools()
    assert [t.evidence_resolution for t in tools] == [
        "declared_fixture",
        "declared_fixture",
        "deterministic_empty",
    ]
    assert all(t.executed and t.exposed_result_count == 0 for t in tools)
    assert not evidence.tool_failure and not evidence.exposed_evidence
    assert "private-empty-query" not in "".join(t.model_dump_json() for t in tools)


async def test_registry_failure_after_adapter_return_does_not_count_exposure():
    class FailedDelivery:
        def __init__(self, runtime):
            self.runtime = runtime

        def model_tools(self):
            return self.runtime.model_tools()

        def validate_call(self, call):
            self.runtime.validate_call(call)

        async def execute(self, call):
            await self.runtime.execute(call)
            raise live.ToolInputValidationError("simulated delivery failure")

    case = next(c for c in load_eval_dataset() if c.case_id == "normal_application")
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence, delegate_wrapper=FailedDelivery)
    call = ModelToolCall(
        call_id="not-delivered", name="search_web", arguments={"query": "q", "max_results": 8}
    )
    runtime.propose((call,))
    with pytest.raises(live.ToolInputValidationError):
        await runtime.execute(call)
    assert not evidence.exposed_evidence
    assert evidence.tools()[0].evidence_resolution == "none"
    assert not evidence.tools()[0].executed
    assert evidence.tools()[0].disposition == "execution_failed"


@pytest.mark.parametrize("bound", ["8", 0, 9])
async def test_v2_search_bounds_still_belong_to_production_registry(bound):
    case = next(c for c in load_eval_dataset() if c.case_id == "normal_application")
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence)
    call = ModelToolCall(
        call_id="bad-bound", name="search_web", arguments={"query": "q", "max_results": bound}
    )
    runtime.propose((call,))
    with pytest.raises(live.ToolInputValidationError):
        await runtime.execute(call)
    assert evidence.tool_failure and not evidence.queries and not evidence.exposed_evidence
    assert not evidence.tools()[0].executed


async def test_v2_resolver_does_not_override_disallowed_registry_tool():
    from app.tools.web_search import create_search_web_tool_registry

    case = next(c for c in load_eval_dataset() if c.case_id == "document_only_resume")
    evidence = live.ExecutionEvidence()
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    fixed = live.FixedEvidence(case, tenant, evidence)
    # A narrower production policy still forbids retrieval, even with declared evidence.
    registry = create_search_web_tool_registry(search_port=fixed)
    runtime = live.ObservedTools(
        registry.bind(
            policy_name=live.RESEARCH_TOOL_POLICY_NAME,
            context=live.ToolRunContext(
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                run_id=uuid4(),
                action_intent_id=None,
                approval_request_id=None,
                trusted_target={},
                deadline=live.monotonic() + 60,
                cancellation=live._Cancellation(asyncio.Event()),
            ),
        ),
        evidence,
    )
    call = ModelToolCall(call_id="disallowed", name="retrieve_documents", arguments={"query": "q"})
    runtime.propose((call,))
    with pytest.raises(live.ToolNotAllowedError):
        await runtime.execute(call)
    assert evidence.tool_failure and not evidence.queries and not evidence.exposed_evidence
    assert not evidence.tools()[0].executed


async def test_foreign_document_fixture_is_configuration_and_hard_failure():
    case = next(c for c in load_eval_dataset() if c.case_id == "document_only_resume")
    retrieval = case.document_retrievals[0]
    bad_result = retrieval.results[0].model_copy(update={"document_id": uuid4()})
    case = case.model_copy(
        update={"document_retrievals": (retrieval.model_copy(update={"results": (bad_result,)}),)}
    )
    evidence = live.ExecutionEvidence()
    runtime = fixed_runtime(case, evidence)
    call = ModelToolCall(
        call_id="foreign-fixture", name="retrieve_documents", arguments={"query": "q"}
    )
    runtime.propose((call,))
    with pytest.raises(ToolExecutionError):  # Registry translates adapter failures.
        await runtime.execute(call)
    assert evidence.configuration_failure
    assert evidence.hard["cross_workspace_leakage_count"] == 1
    assert not evidence.exposed_evidence and not evidence.tools()[0].executed
