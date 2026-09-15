"""Scripted graph governance tests; no semantic accuracy or live-quality claim."""

import asyncio
import json
import traceback
from dataclasses import replace
from itertools import pairwise
from time import monotonic
from uuid import uuid4

import pytest

from app.agents.contracts import AgentLoopControl, AgentLoopLimitsV1
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1
from app.domain.runs import DEFAULT_RUN_LIMITS
from app.domain.tenancy import TenantContext
from app.domain.tool_invocations import ToolInvocationAuthorizationError
from app.llm.factory import LLMFactory, LLMRetryPolicy
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import LLMInvocationAuthorizationError, LLMInvocationContext
from app.llm.ports import LOCKED_CHAT_MODEL, ChatModelResult, ModelToolCall, ProviderAdapterError
from app.obs.agent_loop import StructlogAgentLoopObserver
from app.retrieval.documents import RetrievedDocumentChunk
from app.tools.contracts import ToolRunContext
from app.tools.document_retrieval import create_research_tool_registry
from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.search import SearchResult, search_source_id
from tests.evals import quality_e7a_graph as module
from tests.evals.quality_e7a_contracts import (
    E7ABudgetUsageV1,
    EvidenceAssessmentV1,
    EvidenceGapV1,
    allocate_budget,
)
from tests.evals.quality_e7a_graph import (
    E7AGraphError,
    E7AGraphRuntime,
    E7AResearchStateV1,
    E7ASourceScope,
    E7AUsageOwner,
    build_e7a_research_graph,
    build_gap_followup_plan,
    evidence_context,
    query_identity,
)
from tests.unit.llm.test_factory import _RecordingRecorder, _RecordingTraceSink

CANARY = "e7a4-private-content-canary"


class Cancellation:
    cancelled = False

    def is_cancelled(self):
        return self.cancelled


class Observer:
    def __init__(self):
        self.records = []
        self.logger = StructlogAgentLoopObserver()

    def observe(self, observation):
        self.records.append(observation)
        self.logger.observe(observation)


class Adapter:
    provider = "fake"
    model = LOCKED_CHAT_MODEL

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self.owner = None

    async def invoke(self, messages, tools, metadata, *, attempt):
        assert self.owner.usage.model_calls > 0
        self.calls.append((messages, tools, metadata, attempt, self.owner.usage))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return await step(messages, tools, metadata)
        return step


def web(text="Initial evidence", suffix="one"):
    url = f"https://example.test/{suffix}"
    return SearchResult(source_id=search_source_id(url), title="Source", url=url, snippet=text)


def document(doc, text="Resume experience", *, chunk=None):
    return RetrievedDocumentChunk(
        document_id=doc,
        chunk_id=chunk or uuid4(),
        source_name="resume.md",
        section="Experience",
        ordinal=0,
        cosine_distance=0.1,
        text=text,
    )


class Search:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def search(self, query, max_results, deadline):
        self.calls.append((query, max_results, deadline))
        result = self.responses.get(query, ())
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return await result()
        return tuple(result)[:max_results]


class Documents:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.get(kwargs["query"], ())
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return await result()
        return tuple(result)


class ToolRecorder(InMemoryToolInvocationRecorder):
    def __init__(self):
        super().__init__()
        self.events = []
        self.error = None

    async def reserve(self, **kwargs):
        self.events.append(kwargs)
        if self.error:
            raise self.error
        return await super().reserve(**kwargs)


def plan():
    return ChatModelResult(content=json.dumps({"queries": ["initial"]}))


def calls(*queries, name="search_web", arguments=None):
    return ChatModelResult(
        tool_calls=tuple(
            ModelToolCall(
                call_id=f"call-{uuid4()}",
                name=name,
                arguments=arguments
                if arguments is not None
                else (
                    {"query": query, "max_results": 8} if name == "search_web" else {"query": query}
                ),
            )
            for query in queries
        )
    )


DONE = ChatModelResult(content="Research complete")


def assess(outcome="sufficient", *, code="missing_job_fact", topic="gap"):
    async def response(messages, tools, metadata):
        assert metadata["graph_node"] == "evidence_assessment" and not tools
        payload = json.loads(messages[-1].content)
        ids = tuple(x["evidence_id"] for x in payload["evidence"])
        return ChatModelResult(
            content=EvidenceAssessmentV1(
                outcome=outcome,
                evidence_ids=ids if outcome != "insufficient" else (),
                gaps=()
                if outcome == "sufficient"
                else (
                    EvidenceGapV1(
                        code="source_conflict" if outcome == "conflicting" else code,
                        topic=topic,
                    ),
                ),
            ).model_dump_json()
        )

    return response


def setup(*steps, scope=None, searches=None, documents=None, retries=1, recorder=None):
    cancellation = Cancellation()
    control = AgentLoopControl(
        limits=AgentLoopLimitsV1(**dict(DEFAULT_RUN_LIMITS)),
        deadline=monotonic() + 30,
        cancellation=cancellation,
    )
    scope = scope or E7ASourceScope(
        web_available=True, job_tools=("search_web",), task_tools=("search_web",)
    )
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    context = LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=uuid4())
    adapter = Adapter(steps)
    recorder = recorder or _RecordingRecorder()
    trace = _RecordingTraceSink()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=adapter,
        embedding_adapter=FakeEmbeddingModel(),
        trace_sink=trace,
        retry_policy=LLMRetryPolicy(
            max_attempts=retries, base_delay_seconds=0.001, max_delay_seconds=0.001
        ),
    )
    search = Search(
        searches
        if searches is not None
        else {"initial": (web(),), "gap": (web("New evidence", "two"),)}
    )
    docs = Documents(documents or {})
    tools = ToolRecorder()
    registry = create_research_tool_registry(
        search_port=search,
        retrieval_service=docs,
        tenant=tenant,
        allowed_document_ids=scope.allowed_document_ids,
        recorder=tools,
    )
    bound = registry.bind(
        policy_name="research_agent",
        context=ToolRunContext(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            run_id=context.run_id,
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=control.deadline,
            cancellation=cancellation,
        ),
    )
    published = []
    owner = E7AUsageOwner(published.append)
    adapter.owner = owner
    runtime = E7AGraphRuntime(factory, context, bound, scope, control, Observer(), owner)
    return runtime, adapter, search, docs, tools, recorder, trace, published


async def run(runtime, *, draft=False, query="Research role"):
    result = await build_e7a_research_graph().ainvoke(
        {
            "payload": {
                "request": ResearchRequestV1(
                    query=query, include_application_draft=draft
                ).model_dump(mode="json")
            }
        },
        context=runtime,
    )
    return E7AResearchStateV1.model_validate_json(
        json.dumps(result["payload"], allow_nan=False), strict=True
    )


@pytest.mark.parametrize("outcome", ["sufficient", "partial", "insufficient", "conflicting"])
async def test_outcome_routes_and_shared_governance(outcome):
    steps = [plan(), calls("initial"), DONE, assess(outcome)]
    if outcome in {"partial", "insufficient"}:
        steps += [calls("gap"), DONE, assess()]
    runtime, adapter, search, _, tools, recorder, _, published = setup(*steps)
    result = await run(runtime)
    second = outcome in {"partial", "insufficient"}
    assert result.pass_count == (2 if second else 1)
    assert result.stop_reason == ("conflicting" if outcome == "conflicting" else "sufficient")
    assert [x[0] for x in search.calls] == (["initial", "gap"] if second else ["initial"])
    assert result.usage.plan_calls == 1
    assert result.usage.research_calls == ((2, 2) if second else (2, 0))
    assert result.usage.assessment_calls == ((1, 1) if second else (1, 0))
    assert result.usage.tool_calls == ((1, 1) if second else (1, 0))
    assert (
        result.usage.writer_calls == 0
        and allocate_budget(result.usage, stage="writer", assessment=result.assessment).model_calls
        == 2
    )
    assert len(adapter.calls) == result.usage.model_calls
    assert len(tools.events) == sum(result.usage.tool_calls)
    assert len(recorder.events) == len(adapter.calls) * 2
    assert published[-1] == runtime.usage_owner.usage == result.usage
    for previous, current in pairwise(published):
        assert current.model_calls >= previous.model_calls
        assert sum(current.tool_calls) >= sum(previous.tool_calls)
    if second:
        message = adapter.calls[4][0][-1].content
        assert "<untrusted_gap_queries>" in message and '"query": "gap"' in message
        assert result.summaries[1].gap_codes == ("missing_job_fact",)


@pytest.mark.parametrize("second", ["partial", "insufficient", "conflicting"])
async def test_no_third_pass_even_with_new_evidence_and_unused_budget(second):
    runtime, adapter, search, *_ = setup(
        plan(), calls("initial"), DONE, assess("partial"), calls("gap"), DONE, assess(second)
    )
    result = await run(runtime)
    assert result.pass_count == 2 and result.assessment.outcome == second
    assert result.stop_reason == ("conflicting" if second == "conflicting" else "pass_limit")
    assert len(search.calls) == 2 and len(adapter.calls) == 7


@pytest.mark.parametrize(
    "hits", [(), (web(),), (web("Initial evidence", "copy"),), (web("Initial   evidence", "copy"),)]
)
async def test_second_pass_without_new_content_skips_assessor(hits):
    runtime, adapter, _, _, _, _, _, _ = setup(
        plan(),
        calls("initial"),
        DONE,
        assess("partial"),
        calls("gap"),
        DONE,
        searches={"initial": (web(),), "gap": hits},
    )
    result = await run(runtime)
    assert result.stop_reason == "no_new_evidence"
    assert result.assessment.outcome == "partial" and result.usage.assessment_calls == (1, 0)
    assert not result.summaries[1].obtained_new_evidence
    assert result.summaries[1].new_content_count == 0
    assert len(adapter.calls) == 6


async def test_empty_first_pass_is_deterministic_and_can_fill_gap_once():
    topic = "Requested facts have no delivered evidence"
    runtime, adapter, _, *_ = setup(
        plan(), calls("initial"), DONE, calls(topic), DONE, assess(), searches={topic: (web(),)}
    )
    result = await run(runtime)
    assert result.usage.assessment_calls == (0, 1)
    assert result.assessment.outcome == "sufficient" and len(adapter.calls) == 6


@pytest.mark.parametrize(
    "topic", ["initial", "  INITIAL  ", "\uff29\uff2e\uff29\uff34\uff29\uff21\uff2c"]
)
async def test_previous_equivalent_query_removes_followup_before_model_call(topic):
    runtime, adapter, search, *_ = setup(
        plan(), calls("initial"), DONE, assess("partial", topic=topic)
    )
    result = await run(runtime)
    assert result.stop_reason == "no_executable_query" and result.pass_count == 1
    assert len(adapter.calls) == 4 and len(search.calls) == 1


async def test_same_batch_and_later_batch_query_dedup_are_real_execution_filters():
    runtime, adapter, search, *_ = setup(
        plan(),
        calls("initial", "  INITIAL  ", "\uff29\uff2e\uff29\uff34\uff29\uff21\uff2c"),
        calls("initial"),
        assess(),
    )
    result = await run(runtime)
    assert len(search.calls) == 1 and result.usage.tool_calls == (1, 0)
    assert result.summaries[0].duplicate_query_count == 3
    assert len(adapter.calls) == 4


async def test_max_results_cannot_evade_query_identity():
    runtime, *_ = setup()
    assert query_identity(runtime, "search_web", " A\tB ") == query_identity(
        runtime, "search_web", "\uff41 b"
    )
    assert query_identity(runtime, "search_web", "a b") != query_identity(
        runtime, "retrieve_documents", "a b"
    )
    state = E7AResearchStateV1(request=ResearchRequestV1(query="Task"))
    adapter = module._ResearchTools(runtime, state, 1, 4, 7)
    response = calls("initial", "INITIAL")
    response = response.model_copy(
        update={
            "tool_calls": (
                response.tool_calls[0],
                response.tool_calls[1].model_copy(
                    update={"arguments": {"query": "INITIAL", "max_results": 1}}
                ),
            )
        }
    )
    assert len(adapter.filter_response(response).tool_calls) == 1


@pytest.mark.parametrize(
    "proposal",
    [
        calls("gap", name="invented_tool"),
        calls("different topic"),
        calls("gap", arguments={"query": "gap", "max_results": 8, "workspace_id": "foreign"}),
        calls("gap", name="retrieve_documents"),
    ],
)
async def test_followup_rejects_new_tools_scope_fields_and_unplanned_queries(proposal):
    runtime, _, search, _, tools, *_ = setup(
        plan(), calls("initial"), DONE, assess("partial"), proposal
    )
    result = await run(runtime)
    assert result.stop_reason == "no_new_evidence" and len(search.calls) == len(tools.events) == 1
    assert result.summaries[1].rejected_proposal_count == 1
    assert result.usage.research_calls == (2, 1) and result.usage.tool_calls == (1, 0)


@pytest.mark.parametrize(
    "code", ["missing_resume_fact", "unsupported_claim_strength", "missing_task_fact"]
)
async def test_missing_trusted_source_cannot_fall_back_to_web(code):
    runtime, adapter, search, *_ = setup(
        plan(),
        calls("initial"),
        DONE,
        assess("partial", code=code),
        scope=E7ASourceScope(web_available=True, job_tools=("search_web",)),
    )
    result = await run(runtime)
    assert result.stop_reason == "no_retrievable_gap" and len(adapter.calls) == 4
    assert len(search.calls) == 1


@pytest.mark.parametrize(
    "code",
    ["missing_resume_fact", "unsupported_claim_strength", "missing_job_fact", "missing_task_fact"],
)
async def test_document_gap_keeps_tenant_allowlist_and_source_type(code):
    doc = uuid4()
    scope = E7ASourceScope((doc,), doc, False, ("retrieve_documents",), ("retrieve_documents",))
    runtime, adapter, search, docs, tools, *_ = setup(
        plan(),
        calls("initial", name="retrieve_documents"),
        DONE,
        assess("partial", code=code),
        calls("gap", name="retrieve_documents"),
        DONE,
        assess(),
        scope=scope,
        documents={"initial": (document(doc),), "gap": (document(doc, "Additional experience"),)},
    )
    result = await run(runtime, draft=True)
    assert result.assessment.outcome == "sufficient" and result.pass_count == 2
    assert not search.calls and len(docs.calls) == len(tools.events) == 2
    assert all(x["allowed_document_ids"] == (doc,) for x in docs.calls)
    assert all(
        x["tenant"].workspace_id == runtime.invocation_context.workspace_id for x in docs.calls
    )
    assert all(
        x.source_type == "workspace_document" for x in evidence_context(result, scope).evidence
    )
    assert {x.name for x in adapter.calls[4][1]} == {"retrieve_documents"}


@pytest.mark.parametrize("foreign", [True, False])
async def test_document_scope_or_changed_chunk_identity_fails_without_second_assessment(foreign):
    doc, chunk = uuid4(), uuid4()
    scope = E7ASourceScope((doc,), doc, False, (), ("retrieve_documents",))
    runtime, adapter, _, docs, *_ = setup(
        plan(),
        calls("initial", name="retrieve_documents"),
        DONE,
        assess("partial", code="missing_resume_fact"),
        calls("gap", name="retrieve_documents"),
        scope=scope,
        documents={
            "initial": (document(doc, chunk=chunk),),
            "gap": (document(uuid4() if foreign else doc, "Changed content", chunk=chunk),),
        },
    )
    with pytest.raises(E7AGraphError) as error:
        await run(runtime, draft=True)
    assert error.value.category == (
        "invalid_source_scope" if foreign else "evidence_identity_conflict"
    )
    assert len(docs.calls) == 2 and len(adapter.calls) == 5
    assert runtime.usage_owner.usage.tool_calls == (1, 1)


async def test_same_pass_changed_document_chunk_is_not_silently_dropped():
    doc, chunk = uuid4(), uuid4()
    runtime, *_ = setup(
        plan(),
        calls("initial", "gap", name="retrieve_documents"),
        scope=E7ASourceScope((doc,), doc),
        documents={
            "initial": (document(doc, chunk=chunk),),
            "gap": (document(doc, "Changed", chunk=chunk),),
        },
    )
    with pytest.raises(E7AGraphError, match="evidence_identity_conflict"):
        await run(runtime, draft=True)
    assert runtime.usage_owner.usage.tool_calls == (2, 0)


async def test_duplicate_support_keeps_references_but_assessor_gets_one_copy():
    runtime, adapter, *_ = setup(
        plan(), calls("initial"), DONE, assess(), searches={"initial": (web(), web(suffix="copy"))}
    )
    result = await run(runtime)
    assert len(result.research.evidence) == 2 and len(result.research.sources) == 2
    assert (
        result.summaries[0].new_content_count == 1
        and result.summaries[0].duplicate_content_count == 1
    )
    assert len(json.loads(adapter.calls[3][0][-1].content)["evidence"]) == 1


async def test_plan_repair_and_full_research_respect_assessor_and_writer_reservations():
    steps = [ChatModelResult(content="invalid"), plan()]
    steps += [
        calls("initial"),
        calls("second"),
        calls("third"),
        calls("last-must-not-execute"),
        assess("partial"),
    ]
    steps += [calls("gap"), DONE, assess()]
    runtime, adapter, search, *_ = setup(*steps)
    result = await run(runtime)
    assert result.usage.plan_calls == 2 and result.usage.research_calls == (4, 2)
    assert result.usage.assessment_calls == (1, 1) and result.usage.model_calls == 10
    assert len(adapter.calls) == 10
    assert "last-must-not-execute" not in [x[0] for x in search.calls]
    assert (
        allocate_budget(result.usage, stage="writer", assessment=result.assessment).model_calls == 2
    )


async def test_seven_first_pass_tools_leave_exactly_one_for_followup():
    initial = tuple(f"first-{i}" for i in range(7))
    runtime, _, search, _, tools, *_ = setup(
        plan(),
        calls(*initial),
        DONE,
        assess("partial"),
        calls("gap"),
        DONE,
        assess(),
        searches={
            **{x: (web(suffix=str(i)),) for i, x in enumerate(initial)},
            "gap": (web("New", "last"),),
        },
    )
    result = await run(runtime)
    assert result.usage.tool_calls == (7, 1) and len(search.calls) == len(tools.events) == 8


@pytest.mark.parametrize(
    "remaining_tools,used_models,expected", [(0, 4, False), (1, 8, False), (1, 7, True)]
)
async def test_followup_budget_boundary_uses_latest_cumulative_usage(
    remaining_tools, used_models, expected
):
    runtime, *_ = setup(plan(), calls("initial"), DONE, assess("partial"))
    # Finish the first pass through the real graph with no routable gap, then inspect admission.
    no_route = replace(runtime, scope=E7ASourceScope(web_available=True))
    state = await run(no_route)
    usage = E7ABudgetUsageV1(
        plan_calls=1,
        research_calls=(used_models - 2, 0),
        assessment_calls=(1, 0),
        tool_calls=(8 - remaining_tools, 0),
    )
    runtime.usage_owner.usage = usage
    state = state.model_copy(update={"usage": usage})
    queries = build_gap_followup_plan(state, runtime)
    assert bool(queries) is expected


async def test_no_available_tools_skip_loop_and_empty_assessment_calls():
    runtime, adapter, search, docs, *_ = setup(plan(), scope=E7ASourceScope())
    result = await run(runtime)
    assert (
        result.assessment.outcome == "insufficient" and result.stop_reason == "no_retrievable_gap"
    )
    assert result.usage.model_calls == 1 and result.usage.assessment_calls == (0, 0)
    assert len(adapter.calls) == 1 and not search.calls and not docs.calls


@pytest.mark.parametrize("stage", ["plan", "research", "assessment"])
async def test_provider_failure_preserves_admitted_usage_and_safe_errors(stage, capsys):
    error = ProviderAdapterError(
        category="provider_timeout", retryable=False, provider="fake", model=LOCKED_CHAT_MODEL
    )
    steps = {
        "plan": [error],
        "research": [plan(), error],
        "assessment": [plan(), calls("initial"), DONE, error],
    }[stage]
    runtime, _, _, _, _, _, _, published = setup(*steps)
    with pytest.raises(E7AGraphError, match="provider_timeout") as captured:
        await run(runtime, query=CANARY)
    usage = runtime.usage_owner.usage
    assert usage.model_calls == {"plan": 1, "research": 2, "assessment": 4}[stage]
    assert published[-1] == usage
    assert (
        CANARY not in "".join(traceback.format_exception(captured.value)) + capsys.readouterr().out
    )


async def test_factory_retry_is_two_attempts_but_one_logical_plan_call():
    transient = ProviderAdapterError(
        category="provider_timeout", retryable=True, provider="fake", model=LOCKED_CHAT_MODEL
    )
    runtime, adapter, _, _, _, recorder, *_ = setup(
        transient, plan(), calls("initial"), DONE, assess(), retries=2
    )
    result = await run(runtime)
    assert result.usage.model_calls == 4 and len(adapter.calls) == 5
    assert len(recorder.events) == 10 and result.usage.plan_calls == 1


@pytest.mark.parametrize("failure", ["provider", "accounting", "authorization"])
async def test_technical_failures_never_become_insufficient(failure):
    recorder = _RecordingRecorder()
    if failure == "accounting":
        recorder.prepare_error = RuntimeError(CANARY)
    elif failure == "authorization":
        recorder.prepare_error = LLMInvocationAuthorizationError()
    steps = [RuntimeError(CANARY)] if failure == "provider" else [plan()]
    runtime, *_ = setup(*steps, recorder=recorder)
    with pytest.raises(E7AGraphError) as captured:
        await run(runtime)
    assert captured.value.category in {
        "provider_unavailable",
        "model_invocation_failed",
        "cancelled",
    }
    assert runtime.usage_owner.usage.plan_calls == 1
    assert CANARY not in str(captured.value)


async def test_tool_membership_revocation_stops_before_handler_and_retains_count():
    runtime, _, search, _, tools, *_ = setup(plan(), calls("initial"))
    tools.error = ToolInvocationAuthorizationError()
    with pytest.raises(E7AGraphError, match="cancelled"):
        await run(runtime)
    assert not search.calls and runtime.usage_owner.usage.tool_calls == (1, 0)


@pytest.mark.parametrize("where", ["before", "model", "tool", "after_tool"])
async def test_cancellation_boundaries_do_not_refund_usage(where):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def blocking(*args):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runtime, _, search, *_ = setup(
        plan(), blocking if where == "model" else calls("initial"), DONE, assess()
    )
    if where == "before":
        runtime.control.cancellation.cancelled = True
    if where == "tool":
        search.responses["initial"] = blocking
    if where == "after_tool":

        async def cancel_after():
            runtime.control.cancellation.cancelled = True
            return (web(),)

        search.responses["initial"] = cancel_after
    task = asyncio.create_task(run(runtime))
    if where in {"model", "tool"}:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
    else:
        with pytest.raises(E7AGraphError, match="cancelled"):
            await task
    usage = runtime.usage_owner.usage
    assert usage.model_calls == (0 if where == "before" else 2)
    assert sum(usage.tool_calls) == (1 if where in {"tool", "after_tool"} else 0)


async def test_deadline_at_entry_and_after_model_return():
    runtime, adapter, *_ = setup(plan())
    expired = replace(runtime, control=replace(runtime.control, deadline=1.0))
    with pytest.raises(E7AGraphError, match="deadline_exceeded"):
        await run(expired)
    assert not adapter.calls
    clock = [10.0]

    async def expire(*args):
        clock[0] = 31.0
        return plan()

    runtime, *_ = setup(expire)
    runtime = replace(
        runtime, control=replace(runtime.control, deadline=30.0, clock=lambda: clock[0])
    )
    with pytest.raises(E7AGraphError, match="deadline_exceeded"):
        await run(runtime)
    assert runtime.usage_owner.usage.plan_calls == 1


async def test_admission_notification_failure_prevents_provider_submission():
    runtime, adapter, *_ = setup(plan())

    def fail(_usage):
        raise RuntimeError(CANARY)

    runtime.usage_owner.on_admitted = fail
    with pytest.raises(E7AGraphError):
        await run(runtime)
    assert runtime.usage_owner.usage.plan_calls == 1 and not adapter.calls


@pytest.mark.parametrize(
    "extra",
    [
        {"workspace_id": "foreign"},
        {"pass_count": 2},
        {"usage": {}},
        {"graph_version": "wrong"},
        {"request": {"query": "Q", "role": "admin"}},
    ],
)
async def test_strict_graph_input_rejects_context_and_state_injection(extra):
    runtime, adapter, *_ = setup(plan())
    payload = {"request": {"query": "Task"}, **extra}
    with pytest.raises(E7AGraphError, match="invalid_schema"):
        await build_e7a_research_graph().ainvoke({"payload": payload}, context=runtime)
    assert not adapter.calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"web_available": "true"},
        {"allowed_document_ids": ("foreign",)},
        {"task_tools": ("unknown",)},
        {"job_tools": ("search_web",)},
        {"embedding_profile": "other"},
        {"resume_document_id": uuid4()},
    ],
)
def test_untrusted_or_inconsistent_source_scope_is_rejected(kwargs):
    with pytest.raises(E7AGraphError, match="invalid_source_scope"):
        E7ASourceScope(**kwargs)


async def test_runtime_cannot_be_reused_to_reset_budget():
    runtime, adapter, *_ = setup(plan(), calls("initial"), DONE, assess())
    await run(runtime)
    with pytest.raises(E7AGraphError, match="configuration_error"):
        await run(runtime)
    assert len(adapter.calls) == 4


async def test_json_result_and_actual_observation_surfaces_keep_private_data_out(capsys):
    runtime, adapter, _, _, _, _, trace, _ = setup(
        plan(),
        calls("initial"),
        DONE,
        assess("partial", topic=CANARY),
        calls(CANARY),
        DONE,
        searches={"initial": (web(CANARY),)},
    )
    result = await run(runtime, query=CANARY)
    raw = result.model_dump_json()
    assert CANARY in raw  # Authorized business state, not a public artifact.
    summaries = json.dumps([x.model_dump() for x in result.summaries])
    observations = json.dumps([x.model_dump(mode="json") for x in runtime.observer.records])
    surfaces = (
        summaries
        + observations
        + repr(trace.starts)
        + repr(trace.finishes)
        + capsys.readouterr().out
    )
    assert CANARY not in surfaces
    assert str(runtime.invocation_context.workspace_id) not in adapter.calls[0][0][-1].content
    assert result.graph_version == module.GRAPH_VERSION and result.stop_reason == "no_new_evidence"
    assert (
        not {"action_intent_id", "approval_request_id", "application_draft", "run_status"}
        & result.model_dump().keys()
    )
    assert E7AResearchStateV1.model_validate_json(raw, strict=True) == result


async def test_trace_failure_is_lossy_and_does_not_change_research_result():
    runtime, _, _, _, _, _, trace, _ = setup(plan(), calls("initial"), DONE, assess())
    trace.start_error = RuntimeError(CANARY)
    trace.finish_error = RuntimeError(CANARY)
    result = await run(runtime)
    assert result.assessment.outcome == "sufficient"


async def test_registry_timeout_retains_tool_usage_and_stops_without_assessment(monkeypatch):
    from app.tools import document_retrieval

    monkeypatch.setattr(document_retrieval, "RETRIEVE_DOCUMENTS_TIMEOUT_SECONDS", 0.02)
    doc = uuid4()
    stopped = asyncio.Event()

    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runtime, adapter, _, _, tools, *_ = setup(
        plan(),
        calls("initial", name="retrieve_documents"),
        scope=E7ASourceScope((doc,), doc),
        documents={"initial": slow},
    )
    with pytest.raises(E7AGraphError, match="provider_timeout"):
        await run(runtime, draft=True)
    assert stopped.is_set() and len(tools.events) == 1 and len(adapter.calls) == 2
    assert runtime.usage_owner.usage.tool_calls == (1, 0)


async def test_real_retrieval_service_keeps_factory_embedding_and_original_profile():
    from app.retrieval.documents import EMBEDDING_PROFILE, DocumentRetrievalService

    doc = uuid4()
    hit = document(doc)
    runtime, adapter, _, _, tools, recorder, *_ = setup(
        plan(),
        calls("initial", name="retrieve_documents"),
        DONE,
        assess(),
        scope=E7ASourceScope((doc,), doc),
    )
    recorded = []

    class Repository:
        async def search(self, **kwargs):
            recorded.append(kwargs)
            return (hit,)

    service = DocumentRetrievalService(
        Repository(), runtime.factory.create_embedding_model(runtime.invocation_context)
    )
    tenant = TenantContext(
        runtime.invocation_context.workspace_id,
        runtime.invocation_context.actor_user_id,
        WorkspaceRole.ADMIN,
    )
    registry = create_research_tool_registry(
        search_port=Search({}),
        retrieval_service=service,
        tenant=tenant,
        allowed_document_ids=(doc,),
        recorder=tools,
    )
    bound = registry.bind(
        policy_name="research_agent",
        context=ToolRunContext(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            run_id=runtime.invocation_context.run_id,
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=runtime.control.deadline,
            cancellation=runtime.control.cancellation,
        ),
    )
    result = await run(replace(runtime, tool_runtime=bound), draft=True)
    assert result.usage.model_calls == len(adapter.calls) == 4
    assert len(recorder.events) == 10  # Four chats plus one independently recorded embedding.
    assert len(recorded) == 1 and recorded[0]["allowed_document_ids"] == (doc,)
    assert recorded[0]["embedding_model"] == EMBEDDING_PROFILE
    assert recorded[0]["tenant"] == tenant


async def test_followup_plan_preserves_gap_order_and_truncates_to_remaining_tools():
    runtime, *_ = setup(plan(), calls("initial"), DONE, assess("partial"))
    state = await run(replace(runtime, scope=E7ASourceScope(web_available=True)))
    assessment = state.assessment.model_copy(
        update={
            "gaps": (
                EvidenceGapV1(code="missing_job_fact", topic="first gap"),
                EvidenceGapV1(code="missing_task_fact", topic=" FIRST  GAP "),
                EvidenceGapV1(code="missing_job_fact", topic="second gap"),
                EvidenceGapV1(code="missing_job_fact", topic="third gap"),
            )
        }
    )
    state = state.model_copy(update={"assessment": assessment})
    runtime.usage_owner.usage = runtime.usage_owner.usage.model_copy(update={"tool_calls": (6, 0)})
    result = build_gap_followup_plan(state, runtime)
    assert [x.query for x in result] == ["first gap", "second gap"]
    assert all(x.tool_name == "search_web" for x in result)


@pytest.mark.parametrize(
    "field,value",
    [("graph_version", "production"), ("pass_count", 3), ("usage", {"plan_calls": 99})],
)
def test_internal_json_state_rejects_version_pass_and_budget_drift(field, value):
    state = E7AResearchStateV1(request=ResearchRequestV1(query="Task")).model_dump(mode="json")
    state[field] = value
    with pytest.raises(E7AGraphError, match="invalid_schema"):
        module._checked(E7AResearchStateV1, state)


@pytest.mark.parametrize("where", ["model", "tool"])
async def test_trusted_cancellation_during_wait_uses_cancelled_category(where):
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def blocking(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runtime, _, search, *_ = setup(plan(), blocking if where == "model" else calls("initial"))
    if where == "tool":
        search.responses["initial"] = blocking
    task = asyncio.create_task(run(runtime))
    await asyncio.wait_for(entered.wait(), 5)
    runtime.control.cancellation.cancelled = True
    with pytest.raises(E7AGraphError, match="cancelled"):
        await asyncio.wait_for(task, 5)
    assert stopped.is_set()
    assert runtime.usage_owner.usage.model_calls == 2
    assert sum(runtime.usage_owner.usage.tool_calls) == (1 if where == "tool" else 0)
