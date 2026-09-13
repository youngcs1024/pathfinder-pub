"""Small offline E5.3 contracts; no capacity run, provider network or Docker."""

from __future__ import annotations

import asyncio
import json
import stat
from datetime import UTC, datetime
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from app.domain.actions import SubmitApplicationArgsV1
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError, LLMRetryPolicy
from app.llm.invocations import LLMInvocationContext, TraceIdentifiers
from app.llm.ports import ChatMessage, ModelToolCall, ProviderAdapterError, ProviderAttemptContext
from app.mock_portal.contracts import MockSubmissionRequestV1, mock_submission_payload_digest
from app.tools.adapters.mock_portal import MockPortalTransportError
from app.tools.contracts import ToolRunContext
from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.registry import ToolExecutionError, ToolInputValidationError
from app.tools.search import SearchPermanentError, SearchTransientError
from app.tools.web_search import RESEARCH_TOOL_POLICY_NAME, create_search_web_tool_registry
from tests.performance.adapters import (
    QUERY,
    CallLimitReached,
    Calls,
    Chat,
    Embedding,
    Mock,
    Search,
    worker_adapters,
)
from tests.performance.environment import EnvironmentError
from tests.performance.workload import (
    KINDS,
    CallRecord,
    Delay,
    Fault,
    parse_profile,
    profile,
    publish,
)

CANARY = "E53_BODY_SECRET_CANARY"


def configured(*, faults=(), name="instant-v1", **kwargs):
    value = profile(name).model_dump(mode="json")
    value.update(kwargs, faults=[item.model_dump(mode="json") for item in faults])
    return parse_profile(value)


def test_profiles_are_complete_bounded_and_json_round_trip():
    for name in ("instant-v1", "delayed-v1"):
        policy = profile(name)
        assert parse_profile(policy.model_dump(mode="json")) == policy
        assert set(policy.delays) == set(KINDS)
    assert all(delay.maximum == 0 for delay in profile().delays.values())
    assert profile("delayed-v1").delays["chat"] == Delay(minimum=0.05, maximum=0.15)
    assert profile().max_calls + 1 == 64


@pytest.mark.parametrize(
    "mutation",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": "53"},
        {"seed": 2**32},
        {"max_calls": 64},
        {"max_calls": 0},
        {"target": CANARY},
        {"name": CANARY},
        {"delays": {}},
        {"schema_version": 2},
        {"delays": {kind: {"minimum": 0, "maximum": 3} for kind in KINDS}},
        {"delays": {kind: {"minimum": 0.1, "maximum": 0.0} for kind in KINDS}},
        {"delays": {kind: {"minimum": 0, "maximum": float("nan")} for kind in KINDS}},
        {"delays": {kind: {"minimum": 0.1, "maximum": 0.1} for kind in KINDS}},
        {"faults": [{"call": "embedding", "ordinal": 1, "kind": "response_lost"}]},
        {"faults": [{"call": "chat", "ordinal": 0, "kind": "transient"}]},
        {"faults": [{"call": "mock_lookup", "ordinal": 1, "kind": "permanent"}]},
        {"faults": [{"call": "chat", "ordinal": 1, "kind": "transient"}] * 2},
    ],
)
def test_invalid_profile_fails_closed_without_input_echo(mutation):
    value = profile().model_dump(mode="json")
    value.update(mutation)
    with pytest.raises(EnvironmentError, match=r"^invalid_profile$"):
        parse_profile(value)


async def noop():
    return "ok"


async def test_seed_is_reproducible_and_each_kind_has_an_independent_sequence():
    first, second = Calls(profile("delayed-v1")), Calls(profile("delayed-v1"))
    await first.invoke("chat", noop)
    await first.invoke("search", noop)
    await first.invoke("chat", noop)
    await second.invoke("chat", noop)
    await second.invoke("chat", noop)

    def values(calls):
        return [
            item.delay_seconds
            for item in calls.records
            if item.call == "chat" and item.phase == "started"
        ]

    assert values(first) == values(second)
    assert all(0.05 <= value <= 0.15 for value in values(first))
    third = Calls(configured(name="delayed-v1", seed=54))
    await third.invoke("chat", noop)
    assert values(third)[0] != values(first)[0]


async def test_cancellation_interrupts_sleep_without_dispatching_or_blocking_loop():
    calls = Calls(
        configured(
            name="delayed-v1", delays={kind: {"minimum": 2.0, "maximum": 2.0} for kind in KINDS}
        )
    )
    dispatched = []

    async def operation():
        dispatched.append(True)

    task = asyncio.create_task(calls.invoke("chat", operation))
    async with asyncio.timeout(0.5):
        # A sibling task runs while the adapter waits; no time.sleep in the event loop.
        while not calls.records:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert dispatched == []
    assert [item.outcome for item in calls.records] == ["pending", "cancelled"]


async def test_deadline_bounds_sleep_and_accepted_call_limit_is_global_across_kinds():
    calls = Calls(configured(name="delayed-v1", max_calls=1))
    with pytest.raises(TimeoutError):
        await calls.invoke("chat", noop, timeout=0.001)
    assert calls.records[-1].outcome == "timeout"
    with pytest.raises(CallLimitReached):
        await calls.invoke("search", noop)
    assert sum(calls.counts.values()) == 1
    ingest = Calls(profile(), process="ingest")
    await ingest.invoke("embedding", noop)
    with pytest.raises(CallLimitReached):
        await ingest.invoke("embedding", noop)


@pytest.mark.parametrize(
    "kind,error",
    [
        ("chat", ProviderAdapterError),
        ("embedding", ProviderAdapterError),
        ("search", SearchTransientError),
        ("mock_lookup", MockPortalTransportError),
    ],
)
async def test_fixed_fault_occurs_on_exact_ordinal_without_adapter_retry(kind, error):
    calls = Calls(configured(faults=(Fault(call=kind, ordinal=2, kind="transient"),)))
    assert await calls.invoke(kind, noop) == "ok"
    with pytest.raises(error):
        await calls.invoke(kind, noop)
    assert await calls.invoke(kind, noop) == "ok"
    assert [item.outcome for item in calls.records if item.phase == "finished"] == [
        "succeeded",
        "failed",
        "succeeded",
    ]


class Recorder:
    def __init__(self, fail_prepare=False):
        self.starts = []
        self.ends = []
        self.fail_prepare = fail_prepare

    async def prepare(self, attempt):
        if self.fail_prepare:
            raise RuntimeError(CANARY)
        self.starts.append(attempt)

    async def finalize(self, attempt, outcome):
        self.ends.append((attempt, outcome))


class Trace:
    def __init__(self):
        self.starts = []
        self.ends = []

    def start(self, trace):
        self.starts.append(trace)
        return TraceIdentifiers(
            trace_id=trace.invocation_id.hex, observation_id=trace.invocation_id.hex[:16]
        )

    def finish(self, identity, outcome):
        self.ends.append((identity, outcome))


def factory(calls, recorder, trace=None):
    kwargs = {"trace_sink": trace} if trace is not None else {}
    return LLMFactory(
        recorder=recorder,
        chat_adapter=Chat(calls),
        embedding_adapter=Embedding(calls),
        retry_policy=LLMRetryPolicy(base_delay_seconds=0.001, max_delay_seconds=0.001),
        **kwargs,
    )


def context():
    return LLMInvocationContext(uuid4(), uuid4(), run_id=uuid4())


def messages():
    return (
        ChatMessage(
            role="user",
            content='<untrusted_research_request>\n{"normalized_query":"'
            + CANARY
            + '"}\n</untrusted_research_request>',
        ),
    )


async def test_factory_records_retry_attempts_traces_and_embedding_without_body_leak(tmp_path):
    calls = Calls(
        configured(faults=(Fault(call="chat", ordinal=1, kind="transient"),)), directory=tmp_path
    )
    recorder, trace = Recorder(), Trace()
    models = factory(calls, recorder, trace)
    await models.create_chat_model(context()).invoke(
        messages(), (), {"graph_node": "plan", "prompt_version": "sha256:" + "a" * 64}
    )
    await models.create_embedding_model(context()).embed(
        [CANARY], {"graph_node": "retrieve_documents"}
    )
    received = [item for item in calls.records if item.phase == "started"]
    assert [item.invocation_id for item in received] == [
        item.invocation_id for item in recorder.starts
    ]
    assert [outcome.status for _, outcome in recorder.ends] == ["failed", "succeeded", "succeeded"]
    assert len(trace.starts) == len(trace.ends) == 3
    assert trace.starts[1].attempt_number == 2
    for path in tmp_path.iterdir():
        if CANARY in path.read_text():
            pytest.fail("call_record_leak", pytrace=False)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_factory_prepare_failure_prevents_adapter_dispatch():
    calls = Calls(profile())
    with pytest.raises(LLMAccountingError):
        await (
            factory(calls, Recorder(fail_prepare=True))
            .create_embedding_model(context())
            .embed([CANARY], {"graph_node": "ingest"})
        )
    assert calls.records == []


async def test_factory_permanent_failure_does_not_retry_and_exhaustion_is_bounded():
    for fault, count in (("permanent", 1), ("transient", 3)):
        calls = Calls(
            configured(
                faults=tuple(
                    Fault(call="embedding", ordinal=index, kind=fault) for index in range(1, 4)
                )
            )
        )
        recorder = Recorder()
        with pytest.raises(LLMProviderError):
            await (
                factory(calls, recorder)
                .create_embedding_model(context())
                .embed([CANARY], {"graph_node": "ingest"})
            )
        assert len(recorder.starts) == len(recorder.ends) == calls.counts["embedding"] == count


async def test_factory_cancellation_is_accounted_and_not_retried():
    calls = Calls(
        configured(
            name="delayed-v1", delays={kind: {"minimum": 2.0, "maximum": 2.0} for kind in KINDS}
        )
    )
    recorder = Recorder()
    task = asyncio.create_task(
        factory(calls, recorder)
        .create_embedding_model(context())
        .embed([CANARY], {"graph_node": "ingest"})
    )
    async with asyncio.timeout(0.5):
        while not calls.records:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(recorder.ends) == 1
    assert recorder.ends[0][1].error_category == "cancelled"


async def test_finish_report_failure_does_not_replace_cancellation():
    class FailingFinish(Calls):
        def record(self, record):
            if record.phase == "finished":
                raise EnvironmentError("report_failed")
            super().record(record)

    calls = FailingFinish(configured(name="delayed-v1"))
    task = asyncio.create_task(calls.invoke("chat", noop))
    async with asyncio.timeout(0.5):
        while not calls.records:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls.records) == 1


class ToolRecorder(InMemoryToolInvocationRecorder):
    def __init__(self):
        super().__init__()
        self.attempts = []

    async def start_attempt(self, **kwargs):
        self.attempts.append(kwargs)


def search_runtime(calls):
    recorder = ToolRecorder()
    runtime = create_search_web_tool_registry(Search(calls), recorder=recorder).bind(
        policy_name=RESEARCH_TOOL_POLICY_NAME,
        context=ToolRunContext(
            workspace_id=uuid4(),
            actor_user_id=uuid4(),
            run_id=uuid4(),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=monotonic() + 5,
            cancellation=SimpleNamespace(is_cancelled=lambda: False),
        ),
    )
    return runtime, recorder


async def test_registry_owns_search_retry_and_rejects_trusted_input_fields():
    calls = Calls(configured(faults=(Fault(call="search", ordinal=1, kind="transient"),)))
    runtime, recorder = search_runtime(calls)
    output = await runtime.execute(
        ModelToolCall(
            call_id="search-1", name="search_web", arguments={"query": QUERY, "max_results": 1}
        )
    )
    assert json.loads(output)["result_count"] == 1
    assert len(recorder.attempts) == calls.counts["search"] == 2
    with pytest.raises(ToolInputValidationError):
        await runtime.execute(
            ModelToolCall(
                call_id="search-2",
                name="search_web",
                arguments={"query": QUERY, "max_results": 1, "workspace_id": str(uuid4())},
            )
        )
    assert calls.counts["search"] == 2


async def test_registry_propagates_permanent_search_failure_without_retry():
    calls = Calls(configured(faults=(Fault(call="search", ordinal=1, kind="permanent"),)))
    runtime, recorder = search_runtime(calls)
    with pytest.raises(ToolExecutionError):
        await runtime.execute(
            ModelToolCall(
                call_id="search-1", name="search_web", arguments={"query": QUERY, "max_results": 1}
            )
        )
    assert len(recorder.attempts) == calls.counts["search"] == 1


async def test_search_typed_timeout_fault_and_expired_deadline():
    calls = Calls(configured(faults=(Fault(call="search", ordinal=1, kind="timeout"),)))
    with pytest.raises(SearchTransientError) as failure:
        await Search(calls).search(QUERY, 1, monotonic() + 1)
    assert failure.value.category == "provider_timeout"
    with pytest.raises(TimeoutError):
        await Search(calls).search(QUERY, 1, monotonic() - 1)


async def test_mock_lost_response_occurs_after_actual_adapter_response_and_only_explicit_lookup():
    action_id = uuid4()
    args = SubmitApplicationArgsV1(
        job_ref="synthetic", resume_document_id=uuid4(), cover_letter=CANARY
    )
    request = MockSubmissionRequestV1(
        workspace_id=uuid4(),
        originating_actor_user_id=uuid4(),
        run_id=uuid4(),
        action_intent_id=action_id,
        payload=args,
    )
    methods = []
    response = {
        **request.model_dump(mode="json", exclude={"payload"}),
        "id": str(uuid4()),
        "idempotency_key": str(action_id),
        "payload_digest": mock_submission_payload_digest(request),
        "external_ref": "mock-e53",
        "created_at": datetime.now(UTC).isoformat(),
        "created": True,
    }

    def handler(req):
        methods.append(req.method)
        return httpx.Response(201 if req.method == "POST" else 200, json=response)

    calls = Calls(
        configured(
            faults=(
                Fault(call="mock_submit", ordinal=1, kind="response_lost"),
                Fault(call="mock_lookup", ordinal=1, kind="transient"),
            )
        )
    )
    kwargs = dict(
        workspace_id=request.workspace_id,
        actor_user_id=request.originating_actor_user_id,
        run_id=request.run_id,
        action_intent_id=action_id,
        idempotency_key=str(action_id),
        payload=args,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:1234", trust_env=False
    ) as client:
        adapter = Mock(client, calls)
        with pytest.raises(MockPortalTransportError):
            await adapter.submit(**kwargs)
        assert methods == ["POST"]
        with pytest.raises(MockPortalTransportError):
            await adapter.get_by_idempotency_key(**kwargs)
        assert methods == ["POST"]
        result = await adapter.get_by_idempotency_key(**kwargs)
        assert result.payload_digest == response["payload_digest"]
    assert methods == ["POST", "GET"]
    assert [item.outcome for item in calls.records if item.phase == "finished"] == [
        "response_lost",
        "failed",
        "succeeded",
    ]


def test_patch_context_restores_each_constructor_after_error():
    names = (
        "DeterministicResearchFakeChatAdapter",
        "FakeEmbeddingModel",
        "FakeSearch",
        "MockPortalHTTPAdapter",
    )
    worker = SimpleNamespace(**{name: object() for name in names})
    original = vars(worker).copy()
    with pytest.raises(RuntimeError), worker_adapters(worker, Calls(profile())):
        assert isinstance(worker.DeterministicResearchFakeChatAdapter(), Chat)
        assert isinstance(worker.FakeEmbeddingModel(), Embedding)
        raise RuntimeError("synthetic_stop")
    assert vars(worker) == original


async def test_report_failure_prevents_dispatch_and_existing_files_are_unchanged(tmp_path):
    record = CallRecord(
        process="worker",
        sequence=1,
        call="chat",
        ordinal=1,
        delay_seconds=0.0,
        phase="started",
        outcome="pending",
        elapsed_seconds=0.0,
    )
    name = "calls-worker-001-started.json"
    publish(tmp_path, name, record)
    original = (tmp_path / name).read_bytes()
    called = []

    async def operation():
        called.append(True)

    with pytest.raises(EnvironmentError, match="report_failed"):
        await Calls(profile(), directory=tmp_path).invoke("chat", operation)
    assert called == [] and (tmp_path / name).read_bytes() == original
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "smoke-result.json", record.model_copy(update={"outcome": CANARY}))
    assert not (tmp_path / "smoke-result.json").exists()


def test_record_schema_rejects_business_body_and_arbitrary_diagnostics():
    with pytest.raises(ValidationError):
        CallRecord(
            process="worker",
            sequence=1,
            call="chat",
            ordinal=1,
            delay_seconds=0.0,
            phase="finished",
            outcome="succeeded",
            elapsed_seconds=0.0,
            query=CANARY,
        )


async def test_wrappers_forward_provider_attempt_identity_without_metadata_changes():
    calls = Calls(profile())
    attempt = ProviderAttemptContext(invocation_id=uuid4(), timeout_seconds=1.0)
    result = await Embedding(calls).embed([CANARY], {"graph_node": "ingest"}, attempt=attempt)
    assert len(result.vectors[0]) == 1536
    assert calls.records[0].invocation_id == attempt.invocation_id


@pytest.mark.parametrize("kind", ["chat", "embedding"])
async def test_model_timeout_fault_uses_provider_typed_error(kind):
    calls = Calls(configured(faults=(Fault(call=kind, ordinal=1, kind="timeout"),)))
    with pytest.raises(ProviderAdapterError) as error:
        await calls.invoke(kind, noop)
    assert error.value.category == "provider_timeout"


def test_invalid_call_process_and_search_permanent_error():
    with pytest.raises(ValueError, match="invalid_process"):
        Calls(profile(), process=CANARY)
    with pytest.raises(SearchPermanentError):
        Calls.fail("search", "permanent")
