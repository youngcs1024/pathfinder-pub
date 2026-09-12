from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from uuid import UUID

import pytest

from app.domain.errors import DomainUnavailableError
from app.llm.factory import (
    LLMAccountingError,
    LLMFactory,
    LLMFactoryConfigurationError,
    LLMProviderError,
    LLMRetryPolicy,
)
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import (
    InvocationRecorderPort,
    LLMInvocationAttempt,
    LLMInvocationContext,
    LLMInvocationOutcome,
    LLMTraceStart,
    NoOpTraceSink,
    TraceIdentifiers,
    TraceSinkPort,
)
from app.llm.ports import (
    ChatMessage,
    ChatModelResult,
    EmbeddingResult,
    ModelToolSchema,
    ModelUsage,
    ProviderAdapterError,
    ProviderAttemptContext,
)
from app.llm.pricing import QWEN_BEIJING_PRICE_BOOK, QWEN_BEIJING_PRICING_VERSION

PROMPT_VERSION = f"sha256:{'a' * 64}"
CHAT_METADATA = {
    "graph_node": "plan",
    "prompt_version": PROMPT_VERSION,
    "request_label": "metadata-canary",
}
CHAT_MESSAGES = (ChatMessage(role="user", content="sensitive-request-canary"),)


@dataclass
class _RecordingRecorder:
    events: list[tuple[str, LLMInvocationAttempt, LLMInvocationOutcome | None]] = field(
        default_factory=list
    )
    prepare_error: Exception | None = None
    finalize_error: Exception | None = None

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        self.events.append(("prepare", attempt, None))
        if self.prepare_error is not None:
            raise self.prepare_error

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        self.events.append(("finalize", attempt, outcome))
        if self.finalize_error is not None:
            raise self.finalize_error


class _BlockingFinalizeRecorder(_RecordingRecorder):
    def __init__(self) -> None:
        super().__init__()
        self.finalize_started = asyncio.Event()
        self.release_finalize = asyncio.Event()

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        self.finalize_started.set()
        await self.release_finalize.wait()
        await super().finalize(attempt, outcome)


@dataclass
class _RecordingTraceSink:
    starts: list[LLMTraceStart] = field(default_factory=list)
    finishes: list[tuple[TraceIdentifiers, LLMInvocationOutcome | None]] = field(
        default_factory=list
    )
    start_error: Exception | None = None
    finish_error: Exception | None = None

    def start(self, trace: LLMTraceStart) -> TraceIdentifiers:
        self.starts.append(trace)
        if self.start_error is not None:
            raise self.start_error
        correlation_id = trace.run_id or trace.request_id or trace.invocation_id
        trace_value = max(correlation_id.int & ((1 << 128) - 1), 1)
        observation_value = max(trace.invocation_id.int & ((1 << 64) - 1), 1)
        return TraceIdentifiers(
            trace_id=f"{trace_value:032x}",
            observation_id=f"{observation_value:016x}",
        )

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None:
        self.finishes.append((identifiers, outcome))
        if self.finish_error is not None:
            raise self.finish_error


class _ClockAdvancingRecorder(_RecordingRecorder):
    def __init__(self, clock: _MutableClock) -> None:
        super().__init__()
        self.clock = clock

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        await super().prepare(attempt)
        self.clock.value = 11.0


class _CountingChatAdapter:
    provider = "fake"
    model = "qwen3.6-flash-2026-04-16"

    def __init__(self, recorder: _RecordingRecorder) -> None:
        self.recorder = recorder
        self.call_count = 0

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        assert [event[0] for event in self.recorder.events] == ["prepare"]
        self.call_count += 1
        return ChatModelResult(
            content="accounted fake result",
            usage=ModelUsage(input_tokens=11, output_tokens=13),
        )


class _FailingChatAdapter:
    provider = "fake"
    model = "qwen3.6-flash-2026-04-16"

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.call_count = 0

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        self.call_count += 1
        raise self.error


class _InvalidChatAdapter:
    provider = "fake"
    model = "qwen3.6-flash-2026-04-16"

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> object:
        return object()


class _BlockingChatAdapter:
    provider = "fake"
    model = "qwen3.6-flash-2026-04-16"

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocking adapter unexpectedly resumed")


class _ScriptedQwenChatAdapter:
    provider = "qwen"
    model = "qwen3.6-flash-2026-04-16"

    def __init__(self, script: Sequence[ChatModelResult | ProviderAdapterError]) -> None:
        self.script = list(script)
        self.attempts: list[ProviderAttemptContext] = []

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        self.attempts.append(attempt)
        step = self.script.pop(0)
        if isinstance(step, ProviderAdapterError):
            raise step
        return step


class _UnusedQwenEmbeddingAdapter:
    provider = "qwen"
    model = "text-embedding-v4"

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> EmbeddingResult:
        raise AssertionError("embedding adapter should not be called by chat tests")


class _CostedQwenEmbeddingAdapter:
    provider = "qwen"
    model = "text-embedding-v4"

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=tuple(
                tuple(float(index) for _ in range(1536)) for index, _ in enumerate(texts, start=1)
            ),
            usage=ModelUsage(input_tokens=17, output_tokens=0, total_tokens=17),
            provider="qwen",
        )


class _ExplodingPriceBook:
    version = QWEN_BEIJING_PRICING_VERSION
    currency = "CNY"

    def estimate(self, **kwargs: object) -> None:
        raise RuntimeError("pricing-secret-canary")


class _Clock:
    def __init__(self, *values: float) -> None:
        if not values:
            raise ValueError("clock requires at least one value")
        self.values = iter(values)
        self.last = values[-1]

    def __call__(self) -> float:
        try:
            self.last = next(self.values)
        except StopIteration:
            pass
        return self.last


class _MutableClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _factory(
    recorder: InvocationRecorderPort,
    *,
    chat_adapter: object | None = None,
    clock: object | None = None,
    provider: str = "fake",
    retry_policy: LLMRetryPolicy | None = None,
    invocation_id_factory: object | None = None,
    random_source: object | None = None,
    sleeper: object | None = None,
    trace_sink: TraceSinkPort | None = None,
    price_book: object | None = None,
) -> LLMFactory:
    return LLMFactory(
        recorder=recorder,
        chat_adapter=chat_adapter or FakeChatModel(),  # type: ignore[arg-type]
        embedding_adapter=FakeEmbeddingModel(),
        provider=provider,  # type: ignore[arg-type]
        retry_policy=retry_policy or LLMRetryPolicy(),
        invocation_id_factory=invocation_id_factory or (lambda: UUID(int=10)),  # type: ignore[arg-type]
        clock=clock or _Clock(10.0, 10.0, 10.0, 10.007),  # type: ignore[arg-type]
        random_source=random_source or (lambda: 0.5),  # type: ignore[arg-type]
        sleeper=sleeper or asyncio.sleep,  # type: ignore[arg-type]
        trace_sink=trace_sink or NoOpTraceSink(),
        price_book=price_book or QWEN_BEIJING_PRICE_BOOK,  # type: ignore[arg-type]
    )


def _qwen_factory(
    recorder: InvocationRecorderPort,
    *,
    chat_adapter: _ScriptedQwenChatAdapter,
    clock: object,
    invocation_id_factory: object,
    random_source: object = lambda: 0.5,
    sleeper: object = asyncio.sleep,
    retry_policy: LLMRetryPolicy | None = None,
    trace_sink: TraceSinkPort | None = None,
) -> LLMFactory:
    return LLMFactory(
        recorder=recorder,
        chat_adapter=chat_adapter,
        embedding_adapter=_UnusedQwenEmbeddingAdapter(),
        provider="qwen",
        retry_policy=retry_policy or LLMRetryPolicy(),
        invocation_id_factory=invocation_id_factory,  # type: ignore[arg-type]
        clock=clock,  # type: ignore[arg-type]
        random_source=random_source,  # type: ignore[arg-type]
        sleeper=sleeper,  # type: ignore[arg-type]
        trace_sink=trace_sink or NoOpTraceSink(),
    )


def _context(*, request_id: UUID | None = None, run_id: UUID | None = None) -> LLMInvocationContext:
    return LLMInvocationContext(
        workspace_id=UUID(int=1),
        actor_user_id=UUID(int=2),
        request_id=request_id,
        run_id=run_id,
    )


async def test_chat_prepare_commits_before_adapter_and_success_finalizes_usage() -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    adapter = _CountingChatAdapter(recorder)
    model = _factory(recorder, chat_adapter=adapter, trace_sink=trace_sink).create_chat_model(
        _context(request_id=UUID(int=3), run_id=UUID(int=4))
    )

    result = await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert result.content == "accounted fake result"
    assert adapter.call_count == 1
    assert [event[0] for event in recorder.events] == ["prepare", "finalize"]
    attempt = recorder.events[0][1]
    outcome = recorder.events[1][2]
    assert attempt.workspace_id == _context().workspace_id
    assert attempt.actor_user_id == _context().actor_user_id
    assert attempt.run_id == UUID(int=4)
    assert attempt.provider == "fake"
    assert attempt.model == "qwen3.6-flash-2026-04-16"
    assert attempt.graph_node == "plan"
    assert attempt.prompt_version == PROMPT_VERSION
    assert "sensitive" not in attempt.request_hash
    assert "metadata" not in attempt.request_hash
    expected_trace_ids = TraceIdentifiers(trace_id=f"{4:032x}", observation_id=f"{10:016x}")
    assert outcome == LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(input_tokens=11, output_tokens=13),
        latency_ms=7,
        trace_ids=expected_trace_ids,
    )
    assert [item.run_id for item in trace_sink.starts] == [UUID(int=4)]
    assert trace_sink.finishes == [(expected_trace_ids, outcome)]
    trace_payload = repr((trace_sink.starts, trace_sink.finishes))
    for canary in (
        "sensitive-request-canary",
        "metadata-canary",
        "prompt-document-canary",
        "trusted-context-canary",
    ):
        assert canary not in trace_payload


async def test_embedding_uses_the_same_prepare_call_finalize_boundary() -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    model = _factory(recorder, trace_sink=trace_sink).create_embedding_model(_context())

    result = await model.embed(("alpha", "beta"), {"graph_node": "ingest_documents"})

    assert isinstance(result, EmbeddingResult)
    attempt = recorder.events[0][1]
    outcome = recorder.events[1][2]
    assert attempt.invocation_kind == "embedding"
    assert attempt.model == "text-embedding-v4"
    assert attempt.prompt_version is None
    assert outcome == LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(input_tokens=0, output_tokens=0),
        latency_ms=7,
        trace_ids=TraceIdentifiers(trace_id=f"{10:032x}", observation_id=f"{10:016x}"),
    )
    assert trace_sink.starts[0].invocation_kind == "embedding"
    assert trace_sink.finishes[0][1] == outcome


async def test_run_id_is_accounting_context_not_request_hash_input() -> None:
    recorder = _RecordingRecorder()
    first_run_id = UUID(int=101)
    second_run_id = UUID(int=102)

    first = _factory(
        recorder,
        invocation_id_factory=lambda: UUID(int=201),
    ).create_chat_model(_context(run_id=first_run_id))
    second = _factory(
        recorder,
        invocation_id_factory=lambda: UUID(int=202),
    ).create_chat_model(_context(run_id=second_run_id))

    await first.invoke(CHAT_MESSAGES, (), CHAT_METADATA)
    await second.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    prepared = [event[1] for event in recorder.events if event[0] == "prepare"]
    assert [attempt.run_id for attempt in prepared] == [first_run_id, second_run_id]
    assert len({attempt.request_hash for attempt in prepared}) == 1


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (TimeoutError("timeout-secret-canary"), "provider_timeout"),
        (RuntimeError("provider-secret-canary"), "provider_error"),
    ],
)
async def test_provider_failures_are_safely_categorized_and_finalized(
    error: Exception,
    expected_category: str,
) -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    adapter = _FailingChatAdapter(error)
    model = _factory(recorder, chat_adapter=adapter, trace_sink=trace_sink).create_chat_model(
        _context()
    )

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.category == expected_category
    assert "secret-canary" not in str(captured.value)
    assert adapter.call_count == 1
    outcome = recorder.events[-1][2]
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.error_category == expected_category
    assert len(trace_sink.starts) == 1
    assert trace_sink.finishes == [(outcome.trace_ids, outcome)]


async def test_invalid_adapter_result_is_failed_not_returned() -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    model = _factory(
        recorder,
        chat_adapter=_InvalidChatAdapter(),
        trace_sink=trace_sink,
    ).create_chat_model(_context())

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.category == "invalid_provider_response"
    assert recorder.events[-1][2] is not None
    assert recorder.events[-1][2].error_category == "invalid_provider_response"
    assert len(trace_sink.finishes) == 1
    assert trace_sink.finishes[0][1] == recorder.events[-1][2]


@pytest.mark.parametrize("with_parent", [False, True])
async def test_cancellation_is_recorded_then_propagated(with_parent) -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    adapter = _BlockingChatAdapter()
    model = _factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.005),
        trace_sink=trace_sink,
    ).create_chat_model(_context(run_id=UUID(int=19)))
    from app.domain.tracing import bind_trace_scope

    scope = _application_scope() if with_parent else None
    with bind_trace_scope(scope):
        task = asyncio.create_task(model.invoke(CHAT_MESSAGES, (), CHAT_METADATA))
    await adapter.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    outcome = recorder.events[-1][2]
    assert outcome is not None
    assert outcome.status == "failed"
    assert outcome.latency_ms == 5
    assert outcome.error_category == "cancelled"
    assert trace_sink.finishes == [(outcome.trace_ids, outcome)]


async def test_cancellation_during_success_finalize_waits_for_accounting() -> None:
    recorder = _BlockingFinalizeRecorder()
    trace_sink = _RecordingTraceSink()
    model = _factory(recorder, trace_sink=trace_sink).create_embedding_model(_context())
    task = asyncio.create_task(model.embed(("alpha",), {"graph_node": "retrieve_documents"}))
    await recorder.finalize_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    recorder.release_finalize.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [event[0] for event in recorder.events] == ["prepare", "finalize"]
    outcome = recorder.events[-1][2]
    assert outcome is not None
    assert outcome.status == "succeeded"
    assert trace_sink.finishes == [(outcome.trace_ids, outcome)]


@pytest.mark.parametrize("temporary", [False, True])
async def test_prepare_failure_prevents_provider_call(temporary) -> None:
    error = DomainUnavailableError() if temporary else RuntimeError("database-secret-canary")
    recorder = _RecordingRecorder(prepare_error=error)
    trace_sink = _RecordingTraceSink()
    adapter = _CountingChatAdapter(recorder)
    model = _factory(recorder, chat_adapter=adapter, trace_sink=trace_sink).create_chat_model(
        _context()
    )

    with pytest.raises(LLMAccountingError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.phase == "prepare"
    assert captured.value.retryable is temporary
    assert "secret-canary" not in str(captured.value)
    assert adapter.call_count == 0
    assert [event[0] for event in recorder.events] == ["prepare"]
    assert trace_sink.starts == []
    assert trace_sink.finishes == []


@pytest.mark.parametrize("temporary", [False, True])
async def test_finalize_failure_never_returns_provider_result(temporary) -> None:
    error = DomainUnavailableError() if temporary else RuntimeError("database-secret-canary")
    recorder = _RecordingRecorder(finalize_error=error)
    trace_sink = _RecordingTraceSink()
    adapter = _CountingChatAdapter(recorder)
    model = _factory(recorder, chat_adapter=adapter, trace_sink=trace_sink).create_chat_model(
        _context()
    )

    with pytest.raises(LLMAccountingError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.phase == "finalize"
    assert captured.value.retryable is temporary
    assert "secret-canary" not in str(captured.value)
    assert adapter.call_count == 1
    assert len(trace_sink.finishes) == 1
    assert trace_sink.finishes[0][1] is None


@pytest.mark.parametrize("failure_phase", ["start", "finish"])
@pytest.mark.parametrize("with_parent", [False, True])
async def test_trace_sink_failures_do_not_change_provider_or_accounting_behavior(
    failure_phase: str,
    with_parent: bool,
) -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink(
        start_error=RuntimeError("trace-start-secret-canary") if failure_phase == "start" else None,
        finish_error=RuntimeError("trace-finish-secret-canary")
        if failure_phase == "finish"
        else None,
    )
    adapter = _CountingChatAdapter(recorder)
    model = _factory(
        recorder,
        chat_adapter=adapter,
        trace_sink=trace_sink,
    ).create_chat_model(_context(run_id=UUID(int=19)))

    from app.domain.tracing import bind_trace_scope

    scope = _application_scope() if with_parent else None
    with bind_trace_scope(scope):
        result = await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)
    assert trace_sink.starts[0].parent_context == (scope.parent if scope else None)

    assert result.content == "accounted fake result"
    assert [event[0] for event in recorder.events] == ["prepare", "finalize"]
    assert {event[1].workspace_id for event in recorder.events} == {_context().workspace_id}
    assert {event[1].actor_user_id for event in recorder.events} == {_context().actor_user_id}
    outcome = recorder.events[-1][2]
    assert outcome is not None
    if failure_phase == "start":
        assert outcome.trace_ids is None
        assert trace_sink.finishes == []
    else:
        assert outcome.trace_ids is not None
        assert len(trace_sink.finishes) == 1


async def test_pricing_failure_closes_observation_as_accounting_failure() -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    adapter = _CountingChatAdapter(recorder)
    model = _factory(
        recorder,
        chat_adapter=adapter,
        trace_sink=trace_sink,
        price_book=_ExplodingPriceBook(),
    ).create_chat_model(_context())

    with pytest.raises(LLMAccountingError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.phase == "pricing"
    assert [event[0] for event in recorder.events] == ["prepare"]
    assert len(trace_sink.finishes) == 1
    assert trace_sink.finishes[0][1] is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("provider", "anthropic"),
        ("chat_model", "request-selected-model"),
        ("embedding_model", "request-selected-embedding"),
        ("price_book", object()),
    ],
)
def test_factory_rejects_unknown_provider_and_profile_overrides(
    field_name: str,
    field_value: object,
) -> None:
    arguments: dict[str, object] = {
        "recorder": _RecordingRecorder(),
        "chat_adapter": FakeChatModel(),
        "embedding_adapter": FakeEmbeddingModel(),
        field_name: field_value,
    }
    with pytest.raises(LLMFactoryConfigurationError):
        LLMFactory(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("with_parent", [False, True])
async def test_qwen_rate_limit_retry_creates_distinct_accounted_attempts(with_parent) -> None:
    recorder = _RecordingRecorder()
    trace_sink = _RecordingTraceSink()
    adapter = _ScriptedQwenChatAdapter(
        (
            ProviderAdapterError(
                category="rate_limited",
                retryable=True,
                provider_response_id="req_rate_limited",
            ),
            ChatModelResult(
                content="Recovered response",
                usage=ModelUsage(
                    input_tokens=7,
                    output_tokens=9,
                    total_tokens=16,
                    cached_input_tokens=0,
                    reasoning_output_tokens=4,
                ),
                provider="qwen",
                provider_response_id="req_success",
            ),
        )
    )
    invocation_ids = iter((UUID(int=21), UUID(int=22)))
    sleeps: list[float] = []

    async def sleeper(delay: float) -> None:
        sleeps.append(delay)

    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.01, 10.01, 10.01, 10.01, 10.02),
        invocation_id_factory=lambda: next(invocation_ids),
        random_source=lambda: 0.5,
        sleeper=sleeper,
        trace_sink=trace_sink,
    ).create_chat_model(_context(request_id=UUID(int=20), run_id=UUID(int=19)))

    from app.domain.tracing import bind_trace_scope

    scope = _application_scope(run_id=UUID(int=19)) if with_parent else None
    with bind_trace_scope(scope):
        result = await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)
    assert [t.parent_context for t in trace_sink.starts] == [scope.parent if scope else None] * 2
    assert [t.attempt_number for t in trace_sink.starts] == [1, 2]

    assert result.content == "Recovered response"
    assert sleeps == [0.25]
    prepared = [event[1] for event in recorder.events if event[0] == "prepare"]
    finalized = [event[2] for event in recorder.events if event[0] == "finalize"]
    assert [item.invocation_id for item in prepared] == [UUID(int=21), UUID(int=22)]
    assert len({item.request_hash for item in prepared}) == 1
    assert {item.workspace_id for item in prepared} == {_context().workspace_id}
    assert {item.actor_user_id for item in prepared} == {_context().actor_user_id}
    assert {item.run_id for item in prepared} == {UUID(int=19)}
    assert [item.invocation_id for item in adapter.attempts] == [UUID(int=21), UUID(int=22)]
    assert finalized[0] is not None
    assert finalized[0].error_category == "rate_limited"
    assert finalized[0].provider_response_id == "req_rate_limited"
    assert finalized[0].pricing_version is None
    assert finalized[0].estimated_cost is None
    assert finalized[1] is not None
    assert finalized[1].status == "succeeded"
    assert finalized[1].provider_response_id == "req_success"
    assert finalized[1].pricing_version == QWEN_BEIJING_PRICING_VERSION
    assert finalized[1].currency == "CNY"
    assert finalized[1].estimated_cost == Decimal("0.0000732")
    assert len(trace_sink.starts) == 2
    assert {item.request_id for item in trace_sink.starts} == {UUID(int=20)}
    assert {item.run_id for item in trace_sink.starts} == {UUID(int=19)}
    trace_ids = [item[0] for item in trace_sink.finishes]
    assert [item.trace_id for item in trace_ids] == [f"{19:032x}", f"{19:032x}"]
    assert [item.observation_id for item in trace_ids] == [f"{21:016x}", f"{22:016x}"]
    assert [item[1] for item in trace_sink.finishes] == finalized


async def test_qwen_nonzero_cache_usage_is_persisted_without_guessing_cost() -> None:
    recorder = _RecordingRecorder()
    usage = ModelUsage(
        input_tokens=7,
        output_tokens=9,
        total_tokens=16,
        cached_input_tokens=2,
    )
    adapter = _ScriptedQwenChatAdapter(
        (ChatModelResult(content="Unpriced response", usage=usage, provider="qwen"),)
    )
    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.01),
        invocation_id_factory=lambda: UUID(int=23),
    ).create_chat_model(_context())

    result = await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert result.content == "Unpriced response"
    outcome = recorder.events[-1][2]
    assert outcome is not None
    assert outcome.status == "succeeded"
    assert outcome.token_usage == usage
    assert outcome.pricing_version is None
    assert outcome.currency is None
    assert outcome.estimated_cost is None


async def test_qwen_embedding_success_is_costed_on_the_same_attempt() -> None:
    recorder = _RecordingRecorder()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=_ScriptedQwenChatAdapter(()),
        embedding_adapter=_CostedQwenEmbeddingAdapter(),
        provider="qwen",
        invocation_id_factory=lambda: UUID(int=24),
        clock=_Clock(10.0, 10.0, 10.0, 10.01),
    )

    result = await factory.create_embedding_model(_context()).embed(
        ("alpha", "beta"),
        {"graph_node": "ingest_documents"},
    )

    assert result.usage == ModelUsage(input_tokens=17, output_tokens=0, total_tokens=17)
    outcome = recorder.events[-1][2]
    assert outcome is not None
    assert outcome.pricing_version == QWEN_BEIJING_PRICING_VERSION
    assert outcome.currency == "CNY"
    assert outcome.estimated_cost == Decimal("0.0000085")


async def test_retry_after_is_honored_without_jitter() -> None:
    recorder = _RecordingRecorder()
    adapter = _ScriptedQwenChatAdapter(
        (
            ProviderAdapterError(
                category="provider_unavailable",
                retryable=True,
                retry_after_seconds=1.75,
            ),
            ChatModelResult(content="Recovered", provider="qwen"),
        )
    )
    sleeps: list[float] = []

    async def sleeper(delay: float) -> None:
        sleeps.append(delay)

    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.01, 10.01, 10.01, 10.01, 10.02),
        invocation_id_factory=iter((UUID(int=31), UUID(int=32))).__next__,
        random_source=lambda: (_ for _ in ()).throw(AssertionError("jitter must not run")),
        sleeper=sleeper,
    ).create_chat_model(_context())

    await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert sleeps == [1.75]


async def test_retry_after_that_exceeds_remaining_deadline_stops_before_next_attempt() -> None:
    recorder = _RecordingRecorder()
    adapter = _ScriptedQwenChatAdapter(
        (
            ProviderAdapterError(
                category="rate_limited",
                retryable=True,
                retry_after_seconds=5.0,
            ),
        )
    )
    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.2, 10.2),
        invocation_id_factory=lambda: UUID(int=41),
        retry_policy=LLMRetryPolicy(deadline_seconds=1.0),
        sleeper=asyncio.sleep,
    ).create_chat_model(_context())

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.category == "rate_limited"
    assert len([event for event in recorder.events if event[0] == "prepare"]) == 1
    assert len(adapter.attempts) == 1


@pytest.mark.parametrize(
    ("category", "retryable"),
    [
        ("provider_authentication", False),
        ("provider_rejected", False),
        ("invalid_provider_response", False),
    ],
)
async def test_non_retryable_qwen_failures_stop_after_one_attempt(
    category: str,
    retryable: bool,
) -> None:
    recorder = _RecordingRecorder()
    adapter = _ScriptedQwenChatAdapter(
        (
            ProviderAdapterError(
                category=category,  # type: ignore[arg-type]
                retryable=retryable,
            ),
        )
    )
    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=_Clock(10.0, 10.0, 10.0, 10.01),
        invocation_id_factory=lambda: UUID(int=51),
    ).create_chat_model(_context())

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.category == category
    assert len(adapter.attempts) == 1
    assert len([event for event in recorder.events if event[0] == "prepare"]) == 1


def test_factory_rejects_provider_adapter_profile_mismatch_before_accounting() -> None:
    with pytest.raises(LLMFactoryConfigurationError):
        LLMFactory(
            recorder=_RecordingRecorder(),
            chat_adapter=FakeChatModel(),
            embedding_adapter=FakeEmbeddingModel(),
            provider="qwen",
        )


async def test_deadline_consumed_by_prepare_finalizes_without_provider_request() -> None:
    clock = _MutableClock(10.0)
    recorder = _ClockAdvancingRecorder(clock)
    trace_sink = _RecordingTraceSink()
    adapter = _ScriptedQwenChatAdapter((ChatModelResult(content="must not run", provider="qwen"),))
    model = _qwen_factory(
        recorder,
        chat_adapter=adapter,
        clock=clock,
        invocation_id_factory=lambda: UUID(int=61),
        retry_policy=LLMRetryPolicy(deadline_seconds=1.0),
        trace_sink=trace_sink,
    ).create_chat_model(_context())

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)

    assert captured.value.category == "provider_timeout"
    assert adapter.attempts == []
    assert [event[0] for event in recorder.events] == ["prepare", "finalize"]
    assert recorder.events[-1][2] is not None
    assert recorder.events[-1][2].error_category == "provider_timeout"
    assert len(trace_sink.starts) == 1
    assert trace_sink.finishes[0][1] == recorder.events[-1][2]


def _application_scope(*, workspace_id=None, run_id=None, parent=True):
    from app.domain.tracing import (
        ActiveTraceScope,
        ExecutionSegmentIdentity,
        TraceIdentity,
        TraceParentContext,
    )

    identity = TraceIdentity(
        workspace_id=workspace_id or UUID(int=1), run_id=run_id or UUID(int=19)
    )
    segment = ExecutionSegmentIdentity(job_id=UUID(int=30), attempt=1)
    return ActiveTraceScope(
        trace_identity=identity,
        segment_identity=segment,
        parent=TraceParentContext(
            trace_identity=identity,
            segment_identity=segment,
            span_kind="graph_node",
            context_id=UUID(int=40),
        )
        if parent
        else None,
    )


@pytest.mark.parametrize("mode", ["valid", "absent", "no_parent", "workspace", "run"])
@pytest.mark.parametrize("kind", ["chat", "embedding"])
async def test_application_parent_capture_drops_incompatible_scope_only(mode, kind):
    from app.domain.tracing import bind_trace_scope

    scope = (
        None
        if mode == "absent"
        else _application_scope(
            workspace_id=UUID(int=99) if mode == "workspace" else UUID(int=1),
            run_id=UUID(int=99) if mode == "run" else UUID(int=19),
            parent=mode != "no_parent",
        )
    )
    recorder = _RecordingRecorder()
    sink = _RecordingTraceSink()
    factory = _factory(recorder, trace_sink=sink)
    with bind_trace_scope(scope):
        if kind == "chat":
            await factory.create_chat_model(_context(run_id=UUID(int=19))).invoke(
                CHAT_MESSAGES, (), CHAT_METADATA
            )
        else:
            await factory.create_embedding_model(_context(run_id=UUID(int=19))).embed(
                ("PRIVATE",), {"graph_node": "retrieve_documents"}
            )
    assert [e[0] for e in recorder.events] == ["prepare", "finalize"]
    assert recorder.events[-1][2].status == "succeeded"
    if mode in {"valid", "absent"}:
        assert len(sink.starts) == len(sink.finishes) == 1
        assert sink.starts[0].parent_context == (scope.parent if scope else None)
    else:
        assert sink.starts == sink.finishes == []
        assert recorder.events[-1][2].trace_ids is None


@pytest.mark.parametrize("field", ["workspace_id", "run_id"])
def test_llm_parent_contract_rejects_identity_mismatch(field):
    from pydantic import ValidationError

    values = dict(
        parent_context=_application_scope().parent,
        workspace_id=UUID(int=1),
        run_id=UUID(int=19),
        invocation_id=UUID(int=10),
        graph_node="plan",
        prompt_version=PROMPT_VERSION,
        provider="fake",
        model="qwen3.6-flash-2026-04-16",
        invocation_kind="chat",
        attempt_number=1,
    )
    values[field] = UUID(int=99)
    with pytest.raises(ValidationError, match="parent identity mismatch"):
        LLMTraceStart(**values)


async def test_provider_cancellation_is_not_replaced_by_accounting_failure():
    recorder = _RecordingRecorder(finalize_error=DomainUnavailableError())
    adapter = _FailingChatAdapter(asyncio.CancelledError())
    model = _factory(recorder, chat_adapter=adapter).create_chat_model(_context())
    with pytest.raises(asyncio.CancelledError):
        await model.invoke(CHAT_MESSAGES, (), CHAT_METADATA)
    assert adapter.call_count == 1
    assert [event[0] for event in recorder.events] == ["prepare", "finalize"]
