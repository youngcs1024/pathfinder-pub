from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from random import random
from time import monotonic
from typing import Literal, Never
from uuid import UUID, uuid4

from app.domain.errors import DomainUnavailableError
from app.domain.tracing import current_trace_scope
from app.llm.invocations import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_MODEL,
    InvocationRecorderPort,
    LLMInvocationAttempt,
    LLMInvocationAuthorizationError,
    LLMInvocationContext,
    LLMInvocationErrorCategory,
    LLMInvocationInvariantError,
    LLMInvocationOutcome,
    LLMInvocationProvider,
    LLMTraceStart,
    NoOpTraceSink,
    TraceIdentifiers,
    TraceSinkPort,
    chat_request_hash,
    embedding_request_hash,
    invocation_metadata,
)
from app.llm.ports import (
    ChatAdapterPort,
    ChatMessage,
    ChatModelPort,
    ChatModelResult,
    EmbeddingAdapterPort,
    EmbeddingPort,
    EmbeddingResult,
    ModelToolSchema,
    ModelUsage,
    ProviderAdapterError,
    ProviderAttemptContext,
)
from app.llm.pricing import (
    QWEN_BEIJING_PRICE_BOOK,
    InvocationCost,
    PriceBookPort,
)

type AccountingPhase = Literal["prepare", "finalize", "pricing", "timing"]
type InvocationIdFactory = Callable[[], UUID]
type MonotonicClock = Callable[[], float]
type RandomSource = Callable[[], float]
type AsyncSleeper = Callable[[float], Awaitable[None]]


class LLMFactoryConfigurationError(Exception):
    def __init__(self) -> None:
        super().__init__("LLM factory configuration is invalid")


class LLMAccountingError(Exception):
    def __init__(self, *, phase: AccountingPhase, retryable: bool = False) -> None:
        self.phase = phase
        self.retryable = retryable
        super().__init__(f"LLM invocation accounting failed during {phase}")


class LLMProviderError(Exception):
    def __init__(self, *, category: LLMInvocationErrorCategory) -> None:
        self.category = category
        super().__init__(f"LLM provider invocation failed: {category}")


@dataclass(slots=True, repr=False)
class _TraceAttemptSession:
    sink: TraceSinkPort
    identifiers: TraceIdentifiers | None
    closed: bool = False

    def finish(self, outcome: LLMInvocationOutcome | None) -> None:
        if self.closed:
            return
        self.closed = True
        if self.identifiers is None:
            return
        try:
            self.sink.finish(self.identifiers, outcome)
        except Exception:
            return


def _clock_value(clock: MonotonicClock) -> float:
    try:
        value = clock()
    except Exception:
        raise LLMAccountingError(phase="timing") from None
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(float(value)):
        raise LLMAccountingError(phase="timing")
    return float(value)


def _elapsed_ms(clock: MonotonicClock, started_at: float) -> int:
    elapsed = _clock_value(clock) - started_at
    if elapsed < 0:
        raise LLMAccountingError(phase="timing")
    return round(elapsed * 1000)


@dataclass(frozen=True, slots=True, repr=False)
class LLMRetryPolicy:
    max_attempts: int = 3
    deadline_seconds: float = 120.0
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0

    def __post_init__(self) -> None:
        numeric_values = (
            self.deadline_seconds,
            self.base_delay_seconds,
            self.max_delay_seconds,
        )
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or not 1 <= self.max_attempts <= 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(float(value))
                or float(value) <= 0
                for value in numeric_values
            )
            or self.base_delay_seconds > self.max_delay_seconds
        ):
            raise LLMFactoryConfigurationError


@dataclass(frozen=True, slots=True, repr=False)
class LLMFactory:
    recorder: InvocationRecorderPort
    chat_adapter: ChatAdapterPort
    embedding_adapter: EmbeddingAdapterPort
    trace_sink: TraceSinkPort = field(default_factory=NoOpTraceSink)
    provider: LLMInvocationProvider = "fake"
    chat_model: str = LOCKED_CHAT_MODEL
    embedding_model: str = LOCKED_EMBEDDING_MODEL
    price_book: PriceBookPort = QWEN_BEIJING_PRICE_BOOK
    retry_policy: LLMRetryPolicy = LLMRetryPolicy()
    invocation_id_factory: InvocationIdFactory = uuid4
    clock: MonotonicClock = monotonic
    random_source: RandomSource = random
    sleeper: AsyncSleeper = asyncio.sleep

    def __post_init__(self) -> None:
        if (
            not isinstance(self.recorder, InvocationRecorderPort)
            or not isinstance(self.chat_adapter, ChatAdapterPort)
            or not isinstance(self.embedding_adapter, EmbeddingAdapterPort)
            or not isinstance(self.trace_sink, TraceSinkPort)
            or self.provider not in {"fake", "qwen"}
            or self.chat_adapter.provider != self.provider
            or self.embedding_adapter.provider != self.provider
            or self.chat_adapter.model != self.chat_model
            or self.embedding_adapter.model != self.embedding_model
            or self.chat_model != LOCKED_CHAT_MODEL
            or self.embedding_model != LOCKED_EMBEDDING_MODEL
            or not isinstance(self.price_book, PriceBookPort)
            or not isinstance(self.retry_policy, LLMRetryPolicy)
            or not callable(self.invocation_id_factory)
            or not callable(self.clock)
            or not callable(self.random_source)
            or not callable(self.sleeper)
        ):
            raise LLMFactoryConfigurationError

    def create_chat_model(self, context: LLMInvocationContext) -> ChatModelPort:
        if not isinstance(context, LLMInvocationContext):
            raise LLMFactoryConfigurationError
        return _AccountedChatModel(
            context=context,
            recorder=self.recorder,
            adapter=self.chat_adapter,
            provider=self.provider,
            model=self.chat_model,
            price_book=self.price_book,
            retry_policy=self.retry_policy,
            invocation_id_factory=self.invocation_id_factory,
            clock=self.clock,
            random_source=self.random_source,
            sleeper=self.sleeper,
            trace_sink=self.trace_sink,
        )

    def create_embedding_model(self, context: LLMInvocationContext) -> EmbeddingPort:
        if not isinstance(context, LLMInvocationContext):
            raise LLMFactoryConfigurationError
        return _AccountedEmbeddingModel(
            context=context,
            recorder=self.recorder,
            adapter=self.embedding_adapter,
            provider=self.provider,
            model=self.embedding_model,
            price_book=self.price_book,
            retry_policy=self.retry_policy,
            invocation_id_factory=self.invocation_id_factory,
            clock=self.clock,
            random_source=self.random_source,
            sleeper=self.sleeper,
            trace_sink=self.trace_sink,
        )


@dataclass(frozen=True, slots=True, repr=False)
class _AccountedModel:
    context: LLMInvocationContext
    recorder: InvocationRecorderPort
    provider: LLMInvocationProvider
    model: str
    price_book: PriceBookPort
    retry_policy: LLMRetryPolicy
    invocation_id_factory: InvocationIdFactory
    clock: MonotonicClock
    random_source: RandomSource
    sleeper: AsyncSleeper
    trace_sink: TraceSinkPort

    def _invocation_id(self) -> UUID:
        try:
            invocation_id = self.invocation_id_factory()
        except Exception:
            raise LLMFactoryConfigurationError from None
        if not isinstance(invocation_id, UUID):
            raise LLMFactoryConfigurationError
        return invocation_id

    async def _prepare(self, attempt: LLMInvocationAttempt) -> None:
        try:
            await self.recorder.prepare(attempt)
        except (LLMInvocationAuthorizationError, LLMInvocationInvariantError):
            raise
        except DomainUnavailableError:
            raise LLMAccountingError(phase="prepare", retryable=True) from None
        except Exception:
            raise LLMAccountingError(phase="prepare") from None

    async def _finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        try:
            await self.recorder.finalize(attempt, outcome)
        except LLMInvocationInvariantError:
            raise
        except DomainUnavailableError:
            raise LLMAccountingError(phase="finalize", retryable=True) from None
        except Exception:
            raise LLMAccountingError(phase="finalize") from None

    def _start_trace(
        self,
        attempt: LLMInvocationAttempt,
        *,
        attempt_number: int,
    ) -> _TraceAttemptSession:
        try:
            scope = current_trace_scope()
            if scope is not None and (
                scope.parent is None
                or scope.trace_identity.workspace_id != attempt.workspace_id
                or scope.trace_identity.run_id != attempt.run_id
            ):
                return _TraceAttemptSession(sink=self.trace_sink, identifiers=None)
            identifiers = self.trace_sink.start(
                LLMTraceStart(
                    parent_context=scope.parent if scope is not None else None,
                    request_id=self.context.request_id,
                    workspace_id=attempt.workspace_id,
                    run_id=self.context.run_id,
                    invocation_id=attempt.invocation_id,
                    graph_node=attempt.graph_node,
                    prompt_version=attempt.prompt_version,
                    provider=attempt.provider,
                    model=attempt.model,
                    invocation_kind=attempt.invocation_kind,
                    attempt_number=attempt_number,
                )
            )
        except Exception:
            identifiers = None
        if identifiers is not None and not isinstance(identifiers, TraceIdentifiers):
            identifiers = None
        return _TraceAttemptSession(sink=self.trace_sink, identifiers=identifiers)

    async def _complete(
        self,
        *,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
        trace: _TraceAttemptSession,
    ) -> None:
        finalize_task = asyncio.create_task(self._finalize(attempt, outcome))
        cancellation: asyncio.CancelledError | None = None
        try:
            while True:
                try:
                    await asyncio.shield(finalize_task)
                    break
                except asyncio.CancelledError as error:
                    if finalize_task.cancelled():
                        raise
                    if cancellation is None:
                        cancellation = error
        except BaseException:
            trace.finish(None)
            if cancellation is not None:
                raise cancellation from None
            raise
        else:
            trace.finish(outcome)
        if cancellation is not None:
            raise cancellation

    def _estimate_cost(self, usage: ModelUsage) -> InvocationCost | None:
        try:
            cost = self.price_book.estimate(
                provider=self.provider,
                model=self.model,
                usage=usage,
            )
        except Exception:
            raise LLMAccountingError(phase="pricing") from None
        if cost is not None and not isinstance(cost, InvocationCost):
            raise LLMAccountingError(phase="pricing")
        return cost

    def _deadline(self) -> float:
        return _clock_value(self.clock) + self.retry_policy.deadline_seconds

    def _remaining(self, deadline: float) -> float:
        return deadline - _clock_value(self.clock)

    def _retry_delay(self, *, attempt_number: int, retry_after_seconds: float | None) -> float:
        if retry_after_seconds is not None:
            return retry_after_seconds
        try:
            sample = self.random_source()
        except Exception:
            raise LLMFactoryConfigurationError from None
        if (
            isinstance(sample, bool)
            or not isinstance(sample, int | float)
            or not isfinite(float(sample))
            or not 0 <= float(sample) <= 1
        ):
            raise LLMFactoryConfigurationError
        ceiling = min(
            self.retry_policy.max_delay_seconds,
            self.retry_policy.base_delay_seconds * (2 ** (attempt_number - 1)),
        )
        return ceiling * float(sample)

    async def _wait_before_retry(
        self,
        *,
        attempt_number: int,
        deadline: float,
        retry_after_seconds: float | None,
        final_error: LLMProviderError,
    ) -> None:
        delay = self._retry_delay(
            attempt_number=attempt_number,
            retry_after_seconds=retry_after_seconds,
        )
        if delay >= self._remaining(deadline):
            raise final_error
        try:
            await self.sleeper(delay)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise LLMFactoryConfigurationError from None
        if self._remaining(deadline) <= 0:
            raise final_error

    async def _finalize_failure(
        self,
        *,
        attempt: LLMInvocationAttempt,
        started_at: float,
        category: LLMInvocationErrorCategory,
        trace: _TraceAttemptSession,
        provider_response_id: str | None = None,
    ) -> LLMProviderError:
        await self._complete(
            attempt=attempt,
            outcome=LLMInvocationOutcome(
                status="failed",
                provider_response_id=provider_response_id,
                latency_ms=_elapsed_ms(self.clock, started_at),
                error_category=category,
                trace_ids=trace.identifiers,
            ),
            trace=trace,
        )
        return LLMProviderError(category=category)

    async def _deadline_exhausted_after_prepare(
        self,
        *,
        attempt: LLMInvocationAttempt,
        started_at: float,
        trace: _TraceAttemptSession,
    ) -> Never:
        error = await self._finalize_failure(
            attempt=attempt,
            started_at=started_at,
            category="provider_timeout",
            trace=trace,
        )
        raise error


@dataclass(frozen=True, slots=True, repr=False)
class _AccountedChatModel(_AccountedModel):
    adapter: ChatAdapterPort

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
    ) -> ChatModelResult:
        graph_node, prompt_version = invocation_metadata(metadata, require_prompt_version=True)
        request_hash = chat_request_hash(model=self.model, messages=messages, tools=tools)
        deadline = self._deadline()
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            remaining = self._remaining(deadline)
            if remaining <= 0:
                raise LLMProviderError(category="provider_timeout")
            attempt = LLMInvocationAttempt(
                invocation_id=self._invocation_id(),
                workspace_id=self.context.workspace_id,
                actor_user_id=self.context.actor_user_id,
                run_id=self.context.run_id,
                invocation_kind="chat",
                provider=self.provider,
                model=self.model,
                graph_node=graph_node,
                prompt_version=prompt_version,
                request_hash=request_hash,
            )
            await self._prepare(attempt)
            trace = self._start_trace(attempt, attempt_number=attempt_number)
            try:
                started_at = _clock_value(self.clock)
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    await self._deadline_exhausted_after_prepare(
                        attempt=attempt,
                        started_at=started_at,
                        trace=trace,
                    )
                try:
                    async with asyncio.timeout(remaining):
                        result = await self.adapter.invoke(
                            messages,
                            tools,
                            dict(metadata),
                            attempt=ProviderAttemptContext(
                                invocation_id=attempt.invocation_id,
                                timeout_seconds=remaining,
                            ),
                        )
                except asyncio.CancelledError as cancellation:
                    try:
                        await self._complete(
                            attempt=attempt,
                            outcome=LLMInvocationOutcome(
                                status="failed",
                                latency_ms=_elapsed_ms(self.clock, started_at),
                                error_category="cancelled",
                                trace_ids=trace.identifiers,
                            ),
                            trace=trace,
                        )
                    except Exception:
                        raise cancellation from None
                    raise
                except ProviderAdapterError as error:
                    final_error = await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category=error.category,
                        trace=trace,
                        provider_response_id=error.provider_response_id,
                    )
                    if not error.retryable or attempt_number == self.retry_policy.max_attempts:
                        raise final_error from None
                    await self._wait_before_retry(
                        attempt_number=attempt_number,
                        deadline=deadline,
                        retry_after_seconds=error.retry_after_seconds,
                        final_error=final_error,
                    )
                    continue
                except TimeoutError:
                    final_error = await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category="provider_timeout",
                        trace=trace,
                    )
                    if self.provider != "qwen" or attempt_number == self.retry_policy.max_attempts:
                        raise final_error from None
                    await self._wait_before_retry(
                        attempt_number=attempt_number,
                        deadline=deadline,
                        retry_after_seconds=None,
                        final_error=final_error,
                    )
                    continue
                except Exception:
                    raise (
                        await self._finalize_failure(
                            attempt=attempt,
                            started_at=started_at,
                            category="provider_error",
                            trace=trace,
                        )
                    ) from None

                if not self._valid_result(result):
                    raise await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category="invalid_provider_response",
                        trace=trace,
                    )
                cost = self._estimate_cost(result.usage)
                await self._complete(
                    attempt=attempt,
                    outcome=LLMInvocationOutcome(
                        status="succeeded",
                        token_usage=result.usage,
                        provider_response_id=result.provider_response_id,
                        latency_ms=_elapsed_ms(self.clock, started_at),
                        pricing_version=cost.pricing_version if cost is not None else None,
                        currency=cost.currency if cost is not None else None,
                        estimated_cost=cost.estimated_cost if cost is not None else None,
                        trace_ids=trace.identifiers,
                    ),
                    trace=trace,
                )
                return result.model_copy(deep=True)
            finally:
                trace.finish(None)
        raise AssertionError("bounded chat retry loop did not terminate")

    def _valid_result(self, result: object) -> bool:
        return (
            isinstance(result, ChatModelResult)
            and result.provider == self.provider
            and result.model == self.model
        )


@dataclass(frozen=True, slots=True, repr=False)
class _AccountedEmbeddingModel(_AccountedModel):
    adapter: EmbeddingAdapterPort

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
    ) -> EmbeddingResult:
        graph_node, _ = invocation_metadata(metadata, require_prompt_version=False)
        request_hash = embedding_request_hash(model=self.model, texts=texts)
        deadline = self._deadline()
        for attempt_number in range(1, self.retry_policy.max_attempts + 1):
            remaining = self._remaining(deadline)
            if remaining <= 0:
                raise LLMProviderError(category="provider_timeout")
            attempt = LLMInvocationAttempt(
                invocation_id=self._invocation_id(),
                workspace_id=self.context.workspace_id,
                actor_user_id=self.context.actor_user_id,
                run_id=self.context.run_id,
                invocation_kind="embedding",
                provider=self.provider,
                model=self.model,
                graph_node=graph_node,
                prompt_version=None,
                request_hash=request_hash,
            )
            await self._prepare(attempt)
            trace = self._start_trace(attempt, attempt_number=attempt_number)
            try:
                started_at = _clock_value(self.clock)
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    await self._deadline_exhausted_after_prepare(
                        attempt=attempt,
                        started_at=started_at,
                        trace=trace,
                    )
                try:
                    async with asyncio.timeout(remaining):
                        result = await self.adapter.embed(
                            texts,
                            dict(metadata),
                            attempt=ProviderAttemptContext(
                                invocation_id=attempt.invocation_id,
                                timeout_seconds=remaining,
                            ),
                        )
                except asyncio.CancelledError as cancellation:
                    try:
                        await self._complete(
                            attempt=attempt,
                            outcome=LLMInvocationOutcome(
                                status="failed",
                                latency_ms=_elapsed_ms(self.clock, started_at),
                                error_category="cancelled",
                                trace_ids=trace.identifiers,
                            ),
                            trace=trace,
                        )
                    except Exception:
                        raise cancellation from None
                    raise
                except ProviderAdapterError as error:
                    final_error = await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category=error.category,
                        trace=trace,
                        provider_response_id=error.provider_response_id,
                    )
                    if not error.retryable or attempt_number == self.retry_policy.max_attempts:
                        raise final_error from None
                    await self._wait_before_retry(
                        attempt_number=attempt_number,
                        deadline=deadline,
                        retry_after_seconds=error.retry_after_seconds,
                        final_error=final_error,
                    )
                    continue
                except TimeoutError:
                    final_error = await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category="provider_timeout",
                        trace=trace,
                    )
                    if self.provider != "qwen" or attempt_number == self.retry_policy.max_attempts:
                        raise final_error from None
                    await self._wait_before_retry(
                        attempt_number=attempt_number,
                        deadline=deadline,
                        retry_after_seconds=None,
                        final_error=final_error,
                    )
                    continue
                except Exception:
                    raise (
                        await self._finalize_failure(
                            attempt=attempt,
                            started_at=started_at,
                            category="provider_error",
                            trace=trace,
                        )
                    ) from None

                if not self._valid_result(result, expected_count=len(texts)):
                    raise await self._finalize_failure(
                        attempt=attempt,
                        started_at=started_at,
                        category="invalid_provider_response",
                        trace=trace,
                    )
                cost = self._estimate_cost(result.usage)
                await self._complete(
                    attempt=attempt,
                    outcome=LLMInvocationOutcome(
                        status="succeeded",
                        token_usage=result.usage,
                        provider_response_id=result.provider_response_id,
                        latency_ms=_elapsed_ms(self.clock, started_at),
                        pricing_version=cost.pricing_version if cost is not None else None,
                        currency=cost.currency if cost is not None else None,
                        estimated_cost=cost.estimated_cost if cost is not None else None,
                        trace_ids=trace.identifiers,
                    ),
                    trace=trace,
                )
                return result.model_copy(deep=True)
            finally:
                trace.finish(None)
        raise AssertionError("bounded embedding retry loop did not terminate")

    def _valid_result(self, result: object, *, expected_count: int) -> bool:
        return (
            isinstance(result, EmbeddingResult)
            and result.provider == self.provider
            and result.model == self.model
            and len(result.vectors) == expected_count
        )
