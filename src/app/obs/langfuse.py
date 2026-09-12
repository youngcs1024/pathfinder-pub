from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from threading import Lock
from typing import Any, Protocol
from uuid import UUID, uuid4

from langfuse import Langfuse, is_langfuse_span

from app.config import Settings
from app.domain.tracing import (
    TraceIdentity,
    TraceParentContext,
    TraceSpanFinish,
    TraceSpanStart,
)
from app.domain.tracing import TraceSinkPort as AgentTraceSinkPort
from app.llm.invocations import (
    LLMInvocationOutcome,
    LLMTraceStart,
    NoOpTraceSink,
    TraceIdentifiers,
    TraceSinkPort,
)
from app.llm.ports import ModelUsage
from app.obs.logging import configure_vendor_logging
from app.obs.redaction import redact_value


class _LangfuseObservation(Protocol):
    trace_id: str
    id: str

    def update(self, **kwargs: Any) -> object: ...

    def end(self) -> object: ...


class _LangfuseClient(Protocol):
    def create_trace_id(self, *, seed: str | None = None) -> str: ...

    def start_observation(self, **kwargs: Any) -> _LangfuseObservation: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _OpenObservation:
    observation: _LangfuseObservation
    start: LLMTraceStart


def configure_langfuse_vendor_logging() -> None:
    configure_vendor_logging()


def _safe_payload(value: dict[str, object]) -> dict[str, object]:
    redacted = redact_value(value)
    if not isinstance(redacted, dict):
        raise TypeError("trace redaction must preserve a mapping")
    return redacted


def _mask_langfuse_data(*, data: Any) -> Any:
    return redact_value(data)


def _identity_trace_seed(identity: TraceIdentity) -> str:
    return (
        "pathfinder-langfuse-trace-v1\x00"
        f"workspace={identity.workspace_id}\x00{identity.correlation_kind}={identity.correlation_id}"
    )


def _trace_seed(trace: LLMTraceStart) -> str:
    return _identity_trace_seed(
        TraceIdentity(
            workspace_id=trace.workspace_id,
            run_id=trace.run_id,
            request_id=trace.request_id,
            invocation_id=trace.invocation_id,
        )
    )


def _start_metadata(trace: LLMTraceStart) -> dict[str, object]:
    payload: dict[str, object] = {
        "workspace_id": str(trace.workspace_id),
        "invocation_id": str(trace.invocation_id),
        "graph_node": trace.graph_node,
        "provider": trace.provider,
        "model": trace.model,
        "invocation_kind": trace.invocation_kind,
        "attempt_number": trace.attempt_number,
    }
    if trace.request_id is not None:
        payload["request_id"] = str(trace.request_id)
    if trace.run_id is not None:
        payload["run_id"] = str(trace.run_id)
    if trace.prompt_version is not None:
        payload["prompt_version"] = trace.prompt_version
    return _safe_payload(payload)


def _exclusive_usage(usage: ModelUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    cached = usage.cached_input_tokens or 0
    cache_write = usage.cache_write_input_tokens or 0
    reasoning = usage.reasoning_output_tokens or 0
    details: dict[str, int] = {
        "input": usage.input_tokens - cached - cache_write,
        "output": usage.output_tokens - reasoning,
    }
    if usage.cached_input_tokens is not None:
        details["input_cached_tokens"] = cached
    if usage.cache_write_input_tokens is not None:
        details["input_cache_write_tokens"] = cache_write
    if usage.reasoning_output_tokens is not None:
        details["output_reasoning_tokens"] = reasoning
    if usage.total_tokens is not None:
        details["total"] = usage.total_tokens
    return details


def _finish_metadata(
    open_observation: _OpenObservation,
    identifiers: TraceIdentifiers,
    outcome: LLMInvocationOutcome | None,
) -> dict[str, object]:
    payload = _start_metadata(open_observation.start)
    payload["trace_id"] = identifiers.trace_id
    payload["observation_id"] = identifiers.observation_id
    if outcome is None:
        payload["status"] = "accounting_failed"
        payload["error_category"] = "accounting_error"
        return _safe_payload(payload)

    payload["status"] = outcome.status
    payload["latency_ms"] = outcome.latency_ms
    if outcome.error_category is not None:
        payload["error_category"] = outcome.error_category
    if outcome.pricing_version is not None:
        payload["pricing_version"] = outcome.pricing_version
    if outcome.currency is not None:
        payload["currency"] = outcome.currency
    if outcome.estimated_cost is not None:
        payload["estimated_cost"] = format(outcome.estimated_cost, "f")
    return _safe_payload(payload)


class LangfuseTraceSink:
    def __init__(self, client: _LangfuseClient) -> None:
        self._client = client
        self.agent_sink = LangfuseAgentTraceSink(client)
        self._open: dict[str, _OpenObservation] = {}
        self._lock = Lock()

    def start(self, trace: LLMTraceStart) -> TraceIdentifiers | None:
        if not isinstance(trace, LLMTraceStart):
            return None
        observation: _LangfuseObservation | None = None
        try:
            trace_id = self._client.create_trace_id(seed=_trace_seed(trace))
            trace_context = {"trace_id": trace_id}
            if trace.parent_context is not None:
                parent = self.agent_sink._resolve_parent(trace.parent_context)
                if parent is None or parent.trace_id != trace_id:
                    return None
                trace_context["parent_span_id"] = parent.observation_id
            observation = self._client.start_observation(
                trace_context=trace_context,
                name=f"pathfinder.llm.{trace.invocation_kind}",
                as_type="generation" if trace.invocation_kind == "chat" else "embedding",
                metadata=_start_metadata(trace),
                version=trace.prompt_version,
                model=trace.model,
            )
            identifiers = TraceIdentifiers(
                trace_id=observation.trace_id,
                observation_id=observation.id,
            )
            if identifiers.trace_id != trace_id:
                raise ValueError("Langfuse returned an unexpected trace identifier")
            with self._lock:
                if identifiers.observation_id in self._open:
                    raise ValueError("Langfuse returned a duplicate observation identifier")
                self._open[identifiers.observation_id] = _OpenObservation(
                    observation=observation,
                    start=trace,
                )
            return identifiers
        except Exception:
            if observation is not None:
                try:
                    observation.end()
                except Exception:
                    pass
            return None

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None:
        if not isinstance(identifiers, TraceIdentifiers):
            return
        with self._lock:
            open_observation = self._open.pop(identifiers.observation_id, None)
        if open_observation is None:
            return

        level = "ERROR" if outcome is None or outcome.status == "failed" else "DEFAULT"
        status_message = (
            "accounting_error"
            if outcome is None
            else outcome.error_category
            if outcome.status == "failed"
            else None
        )
        try:
            open_observation.observation.update(
                metadata=_finish_metadata(open_observation, identifiers, outcome),
                level=level,
                status_message=status_message,
                usage_details=_exclusive_usage(outcome.token_usage)
                if outcome is not None
                else None,
            )
        except Exception:
            pass
        finally:
            try:
                open_observation.observation.end()
            except Exception:
                pass

    def flush(self) -> None:
        self.flush_and_check()

    def flush_and_check(self) -> bool:
        try:
            self._client.flush()
        except Exception:
            return False
        return True

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:
            return


@dataclass(frozen=True, slots=True)
class _OpenAgentObservation:
    observation: _LangfuseObservation
    context: TraceParentContext
    start: TraceSpanStart


def _agent_metadata(
    span: TraceSpanStart, outcome: TraceSpanFinish | None = None
) -> dict[str, object]:
    identity = span.trace_identity
    payload: dict[str, object] = {
        "workspace_id": str(identity.workspace_id),
        f"{identity.correlation_kind}_id": str(identity.correlation_id),
        "span_kind": span.span_kind,
        **dict(span.metadata),
    }
    if span.segment_identity is not None:
        payload["job_id"] = str(span.segment_identity.job_id)
        payload["attempt"] = span.segment_identity.attempt
    if outcome is not None:
        payload.update(outcome.metadata)
        payload["status"] = outcome.status
        payload["latency_ms"] = outcome.latency_ms
        if outcome.error_category is not None:
            payload["error_category"] = outcome.error_category
    return _safe_payload(
        {
            key: str(value) if isinstance(value, (UUID, Decimal)) else value
            for key, value in payload.items()
        }
    )


class LangfuseAgentTraceSink:
    """Generic adapter view of the existing client; owns no exporter lifecycle."""

    def __init__(self, client: _LangfuseClient) -> None:
        self._client = client
        self._open: dict[UUID, _OpenAgentObservation] = {}
        self._lock = Lock()

    def _resolve_parent(self, context: TraceParentContext) -> TraceIdentifiers | None:
        """Resolve only a currently open, exactly matching application context."""
        with self._lock:
            opened = self._open.get(context.context_id)
            if opened is None or opened.context != context:
                return None
            return TraceIdentifiers(
                trace_id=opened.observation.trace_id,
                observation_id=opened.observation.id,
            )

    def start(self, span: TraceSpanStart) -> TraceParentContext | None:
        if not isinstance(span, TraceSpanStart):
            return None
        observation = None
        try:
            trace_id = self._client.create_trace_id(seed=_identity_trace_seed(span.trace_identity))
            trace_context = {"trace_id": trace_id}
            if span.parent is not None:
                with self._lock:
                    parent = self._open.get(span.parent.context_id)
                if parent is None or parent.context != span.parent:
                    return None
                trace_context["parent_span_id"] = parent.observation.id
            observation = self._client.start_observation(
                trace_context=trace_context,
                name=f"pathfinder.{span.name}",
                as_type="span",
                metadata=_agent_metadata(span),
            )
            if observation.trace_id != trace_id:
                raise ValueError("Langfuse returned an unexpected trace identifier")
            context = TraceParentContext(
                trace_identity=span.trace_identity,
                segment_identity=span.segment_identity,
                span_kind=span.span_kind,
                context_id=uuid4(),
            )
            with self._lock:
                if any(item.observation.id == observation.id for item in self._open.values()):
                    raise ValueError("Langfuse returned a duplicate observation identifier")
                self._open[context.context_id] = _OpenAgentObservation(observation, context, span)
            return context
        except Exception:
            if observation is not None:
                try:
                    observation.end()
                except Exception:
                    pass
            return None

    def finish(self, context: TraceParentContext, outcome: TraceSpanFinish) -> None:
        if not isinstance(context, TraceParentContext) or not isinstance(outcome, TraceSpanFinish):
            return
        with self._lock:
            opened = self._open.get(context.context_id)
            if (
                opened is None
                or opened.context != context
                or outcome.span_kind != context.span_kind
            ):
                return
            self._open.pop(context.context_id)
        try:
            opened.observation.update(
                metadata=_agent_metadata(opened.start, outcome),
                level="ERROR" if outcome.status == "failed" else "DEFAULT",
                status_message=outcome.error_category,
            )
        except Exception:
            pass
        finally:
            try:
                opened.observation.end()
            except Exception:
                pass


def agent_trace_sink(sink: TraceSinkPort) -> AgentTraceSinkPort | None:
    """Select the generic view without building another client, including in off mode."""
    return sink.agent_sink if isinstance(sink, LangfuseTraceSink) else None


def build_trace_sink(settings: Settings) -> TraceSinkPort:
    if not isinstance(settings, Settings):
        raise TypeError("trace sink settings use the wrong contract")
    if settings.trace_mode == "off":
        return NoOpTraceSink()

    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    base_url = settings.langfuse_base_url
    if public_key is None or secret_key is None or base_url is None:
        return NoOpTraceSink()

    configure_langfuse_vendor_logging()
    try:
        client = Langfuse(
            public_key=public_key.get_secret_value().strip(),
            secret_key=secret_key.get_secret_value().strip(),
            base_url=base_url,
            sample_rate=settings.langfuse_sample_rate,
            mask=_mask_langfuse_data,
            should_export_span=is_langfuse_span,
        )
    except Exception:
        return NoOpTraceSink()
    return LangfuseTraceSink(client)
