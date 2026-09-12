"""Application-owned observation contracts and process-local scope; no exporter.

Correlation is not authorization. PostgreSQL remains the business authority; these
objects must never decide execution, retry, approval, or checkpoint recovery.
Callers supply code-owned identifiers/categories, never business text. Validation
fails closed without echoing rejected input; export failure handling belongs to obs.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from time import monotonic
from types import MappingProxyType
from typing import Literal, Protocol, get_args
from uuid import UUID

type CorrelationKind = Literal["run", "request", "invocation"]
type SpanKind = Literal[
    "execution_segment",
    "graph_node",
    "tool",
    "mcp_transport",
    "retrieval",
    "approval_event",
    "action_recovery",
    "llm_generation",
    "llm_embedding",
]
type SpanStatus = Literal["succeeded", "failed", "cancelled", "skipped", "no_result"]
type TraceMetadataValue = str | int | UUID | Decimal

_SPAN_KINDS = frozenset(get_args(SpanKind.__value__))
_SPAN_STATUSES = frozenset(get_args(SpanStatus.__value__))
_APPROVAL_NAMES = frozenset(
    {
        "approval.request_created",
        "approval.decision_recorded",
        "approval.resume_consumed",
        "approval.expired",
    }
)
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]{0,99}")
_MACHINE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,99}")
_PROMPT_VERSION = re.compile(r"sha256:[0-9a-f]{64}")
_UUID_FIELDS = frozenset(
    {"tool_invocation_id", "approval_request_id", "action_intent_id", "llm_invocation_id"}
)
_POSITIVE_FIELDS = frozenset({"attempt_number", "occurrence_ordinal", "binding_version", "top_k"})
_TOKEN_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "reasoning_output_tokens",
    }
)
_NONNEGATIVE_FIELDS = _TOKEN_FIELDS | frozenset(
    {
        "retry_count",
        "output_bytes",
        "allowed_document_count",
        "returned_chunk_count",
        "context_bytes",
        "approval_duration_ms",
    }
)
_MACHINE_FIELDS = frozenset(
    {"graph_version", "provider", "model", "embedding_profile", "pricing_version"}
)
_SNAKE_FIELDS = frozenset({"node_name", "graph_node", "tool_name"})
_ENUM_FIELDS = MappingProxyType(
    {
        "disposition": frozenset({"execute", "recover_action", "finished", "lease_lost"}),
        "tool_effect": frozenset({"read_only", "reversible", "irreversible"}),
        "decision_type": frozenset({"approve", "reject"}),
        "actor_role": frozenset({"member", "reviewer", "admin"}),
        "currency": frozenset({"CNY"}),
        "transport": frozenset({"stdio"}),
    }
)
_LLM_FIELDS = _TOKEN_FIELDS | frozenset(
    {
        "llm_invocation_id",
        "graph_node",
        "provider",
        "model",
        "attempt_number",
        "pricing_version",
        "currency",
        "estimated_cost",
    }
)
METADATA_ALLOWLIST: Mapping[SpanKind, frozenset[str]] = MappingProxyType(
    {
        "execution_segment": frozenset({"graph_version", "disposition"}),
        "graph_node": frozenset(
            {"graph_version", "node_name", "occurrence_ordinal", "prompt_version"}
        ),
        "tool": frozenset(
            {
                "tool_name",
                "tool_effect",
                "tool_invocation_id",
                "attempt_number",
                "retry_count",
                "output_bytes",
            }
        ),
        "mcp_transport": frozenset({"transport"}),
        "retrieval": frozenset(
            {
                "embedding_profile",
                "top_k",
                "allowed_document_count",
                "returned_chunk_count",
                "context_bytes",
            }
        ),
        "approval_event": frozenset(
            {
                "approval_request_id",
                "action_intent_id",
                "decision_type",
                "actor_role",
                "binding_version",
                "approval_duration_ms",
            }
        ),
        "action_recovery": frozenset({"action_intent_id", "approval_request_id"}),
        "llm_generation": _LLM_FIELDS | {"prompt_version"},
        "llm_embedding": _LLM_FIELDS,
    }
)


def _uuid(value: object) -> None:
    if type(value) is not UUID:
        raise ValueError("trace identity fields must be UUID values")


def _integer(value: object, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError("trace integer is invalid")


def _kind(value: object) -> None:
    if type(value) is not str or value not in _SPAN_KINDS:
        raise ValueError("trace span kind is invalid")


def _matches(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _freeze_metadata(
    kind: SpanKind, metadata: Mapping[str, TraceMetadataValue]
) -> Mapping[str, TraceMetadataValue]:
    if not isinstance(metadata, Mapping):
        raise ValueError("trace metadata must be a mapping")
    copied = dict(metadata)
    for key, value in copied.items():
        if type(key) is not str or key not in METADATA_ALLOWLIST[kind]:
            raise ValueError("trace metadata key is not allowed")
        if key in _UUID_FIELDS:
            _uuid(value)
        elif key in _POSITIVE_FIELDS:
            _integer(value, minimum=1)
        elif key in _NONNEGATIVE_FIELDS:
            _integer(value, minimum=0)
        elif key in _ENUM_FIELDS:
            if type(value) is not str or value not in _ENUM_FIELDS[key]:
                raise ValueError("trace metadata enum is invalid")
        elif key == "estimated_cost":
            if type(value) is not Decimal or not value.is_finite() or value < 0:
                raise ValueError("trace estimated cost must be a finite nonnegative Decimal")
        else:
            pattern = (
                _PROMPT_VERSION
                if key == "prompt_version"
                else _SNAKE_CASE
                if key in _SNAKE_FIELDS
                else _MACHINE_ID
                if key in _MACHINE_FIELDS
                else None
            )
            if pattern is None or not _matches(value, pattern):
                raise ValueError("trace metadata identifier is invalid")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceIdentity:
    """Existing correlation identities only; never a permission or vendor trace ID.

    Reuse the same full identity within a parent tree. Per-call LLM/tool invocation
    IDs belong in metadata, not in the logical invocation fallback field.
    """

    workspace_id: UUID
    run_id: UUID | None = None
    request_id: UUID | None = None
    invocation_id: UUID | None = None

    def __post_init__(self) -> None:
        _uuid(self.workspace_id)
        for value in (self.run_id, self.request_id, self.invocation_id):
            if value is not None:
                _uuid(value)
        if self.run_id is None and self.request_id is None and self.invocation_id is None:
            raise ValueError("trace requires a correlation identity")

    @property
    def correlation_kind(self) -> CorrelationKind:
        if self.run_id is not None:
            return "run"
        return "request" if self.request_id is not None else "invocation"

    @property
    def correlation_id(self) -> UUID:
        if self.run_id is not None:
            return self.run_id
        if self.request_id is not None:
            return self.request_id
        assert self.invocation_id is not None
        return self.invocation_id


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSegmentIdentity:
    """One bounded segment per claimed worker job, identified by (job_id, attempt)."""

    job_id: UUID
    attempt: int

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _integer(self.attempt, minimum=1)


def _context_fields(
    trace_identity: TraceIdentity,
    segment_identity: ExecutionSegmentIdentity | None,
    span_kind: SpanKind,
) -> None:
    if type(trace_identity) is not TraceIdentity:
        raise ValueError("trace identity uses the wrong contract")
    if segment_identity is not None and type(segment_identity) is not ExecutionSegmentIdentity:
        raise ValueError("trace segment uses the wrong contract")
    _kind(span_kind)
    if span_kind == "execution_segment" and segment_identity is None:
        raise ValueError("execution segment requires segment identity")


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceParentContext:
    """Sink-created, process-local token; may be lost on crash or sampling.

    context_id is neither a business ID nor a vendor observation ID. Never persist
    it in DB/checkpoint/API. An obs adapter may privately map it to a vendor ID.
    """

    trace_identity: TraceIdentity
    span_kind: SpanKind
    context_id: UUID
    segment_identity: ExecutionSegmentIdentity | None = None

    def __post_init__(self) -> None:
        _context_fields(self.trace_identity, self.segment_identity, self.span_kind)
        _uuid(self.context_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceSpanStart:
    trace_identity: TraceIdentity
    span_kind: SpanKind
    name: str
    segment_identity: ExecutionSegmentIdentity | None = None
    parent: TraceParentContext | None = None
    metadata: Mapping[str, TraceMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _context_fields(self.trace_identity, self.segment_identity, self.span_kind)
        if type(self.name) is not str or (
            self.name not in _APPROVAL_NAMES
            if self.span_kind == "approval_event"
            else self.name != self.span_kind
        ):
            raise ValueError("trace span name is invalid")
        if self.parent is not None:
            if type(self.parent) is not TraceParentContext:
                raise ValueError("trace parent uses the wrong contract")
            if self.span_kind == "execution_segment":
                raise ValueError("execution segment must be a root")
            if self.parent.trace_identity != self.trace_identity:
                raise ValueError("trace parent identity mismatch")
            if (
                self.parent.segment_identity is not None
                and self.segment_identity is not None
                and self.parent.segment_identity != self.segment_identity
            ):
                raise ValueError("trace parent segment mismatch")
        object.__setattr__(self, "metadata", _freeze_metadata(self.span_kind, self.metadata))


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceSpanFinish:
    """Terminal observation only, not a workflow state machine.

    Finish metadata uses the same per-kind allowlist for counts/cost known only
    after execution. Missing usage/cost stays absent, never inferred as zero.
    """

    span_kind: SpanKind
    status: SpanStatus
    latency_ms: int
    error_category: str | None = None
    metadata: Mapping[str, TraceMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _kind(self.span_kind)
        if type(self.status) is not str or self.status not in _SPAN_STATUSES:
            raise ValueError("trace span status is invalid")
        _integer(self.latency_ms, minimum=0)
        if self.error_category is not None and not _matches(self.error_category, _SNAKE_CASE):
            raise ValueError("trace error category is invalid")
        if self.status in {"failed", "cancelled"}:
            if self.error_category is None:
                raise ValueError("failed or cancelled trace requires an error category")
        elif self.error_category is not None:
            raise ValueError("trace status forbids an error category")
        if self.status == "no_result" and self.span_kind != "retrieval":
            raise ValueError("no_result is only valid for retrieval")
        object.__setattr__(self, "metadata", _freeze_metadata(self.span_kind, self.metadata))


class TraceSinkPort(Protocol):
    """Narrow synchronous, lossy observation lifecycle.

    A sink creates a unique local context for each accepted start, or returns None
    for a dropped observation. Finish must match that context's kind and complete
    it at most once. No exporter result can become a business prerequisite.
    """

    def start(self, span: TraceSpanStart) -> TraceParentContext | None: ...

    def finish(self, context: TraceParentContext, outcome: TraceSpanFinish) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveTraceScope:
    """Coroutine-local correlation only; never persisted or used for authorization."""

    trace_identity: TraceIdentity
    segment_identity: ExecutionSegmentIdentity
    parent: TraceParentContext | None = None
    sink: TraceSinkPort | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _context_fields(self.trace_identity, self.segment_identity, "execution_segment")
        if self.parent is not None and (
            self.parent.trace_identity != self.trace_identity
            or self.parent.segment_identity != self.segment_identity
        ):
            raise ValueError("trace scope parent mismatch")


_ACTIVE_TRACE_SCOPE: ContextVar[ActiveTraceScope | None] = ContextVar(
    "pathfinder_active_trace_scope", default=None
)


def current_trace_scope() -> ActiveTraceScope | None:
    return _ACTIVE_TRACE_SCOPE.get()


@contextmanager
def bind_trace_scope(scope: ActiveTraceScope | None) -> Iterator[None]:
    token = _ACTIVE_TRACE_SCOPE.set(scope)
    try:
        yield
    finally:
        _ACTIVE_TRACE_SCOPE.reset(token)


def start_trace_span(
    scope: ActiveTraceScope,
    *,
    span_kind: SpanKind,
    metadata: Mapping[str, TraceMetadataValue],
    name: str | None = None,
) -> TraceParentContext | None:
    """Reject invalid metadata/contexts and drop sink failures without business effects."""
    if scope.sink is None or (span_kind != "execution_segment" and scope.parent is None):
        return None
    try:
        context = scope.sink.start(
            TraceSpanStart(
                trace_identity=scope.trace_identity,
                segment_identity=scope.segment_identity,
                parent=scope.parent,
                span_kind=span_kind,
                name=name or span_kind,
                metadata=metadata,
            )
        )
        if (
            type(context) is TraceParentContext
            and context.trace_identity == scope.trace_identity
            and context.segment_identity == scope.segment_identity
            and context.span_kind == span_kind
        ):
            return context
    except Exception:
        pass
    return None


def finish_trace_span(
    scope: ActiveTraceScope,
    context: TraceParentContext | None,
    *,
    started_at: float,
    status: SpanStatus,
    error_category: str | None = None,
    metadata: Mapping[str, TraceMetadataValue] | None = None,
) -> None:
    if scope.sink is None or context is None:
        return
    try:
        scope.sink.finish(
            context,
            TraceSpanFinish(
                span_kind=context.span_kind,
                status=status,
                latency_ms=max(0, int((monotonic() - started_at) * 1000)),
                error_category=error_category,
                metadata=metadata or {},
            ),
        )
    except Exception:
        pass
