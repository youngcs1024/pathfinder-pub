"""Collecting vendor-neutral sink shared by trace boundary and runtime tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from time import monotonic
from uuid import UUID, uuid4

from app.domain.tracing import (
    ActiveTraceScope,
    ExecutionSegmentIdentity,
    TraceIdentity,
    TraceParentContext,
    TraceSpanFinish,
    TraceSpanStart,
    bind_trace_scope,
    current_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.llm.invocations import TraceIdentifiers


@dataclass
class CollectingTraceSink:
    context_id_factory: Callable[[], UUID] = uuid4
    starts: dict[UUID, tuple[TraceParentContext, TraceSpanStart]] = field(default_factory=dict)
    finishes: dict[UUID, TraceSpanFinish] = field(default_factory=dict)

    def start(self, span: TraceSpanStart) -> TraceParentContext:
        context = TraceParentContext(
            trace_identity=span.trace_identity,
            segment_identity=span.segment_identity,
            span_kind=span.span_kind,
            context_id=self.context_id_factory(),
        )
        if context.context_id in self.starts:
            raise ValueError("duplicate context")
        self.starts[context.context_id] = (context, span)
        return context

    def finish(self, context: TraceParentContext, outcome: TraceSpanFinish) -> None:
        started = self.starts.get(context.context_id)
        if started is None or started[0] != context or context.span_kind != outcome.span_kind:
            raise ValueError("finish context mismatch")
        if context.context_id in self.finishes:
            raise ValueError("duplicate finish")
        self.finishes[context.context_id] = outcome

    def safe_json(self) -> str:
        return json.dumps(
            [
                {
                    "context_id": str(context.context_id),
                    "workspace_id": str(span.trace_identity.workspace_id),
                    "correlation_kind": span.trace_identity.correlation_kind,
                    "correlation_id": str(span.trace_identity.correlation_id),
                    "segment": None
                    if span.segment_identity is None
                    else {
                        "job_id": str(span.segment_identity.job_id),
                        "attempt": span.segment_identity.attempt,
                    },
                    "parent": str(span.parent.context_id) if span.parent else None,
                    "kind": span.span_kind,
                    "name": span.name,
                    "metadata": dict(span.metadata),
                    "finish": None
                    if context.context_id not in self.finishes
                    else {
                        "status": self.finishes[context.context_id].status,
                        "latency_ms": self.finishes[context.context_id].latency_ms,
                        "error_category": self.finishes[context.context_id].error_category,
                        "metadata": dict(self.finishes[context.context_id].metadata),
                    },
                }
                for context, span in self.starts.values()
            ],
            default=str,
            sort_keys=True,
        )


@dataclass
class FaultTraceSink:
    """Deterministic, test-only loss seam; never changes the production contract."""

    delegate: object
    failure: str
    selected_kind: str = "execution_segment"
    start_calls: list[TraceSpanStart] = field(default_factory=list)
    finish_calls: list[TraceParentContext] = field(default_factory=list)

    def start(self, span):
        self.start_calls.append(span)
        if span.span_kind == self.selected_kind:
            if self.failure == "drop":
                return None
            if self.failure == "start":
                raise RuntimeError("TRACE_SECRET_CANARY_START")
        return self.delegate.start(span)

    def finish(self, context, outcome):
        self.finish_calls.append(context)
        if self.failure == "finish":
            raise RuntimeError("TRACE_SECRET_CANARY_FINISH")
        self.delegate.finish(context, outcome)


class CollectingLLMTraceSink:
    """Observe the existing factory bridge in the same collector as its parents."""

    def __init__(self):
        self.open = {}
        self.starts = []

    def start(self, trace):
        self.starts.append(trace)
        scope = current_trace_scope()
        assert scope is not None and scope.parent == trace.parent_context
        metadata = {
            "llm_invocation_id": trace.invocation_id,
            "graph_node": trace.graph_node,
            "provider": trace.provider,
            "model": trace.model,
            "attempt_number": trace.attempt_number,
        }
        if trace.prompt_version is not None:
            metadata["prompt_version"] = trace.prompt_version
        parent = start_trace_span(
            scope,
            span_kind="llm_embedding" if trace.invocation_kind == "embedding" else "llm_generation",
            metadata=metadata,
        )
        if parent is None:
            return None
        identifiers = TraceIdentifiers(
            trace_id=scope.trace_identity.correlation_id.hex,
            observation_id=parent.context_id.hex[:16],
        )
        self.open[identifiers.observation_id] = (scope, parent, monotonic())
        return identifiers

    def finish(self, identifiers, outcome):
        scope, parent, started_at = self.open.pop(identifiers.observation_id)
        finish_trace_span(
            scope,
            parent,
            started_at=started_at,
            status=outcome.status if outcome is not None else "failed",
            error_category=outcome.error_category if outcome is not None else "accounting_error",
        )


def assert_closed_trace_tree(sink: CollectingTraceSink, *, segments: int) -> None:
    """Check every edge, including repeated nodes and factory-generated children."""
    from app.domain.tracing import METADATA_ALLOWLIST

    assert sink.starts.keys() == sink.finishes.keys()
    roots = [c for c, s in sink.starts.values() if s.span_kind == "execution_segment"]
    assert len(roots) == segments
    ordinals = {}
    for context, span in sink.starts.values():
        assert set(span.metadata) <= METADATA_ALLOWLIST[span.span_kind]
        assert set(sink.finishes[context.context_id].metadata) <= METADATA_ALLOWLIST[span.span_kind]
        if span.span_kind == "execution_segment":
            assert span.parent is None
            continue
        assert span.parent is not None
        assert sink.starts[span.parent.context_id][0] == span.parent
        ancestor = span.parent
        visited = {context.context_id}
        while ancestor is not None:
            assert ancestor.context_id not in visited
            visited.add(ancestor.context_id)
            assert ancestor.trace_identity == context.trace_identity
            assert ancestor.segment_identity == context.segment_identity
            ancestor_span = sink.starts[ancestor.context_id][1]
            if ancestor_span.parent is None:
                assert ancestor in roots
            ancestor = ancestor_span.parent
        if span.span_kind == "graph_node":
            assert span.parent.span_kind == "execution_segment"
            key = (span.parent.context_id, span.metadata["node_name"])
            ordinals[key] = ordinals.get(key, 0) + 1
            assert span.metadata["occurrence_ordinal"] == ordinals[key]
        if span.span_kind in {"tool", "llm_generation"}:
            assert span.parent.span_kind == "graph_node"
        if span.span_kind in {"retrieval", "mcp_transport"}:
            assert span.parent.span_kind == "tool"
        if span.span_kind == "llm_embedding":
            assert span.parent.span_kind == "retrieval"


@contextmanager
def collecting_segment(sink: CollectingTraceSink, *, trace_identity: TraceIdentity | None = None):
    scope = ActiveTraceScope(
        trace_identity=trace_identity or TraceIdentity(workspace_id=uuid4(), run_id=uuid4()),
        segment_identity=ExecutionSegmentIdentity(job_id=uuid4(), attempt=1),
        sink=sink,
    )
    started_at = monotonic()
    parent = start_trace_span(scope, span_kind="execution_segment", metadata={})
    scope = replace(scope, parent=parent)
    with bind_trace_scope(scope):
        try:
            yield scope
        finally:
            finish_trace_span(scope, parent, started_at=started_at, status="succeeded")


@contextmanager
def collecting_node(sink: CollectingTraceSink):
    with collecting_segment(sink) as scope:
        started_at = monotonic()
        parent = start_trace_span(
            scope, span_kind="graph_node", metadata={"node_name": "research_agent"}
        )
        scope = replace(scope, parent=parent)
        with bind_trace_scope(scope):
            try:
                yield scope
            finally:
                finish_trace_span(scope, parent, started_at=started_at, status="succeeded")
