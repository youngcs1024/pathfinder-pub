from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from itertools import count
from uuid import UUID

import pytest

from app.domain.tracing import (
    METADATA_ALLOWLIST,
    ExecutionSegmentIdentity,
    TraceIdentity,
    TraceSinkPort,
    TraceSpanFinish,
    TraceSpanStart,
)
from tests.tracing import CollectingTraceSink as _CollectingTraceSink

TRACE = TraceIdentity(workspace_id=UUID(int=1), run_id=UUID(int=2))
SEGMENT = ExecutionSegmentIdentity(job_id=UUID(int=3), attempt=1)
PROMPT = "sha256:" + "a" * 64
LLM_METADATA = {
    "llm_invocation_id": UUID(int=4),
    "graph_node": "research_agent",
    "provider": "qwen",
    "model": "qwen3.6-flash-2026-04-16",
    "attempt_number": 1,
    "input_tokens": 10,
    "output_tokens": 2,
    "total_tokens": 12,
    "cached_input_tokens": 0,
    "cache_write_input_tokens": 0,
    "reasoning_output_tokens": 1,
    "pricing_version": "qwen-cn-beijing-cny-2026-08-13-v1",
    "currency": "CNY",
    "estimated_cost": Decimal("0.0001"),
}
PAYLOADS = {
    "execution_segment": {"graph_version": "research-v1", "disposition": "execute"},
    "graph_node": {
        "graph_version": "research-v1",
        "node_name": "research_agent",
        "occurrence_ordinal": 2,
        "prompt_version": PROMPT,
    },
    "tool": {
        "tool_name": "search_web",
        "tool_effect": "read_only",
        "tool_invocation_id": UUID(int=5),
        "attempt_number": 1,
        "retry_count": 0,
        "output_bytes": 25,
    },
    "mcp_transport": {"transport": "stdio"},
    "retrieval": {
        "embedding_profile": "qwen-beijing-text-embedding-v4-1536-v1",
        "top_k": 5,
        "allowed_document_count": 2,
        "returned_chunk_count": 1,
        "context_bytes": 50,
    },
    "approval_event": {
        "approval_request_id": UUID(int=6),
        "action_intent_id": UUID(int=7),
        "decision_type": "approve",
        "actor_role": "admin",
        "binding_version": 1,
        "approval_duration_ms": 100,
    },
    "action_recovery": {"action_intent_id": UUID(int=7), "approval_request_id": UUID(int=6)},
    "llm_generation": {**LLM_METADATA, "prompt_version": PROMPT},
    "llm_embedding": {**LLM_METADATA, "model": "text-embedding-v4"},
}
PRIVATE_KEYS = (
    "query",
    "prompt",
    "messages",
    "document_body",
    "document_text",
    "chunk_text",
    "tool_args",
    "tool_output",
    "approval_reason",
    "action_args",
    "target",
    "trusted_target",
    "authorization",
    "credential",
    "secret",
    "checkpoint_payload",
    "checkpoint_state",
    "generated_report",
    "application_draft",
    "hidden_reasoning",
)
CANARIES = (
    "prompt-secret-canary",
    "document-secret-canary",
    "approval-secret-canary",
    "trusted-context-secret-canary",
)


def _start(kind="graph_node", **kwargs) -> TraceSpanStart:
    return TraceSpanStart(
        **{
            "trace_identity": TRACE,
            "segment_identity": SEGMENT,
            "span_kind": kind,
            "name": "approval.request_created" if kind == "approval_event" else kind,
            **kwargs,
        }
    )


def _finish(kind="graph_node", **kwargs) -> TraceSpanFinish:
    return TraceSpanFinish(
        **{
            "span_kind": kind,
            "status": "succeeded",
            "latency_ms": 0,
            **kwargs,
        }
    )


def _collector() -> _CollectingTraceSink:
    identifiers = count(100)
    return _CollectingTraceSink(lambda: UUID(int=next(identifiers)))


@pytest.mark.parametrize(
    ("fields", "kind", "value"),
    [
        ({"run_id": UUID(int=2)}, "run", 2),
        ({"request_id": UUID(int=3), "invocation_id": UUID(int=4)}, "request", 3),
        ({"invocation_id": UUID(int=4)}, "invocation", 4),
        (
            {"run_id": UUID(int=2), "request_id": UUID(int=3), "invocation_id": UUID(int=4)},
            "run",
            2,
        ),
    ],
)
def test_correlation_priority(fields, kind, value) -> None:
    identity = TraceIdentity(workspace_id=UUID(int=1), **fields)
    assert (identity.correlation_kind, identity.correlation_id) == (kind, UUID(int=value))


def test_missing_correlation_rejected() -> None:
    with pytest.raises(ValueError):
        TraceIdentity(workspace_id=UUID(int=1))


@pytest.mark.parametrize("key", ["workspace_id", "run_id", "request_id", "invocation_id"])
@pytest.mark.parametrize("value", ["not-uuid", 1, True, object()])
def test_identity_types(key, value) -> None:
    with pytest.raises(ValueError):
        TraceIdentity(**{"workspace_id": UUID(int=1), "run_id": UUID(int=2), key: value})


@pytest.mark.parametrize("attempt", [0, -1, True, False, 1.0, "1"])
def test_invalid_segment_attempt(attempt) -> None:
    with pytest.raises(ValueError):
        ExecutionSegmentIdentity(job_id=UUID(int=3), attempt=attempt)


def test_segment_identity() -> None:
    assert SEGMENT == ExecutionSegmentIdentity(job_id=UUID(int=3), attempt=1)
    with pytest.raises(ValueError):
        ExecutionSegmentIdentity(job_id="job", attempt=1)


@pytest.mark.parametrize("kind", PAYLOADS)
def test_kinds_and_exact_allowlists(kind) -> None:
    assert set(METADATA_ALLOWLIST) == set(PAYLOADS)
    assert METADATA_ALLOWLIST[kind] == PAYLOADS[kind].keys()
    assert dict(_start(kind, metadata=PAYLOADS[kind]).metadata) == PAYLOADS[kind]
    assert dict(_finish(kind, metadata=PAYLOADS[kind]).metadata) == PAYLOADS[kind]


@pytest.mark.parametrize(
    "name",
    [
        "approval.request_created",
        "approval.decision_recorded",
        "approval.resume_consumed",
        "approval.expired",
    ],
)
def test_approval_names(name) -> None:
    assert _start("approval_event", name=name).name == name


@pytest.mark.parametrize(
    "kind,name",
    [
        ("unknown", "unknown"),
        ("approval_event", "approval.other"),
        ("approval_event", "approval_event"),
        ("tool", "search_web"),
        ("graph_node", "prompt-secret-canary"),
        ("retrieval", "my document body"),
    ],
)
def test_invalid_kinds_names(kind, name) -> None:
    with pytest.raises(ValueError):
        _start(kind, name=name)


@pytest.mark.parametrize("kind", PAYLOADS)
@pytest.mark.parametrize("boundary", [_start, _finish])
def test_unknown_and_structural_metadata_keys(kind, boundary) -> None:
    for key in (
        "unknown",
        "workspace_id",
        "run_id",
        "request_id",
        "invocation_id",
        "parent",
        "segment_identity",
        "status",
        "latency_ms",
        "error_category",
        "actor_user_id",
    ):
        with pytest.raises(ValueError, match="key is not allowed"):
            boundary(kind, metadata={key: UUID(int=1)})


@pytest.mark.parametrize("key", PRIVATE_KEYS)
def test_private_data_cannot_enter_collector_or_errors(key) -> None:
    sink = _collector()
    for kind in PAYLOADS:
        context = sink.start(_start(kind, metadata=PAYLOADS[kind]))
        for boundary in (_start, _finish):
            for canary in CANARIES:
                with pytest.raises(ValueError) as error:
                    boundary(kind, metadata={key: canary})
                assert canary not in str(error.value)
        sink.finish(context, _finish(kind))
    for canary in CANARIES:
        assert canary not in repr(sink)
        assert canary not in sink.safe_json()


INVALID_VALUES = [
    ("tool", "tool_invocation_id", "uuid"),
    ("approval_event", "approval_request_id", "uuid"),
    ("action_recovery", "action_intent_id", "uuid"),
    ("llm_generation", "llm_invocation_id", "uuid"),
    ("graph_node", "node_name", "node with business text"),
    ("graph_node", "node_name", "x" * 101),
    ("graph_node", "prompt_version", "sha256:bad"),
    ("graph_node", "prompt_version", "sha256:" + "A" * 64),
    ("execution_segment", "disposition", "retry"),
    ("tool", "tool_effect", "unknown"),
    ("approval_event", "decision_type", "allow"),
    ("approval_event", "actor_role", "owner"),
    ("llm_generation", "currency", "USD"),
    ("llm_generation", "estimated_cost", Decimal("-1")),
    ("llm_generation", "estimated_cost", Decimal("NaN")),
    ("llm_generation", "estimated_cost", Decimal("sNaN")),
    ("llm_generation", "estimated_cost", Decimal("Infinity")),
    ("llm_generation", "estimated_cost", 0),
    ("llm_embedding", "prompt_version", PROMPT),
]


@pytest.mark.parametrize("kind,key,value", INVALID_VALUES)
@pytest.mark.parametrize("boundary", [_start, _finish])
def test_invalid_metadata_values(kind, key, value, boundary) -> None:
    with pytest.raises(ValueError):
        boundary(kind, metadata={key: value})


@pytest.mark.parametrize("kind", PAYLOADS)
@pytest.mark.parametrize("boundary", [_start, _finish])
def test_every_metadata_field_rejects_nested_or_arbitrary_values(kind, boundary) -> None:
    for key, valid in PAYLOADS[kind].items():
        for value in ({"secret": CANARIES[0]}, [CANARIES[1]], object(), RuntimeError(CANARIES[2])):
            with pytest.raises(ValueError):
                boundary(kind, metadata={key: value})
        if type(valid) is int:
            for value in (True, False, -1, 1.5, "1"):
                with pytest.raises(ValueError):
                    boundary(kind, metadata={key: value})
        if type(valid) is str and key not in {"currency", "prompt_version"}:
            for value in ("x" * 101, "private body\nsecond line", "", "é"):
                with pytest.raises(ValueError):
                    boundary(kind, metadata={key: value})


@pytest.mark.parametrize(
    "kind,key",
    [
        ("tool", "attempt_number"),
        ("graph_node", "occurrence_ordinal"),
        ("approval_event", "binding_version"),
        ("retrieval", "top_k"),
    ],
)
def test_positive_metadata_rejects_zero(kind, key) -> None:
    for boundary in (_start, _finish):
        with pytest.raises(ValueError):
            boundary(kind, metadata={key: 0})


@pytest.mark.parametrize("boundary", [_start, _finish])
def test_metadata_copied_and_frozen(boundary) -> None:
    metadata = {"node_name": "plan"}
    span = boundary(metadata=metadata)
    metadata["node_name"] = CANARIES[0]
    assert dict(span.metadata) == {"node_name": "plan"}
    with pytest.raises(TypeError):
        span.metadata["node_name"] = "write_report"
    with pytest.raises(FrozenInstanceError):
        span.metadata = {}
    assert CANARIES[0] not in repr(span)
    with pytest.raises(TypeError):
        METADATA_ALLOWLIST["tool"] = frozenset({"secret"})


@pytest.mark.parametrize(
    "status,category",
    [
        ("succeeded", None),
        ("failed", "provider_timeout"),
        ("cancelled", "lease_lost"),
        ("skipped", None),
        ("no_result", None),
    ],
)
def test_valid_status(status, category) -> None:
    assert _finish("retrieval", status=status, error_category=category).status == status


@pytest.mark.parametrize(
    "status,category",
    [
        ("succeeded", "error"),
        ("failed", None),
        ("cancelled", None),
        ("skipped", "error"),
        ("no_result", "error"),
        ("running", None),
        ("failed", "Exception: private body"),
        ("failed", "a" * 101),
        ("failed", "Upper"),
        ("failed", "error\n"),
        ("failed", RuntimeError(CANARIES[0])),
        ("failed", ""),
    ],
)
def test_invalid_status_error(status, category) -> None:
    with pytest.raises(ValueError):
        _finish("retrieval", status=status, error_category=category)


@pytest.mark.parametrize("kind", [kind for kind in PAYLOADS if kind != "retrieval"])
def test_no_result_is_retrieval_only(kind) -> None:
    with pytest.raises(ValueError):
        _finish(kind, status="no_result")


@pytest.mark.parametrize("value", [True, -1, 0.5, "0", float("nan"), float("inf")])
def test_invalid_latency(value) -> None:
    with pytest.raises(ValueError):
        _finish(latency_ms=value)


def test_parent_and_root_validation() -> None:
    parent = _collector().start(_start("execution_segment"))
    assert _start(parent=parent).parent == parent
    assert _start(parent=None).parent is None
    assert _start(parent=parent, segment_identity=None).parent == parent
    assert _start(parent=replace(parent, span_kind="approval_event", segment_identity=None))
    with pytest.raises(ValueError):
        _start("execution_segment", parent=parent)
    with pytest.raises(ValueError):
        _start("execution_segment", segment_identity=None)
    for identity in (
        replace(TRACE, workspace_id=UUID(int=9)),
        replace(TRACE, run_id=UUID(int=9)),
        replace(TRACE, request_id=UUID(int=9)),
    ):
        with pytest.raises(ValueError, match="parent identity mismatch"):
            _start(parent=parent, trace_identity=identity)
    for segment in (replace(SEGMENT, attempt=2), replace(SEGMENT, job_id=UUID(int=9))):
        with pytest.raises(ValueError, match="parent segment mismatch"):
            _start(parent=parent, segment_identity=segment)


def test_context_contract_fields() -> None:
    parent = _collector().start(_start())
    for fields in (
        {"context_id": "vendor-id"},
        {"span_kind": "unknown"},
        {"trace_identity": object()},
        {"segment_identity": "job"},
        {"span_kind": "execution_segment", "segment_identity": None},
    ):
        with pytest.raises(ValueError):
            replace(parent, **fields)
    for fields in ({"trace_identity": object()}, {"segment_identity": "job"}, {"parent": object()}):
        with pytest.raises(ValueError):
            _start(**fields)
    with pytest.raises(FrozenInstanceError):
        parent.context_id = UUID(int=10)
    with pytest.raises(FrozenInstanceError):
        TRACE.run_id = UUID(int=10)
    with pytest.raises(FrozenInstanceError):
        SEGMENT.attempt = 2


def test_collecting_sink_topology_and_pairing() -> None:
    sink = _collector()
    port: TraceSinkPort = sink
    root = port.start(_start("execution_segment", metadata=PAYLOADS["execution_segment"]))
    node = port.start(_start(parent=root, metadata=PAYLOADS["graph_node"]))
    children = [
        port.start(_start(kind, parent=node, metadata=PAYLOADS[kind]))
        for kind in ("llm_generation", "tool", "retrieval")
    ]
    assert root is not None and node is not None and all(children)
    assert len(sink.starts) == 5
    assert list(sink.starts) == [UUID(int=n) for n in range(100, 105)]
    assert sink.starts[root.context_id][1].parent is None
    assert sink.starts[node.context_id][1].parent == root
    for child in children:
        assert child is not None
        assert sink.starts[child.context_id][1].parent == node
    for context, span in reversed(list(sink.starts.values())):
        assert context.trace_identity == span.trace_identity == TRACE
        assert context.segment_identity == span.segment_identity == SEGMENT
        port.finish(context, _finish(span.span_kind, metadata=PAYLOADS[span.span_kind]))
    assert sink.starts.keys() == sink.finishes.keys()
    with pytest.raises(ValueError, match="duplicate finish"):
        port.finish(root, _finish("execution_segment"))
    second = replace(TRACE, workspace_id=UUID(int=20), run_id=UUID(int=21))
    root_b = port.start(_start("execution_segment", trace_identity=second))
    assert root_b is not None and root_b.trace_identity == second
    with pytest.raises(ValueError, match="parent identity mismatch"):
        port.start(_start(trace_identity=second, parent=node))
    port.finish(root_b, _finish("execution_segment"))
    assert sink.starts.keys() == sink.finishes.keys()
    for canary in CANARIES:
        assert canary not in repr(sink)
        assert canary not in sink.safe_json()


def test_collector_rejects_mismatched_or_unknown_finish() -> None:
    sink = _collector()
    context = sink.start(_start())
    with pytest.raises(ValueError, match="context mismatch"):
        sink.finish(context, _finish("tool"))
    for forged in (
        replace(context, context_id=UUID(int=999)),
        replace(context, trace_identity=replace(TRACE, run_id=UUID(int=99))),
        replace(context, segment_identity=replace(SEGMENT, attempt=2)),
        replace(context, span_kind="tool"),
    ):
        with pytest.raises(ValueError, match="context mismatch"):
            sink.finish(forged, _finish(forged.span_kind))
    assert sink.finishes == {}
    sink.finish(context, _finish())


def test_collector_rejects_duplicate_context_id() -> None:
    sink = _CollectingTraceSink(lambda: UUID(int=100))
    sink.start(_start())
    with pytest.raises(ValueError, match="duplicate context"):
        sink.start(_start())
    assert len(sink.starts) == 1


async def test_active_scope_isolated_between_interleaved_tasks_and_reset() -> None:
    import asyncio

    from app.domain.tracing import ActiveTraceScope, bind_trace_scope, current_trace_scope

    scopes = [
        ActiveTraceScope(
            trace_identity=replace(TRACE, workspace_id=UUID(int=n), run_id=UUID(int=n + 10)),
            segment_identity=replace(SEGMENT, job_id=UUID(int=n + 20)),
        )
        for n in (50, 60)
    ]
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def observe(index):
        assert current_trace_scope() is None
        with bind_trace_scope(scopes[index]):
            entered[index].set()
            await entered[1 - index].wait()
            assert current_trace_scope() is scopes[index]
            await release.wait()
            assert current_trace_scope() is scopes[index]
        assert current_trace_scope() is None

    tasks = [asyncio.create_task(observe(index)) for index in (0, 1)]
    await asyncio.gather(*(event.wait() for event in entered))
    assert current_trace_scope() is None
    release.set()
    await asyncio.gather(*tasks)
    assert current_trace_scope() is None


def test_scope_rejects_other_workspace_or_segment_parent() -> None:
    from app.domain.tracing import ActiveTraceScope

    parent = _collector().start(_start("execution_segment"))
    for bad in (
        replace(parent, trace_identity=replace(TRACE, workspace_id=UUID(int=999))),
        replace(parent, segment_identity=replace(SEGMENT, attempt=2)),
    ):
        with pytest.raises(ValueError, match="scope parent mismatch"):
            ActiveTraceScope(trace_identity=TRACE, segment_identity=SEGMENT, parent=bad)


@pytest.mark.parametrize(
    "name",
    [
        "approval.request_created",
        "approval.decision_recorded",
        "approval.resume_consumed",
        "approval.expired",
    ],
)
def test_runtime_helper_accepts_explicit_approval_names(name):
    from time import monotonic

    from app.domain.tracing import finish_trace_span, start_trace_span
    from tests.tracing import CollectingTraceSink, collecting_node

    sink = CollectingTraceSink()
    with collecting_node(sink) as scope:
        context = start_trace_span(scope, span_kind="approval_event", name=name, metadata={})
        assert context is not None
        assert sink.starts[context.context_id][1].name == name
        finish_trace_span(scope, context, started_at=monotonic(), status="succeeded")
        for bad_name, metadata in [("approval.unknown", {}), (name, {"reason": "CANARY"})]:
            assert (
                start_trace_span(
                    scope, span_kind="approval_event", name=bad_name, metadata=metadata
                )
                is None
            )
    assert sink.starts.keys() == sink.finishes.keys()
    assert list(sink.starts.values())[1][1].name == "graph_node"


def test_runtime_approval_helper_drops_sink_exceptions():
    from app.domain.tracing import start_trace_span
    from tests.tracing import CollectingTraceSink, collecting_node

    class Broken:
        def start(self, span):
            raise RuntimeError("PRIVATE")

    with collecting_node(CollectingTraceSink()) as scope:
        assert (
            start_trace_span(
                replace(scope, sink=Broken()),
                span_kind="approval_event",
                name="approval.expired",
                metadata={},
            )
            is None
        )


@pytest.mark.parametrize("value", ["http", "secret-canary", 1, None])
def test_mcp_transport_only_accepts_stdio(value):
    with pytest.raises(ValueError):
        _start("mcp_transport", metadata={"transport": value})


@pytest.mark.parametrize(
    "key", ["record_id", "workspace_id", "actor_user_id", "deadline", "raw_protocol", "stderr"]
)
def test_mcp_transport_rejects_context_and_body(key):
    with pytest.raises(ValueError):
        _start("mcp_transport", metadata={key: "secret-canary"})
