from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest

from app.config import Settings
from app.llm.invocations import (
    LLMInvocationOutcome,
    LLMTraceStart,
    NoOpTraceSink,
    TraceIdentifiers,
)
from app.llm.ports import ModelUsage
from app.obs.langfuse import (
    LangfuseTraceSink,
    build_trace_sink,
    configure_langfuse_vendor_logging,
    is_langfuse_span,
)
from app.obs.logging import configure_logging

PROMPT_VERSION = f"sha256:{'a' * 64}"


@dataclass
class _FakeObservation:
    trace_id: str
    id: str
    update_error: Exception | None = None
    end_error: Exception | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)
    end_calls: int = 0

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)
        if self.update_error is not None:
            raise self.update_error

    def end(self) -> None:
        self.end_calls += 1
        if self.end_error is not None:
            raise self.end_error


@dataclass
class _FakeLangfuseClient:
    start_error: Exception | None = None
    update_error: Exception | None = None
    end_error: Exception | None = None
    mismatched_trace_id: bool = False
    flush_error: Exception | None = None
    shutdown_error: Exception | None = None
    starts: list[dict[str, Any]] = field(default_factory=list)
    observations: list[_FakeObservation] = field(default_factory=list)
    flush_calls: int = 0
    shutdown_calls: int = 0

    def create_trace_id(self, *, seed: str | None = None) -> str:
        assert seed is not None
        return hashlib.sha256(seed.encode()).hexdigest()[:32]

    def start_observation(self, **kwargs: Any) -> _FakeObservation:
        self.starts.append(kwargs)
        if self.start_error is not None:
            raise self.start_error
        trace_id = kwargs["trace_context"]["trace_id"]
        if self.mismatched_trace_id:
            trace_id = "f" * 32
        observation = _FakeObservation(
            trace_id=trace_id,
            id=f"{len(self.observations) + 1:016x}",
            update_error=self.update_error,
            end_error=self.end_error,
        )
        self.observations.append(observation)
        return observation

    def flush(self) -> None:
        self.flush_calls += 1
        if self.flush_error is not None:
            raise self.flush_error

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


def _trace(
    *,
    invocation_id: UUID,
    invocation_kind: str = "chat",
    request_id: UUID | None = None,
    run_id: UUID | None = None,
) -> LLMTraceStart:
    is_chat = invocation_kind == "chat"
    return LLMTraceStart(
        request_id=request_id,
        workspace_id=UUID(int=1),
        run_id=run_id,
        invocation_id=invocation_id,
        graph_node="plan" if is_chat else "ingest_documents",
        prompt_version=PROMPT_VERSION if is_chat else None,
        provider="qwen",
        model="qwen3.6-flash-2026-04-16" if is_chat else "text-embedding-v4",
        invocation_kind=invocation_kind,
        attempt_number=1,
    )


def _success_outcome(*, trace_ids: TraceIdentifiers) -> LLMInvocationOutcome:
    return LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(
            input_tokens=11,
            output_tokens=13,
            total_tokens=24,
            cached_input_tokens=2,
            cache_write_input_tokens=3,
            reasoning_output_tokens=5,
        ),
        latency_ms=17,
        pricing_version="qwen-cn-beijing-cny-2026-08-13-v1",
        currency="CNY",
        estimated_cost=Decimal("0.000175900000"),
        trace_ids=trace_ids,
    )


def test_adapter_uses_fixed_observation_types_safe_metadata_and_exclusive_usage() -> None:
    client = _FakeLangfuseClient()
    sink = LangfuseTraceSink(client)

    chat_ids = sink.start(_trace(invocation_id=UUID(int=10), request_id=UUID(int=20)))
    embedding_ids = sink.start(
        _trace(invocation_id=UUID(int=11), invocation_kind="embedding", request_id=UUID(int=20))
    )

    assert chat_ids is not None
    assert embedding_ids is not None
    assert chat_ids.trace_id == embedding_ids.trace_id
    assert chat_ids.observation_id != embedding_ids.observation_id
    assert [(item["name"], item["as_type"]) for item in client.starts] == [
        ("pathfinder.llm.chat", "generation"),
        ("pathfinder.llm.embedding", "embedding"),
    ]
    assert set(client.starts[0]["metadata"]) == {
        "workspace_id",
        "request_id",
        "invocation_id",
        "graph_node",
        "prompt_version",
        "provider",
        "model",
        "invocation_kind",
        "attempt_number",
    }
    assert "input" not in client.starts[0]
    assert "output" not in client.starts[0]

    outcome = _success_outcome(trace_ids=chat_ids)
    sink.finish(chat_ids, outcome)
    update = client.observations[0].updates[0]
    assert update["usage_details"] == {
        "input": 6,
        "output": 8,
        "input_cached_tokens": 2,
        "input_cache_write_tokens": 3,
        "output_reasoning_tokens": 5,
        "total": 24,
    }
    assert update["metadata"]["estimated_cost"] == "0.000175900000"
    assert "cost_details" not in update
    assert client.observations[0].end_calls == 1


def test_trace_grouping_prefers_run_then_request_then_invocation() -> None:
    client = _FakeLangfuseClient()
    sink = LangfuseTraceSink(client)

    run_ids = [
        sink.start(
            _trace(
                invocation_id=UUID(int=value),
                request_id=UUID(int=40),
                run_id=UUID(int=30),
            )
        )
        for value in (1, 2)
    ]
    request_ids = [
        sink.start(_trace(invocation_id=UUID(int=value), request_id=UUID(int=40)))
        for value in (3, 4)
    ]
    invocation_ids = [sink.start(_trace(invocation_id=UUID(int=value))) for value in (5, 6)]

    assert all(item is not None for item in (*run_ids, *request_ids, *invocation_ids))
    assert run_ids[0].trace_id == run_ids[1].trace_id  # type: ignore[union-attr]
    assert request_ids[0].trace_id == request_ids[1].trace_id  # type: ignore[union-attr]
    assert run_ids[0].trace_id != request_ids[0].trace_id  # type: ignore[union-attr]
    assert invocation_ids[0].trace_id != invocation_ids[1].trace_id  # type: ignore[union-attr]


@pytest.mark.parametrize("failure", ["start", "mismatch", "update", "end"])
def test_sdk_failures_are_swallowed_and_each_started_observation_is_closed(
    failure: str,
) -> None:
    client = _FakeLangfuseClient(
        start_error=RuntimeError("start-secret-canary") if failure == "start" else None,
        update_error=RuntimeError("update-secret-canary") if failure == "update" else None,
        end_error=RuntimeError("end-secret-canary") if failure == "end" else None,
        mismatched_trace_id=failure == "mismatch",
    )
    sink = LangfuseTraceSink(client)

    identifiers = sink.start(_trace(invocation_id=UUID(int=10)))

    if failure in {"start", "mismatch"}:
        assert identifiers is None
        assert sum(item.end_calls for item in client.observations) == (failure == "mismatch")
        return
    assert identifiers is not None
    sink.finish(identifiers, _success_outcome(trace_ids=identifiers))
    sink.finish(identifiers, _success_outcome(trace_ids=identifiers))
    assert client.observations[0].end_calls == 1


def test_accounting_failure_uses_only_fixed_error_metadata() -> None:
    client = _FakeLangfuseClient()
    sink = LangfuseTraceSink(client)
    identifiers = sink.start(_trace(invocation_id=UUID(int=10)))
    assert identifiers is not None

    sink.finish(identifiers, None)

    update = client.observations[0].updates[0]
    assert update["metadata"]["status"] == "accounting_failed"
    assert update["metadata"]["error_category"] == "accounting_error"
    assert update["status_message"] == "accounting_error"
    assert update["usage_details"] is None


def test_lifecycle_failures_degrade_and_are_not_called_per_observation() -> None:
    client = _FakeLangfuseClient(
        flush_error=RuntimeError("flush-secret-canary"),
        shutdown_error=RuntimeError("shutdown-secret-canary"),
    )
    sink = LangfuseTraceSink(client)
    identifiers = sink.start(_trace(invocation_id=UUID(int=10)))
    assert identifiers is not None
    sink.finish(identifiers, _success_outcome(trace_ids=identifiers))

    assert client.flush_calls == 0
    assert client.shutdown_calls == 0
    assert sink.flush_and_check() is False
    sink.shutdown()
    assert client.flush_calls == 1
    assert client.shutdown_calls == 1


def test_off_mode_does_not_construct_a_client_and_enabled_builder_is_sdk_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    construction_calls: list[dict[str, Any]] = []
    fake_client = _FakeLangfuseClient()

    def constructor(**kwargs: Any) -> _FakeLangfuseClient:
        construction_calls.append(kwargs)
        return fake_client

    monkeypatch.setattr("app.obs.langfuse.Langfuse", constructor)
    assert isinstance(build_trace_sink(Settings()), NoOpTraceSink)
    assert construction_calls == []

    sink = build_trace_sink(
        Settings(
            trace_mode="langfuse",
            langfuse_public_key="public-key-canary",
            langfuse_secret_key="secret-key-canary",
            langfuse_base_url="https://us.cloud.langfuse.com",
            langfuse_sample_rate=0.25,
        )
    )

    assert isinstance(sink, LangfuseTraceSink)
    assert len(construction_calls) == 1
    assert construction_calls[0]["base_url"] == "https://us.cloud.langfuse.com"
    assert construction_calls[0]["sample_rate"] == 0.25
    assert construction_calls[0]["should_export_span"] is is_langfuse_span
    masked = construction_calls[0]["mask"](
        data={"secret": "credential-canary", "nested": {"authorization": "Bearer canary"}}
    )
    assert masked == {"secret": "[REDACTED]", "nested": {"authorization": "[REDACTED]"}}
    assert fake_client.flush_calls == 0
    assert fake_client.shutdown_calls == 0


def test_vendor_log_bridge_drops_raw_messages_and_exception_text() -> None:
    stream = io.StringIO()
    configure_logging(log_level="INFO", stream=stream)
    configure_langfuse_vendor_logging()

    try:
        raise RuntimeError("vendor-exception-secret-canary")
    except RuntimeError:
        logging.getLogger("langfuse.dynamic-logger-secret-canary").error(
            "vendor-message-prompt-document-trusted-context-canary",
            exc_info=True,
        )

    output = stream.getvalue()
    assert "vendor.observability" in output
    assert '"error_type": "RuntimeError"' in output
    assert "vendor-message-prompt-document-trusted-context-canary" not in output
    assert "vendor-exception-secret-canary" not in output
    assert "dynamic-logger-secret-canary" not in output


def _agent_start(kind="execution_segment", **kwargs):
    from app.domain.tracing import ExecutionSegmentIdentity, TraceIdentity, TraceSpanStart

    return TraceSpanStart(
        **{
            "span_kind": kind,
            "name": kind,
            "trace_identity": TraceIdentity(workspace_id=UUID(int=1), run_id=UUID(int=20)),
            "segment_identity": ExecutionSegmentIdentity(job_id=UUID(int=30), attempt=1),
            **kwargs,
        }
    )


def _agent_finish(kind="execution_segment", **kwargs):
    from app.domain.tracing import TraceSpanFinish

    return TraceSpanFinish(span_kind=kind, status="succeeded", latency_ms=10, **kwargs)


def test_generic_tree_and_legacy_llm_share_client_seed_and_shutdown():
    from app.obs.langfuse import agent_trace_sink

    client = _FakeLangfuseClient()
    legacy = LangfuseTraceSink(client)
    sink = agent_trace_sink(legacy)
    assert sink is legacy.agent_sink
    root = sink.start(_agent_start(metadata={"graph_version": "pathfinder-research-v3"}))
    assert root is not None
    children = [
        sink.start(
            _agent_start(
                "graph_node",
                parent=root,
                metadata={
                    "node_name": "research_agent",
                    "graph_version": "pathfinder-research-v3",
                    "occurrence_ordinal": ordinal,
                },
            )
        )
        for ordinal in (1, 2)
    ]
    ids = legacy.start(_trace(invocation_id=UUID(int=100), run_id=UUID(int=20)))
    assert ids is not None
    assert len({item.trace_id for item in client.observations}) == 1
    assert len({item.id for item in client.observations}) == 4
    assert children[0] != children[1]
    for start in client.starts[1:3]:
        assert start["trace_context"]["parent_span_id"] == client.observations[0].id
        assert set(start["metadata"]) == {
            "workspace_id",
            "run_id",
            "job_id",
            "attempt",
            "span_kind",
            "node_name",
            "graph_version",
            "occurrence_ordinal",
        }
        assert "input" not in start and "output" not in start
    for child in children:
        sink.finish(child, _agent_finish("graph_node"))
        sink.finish(child, _agent_finish("graph_node"))
    sink.finish(root, _agent_finish(metadata={"disposition": "execute"}))
    legacy.finish(ids, _success_outcome(trace_ids=ids))
    assert [item.end_calls for item in client.observations] == [1, 1, 1, 1]
    assert client.observations[0].updates[0]["metadata"]["disposition"] == "execute"
    legacy.shutdown()
    assert client.shutdown_calls == 1


@pytest.mark.parametrize("field", ["start_error", "update_error", "end_error"])
def test_generic_sdk_failures_are_dropped_and_finish_is_idempotent(field):
    client = _FakeLangfuseClient(**{field: RuntimeError("TRACE-BUSINESS-BODY-CANARY")})
    sink = LangfuseTraceSink(client).agent_sink
    context = sink.start(_agent_start())
    if field == "start_error":
        assert context is None
    else:
        assert context is not None
        sink.finish(context, _agent_finish())
        sink.finish(context, _agent_finish())
        assert client.observations[0].end_calls == 1
        assert "TRACE-BUSINESS-BODY-CANARY" not in repr(client.observations[0].updates)
    assert "TRACE-BUSINESS-BODY-CANARY" not in repr(client.starts)


def test_generic_missing_parent_unknown_finish_and_mismatch_degrade():
    from dataclasses import replace

    client = _FakeLangfuseClient()
    sink = LangfuseTraceSink(client).agent_sink
    root = sink.start(_agent_start())
    assert root is not None
    for forged in (
        replace(root, context_id=UUID(int=999)),
        replace(root, trace_identity=replace(root.trace_identity, run_id=UUID(int=99))),
    ):
        sink.finish(forged, _agent_finish())
    sink.finish(root, _agent_finish("graph_node"))
    assert client.observations[0].end_calls == 0
    sink.finish(root, _agent_finish())
    assert sink.start(_agent_start("graph_node", parent=root)) is None
    sink.finish(root, _agent_finish())
    assert client.observations[0].end_calls == 1
    client.mismatched_trace_id = True
    assert sink.start(_agent_start()) is None
    assert client.observations[-1].end_calls == 1


@pytest.mark.parametrize("kind", ["run", "request", "invocation"])
def test_generic_seed_matches_legacy_fallback_priority(kind):
    from app.domain.tracing import TraceIdentity

    fields = (
        {"run_id": UUID(int=20)}
        if kind == "run"
        else ({"request_id": UUID(int=21)} if kind == "request" else {})
    )
    invocation_id = UUID(int=22)
    client = _FakeLangfuseClient()
    legacy = LangfuseTraceSink(client)
    ids = legacy.start(_trace(invocation_id=invocation_id, **fields))
    parent = legacy.agent_sink.start(
        _agent_start(
            trace_identity=TraceIdentity(
                workspace_id=UUID(int=1),
                invocation_id=invocation_id,
                **fields,
            )
        )
    )
    assert parent is not None and ids is not None
    assert client.observations[-1].trace_id == ids.trace_id


def test_generic_view_builds_no_second_client_and_off_constructs_none(monkeypatch):
    import importlib

    from app.obs.langfuse import agent_trace_sink

    module = importlib.import_module("app.obs.langfuse")
    calls = []
    client = _FakeLangfuseClient()

    def construct(**kwargs):
        calls.append(kwargs)
        return client

    monkeypatch.setattr(module, "Langfuse", construct)
    off = build_trace_sink(Settings(trace_mode="off"))
    assert agent_trace_sink(off) is None
    assert calls == []
    legacy = build_trace_sink(
        Settings(
            trace_mode="langfuse",
            langfuse_public_key="public-key",
            langfuse_secret_key="secret-key",
            langfuse_base_url="https://us.cloud.langfuse.com",
        )
    )
    sink = agent_trace_sink(legacy)
    assert len(calls) == 1
    assert sink.start(_agent_start()) is not None
    assert legacy.start(_trace(invocation_id=UUID(int=10))) is not None
    assert len(calls) == 1
    client.shutdown_error = RuntimeError("sensitive shutdown failure")
    legacy.shutdown()
    assert client.shutdown_calls == 1


@pytest.mark.parametrize(
    ("kind", "parent_kind", "as_type"),
    [("chat", "graph_node", "generation"), ("embedding", "retrieval", "embedding")],
)
@pytest.mark.parametrize("failure", [None, "missing", "finished", "context", "trace", "start"])
def test_llm_uses_open_application_parent_or_drops_without_orphan(
    kind, parent_kind, as_type, failure
):
    from dataclasses import replace

    client = _FakeLangfuseClient()
    legacy = LangfuseTraceSink(client)
    sink = legacy.agent_sink
    root = sink.start(_agent_start())
    parent = sink.start(_agent_start(parent_kind, parent=root))
    assert parent is not None
    expected = client.observations[-1]
    if failure == "missing":
        parent = replace(parent, context_id=UUID(int=999))
    elif failure == "finished":
        sink.finish(parent, _agent_finish(parent_kind))
    elif failure == "context":
        parent = replace(parent, segment_identity=replace(parent.segment_identity, attempt=2))
    elif failure == "trace":
        expected.trace_id = "f" * 32
    elif failure == "start":
        client.start_error = RuntimeError("PRIVATE")
    trace = _trace(
        invocation_id=UUID(int=100), invocation_kind=kind, run_id=UUID(int=20)
    ).model_copy(update={"parent_context": parent})
    before = len(client.observations)
    ids = legacy.start(trace)
    if failure:
        assert ids is None
        assert len(client.observations) == before
    else:
        assert ids is not None
        assert client.starts[-1]["trace_context"] == {
            "trace_id": expected.trace_id,
            "parent_span_id": expected.id,
        }
        assert client.starts[-1]["as_type"] == as_type
        assert "parent_span_id" not in repr(client.starts[-1]["metadata"])
        legacy.finish(ids, None)
    legacy.shutdown()
    assert client.shutdown_calls == 1
