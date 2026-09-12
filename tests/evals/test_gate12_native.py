"""Native experiment evidence; no MCP, database, wall-clock thresholds or sleeps."""

import json
import os
import socket
from dataclasses import replace
from itertools import count
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.tools.contracts import ToolTransientError
from app.tools.invocations import (
    InMemoryToolInvocationRecorder,
    canonical_args_digest,
    output_summary,
)
from app.tools.registry import (
    ToolCallLimitExceededError,
    ToolCancelledError,
    ToolDeadlineExceededError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolOutputValidationError,
    ToolRegistry,
    ToolTimeoutError,
    ToolUnavailableError,
)
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    MANUAL_LEARNING_POLICY,
    MANUAL_LEARNING_POLICY_NAME,
)
from tests.evals import gate12_native as native
from tests.tracing import CollectingTraceSink, collecting_node

CANARY = "gate12-private-context-secret-canary"


class SpyRecorder(InMemoryToolInvocationRecorder):
    def __init__(self):
        super().__init__()
        self.events = []

    async def reserve(self, **kwargs):
        reservation = await super().reserve(**kwargs)
        self.events.append(("reserve", kwargs))
        return reservation

    async def start_attempt(self, **kwargs):
        self.events.append(("start_attempt", kwargs))

    async def succeed(self, **kwargs):
        self.events.append(("succeed", kwargs))

    async def fail(self, **kwargs):
        self.events.append(("fail", kwargs))


class Cancellation:
    def __init__(self, *values):
        self.values = iter(values)

    def is_cancelled(self):
        return next(self.values)


def harness(*, fault=None, cancellation=None, clock=lambda: 10.0, trusted_target=None):
    calls = []
    recorder = SpyRecorder()

    async def handler(tool_input, context):
        calls.append((tool_input, context))
        if fault == "malformed":
            return {"record_id": "record-1", "found": True, "summary": None}
        if fault is ToolTransientError:
            raise ToolTransientError
        if fault is not None:
            raise fault(CANARY)
        return await LOOKUP_SYNTHETIC_RECORD_SPEC.handler(tool_input, context)

    registry = ToolRegistry(
        specs=(replace(LOOKUP_SYNTHETIC_RECORD_SPEC, handler=handler),),
        policies=(MANUAL_LEARNING_POLICY,),
        recorder=recorder,
    )
    context = replace(
        native.native_context(0),
        cancellation=cancellation or native.NotCancelled(),
        deadline=100.0,
        trusted_target=trusted_target,
    )
    ids = count(10000)
    runtime = registry.bind(
        policy_name=MANUAL_LEARNING_POLICY_NAME,
        context=context,
        invocation_id_factory=lambda: UUID(int=next(ids)),
        clock=clock,
    )
    return runtime, recorder, calls


def tool_spans(sink):
    return [
        (span, sink.finishes[context.context_id])
        for context, span in sink.starts.values()
        if span.span_kind == "tool"
    ]


@pytest.mark.parametrize(
    "record_id,summary", list(zip(native.CASES, native.EXPECTED_SUMMARIES, strict=True))
)
async def test_native_results_accounting_and_span(record_id, summary):
    runtime, recorder, calls = harness()
    sink = CollectingTraceSink()
    with collecting_node(sink):
        serialized = await runtime.execute(native.native_call(record_id))
    assert json.loads(serialized) == {
        "record_id": record_id,
        "found": summary is not None,
        "summary": summary,
    }
    assert len(calls) == 1
    assert [name for name, _ in recorder.events] == ["reserve", "start_attempt", "succeed"]
    reserve, start, succeeded = [data for _, data in recorder.events]
    assert reserve["args_digest"] == canonical_args_digest({"record_id": record_id})
    assert reserve["invocation_id"] == start["reservation"].invocation_id == UUID(int=10000)
    assert succeeded["reservation"] == start["reservation"]
    assert start["attempt"] == 1
    assert succeeded["result_summary"] == output_summary(serialized)
    assert reserve["workspace_id"] == start["workspace_id"] == succeeded["workspace_id"]
    assert reserve["run_id"] == start["run_id"] == succeeded["run_id"]
    assert calls[0][1].invocation_id == reserve["invocation_id"]
    assert calls[0][1].trusted_target is None
    [(span, finish)] = tool_spans(sink)
    assert dict(span.metadata) == {
        "tool_name": "lookup_synthetic_record",
        "tool_effect": "read_only",
        "tool_invocation_id": UUID(int=10000),
    }
    assert finish.status == "succeeded"
    assert finish.error_category is None
    assert dict(finish.metadata) == {
        "attempt_number": 1,
        "retry_count": 0,
        "output_bytes": len(serialized.encode("utf-8")),
    }
    assert finish.latency_ms >= 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"record_id": " "},
        {"record_id": 7},
        {"record_id": "record-1", "workspace_id": CANARY},
    ],
)
async def test_input_rejection_precedes_reservation_and_handler(arguments):
    runtime, recorder, calls = harness()
    call = native.native_call("record-1").model_copy(update={"arguments": arguments})
    sink = CollectingTraceSink()
    with collecting_node(sink), pytest.raises(ToolInputValidationError):
        await runtime.execute(call)
    assert calls == recorder.events == []
    assert tool_spans(sink) == []


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"cancellation": Cancellation(True)}, ToolCancelledError),
        ({"clock": lambda: 100.0}, ToolDeadlineExceededError),
    ],
)
async def test_cancel_deadline_before_reservation(kwargs, error):
    runtime, recorder, calls = harness(**kwargs)
    sink = CollectingTraceSink()
    with collecting_node(sink), pytest.raises(error):
        await runtime.execute(native.native_call("record-1"))
    assert calls == recorder.events == []
    assert tool_spans(sink) == []


@pytest.mark.parametrize(
    "fault,error,category",
    [
        (ToolTransientError, ToolUnavailableError, "provider_unavailable"),
        (TimeoutError, ToolTimeoutError, "provider_timeout"),
        (RuntimeError, ToolExecutionError, "tool_execution_failed"),
        ("malformed", ToolOutputValidationError, "invalid_tool_output"),
    ],
)
async def test_admitted_failure_has_one_attempt_and_safe_accounting(fault, error, category):
    runtime, recorder, calls = harness(fault=fault)
    sink = CollectingTraceSink()
    with collecting_node(sink), pytest.raises(error) as caught:
        await runtime.execute(native.native_call("record-1"))
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.max_attempts == 1
    assert len(calls) == 1
    assert [name for name, _ in recorder.events] == ["reserve", "start_attempt", "fail"]
    assert recorder.events[-1][1]["error_category"] == category
    assert recorder.events[-1][1]["reservation"] == recorder.events[1][1]["reservation"]
    [(_, finish)] = tool_spans(sink)
    assert (finish.status, finish.error_category) == ("failed", category)
    assert dict(finish.metadata) == {"attempt_number": 1, "retry_count": 0}
    assert CANARY not in sink.safe_json() + str(caught.value) + repr(recorder.events)


async def test_cancel_after_reservation_records_failure_without_handler_attempt():
    runtime, recorder, calls = harness(cancellation=Cancellation(False, True))
    sink = CollectingTraceSink()
    with collecting_node(sink), pytest.raises(ToolCancelledError):
        await runtime.execute(native.native_call("record-1"))
    assert calls == []
    assert [name for name, _ in recorder.events] == ["reserve", "fail"]
    assert recorder.events[-1][1]["error_category"] == "cancelled"
    [(_, finish)] = tool_spans(sink)
    assert (finish.status, finish.error_category) == ("cancelled", "cancelled")
    assert dict(finish.metadata) == {}


async def test_fourth_call_is_not_admitted():
    runtime, recorder, calls = harness()
    sink = CollectingTraceSink()
    with collecting_node(sink):
        for record_id in native.CASES:
            await runtime.execute(native.native_call(record_id))
        with pytest.raises(ToolCallLimitExceededError):
            await runtime.execute(native.native_call("record-1"))
    assert len(calls) == len(tool_spans(sink)) == 3
    assert [name for name, _ in recorder.events] == ["reserve", "start_attempt", "succeed"] * 3
    assert [
        data["reservation"].call_number for name, data in recorder.events if name == "succeed"
    ] == [1, 2, 3]


async def test_offline_report_and_trace_do_not_read_or_expose_secrets(monkeypatch, caplog, capsys):
    keys = (
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    )
    for key in keys:
        monkeypatch.setenv(key, CANARY)
    original_getitem = type(os.environ).__getitem__

    def checked_getitem(env, key):
        assert key not in keys, "native runner read provider secret"
        return original_getitem(env, key)

    def deny_network(*args, **kwargs):
        raise AssertionError("native runner attempted network access")

    monkeypatch.setattr(type(os.environ), "__getitem__", checked_getitem)
    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    report = await native.run_native_baseline()
    runtime, _, _ = harness(trusted_target={"opaque": CANARY})
    sink = CollectingTraceSink()
    with collecting_node(sink):
        await runtime.execute(native.native_call(CANARY))
        await runtime.execute(native.native_call("record-1"))
    traces = sink.safe_json()
    captured = capsys.readouterr()
    observable = (
        native.serialize_report(report) + traces + caplog.text + captured.out + captured.err
    )
    assert CANARY not in observable
    assert "Synthetic backend role record." not in traces
    for field in (
        "record_id",
        "summary",
        "trusted_target",
        "actor_user_id",
        "credential",
        "deadline",
    ):
        assert field not in traces
    assert '"workspace_id"' not in native.serialize_report(report)
    assert report.external_network_used is report.external_credentials_used is False


@pytest.mark.parametrize(
    "samples,expected",
    [
        ([0], (0, 0)),
        ([8, 2], (2, 8)),
        ([3, 1, 2], (2, 3)),
        (list(range(1, 21)), (10, 19)),
        (list(range(1, 22)), (11, 20)),
    ],
)
def test_nearest_rank_boundaries(samples, expected):
    assert native.latency_percentiles(samples) == expected


@pytest.mark.parametrize("samples", [[], [-1], [True], [1.5], [float("nan")]])
def test_invalid_latency_samples_fail_closed(samples):
    with pytest.raises(native.NativeBaselineError):
        native.latency_percentiles(samples)


async def test_report_samples_come_from_tool_span_and_independent_runs(monkeypatch):
    original_finish = CollectingTraceSink.finish
    invocations = []
    runs = []

    def finish(self, context, outcome):
        if outcome.span_kind == "tool":
            invocations.append(self.starts[context.context_id][1].metadata["tool_invocation_id"])
            runs.append(context.trace_identity.run_id)
            assert context.trace_identity.workspace_id == UUID(int=1)
            outcome = replace(outcome, latency_ms=100 if len(invocations) <= 3 else 17)
        else:
            outcome = replace(outcome, latency_ms=999)
        original_finish(self, context, outcome)

    monkeypatch.setattr(CollectingTraceSink, "finish", finish)
    report = await native.run_native_baseline()
    assert len(invocations) == len(set(invocations)) == 12
    assert runs == [UUID(int=100 + sequence) for sequence in range(12)]
    assert report.sample_count == 9
    assert report.steady_state_samples_ms == (17,) * 9
    assert report.p50_ms == report.p95_ms == 17
    serialized = native.serialize_report(report)
    assert native.NativeReport.model_validate_json(serialized) == report
    assert json.loads(serialized)["latency_source"] == "TraceSpanFinish.latency_ms"


@pytest.mark.parametrize(
    "change",
    [
        {"native_setup_ms": float("nan")},
        {"sample_count": 8},
        {"p95_ms": -1},
        {"steady_state_samples_ms": (0,)},
        {"functional_cases": ()},
        {"external_network_used": True},
    ],
)
async def test_malformed_internal_report_is_not_serialized(change):
    report = await native.run_native_baseline()
    with pytest.raises(ValidationError):
        native.serialize_report(report.model_copy(update=change))


async def test_missing_trace_cannot_be_reported_as_zero_latency(monkeypatch):
    monkeypatch.setattr(CollectingTraceSink, "start", lambda self, span: None)
    with pytest.raises(native.NativeBaselineError):
        await native.run_native_baseline()


@pytest.mark.parametrize(
    "serialized",
    [
        '{"record_id":"record-1","found":true,"summary":null}',
        '{"record_id":"record-2","found":true,"summary":"wrong"}',
        '{"record_id":"record-1","found":true,"summary":"wrong"}',
        "not-json-private-canary",
    ],
)
async def test_malformed_internal_runtime_result_fails_closed(serialized):
    class BrokenRuntime:
        async def execute(self, call):
            return serialized

    with pytest.raises(native.NativeBaselineError, match="native result invalid"):
        await native.measure_case(BrokenRuntime(), "record-1")


def test_cli_is_strict_json(capsys):
    assert native.main() == 0
    captured = capsys.readouterr()
    assert native.NativeReport.model_validate_json(captured.out).sample_count == 9
    assert captured.err == ""


@pytest.mark.parametrize("error", [native.NativeBaselineError, RuntimeError])
def test_cli_failure_is_sanitized(monkeypatch, capsys, error):
    async def broken():
        raise error(CANARY)

    monkeypatch.setattr(native, "run_native_baseline", broken)
    assert native.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error_category": "native_baseline_failed"}
    assert CANARY not in captured.err
