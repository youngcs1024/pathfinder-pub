"""Real SDK/child fault evidence, no DB, providers, or production fault settings."""

import asyncio
import json
import logging
import traceback
from pathlib import Path
from time import monotonic
from unittest.mock import AsyncMock

import pytest

from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.mcp_experiment import client as experiment
from app.tools.registry import ToolExecutionError, ToolOutputTooLargeError, ToolTimeoutError
from tests.tracing import collecting_node
from tests.unit.tools.test_mcp_experiment_client import bind, call

CANARY = "GATE12_RAW_SECRET_CANARY"
SECRET_KEYS = (
    "DASHSCOPE_API_KEY",
    "TAVILY_API_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "LANGFUSE_SECRET_KEY",
    "PF_GATE12_SECRET_CANARY",
)


def launch_fault(monkeypatch, tmp_path, mode):
    parameters = experiment._launch_parameters()
    parameters.args = [
        str(Path("tests/fixtures/tools/mcp_fault_server.py").resolve()),
        mode,
        str(tmp_path),
    ]
    monkeypatch.setattr(experiment, "_launch_parameters", lambda: parameters)
    # Real imports get enough time; failures still finish well inside production budget.
    monkeypatch.setattr(experiment, "DISCOVERY_SECONDS", 2.0)


def recorder_spy():
    recorder = InMemoryToolInvocationRecorder()
    for name in ("reserve", "start_attempt", "succeed", "fail"):
        setattr(recorder, name, AsyncMock(wraps=getattr(recorder, name)))
    return recorder


async def marker(path):
    async with asyncio.timeout(5):
        while not path.exists():
            # Poll a concrete child milestone, never assume readiness after a fixed delay.
            await asyncio.sleep(0.01)


def assert_reaped(tmp_path):
    identity = json.loads(tmp_path.joinpath("pid").read_text())
    stat = Path(f"/proc/{identity['pid']}/stat")
    if stat.exists():
        assert stat.read_text().split()[21] != identity["start"], "same child survived cleanup"


async def fresh_experiment():
    async with experiment.open_mcp_experiment() as registry:
        assert json.loads(await bind(registry).execute(call()))["found"] is True


def assert_call_trace(sink, *, transport_status, tool_category=None):
    assert sink.starts.keys() == sink.finishes.keys()
    transports = [(c, s) for c, s in sink.starts.values() if s.span_kind == "mcp_transport"]
    assert len(transports) == 1
    context, span = transports[0]
    assert dict(span.metadata) == {"transport": "stdio"}
    assert span.parent.span_kind == "tool"
    tool_context, tool = sink.starts[span.parent.context_id]
    assert tool.parent.span_kind == "graph_node"
    assert span.trace_identity == tool.trace_identity
    assert span.segment_identity == tool.segment_identity
    finish = sink.finishes[context.context_id]
    assert finish.status == transport_status
    assert (
        finish.error_category
        == {"succeeded": None, "failed": "mcp_call_failed", "cancelled": "cancelled"}[
            transport_status
        ]
    )
    tool_finish = sink.finishes[tool_context.context_id]
    assert tool_finish.error_category == tool_category
    assert tool_finish.status == (
        "succeeded"
        if tool_category is None
        else "cancelled"
        if tool_category == "cancelled"
        else "failed"
    )
    assert "record-1" not in sink.safe_json()
    assert "private-trusted-context-canary" not in sink.safe_json()


def assert_redacted(caught, caplog, capfd, sink=None, recorder=None):
    capture = capfd.readouterr()
    observed = capture.out + capture.err + caplog.text
    if caught is not None:
        observed += "".join(traceback.format_exception(caught.value))
    if sink is not None:
        observed += sink.safe_json()
    if recorder is not None:
        observed += repr(
            [
                getattr(recorder, name).call_args_list
                for name in ("reserve", "start_attempt", "succeed", "fail")
            ]
        )
    assert CANARY not in observed


@pytest.mark.parametrize(
    "mode",
    ["silent_discovery", "early_exit", "malformed_json", "stdout_flood", "stdout_unterminated"],
)
async def test_discovery_faults_reap_and_reopen(monkeypatch, tmp_path, caplog, capfd, mode):
    caplog.set_level(logging.DEBUG)
    recorder = recorder_spy()
    started = monotonic()
    with monkeypatch.context() as patch:
        launch_fault(patch, tmp_path, mode)
        with pytest.raises(experiment.MCPDiscoveryError) as caught:
            async with experiment.open_mcp_experiment(recorder=recorder):
                pytest.fail("discovery exposed a Registry")
    assert monotonic() - started < 10
    assert_reaped(tmp_path)
    recorder.reserve.assert_not_awaited()
    recorder.start_attempt.assert_not_awaited()
    if mode.startswith("stdout_"):
        assert tmp_path.joinpath("emitted").exists()
    assert_redacted(caught, caplog, capfd)
    await fresh_experiment()


@pytest.mark.parametrize(
    ("mode", "error", "category", "transport_status"),
    [
        ("call_hang", ToolTimeoutError, "provider_timeout", "cancelled"),
        ("crash_on_call", ToolExecutionError, "tool_execution_failed", "failed"),
        ("malformed_call", ToolTimeoutError, "provider_timeout", "cancelled"),
        ("oversized_output", ToolOutputTooLargeError, "invalid_tool_output", "succeeded"),
    ],
)
async def test_admitted_failure_accounting_and_trace(
    monkeypatch,
    tmp_path,
    collecting_trace_sink,
    caplog,
    capfd,
    mode,
    error,
    category,
    transport_status,
):
    caplog.set_level(logging.DEBUG)
    recorder = recorder_spy()
    with monkeypatch.context() as patch:
        launch_fault(patch, tmp_path, mode)
        async with experiment.open_mcp_experiment(recorder=recorder) as registry:
            with collecting_node(collecting_trace_sink), pytest.raises(error) as caught:
                await bind(registry).execute(call())
    assert tmp_path.joinpath("called").exists()
    assert_reaped(tmp_path)
    for name in ("reserve", "start_attempt", "fail"):
        getattr(recorder, name).assert_awaited_once()
    recorder.succeed.assert_not_awaited()
    assert recorder.fail.call_args.kwargs["error_category"] == category
    assert_call_trace(
        collecting_trace_sink, transport_status=transport_status, tool_category=category
    )
    assert_redacted(caught, caplog, capfd, collecting_trace_sink, recorder)
    await fresh_experiment()


async def test_real_active_io_cancellation(monkeypatch, tmp_path, collecting_trace_sink):
    recorder = recorder_spy()
    with monkeypatch.context() as patch:
        launch_fault(patch, tmp_path, "call_hang")
        async with experiment.open_mcp_experiment(recorder=recorder) as registry:
            with collecting_node(collecting_trace_sink):
                task = asyncio.create_task(bind(registry).execute(call()))
                try:
                    await marker(tmp_path / "called")
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                finally:
                    if not task.done():
                        task.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await task
    assert_reaped(tmp_path)
    for name in ("reserve", "start_attempt", "fail"):
        getattr(recorder, name).assert_awaited_once()
    recorder.succeed.assert_not_awaited()
    assert recorder.fail.call_args.kwargs["error_category"] == "cancelled"
    assert_call_trace(
        collecting_trace_sink, transport_status="cancelled", tool_category="cancelled"
    )
    await fresh_experiment()


@pytest.mark.parametrize("mode", ["stderr_flood", "env_probe", "linger_on_shutdown"])
async def test_real_env_flood_and_shutdown(
    monkeypatch, tmp_path, caplog, capfd, collecting_trace_sink, mode
):
    for key in (*SECRET_KEYS, "LC_CTYPE"):
        monkeypatch.setenv(key, CANARY)
    recorder = recorder_spy()
    with monkeypatch.context() as patch:
        launch_fault(patch, tmp_path, mode)
        async with experiment.open_mcp_experiment(recorder=recorder) as registry:
            with collecting_node(collecting_trace_sink):
                assert json.loads(await bind(registry).execute(call()))["found"] is True
            closing = monotonic()
    assert monotonic() - closing < experiment.SHUTDOWN_SECONDS
    assert_reaped(tmp_path)
    if mode == "env_probe":
        report = json.loads(tmp_path.joinpath("env").read_text())
        keys = set(report["runtime_keys"])
        inherited = set(report["inherited_keys"])
        assert not keys.intersection(SECRET_KEYS)
        assert not any(key.startswith("PF_") for key in keys)
        assert inherited <= {"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
        # CPython locale coercion adds LC_CTYPE after exec; not inherited from the parent.
        generated = keys - inherited
        assert generated <= {"LC_CTYPE"}
        if generated:
            assert report["python_locale_coercion"] is True
    if mode == "linger_on_shutdown":
        assert tmp_path.joinpath("stdin_closed").exists()
        assert tmp_path.joinpath("term").exists()
    for name in ("reserve", "start_attempt", "succeed"):
        getattr(recorder, name).assert_awaited_once()
    recorder.fail.assert_not_awaited()
    assert_call_trace(collecting_trace_sink, transport_status="succeeded")
    assert_redacted(None, caplog, capfd, collecting_trace_sink, recorder)
    await fresh_experiment()
