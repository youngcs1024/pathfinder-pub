import asyncio
import json
import sys
import traceback
from contextlib import asynccontextmanager
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.llm.ports import ModelToolCall
from app.tools.contracts import ToolRunContext
from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.mcp_experiment import client as experiment
from app.tools.registry import (
    ToolCallLimitExceededError,
    ToolCancelledError,
    ToolDeadlineExceededError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolNotAllowedError,
    ToolOutputTooLargeError,
    ToolOutputValidationError,
)
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    LookupSyntheticRecordInput,
    LookupSyntheticRecordOutput,
    resolve_synthetic_record,
)
from tests.tracing import collecting_node

CANARY = "private-trusted-context-canary"


def run_context() -> ToolRunContext:
    return ToolRunContext(
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        run_id=uuid4(),
        action_intent_id=uuid4(),
        approval_request_id=uuid4(),
        trusted_target={"credential": CANARY, "target": CANARY},
        deadline=100.0,
        cancellation=SimpleNamespace(is_cancelled=lambda: False),
    )


def bind(registry, context=None):
    return registry.bind(
        policy_name=experiment.MCP_EXPERIMENT_POLICY.name,
        context=context or run_context(),
        clock=lambda: 10.0,
    )


def call(record_id="record-1", **extra):
    return ModelToolCall(
        call_id=str(uuid4()),
        name=LOOKUP_SYNTHETIC_RECORD_SPEC.name,
        arguments={"record_id": record_id, **extra},
    )


@pytest.fixture
def recorder():
    recorder = InMemoryToolInvocationRecorder()
    for name in ("reserve", "start_attempt", "succeed", "fail"):
        setattr(recorder, name, AsyncMock(wraps=getattr(recorder, name)))
    return recorder


@pytest.fixture
def client():
    async def lookup(_name, arguments):
        output = resolve_synthetic_record(LookupSyntheticRecordInput(**arguments))
        return SimpleNamespace(is_error=False, structured_content=output.model_dump())

    return SimpleNamespace(
        protocol_version="2026-07-28",
        server_info=SimpleNamespace(name="pathfinder-gate12-teaching", version="0.1.0"),
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                next_cursor=None,
                tools=[
                    SimpleNamespace(
                        name=LOOKUP_SYNTHETIC_RECORD_SPEC.name,
                        input_schema={
                            "type": "object",
                            "properties": {"record_id": {"type": "string", "maxLength": 64}},
                            "required": ["record_id"],
                        },
                        output_schema=LookupSyntheticRecordOutput.model_json_schema(),
                        annotations={"readOnlyHint": False},
                    )
                ],
            )
        ),
        list_resources=AsyncMock(return_value=SimpleNamespace(resources=[], next_cursor=None)),
        list_resource_templates=AsyncMock(
            return_value=SimpleNamespace(resource_templates=[], next_cursor=None)
        ),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[], next_cursor=None)),
        call_tool=AsyncMock(side_effect=lookup),
    )


async def test_discovery_and_local_contract_authority(client, recorder):
    registry = await experiment._validated_registry(client, recorder)
    spec = registry._specs[LOOKUP_SYNTHETIC_RECORD_SPEC.name]
    for field in fields(spec):
        if field.name != "handler":
            assert getattr(spec, field.name) == getattr(LOOKUP_SYNTHETIC_RECORD_SPEC, field.name)
    assert spec.handler is not LOOKUP_SYNTHETIC_RECORD_SPEC.handler
    assert (spec.effect.value, spec.credential_source.value) == ("read_only", "none")
    assert (spec.timeout_seconds, spec.max_attempts, spec.per_run_call_limit) == (1.0, 1, 3)
    assert spec.max_output_bytes == 4096
    runtime = bind(registry)
    assert [tool.name for tool in runtime.model_tools()] == [spec.name]
    client.list_tools.assert_awaited_once_with()
    with pytest.raises(ToolNotAllowedError):
        await runtime.execute(ModelToolCall(call_id="bad", name="search_web", arguments={}))
    recorder.reserve.assert_not_awaited()
    client.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "drift",
    [
        "name",
        "version",
        "protocol",
        "no_identity",
        "missing",
        "extra",
        "pagination",
        "object_type",
        "field",
        "type",
        "required",
        "max_length",
        "output",
        "resources",
        "templates",
        "prompts",
    ],
)
async def test_discovery_drift_never_constructs_registry(client, recorder, monkeypatch, drift):
    tool = client.list_tools.return_value.tools[0]
    schema = tool.input_schema
    if drift == "name":
        client.server_info.name = CANARY
    elif drift == "version":
        client.server_info.version = "0.2.0"
    elif drift == "protocol":
        client.protocol_version = "2025-11-25"
    elif drift == "no_identity":
        client.server_info = None
    elif drift == "missing":
        client.list_tools.return_value.tools = []
    elif drift == "extra":
        client.list_tools.return_value.tools.append(SimpleNamespace(name=CANARY))
    elif drift == "pagination":
        client.list_tools.return_value.next_cursor = CANARY
    elif drift == "object_type":
        schema["type"] = "array"
    elif drift == "field":
        schema["properties"] = {"workspace_id": {"type": "string"}}
    elif drift == "type":
        schema["properties"]["record_id"]["type"] = "integer"
    elif drift == "required":
        schema["required"] = []
    elif drift == "max_length":
        schema["properties"]["record_id"]["maxLength"] = 65
    elif drift == "output":
        tool.output_schema = {"type": "object"}
    elif drift == "resources":
        client.list_resources.return_value.resources = [CANARY]
    elif drift == "templates":
        client.list_resource_templates.return_value.resource_templates = [CANARY]
    elif drift == "prompts":
        client.list_prompts.return_value.prompts = [CANARY]
    constructor = Mock()
    monkeypatch.setattr(experiment, "ToolRegistry", constructor)
    with pytest.raises(experiment.MCPDiscoveryError) as caught:
        await experiment._validated_registry(client, recorder)
    constructor.assert_not_called()
    client.call_tool.assert_not_awaited()
    assert CANARY not in str(caught.value)


@pytest.mark.parametrize("record_id", ["record-1", "missing-record", "record-2", "x" * 64])
async def test_results_and_only_business_input_crosses(client, recorder, record_id):
    runtime = bind(await experiment._validated_registry(client, recorder))
    result = await runtime.execute(call(record_id))
    assert (
        json.loads(result)
        == resolve_synthetic_record(LookupSyntheticRecordInput(record_id=record_id)).model_dump()
    )
    client.call_tool.assert_awaited_once_with(
        LOOKUP_SYNTHETIC_RECORD_SPEC.name, {"record_id": record_id}
    )
    recorder.start_attempt.assert_awaited_once()
    recorder.succeed.assert_awaited_once()
    recorder.fail.assert_not_awaited()
    assert CANARY not in result


@pytest.mark.parametrize(
    "arguments",
    [
        {"record_id": " "},
        {"record_id": 123},
        {"record_id": "x" * 65},
        *[{"record_id": "record-1", key: CANARY} for key in ("workspace_id", "role", "target")],
    ],
)
async def test_input_rejected_before_reservation_or_mcp(client, recorder, arguments):
    runtime = bind(await experiment._validated_registry(client, recorder))
    with pytest.raises(ToolInputValidationError):
        await runtime.execute(ModelToolCall(call_id="bad", name=call().name, arguments=arguments))
    client.call_tool.assert_not_awaited()
    recorder.reserve.assert_not_awaited()
    recorder.start_attempt.assert_not_awaited()


@pytest.mark.parametrize("raises", [False, True])
async def test_call_failure_is_sanitized_accounted_once(
    client, recorder, collecting_trace_sink, capsys, caplog, raises
):
    client.call_tool.side_effect = RuntimeError(CANARY) if raises else None
    client.call_tool.return_value = SimpleNamespace(is_error=True, content=[CANARY])
    runtime = bind(await experiment._validated_registry(client, recorder))
    with collecting_node(collecting_trace_sink), pytest.raises(ToolExecutionError) as caught:
        await runtime.execute(call())
    client.call_tool.assert_awaited_once()
    recorder.start_attempt.assert_awaited_once()
    recorder.fail.assert_awaited_once()
    assert recorder.fail.call_args.kwargs["error_category"] == "tool_execution_failed"
    recorder.succeed.assert_not_awaited()
    captured = capsys.readouterr()
    observable = (
        "".join(traceback.format_exception(caught.value))
        + captured.out
        + captured.err
        + caplog.text
        + repr(collecting_trace_sink.starts)
        + repr(collecting_trace_sink.finishes)
    )
    assert CANARY not in observable


@pytest.mark.parametrize(
    "output",
    [
        {"record_id": "record-1", "found": True, "summary": None},
        {"record_id": "record-1", "found": False},
        {"record_id": "record-1", "found": "false", "summary": None},
        *[
            {"record_id": "record-1", "found": False, "summary": None, key: CANARY}
            for key in ("role", "effect", "target", "context", "workspace")
        ],
    ],
)
async def test_registry_is_final_output_authority(client, recorder, output, collecting_trace_sink):
    client.call_tool.side_effect = None
    client.call_tool.return_value = SimpleNamespace(is_error=False, structured_content=output)
    registry = await experiment._validated_registry(client, recorder)
    context = run_context()
    target = dict(context.trusted_target)
    spec = registry._specs[LOOKUP_SYNTHETIC_RECORD_SPEC.name]
    runtime = bind(registry, context)
    with collecting_node(collecting_trace_sink), pytest.raises(ToolOutputValidationError):
        await runtime.execute(call())
    assert registry._specs[spec.name] is spec
    assert spec.effect == LOOKUP_SYNTHETIC_RECORD_SPEC.effect
    assert context.trusted_target == target
    with pytest.raises(ToolNotAllowedError):
        await runtime.execute(ModelToolCall(call_id="forged", name="search_web", arguments={}))
    client.call_tool.assert_awaited_once()
    recorder.reserve.assert_awaited_once()
    assert recorder.fail.call_args.kwargs["error_category"] == "invalid_tool_output"
    recorder.succeed.assert_not_awaited()
    assert any(
        finish.error_category == "invalid_tool_output"
        for finish in collecting_trace_sink.finishes.values()
    )


async def test_registry_output_cap_remains_effective(client, recorder):
    client.call_tool.side_effect = None
    client.call_tool.return_value = SimpleNamespace(
        is_error=False,
        structured_content={"record_id": "record-1", "found": True, "summary": "x" * 4096},
    )
    runtime = bind(await experiment._validated_registry(client, recorder))
    with pytest.raises(ToolOutputTooLargeError):
        await runtime.execute(call())
    assert recorder.fail.call_args.kwargs["error_category"] == "invalid_tool_output"


async def test_fourth_call_is_rejected_locally(client, recorder):
    runtime = bind(await experiment._validated_registry(client, recorder))
    for record_id in ("record-1", "missing-record", "record-2"):
        await runtime.execute(call(record_id))
    with pytest.raises(ToolCallLimitExceededError):
        await runtime.execute(call())
    assert client.call_tool.await_count == recorder.start_attempt.await_count == 3


@pytest.mark.parametrize("cancelled", [True, False])
async def test_preflight_cancellation_and_deadline(client, recorder, cancelled):
    context = run_context()
    context = (
        replace(context, cancellation=SimpleNamespace(is_cancelled=lambda: True))
        if cancelled
        else replace(context, deadline=9.0)
    )
    runtime = bind(await experiment._validated_registry(client, recorder), context)
    with pytest.raises(ToolCancelledError if cancelled else ToolDeadlineExceededError):
        await runtime.execute(call())
    client.call_tool.assert_not_awaited()
    recorder.reserve.assert_not_awaited()


async def test_adapter_preserves_cancellation(client, recorder):
    client.call_tool.side_effect = asyncio.CancelledError
    runtime = bind(await experiment._validated_registry(client, recorder))
    with pytest.raises(asyncio.CancelledError):
        await runtime.execute(call())
    assert recorder.fail.call_args.kwargs["error_category"] == "cancelled"


async def test_static_launch_auto_mode_and_one_session(client, monkeypatch):
    lifecycle = []

    @asynccontextmanager
    async def transport(parameters, *, errlog):
        assert parameters.command == str(Path(sys.executable).absolute())
        assert parameters.args == ["-m", "app.tools.mcp_experiment.server"]
        assert parameters.cwd == experiment.PROJECT_ROOT
        assert parameters.env == {}
        assert errlog.name == "/dev/null"
        lifecycle.append("spawn")
        try:
            yield ("read", "write")
        finally:
            lifecycle.append("close")

    @asynccontextmanager
    async def sdk_client(streams, **kwargs):
        assert kwargs == {"mode": "auto", "cache": None, "input_required_max_rounds": 0}
        async with streams as opened:
            assert opened == ("read", "write")
            yield client

    monkeypatch.setattr(experiment, "stdio_client", transport)
    monkeypatch.setattr(experiment, "Client", sdk_client)
    async with experiment.open_mcp_experiment() as registry:
        runtime = bind(registry)
        await runtime.execute(call("record-1"))
        await runtime.execute(call("record-2"))
        assert lifecycle == ["spawn"]
    assert lifecycle == ["spawn", "close"]
    client.list_tools.assert_awaited_once()


async def test_nonexistent_executable_is_startup_failure(monkeypatch, recorder):
    parameters = experiment._launch_parameters()
    parameters.command = "/nonexistent/pathfinder-gate12-python"
    monkeypatch.setattr(experiment, "_launch_parameters", lambda: parameters)
    with pytest.raises(experiment.MCPStartupError):
        async with experiment.open_mcp_experiment(recorder=recorder):
            pytest.fail("startup exposed a Registry")
    recorder.reserve.assert_not_awaited()


async def test_outer_startup_timeout(monkeypatch, recorder):
    entered = asyncio.Event()

    @asynccontextmanager
    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield

    monkeypatch.setattr(experiment, "stdio_client", blocked)
    monkeypatch.setattr(experiment, "STARTUP_SECONDS", 0.01)
    with pytest.raises(experiment.MCPStartupError):
        async with experiment.open_mcp_experiment(recorder=recorder):
            pytest.fail("startup exposed a Registry")
    assert entered.is_set()
    recorder.reserve.assert_not_awaited()


@pytest.mark.parametrize("cleanup", ["timeout", "exception", "survivor", "success"])
@pytest.mark.parametrize("body", ["success", "exception", "cancelled"])
async def test_cleanup_precedence_and_budget(client, monkeypatch, cleanup, body):
    import logging

    from app.tools.mcp_experiment._logging import _SURVIVOR_WARNING

    exiting = asyncio.Event()

    @asynccontextmanager
    async def transport(*args, **kwargs):
        try:
            yield ("read", "write")
        finally:
            exiting.set()
            if cleanup == "timeout":
                await asyncio.Event().wait()
            elif cleanup == "exception":
                raise RuntimeError(CANARY)
            elif cleanup == "survivor":
                logging.getLogger("mcp.client.stdio").warning(_SURVIVOR_WARNING, 123)

    @asynccontextmanager
    async def sdk_client(*args, **kwargs):
        yield client

    monkeypatch.setattr(experiment, "stdio_client", transport)
    monkeypatch.setattr(experiment, "Client", sdk_client)
    monkeypatch.setattr(experiment, "SHUTDOWN_SECONDS", 0.01)
    original = ValueError("body error") if body == "exception" else asyncio.CancelledError()

    async def run():
        async with experiment.open_mcp_experiment():
            if body != "success":
                raise original

    if cleanup != "success":
        with pytest.raises(experiment.MCPCleanupError) as caught:
            await run()
        assert CANARY not in "".join(traceback.format_exception(caught.value))
    elif body != "success":
        with pytest.raises(type(original)) as caught:
            await run()
        assert caught.value is original
    else:
        await run()
    assert exiting.is_set()


def test_pinned_sdk_parser_logging_guard(caplog):
    import inspect
    import logging
    from importlib.metadata import version

    import mcp.client.stdio as stdio
    import mcp.shared.jsonrpc_dispatcher as dispatcher

    from app.tools.mcp_experiment._logging import _SURVIVOR_WARNING, _sdk_logging_guard

    assert version("mcp") == "2.1.1", "SDK upgrade requires logging/lifecycle re-review"
    assert _SURVIVOR_WARNING in inspect.getsource(stdio)
    assert "transport yielded exception: %r" in inspect.getsource(dispatcher)
    logger = logging.getLogger("mcp.client.stdio")
    original_filters, original_level = list(logger.filters), logger.level
    caplog.set_level(logging.DEBUG)
    with _sdk_logging_guard() as outer:
        error = stdio._parse_line(CANARY)
        logging.getLogger("mcp.shared.jsonrpc_dispatcher").debug(
            "transport yielded exception: %r", error
        )
        assert isinstance(error, Exception)
        with _sdk_logging_guard() as inner:
            logger.warning(_SURVIVOR_WARNING, 123)
            assert inner.survivor
        assert not outer.survivor
        logger.warning(_SURVIVOR_WARNING, 123)
        assert outer.survivor
    assert CANARY not in caplog.text
    assert logger.filters == original_filters
    assert logger.level == original_level


@pytest.mark.parametrize("failure", ["drop", "start", "finish"])
async def test_transport_trace_sink_failure_is_lossy(client, recorder, failure):
    from tests.tracing import CollectingTraceSink, FaultTraceSink

    sink = FaultTraceSink(CollectingTraceSink(), failure, selected_kind="mcp_transport")
    runtime = bind(await experiment._validated_registry(client, recorder))
    with collecting_node(sink):
        assert json.loads(await runtime.execute(call()))["found"] is True
    recorder.succeed.assert_awaited_once()
    recorder.fail.assert_not_awaited()
