import logging
import os
import socket
from unittest.mock import Mock

import pytest
from mcp import Client

from app.domain.tool_effects import ToolEffect
from app.tools.mcp_experiment import server
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    LookupSyntheticRecordInput,
    LookupSyntheticRecordOutput,
    resolve_synthetic_record,
)


async def test_identity_protocol_and_only_teaching_tool_surface() -> None:
    async with Client(server.build_server()) as client:
        assert client.server_info.name == "pathfinder-gate12-teaching"
        assert client.server_info.version == "0.1.0"
        assert client.protocol_version == server.PROTOCOL_TARGET == "2026-07-28"
        tools = (await client.list_tools()).tools
        assert [tool.name for tool in tools] == [LOOKUP_SYNTHETIC_RECORD_TOOL_NAME]
        assert (await client.list_resources()).resources == []
        assert (await client.list_resource_templates()).resource_templates == []
        assert (await client.list_prompts()).prompts == []

    tool = tools[0]
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"record_id"}
    assert schema["required"] == ["record_id"]
    assert schema["properties"]["record_id"]["type"] == "string"
    assert schema["properties"]["record_id"]["maxLength"] == 64
    assert tool.output_schema == LookupSyntheticRecordOutput.model_json_schema()
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.open_world_hint is False
    # MCP annotations describe capability; the local ToolSpec remains effect authority.
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.effect is ToolEffect.READ_ONLY


async def test_normal_results_match_native_contract_and_are_deterministic() -> None:
    async with Client(server.build_server()) as client:
        for record_id, summary in (
            ("record-1", "Synthetic backend role record."),
            ("missing-record", None),
            ("record-2", "Synthetic platform engineering record."),
            ("x" * 64, None),
        ):
            expected = LookupSyntheticRecordOutput(
                record_id=record_id, found=summary is not None, summary=summary
            )
            assert (
                resolve_synthetic_record(LookupSyntheticRecordInput(record_id=record_id))
                == expected
            )
            for _ in range(3):
                result = await client.call_tool(
                    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, {"record_id": record_id}
                )
                assert not result.is_error
                assert (
                    LookupSyntheticRecordOutput.model_validate(
                        result.structured_content, strict=True
                    )
                    == expected
                )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"record_id": " "},
        {"record_id": ""},
        {"record_id": 123},
        {"record_id": True},
        {"record_id": None},
        {"record_id": ["private-canary"]},
        {"record_id": "private-canary" * 5},
        {"record_id": "x" * 65},
        *[
            {"record_id": "record-1", field: "private-canary"}
            for field in (
                "workspace_id",
                "actor_user_id",
                "run_id",
                "invocation_id",
                "role",
                "credential",
                "target",
                "trusted_target",
                "action_intent_id",
                "approval_request_id",
                "deadline",
                "budget",
                "cancellation",
            )
        ],
    ],
)
async def test_invalid_input_fails_before_resolver_without_echo(
    arguments: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resolver = Mock(side_effect=AssertionError("private-canary"))
    monkeypatch.setattr(server, "resolve_synthetic_record", resolver)
    caplog.set_level(logging.INFO)
    async with Client(server.build_server()) as client:
        result = await client.call_tool(LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, arguments)
    assert result.is_error
    assert result.structured_content is None
    assert [item.text for item in result.content] == [server.INPUT_FAILURE]
    resolver.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-canary" not in result.model_dump_json() + captured.err + caplog.text


@pytest.mark.parametrize("bad_output", [False, True])
async def test_resolver_failure_is_sanitized_and_next_call_works(
    bad_output: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    resolver = Mock(
        return_value={"record_id": "private-canary", "found": True, "summary": None},
        side_effect=None if bad_output else RuntimeError("private-canary"),
    )
    monkeypatch.setattr(server, "resolve_synthetic_record", resolver)
    caplog.set_level(logging.INFO)
    async with Client(server.build_server()) as client:
        result = await client.call_tool(
            LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, {"record_id": "record-1"}
        )
        assert result.is_error
        assert result.structured_content is None
        assert server.LOOKUP_FAILURE in result.content[0].text
        resolver.assert_called_once_with(LookupSyntheticRecordInput(record_id="record-1"))
        monkeypatch.setattr(server, "resolve_synthetic_record", resolve_synthetic_record)
        assert not (
            await client.call_tool(LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, {"record_id": "record-2"})
        ).is_error
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-canary" not in result.model_dump_json() + captured.err + caplog.text
    assert "Traceback" not in captured.err + caplog.text


async def test_server_path_needs_no_network_or_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_names = {
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    }
    for key in secret_names:
        monkeypatch.setenv(key, "private-canary")
    original_getitem = type(os.environ).__getitem__

    def checked_getitem(environment, key):
        assert key not in secret_names and not key.startswith("PF_")
        return original_getitem(environment, key)

    def deny_network(*_args, **_kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(type(os.environ), "__getitem__", checked_getitem)
    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    async with Client(server.build_server()) as client:
        result = await client.call_tool(
            LOOKUP_SYNTHETIC_RECORD_TOOL_NAME, {"record_id": "record-1"}
        )
    assert not result.is_error
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private-canary" not in result.model_dump_json() + captured.err + caplog.text
    # In-memory code-path evidence only; no child environment isolation is claimed.


def test_entrypoint_uses_only_sdk_stdio_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    built = Mock()
    monkeypatch.setattr(server, "build_server", Mock(return_value=built))
    server.main()
    built.run.assert_called_once_with(transport="stdio")
