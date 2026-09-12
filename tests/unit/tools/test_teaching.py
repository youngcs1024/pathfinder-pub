import json
import socket
from uuid import UUID

import pytest

from app.domain.tool_effects import ToolEffect
from app.llm.ports import ModelToolCall
from app.tools import teaching
from app.tools.contracts import CredentialSource, ToolRunContext
from app.tools.registry import ToolCallLimitExceededError
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    MANUAL_LEARNING_POLICY,
    MANUAL_LEARNING_POLICY_NAME,
    LookupSyntheticRecordInput,
    LookupSyntheticRecordOutput,
    create_teaching_tool_registry,
)


class _NotCancelled:
    def is_cancelled(self) -> bool:
        return False


class _InvocationIds:
    def __init__(self) -> None:
        self._value = 100

    def __call__(self) -> UUID:
        invocation_id = UUID(int=self._value)
        self._value += 1
        return invocation_id


def _runtime(*, trusted_target: dict[str, str] | None = None):
    return create_teaching_tool_registry().bind(
        policy_name=MANUAL_LEARNING_POLICY_NAME,
        context=ToolRunContext(
            workspace_id=UUID(int=1),
            actor_user_id=UUID(int=2),
            run_id=UUID(int=3),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=trusted_target,
            deadline=100.0,
            cancellation=_NotCancelled(),
        ),
        invocation_id_factory=_InvocationIds(),
        clock=lambda: 10.0,
    )


def _call(call_id: str, record_id: str) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name=LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
        arguments={"record_id": record_id},
    )


def test_teaching_tool_uses_the_locked_read_only_contract() -> None:
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.effect is ToolEffect.READ_ONLY
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.credential_source is CredentialSource.NONE
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.timeout_seconds == 1.0
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.max_attempts == 1
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.per_run_call_limit == 3
    assert LOOKUP_SYNTHETIC_RECORD_SPEC.max_output_bytes == 4096
    assert MANUAL_LEARNING_POLICY.allowed_tool_names == frozenset(
        {LOOKUP_SYNTHETIC_RECORD_TOOL_NAME}
    )
    assert MANUAL_LEARNING_POLICY.allowed_effects == frozenset({ToolEffect.READ_ONLY})

    tools = _runtime().model_tools()
    assert len(tools) == 1
    assert tools[0].name == LOOKUP_SYNTHETIC_RECORD_TOOL_NAME
    assert set(tools[0].input_schema["properties"]) == {"record_id"}


@pytest.mark.asyncio
async def test_teaching_tool_returns_known_and_unknown_records_then_enforces_limit() -> None:
    runtime = _runtime()

    known = await runtime.execute(_call("call-1", "record-1"))
    unknown = await runtime.execute(_call("call-2", "missing-record"))
    second = await runtime.execute(_call("call-3", "record-2"))

    assert json.loads(known) == {
        "found": True,
        "record_id": "record-1",
        "summary": "Synthetic backend role record.",
    }
    assert json.loads(unknown) == {
        "found": False,
        "record_id": "missing-record",
        "summary": None,
    }
    assert json.loads(second)["found"] is True
    with pytest.raises(ToolCallLimitExceededError, match="limit"):
        await runtime.execute(_call("call-4", "record-1"))


async def test_registry_handler_uses_shared_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = []

    def resolver(value: LookupSyntheticRecordInput) -> LookupSyntheticRecordOutput:
        inputs.append(value)
        return LookupSyntheticRecordOutput(record_id=value.record_id, found=False, summary=None)

    monkeypatch.setattr(teaching, "resolve_synthetic_record", resolver)
    result = await _runtime().execute(_call("shared-resolver", "record-1"))
    assert inputs == [LookupSyntheticRecordInput(record_id="record-1")]
    assert json.loads(result) == {"record_id": "record-1", "found": False, "summary": None}


@pytest.mark.asyncio
async def test_teaching_tool_is_offline_and_does_not_read_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "teaching-tool-provider-secret-canary"
    for name in (
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.setenv(name, canary)

    def deny_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("teaching tool attempted network access")

    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    runtime = _runtime(trusted_target={"opaque": canary})

    result = await runtime.execute(_call("call-offline", "record-1"))
    captured = capsys.readouterr()

    observable = "\n".join(
        (
            result,
            repr(runtime),
            json.dumps([tool.model_dump(mode="json") for tool in runtime.model_tools()]),
            caplog.text,
            captured.out,
            captured.err,
        )
    )
    assert canary not in observable
