import json
from collections.abc import Mapping, Sequence
from uuid import UUID

import pytest

from app.agents.contracts import AgentLoopResultV1
from app.agents.manual_loop import (
    ManualLoopLimitExceeded,
    ManualLoopProtocolError,
    run_manual_tool_loop,
)
from app.domain.tool_effects import ToolEffect
from app.llm.fake import (
    ScriptedFakeChatModel,
    ScriptedFakeFailure,
    ScriptedFakeProviderError,
)
from app.llm.ports import (
    ChatMessage,
    ChatModelResult,
    ModelToolCall,
    ModelToolSchema,
    ModelUsage,
)
from app.tools.contracts import (
    CredentialSource,
    GraphToolPolicy,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolRunContext,
    ToolSpec,
)
from app.tools.registry import (
    ToolInputValidationError,
    ToolNotAllowedError,
    ToolOutputValidationError,
    ToolRegistry,
)


class _RecordingModel:
    def __init__(self, responses: Sequence[ChatModelResult]) -> None:
        self._responses = list(responses)
        self.message_snapshots: list[tuple[ChatMessage, ...]] = []
        self.tool_snapshots: list[tuple[ModelToolSchema, ...]] = []
        self.metadata_snapshots: list[dict[str, str]] = []

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
    ) -> ChatModelResult:
        self.message_snapshots.append(tuple(message.model_copy(deep=True) for message in messages))
        self.tool_snapshots.append(tuple(tool.model_copy(deep=True) for tool in tools))
        self.metadata_snapshots.append(dict(metadata))
        if not self._responses:
            raise AssertionError("recording model received an unexpected invocation")
        return self._responses.pop(0)


class _TestToolInput(ToolInputModel):
    record_id: str


class _TestToolOutput(ToolOutputModel):
    record_id: str
    status: str


class _NotCancelled:
    def is_cancelled(self) -> bool:
        return False


class _RecordingHandler:
    def __init__(self) -> None:
        self.next_result: object = _TestToolOutput(
            record_id="record-default",
            status="synthetic",
        )
        self.contexts: list[ToolExecutionContext] = []

    async def __call__(
        self,
        _tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object:
        self.contexts.append(context)
        return self.next_result


class _RecordingToolRuntime:
    def __init__(
        self,
        results: Mapping[str, object],
        *,
        trusted_target: Mapping[str, str] | None = None,
    ) -> None:
        self._results = dict(results)
        self.calls: list[ModelToolCall] = []
        self._handler = _RecordingHandler()
        spec = ToolSpec(
            name="lookup",
            description="Look up a synthetic record",
            input_model=_TestToolInput,
            output_model=_TestToolOutput,
            effect=ToolEffect.READ_ONLY,
            credential_source=CredentialSource.NONE,
            timeout_seconds=1.0,
            max_attempts=1,
            per_run_call_limit=20,
            max_output_bytes=4096,
            handler=self._handler,
        )
        registry = ToolRegistry(
            specs=(spec,),
            policies=(
                GraphToolPolicy(
                    name="manual_learning_loop",
                    allowed_tool_names=frozenset({spec.name}),
                    allowed_effects=frozenset({ToolEffect.READ_ONLY}),
                ),
            ),
        )
        self._runtime = registry.bind(
            policy_name="manual_learning_loop",
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
            invocation_id_factory=self._invocation_id,
            clock=lambda: 10.0,
        )
        self._next_invocation_id = 100

    def _invocation_id(self) -> UUID:
        invocation_id = UUID(int=self._next_invocation_id)
        self._next_invocation_id += 1
        return invocation_id

    def model_tools(self) -> tuple[ModelToolSchema, ...]:
        return self._runtime.model_tools()

    def validate_call(self, call: ModelToolCall) -> None:
        self._runtime.validate_call(call)

    @property
    def handler_contexts(self) -> tuple[ToolExecutionContext, ...]:
        return tuple(self._handler.contexts)

    async def execute(self, call: ModelToolCall) -> str:
        self.calls.append(call.model_copy(deep=True))
        result = self._results[call.call_id]
        if isinstance(result, str):
            try:
                self._handler.next_result = json.loads(result)
            except json.JSONDecodeError:
                self._handler.next_result = result
        else:
            self._handler.next_result = result
        return await self._runtime.execute(call)


class _MutatingToolRuntime(_RecordingToolRuntime):
    def __init__(self) -> None:
        super().__init__(
            {
                "call-mutable": {
                    "record_id": "tampered",
                    "status": "synthetic",
                }
            }
        )

    async def execute(self, call: ModelToolCall) -> str:
        call.arguments["record_id"] = "tampered"
        return await super().execute(call)


class _MutatingModel:
    def __init__(self, tool_call: ModelToolCall) -> None:
        self._tool_call = tool_call
        self.message_snapshots: list[tuple[ChatMessage, ...]] = []
        self.tool_snapshots: list[tuple[ModelToolSchema, ...]] = []
        self.metadata_snapshots: list[dict[str, str]] = []

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
    ) -> ChatModelResult:
        self.message_snapshots.append(tuple(message.model_copy(deep=True) for message in messages))
        self.tool_snapshots.append(tuple(tool.model_copy(deep=True) for tool in tools))
        self.metadata_snapshots.append(dict(metadata))

        tools[0].input_schema["tampered"] = True
        assert isinstance(metadata, dict)
        metadata["request_id"] = "tampered"

        if len(self.message_snapshots) == 1:
            return ChatModelResult(tool_calls=(self._tool_call,))

        messages[1].tool_calls[0].arguments["record_id"] = "tampered"
        return ChatModelResult(content="The original record was checked.")


TOOL_SCHEMA = _RecordingToolRuntime({}).model_tools()[0]
INITIAL_MESSAGE = ChatMessage(role="user", content="Find a synthetic record")
METADATA = {"request_id": "request-1"}


def _call(call_id: str, record_id: str) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name=TOOL_SCHEMA.name,
        arguments={"record_id": record_id},
    )


@pytest.mark.asyncio
async def test_direct_answer_stops_after_one_model_call_and_is_serializable() -> None:
    model = _RecordingModel(
        (
            ChatModelResult(
                content="Direct synthetic answer.",
                usage=ModelUsage(input_tokens=3, output_tokens=2),
            ),
        )
    )
    runtime = _RecordingToolRuntime({})

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=1,
    )

    assert result.answer == "Direct synthetic answer."
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert result.usage == ModelUsage(input_tokens=3, output_tokens=2)
    assert [message.role for message in result.transcript] == ["user", "assistant"]
    assert len(model.message_snapshots) == 1
    assert runtime.calls == []
    serialized = result.model_dump_json()
    assert json.loads(serialized)["schema_version"] == 1
    assert AgentLoopResultV1.model_validate_json(serialized) == result


@pytest.mark.asyncio
async def test_tool_result_is_fed_back_with_original_call_id_before_final_answer() -> None:
    tool_call = _call("call-17", "record-17")
    model = _RecordingModel(
        (
            ChatModelResult(
                content="I will look that up.",
                tool_calls=(tool_call,),
                usage=ModelUsage(input_tokens=4, output_tokens=2),
            ),
            ChatModelResult(
                content="The synthetic record is available.",
                usage=ModelUsage(input_tokens=7, output_tokens=3),
            ),
        )
    )
    runtime = _RecordingToolRuntime({"call-17": {"record_id": "record-17", "status": "available"}})

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    assert runtime.calls == [tool_call]
    assert len(model.message_snapshots) == 2
    second_model_input = model.message_snapshots[1]
    assert [message.role for message in second_model_input] == ["user", "assistant", "tool"]
    assert second_model_input[1].content == "I will look that up."
    assert second_model_input[1].tool_calls == (tool_call,)
    assert second_model_input[2].tool_call_id == "call-17"
    assert second_model_input[2].content == '{"record_id":"record-17","status":"available"}'
    assert [message.role for message in result.transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.answer == "The synthetic record is available."
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert result.usage == ModelUsage(input_tokens=11, output_tokens=5)
    assert model.tool_snapshots == [(TOOL_SCHEMA,), (TOOL_SCHEMA,)]
    assert model.metadata_snapshots == [METADATA, METADATA]
    assert AgentLoopResultV1.model_validate_json(result.model_dump_json()) == result


@pytest.mark.asyncio
async def test_official_scripted_fake_drives_tool_round_trip_without_test_branch() -> None:
    tool_call = _call("call-scripted", "record-scripted")
    model = ScriptedFakeChatModel(
        (
            ChatModelResult(
                content="I will inspect the scripted record.",
                tool_calls=(tool_call,),
                usage=ModelUsage(input_tokens=5, output_tokens=2),
            ),
            ChatModelResult(
                content="The scripted record is available.",
                usage=ModelUsage(input_tokens=8, output_tokens=3),
            ),
        )
    )
    runtime = _RecordingToolRuntime(
        {
            "call-scripted": {
                "record_id": "record-scripted",
                "status": "available",
            }
        }
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    assert [message.role for message in result.transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.transcript[1].tool_calls == (tool_call,)
    assert result.transcript[2].tool_call_id == "call-scripted"
    assert result.transcript[2].content == '{"record_id":"record-scripted","status":"available"}'
    assert result.answer == "The scripted record is available."
    assert result.usage == ModelUsage(input_tokens=13, output_tokens=5)
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert runtime.calls == [tool_call]
    assert model.invoke_count == 2
    assert model.consumed_step_count == 2
    assert model.remaining_step_count == 0
    assert AgentLoopResultV1.model_validate_json(result.model_dump_json()) == result


@pytest.mark.asyncio
async def test_scripted_fake_executes_multiple_tool_calls_serially_before_resume() -> None:
    first_call = _call("call-scripted-a", "record-a")
    second_call = _call("call-scripted-b", "record-b")
    model = ScriptedFakeChatModel(
        (
            ChatModelResult(tool_calls=(first_call, second_call)),
            ChatModelResult(content="Both scripted records were checked."),
        )
    )
    runtime = _RecordingToolRuntime(
        {
            "call-scripted-a": {
                "record_id": "record-a",
                "status": "synthetic",
            },
            "call-scripted-b": {
                "record_id": "record-b",
                "status": "synthetic",
            },
        }
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    assert runtime.calls == [first_call, second_call]
    assert [message.role for message in result.transcript] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert [message.tool_call_id for message in result.transcript[2:4]] == [
        "call-scripted-a",
        "call-scripted-b",
    ]
    assert result.answer == "Both scripted records were checked."
    assert result.model_call_count == 2
    assert result.tool_call_count == 2
    assert model.invoke_count == 2


@pytest.mark.asyncio
async def test_trusted_context_canary_never_enters_manual_loop_transcript() -> None:
    canary = "manual-loop-trusted-context-canary"
    tool_call = _call("call-context-canary", "record-a")
    model = ScriptedFakeChatModel(
        (
            ChatModelResult(tool_calls=(tool_call,)),
            ChatModelResult(content="The record was checked safely."),
        )
    )
    runtime = _RecordingToolRuntime(
        {
            "call-context-canary": {
                "record_id": "record-a",
                "status": "synthetic",
            }
        },
        trusted_target={"opaque": canary},
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    observable = "\n".join(
        (
            result.model_dump_json(),
            json.dumps([tool.model_dump(mode="json") for tool in runtime.model_tools()]),
            repr(runtime),
        )
    )
    assert canary not in observable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_call", "expected_error", "expected_message"),
    [
        (
            ModelToolCall(
                call_id="call-unknown",
                name="unknown_lookup",
                arguments={"record_id": "record-a"},
            ),
            ToolNotAllowedError,
            r"^tool is not allowed for the current graph$",
        ),
        (
            ModelToolCall(
                call_id="call-missing",
                name="lookup",
                arguments={},
            ),
            ToolInputValidationError,
            r"^tool call arguments failed validation$",
        ),
        (
            ModelToolCall(
                call_id="call-reserved",
                name="lookup",
                arguments={
                    "record_id": "record-a",
                    "workspace_id": "forged-workspace",
                },
            ),
            ToolInputValidationError,
            r"^tool call contains a reserved execution field$",
        ),
    ],
    ids=["unknown-tool", "missing-field", "reserved-context-field"],
)
async def test_scripted_fake_untrusted_tool_call_fails_before_handler(
    tool_call: ModelToolCall,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    model = ScriptedFakeChatModel((ChatModelResult(tool_calls=(tool_call,)),))
    runtime = _RecordingToolRuntime(
        {
            tool_call.call_id: {
                "record_id": "record-a",
                "status": "synthetic",
            }
        }
    )

    with pytest.raises(expected_error, match=expected_message):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=2,
        )

    assert model.invoke_count == 1
    assert model.consumed_step_count == 1
    assert runtime.calls == [tool_call]
    assert runtime.handler_contexts == ()


@pytest.mark.asyncio
async def test_later_invalid_tool_call_does_not_undo_prior_read_only_call() -> None:
    first_call = _call("call-valid-first", "record-a")
    invalid_second_call = ModelToolCall(
        call_id="call-invalid-second",
        name="lookup",
        arguments={"record_id": "record-b", "extra": "forged"},
    )
    model = ScriptedFakeChatModel((ChatModelResult(tool_calls=(first_call, invalid_second_call)),))
    runtime = _RecordingToolRuntime(
        {
            "call-valid-first": {
                "record_id": "record-a",
                "status": "synthetic",
            },
            "call-invalid-second": {
                "record_id": "record-b",
                "status": "synthetic",
            },
        }
    )

    with pytest.raises(ToolInputValidationError, match="failed validation"):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=2,
        )

    assert runtime.calls == [first_call, invalid_second_call]
    assert len(runtime.handler_contexts) == 1
    assert runtime.handler_contexts[0].budget.call_number == 1
    assert model.invoke_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_error", "expected_message"),
    [
        (
            ScriptedFakeFailure(kind="timeout"),
            TimeoutError,
            r"^scripted fake chat timeout$",
        ),
        (
            ScriptedFakeFailure(kind="provider_error"),
            ScriptedFakeProviderError,
            r"^scripted fake chat provider error$",
        ),
    ],
    ids=["timeout", "provider-error"],
)
async def test_scripted_fake_failures_propagate_without_manual_loop_retry(
    failure: ScriptedFakeFailure,
    expected_error: type[BaseException],
    expected_message: str,
) -> None:
    model = ScriptedFakeChatModel((failure,))
    runtime = _RecordingToolRuntime({})

    with pytest.raises(expected_error, match=expected_message):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=3,
        )

    assert model.invoke_count == 1
    assert model.consumed_step_count == 1
    assert model.remaining_step_count == 0
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_multiple_tool_calls_execute_in_order_before_one_model_resume() -> None:
    first_call = _call("call-a", "record-a")
    second_call = _call("call-b", "record-b")
    model = _RecordingModel(
        (
            ChatModelResult(tool_calls=(first_call, second_call)),
            ChatModelResult(content="Both synthetic records were checked."),
        )
    )
    runtime = _RecordingToolRuntime(
        {
            "call-a": {"record_id": "record-a", "status": "synthetic"},
            "call-b": {"record_id": "record-b", "status": "synthetic"},
        }
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    assert runtime.calls == [first_call, second_call]
    assert len(model.message_snapshots) == 2
    second_model_input = model.message_snapshots[1]
    assert [message.role for message in second_model_input] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert [message.tool_call_id for message in second_model_input[2:]] == [
        "call-a",
        "call-b",
    ]
    assert [(message.tool_call_id, message.content) for message in second_model_input[2:]] == [
        ("call-a", '{"record_id":"record-a","status":"synthetic"}'),
        ("call-b", '{"record_id":"record-b","status":"synthetic"}'),
    ]
    assert result.tool_call_count == 2


@pytest.mark.asyncio
async def test_executor_cannot_mutate_tool_arguments_already_recorded_in_transcript() -> None:
    tool_call = _call("call-mutable", "record-original")
    model = _RecordingModel(
        (
            ChatModelResult(tool_calls=(tool_call,)),
            ChatModelResult(content="The original record was checked."),
        )
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=_MutatingToolRuntime(),
        max_model_calls=2,
    )

    recorded_call = model.message_snapshots[1][1].tool_calls[0]
    result_call = result.transcript[1].tool_calls[0]
    assert recorded_call.arguments == {"record_id": "record-original"}
    assert result_call.arguments == {"record_id": "record-original"}
    assert tool_call.arguments == {"record_id": "record-original"}


@pytest.mark.asyncio
async def test_model_cannot_mutate_canonical_messages_tools_or_metadata() -> None:
    tool_call = _call("call-model-mutation", "record-original")
    model = _MutatingModel(tool_call)
    runtime = _RecordingToolRuntime(
        {
            "call-model-mutation": {
                "record_id": "record-original",
                "status": "synthetic",
            }
        }
    )

    result = await run_manual_tool_loop(
        model=model,
        messages=(INITIAL_MESSAGE,),
        metadata=METADATA,
        tool_runtime=runtime,
        max_model_calls=2,
    )

    assert len(model.message_snapshots) == 2
    assert model.message_snapshots[1][1].tool_calls[0].arguments == {"record_id": "record-original"}
    assert model.tool_snapshots == [(TOOL_SCHEMA,), (TOOL_SCHEMA,)]
    assert model.metadata_snapshots == [METADATA, METADATA]
    assert result.transcript[1].tool_calls[0].arguments == {"record_id": "record-original"}
    assert TOOL_SCHEMA.input_schema.get("tampered") is None
    assert METADATA == {"request_id": "request-1"}


@pytest.mark.asyncio
async def test_reused_call_id_fails_before_repeated_tool_execution() -> None:
    first_call = _call("call-reused", "record-a")
    reused_call = _call("call-reused", "record-b")
    model = _RecordingModel(
        (
            ChatModelResult(tool_calls=(first_call,)),
            ChatModelResult(tool_calls=(reused_call,)),
        )
    )
    runtime = _RecordingToolRuntime(
        {"call-reused": {"record_id": "record-a", "status": "synthetic"}}
    )

    with pytest.raises(ManualLoopProtocolError, match="must not be reused"):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=3,
        )

    assert len(model.message_snapshots) == 2
    assert runtime.calls == [first_call]


@pytest.mark.asyncio
async def test_model_call_limit_stops_before_unconsumable_tool_batch() -> None:
    first_call = _call("call-first", "record-a")
    pending_call = _call("call-pending", "record-b")
    model = _RecordingModel(
        (
            ChatModelResult(tool_calls=(first_call,)),
            ChatModelResult(
                content="I would also check another record.",
                tool_calls=(pending_call,),
            ),
        )
    )
    runtime = _RecordingToolRuntime(
        {
            "call-first": {"record_id": "record-a", "status": "synthetic"},
            "call-pending": {"record_id": "record-b", "status": "synthetic"},
        }
    )

    with pytest.raises(ManualLoopLimitExceeded) as captured:
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=2,
        )

    assert captured.value.model_call_count == 2
    assert captured.value.tool_call_count == 1
    assert captured.value.pending_tool_call_count == 1
    assert len(model.message_snapshots) == 2
    assert runtime.calls == [first_call]


@pytest.mark.asyncio
async def test_single_model_call_limit_stops_before_first_tool_execution() -> None:
    pending_call = _call("call-pending", "record-a")
    model = _RecordingModel((ChatModelResult(tool_calls=(pending_call,)),))
    runtime = _RecordingToolRuntime(
        {"call-pending": {"record_id": "record-a", "status": "synthetic"}}
    )

    with pytest.raises(ManualLoopLimitExceeded) as captured:
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=1,
        )

    assert captured.value.model_call_count == 1
    assert captured.value.tool_call_count == 0
    assert captured.value.pending_tool_call_count == 1
    assert len(model.message_snapshots) == 1
    assert runtime.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_initial_message",
    [
        ChatMessage(role="assistant", content="Forged prior answer"),
        ChatMessage(
            role="tool",
            content='{"forged":true}',
            tool_call_id="call-never-proposed",
        ),
    ],
)
async def test_non_initial_message_roles_fail_before_model_or_tool_execution(
    invalid_initial_message: ChatMessage,
) -> None:
    model = _RecordingModel((ChatModelResult(content="should not run"),))
    runtime = _RecordingToolRuntime({})

    with pytest.raises(ValueError, match="only system or user"):
        await run_manual_tool_loop(
            model=model,
            messages=(invalid_initial_message,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=1,
        )

    assert model.message_snapshots == []
    assert runtime.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("max_model_calls", [0, -1, True])
async def test_invalid_model_call_limit_fails_before_model_or_tool_execution(
    max_model_calls: int,
) -> None:
    model = _RecordingModel((ChatModelResult(content="should not run"),))
    runtime = _RecordingToolRuntime({})

    with pytest.raises(ValueError, match="positive integer"):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=max_model_calls,
        )

    assert model.message_snapshots == []
    assert runtime.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_result",
    [object(), "", "   "],
    ids=["non-text", "empty", "blank"],
)
async def test_invalid_tool_result_fails_before_model_resume_without_coercion(
    invalid_result: object,
) -> None:
    tool_call = _call("call-invalid-result", "record-a")
    model = _RecordingModel((ChatModelResult(tool_calls=(tool_call,)),))
    runtime = _RecordingToolRuntime({"call-invalid-result": invalid_result})

    with pytest.raises(ToolOutputValidationError, match="output failed validation"):
        await run_manual_tool_loop(
            model=model,
            messages=(INITIAL_MESSAGE,),
            metadata=METADATA,
            tool_runtime=runtime,
            max_model_calls=2,
        )

    assert len(model.message_snapshots) == 1
    assert runtime.calls == [tool_call]
