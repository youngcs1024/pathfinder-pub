import json

import pytest
from pydantic import ValidationError

from app.llm.fake import (
    FAKE_EMBEDDING_DIMENSION,
    FakeChatModel,
    FakeEmbeddingModel,
    ScriptedFakeChatModel,
    ScriptedFakeExhaustedError,
    ScriptedFakeFailure,
    ScriptedFakeProviderError,
)
from app.llm.ports import (
    ChatMessage,
    ChatModelPort,
    ChatModelResult,
    EmbeddingPort,
    EmbeddingResult,
    ModelToolCall,
    ModelToolSchema,
    ModelUsage,
)


def test_chat_contract_models_are_frozen_strict_and_json_serializable() -> None:
    message = ChatMessage(role="user", content="Find backend roles")
    tool = ModelToolSchema(
        name="lookup",
        description="Look up a synthetic record",
        input_schema={
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
        },
    )
    result = ChatModelResult(
        tool_calls=(
            ModelToolCall(
                call_id="call-1",
                name=tool.name,
                arguments={"record_id": "record-1"},
            ),
        ),
        usage=ModelUsage(input_tokens=3, output_tokens=2),
    )
    assistant_message = ChatMessage(role="assistant", tool_calls=result.tool_calls)
    tool_message = ChatMessage(
        role="tool",
        content='{"record":"synthetic"}',
        tool_call_id="call-1",
    )

    payload = json.loads(result.model_dump_json())

    assert payload == {
        "content": None,
        "tool_calls": [
            {
                "call_id": "call-1",
                "name": "lookup",
                "arguments": {"record_id": "record-1"},
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
        "provider": "fake",
        "model": "qwen3.6-flash-2026-04-16",
        "provider_response_id": None,
        "finish_status": "completed",
    }
    assert json.loads(message.model_dump_json()) == {
        "role": "user",
        "content": "Find backend roles",
        "tool_calls": [],
        "tool_call_id": None,
    }
    assert json.loads(assistant_message.model_dump_json()) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "call_id": "call-1",
                "name": "lookup",
                "arguments": {"record_id": "record-1"},
            }
        ],
        "tool_call_id": None,
    }
    assert json.loads(tool_message.model_dump_json()) == {
        "role": "tool",
        "content": '{"record":"synthetic"}',
        "tool_calls": [],
        "tool_call_id": "call-1",
    }
    assert json.loads(tool.model_dump_json())["name"] == "lookup"

    with pytest.raises(ValidationError, match="frozen"):
        message.content = "changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ChatMessage.model_validate({"role": "user", "content": "hello", "workspace_id": "forged"})


def test_model_usage_preserves_available_details_and_rejects_inconsistent_totals() -> None:
    usage = ModelUsage(
        input_tokens=11,
        output_tokens=13,
        total_tokens=24,
        cached_input_tokens=2,
        cache_write_input_tokens=3,
        reasoning_output_tokens=5,
    )

    assert usage.model_dump() == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 24,
        "cached_input_tokens": 2,
        "cache_write_input_tokens": 3,
        "reasoning_output_tokens": 5,
    }
    assert ModelUsage(input_tokens=1, output_tokens=2).model_dump() == {
        "input_tokens": 1,
        "output_tokens": 2,
    }

    for invalid in (
        {"input_tokens": 2, "output_tokens": 3, "total_tokens": 6},
        {
            "input_tokens": 4,
            "output_tokens": 1,
            "cached_input_tokens": 3,
            "cache_write_input_tokens": 2,
        },
        {"input_tokens": 1, "output_tokens": 2, "reasoning_output_tokens": 3},
    ):
        with pytest.raises(ValidationError):
            ModelUsage.model_validate(invalid)


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "user", "content": " "},
        {"role": "user"},
        {
            "role": "user",
            "content": "hello",
            "tool_calls": [{"call_id": "call-1", "name": "lookup"}],
        },
        {"role": "assistant"},
        {"role": "assistant", "content": "hello", "tool_call_id": "call-1"},
        {"role": "tool", "content": "result"},
        {"role": "tool", "content": "result", "tool_call_id": " "},
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call-1",
            "tool_calls": [{"call_id": "call-2", "name": "lookup"}],
        },
        {"role": "unknown", "content": "hello"},
    ],
)
def test_chat_message_rejects_invalid_role_field_combinations(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(payload)


def test_assistant_message_allows_content_with_tool_calls() -> None:
    message = ChatMessage(
        role="assistant",
        content="I will look that up.",
        tool_calls=(
            ModelToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"record_id": "record-1"},
            ),
        ),
    )

    assert message.content == "I will look that up."
    assert message.tool_calls[0].call_id == "call-1"


@pytest.mark.parametrize("call_id", ["", " ", "\t\n"])
def test_model_tool_call_rejects_blank_call_id(call_id: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        ModelToolCall(call_id=call_id, name="lookup")


def test_chat_result_allows_empty_provider_result_and_requires_unique_tool_calls() -> None:
    assert ChatModelResult().content is None
    assert ChatModelResult().tool_calls == ()
    duplicate_call = ModelToolCall(call_id="call-1", name="lookup")
    with pytest.raises(ValidationError, match="tool call ids must be unique"):
        ChatModelResult(tool_calls=(duplicate_call, duplicate_call))
    with pytest.raises(ValidationError, match="tool call ids must be unique"):
        ChatMessage(role="assistant", tool_calls=(duplicate_call, duplicate_call))


@pytest.mark.asyncio
async def test_fake_chat_implements_port_and_returns_direct_deterministic_result() -> None:
    model = FakeChatModel()
    messages = (ChatMessage(role="user", content="A synthetic request"),)
    tools = (
        ModelToolSchema(
            name="teaching_tool",
            description="A model-visible schema only",
            input_schema={"type": "object"},
        ),
    )

    first = await model.invoke(messages, tools, {"request_id": "request-1"})
    second = await model.invoke(messages, tools, {"request_id": "request-1"})

    assert isinstance(model, ChatModelPort)
    assert first == second
    assert first is not second
    assert first.content == "Offline fake response."
    assert first.tool_calls == ()
    assert first.usage == ModelUsage(input_tokens=0, output_tokens=0)


@pytest.mark.asyncio
async def test_fake_chat_rejects_empty_messages_and_invalid_metadata() -> None:
    model = FakeChatModel()

    with pytest.raises(ValueError, match="messages must not be empty"):
        await model.invoke((), (), {})
    with pytest.raises(ValueError, match="metadata keys and values must be strings"):
        await model.invoke(
            (ChatMessage(role="user", content="hello"),),
            (),
            {"request_id": 123},  # type: ignore[dict-item]
        )


@pytest.mark.asyncio
async def test_scripted_fake_implements_port_and_consumes_results_in_order() -> None:
    first_step = ChatModelResult(
        content="First scripted response.",
        usage=ModelUsage(input_tokens=2, output_tokens=1),
    )
    second_step = ChatModelResult(
        content="Second scripted response.",
        usage=ModelUsage(input_tokens=4, output_tokens=3),
    )
    model = ScriptedFakeChatModel((first_step, second_step))
    messages = (ChatMessage(role="user", content="A synthetic request"),)

    first = await model.invoke(messages, (), {"request_id": "request-1"})
    second = await model.invoke(messages, (), {"request_id": "request-1"})

    assert isinstance(model, ChatModelPort)
    assert first == first_step
    assert second == second_step
    assert first is not first_step
    assert second is not second_step
    assert model.invoke_count == 2
    assert model.consumed_step_count == 2
    assert model.remaining_step_count == 0


@pytest.mark.asyncio
async def test_scripted_fake_deep_copies_constructor_and_returned_results() -> None:
    tool_call = ModelToolCall(
        call_id="call-copy",
        name="lookup",
        arguments={"record_id": "record-original"},
    )
    scripted_result = ChatModelResult(tool_calls=(tool_call,))
    model = ScriptedFakeChatModel((scripted_result, scripted_result))
    tool_call.arguments["record_id"] = "changed-after-construction"

    first = await model.invoke(
        (ChatMessage(role="user", content="A synthetic request"),),
        (),
        {},
    )
    first.tool_calls[0].arguments["record_id"] = "changed-after-return"
    second = await model.invoke(
        (ChatMessage(role="user", content="A synthetic request"),),
        (),
        {},
    )

    assert second.tool_calls[0].arguments == {"record_id": "record-original"}
    assert scripted_result.tool_calls[0].arguments == {"record_id": "changed-after-construction"}


def test_scripted_fake_rejects_empty_or_invalid_script() -> None:
    with pytest.raises(ValueError, match="script must not be empty"):
        ScriptedFakeChatModel(())
    with pytest.raises(ValueError, match="must contain ChatModelResult"):
        ScriptedFakeChatModel((object(),))  # type: ignore[arg-type]
    canary = "scripted-failure-secret-canary"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted") as captured:
        ScriptedFakeFailure.model_validate({"kind": "timeout", "message": canary})
    assert canary not in str(captured.value)


@pytest.mark.asyncio
async def test_invalid_invocation_does_not_consume_script_or_increment_count() -> None:
    model = ScriptedFakeChatModel((ChatModelResult(content="Still available."),))

    with pytest.raises(ValueError, match="messages must not be empty"):
        await model.invoke((), (), {})
    with pytest.raises(ValueError, match="metadata keys and values must be strings"):
        await model.invoke(
            (ChatMessage(role="user", content="A synthetic request"),),
            (),
            {"request_id": object()},  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="tools must contain ModelToolSchema values"):
        await model.invoke(
            (ChatMessage(role="user", content="A synthetic request"),),
            (object(),),  # type: ignore[arg-type]
            {},
        )

    assert model.invoke_count == 0
    assert model.consumed_step_count == 0
    assert model.remaining_step_count == 1
    assert await model.invoke(
        (ChatMessage(role="user", content="A synthetic request"),),
        (),
        {},
    ) == ChatModelResult(content="Still available.")


@pytest.mark.asyncio
async def test_scripted_failures_are_consumed_before_the_next_result() -> None:
    model = ScriptedFakeChatModel(
        (
            ScriptedFakeFailure(kind="timeout"),
            ScriptedFakeFailure(kind="provider_error"),
            ChatModelResult(content="Recovered scripted response."),
        )
    )
    messages = (ChatMessage(role="user", content="A synthetic request"),)

    with pytest.raises(TimeoutError, match=r"^scripted fake chat timeout$"):
        await model.invoke(messages, (), {})
    assert (model.invoke_count, model.consumed_step_count, model.remaining_step_count) == (
        1,
        1,
        2,
    )

    with pytest.raises(
        ScriptedFakeProviderError,
        match=r"^scripted fake chat provider error$",
    ):
        await model.invoke(messages, (), {})
    assert (model.invoke_count, model.consumed_step_count, model.remaining_step_count) == (
        2,
        2,
        1,
    )

    assert await model.invoke(messages, (), {}) == ChatModelResult(
        content="Recovered scripted response."
    )
    assert (model.invoke_count, model.consumed_step_count, model.remaining_step_count) == (
        3,
        3,
        0,
    )


@pytest.mark.asyncio
async def test_script_exhaustion_is_explicit_and_does_not_repeat_last_result() -> None:
    model = ScriptedFakeChatModel((ChatModelResult(content="Only scripted response."),))
    messages = (ChatMessage(role="user", content="A synthetic request"),)

    assert await model.invoke(messages, (), {}) == ChatModelResult(
        content="Only scripted response."
    )
    with pytest.raises(
        ScriptedFakeExhaustedError,
        match=r"^scripted fake chat script exhausted$",
    ):
        await model.invoke(messages, (), {})

    assert model.invoke_count == 2
    assert model.consumed_step_count == 1
    assert model.remaining_step_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_call",
    [
        ModelToolCall(
            call_id="call-unknown",
            name="unknown_tool",
            arguments={"record_id": "record-1"},
        ),
        ModelToolCall(call_id="call-missing", name="lookup", arguments={}),
        ModelToolCall(
            call_id="call-reserved",
            name="lookup",
            arguments={"workspace_id": "forged", "actor_user_id": "forged"},
        ),
    ],
    ids=["unknown-tool", "schema-invalid", "reserved-context-fields"],
)
async def test_scripted_fake_can_emit_semantically_invalid_tool_calls(
    tool_call: ModelToolCall,
) -> None:
    model = ScriptedFakeChatModel((ChatModelResult(tool_calls=(tool_call,)),))
    declared_tool = ModelToolSchema(
        name="lookup",
        description="Look up a synthetic record",
        input_schema={
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
    )

    result = await model.invoke(
        (ChatMessage(role="user", content="A synthetic request"),),
        (declared_tool,),
        {},
    )

    assert result.tool_calls == (tool_call,)


def test_embedding_result_rejects_empty_or_inconsistent_vectors() -> None:
    with pytest.raises(ValidationError, match="at least one vector"):
        EmbeddingResult(vectors=())
    with pytest.raises(ValidationError, match="exactly 1536 dimensions"):
        EmbeddingResult(vectors=((0.1,), (0.1, 0.2)))
    with pytest.raises(ValidationError, match="exactly 1536 dimensions"):
        EmbeddingResult(vectors=((),))
    with pytest.raises(ValidationError, match="finite number"):
        EmbeddingResult(vectors=((float("nan"),),))
    with pytest.raises(ValidationError, match="text-embedding-v4"):
        EmbeddingResult.model_validate({"vectors": [[0.1] * 1536], "model": "forged"})


@pytest.mark.asyncio
async def test_fake_embedding_implements_port_and_is_stable() -> None:
    model = FakeEmbeddingModel()

    first = await model.embed(("alpha", "beta"), {"request_id": "request-1"})
    second = await model.embed(("alpha", "beta"), {"request_id": "request-2"})

    assert isinstance(model, EmbeddingPort)
    assert first == second
    assert len(first.vectors) == 2
    assert {len(vector) for vector in first.vectors} == {FAKE_EMBEDDING_DIMENSION}
    assert first.vectors[0] != first.vectors[1]
    assert all(0.0 <= value <= 1.0 for vector in first.vectors for value in vector)
    assert json.loads(first.model_dump_json())["vectors"]
    assert first.provider == "fake"
    assert first.model == "text-embedding-v4"
    assert first.provider_response_id is None
    assert first.usage == ModelUsage(input_tokens=0, output_tokens=0)


@pytest.mark.asyncio
async def test_fake_embedding_rejects_empty_or_blank_input() -> None:
    model = FakeEmbeddingModel()

    with pytest.raises(ValueError, match="texts must not be empty"):
        await model.embed((), {})
    with pytest.raises(ValueError, match="non-blank strings"):
        await model.embed(("valid", " "), {})
