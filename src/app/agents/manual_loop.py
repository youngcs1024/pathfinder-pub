from collections.abc import Mapping, Sequence

from app.agents.contracts import AgentLoopResultV1
from app.llm.ports import (
    ChatMessage,
    ChatModelPort,
    ChatModelResult,
    ModelToolSchema,
    ModelUsage,
)
from app.tools.contracts import ToolRuntime


class ManualLoopError(Exception):
    """Base error for the learning-only manual tool loop."""


class ManualLoopProtocolError(ManualLoopError):
    """A model or tool result cannot be represented by the loop contract."""


class ManualLoopLimitExceeded(ManualLoopError):
    def __init__(
        self,
        *,
        model_call_count: int,
        tool_call_count: int,
        pending_tool_call_count: int,
    ) -> None:
        self.model_call_count = model_call_count
        self.tool_call_count = tool_call_count
        self.pending_tool_call_count = pending_tool_call_count
        super().__init__("manual loop model-call limit reached before executing pending tool calls")


def _validate_inputs(
    *,
    messages: Sequence[ChatMessage],
    tool_runtime: ToolRuntime,
    metadata: Mapping[str, str],
    max_model_calls: int,
) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    if any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("messages must contain ChatMessage values")
    if any(message.role not in {"system", "user"} for message in messages):
        raise ValueError("initial messages must contain only system or user messages")
    if not isinstance(tool_runtime, ToolRuntime):
        raise ValueError("tool_runtime must implement ToolRuntime")
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("metadata keys and values must be strings")
    if (
        isinstance(max_model_calls, bool)
        or not isinstance(max_model_calls, int)
        or max_model_calls <= 0
    ):
        raise ValueError("max_model_calls must be a positive integer")


async def run_manual_tool_loop(
    *,
    model: ChatModelPort,
    messages: Sequence[ChatMessage],
    metadata: Mapping[str, str],
    tool_runtime: ToolRuntime,
    max_model_calls: int,
) -> AgentLoopResultV1:
    """Run the learning-only model -> tool -> model protocol with a hard stop."""

    _validate_inputs(
        messages=messages,
        tool_runtime=tool_runtime,
        metadata=metadata,
        max_model_calls=max_model_calls,
    )
    transcript = [message.model_copy(deep=True) for message in messages]
    runtime_tools = tool_runtime.model_tools()
    if not isinstance(runtime_tools, tuple) or any(
        not isinstance(tool, ModelToolSchema) for tool in runtime_tools
    ):
        raise ManualLoopProtocolError("tool runtime must return ModelToolSchema values")
    model_tools = tuple(tool.model_copy(deep=True) for tool in runtime_tools)
    model_metadata = dict(metadata)
    seen_tool_call_ids: set[str] = set()
    model_call_count = 0
    tool_call_count = 0
    input_tokens = 0
    output_tokens = 0

    while model_call_count < max_model_calls:
        response = await model.invoke(
            tuple(message.model_copy(deep=True) for message in transcript),
            tuple(tool.model_copy(deep=True) for tool in model_tools),
            dict(model_metadata),
        )
        if not isinstance(response, ChatModelResult):
            raise ManualLoopProtocolError("model must return ChatModelResult")

        model_call_count += 1
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        pending_tool_calls = tuple(call.model_copy(deep=True) for call in response.tool_calls)

        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            tool_calls=pending_tool_calls,
        )
        transcript.append(assistant_message)

        if not pending_tool_calls:
            if response.content is None:
                raise ManualLoopProtocolError("final model result must contain content")
            return AgentLoopResultV1(
                answer=response.content,
                transcript=tuple(transcript),
                model_call_count=model_call_count,
                tool_call_count=tool_call_count,
                usage=ModelUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        pending_call_ids = {call.call_id for call in pending_tool_calls}
        if seen_tool_call_ids.intersection(pending_call_ids):
            raise ManualLoopProtocolError("assistant tool call ids must not be reused")
        seen_tool_call_ids.update(pending_call_ids)

        if model_call_count == max_model_calls:
            raise ManualLoopLimitExceeded(
                model_call_count=model_call_count,
                tool_call_count=tool_call_count,
                pending_tool_call_count=len(pending_tool_calls),
            )

        for call in pending_tool_calls:
            result_content = await tool_runtime.execute(call.model_copy(deep=True))
            if not isinstance(result_content, str) or not result_content.strip():
                raise ManualLoopProtocolError(
                    "tool runtime must return non-blank model-visible text"
                )
            transcript.append(
                ChatMessage(
                    role="tool",
                    content=result_content,
                    tool_call_id=call.call_id,
                )
            )
            tool_call_count += 1

    raise AssertionError("manual loop exhausted without returning or raising a typed limit")
