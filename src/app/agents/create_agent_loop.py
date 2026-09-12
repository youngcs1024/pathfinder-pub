from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import Context
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, cast

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.constants import CONFIG_KEY_CHECKPOINTER
from langgraph.errors import GraphRecursionError
from langsmith import tracing_context
from pydantic import ConfigDict, PrivateAttr, ValidationError

from app.agents.contracts import (
    AgentLoopControl,
    AgentLoopErrorCategory,
    AgentLoopEvent,
    AgentLoopLimitKind,
    AgentLoopObservationV1,
    AgentLoopObserver,
    AgentLoopResultV1,
)
from app.agents.prompting import (
    AgentPromptBundleError,
    AgentPromptProfile,
    load_agent_prompt_bundle,
)
from app.domain.tracing import bind_trace_scope, current_trace_scope
from app.llm.ports import (
    ChatMessage,
    ChatModelPort,
    ChatModelResult,
    ModelToolCall,
    ModelToolSchema,
    ModelUsage,
)
from app.tools.contracts import ToolRuntime
from app.tools.registry import (
    ToolInputValidationError,
    ToolNotAllowedError,
)


class CreateAgentLoopError(Exception):
    """Base error for the production create_agent bridge."""


class CreateAgentConfigurationError(CreateAgentLoopError):
    """Trusted framework or Registry configuration is inconsistent."""


class CreateAgentProtocolError(CreateAgentLoopError):
    """A framework/model value cannot be represented by Pathfinder contracts."""


type ToolProposalErrorCategory = Literal[
    "duplicate_tool_call_id",
    "tool_not_allowed",
    "invalid_tool_arguments",
]


class CreateAgentToolProposalError(CreateAgentLoopError):
    """A model-proposed Tool batch failed deterministic application policy."""

    def __init__(self, category: ToolProposalErrorCategory) -> None:
        self.category = category
        super().__init__("model tool proposal failed application validation")


class AgentLoopLimitExceeded(CreateAgentLoopError):
    """A caller-supplied Agent loop limit would be exceeded."""

    def __init__(
        self,
        *,
        limit_kind: AgentLoopLimitKind,
        limit: int,
        current_count: int,
        requested_count: int,
    ) -> None:
        self.limit_kind = limit_kind
        self.limit = limit
        self.current_count = current_count
        self.requested_count = requested_count
        super().__init__("agent loop limit exceeded")


class AgentLoopDeadlineExceeded(CreateAgentLoopError):
    """The trusted absolute Agent deadline has expired."""

    def __init__(self) -> None:
        super().__init__("agent loop deadline exceeded")


class AgentLoopCancelled(CreateAgentLoopError):
    """The trusted cancellation check requested termination."""

    def __init__(self) -> None:
        super().__init__("agent loop was cancelled")


_CANCELLATION_POLL_SECONDS = 0.01
_UNRECOGNIZED_TOOL_OBSERVATION_NAME = "unrecognized"
_RESEARCH_PASS_COMPLETION = "Bounded research pass complete with the evidence collected so far."


@dataclass(slots=True)
class _ModelInvocationTracker:
    control: AgentLoopControl
    observer: AgentLoopObserver
    prompt_version: str
    run_started_at: float
    call_count: int = 0
    tool_call_count: int = 0
    tool_result_count: int = 0
    iteration_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seen_tool_call_ids: set[str] = field(default_factory=set)
    current_batch_call_ids: tuple[str, ...] = ()
    current_batch_iteration: int = 0
    sequence: int = 0
    last_failure_category: AgentLoopErrorCategory | None = None

    def now(self) -> float:
        try:
            value = self.control.clock()
        except Exception:
            raise _configuration_error("Agent loop clock failed") from None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
        ):
            raise _configuration_error("Agent loop clock returned an invalid value")
        return float(value)

    def elapsed_ms(self, started_at: float) -> float:
        return max(0.0, (self.now() - started_at) * 1000.0)

    def _is_cancelled(self) -> bool:
        try:
            value = self.control.cancellation.is_cancelled()
        except Exception:
            raise _configuration_error("Agent loop cancellation check failed") from None
        if type(value) is not bool:
            raise _configuration_error("Agent loop cancellation check returned an invalid value")
        return value

    def check_boundary(self) -> None:
        if self._is_cancelled():
            raise AgentLoopCancelled()
        if self.now() >= self.control.deadline:
            raise AgentLoopDeadlineExceeded()

    def _raise_limit(
        self,
        *,
        kind: AgentLoopLimitKind,
        limit: int,
        current: int,
        requested: int,
    ) -> None:
        raise AgentLoopLimitExceeded(
            limit_kind=kind,
            limit=limit,
            current_count=current,
            requested_count=requested,
        )

    def begin_model(self) -> tuple[int, float]:
        self.check_boundary()
        limits = self.control.limits
        if self.call_count >= limits.max_model_calls:
            self._raise_limit(
                kind="model_calls",
                limit=limits.max_model_calls,
                current=self.call_count,
                requested=1,
            )
        if self.iteration_count >= limits.max_iterations:
            self._raise_limit(
                kind="iterations",
                limit=limits.max_iterations,
                current=self.iteration_count,
                requested=1,
            )
        if self.current_batch_call_ids:
            raise _protocol_error("create_agent started a model before its tool batch completed")
        self.call_count += 1
        self.iteration_count += 1
        return self.iteration_count, self.now()

    def preflight_tool_batch(self, calls: Sequence[ModelToolCall]) -> None:
        self.check_boundary()
        if not calls:
            return
        if self.current_batch_call_ids:
            raise _protocol_error("create_agent proposed overlapping tool batches")

        requested = len(calls)
        limits = self.control.limits
        if self.call_count >= limits.max_model_calls:
            self._raise_limit(
                kind="model_calls",
                limit=limits.max_model_calls,
                current=self.call_count,
                requested=1,
            )
        if self.iteration_count + 2 > limits.max_iterations:
            self._raise_limit(
                kind="iterations",
                limit=limits.max_iterations,
                current=self.iteration_count,
                requested=2,
            )
        if self.tool_call_count + requested > limits.max_tool_calls:
            self._raise_limit(
                kind="tool_calls",
                limit=limits.max_tool_calls,
                current=self.tool_call_count,
                requested=requested,
            )
        if self.tool_result_count + requested > limits.max_tool_results:
            self._raise_limit(
                kind="tool_results",
                limit=limits.max_tool_results,
                current=self.tool_result_count,
                requested=requested,
            )

        self.iteration_count += 1
        self.current_batch_iteration = self.iteration_count
        self.current_batch_call_ids = tuple(call.call_id for call in calls)

    def batch_ordinal(self, call_id: str) -> int:
        try:
            return self.current_batch_call_ids.index(call_id) + 1
        except ValueError:
            raise _protocol_error("LangChain tool call was not in the proposed batch") from None

    def begin_tool(self, call_id: str) -> tuple[int, int, float]:
        self.check_boundary()
        ordinal = self.batch_ordinal(call_id)
        self.tool_call_count += 1
        return self.current_batch_iteration, ordinal, self.now()

    def complete_tool(self, *, ordinal: int) -> None:
        self.tool_result_count += 1
        if ordinal == len(self.current_batch_call_ids):
            self.current_batch_call_ids = ()
            self.current_batch_iteration = 0

    def record_usage(self, usage: ModelUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    async def await_controlled(self, awaitable: Awaitable[Any]) -> Any:
        self.check_boundary()
        remaining = self.control.deadline - self.now()
        if remaining <= 0:
            raise AgentLoopDeadlineExceeded()

        task = asyncio.ensure_future(awaitable)
        timeout = asyncio.timeout(remaining)
        try:
            try:
                async with timeout:
                    while not task.done():
                        await asyncio.wait(
                            (task,),
                            timeout=min(_CANCELLATION_POLL_SECONDS, remaining),
                        )
                        if task.done():
                            break
                        if self._is_cancelled():
                            raise AgentLoopCancelled()
                        remaining = self.control.deadline - self.now()
                        if remaining <= 0:
                            raise AgentLoopDeadlineExceeded()
                    result = await task
            except TimeoutError:
                if timeout.expired():
                    raise AgentLoopDeadlineExceeded() from None
                raise
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        self.check_boundary()
        return result

    def emit(
        self,
        *,
        event: AgentLoopEvent,
        iteration: int,
        duration_ms: float,
        usage: ModelUsage | None = None,
        output_bytes: int | None = None,
        proposed_tool_call_count: int | None = None,
        tool_name: str | None = None,
        batch_ordinal: int | None = None,
        result_bytes: int | None = None,
        error_category: AgentLoopErrorCategory | None = None,
    ) -> None:
        self.sequence += 1
        observation = AgentLoopObservationV1(
            event=event,
            sequence=self.sequence,
            iteration=iteration,
            prompt_version=self.prompt_version,
            duration_ms=duration_ms,
            model_call_count=self.call_count,
            tool_call_count=self.tool_call_count,
            tool_result_count=self.tool_result_count,
            usage=usage,
            output_bytes=output_bytes,
            proposed_tool_call_count=proposed_tool_call_count,
            tool_name=tool_name,
            batch_ordinal=batch_ordinal,
            result_bytes=result_bytes,
            error_category=error_category,
        )
        if error_category is not None:
            self.last_failure_category = error_category
        try:
            self.observer.observe(observation)
        except Exception:
            return


def _protocol_error(message: str) -> CreateAgentProtocolError:
    return CreateAgentProtocolError(message)


def _configuration_error(message: str) -> CreateAgentConfigurationError:
    return CreateAgentConfigurationError(message)


def _error_category(
    error: BaseException,
    *,
    stage: str,
) -> AgentLoopErrorCategory:
    if isinstance(error, AgentLoopLimitExceeded):
        return "recursion_limit" if error.limit_kind == "recursion" else "agent_limit_exceeded"
    if isinstance(error, AgentLoopDeadlineExceeded):
        return "agent_deadline_exceeded"
    if isinstance(error, AgentLoopCancelled):
        return "agent_cancelled"
    if isinstance(error, asyncio.CancelledError):
        return "external_cancelled"
    if isinstance(error, CreateAgentConfigurationError | AgentPromptBundleError):
        return "configuration_error"
    if isinstance(error, CreateAgentProtocolError):
        return "protocol_error"
    if isinstance(error, CreateAgentToolProposalError):
        return "invalid_tool_proposal"
    if stage == "model":
        return "provider_timeout" if isinstance(error, TimeoutError) else "provider_error"
    return "tool_error" if stage == "tool" else "protocol_error"


def _copy_tool_schema(schema: ModelToolSchema) -> ModelToolSchema:
    return schema.model_copy(deep=True)


def _copy_tool_call(call: ModelToolCall) -> ModelToolCall:
    return call.model_copy(deep=True)


def _pathfinder_message_to_langchain(message: ChatMessage) -> BaseMessage:
    if message.role == "system":
        if message.content is None:
            raise _protocol_error("system message is missing text content")
        return SystemMessage(content=message.content)
    if message.role == "user":
        if message.content is None:
            raise _protocol_error("user message is missing text content")
        return HumanMessage(content=message.content)
    if message.role == "assistant":
        tool_calls = [
            {
                "name": call.name,
                "args": deepcopy(call.arguments),
                "id": call.call_id,
                "type": "tool_call",
            }
            for call in message.tool_calls
        ]
        return AIMessage(content=message.content or "", tool_calls=tool_calls)
    if message.role == "tool":
        if message.content is None or message.tool_call_id is None:
            raise _protocol_error("tool message is missing required fields")
        return ToolMessage(content=message.content, tool_call_id=message.tool_call_id)
    raise _protocol_error("chat message uses an unsupported role")


def _string_content(message: BaseMessage) -> str:
    if type(message.content) is not str:
        raise _protocol_error("LangChain message content must be plain text")
    return message.content


def _langchain_tool_call_to_pathfinder(value: object) -> ModelToolCall:
    if not isinstance(value, Mapping):
        raise _protocol_error("LangChain tool call uses an unsupported representation")
    call_id = value.get("id")
    name = value.get("name")
    arguments = value.get("args")
    if (
        not isinstance(call_id, str)
        or not call_id.strip()
        or not isinstance(name, str)
        or not name.strip()
        or not isinstance(arguments, dict)
        or any(not isinstance(key, str) for key in arguments)
    ):
        raise _protocol_error("LangChain tool call is missing required JSON fields")
    try:
        return ModelToolCall(call_id=call_id, name=name, arguments=deepcopy(arguments))
    except (TypeError, ValueError, ValidationError):
        raise _protocol_error("LangChain tool call failed contract validation") from None


def _langchain_message_to_pathfinder(message: BaseMessage) -> ChatMessage:
    if type(message) is SystemMessage:
        content = _string_content(message)
        try:
            return ChatMessage(role="system", content=content)
        except ValidationError:
            raise _protocol_error("LangChain system message failed contract validation") from None

    if type(message) is HumanMessage:
        content = _string_content(message)
        try:
            return ChatMessage(role="user", content=content)
        except ValidationError:
            raise _protocol_error("LangChain user message failed contract validation") from None

    if type(message) is AIMessage:
        content = _string_content(message)
        if message.invalid_tool_calls:
            raise _protocol_error("LangChain assistant message contains invalid tool calls")
        tool_calls = tuple(_langchain_tool_call_to_pathfinder(call) for call in message.tool_calls)
        normalized_content = content if content.strip() else None
        try:
            return ChatMessage(
                role="assistant",
                content=normalized_content,
                tool_calls=tool_calls,
            )
        except ValidationError:
            raise _protocol_error(
                "LangChain assistant message failed contract validation"
            ) from None

    if type(message) is ToolMessage:
        content = _string_content(message)
        if message.status != "success" or message.artifact is not None:
            raise _protocol_error("LangChain tool message contains unsupported result metadata")
        if not isinstance(message.tool_call_id, str) or not message.tool_call_id.strip():
            raise _protocol_error("LangChain tool message is missing its call id")
        try:
            return ChatMessage(
                role="tool",
                content=content,
                tool_call_id=message.tool_call_id,
            )
        except ValidationError:
            raise _protocol_error("LangChain tool message failed contract validation") from None

    raise _protocol_error("LangChain returned an unsupported message type")


def _result_to_ai_message(result: ChatModelResult) -> AIMessage:
    return AIMessage(
        content=result.content or "",
        tool_calls=[
            {
                "name": call.name,
                "args": deepcopy(call.arguments),
                "id": call.call_id,
                "type": "tool_call",
            }
            for call in result.tool_calls
        ],
        usage_metadata={
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
        },
    )


class _ChatModelPortAdapter(BaseChatModel):
    """Async-only LangChain model facade over Pathfinder's governed chat port."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _port: ChatModelPort = PrivateAttr()
    _metadata: dict[str, str] = PrivateAttr()
    _schemas: tuple[ModelToolSchema, ...] = PrivateAttr()
    _static_tools: tuple[StructuredTool, ...] = PrivateAttr()
    _tool_runtime: ToolRuntime = PrivateAttr()
    _tracker: _ModelInvocationTracker = PrivateAttr()
    _bound: bool = PrivateAttr()
    _research_budget_feedback_enabled: bool = PrivateAttr()

    def __init__(
        self,
        *,
        port: ChatModelPort,
        metadata: Mapping[str, str],
        schemas: Sequence[ModelToolSchema],
        static_tools: Sequence[StructuredTool],
        tool_runtime: ToolRuntime,
        tracker: _ModelInvocationTracker,
        research_budget_feedback_enabled: bool = False,
        bound: bool = False,
    ) -> None:
        super().__init__()
        self._port = port
        self._metadata = dict(metadata)
        self._schemas = tuple(_copy_tool_schema(schema) for schema in schemas)
        self._static_tools = tuple(static_tools)
        self._tool_runtime = tool_runtime
        self._tracker = tracker
        self._research_budget_feedback_enabled = research_budget_feedback_enabled
        self._bound = bound

    @property
    def _llm_type(self) -> str:
        return "pathfinder-chat-model-port"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"adapter": "pathfinder-chat-model-port"}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ChatModelPortAdapter:
        if tool_choice is not None or kwargs:
            raise _configuration_error("unsupported LangChain model binding options")
        if len(tools) != len(self._static_tools):
            raise _configuration_error("LangChain tool collection differs from Registry")

        seen_names: set[str] = set()
        expected_by_name = {
            schema.name: (schema, tool)
            for schema, tool in zip(self._schemas, self._static_tools, strict=True)
        }
        for candidate in tools:
            if not isinstance(candidate, BaseTool):
                raise _configuration_error("LangChain attempted to bind a dynamic tool")
            if candidate.name in seen_names:
                raise _configuration_error("LangChain attempted to bind duplicate tools")
            seen_names.add(candidate.name)
            expected = expected_by_name.get(candidate.name)
            if expected is None:
                raise _configuration_error("LangChain tool collection differs from Registry")
            schema, static_tool = expected
            if candidate is not static_tool:
                raise _configuration_error("LangChain attempted to replace a Registry tool")
            if (
                candidate.description != schema.description
                or not isinstance(candidate.args_schema, dict)
                or candidate.args_schema != schema.input_schema
            ):
                raise _configuration_error("LangChain tool schema differs from Registry")

        return _ChatModelPortAdapter(
            port=self._port,
            metadata=self._metadata,
            schemas=self._schemas,
            static_tools=self._static_tools,
            tool_runtime=self._tool_runtime,
            tracker=self._tracker,
            research_budget_feedback_enabled=self._research_budget_feedback_enabled,
            bound=True,
        )

    def _model_messages(
        self,
        messages: Sequence[BaseMessage],
    ) -> tuple[ChatMessage, ...]:
        pathfinder_messages = tuple(
            _langchain_message_to_pathfinder(message) for message in messages
        )
        if not self._research_budget_feedback_enabled:
            return pathfinder_messages

        system_indexes = tuple(
            index for index, message in enumerate(pathfinder_messages) if message.role == "system"
        )
        if len(system_indexes) != 1:
            raise _protocol_error("research model invocation requires one system message")

        limits = self._tracker.control.limits
        remaining_tool_calls = limits.max_tool_calls - self._tracker.tool_call_count
        remaining_tool_results = limits.max_tool_results - self._tracker.tool_result_count
        if remaining_tool_calls < 0 or remaining_tool_results < 0:
            raise _protocol_error("research tool accounting exceeded its hard limit")
        runtime_budget_notice = (
            "<trusted_runtime_budget>\n"
            f"Remaining tool calls: {remaining_tool_calls}.\n"
            f"Remaining tool results: {remaining_tool_results}.\n\n"
            "Allowed tool names:\n"
            + "".join(f"- {name}\n" for name in sorted(schema.name for schema in self._schemas))
            + "\nUse only these exact tool names. Do not invent aliases or additional tools.\n\n"
            "Any tool-call batch in this response must contain no more than the remaining "
            "tool-call and tool-result budgets. If the available evidence is already sufficient, "
            "stop calling tools and return the short completion note. Do not call tools merely "
            "to exhaust the budget. If either remaining budget is zero, return the completion "
            "note with no tool calls.\n"
            "</trusted_runtime_budget>"
        )
        copied_messages = [message.model_copy(deep=True) for message in pathfinder_messages]
        system_index = system_indexes[0]
        system_message = copied_messages[system_index]
        if system_message.content is None:
            raise _protocol_error("research system message is missing text content")
        copied_messages[system_index] = system_message.model_copy(
            update={"content": f"{system_message.content}\n\n{runtime_budget_notice}"},
            deep=True,
        )
        return tuple(copied_messages)

    def _accepted_response(self, response: ChatModelResult) -> ChatModelResult:
        if (
            self._research_budget_feedback_enabled
            and not response.tool_calls
            and response.content is None
        ):
            return self._research_pass_completion(response)
        call_ids = tuple(call.call_id for call in response.tool_calls)
        unique_call_ids = set(call_ids)
        if len(call_ids) != len(unique_call_ids):
            if self._research_budget_feedback_enabled:
                return self._research_pass_completion(response)
            raise CreateAgentToolProposalError("duplicate_tool_call_id")
        if self._tracker.seen_tool_call_ids.intersection(unique_call_ids):
            if self._research_budget_feedback_enabled:
                return self._research_pass_completion(response)
            raise CreateAgentToolProposalError("duplicate_tool_call_id")

        try:
            for call in response.tool_calls:
                self._tool_runtime.validate_call(_copy_tool_call(call))
        except ToolNotAllowedError:
            if not self._research_budget_feedback_enabled:
                raise CreateAgentToolProposalError("tool_not_allowed") from None
            return self._research_pass_completion(response)
        except ToolInputValidationError:
            if not self._research_budget_feedback_enabled:
                raise CreateAgentToolProposalError("invalid_tool_arguments") from None
            return self._research_pass_completion(response)

        try:
            self._tracker.preflight_tool_batch(response.tool_calls)
        except AgentLoopLimitExceeded as error:
            if not (
                self._research_budget_feedback_enabled
                and error.limit_kind in {"tool_calls", "tool_results"}
            ):
                raise
            return self._research_pass_completion(response)

        self._tracker.seen_tool_call_ids.update(unique_call_ids)
        return response

    @staticmethod
    def _research_pass_completion(response: ChatModelResult) -> ChatModelResult:
        return response.model_copy(
            update={"content": _RESEARCH_PASS_COMPLETION, "tool_calls": ()},
            deep=True,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise _protocol_error("synchronous LangChain model invocation is not supported")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self._bound:
            raise _configuration_error("LangChain model invocation was not Registry-bound")
        if stop is not None or kwargs:
            raise _protocol_error("unsupported LangChain model invocation options")

        iteration, started_at = self._tracker.begin_model()
        safe_response: ChatModelResult | None = None
        try:
            pathfinder_messages = self._model_messages(messages)
            invocation_metadata = {
                **self._metadata,
                "prompt_version": self._tracker.prompt_version,
                "agent_iteration": str(iteration),
            }
            response = await self._tracker.await_controlled(
                self._port.invoke(
                    tuple(message.model_copy(deep=True) for message in pathfinder_messages),
                    tuple(_copy_tool_schema(schema) for schema in self._schemas),
                    dict(invocation_metadata),
                )
            )
            if not isinstance(response, ChatModelResult):
                raise _protocol_error("chat model port returned the wrong result contract")
            safe_response = response.model_copy(deep=True)
            self._tracker.record_usage(safe_response.usage)
        except asyncio.CancelledError as error:
            self._tracker.emit(
                event="agent.model.failed",
                iteration=iteration,
                duration_ms=self._tracker.elapsed_ms(started_at),
                usage=safe_response.usage if safe_response is not None else ModelUsage(),
                output_bytes=(
                    len(safe_response.model_dump_json().encode("utf-8"))
                    if safe_response is not None
                    else 0
                ),
                proposed_tool_call_count=(
                    len(safe_response.tool_calls) if safe_response is not None else 0
                ),
                error_category=_error_category(error, stage="model"),
            )
            raise
        except Exception as error:
            self._tracker.emit(
                event="agent.model.failed",
                iteration=iteration,
                duration_ms=self._tracker.elapsed_ms(started_at),
                usage=safe_response.usage if safe_response is not None else ModelUsage(),
                output_bytes=(
                    len(safe_response.model_dump_json().encode("utf-8"))
                    if safe_response is not None
                    else 0
                ),
                proposed_tool_call_count=(
                    len(safe_response.tool_calls) if safe_response is not None else 0
                ),
                error_category=_error_category(error, stage="model"),
            )
            raise

        output_bytes = len(safe_response.model_dump_json().encode("utf-8"))
        self._tracker.emit(
            event="agent.model.completed",
            iteration=iteration,
            duration_ms=self._tracker.elapsed_ms(started_at),
            usage=safe_response.usage,
            output_bytes=output_bytes,
            proposed_tool_call_count=len(safe_response.tool_calls),
        )
        safe_response = self._accepted_response(safe_response)
        return ChatResult(
            generations=[ChatGeneration(message=_result_to_ai_message(safe_response))]
        )


async def _registry_bypass_stub(**_arguments: object) -> str:
    raise _configuration_error("LangChain tool execution bypassed the Registry middleware")


def _build_static_tools(schemas: Sequence[ModelToolSchema]) -> tuple[StructuredTool, ...]:
    tools: list[StructuredTool] = []
    for schema in schemas:
        try:
            tool = StructuredTool.from_function(
                coroutine=_registry_bypass_stub,
                name=schema.name,
                description=schema.description,
                args_schema=deepcopy(schema.input_schema),
                infer_schema=False,
            )
        except Exception:
            raise _configuration_error(
                "Registry schema could not be wrapped for LangChain"
            ) from None
        tools.append(tool)
    return tuple(tools)


class _SerialRegistryToolMiddleware(AgentMiddleware):
    def __init__(
        self,
        runtime: ToolRuntime,
        tracker: _ModelInvocationTracker,
        static_tool_names: frozenset[str],
    ) -> None:
        self._runtime = runtime
        self._tracker = tracker
        self._static_tool_names = static_tool_names
        self._condition = asyncio.Condition()
        self._failed = False
        self._first_failure: BaseException | None = None
        self._next_batch_ordinal = 1

    @property
    def completed_call_count(self) -> int:
        return self._tracker.tool_result_count

    @property
    def first_failure(self) -> BaseException | None:
        return self._first_failure

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[object]],
    ) -> ToolMessage:
        del handler
        call = _langchain_tool_call_to_pathfinder(request.tool_call)
        observed_tool_name = (
            call.name
            if call.name in self._static_tool_names
            else _UNRECOGNIZED_TOOL_OBSERVATION_NAME
        )
        ordinal = self._tracker.batch_ordinal(call.call_id)
        async with self._condition:
            while not self._failed and ordinal != self._next_batch_ordinal:
                await self._condition.wait()
            if self._failed:
                raise _protocol_error("tool batch was aborted after an earlier failure")

            iteration = self._tracker.current_batch_iteration
            started_at: float | None = None
            try:
                iteration, checked_ordinal, started_at = self._tracker.begin_tool(call.call_id)
                if checked_ordinal != ordinal:
                    raise _protocol_error("tool batch ordinal changed during execution")
                content = await self._tracker.await_controlled(
                    self._runtime.execute(_copy_tool_call(call))
                )
                if not isinstance(content, str) or not content.strip():
                    raise _protocol_error("Registry returned invalid model-visible tool text")
            except asyncio.CancelledError as error:
                self._failed = True
                self._first_failure = error
                if started_at is not None:
                    self._tracker.emit(
                        event="agent.tool.failed",
                        iteration=iteration,
                        duration_ms=self._tracker.elapsed_ms(started_at),
                        tool_name=observed_tool_name,
                        batch_ordinal=ordinal,
                        error_category=_error_category(error, stage="tool"),
                    )
                self._condition.notify_all()
                raise
            except Exception as error:
                self._failed = True
                self._first_failure = error
                if started_at is not None:
                    self._tracker.emit(
                        event="agent.tool.failed",
                        iteration=iteration,
                        duration_ms=self._tracker.elapsed_ms(started_at),
                        tool_name=observed_tool_name,
                        batch_ordinal=ordinal,
                        error_category=_error_category(error, stage="tool"),
                    )
                self._condition.notify_all()
                raise

            result_bytes = len(content.encode("utf-8"))
            self._tracker.complete_tool(ordinal=ordinal)
            self._tracker.emit(
                event="agent.tool.completed",
                iteration=iteration,
                duration_ms=self._tracker.elapsed_ms(cast(float, started_at)),
                tool_name=observed_tool_name,
                batch_ordinal=ordinal,
                result_bytes=result_bytes,
            )
            self._next_batch_ordinal += 1
            if not self._tracker.current_batch_call_ids:
                self._next_batch_ordinal = 1
            self._condition.notify_all()
            return ToolMessage(
                content=content,
                name=call.name,
                tool_call_id=call.call_id,
            )


def _validate_inputs(
    *,
    model: ChatModelPort,
    messages: Sequence[ChatMessage],
    metadata: Mapping[str, str],
    tool_runtime: ToolRuntime,
    control: AgentLoopControl,
    observer: AgentLoopObserver,
    prompt_profile: AgentPromptProfile,
) -> tuple[
    tuple[ChatMessage, ...],
    dict[str, str],
    tuple[ModelToolSchema, ...],
    str,
    str,
]:
    if not isinstance(model, ChatModelPort):
        raise ValueError("model must implement ChatModelPort")
    if not messages:
        raise ValueError("messages must not be empty")
    if any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("messages must contain ChatMessage values")
    if any(message.role != "user" for message in messages):
        raise ValueError("initial messages must contain only user messages")
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("metadata keys and values must be strings")
    if not isinstance(tool_runtime, ToolRuntime):
        raise ValueError("tool_runtime must implement ToolRuntime")
    if not isinstance(control, AgentLoopControl):
        raise ValueError("control must use AgentLoopControl")
    if not isinstance(observer, AgentLoopObserver) or inspect.iscoroutinefunction(observer.observe):
        raise ValueError("observer must implement AgentLoopObserver")
    if {"prompt_version", "agent_iteration"}.intersection(metadata):
        raise ValueError("metadata contains a code-owned Agent key")

    try:
        runtime_schemas = tool_runtime.model_tools()
    except Exception:
        raise _configuration_error("Registry runtime failed to provide model tools") from None
    if (
        not isinstance(runtime_schemas, tuple)
        or not runtime_schemas
        or any(not isinstance(schema, ModelToolSchema) for schema in runtime_schemas)
    ):
        raise _configuration_error("Registry runtime returned invalid model tools")
    names = [schema.name for schema in runtime_schemas]
    if len(names) != len(set(names)):
        raise _configuration_error("Registry runtime returned duplicate model tools")

    try:
        prompt_bundle = load_agent_prompt_bundle(prompt_profile)
    except AgentPromptBundleError:
        raise _configuration_error("Agent prompt bundle could not be loaded") from None

    return (
        tuple(message.model_copy(deep=True) for message in messages),
        dict(metadata),
        tuple(_copy_tool_schema(schema) for schema in runtime_schemas),
        prompt_bundle.system_prompt,
        prompt_bundle.version,
    )


def _extract_transcript(state: object) -> tuple[ChatMessage, ...]:
    if not isinstance(state, Mapping):
        raise _protocol_error("create_agent returned an invalid final state")
    messages = state.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(message, BaseMessage) for message in messages
    ):
        raise _protocol_error("create_agent returned invalid final messages")
    return tuple(_langchain_message_to_pathfinder(message) for message in messages)


def _remove_langgraph_task_notes(error: BaseException) -> None:
    notes = getattr(error, "__notes__", None)
    if not isinstance(notes, list):
        return
    retained = [
        note
        for note in notes
        if not (
            isinstance(note, str)
            and note.startswith("During task with name '")
            and "' and id '" in note
        )
    ]
    if retained:
        error.__notes__ = retained
    else:
        delattr(error, "__notes__")


async def run_create_agent_tool_loop(
    *,
    model: ChatModelPort,
    messages: Sequence[ChatMessage],
    metadata: Mapping[str, str],
    tool_runtime: ToolRuntime,
    control: AgentLoopControl,
    observer: AgentLoopObserver,
    prompt_profile: AgentPromptProfile = "base",
) -> AgentLoopResultV1:
    """Run the sole production Agent loop through LangChain v1 create_agent."""

    initial_messages, safe_metadata, schemas, system_prompt, prompt_version = _validate_inputs(
        model=model,
        messages=messages,
        metadata=metadata,
        tool_runtime=tool_runtime,
        control=control,
        observer=observer,
        prompt_profile=prompt_profile,
    )
    static_tools = _build_static_tools(schemas)
    tracker = _ModelInvocationTracker(
        control=control,
        observer=observer,
        prompt_version=prompt_version,
        run_started_at=0.0,
    )
    tracker.run_started_at = tracker.now()
    adapter = _ChatModelPortAdapter(
        port=model,
        metadata=safe_metadata,
        schemas=schemas,
        static_tools=static_tools,
        tool_runtime=tool_runtime,
        tracker=tracker,
        research_budget_feedback_enabled=prompt_profile == "research",
    )
    bridge = _SerialRegistryToolMiddleware(
        tool_runtime,
        tracker,
        frozenset(schema.name for schema in schemas),
    )

    try:
        agent = create_agent(
            model=adapter,
            tools=static_tools,
            system_prompt=system_prompt,
            middleware=(bridge,),
            response_format=None,
            # This nested loop is ephemeral. The outer research graph owns recovery,
            # and inheriting its saver would persist LangChain message objects.
            checkpointer=False,
            store=None,
        )
        langchain_messages = [
            _pathfinder_message_to_langchain(message.model_copy(deep=True))
            for message in initial_messages
        ]
        trace_scope = current_trace_scope()

        async def invoke_isolated_agent() -> Any:
            # Carry only application tracing across the empty runtime Context.
            # The outer LangGraph saver/configuration must remain isolated.
            with bind_trace_scope(trace_scope):
                return await agent.ainvoke(
                    {"messages": langchain_messages},
                    config={
                        "recursion_limit": control.limits.max_iterations + 1,
                        "configurable": {CONFIG_KEY_CHECKPOINTER: False},
                    },
                )

        with tracing_context(enabled=False):
            # Isolate the nested LangGraph context so it cannot inherit the outer
            # recovery saver or emit its LangChain message objects into checkpoints.
            agent_task = asyncio.create_task(invoke_isolated_agent(), context=Context())
            state = await agent_task

        transcript = _extract_transcript(state)
        if transcript[: len(initial_messages)] != initial_messages:
            raise _protocol_error("create_agent changed the initial transcript")
        if not transcript:
            raise _protocol_error("create_agent returned an empty transcript")
        final_message = transcript[-1]
        if (
            final_message.role != "assistant"
            or final_message.content is None
            or not final_message.content.strip()
            or final_message.tool_calls
        ):
            raise _protocol_error("create_agent did not finish with a final assistant answer")

        generated_messages = transcript[len(initial_messages) :]
        if sum(message.role == "assistant" for message in generated_messages) != tracker.call_count:
            raise _protocol_error("create_agent model-call transcript is inconsistent")
        transcript_tool_count = sum(message.role == "tool" for message in generated_messages)
        if (
            transcript_tool_count != bridge.completed_call_count
            or tracker.tool_call_count != tracker.tool_result_count
        ):
            raise _protocol_error("create_agent tool-call transcript is inconsistent")

        tracker.check_boundary()
        result = AgentLoopResultV1(
            answer=cast(str, final_message.content),
            transcript=transcript,
            model_call_count=tracker.call_count,
            tool_call_count=tracker.tool_call_count,
            usage=ModelUsage(
                input_tokens=tracker.input_tokens,
                output_tokens=tracker.output_tokens,
            ),
        )
    except asyncio.CancelledError as error:
        public_error = bridge.first_failure if bridge.first_failure is not None else error
        _remove_langgraph_task_notes(public_error)
        tracker.emit(
            event="agent.loop.failed",
            iteration=tracker.iteration_count,
            duration_ms=tracker.elapsed_ms(tracker.run_started_at),
            error_category=(
                tracker.last_failure_category or _error_category(public_error, stage="loop")
            ),
        )
        if public_error is error:
            raise
        raise public_error from None
    except Exception as error:
        public_error = bridge.first_failure if bridge.first_failure is not None else error
        if isinstance(public_error, GraphRecursionError):
            public_error = AgentLoopLimitExceeded(
                limit_kind="recursion",
                limit=control.limits.max_iterations,
                current_count=tracker.iteration_count,
                requested_count=1,
            )
        _remove_langgraph_task_notes(public_error)
        tracker.emit(
            event="agent.loop.failed",
            iteration=tracker.iteration_count,
            duration_ms=tracker.elapsed_ms(tracker.run_started_at),
            error_category=(
                tracker.last_failure_category or _error_category(public_error, stage="loop")
            ),
        )
        if public_error is error:
            raise
        raise public_error from None

    tracker.emit(
        event="agent.loop.completed",
        iteration=tracker.iteration_count,
        duration_ms=tracker.elapsed_ms(tracker.run_started_at),
    )
    return result
