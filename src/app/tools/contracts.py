from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue

from app.domain.action_execution import ActionExecutionIdentity
from app.domain.tool_effects import ToolEffect
from app.llm.ports import ModelToolCall, ModelToolSchema


class CredentialSource(StrEnum):
    NONE = "none"
    SERVER_MANAGED = "server_managed"


class ToolInputModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class ToolOutputModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


@dataclass(frozen=True, slots=True, repr=False)
class ToolCallBudget:
    call_number: int
    call_limit: int
    remaining_calls: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.call_number, bool)
            or not isinstance(self.call_number, int)
            or self.call_number <= 0
        ):
            raise ValueError("tool budget call number must be a positive integer")
        if (
            isinstance(self.call_limit, bool)
            or not isinstance(self.call_limit, int)
            or self.call_limit <= 0
        ):
            raise ValueError("tool budget call limit must be a positive integer")
        if self.call_number > self.call_limit:
            raise ValueError("tool budget call number must not exceed its limit")
        if self.remaining_calls != self.call_limit - self.call_number:
            raise ValueError("tool budget remaining calls must match its limit")


@runtime_checkable
class CancellationCheck(Protocol):
    def is_cancelled(self) -> bool: ...


@dataclass(frozen=True, slots=True, repr=False)
class ToolRunContext:
    workspace_id: UUID
    actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID | None
    approval_request_id: UUID | None
    trusted_target: Mapping[str, JsonValue] | None
    deadline: float
    cancellation: CancellationCheck


@dataclass(frozen=True, slots=True, repr=False)
class ToolExecutionContext:
    workspace_id: UUID
    actor_user_id: UUID
    run_id: UUID
    invocation_id: UUID
    action_intent_id: UUID | None
    approval_request_id: UUID | None
    trusted_target: Mapping[str, JsonValue] | None
    deadline: float
    budget: ToolCallBudget
    cancellation: CancellationCheck


class ToolHandler(Protocol):
    async def __call__(
        self,
        tool_input: ToolInputModel,
        context: ToolExecutionContext,
    ) -> object: ...


@dataclass(frozen=True, slots=True, repr=False)
class ToolSpec:
    name: str
    description: str
    input_model: type[ToolInputModel]
    output_model: type[ToolOutputModel]
    effect: ToolEffect
    credential_source: CredentialSource
    timeout_seconds: float
    max_attempts: int
    per_run_call_limit: int
    max_output_bytes: int
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class GraphToolPolicy:
    name: str
    allowed_tool_names: frozenset[str]
    allowed_effects: frozenset[ToolEffect]


class ToolTransientError(Exception):
    def __init__(self) -> None:
        super().__init__("tool handler reported a transient failure")


@runtime_checkable
class ToolRuntime(Protocol):
    def model_tools(self) -> tuple[ModelToolSchema, ...]: ...

    def validate_call(self, call: ModelToolCall) -> None: ...

    async def execute(self, call: ModelToolCall) -> str: ...


class ApprovedActionExecutor(Protocol):
    async def execute_approved_action(
        self,
        identity: ActionExecutionIdentity,
        *,
        deadline: float,
        cancellation: CancellationCheck,
    ) -> str: ...


type InvocationIdFactory = Callable[[], UUID]
type MonotonicClock = Callable[[], float]
