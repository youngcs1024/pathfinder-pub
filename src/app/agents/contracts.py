from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.llm.ports import ChatMessage, ModelUsage
from app.tools.contracts import CancellationCheck

AgentLoopLimitKind = Literal[
    "model_calls",
    "tool_calls",
    "tool_results",
    "iterations",
    "recursion",
]
AgentLoopEvent = Literal[
    "agent.model.completed",
    "agent.model.failed",
    "agent.tool.completed",
    "agent.tool.failed",
    "agent.loop.completed",
    "agent.loop.failed",
]
AgentLoopErrorCategory = Literal[
    "agent_cancelled",
    "agent_deadline_exceeded",
    "agent_limit_exceeded",
    "configuration_error",
    "external_cancelled",
    "protocol_error",
    "provider_error",
    "provider_timeout",
    "recursion_limit",
    "invalid_tool_proposal",
    "tool_error",
]


class AgentLoopLimitsV1(BaseModel):
    """Caller-owned hard limits for one production Agent loop."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    max_model_calls: int = Field(ge=1)
    max_tool_calls: int = Field(ge=0)
    max_tool_results: int = Field(ge=0)
    max_iterations: int = Field(ge=1)

    @model_validator(mode="after")
    def tool_results_must_fit_tool_calls(self) -> AgentLoopLimitsV1:
        if self.max_tool_results > self.max_tool_calls:
            raise ValueError("max_tool_results must not exceed max_tool_calls")
        return self


type AgentLoopClock = Callable[[], float]


@dataclass(frozen=True, slots=True, repr=False)
class AgentLoopControl:
    """Trusted, non-serializable controls for one Agent loop invocation."""

    limits: AgentLoopLimitsV1
    deadline: float
    cancellation: CancellationCheck
    clock: AgentLoopClock = field(default=monotonic)

    def __post_init__(self) -> None:
        if not isinstance(self.limits, AgentLoopLimitsV1):
            raise ValueError("control limits must use AgentLoopLimitsV1")
        if (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, int | float)
            or not isfinite(float(self.deadline))
            or float(self.deadline) <= 0
        ):
            raise ValueError("control deadline must be a positive finite monotonic value")
        if not isinstance(self.cancellation, CancellationCheck):
            raise ValueError("control cancellation must implement CancellationCheck")
        if not callable(self.clock):
            raise ValueError("control clock must be callable")


class AgentLoopObservationV1(BaseModel):
    """Content-free event emitted by the production Agent loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event: AgentLoopEvent
    sequence: int = Field(ge=1)
    iteration: int = Field(ge=0)
    prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    model_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    tool_result_count: int = Field(ge=0)
    usage: ModelUsage | None = None
    output_bytes: int | None = Field(default=None, ge=0)
    proposed_tool_call_count: int | None = Field(default=None, ge=0)
    tool_name: str | None = None
    batch_ordinal: int | None = Field(default=None, ge=1)
    result_bytes: int | None = Field(default=None, ge=0)
    error_category: AgentLoopErrorCategory | None = None

    @model_validator(mode="after")
    def event_specific_fields_must_match(self) -> AgentLoopObservationV1:
        is_model = self.event.startswith("agent.model.")
        is_tool = self.event.startswith("agent.tool.")
        is_failed = self.event.endswith(".failed")

        model_fields = (self.usage, self.output_bytes, self.proposed_tool_call_count)
        tool_fields = (self.tool_name, self.batch_ordinal, self.result_bytes)
        if is_model and any(value is None for value in model_fields):
            raise ValueError("model observations require usage and output summary fields")
        if not is_model and any(value is not None for value in model_fields):
            raise ValueError("non-model observations forbid model summary fields")
        if is_tool and (self.tool_name is None or self.batch_ordinal is None):
            raise ValueError("tool observations require static name and batch ordinal")
        if not is_tool and any(value is not None for value in tool_fields):
            raise ValueError("non-tool observations forbid tool summary fields")
        if is_failed != (self.error_category is not None):
            raise ValueError("only failed observations require an error category")
        return self


@runtime_checkable
class AgentLoopObserver(Protocol):
    def observe(self, observation: AgentLoopObservationV1) -> None: ...


class AgentLoopResultV1(BaseModel):
    """Framework-neutral result returned by both Agent loop implementations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    answer: str
    transcript: tuple[ChatMessage, ...]
    model_call_count: int = Field(ge=1)
    tool_call_count: int = Field(ge=0)
    usage: ModelUsage

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent loop answer must not be blank")
        return value
