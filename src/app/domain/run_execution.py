from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.provisioning import WorkspaceRole
from app.domain.run_payloads import RunInput
from app.domain.runs import RunMode


class RunExecutionLimitsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    max_model_calls: int = Field(ge=1)
    max_tool_calls: int = Field(ge=0)
    max_tool_results: int = Field(ge=0)
    max_iterations: int = Field(ge=1)

    @model_validator(mode="after")
    def tool_results_must_fit_tool_calls(self) -> RunExecutionLimitsV1:
        if self.max_tool_results > self.max_tool_calls:
            raise ValueError("max_tool_results must not exceed max_tool_calls")
        return self


@dataclass(frozen=True, slots=True)
class RunExecutionInput:
    run_id: UUID
    workspace_id: UUID
    actor_user_id: UUID
    conversation_id: UUID
    graph_version: str
    request: RunInput
    limits: RunExecutionLimitsV1
    mode: RunMode = RunMode.RESEARCH
    resume_document_id: UUID | None = None
    role: WorkspaceRole = WorkspaceRole.MEMBER
    resume_approval_request_id: UUID | None = None


class RunExecutionReadError(Exception):
    """Base error for the narrow, persistence-backed execution reader."""


class RunExecutionCancelledError(RunExecutionReadError):
    """The persisted actor authority or run cancellation no longer permits execution."""


class RunExecutionInvalidError(RunExecutionReadError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__("persisted run execution input is invalid")


class RunExecutionReader(Protocol):
    async def read_for_execution(
        self,
        *,
        run_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        graph_version: str,
    ) -> RunExecutionInput: ...

    async def assert_execution_allowed(
        self,
        *,
        run_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        graph_version: str,
    ) -> None: ...

    async def has_unsupported_pending_work(self) -> bool: ...
