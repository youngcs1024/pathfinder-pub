from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import JsonValue

from app.domain.tool_effects import ToolEffect


class ToolInvocationStatus(StrEnum):
    PREPARED = "prepared"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


_TOOL_INVOCATION_TRANSITIONS: dict[ToolInvocationStatus, frozenset[ToolInvocationStatus]] = {
    ToolInvocationStatus.PREPARED: frozenset(
        {ToolInvocationStatus.EXECUTING, ToolInvocationStatus.FAILED}
    ),
    ToolInvocationStatus.EXECUTING: frozenset(
        {
            ToolInvocationStatus.SUCCEEDED,
            ToolInvocationStatus.FAILED,
            ToolInvocationStatus.OUTCOME_UNKNOWN,
        }
    ),
    ToolInvocationStatus.SUCCEEDED: frozenset(),
    ToolInvocationStatus.FAILED: frozenset(),
    ToolInvocationStatus.OUTCOME_UNKNOWN: frozenset(),
}


def is_valid_tool_invocation_transition(
    current: ToolInvocationStatus,
    target: ToolInvocationStatus,
) -> bool:
    return target in _TOOL_INVOCATION_TRANSITIONS[current]


class ToolInvocationRecorderError(Exception):
    """Persistence or authorization prevented safe tool execution accounting."""


class ToolInvocationAuthorizationError(ToolInvocationRecorderError):
    """The trusted actor/run context no longer authorizes a provider send."""


class ToolInvocationLimitError(ToolInvocationRecorderError):
    """The persisted per-run logical invocation limit was exhausted."""


@dataclass(frozen=True, slots=True)
class ToolInvocationReservation:
    invocation_id: UUID
    call_number: int


class ToolInvocationRecorderPort(Protocol):
    async def consumed_call_count(self, *, workspace_id: UUID, run_id: UUID) -> int: ...

    async def reserve(
        self,
        *,
        invocation_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        tool_name: str,
        effect: ToolEffect,
        args_digest: str,
        call_limit: int,
    ) -> ToolInvocationReservation: ...

    async def start_attempt(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        attempt: int,
    ) -> None: ...

    async def succeed(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        run_id: UUID,
        latency_ms: int,
        result_summary: dict[str, JsonValue],
    ) -> None: ...

    async def fail(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        run_id: UUID,
        latency_ms: int,
        error_category: str,
    ) -> None: ...
