from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from app.domain.actions import SubmitApplicationArgsV1, TrustedActionTargetV1


class ActionApprovalExpiredError(Exception):
    category = "approval_expired"


class ActionResultUnconfirmedError(Exception):
    category = "external_action_unconfirmed"


class ActionOutcomeUnknownError(Exception):
    category = "external_outcome_unknown"


class ActionExecutionFailedError(Exception):
    category = "external_action_failed"


@dataclass(frozen=True, slots=True)
class ActionExecutionIdentity:
    workspace_id: UUID
    run_id: UUID
    action_intent_id: UUID
    approval_request_id: UUID

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, UUID)
            for value in (
                self.workspace_id,
                self.run_id,
                self.action_intent_id,
                self.approval_request_id,
            )
        ):
            raise TypeError("action execution identity must use UUIDs")


@dataclass(frozen=True, slots=True, repr=False)
class PreparedActionExecution:
    workspace_id: UUID
    originating_actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID
    approval_request_id: UUID
    invocation_id: UUID
    tool_name: str
    args: SubmitApplicationArgsV1
    target: TrustedActionTargetV1
    idempotency_key: str
    status: Literal["authorized", "executing", "succeeded", "failed", "outcome_unknown"] = (
        "authorized"
    )
    recovery_attempts: int = 0
    result: dict[str, object] | None = None
    evidence: dict[str, object] | None = None


@dataclass(frozen=True, slots=True, repr=False)
class SendAuthorization:
    execution: PreparedActionExecution
    allowed: bool
    error_category: Literal["cancelled_before_send", "aborted_before_send"] | None = None

    def __post_init__(self) -> None:
        if self.allowed != (self.error_category is None):
            raise ValueError("send authorization result is inconsistent")


@dataclass(frozen=True, slots=True, repr=False)
class ReconciliationAuthorization:
    execution: PreparedActionExecution
    recovery_attempt: int
    network_allowed: bool


@dataclass(frozen=True, slots=True, repr=False)
class ResendAuthorization:
    execution: PreparedActionExecution
    recovery_attempt: int
    allowed: bool
    error_category: Literal["cancelled_confirmed_absent", "authorization_revoked"] | None = None

    def __post_init__(self) -> None:
        if self.allowed != (self.error_category is None):
            raise ValueError("recovery resend authorization result is inconsistent")


@dataclass(frozen=True, slots=True)
class ConfirmedActionResult:
    external_ref: str
    payload_digest: str | None = None
    source: Literal["initial_response", "reconciliation_query", "recovery_resend"] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.external_ref, str) or not self.external_ref.strip():
            raise ValueError("confirmed action external ref is invalid")
        if self.payload_digest is not None and (
            not isinstance(self.payload_digest, str)
            or not self.payload_digest.startswith("sha256:")
        ):
            raise ValueError("confirmed action payload digest is invalid")
        if (self.payload_digest is None) != (self.source is None):
            raise ValueError("confirmed action evidence is incomplete")


ActionExecutionStoreFaultPoint = Literal[
    "consume_before_commit",
    "send_transition_before_commit",
    "recovery_attempt_before_commit",
    "resend_transition_before_commit",
]
type ActionExecutionStoreFaultInjector = Callable[
    [ActionExecutionStoreFaultPoint], Awaitable[None] | None
]


class ActionExecutionStore(Protocol):
    async def prepare_execution(
        self,
        identity: ActionExecutionIdentity,
        *,
        now: datetime,
    ) -> PreparedActionExecution: ...

    async def begin_send(
        self,
        identity: ActionExecutionIdentity,
        *,
        now: datetime,
    ) -> SendAuthorization: ...

    async def begin_reconciliation(
        self,
        identity: ActionExecutionIdentity,
        *,
        max_attempts: int,
        now: datetime,
    ) -> ReconciliationAuthorization: ...

    async def authorize_resend(
        self,
        identity: ActionExecutionIdentity,
        *,
        expected_recovery_attempt: int,
        now: datetime,
    ) -> ResendAuthorization: ...

    async def confirm_success(
        self,
        identity: ActionExecutionIdentity,
        *,
        result: ConfirmedActionResult,
        latency_ms: int,
        now: datetime,
    ) -> None: ...

    async def confirm_failure(
        self,
        identity: ActionExecutionIdentity,
        *,
        error_category: str,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None: ...

    async def confirm_outcome_unknown(
        self,
        identity: ActionExecutionIdentity,
        *,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None: ...
