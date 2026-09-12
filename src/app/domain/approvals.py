from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainValidationError,
)
from app.domain.tenancy import TenantContext

DECISION_REASON_MAX_CHARS = 1000
ApprovalDecision = Literal["approve", "reject"]
ApprovalStoreFaultPoint = Literal["decision_before_commit"]
type ApprovalStoreFaultInjector = Callable[[ApprovalStoreFaultPoint], Awaitable[None] | None]


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONSUMED = "consumed"
    EXPIRED = "expired"


_APPROVAL_TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset(
        {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}
    ),
    ApprovalStatus.APPROVED: frozenset({ApprovalStatus.CONSUMED, ApprovalStatus.EXPIRED}),
    ApprovalStatus.REJECTED: frozenset(),
    ApprovalStatus.CONSUMED: frozenset(),
    ApprovalStatus.EXPIRED: frozenset(),
}


def is_valid_approval_transition(current: ApprovalStatus, target: ApprovalStatus) -> bool:
    return target in _APPROVAL_TRANSITIONS[current]


def validate_approval_transition(current: ApprovalStatus, target: ApprovalStatus) -> None:
    if not isinstance(current, ApprovalStatus) or not isinstance(target, ApprovalStatus):
        raise DomainInvariantError("approval transition state is invalid")
    if not is_valid_approval_transition(current, target):
        raise DomainConflictError("approval request transition is not allowed")


def persisted_approval_status(value: object) -> ApprovalStatus:
    try:
        return ApprovalStatus(value)
    except (TypeError, ValueError):
        raise DomainInvariantError("persisted approval status is invalid") from None


class FixedApprovalPolicyV1(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    eligible_roles: tuple[Literal["reviewer", "admin"], ...]
    required_approvals: Literal[1]
    separation_of_duty: Literal[False]

    @model_validator(mode="after")
    def policy_is_the_fixed_mvp_policy(self) -> FixedApprovalPolicyV1:
        if self.eligible_roles != ("reviewer", "admin"):
            raise ValueError("eligible roles must match the fixed MVP policy")
        return self


def fixed_approval_policy() -> FixedApprovalPolicyV1:
    return FixedApprovalPolicyV1(
        eligible_roles=("reviewer", "admin"),
        required_approvals=1,
        separation_of_duty=False,
    )


@dataclass(frozen=True, slots=True)
class ApprovalRequestRecord:
    request_id: UUID
    workspace_id: UUID
    run_id: UUID
    action_intent_id: UUID
    status: ApprovalStatus
    args_digest: str
    target_digest: str
    approval_binding_version: int
    approval_binding_digest: str
    policy_version: int
    policy_snapshot: dict[str, object]
    version: int
    expires_at: datetime
    consumed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ApprovalDecisionRecord:
    approval_decision_id: UUID
    workspace_id: UUID
    approval_request_id: UUID
    action_intent_id: UUID
    actor_user_id: UUID
    decision: ApprovalDecision
    reason: str | None
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class ApprovalDecisionCommand:
    tenant: TenantContext
    action_intent_id: UUID
    decision: ApprovalDecision
    expected_version: int
    reason: str | None
    now: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, TenantContext) or not isinstance(
            self.action_intent_id, UUID
        ):
            raise DomainValidationError("approval decision identity is invalid")
        if self.decision not in {"approve", "reject"}:
            raise DomainValidationError("approval decision is invalid")
        if (
            isinstance(self.expected_version, bool)
            or not isinstance(self.expected_version, int)
            or self.expected_version < 1
        ):
            raise DomainValidationError("approval expected version is invalid")
        if self.reason is not None and (
            not isinstance(self.reason, str) or len(self.reason) > DECISION_REASON_MAX_CHARS
        ):
            raise DomainValidationError("approval decision reason is invalid")
        if not isinstance(self.now, datetime) or self.now.tzinfo is None:
            raise DomainValidationError("approval decision time is invalid")


@dataclass(frozen=True, slots=True)
class ApprovalReviewRecord:
    intent: object
    approval_request: ApprovalRequestRecord
    decision: ApprovalDecisionRecord | None


@dataclass(frozen=True, slots=True)
class ApprovalResumeRecord:
    approval_request: ApprovalRequestRecord
    action_intent_id: UUID
    decision: ApprovalDecision | None


class ApprovalDecisionStore(Protocol):
    async def get_action_review(
        self, *, tenant: TenantContext, action_intent_id: UUID
    ) -> ApprovalReviewRecord: ...

    async def decide(self, command: ApprovalDecisionCommand) -> ApprovalDecisionRecord: ...


class ApprovalResumeResolver(Protocol):
    async def resolve_approval_resume(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        action_intent_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalResumeRecord: ...


class ApprovalService:
    def __init__(self, store: ApprovalDecisionStore) -> None:
        self._store = store

    async def get_action_review(
        self, *, tenant: TenantContext, action_intent_id: UUID
    ) -> ApprovalReviewRecord:
        if not isinstance(tenant, TenantContext) or not isinstance(action_intent_id, UUID):
            raise DomainValidationError("action review identity is invalid")
        return await self._store.get_action_review(tenant=tenant, action_intent_id=action_intent_id)

    async def decide(self, command: ApprovalDecisionCommand) -> ApprovalDecisionRecord:
        if not isinstance(command, ApprovalDecisionCommand):
            raise DomainValidationError("approval decision command is invalid")
        return await self._store.decide(command)


@dataclass(frozen=True, slots=True)
class ApprovalExpirySweepSummary:
    scanned: int
    expired: int
    requeued: int

    def __post_init__(self) -> None:
        values = (self.scanned, self.expired, self.requeued)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("approval expiry sweep counts must be non-negative integers")
        if self.expired > self.scanned or self.requeued > self.expired:
            raise ValueError("approval expiry sweep counts are inconsistent")


class ApprovalRequestExpirySweeper(Protocol):
    async def sweep_due_approval_requests(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> ApprovalExpirySweepSummary: ...
