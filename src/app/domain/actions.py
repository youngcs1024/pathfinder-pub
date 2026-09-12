from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

from app.domain.approvals import ApprovalRequestRecord, fixed_approval_policy
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainValidationError,
)
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect

ACTION_KEY = "submit_application"
SUBMIT_APPLICATION_TOOL_NAME = "submit_mock_application"
INITIAL_ACTION_REVISION = 1
CANONICALIZATION_VERSION = 1
TARGET_CANONICALIZATION_VERSION = 1
APPROVAL_BINDING_VERSION = 1
POLICY_VERSION = 1

_ARGS_DIGEST_PREFIX = b"pathfinder-action-args-v1\0"
_TARGET_DIGEST_PREFIX = b"pathfinder-action-target-v1\0"
_BINDING_DIGEST_PREFIX = b"pathfinder-approval-binding-v1\0"
_MAX_CANONICAL_INTEGER = 2_147_483_647


class ActionIntentStatus(StrEnum):
    PROPOSED = "proposed"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    CANCELLED = "cancelled"


_ACTION_TRANSITIONS: dict[ActionIntentStatus, frozenset[ActionIntentStatus]] = {
    ActionIntentStatus.PROPOSED: frozenset(
        {ActionIntentStatus.AUTHORIZED, ActionIntentStatus.CANCELLED}
    ),
    ActionIntentStatus.AUTHORIZED: frozenset(
        {ActionIntentStatus.EXECUTING, ActionIntentStatus.CANCELLED}
    ),
    ActionIntentStatus.EXECUTING: frozenset(
        {
            ActionIntentStatus.SUCCEEDED,
            ActionIntentStatus.FAILED,
            ActionIntentStatus.OUTCOME_UNKNOWN,
        }
    ),
    ActionIntentStatus.SUCCEEDED: frozenset(),
    ActionIntentStatus.FAILED: frozenset(),
    ActionIntentStatus.OUTCOME_UNKNOWN: frozenset(),
    ActionIntentStatus.CANCELLED: frozenset(),
}


def is_valid_action_intent_transition(
    current: ActionIntentStatus,
    target: ActionIntentStatus,
) -> bool:
    return target in _ACTION_TRANSITIONS[current]


def validate_action_intent_transition(
    current: ActionIntentStatus,
    target: ActionIntentStatus,
) -> None:
    if not isinstance(current, ActionIntentStatus) or not isinstance(target, ActionIntentStatus):
        raise DomainInvariantError("action intent transition state is invalid")
    if not is_valid_action_intent_transition(current, target):
        raise DomainConflictError("action intent transition is not allowed")


def persisted_action_intent_status(value: object) -> ActionIntentStatus:
    try:
        return ActionIntentStatus(value)
    except (TypeError, ValueError):
        raise DomainInvariantError("persisted action intent status is invalid") from None


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("action text must not be blank")
    return value


JobReference = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=500),
    AfterValidator(_non_blank),
]
AnswerKey = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=500),
    AfterValidator(_non_blank),
]
AnswerText = Annotated[
    str,
    StringConstraints(strict=True, max_length=4_000),
]
CoverLetterText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=20_000),
    AfterValidator(_non_blank),
]
DigestText = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$"),
]
ActionName = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    ),
]


class ActionContractModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
    )


class SubmitApplicationArgsV1(ActionContractModel):
    job_ref: JobReference
    resume_document_id: UUID
    answers: dict[AnswerKey, AnswerText] = Field(default_factory=dict, max_length=32)
    cover_letter: CoverLetterText


class TrustedActionTargetV1(ActionContractModel):
    provider: Literal["mock_portal"] = "mock_portal"
    target_type: Literal["internal_mock"] = "internal_mock"
    target_ref: Literal["default"] = "default"
    resource_scope: Literal["application_submission"] = "application_submission"


class ApprovalBindingV1(ActionContractModel):
    schema_version: Literal[1] = 1
    workspace_id: UUID
    run_id: UUID
    action_intent_id: UUID
    action_key: ActionName
    action_revision: int = Field(ge=1, le=_MAX_CANONICAL_INTEGER)
    tool_name: ActionName
    effect: ToolEffect
    args_digest: DigestText
    target_digest: DigestText
    policy_version: int = Field(ge=1, le=_MAX_CANONICAL_INTEGER)


def _validate_canonical_json(value: object) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if type(value) is int:
        if not -_MAX_CANONICAL_INTEGER <= value <= _MAX_CANONICAL_INTEGER:
            raise DomainValidationError("canonical integer is out of bounds")
        return
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise DomainValidationError("NaN and Infinity are not canonical JSON")
        raise DomainValidationError("floats are not accepted in action canonicalization")
    if isinstance(value, list):
        for item in value:
            _validate_canonical_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainValidationError("canonical JSON object keys must be strings")
            _validate_canonical_json(item)
        return
    raise DomainValidationError("value is not accepted by action canonicalization")


def canonical_json_bytes(value: dict[str, object]) -> bytes:
    if not isinstance(value, dict):
        raise DomainValidationError("canonical action value must be a JSON object")
    _validate_canonical_json(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise DomainValidationError("action value cannot be canonicalized") from None


def _digest(prefix: bytes, canonical_value: bytes) -> str:
    return f"sha256:{hashlib.sha256(prefix + canonical_value).hexdigest()}"


def canonicalize_action_args(
    value: SubmitApplicationArgsV1 | object,
) -> tuple[dict[str, object], bytes, str]:
    try:
        validated = SubmitApplicationArgsV1.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        raise DomainValidationError("submit application args are invalid") from None
    snapshot = validated.model_dump(mode="json", round_trip=True)
    canonical = canonical_json_bytes(snapshot)
    return snapshot, canonical, _digest(_ARGS_DIGEST_PREFIX, canonical)


def trusted_action_target() -> TrustedActionTargetV1:
    return TrustedActionTargetV1()


def canonicalize_trusted_target() -> tuple[dict[str, object], bytes, str]:
    snapshot = trusted_action_target().model_dump(mode="json", round_trip=True)
    canonical = canonical_json_bytes(snapshot)
    return snapshot, canonical, _digest(_TARGET_DIGEST_PREFIX, canonical)


def approval_binding_digest(binding: ApprovalBindingV1 | object) -> tuple[bytes, str]:
    try:
        validated = ApprovalBindingV1.model_validate(binding, strict=True)
    except (TypeError, ValidationError, ValueError):
        raise DomainValidationError("approval binding is invalid") from None
    canonical = canonical_json_bytes(validated.model_dump(mode="json", round_trip=True))
    return canonical, _digest(_BINDING_DIGEST_PREFIX, canonical)


@dataclass(frozen=True, slots=True)
class PrepareActionCommand:
    tenant: TenantContext
    run_id: UUID
    action_proposal_id: UUID
    action_key: str
    action_revision: int
    args: SubmitApplicationArgsV1
    now: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, TenantContext):
            raise TypeError("prepare action tenant must be a TenantContext")
        if not isinstance(self.run_id, UUID) or not isinstance(self.action_proposal_id, UUID):
            raise TypeError("prepare action identity must use UUIDs")
        if self.action_proposal_id.version != 4:
            raise DomainValidationError("action proposal identity must be UUID4")
        if self.action_key != ACTION_KEY:
            raise DomainValidationError("prepare action key is unsupported")
        if (
            isinstance(self.action_revision, bool)
            or not isinstance(self.action_revision, int)
            or not 1 <= self.action_revision <= _MAX_CANONICAL_INTEGER
        ):
            raise DomainValidationError("action revision is invalid")
        if not isinstance(self.args, SubmitApplicationArgsV1):
            raise DomainValidationError("prepare action args are invalid")
        if (
            not isinstance(self.now, datetime)
            or not isinstance(self.expires_at, datetime)
            or self.now.tzinfo is None
            or self.expires_at.tzinfo is None
            or self.expires_at <= self.now
        ):
            raise DomainValidationError("approval expiry is invalid")


@dataclass(frozen=True, slots=True)
class SupersedeActionCommand:
    tenant: TenantContext
    run_id: UUID
    old_action_intent_id: UUID
    new_action_proposal_id: UUID
    action_key: str
    action_revision: int
    args: SubmitApplicationArgsV1
    now: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.old_action_intent_id, UUID):
            raise DomainValidationError("old action intent identity is invalid")
        if self.old_action_intent_id.version != 4:
            raise DomainValidationError("old action intent identity must be UUID4")
        self.as_prepare_command()

    def as_prepare_command(self) -> PrepareActionCommand:
        return PrepareActionCommand(
            tenant=self.tenant,
            run_id=self.run_id,
            action_proposal_id=self.new_action_proposal_id,
            action_key=self.action_key,
            action_revision=self.action_revision,
            args=self.args,
            now=self.now,
            expires_at=self.expires_at,
        )


@dataclass(frozen=True, slots=True)
class ActionIntentRecord:
    action_intent_id: UUID
    workspace_id: UUID
    originating_actor_user_id: UUID
    run_id: UUID
    action_key: str
    action_revision: int
    tool_name: str
    effect: ToolEffect
    args_snapshot: dict[str, object]
    canonicalization_version: int
    args_digest: str
    target_snapshot: dict[str, object]
    target_canonicalization_version: int
    target_digest: str
    approval_binding_version: int
    approval_binding_digest: str
    status: ActionIntentStatus
    idempotency_key: str
    recovery_attempts: int
    result: dict[str, object] | None
    evidence: dict[str, object] | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PreparedAction:
    intent: ActionIntentRecord
    approval_request: ApprovalRequestRecord


@dataclass(frozen=True, slots=True)
class CancelActionCommand:
    tenant: TenantContext
    run_id: UUID
    action_intent_id: UUID
    approval_request_id: UUID
    reason: Literal["approval_rejected", "approval_expired"]
    now: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, TenantContext) or any(
            not isinstance(value, UUID)
            for value in (self.run_id, self.action_intent_id, self.approval_request_id)
        ):
            raise DomainValidationError("cancel action identity is invalid")
        if self.reason not in {"approval_rejected", "approval_expired"}:
            raise DomainValidationError("cancel action reason is invalid")
        if not isinstance(self.now, datetime) or self.now.tzinfo is None:
            raise DomainValidationError("cancel action time is invalid")


@dataclass(frozen=True, slots=True)
class CancelledAction:
    intent: ActionIntentRecord
    changed: bool


def validate_exact_approval_binding(
    *,
    intent: ActionIntentRecord,
    request: ApprovalRequestRecord,
) -> None:
    """Recompute the complete persisted snapshot→digest→binding chain."""
    if (
        intent.workspace_id != request.workspace_id
        or intent.run_id != request.run_id
        or intent.action_intent_id != request.action_intent_id
        or intent.action_key != ACTION_KEY
        or intent.action_revision < 1
        or intent.tool_name != SUBMIT_APPLICATION_TOOL_NAME
        or intent.effect is not ToolEffect.IRREVERSIBLE
        or intent.canonicalization_version != CANONICALIZATION_VERSION
        or intent.target_canonicalization_version != TARGET_CANONICALIZATION_VERSION
        or intent.approval_binding_version != APPROVAL_BINDING_VERSION
        or request.approval_binding_version != APPROVAL_BINDING_VERSION
        or request.policy_version != POLICY_VERSION
        or request.args_digest != intent.args_digest
        or request.target_digest != intent.target_digest
        or request.approval_binding_digest != intent.approval_binding_digest
        or request.policy_snapshot
        != fixed_approval_policy().model_dump(mode="json", round_trip=True)
    ):
        raise DomainInvariantError("persisted approval binding facts conflict")
    try:
        persisted_args = SubmitApplicationArgsV1.model_validate_json(
            json.dumps(
                intent.args_snapshot,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        raise DomainInvariantError("persisted approval args snapshot is invalid") from None
    args_snapshot, _args_bytes, args_digest = canonicalize_action_args(persisted_args)
    target_snapshot, _target_bytes, target_digest = canonicalize_trusted_target()
    if (
        args_snapshot != intent.args_snapshot
        or args_digest != intent.args_digest
        or target_snapshot != intent.target_snapshot
        or target_digest != intent.target_digest
    ):
        raise DomainInvariantError("persisted approval snapshot digest is invalid")
    _binding_bytes, binding_digest = approval_binding_digest(
        ApprovalBindingV1(
            workspace_id=intent.workspace_id,
            run_id=intent.run_id,
            action_intent_id=intent.action_intent_id,
            action_key=intent.action_key,
            action_revision=intent.action_revision,
            tool_name=intent.tool_name,
            effect=intent.effect,
            args_digest=intent.args_digest,
            target_digest=intent.target_digest,
            policy_version=request.policy_version,
        )
    )
    if binding_digest != intent.approval_binding_digest:
        raise DomainInvariantError("persisted approval binding digest is invalid")


ActionStoreFaultPoint = Literal[
    "prepare_before_commit",
    "supersede_after_expiry_before_new_proposal",
    "cancel_before_commit",
]
type ActionStoreFaultInjector = Callable[
    [ActionStoreFaultPoint],
    Awaitable[None] | None,
]


class ActionStore(Protocol):
    async def prepare_action(self, command: PrepareActionCommand) -> PreparedAction: ...

    async def supersede_action(self, command: SupersedeActionCommand) -> PreparedAction: ...

    async def cancel_action(self, command: CancelActionCommand) -> CancelledAction: ...
