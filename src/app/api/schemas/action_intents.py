from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.approvals import DECISION_REASON_MAX_CHARS


class ActionIntentAPIModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        hide_input_in_errors=True,
        strict=True,
    )


class ApprovalDecisionRequest(ActionIntentAPIModel):
    decision: Literal["approve", "reject"]
    expected_version: int = Field(ge=1)
    reason: Annotated[str, Field(max_length=DECISION_REASON_MAX_CHARS)] | None = None


class ApprovalDecisionResponse(ActionIntentAPIModel):
    approval_decision_id: UUID
    approval_request_id: UUID
    action_intent_id: UUID
    decision: Literal["approve", "reject"]
    reason: str | None
    decided_at: datetime


class ApprovalRequestResponse(ActionIntentAPIModel):
    request_id: UUID
    status: Literal["pending", "approved", "rejected", "consumed", "expired"]
    version: int = Field(ge=1)
    expires_at: datetime
    args_digest: str
    target_digest: str
    approval_binding_version: int = Field(ge=1)
    approval_binding_digest: str
    policy_version: int = Field(ge=1)
    policy_snapshot: dict[str, object]


class ActionIntentReviewResponse(ActionIntentAPIModel):
    action_intent_id: UUID
    action_key: str
    action_revision: int = Field(ge=1)
    tool_name: str
    effect: Literal["read_only", "reversible", "irreversible"]
    args_snapshot: dict[str, object]
    canonicalization_version: int = Field(ge=1)
    args_digest: str
    target_snapshot: dict[str, object]
    target_canonicalization_version: int = Field(ge=1)
    target_digest: str
    approval_binding_version: int = Field(ge=1)
    approval_binding_digest: str
    status: Literal[
        "proposed",
        "authorized",
        "executing",
        "succeeded",
        "failed",
        "outcome_unknown",
        "cancelled",
    ]
    recovery_attempts: int = Field(ge=0)
    result: dict[str, object] | None
    evidence: dict[str, object] | None
    manual_review_required: bool
    manual_review_instruction: str | None
    approval_request: ApprovalRequestResponse
    decision: ApprovalDecisionResponse | None
