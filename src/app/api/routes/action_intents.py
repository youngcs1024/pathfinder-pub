from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import ApprovalServiceDependency, TenantDependency
from app.api.schemas.action_intents import (
    ActionIntentReviewResponse,
    ApprovalDecisionResponse,
    ApprovalRequestResponse,
)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["action-intents"])

_UNKNOWN_ACTION_INSTRUCTION = (
    "Verify the stable action intent and idempotency identity in the target system; "
    "do not resubmit or retry manually before verification."
)


@router.get(
    "/action-intents/{action_intent_id}",
    response_model=ActionIntentReviewResponse,
    status_code=status.HTTP_200_OK,
)
async def get_action_intent(
    action_intent_id: UUID,
    tenant: TenantDependency,
    service: ApprovalServiceDependency,
) -> ActionIntentReviewResponse:
    review = await service.get_action_review(tenant=tenant, action_intent_id=action_intent_id)
    intent = review.intent
    request = review.approval_request
    return ActionIntentReviewResponse(
        action_intent_id=intent.action_intent_id,
        action_key=intent.action_key,
        action_revision=intent.action_revision,
        tool_name=intent.tool_name,
        effect=intent.effect,
        args_snapshot=intent.args_snapshot,
        canonicalization_version=intent.canonicalization_version,
        args_digest=intent.args_digest,
        target_snapshot=intent.target_snapshot,
        target_canonicalization_version=intent.target_canonicalization_version,
        target_digest=intent.target_digest,
        approval_binding_version=intent.approval_binding_version,
        approval_binding_digest=intent.approval_binding_digest,
        status=intent.status,
        recovery_attempts=intent.recovery_attempts,
        result=intent.result,
        evidence=intent.evidence,
        manual_review_required=intent.status == "outcome_unknown",
        manual_review_instruction=(
            _UNKNOWN_ACTION_INSTRUCTION if intent.status == "outcome_unknown" else None
        ),
        approval_request=ApprovalRequestResponse(
            request_id=request.request_id,
            status=request.status,
            version=request.version,
            expires_at=request.expires_at,
            args_digest=request.args_digest,
            target_digest=request.target_digest,
            approval_binding_version=request.approval_binding_version,
            approval_binding_digest=request.approval_binding_digest,
            policy_version=request.policy_version,
            policy_snapshot=request.policy_snapshot,
        ),
        decision=(
            ApprovalDecisionResponse.model_validate(review.decision)
            if review.decision is not None
            else None
        ),
    )


@router.post("/action-intents/{action_intent_id}/decision", status_code=410)
async def decide_action_intent(
    action_intent_id: UUID,
    tenant: TenantDependency,
    service: ApprovalServiceDependency,
) -> None:
    await service.get_action_review(tenant=tenant, action_intent_id=action_intent_id)
    raise HTTPException(status_code=410, detail="Historical approvals are read-only.")
