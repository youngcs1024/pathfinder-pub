"""Historical write routes: never mounted by the production application."""

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Request, Response, status

from app.api.dependencies import ApprovalServiceDependency, RunServiceDependency, TenantDependency
from app.api.run_request_identity import parse_idempotency_key
from app.api.schemas.action_intents import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
)
from app.api.schemas.runs import (
    RunAcceptedResponse,
    RunCancellationResponse,
    RunCreateRequest,
)
from app.domain.approvals import ApprovalDecisionCommand
from app.domain.runs import RunMode

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}")


@router.post(
    "/runs",
    response_model=RunAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_run(
    workspace_id: UUID,
    request: Request,
    response: Response,
    payload: Annotated[RunCreateRequest, Body()],
    tenant: TenantDependency,
    service: RunServiceDependency,
) -> RunAcceptedResponse:
    accepted = await service.create_run(
        tenant=tenant,
        mode=RunMode(payload.mode),
        query=payload.query,
        resume_document_id=payload.resume_document_id,
        client_request_id=parse_idempotency_key(request.headers.getlist("Idempotency-Key")),
    )
    response.headers["Idempotency-Replayed"] = "true" if accepted.replayed else "false"
    return RunAcceptedResponse(
        run_id=accepted.run_id,
        status=accepted.status,
        events_url=(f"/api/v1/workspaces/{workspace_id}/runs/{accepted.run_id}/events"),
    )


@router.post(
    "/runs/{run_id}/cancel",
    response_model=RunCancellationResponse,
    status_code=status.HTTP_200_OK,
)
async def cancel_run(
    run_id: UUID,
    tenant: TenantDependency,
    service: RunServiceDependency,
) -> RunCancellationResponse:
    cancellation = await service.cancel_run(tenant=tenant, run_id=run_id)
    return RunCancellationResponse.model_validate(cancellation)


@router.post(
    "/action-intents/{action_intent_id}/decision",
    response_model=ApprovalDecisionResponse,
    status_code=status.HTTP_200_OK,
)
async def decide_action_intent(
    action_intent_id: UUID,
    payload: Annotated[ApprovalDecisionRequest, Body()],
    tenant: TenantDependency,
    service: ApprovalServiceDependency,
) -> ApprovalDecisionResponse:
    decision = await service.decide(
        ApprovalDecisionCommand(
            tenant=tenant,
            action_intent_id=action_intent_id,
            decision=payload.decision,
            expected_version=payload.expected_version,
            reason=payload.reason,
            now=datetime.now(UTC),
        )
    )
    return ApprovalDecisionResponse.model_validate(decision)
