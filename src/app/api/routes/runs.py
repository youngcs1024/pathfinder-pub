from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Request, Response, status

from app.api.dependencies import RunServiceDependency, TenantDependency
from app.api.run_request_identity import parse_idempotency_key
from app.api.schemas.runs import (
    RunAcceptedResponse,
    RunCancellationResponse,
    RunCreateRequest,
    RunDetailResponse,
)
from app.domain.runs import RunMode

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["runs"])


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


@router.get(
    "/runs/{run_id}",
    response_model=RunDetailResponse,
    status_code=status.HTTP_200_OK,
)
async def get_run(
    run_id: UUID,
    tenant: TenantDependency,
    service: RunServiceDependency,
) -> RunDetailResponse:
    record = await service.get_run(tenant=tenant, run_id=run_id)
    return RunDetailResponse.model_validate(record)


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
