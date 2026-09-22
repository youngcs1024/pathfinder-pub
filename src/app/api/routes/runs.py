from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import RunServiceDependency, TenantDependency
from app.api.schemas.runs import (
    RunDetailResponse,
)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["runs"])


@router.post("/runs", status_code=410)
async def create_run(tenant: TenantDependency) -> None:
    raise HTTPException(status_code=410, detail="The research and application workflow is retired.")


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


@router.post("/runs/{run_id}/cancel", status_code=410)
async def cancel_run(
    run_id: UUID,
    tenant: TenantDependency,
    service: RunServiceDependency,
) -> None:
    await service.get_run(tenant=tenant, run_id=run_id)
    raise HTTPException(status_code=410, detail="Historical runs are read-only.")
