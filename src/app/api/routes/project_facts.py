from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, status

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.api.schemas.project_facts import (
    FactAddInput,
    FactCommandResponse,
    FactListResponse,
    FactReviewInput,
    FactReviseInput,
    FactSearchResponse,
)
from app.domain.project_facts import ProjectFactService

router = APIRouter(prefix="/api/v2/workspaces/{workspace_id}", tags=["project-facts"])


def _service(request: Request) -> ProjectFactService:
    value = getattr(request.app.state, "project_fact_service", None)
    if not isinstance(value, ProjectFactService):
        raise RuntimeError("project facts are unavailable")
    return value


def _receipt(accepted) -> FactCommandResponse:
    receipt = accepted.receipt
    assert receipt.resource_id is not None
    return FactCommandResponse(
        command_id=receipt.command_id,
        resource_id=receipt.resource_id,
        status="completed",
        replayed=accepted.replayed,
    )


@router.get("/projects/{project_id}/facts", response_model=FactListResponse)
async def list_project_facts(request: Request, tenant: TenantDependency, project_id: UUID):
    return await _service(request).current_facts(tenant, project_id)


@router.post(
    "/projects/{project_id}/facts",
    response_model=FactCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_project_fact(
    request: Request, tenant: TenantDependency, project_id: UUID, body: FactAddInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="material_fact_add",
            project_id=project_id,
            import_id=body.import_id,
            request_id=key,
            candidate=body.candidate,
        )
    )


@router.post(
    "/projects/{project_id}/facts/{fact_id}/versions",
    response_model=FactCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def revise_project_fact(
    request: Request,
    tenant: TenantDependency,
    project_id: UUID,
    fact_id: UUID,
    body: FactReviseInput,
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="material_fact_revise",
            project_id=project_id,
            fact_id=fact_id,
            import_id=body.import_id,
            request_id=key,
            expected_version=body.expected_version,
            candidate=body.candidate,
        )
    )


@router.post(
    "/projects/{project_id}/facts/{fact_id}/reviews",
    response_model=FactCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def review_project_fact(
    request: Request,
    tenant: TenantDependency,
    project_id: UUID,
    fact_id: UUID,
    body: FactReviewInput,
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="material_fact_review",
            project_id=project_id,
            fact_id=fact_id,
            import_id=body.import_id,
            request_id=key,
            expected_version=body.expected_version,
            decision=body.decision,
            attested=body.attested,
        )
    )


@router.get("/facts/search", response_model=list[FactSearchResponse])
async def search_project_facts(
    request: Request,
    tenant: TenantDependency,
    project_ids: Annotated[list[UUID], Query()],
    query: Annotated[str, Query(min_length=1, max_length=200)],
):
    return await _service(request).search_confirmed(tenant, tuple(project_ids), query)
