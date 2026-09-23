from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, status

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.api.schemas.material import (
    AliasResponse,
    ImportInput,
    ImportListItemResponse,
    ImportProgressResponse,
    ImportReceiptResponse,
    ProjectInput,
    ProjectResponse,
    SourceInput,
    SourceResponse,
)
from app.domain.material import MaterialService
from app.material.aliases import MaterialAliasRegistry

router = APIRouter(prefix="/api/v2/workspaces/{workspace_id}", tags=["materials"])


def _service(request: Request) -> MaterialService:
    value = getattr(request.app.state, "material_service", None)
    if not isinstance(value, MaterialService):
        raise RuntimeError("material service is unavailable")
    return value


@router.get("/material-aliases", response_model=list[AliasResponse])
async def list_aliases(request: Request, tenant: TenantDependency) -> list[AliasResponse]:
    await _service(request).list_projects(tenant)
    aliases = getattr(request.app.state, "material_aliases", None)
    if not isinstance(aliases, MaterialAliasRegistry):
        raise RuntimeError("material aliases are unavailable")
    return [
        AliasResponse(name=item.name, kind=item.kind)
        for item in aliases.aliases
        if tenant.workspace_id in item.workspace_ids
    ]


@router.get("/projects", response_model=list[ProjectResponse])
async def list_projects(request: Request, tenant: TenantDependency):
    return await _service(request).list_projects(tenant)


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(request: Request, tenant: TenantDependency, body: ProjectInput):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return await _service(request).create_project(tenant, body.name, key)


@router.get("/projects/{project_id}/material-sources", response_model=list[SourceResponse])
async def list_sources(request: Request, tenant: TenantDependency, project_id: UUID):
    return await _service(request).list_sources(tenant, project_id)


@router.post(
    "/projects/{project_id}/material-sources",
    response_model=SourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_source(
    request: Request, tenant: TenantDependency, project_id: UUID, body: SourceInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return await _service(request).create_source(tenant, project_id, body.alias, key)


@router.post(
    "/projects/{project_id}/imports",
    response_model=ImportReceiptResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_import(
    request: Request, tenant: TenantDependency, project_id: UUID, body: ImportInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    accepted = await _service(request).submit_import(tenant, project_id, body.source_ids, key)
    receipt = accepted.receipt
    assert receipt.run_id is not None and receipt.resource_id is not None
    return ImportReceiptResponse(
        command_id=receipt.command_id,
        run_id=receipt.run_id,
        import_id=receipt.resource_id,
        status="queued",
        replayed=accepted.replayed,
    )


@router.get("/projects/{project_id}/imports", response_model=list[ImportListItemResponse])
async def list_imports(request: Request, tenant: TenantDependency, project_id: UUID):
    return await _service(request).list_imports(tenant, project_id)


@router.get("/projects/{project_id}/imports/{import_id}", response_model=ImportProgressResponse)
async def get_import(request: Request, tenant: TenantDependency, project_id: UUID, import_id: UUID):
    return await _service(request).get_import(tenant, project_id, import_id)
