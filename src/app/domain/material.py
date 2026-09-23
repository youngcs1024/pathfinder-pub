"""Application boundary for workspace material operations."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.domain.resume_commands import CommandAccepted
from app.domain.tenancy import TenantContext


class MaterialPort(Protocol):
    async def list_projects(self, tenant: TenantContext) -> list[dict[str, object]]: ...
    async def create_project(
        self, tenant: TenantContext, name: str, client_request_id: UUID
    ) -> dict[str, object]: ...
    async def list_sources(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]: ...
    async def create_source(
        self,
        tenant: TenantContext,
        project_id: UUID,
        alias_name: str,
        client_request_id: UUID,
    ) -> dict[str, object]: ...
    async def submit_import(
        self,
        tenant: TenantContext,
        project_id: UUID,
        source_ids: tuple[UUID, ...],
        client_request_id: UUID,
    ) -> CommandAccepted: ...
    async def get_import(
        self, tenant: TenantContext, project_id: UUID, import_id: UUID
    ) -> dict[str, object]: ...
    async def list_imports(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]: ...


class MaterialService:
    def __init__(self, port: MaterialPort) -> None:
        self.port = port

    async def list_projects(self, tenant: TenantContext) -> list[dict[str, object]]:
        return await self.port.list_projects(tenant)

    async def create_project(
        self, tenant: TenantContext, name: str, client_request_id: UUID
    ) -> dict[str, object]:
        return await self.port.create_project(tenant, name, client_request_id)

    async def list_sources(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]:
        return await self.port.list_sources(tenant, project_id)

    async def create_source(
        self,
        tenant: TenantContext,
        project_id: UUID,
        alias_name: str,
        client_request_id: UUID,
    ) -> dict[str, object]:
        return await self.port.create_source(tenant, project_id, alias_name, client_request_id)

    async def submit_import(
        self,
        tenant: TenantContext,
        project_id: UUID,
        source_ids: tuple[UUID, ...],
        client_request_id: UUID,
    ) -> CommandAccepted:
        return await self.port.submit_import(tenant, project_id, source_ids, client_request_id)

    async def get_import(
        self, tenant: TenantContext, project_id: UUID, import_id: UUID
    ) -> dict[str, object]:
        return await self.port.get_import(tenant, project_id, import_id)

    async def list_imports(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]:
        return await self.port.list_imports(tenant, project_id)
