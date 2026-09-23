"""Narrow service boundary for private resume profiles."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.domain.tenancy import TenantContext


class ResumeProfilePort(Protocol):
    async def preview(self, tenant: TenantContext, source_tex: str): ...
    async def import_source(self, tenant: TenantContext, source_tex: str, request_id: UUID): ...
    async def get_me(self, tenant: TenantContext): ...
    async def get_profile(self, tenant: TenantContext, profile_id: UUID): ...
    async def command(
        self, tenant: TenantContext, *, kind: str, profile_id: UUID, request_id: UUID, payload
    ): ...


class ResumeProfileService:
    def __init__(self, port: ResumeProfilePort) -> None:
        self.port = port

    async def preview(self, tenant: TenantContext, source_tex: str):
        return await self.port.preview(tenant, source_tex)

    async def import_source(self, tenant: TenantContext, source_tex: str, request_id: UUID):
        return await self.port.import_source(tenant, source_tex, request_id)

    async def get_me(self, tenant: TenantContext):
        return await self.port.get_me(tenant)

    async def get_profile(self, tenant: TenantContext, profile_id: UUID):
        return await self.port.get_profile(tenant, profile_id)

    async def command(
        self, tenant: TenantContext, *, kind: str, profile_id: UUID, request_id: UUID, payload
    ):
        return await self.port.command(
            tenant, kind=kind, profile_id=profile_id, request_id=request_id, payload=payload
        )
