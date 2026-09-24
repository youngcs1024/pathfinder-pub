"""Version-scoped, immutable confirmation and delivery contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.tenancy import TenantContext


class ConfirmationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ConfirmVersionV1(ConfirmationModel):
    version_id: UUID
    expected_session_revision: int = Field(ge=0)
    expected_current_version_id: UUID
    artifact_id: UUID
    tex_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attested: bool


class ConfirmationV1(ConfirmationModel):
    confirmation_id: UUID
    session_id: UUID
    version_id: UUID
    artifact_id: UUID
    tex_sha256: str
    confirmed_by_user_id: UUID
    confirmed_at: datetime


class VersionSummaryV1(ConfirmationModel):
    version_id: UUID
    session_id: UUID
    version: int
    parent_version_id: UUID | None
    artifact_id: UUID
    tex_sha256: str
    created_at: datetime
    confirmation: ConfirmationV1 | None


class ConfirmationReceiptV1(ConfirmationV1):
    command_id: UUID
    replayed: bool


class ResumeConfirmationPort(Protocol):
    async def list_versions(self, tenant: TenantContext, session_id: UUID): ...
    async def confirm(
        self,
        tenant: TenantContext,
        session_id: UUID,
        version_id: UUID,
        request: ConfirmVersionV1,
        key: UUID,
    ): ...
    async def download(
        self,
        tenant: TenantContext,
        session_id: UUID,
        version_id: UUID,
    ) -> tuple[bytes, str, int, bool]: ...


class ResumeConfirmationService:
    def __init__(self, port: ResumeConfirmationPort) -> None:
        self.port = port

    async def list_versions(self, tenant: TenantContext, session_id: UUID):
        return await self.port.list_versions(tenant, session_id)

    async def confirm(
        self,
        tenant: TenantContext,
        session_id: UUID,
        version_id: UUID,
        request: ConfirmVersionV1,
        key: UUID,
    ):
        return await self.port.confirm(tenant, session_id, version_id, request, key)

    async def download(self, tenant: TenantContext, session_id: UUID, version_id: UUID):
        return await self.port.download(tenant, session_id, version_id)
