"""Existing artifact metadata and exact-byte HTTP delivery."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

import httpx

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.errors import DomainNotFoundError
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_artifacts import ResumeArtifactService, TexArtifactInfoV1
from app.domain.tenancy import TenantContext, TenantService
from app.main import create_app


class _Actor:
    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id

    async def get_actor(self, access_token=None):
        return ActorContext(user_id=self.user_id, subject="synthetic")


class _Resolver:
    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant

    async def resolve_tenant(self, *, workspace_id, actor_user_id):
        if workspace_id == self.tenant.workspace_id and actor_user_id == self.tenant.actor_user_id:
            return self.tenant
        raise DomainNotFoundError


class _Port:
    def __init__(self, artifact_id: UUID) -> None:
        self.artifact_id = artifact_id
        self.payload = b"\\documentclass{article}\n\\begin{document}\nSynthetic\\end{document}\n"
        self.digest = hashlib.sha256(self.payload).hexdigest()

    async def get_info(self, tenant, artifact_id):
        if artifact_id != self.artifact_id:
            raise DomainNotFoundError
        return TexArtifactInfoV1(
            artifact_id=artifact_id,
            profile_version_id=uuid4(),
            template_commit="c056056fcc6f362f8aa9a98f553c5915e4e2eabc",
            template_source_sha256="a" * 64,
            preamble_sha256="b" * 64,
            renderer_version="resume-tex-v1",
            tex_sha256=self.digest,
            byte_count=len(self.payload),
        )

    async def get_bytes(self, tenant, artifact_id):
        if artifact_id != self.artifact_id:
            raise DomainNotFoundError
        return self.payload, self.digest


async def test_artifact_routes_return_metadata_bytes_and_safe_headers() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    app = create_app(Settings(log_level="ERROR"))
    app.state.actor_provider = _Actor(tenant.actor_user_id)
    app.state.tenant_service = TenantService(_Resolver(tenant))
    artifact_id = uuid4()
    port = _Port(artifact_id)
    app.state.resume_artifact_service = ResumeArtifactService(port)
    base = f"/api/v2/workspaces/{tenant.workspace_id}/artifacts/{artifact_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        info = await client.get(base)
        assert info.status_code == 200
        assert info.json()["tex_sha256"] == port.digest
        assert info.json()["compile"]["engine"] == "XeLaTeX"
        assert info.json()["compile"]["font"] == "Microsoft YaHei"
        download = await client.get(f"{base}/download")
        assert download.status_code == 200
        assert download.content == port.payload
        assert download.headers["x-content-sha256"] == port.digest
        assert download.headers["content-disposition"] == (
            f'attachment; filename="resume-{artifact_id}.tex"'
        )
        assert download.headers["cache-control"] == "private, no-store"
        assert download.headers["x-content-type-options"] == "nosniff"
        assert download.headers["content-type"].startswith("application/x-tex")
        assert (await client.get(base.replace(str(artifact_id), str(uuid4())))).status_code == 404
        wrong_workspace = f"/api/v2/workspaces/{uuid4()}/artifacts/{artifact_id}"
        assert (await client.get(wrong_workspace)).status_code == 404
