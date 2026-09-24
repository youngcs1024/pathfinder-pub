"""R5.2 version-scoped history, confirmation and byte route contracts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.errors import DomainNotFoundError
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_confirmation import ConfirmVersionV1, ResumeConfirmationService
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
    def __init__(self, session_id: UUID, version_id: UUID) -> None:
        self.session_id = session_id
        self.version_id = version_id
        self.artifact_id = uuid4()
        self.payload = b"\\documentclass{article}\n\\begin{document}Synthetic\\end{document}\n"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.confirmed_at = datetime.now(UTC)
        self.calls: list[tuple[UUID, ConfirmVersionV1, UUID]] = []

    def _check(self, session_id: UUID, version_id: UUID | None = None) -> None:
        if session_id != self.session_id or (
            version_id is not None and version_id != self.version_id
        ):
            raise DomainNotFoundError

    async def list_versions(self, tenant, session_id):
        self._check(session_id)
        return [
            {
                "session_id": session_id,
                "version_id": self.version_id,
                "version": 2,
                "parent_version_id": uuid4(),
                "artifact_id": self.artifact_id,
                "tex_sha256": self.digest,
                "created_at": self.confirmed_at,
                "confirmation": None,
            }
        ]

    async def confirm(self, tenant, session_id, version_id, request, key):
        self._check(session_id, version_id)
        self.calls.append((version_id, request, key))
        return {
            "confirmation_id": uuid4(),
            "session_id": session_id,
            "version_id": version_id,
            "artifact_id": self.artifact_id,
            "tex_sha256": self.digest,
            "confirmed_by_user_id": tenant.actor_user_id,
            "confirmed_at": self.confirmed_at,
            "command_id": uuid4(),
            "replayed": False,
        }

    async def download(self, tenant, session_id, version_id):
        self._check(session_id, version_id)
        return self.payload, self.digest, 2, True


async def test_confirmation_routes_bind_version_digest_and_download_headers() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    app = create_app(Settings(log_level="ERROR"))
    app.state.actor_provider = _Actor(tenant.actor_user_id)
    app.state.tenant_service = TenantService(_Resolver(tenant))
    session_id, version_id = uuid4(), uuid4()
    port = _Port(session_id, version_id)
    app.state.resume_confirmation_service = ResumeConfirmationService(port)
    base = f"/api/v2/workspaces/{tenant.workspace_id}/resume-sessions/{session_id}/versions"
    body = {
        "version_id": str(version_id),
        "expected_session_revision": 3,
        "expected_current_version_id": str(uuid4()),
        "artifact_id": str(port.artifact_id),
        "tex_sha256": port.digest,
        "attested": True,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        listing = await client.get(base)
        assert listing.status_code == 200
        assert listing.json()[0]["tex_sha256"] == port.digest
        assert (await client.post(f"{base}/{version_id}/confirm", json=body)).status_code == 400
        key = uuid4()
        confirmed = await client.post(
            f"{base}/{version_id}/confirm",
            json=body,
            headers={"Idempotency-Key": str(key)},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["version_id"] == str(version_id)
        assert port.calls[0][2] == key
        assert port.calls[0][1].attested is True
        downloaded = await client.get(f"{base}/{version_id}/download")
        assert downloaded.status_code == 200
        assert downloaded.content == port.payload
        assert downloaded.headers["x-content-sha256"] == port.digest
        assert downloaded.headers["x-resume-version-id"] == str(version_id)
        assert downloaded.headers["x-resume-status"] == "confirmed"
        assert downloaded.headers["content-disposition"] == (
            'attachment; filename="resume-v2-confirmed.tex"'
        )
        assert downloaded.headers["cache-control"] == "private, no-store"
        assert (await client.get(f"{base}/{uuid4()}/download")).status_code == 404
        wrong_workspace = base.replace(str(tenant.workspace_id), str(uuid4()))
        assert (await client.get(wrong_workspace)).status_code == 404


def test_confirm_request_rejects_bad_digest_and_missing_attestation() -> None:
    request = {
        "version_id": uuid4(),
        "expected_session_revision": 0,
        "expected_current_version_id": uuid4(),
        "artifact_id": uuid4(),
        "tex_sha256": "not-a-digest",
    }
    with pytest.raises(ValidationError):
        ConfirmVersionV1.model_validate(request)
