from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import httpx

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.material import MaterialService
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandAccepted, CommandReceiptV1
from app.domain.tenancy import TenantContext, TenantService
from app.main import create_app
from app.material.aliases import MaterialAlias, MaterialAliasRegistry


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
        return None


class _Port:
    def __init__(self) -> None:
        self.keys = []

    async def list_projects(self, tenant):
        return []

    async def submit_import(self, tenant, project_id, source_ids, client_request_id):
        self.keys.append(client_request_id)
        return CommandAccepted(
            CommandReceiptV1(
                command_id=uuid4(), run_id=uuid4(), resource_id=uuid4(), status="queued"
            )
        )


async def test_material_api_requires_current_workspace_and_idempotency_key() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    app = create_app(Settings(log_level="ERROR"))
    app.state.actor_provider = _Actor(tenant.actor_user_id)
    app.state.tenant_service = TenantService(_Resolver(tenant))
    port = _Port()
    app.state.material_service = MaterialService(port)
    app.state.material_aliases = MaterialAliasRegistry(
        (
            MaterialAlias(
                "mine", "file", Path("/tmp/materials"), ("test.md",), (tenant.workspace_id,)
            ),
            MaterialAlias("foreign", "file", Path("/tmp/foreign"), ("test.md",), (uuid4(),)),
        )
    )
    base = f"/api/v2/workspaces/{tenant.workspace_id}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        aliases = await client.get(f"{base}/material-aliases")
        assert aliases.status_code == 200
        assert aliases.json() == [{"name": "mine", "kind": "file"}]
        denied = await client.get(f"/api/v2/workspaces/{uuid4()}/material-aliases")
        assert denied.status_code == 404
        path = f"{base}/projects/{uuid4()}/imports"
        missing = await client.post(path, json={"source_ids": [str(uuid4())]})
        assert missing.status_code == 400
        key = uuid4()
        accepted = await client.post(
            path, json={"source_ids": [str(uuid4())]}, headers={"Idempotency-Key": str(key)}
        )
        assert accepted.status_code == 202
        assert port.keys == [key]
