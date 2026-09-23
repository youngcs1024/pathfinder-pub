from __future__ import annotations

from uuid import UUID, uuid4

import httpx

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.project_facts import ProjectFactService
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandAccepted, CommandReceiptV1
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
        return None


class _Port:
    def __init__(self) -> None:
        self.commands = []

    async def current_facts(self, tenant, project_id):
        return {
            "fact_set_id": None,
            "import_id": None,
            "complete": False,
            "issues": [],
            "facts": [],
        }

    async def command(self, tenant, **kwargs):
        self.commands.append(kwargs)
        return CommandAccepted(
            CommandReceiptV1(command_id=uuid4(), resource_id=uuid4(), status="completed")
        )

    async def search_confirmed(self, tenant, project_ids, query):
        return []


async def test_fact_routes_require_scope_key_and_version() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    app = create_app(Settings(log_level="ERROR"))
    app.state.actor_provider = _Actor(tenant.actor_user_id)
    app.state.tenant_service = TenantService(_Resolver(tenant))
    port = _Port()
    app.state.project_fact_service = ProjectFactService(port)
    project_id, import_id, fact_id = uuid4(), uuid4(), uuid4()
    base = f"/api/v2/workspaces/{tenant.workspace_id}/projects/{project_id}/facts"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        assert (await client.get(base)).status_code == 200
        assert (
            await client.get(f"/api/v2/workspaces/{uuid4()}/projects/{project_id}/facts")
        ).status_code == 404
        payload = {
            "import_id": str(import_id),
            "candidate": {
                "claim": "Synthetic responsibility",
                "kind": "personal_statement",
                "conditions": {},
                "evidence": [],
            },
        }
        assert (await client.post(base, json=payload)).status_code == 400
        key = uuid4()
        added = await client.post(base, json=payload, headers={"Idempotency-Key": str(key)})
        assert added.status_code == 201
        assert port.commands[-1]["request_id"] == key
        assert port.commands[-1]["candidate"].kind == "personal_statement"
        reviewed = await client.post(
            f"{base}/{fact_id}/reviews",
            json={
                "import_id": str(import_id),
                "expected_version": 1,
                "decision": "confirm",
                "attested": True,
            },
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert reviewed.status_code == 201
        assert port.commands[-1]["fact_id"] == fact_id
        invalid = await client.post(
            f"{base}/{fact_id}/reviews",
            json={"import_id": str(import_id), "expected_version": 0, "decision": "confirm"},
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert invalid.status_code == 422
        search = await client.get(
            f"/api/v2/workspaces/{tenant.workspace_id}/facts/search",
            params={"project_ids": str(project_id), "query": "Synthetic"},
        )
        assert search.status_code == 200
