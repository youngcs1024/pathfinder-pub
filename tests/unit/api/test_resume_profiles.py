from __future__ import annotations

from uuid import UUID, uuid4

import httpx

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandAccepted, CommandReceiptV1
from app.domain.resume_profiles import ResumeProfileService
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

    async def preview(self, tenant, source_tex):
        return {
            "source_sha256": "a" * 64,
            "complete": False,
            "content": None,
            "claims": [],
            "issues": [{"code": "source_identity_mismatch", "line": 1, "column": 1}],
        }

    async def import_source(self, tenant, source_tex, request_id):
        self.commands.append(("import", request_id, source_tex))
        return CommandAccepted(
            CommandReceiptV1(command_id=uuid4(), resource_id=uuid4(), status="completed")
        )

    async def get_me(self, tenant):
        return {"profile_id": uuid4()}

    async def get_profile(self, tenant, profile_id):
        return {"profile_id": profile_id}

    async def command(self, tenant, *, kind, profile_id, request_id, payload):
        self.commands.append((kind, request_id, profile_id, payload))
        return CommandAccepted(
            CommandReceiptV1(command_id=uuid4(), resource_id=uuid4(), status="completed")
        )


async def test_profile_routes_require_workspace_key_and_typed_payload() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    app = create_app(Settings(log_level="ERROR"))
    app.state.actor_provider = _Actor(tenant.actor_user_id)
    app.state.tenant_service = TenantService(_Resolver(tenant))
    port = _Port()
    app.state.resume_profile_service = ResumeProfileService(port)
    base = f"/api/v2/workspaces/{tenant.workspace_id}/profiles"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        preview = await client.post(f"{base}/import-preview", json={"source_tex": "synthetic"})
        assert preview.status_code == 200
        assert preview.json()["issues"][0]["line"] == 1
        assert (
            await client.post(f"{base}/imports", json={"source_tex": "synthetic"})
        ).status_code == 400
        assert (
            await client.post(
                f"/api/v2/workspaces/{uuid4()}/profiles/import-preview",
                json={"source_tex": "synthetic"},
            )
        ).status_code == 404
        key = uuid4()
        accepted = await client.post(
            f"{base}/imports",
            json={"source_tex": "synthetic"},
            headers={"Idempotency-Key": str(key)},
        )
        assert accepted.status_code == 201
        assert port.commands[-1] == ("import", key, "synthetic")
        profile_id = uuid4()
        edited = await client.post(
            f"{base}/{profile_id}/contact-edits",
            json={
                "expected_version": 1,
                "item_id": str(uuid4()),
                "value": "555-0101",
            },
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert edited.status_code == 201
        assert port.commands[-1][0] == "resume_profile_contact_edit"
        assert (
            await client.post(
                f"{base}/{profile_id}/claim-reviews",
                json={
                    "claim_id": str(uuid4()),
                    "expected_review_version": -1,
                    "decision": "linked",
                    "fact_version_ids": [],
                },
                headers={"Idempotency-Key": str(uuid4())},
            )
        ).status_code == 422
