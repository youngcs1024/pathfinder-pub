from uuid import UUID, uuid4

import httpx
import pytest

from app.auth.contracts import ActorContext
from app.auth.errors import (
    AccessTokenVerificationError,
    JwksUnavailableError,
    TokenVerificationReason,
)
from app.config import Settings
from app.domain.provisioning import WorkspaceKind, WorkspaceRole
from app.domain.tenancy import ActorWorkspaceMembership, TenantContext, TenantService
from app.main import create_app


class _ActorProvider:
    def __init__(self, outcome: ActorContext | BaseException) -> None:
        self.outcome = outcome
        self.tokens: list[str | None] = []

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        self.tokens.append(access_token)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _TenantResolver:
    def __init__(self, memberships: tuple[ActorWorkspaceMembership, ...]) -> None:
        self.memberships = memberships
        self.actor_ids: list[UUID] = []

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        del workspace_id, actor_user_id
        return None

    async def list_active_memberships(
        self,
        *,
        actor_user_id: UUID,
    ) -> tuple[ActorWorkspaceMembership, ...]:
        self.actor_ids.append(actor_user_id)
        return self.memberships


def _application(
    *,
    auth_mode: str = "fake",
    provider_outcome: ActorContext | BaseException | None = None,
):
    actor = ActorContext(user_id=uuid4(), subject="private-supabase-subject")
    workspace = ActorWorkspaceMembership(
        workspace_id=uuid4(),
        kind=WorkspaceKind.PERSONAL,
        name="Personal Workspace",
        role=WorkspaceRole.ADMIN,
    )
    settings = (
        Settings(log_level="ERROR")
        if auth_mode == "fake"
        else Settings(
            log_level="ERROR",
            auth_mode="supabase",
            supabase_project_ref="abcdefghijklmnopqrst",
            supabase_publishable_key="sb_publishable_test-public-key",
        )
    )
    application = create_app(settings)
    provider = _ActorProvider(provider_outcome or actor)
    resolver = _TenantResolver((workspace,))
    application.state.actor_provider = provider
    application.state.tenant_service = TenantService(resolver)
    return application, actor, workspace, provider, resolver


async def _get(application, *, headers: dict[str, str] | None = None) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://testserver",
    ) as client:
        return await client.get("/api/v1/me", headers=headers)


async def test_fake_mode_needs_no_authorization_and_ignores_spoof_headers() -> None:
    application, actor, workspace, provider, resolver = _application()

    response = await _get(
        application,
        headers={
            "Authorization": "Bearer attacker-token",
            "X-Fake-User": str(uuid4()),
            "X-User-ID": str(uuid4()),
            "X-Role": "reviewer",
        },
    )

    assert response.status_code == 200
    assert response.json()["user_id"] == str(actor.user_id)
    assert response.json()["workspaces"][0]["workspace_id"] == str(workspace.workspace_id)
    assert provider.tokens == [None]
    assert resolver.actor_ids == [actor.user_id]


@pytest.mark.parametrize("authorization", [None, "Basic credential", "Digest credential"])
async def test_supabase_mode_requires_bearer_credential(
    authorization: str | None,
) -> None:
    application, _actor, _workspace, provider, _resolver = _application(auth_mode="supabase")
    headers = {} if authorization is None else {"Authorization": authorization}

    response = await _get(application, headers=headers)

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["www-authenticate"] == "Bearer"
    assert provider.tokens == []


async def test_invalid_token_is_401_without_verifier_detail_or_token_leak() -> None:
    error = AccessTokenVerificationError(TokenVerificationReason.INVALID_SIGNATURE)
    application, _actor, _workspace, provider, _resolver = _application(
        auth_mode="supabase",
        provider_outcome=error,
    )

    response = await _get(
        application,
        headers={"Authorization": "Bearer raw-secret-token"},
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert provider.tokens == ["raw-secret-token"]
    assert "raw-secret-token" not in response.text
    assert "signature" not in response.text.lower()


async def test_jwks_unavailable_is_503_and_remains_fail_closed() -> None:
    application, _actor, _workspace, provider, resolver = _application(
        auth_mode="supabase",
        provider_outcome=JwksUnavailableError(),
    )

    response = await _get(application, headers={"Authorization": "Bearer valid-shape"})

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert provider.tokens == ["valid-shape"]
    assert resolver.actor_ids == []


async def test_valid_bearer_maps_internal_actor_and_returns_private_subject_summary() -> None:
    application, actor, workspace, provider, resolver = _application(auth_mode="supabase")

    response = await _get(application, headers={"Authorization": "Bearer exact-token"})

    assert response.status_code == 200
    assert response.json() == {
        "user_id": str(actor.user_id),
        "subject_summary": "sha256:12632e960362",
        "workspaces": [
            {
                "workspace_id": str(workspace.workspace_id),
                "kind": "personal",
                "name": "Personal Workspace",
                "role": "admin",
            }
        ],
    }
    assert actor.subject not in response.text
    assert provider.tokens == ["exact-token"]
    assert resolver.actor_ids == [actor.user_id]
