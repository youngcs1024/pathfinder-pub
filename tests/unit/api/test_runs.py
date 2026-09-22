from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.middleware.cors import CORSMiddleware

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchLimitationV1, ResearchOutputV1, ResearchRequestV1
from app.domain.runs import (
    CURRENT_GRAPH_VERSION,
    RunAccepted,
    RunCancellation,
    RunCreateIdentity,
    RunMode,
    RunRecord,
    RunService,
    RunStatus,
    RunUsageBucket,
    RunUsageSummary,
)
from app.domain.tenancy import TenantContext, TenantService
from tests.legacy_app import create_app


def _empty_bucket() -> RunUsageBucket:
    return RunUsageBucket(
        attempt_count=0,
        succeeded_count=0,
        input_tokens=0,
        output_tokens=0,
        reasoning_output_tokens=0,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        estimated_cost=None,
        currency=None,
        cost_available=False,
    )


class _ActorProvider:
    def __init__(self, actor: ActorContext) -> None:
        self.actor = actor
        self.calls = 0

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        del access_token
        self.calls += 1
        return self.actor


class _TenantResolver:
    def __init__(self, tenant: TenantContext | None) -> None:
        self.tenant = tenant

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        if (
            self.tenant is not None
            and workspace_id == self.tenant.workspace_id
            and actor_user_id == self.tenant.actor_user_id
        ):
            return self.tenant
        return None


class _RunStore:
    def __init__(self) -> None:
        self.identities: list[RunCreateIdentity | None] = []
        self.run_id = uuid4()
        self.created: list[ResearchRequestV1] = []

    async def create_run(
        self,
        *,
        tenant: TenantContext,
        mode: RunMode,
        resume_document_id: UUID | None,
        request: ResearchRequestV1,
        limits: dict[str, int],
        graph_version: str,
        request_identity: RunCreateIdentity | None = None,
    ) -> RunAccepted:
        self.identities.append(request_identity)
        self.created.append(request)
        return RunAccepted(run_id=self.run_id, status=RunStatus.QUEUED)

    async def get_run(self, *, tenant: TenantContext, run_id: UUID) -> RunRecord:
        now = datetime.now(UTC)
        return RunRecord(
            run_id=run_id,
            mode=RunMode.RESEARCH,
            status=RunStatus.COMPLETED,
            graph_version=CURRENT_GRAPH_VERSION,
            result=ResearchOutputV1(
                evidence_sufficient=False,
                limitations=(
                    ResearchLimitationV1(
                        code="insufficient_evidence",
                        detail="No reliable evidence was found.",
                    ),
                ),
            ),
            error_category=None,
            cancel_requested_at=None,
            started_at=now,
            finished_at=now,
            created_at=now,
            updated_at=now,
            usage=RunUsageSummary(chat=_empty_bucket(), embedding=_empty_bucket()),
        )

    async def cancel_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        allow_other_creator: bool,
    ) -> RunCancellation:
        return RunCancellation(
            run_id=run_id,
            status=RunStatus.CANCELLED,
            cancel_requested_at=None,
        )


def _application():
    actor = ActorContext(user_id=uuid4(), subject="unit-actor")
    tenant = TenantContext(
        workspace_id=uuid4(),
        actor_user_id=actor.user_id,
        role=WorkspaceRole.ADMIN,
    )
    provider = _ActorProvider(actor)
    store = _RunStore()
    application = create_app(Settings(log_level="ERROR"))
    application.state.actor_provider = provider
    application.state.tenant_service = TenantService(_TenantResolver(tenant))
    application.state.run_service = RunService(store)
    return application, tenant, provider, store


async def _request(
    application,
    method: str,
    path: str,
    *,
    json: object | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=json, headers=headers)


async def test_create_run_returns_202_and_stable_relative_events_url() -> None:
    application, tenant, provider, store = _application()

    response = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs",
        json={"mode": "research", "query": "Original query"},
        headers={"X-Fake-User": str(uuid4()), "X-Role": "admin"},
    )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": str(store.run_id),
        "status": "queued",
        "events_url": (f"/api/v1/workspaces/{tenant.workspace_id}/runs/{store.run_id}/events"),
    }
    assert store.created == [ResearchRequestV1(query="Original query")]
    assert response.headers["Idempotency-Replayed"] == "false"
    assert store.identities == [None]
    assert provider.calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "research", "query": ""},
        {"mode": "research", "query": "   "},
        {"mode": "research", "query": "x" * 2_001},
        {"mode": "application", "query": "role"},
        {"mode": "research", "query": "role", "user_id": str(uuid4())},
        {"mode": "research", "query": "role", "role": "admin"},
        {"mode": "research", "query": "role", "workspace_id": str(uuid4())},
    ],
)
async def test_create_run_rejects_invalid_or_trusted_extra_fields(payload: object) -> None:
    application, tenant, _provider, store = _application()

    response = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs",
        json=payload,
    )

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert store.created == []
    assert "query" not in response.text


async def test_get_run_returns_typed_result_and_unavailable_empty_cost() -> None:
    application, tenant, _provider, store = _application()

    response = await _request(
        application,
        "GET",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs/{store.run_id}",
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == str(store.run_id)
    assert payload["status"] == "completed"
    assert payload["result"]["evidence_sufficient"] is False
    assert payload["usage"]["chat"] == {
        "attempt_count": 0,
        "succeeded_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "estimated_cost": None,
        "currency": None,
        "cost_available": False,
    }


async def test_cancel_run_returns_current_domain_state() -> None:
    application, tenant, _provider, store = _application()

    response = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs/{store.run_id}/cancel",
    )

    assert response.status_code == 200
    assert response.json() == {
        "run_id": str(store.run_id),
        "status": "cancelled",
        "cancel_requested_at": None,
    }


async def test_nonmember_workspace_is_hidden_as_404() -> None:
    application, _tenant, _provider, store = _application()

    response = await _request(
        application,
        "GET",
        f"/api/v1/workspaces/{uuid4()}/runs/{store.run_id}",
    )

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


def test_supabase_mode_can_build_application_without_jwks_network_access() -> None:
    application = create_app(
        Settings(
            auth_mode="supabase",
            supabase_project_ref="project-ref",
            supabase_publishable_key="sb_publishable_test-public-key",
        )
    )

    assert application.state.settings.auth_mode == "supabase"
    assert "/api/v1/me" in application.openapi()["paths"]


async def test_key_is_parsed_and_replay_header_is_forwarded() -> None:
    application, tenant, _provider, store = _application()
    key = uuid4()
    response = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs",
        json={"mode": "research", "query": "Research"},
        headers={"Idempotency-Key": str(key).upper()},
    )
    assert response.status_code == 202
    assert response.headers["Idempotency-Replayed"] == "false"
    assert store.identities[0].client_request_id == key

    class ReplayStore(_RunStore):
        async def create_run(self, **kwargs) -> RunAccepted:
            accepted = await super().create_run(**kwargs)
            return RunAccepted(accepted.run_id, accepted.status, replayed=True)

    application.state.run_service = RunService(ReplayStore())
    replay = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/runs",
        json={"mode": "research", "query": "Research"},
        headers={"Idempotency-Key": str(key)},
    )
    assert replay.status_code == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay.json()["status"] == "queued"


def test_cors_key_allowlist_does_not_open_origins() -> None:
    application, *_ = _application()
    cors = next(item for item in application.user_middleware if item.cls is CORSMiddleware)
    assert "Idempotency-Key" in cors.kwargs["allow_headers"]
    assert cors.kwargs["allow_origins"] == []
