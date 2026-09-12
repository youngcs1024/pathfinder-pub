"""E3.4 HTTP/service/store contracts; concurrency stress belongs to E3.6."""

from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from sqlalchemy import select, update

from app.api.dependencies import actor_context, tenant_context
from app.auth.contracts import ActorContext
from app.config import Settings
from app.db.models import Run, RunEvent, RunJob, WorkspaceMembership
from app.db.session import transaction
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from app.main import create_app
from tests.integration.db import test_run_api_store as support
from tests.integration.db.test_run_api_store import runtime as runtime

pytestmark = pytest.mark.integration
KEY = "12345678-1234-4234-9234-123456789abc"
QUERY = "E34-PRIVATE-QUERY-CANARY"


def _app(runtime: support._Runtime, tenant: TenantContext):
    application = create_app(Settings(log_level="ERROR"))
    application.state.tenant_service = runtime.tenant_service
    application.state.run_service = runtime.run_service

    async def actor() -> ActorContext:
        return ActorContext(user_id=tenant.actor_user_id, subject="e34-http-actor")

    application.dependency_overrides[actor_context] = actor
    return application


async def _request(application, tenant, *, headers=None, payload=None, run_id=None):
    path = f"/api/v1/workspaces/{tenant.workspace_id}/runs"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://testserver"
    ) as client:
        if run_id is not None:
            return await client.get(f"{path}/{run_id}")
        return await client.post(
            path,
            headers=headers,
            json=payload if payload is not None else {"mode": "research", "query": QUERY},
        )


async def test_legacy_and_keyed_http_receipts(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e34-receipts")
    application = _app(runtime, tenant)
    legacy = [await _request(application, tenant) for _ in range(2)]
    first = await _request(application, tenant, headers={"Idempotency-Key": KEY.upper()})
    replay = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    for response in (*legacy, first, replay):
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
    assert len({response.json()["run_id"] for response in (*legacy, first)}) == 3
    assert replay.json() == first.json()
    assert first.json()["events_url"] == (
        f"/api/v1/workspaces/{tenant.workspace_id}/runs/{first.json()['run_id']}/events"
    )
    assert all(r.headers["Idempotency-Replayed"] == "false" for r in (*legacy, first))
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert await support._business_counts(runtime.session_factory) == (3, 3, 3, 3, 3)


async def test_conflict_is_safe_and_does_not_write(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e34-conflict")
    application = _app(runtime, tenant)
    first = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert first.status_code == 202
    async with runtime.session_factory() as session:
        digest = await session.scalar(select(Run.create_request_digest))
    conflict = await _request(
        application,
        tenant,
        headers={"Idempotency-Key": KEY},
        payload={"mode": "research", "query": "E34-OTHER-PRIVATE-CANARY"},
    )
    assert conflict.status_code == 409
    assert conflict.headers["content-type"] == "application/problem+json"
    assert conflict.json()["type"] == "urn:pathfinder:problem:conflict"
    assert conflict.json()["detail"] == "The request conflicts with the current resource state."
    for private in (QUERY, "E34-OTHER-PRIVATE-CANARY", KEY, digest, first.json()["run_id"]):
        assert private not in conflict.text
    assert "Idempotency-Replayed" not in conflict.headers
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


@pytest.mark.parametrize(
    "values",
    [
        [""],
        [" "],
        [KEY, KEY],
        [KEY, KEY.upper()],
        [f"{KEY},{KEY}"],
        [f" {KEY}"],
        [KEY.replace("-", "")],
        [f"urn:uuid:{KEY}"],
        [f"{{{KEY}}}"],
        ["12345678-1234-1234-9234-123456789abc"],
        ["12345678-1234-4234-7234-123456789abc"],
        ["E34-HEADER-CANARY"],
    ],
)
async def test_invalid_headers_do_not_create_business_facts(
    runtime: support._Runtime, values: list[str]
) -> None:
    tenant = await support._personal_tenant(runtime, "e34-invalid-header")
    response = await _request(
        _app(runtime, tenant), tenant, headers=[("Idempotency-Key", value) for value in values]
    )
    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:pathfinder:problem:invalid-idempotency-key"
    assert response.json()["detail"] == "Idempotency-Key must be a single UUID4."
    assert QUERY not in response.text
    assert KEY not in response.text
    assert "E34-HEADER-CANARY" not in response.text
    assert await support._business_counts(runtime.session_factory) == (0, 0, 0, 0, 0)


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_terminal_replay_is_receipt_then_get_reads_current_state(
    runtime: support._Runtime, status: str
) -> None:
    tenant = await support._personal_tenant(runtime, "e34-completed")
    application = _app(runtime, tenant)
    first = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert first.status_code == 202
    run_id = UUID(first.json()["run_id"])
    now = datetime.now(UTC)
    result = None
    if status == "completed":
        result = {
            "schema_version": 2,
            "evidence_sufficient": False,
            "limitations": [{"code": "insufficient_evidence", "detail": "No evidence available."}],
        }
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(
                status=status,
                error_category="synthetic_failure" if status == "failed" else None,
                started_at=now,
                finished_at=now,
                result_json=result,
            )
        )
        await session.execute(update(RunJob).where(RunJob.run_id == run_id).values(status="done"))

    async def snapshot():
        async with runtime.session_factory() as session:
            return [
                [dict(row) for row in (await session.execute(select(model.__table__))).mappings()]
                for model in (Run, RunJob, RunEvent)
            ]

    before = await snapshot()
    replay = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert replay.headers["Idempotency-Replayed"] == "true"
    current = await _request(application, tenant, run_id=run_id)
    assert current.status_code == 200
    assert current.json()["status"] == status
    assert await snapshot() == before
    async with runtime.session_factory() as session:
        assert await session.scalar(select(RunJob.status)) == "done"
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


async def test_scope_isolation_and_foreign_workspace_hidden(runtime: support._Runtime) -> None:
    first = await support._personal_tenant(runtime, "e34-first")
    second = await support._personal_tenant(runtime, "e34-second")
    first_app = _app(runtime, first)
    receipt = await _request(first_app, first, headers={"Idempotency-Key": KEY})
    assert receipt.status_code == 202
    forbidden = await _request(first_app, second, headers={"Idempotency-Key": KEY})
    assert forbidden.status_code == 404
    assert receipt.json()["run_id"] not in forbidden.text
    async with transaction(runtime.session_factory) as session:
        session.add_all(
            [
                WorkspaceMembership(
                    workspace_id=first.workspace_id, user_id=second.actor_user_id, role="member"
                ),
                WorkspaceMembership(
                    workspace_id=second.workspace_id, user_id=first.actor_user_id, role="member"
                ),
            ]
        )
    other_actor = await _request(_app(runtime, second), first, headers={"Idempotency-Key": KEY})
    other_workspace = await _request(first_app, second, headers={"Idempotency-Key": KEY})
    assert other_actor.status_code == other_workspace.status_code == 202
    assert len({r.json()["run_id"] for r in (receipt, other_actor, other_workspace)}) == 3
    assert other_actor.headers["Idempotency-Replayed"] == "false"
    assert other_workspace.headers["Idempotency-Replayed"] == "false"
    assert await support._business_counts(runtime.session_factory) == (3, 3, 3, 3, 3)


@pytest.mark.parametrize("stale_context", [False, True])
async def test_revoke_blocks_replay_at_resolver_and_store(
    runtime: support._Runtime, stale_context: bool
) -> None:
    tenant = await support._personal_tenant(runtime, "e34-revoked")
    application = _app(runtime, tenant)
    first = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert first.status_code == 202
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == tenant.workspace_id)
            .values(revoked_at=datetime.now(UTC))
        )
    if stale_context:
        application.dependency_overrides[tenant_context] = lambda: tenant
    replay = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert replay.status_code == 404
    assert first.json()["run_id"] not in replay.text
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)


async def test_stale_role_and_body_role_cannot_authorize_replay(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e34-role")
    application = _app(runtime, tenant)
    first = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert first.status_code == 202
    forged_body = await _request(
        application,
        tenant,
        headers={"Idempotency-Key": KEY},
        payload={"mode": "research", "query": QUERY, "role": "invalid-role"},
    )
    assert forged_body.status_code == 422
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.workspace_id == tenant.workspace_id)
            .values(role=WorkspaceRole.MEMBER.value)
        )
    application.dependency_overrides[tenant_context] = lambda: tenant
    replay = await _request(application, tenant, headers={"Idempotency-Key": KEY})
    assert replay.status_code == 404
    assert first.json()["run_id"] not in replay.text
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)
