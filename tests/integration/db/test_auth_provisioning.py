from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError

from app.auth.contracts import ActorContext, VerifiedAuthPrincipal
from app.auth.supabase import SupabaseActorProvider
from app.config import Settings
from app.db.models import User, Workspace, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
    transaction,
)
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService, WorkspaceKind, WorkspaceRole
from app.domain.tenancy import TenantService
from app.main import create_app
from tests.integration.support import connect_database

pytestmark = pytest.mark.integration


class _LocalVerifier:
    def __init__(self, subject: str) -> None:
        self.subject = subject
        self.tokens: list[str] = []

    async def verify_access_token(self, token: str) -> VerifiedAuthPrincipal:
        self.tokens.append(token)
        return VerifiedAuthPrincipal(subject=self.subject)


@dataclass(frozen=True)
class _Runtime:
    application: object
    provider: SupabaseActorProvider
    verifier: _LocalVerifier
    session_factory: AsyncSessionFactory


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(session_factory))
    verifier = _LocalVerifier("private-production-subject")
    provider = SupabaseActorProvider(verifier, provisioning)
    settings = Settings(
        database_url=SecretStr(migrated_database_url),
        auth_mode="supabase",
        supabase_project_ref="abcdefghijklmnopqrst",
        supabase_publishable_key="sb_publishable_test-public-key",
        log_level="ERROR",
    )
    application = create_app(settings)
    application.state.actor_provider = provider
    application.state.tenant_service = TenantService(SqlAlchemyTenantResolver(session_factory))
    try:
        yield _Runtime(application, provider, verifier, session_factory)
    finally:
        await engine.dispose()


async def _table_counts(session_factory: AsyncSessionFactory) -> tuple[int, int, int]:
    async with session_factory() as session:
        return (
            await session.scalar(select(func.count()).select_from(User)) or 0,
            await session.scalar(select(func.count()).select_from(Workspace)) or 0,
            await session.scalar(select(func.count()).select_from(WorkspaceMembership)) or 0,
        )


async def _get_me(runtime: _Runtime, token: str = "local-port-token") -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.application),
        base_url="http://testserver",
    ) as client:
        return await client.get(
            "/api/v1/me",
            headers={"Authorization": f"Bearer {token}"},
        )


async def test_first_repeated_and_response_lost_login_converge_through_production_path(
    runtime: _Runtime,
) -> None:
    discarded_actor = await runtime.provider.get_actor("response-was-lost")

    first = await _get_me(runtime, "client-retry")
    second = await _get_me(runtime, "repeated-request")

    assert first.status_code == second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["user_id"] == second_payload["user_id"] == str(discarded_actor.user_id)
    assert first_payload["workspaces"] == second_payload["workspaces"]
    assert first_payload["workspaces"][0]["kind"] == "personal"
    assert first_payload["workspaces"][0]["role"] == "admin"
    assert "private-production-subject" not in first.text
    assert first_payload["subject_summary"].startswith("sha256:")
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)
    assert runtime.verifier.tokens == [
        "response-was-lost",
        "client-retry",
        "repeated-request",
    ]


async def test_me_returns_active_team_membership_and_omits_revoked_membership(
    runtime: _Runtime,
) -> None:
    actor = await runtime.provider.get_actor("initial-login")
    active_team_id = uuid4()
    revoked_team_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add_all(
            [
                Workspace(
                    id=active_team_id,
                    kind=WorkspaceKind.TEAM.value,
                    name="Active Team",
                    created_by_user_id=actor.user_id,
                ),
                Workspace(
                    id=revoked_team_id,
                    kind=WorkspaceKind.TEAM.value,
                    name="Revoked Team",
                    created_by_user_id=actor.user_id,
                ),
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=active_team_id,
                    user_id=actor.user_id,
                    role=WorkspaceRole.REVIEWER.value,
                ),
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=revoked_team_id,
                    user_id=actor.user_id,
                    role=WorkspaceRole.MEMBER.value,
                    revoked_at=func.now(),
                ),
            ]
        )

    response = await _get_me(runtime)

    assert response.status_code == 200
    workspaces = response.json()["workspaces"]
    assert [workspace["kind"] for workspace in workspaces] == ["personal", "team"]
    assert workspaces[1] == {
        "workspace_id": str(active_team_id),
        "kind": "team",
        "name": "Active Team",
        "role": "reviewer",
    }
    assert str(revoked_team_id) not in response.text


async def test_eight_concurrent_actor_mappings_converge_to_one_identity(
    runtime: _Runtime,
) -> None:
    start = asyncio.Event()

    async def map_actor(index: int) -> ActorContext:
        await start.wait()
        return await runtime.provider.get_actor(f"token-{index}")

    tasks = [asyncio.create_task(map_actor(index)) for index in range(8)]
    start.set()
    actors = await asyncio.gather(*tasks)

    assert len({actor.user_id for actor in actors}) == 1
    assert {actor.subject for actor in actors} == {"private-production-subject"}
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)


async def test_membership_fault_rolls_back_then_retry_succeeds_through_actor_mapping(
    runtime: _Runtime,
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        connection.execute(
            """
            CREATE FUNCTION fail_actor_membership_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected actor membership failure';
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER fail_actor_membership_insert
            BEFORE INSERT ON workspace_memberships
            FOR EACH ROW EXECUTE FUNCTION fail_actor_membership_insert()
            """
        )

    with pytest.raises(DBAPIError):
        await runtime.provider.get_actor("faulted-request")

    assert await _table_counts(runtime.session_factory) == (0, 0, 0)

    with connect_database(migrated_database_url) as connection:
        connection.execute("DROP TRIGGER fail_actor_membership_insert ON workspace_memberships")
        connection.execute("DROP FUNCTION fail_actor_membership_insert()")

    actor = await runtime.provider.get_actor("retry-after-fault")

    assert await _table_counts(runtime.session_factory) == (1, 1, 1)
    async with runtime.session_factory() as session:
        persisted_user_id = await session.scalar(
            select(User.id).where(User.auth_subject == "private-production-subject")
        )
    assert persisted_user_id == actor.user_id


async def test_me_denies_revoked_personal_membership_without_regranting(
    runtime: _Runtime,
) -> None:
    actor = await runtime.provider.get_actor("initial-login")
    revoked_at = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        membership_id = await session.scalar(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.user_id == actor.user_id)
            .values(revoked_at=revoked_at)
            .returning(WorkspaceMembership.id)
        )

    response = await _get_me(runtime)

    assert response.status_code == 403
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:pathfinder:problem:forbidden"
    assert response.json()["detail"] == "Access to the requested workspace has been revoked."
    assert "private-production-subject" not in response.text
    async with runtime.session_factory() as session:
        membership = await session.get(WorkspaceMembership, membership_id)
    assert membership is not None
    assert membership.revoked_at == revoked_at
    assert membership.role == WorkspaceRole.ADMIN.value
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)
