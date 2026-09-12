from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError

from app.db.models import User, Workspace, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
    transaction,
)
from app.domain.errors import DomainForbiddenError, DomainInvariantError
from app.domain.provisioning import (
    PERSONAL_WORKSPACE_NAME,
    ProvisioningService,
    WorkspaceKind,
    WorkspaceRole,
)
from tests.integration.support import connect_database

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Runtime:
    service: ProvisioningService
    session_factory: AsyncSessionFactory


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    store = SqlAlchemyProvisioningStore(session_factory)
    try:
        yield _Runtime(
            service=ProvisioningService(store),
            session_factory=session_factory,
        )
    finally:
        await engine.dispose()


async def _table_counts(session_factory: AsyncSessionFactory) -> tuple[int, int, int]:
    async with session_factory() as session:
        return (
            await session.scalar(select(func.count()).select_from(User)) or 0,
            await session.scalar(select(func.count()).select_from(Workspace)) or 0,
            await session.scalar(select(func.count()).select_from(WorkspaceMembership)) or 0,
        )


async def test_first_and_repeated_provisioning_return_same_complete_identity(
    runtime: _Runtime,
) -> None:
    first = await runtime.service.provision_personal_workspace("fake-user-1")
    second = await runtime.service.provision_personal_workspace("fake-user-1")

    assert first == second
    assert first.kind is WorkspaceKind.PERSONAL
    assert first.role is WorkspaceRole.ADMIN
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)

    async with runtime.session_factory() as session:
        workspace = await session.get(Workspace, first.workspace_id)
        membership = await session.get(WorkspaceMembership, first.membership_id)

    assert workspace is not None
    assert workspace.name == PERSONAL_WORKSPACE_NAME
    assert workspace.created_by_user_id == first.user_id
    assert workspace.created_at.tzinfo is not None
    assert membership is not None
    assert membership.user_id == first.user_id
    assert membership.workspace_id == first.workspace_id
    assert membership.role == WorkspaceRole.ADMIN.value
    assert membership.revoked_at is None


async def test_eight_concurrent_calls_converge_to_one_identity(runtime: _Runtime) -> None:
    start = asyncio.Event()

    async def provision() -> object:
        await start.wait()
        return await runtime.service.provision_personal_workspace("concurrent-user")

    tasks = [asyncio.create_task(provision()) for _ in range(8)]
    start.set()
    results = await asyncio.gather(*tasks)

    assert len(set(results)) == 1
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)


async def test_different_subjects_remain_isolated(runtime: _Runtime) -> None:
    first, second = await asyncio.gather(
        runtime.service.provision_personal_workspace("first-user"),
        runtime.service.provision_personal_workspace("second-user"),
    )

    assert first.user_id != second.user_id
    assert first.workspace_id != second.workspace_id
    assert first.membership_id != second.membership_id
    assert await _table_counts(runtime.session_factory) == (2, 2, 2)


async def test_wrong_role_creator_membership_fails_closed_without_regranting(
    runtime: _Runtime,
) -> None:
    provisioned = await runtime.service.provision_personal_workspace("wrong-role-user")
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.id == provisioned.membership_id)
            .values(role=WorkspaceRole.MEMBER.value)
        )

    with pytest.raises(
        DomainInvariantError,
        match="creator membership is not an active admin membership",
    ):
        await runtime.service.provision_personal_workspace("wrong-role-user")

    async with runtime.session_factory() as session:
        membership = await session.get(WorkspaceMembership, provisioned.membership_id)

    assert membership is not None
    assert membership.role == WorkspaceRole.MEMBER.value
    assert membership.revoked_at is None
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)


async def test_revoked_creator_membership_is_forbidden_without_regranting(
    runtime: _Runtime,
) -> None:
    provisioned = await runtime.service.provision_personal_workspace("revoked-user")
    revoked_at = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.id == provisioned.membership_id)
            .values(revoked_at=revoked_at)
        )

    with pytest.raises(DomainForbiddenError):
        await runtime.service.provision_personal_workspace("revoked-user")

    async with runtime.session_factory() as session:
        membership = await session.get(WorkspaceMembership, provisioned.membership_id)

    assert membership is not None
    assert membership.role == WorkspaceRole.ADMIN.value
    assert membership.revoked_at == revoked_at
    assert await _table_counts(runtime.session_factory) == (1, 1, 1)


async def test_membership_insert_failure_rolls_back_entire_provisioning_transaction(
    runtime: _Runtime,
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        connection.execute(
            """
            CREATE FUNCTION fail_membership_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected membership failure';
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER fail_membership_insert
            BEFORE INSERT ON workspace_memberships
            FOR EACH ROW EXECUTE FUNCTION fail_membership_insert()
            """
        )

    with pytest.raises(DBAPIError):
        await runtime.service.provision_personal_workspace("rollback-user")

    assert await _table_counts(runtime.session_factory) == (0, 0, 0)
