from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.config import Settings
from app.db.errors import classify_database_failure
from app.db.models import Run, ToolInvocation, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy, DatabaseSessionPolicy
from app.db.session import create_database_engine, create_session_factory, transaction
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.errors import DomainUnavailableError
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.runs import RunService
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationAuthorizationError
from tests.legacy_runtime import SqlAlchemyRunStore

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "component",
    [
        DatabaseComponent.API,
        DatabaseComponent.WORKER,
        DatabaseComponent.INGEST,
        DatabaseComponent.WORKER_HEALTHCHECK,
    ],
)
async def test_runtime_policy_applies_to_each_physical_connection_and_reuse(
    database_url: str, component: DatabaseComponent
) -> None:
    settings = Settings(
        db_statement_timeout_ms=8000,
        db_lock_timeout_ms=1500,
        db_idle_in_transaction_timeout_ms=12000,
        db_pool_size=2,
    )
    engine = create_database_engine(
        SecretStr(database_url), policy=DatabasePoolPolicy.from_settings(settings, component)
    )
    expected = {
        "statement_timeout": "8000",
        "lock_timeout": "1500",
        "idle_in_transaction_session_timeout": "12000",
        "application_name": component.value,
    }
    query = text(
        "SELECT name, setting FROM pg_settings WHERE name IN "
        "('statement_timeout', 'lock_timeout', 'idle_in_transaction_session_timeout', "
        "'application_name')"
    )
    try:
        pids = set()
        async with AsyncExitStack() as stack:
            for _ in range(2):
                connection = await stack.enter_async_context(engine.connect())
                assert not connection.in_transaction()
                pids.add(await connection.scalar(text("SELECT pg_backend_pid()")))
                assert dict((await connection.execute(query)).tuples().all()) == expected
                await connection.execute(text("SET LOCAL statement_timeout = '2500ms'"))
                # Closing the connection must roll back its still-open transaction.
        assert len(pids) == 2
        reused_pids = set()
        async with AsyncExitStack() as stack:
            for _ in range(2):
                connection = await stack.enter_async_context(engine.connect())
                assert not connection.in_transaction()
                reused_pids.add(await connection.scalar(text("SELECT pg_backend_pid()")))
                assert dict((await connection.execute(query)).tuples().all()) == expected
        assert reused_pids == pids
    finally:
        await engine.dispose()


@pytest.mark.parametrize("overflow", [0, 1])
async def test_business_pool_enforces_peak_and_recovers_after_release(
    database_url: str, overflow: int
) -> None:
    policy = DatabasePoolPolicy(
        session=DatabaseSessionPolicy(component=DatabaseComponent.API),
        pool_size=1,
        max_overflow=overflow,
        pool_timeout_seconds=0.1,
    )
    engine = create_database_engine(SecretStr(database_url), policy=policy)
    try:
        async with AsyncExitStack() as stack:
            pids = set()
            for _ in range(1 + overflow):
                connection = await stack.enter_async_context(engine.connect())
                pids.add(await connection.scalar(text("SELECT pg_backend_pid()")))
            assert len(pids) == 1 + overflow
            assert engine.pool.checkedout() == 1 + overflow
            async with asyncio.timeout(3):
                with pytest.raises(PoolTimeoutError):
                    async with engine.connect():
                        pytest.fail("pool must not exceed configured capacity")
            assert engine.pool.checkedout() == 1 + overflow
        assert engine.pool.checkedout() == 0
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT 1")) == 1
    finally:
        await engine.dispose()


async def test_concurrent_transactions_use_independent_sessions(database_url: str) -> None:
    engine = create_database_engine(SecretStr(database_url))
    factory = create_session_factory(engine)
    both_connected = asyncio.Event()
    sessions = []
    pids = []

    async def use_session() -> None:
        async with transaction(factory) as session:
            sessions.append(session)
            pids.append(await session.scalar(text("SELECT pg_backend_pid()")))
            if len(pids) == 2:
                both_connected.set()
            await both_connected.wait()
            assert await session.scalar(text("SELECT 1")) == 1

    try:
        async with asyncio.timeout(10), asyncio.TaskGroup() as group:
            group.create_task(use_session())
            group.create_task(use_session())
        assert sessions[0] is not sessions[1]
        assert len(set(pids)) == 2
        assert engine.pool.checkedout() == 0
    finally:
        await engine.dispose()


async def test_sql_error_hides_bound_values_and_transaction_can_recover(
    database_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = create_database_engine(SecretStr(database_url))
    factory = create_session_factory(engine)
    canary = "E23-bound-business-body-canary"
    try:
        with pytest.raises(DBAPIError) as raised:
            async with transaction(factory) as session:
                await session.execute(text("SELECT CAST(:body AS text), 1 / 0"), {"body": canary})
        assert "SQL parameters hidden" in str(raised.value)
        assert canary not in str(raised.value)
        assert canary not in caplog.text
        async with transaction(factory) as session:
            assert await session.scalar(text("SELECT 1")) == 1
    finally:
        await engine.dispose()


def _fault_policy(*, idle_timeout_ms: int = 30_000) -> DatabasePoolPolicy:
    return DatabasePoolPolicy(
        session=DatabaseSessionPolicy(
            component=DatabaseComponent.WORKER,
            statement_timeout_ms=2000,
            lock_timeout_ms=500,
            idle_in_transaction_timeout_ms=idle_timeout_ms,
        ),
        pool_size=1,
        max_overflow=0,
    )


async def _running_run(sessions):
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace("e26-database-fault-user")
    tenant = TenantContext(identity.workspace_id, identity.user_id, WorkspaceRole.ADMIN)
    accepted = await RunService(SqlAlchemyRunStore(sessions)).create_research_run(
        tenant=tenant, query="E26-private-query-canary"
    )
    async with sessions.begin() as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(status="running", started_at=datetime.now(UTC))
        )
    return tenant, accepted.run_id


async def test_row_lock_timeout_rolls_back_and_recovers_after_holder_commit(
    migrated_database_url: str,
) -> None:
    holder_engine = create_database_engine(SecretStr(migrated_database_url))
    waiter_engine = create_database_engine(SecretStr(migrated_database_url), policy=_fault_policy())
    try:
        _, run_id = await _running_run(create_session_factory(holder_engine))
        async with holder_engine.begin() as holder, waiter_engine.connect() as waiter:
            await holder.execute(select(Run.id).where(Run.id == run_id).with_for_update())
            # Awaited FOR UPDATE establishes the barrier before the second connection writes.
            pid = await waiter.scalar(text("SELECT pg_backend_pid()"))
            async with asyncio.timeout(10):
                with pytest.raises(DBAPIError) as raised:
                    await waiter.execute(
                        update(Run).where(Run.id == run_id).values(updated_at=func.now())
                    )
            assert raised.value.orig.sqlstate == "55P03"
            assert classify_database_failure(raised.value) == "lock_unavailable"
            await waiter.rollback()
            assert await waiter.scalar(text("SELECT pg_backend_pid()")) == pid
        # Both transactions have exited; the holder committed, rather than being cancelled.
        async with waiter_engine.begin() as recovered:
            assert await recovered.scalar(select(Run.id).where(Run.id == run_id)) == run_id
            await recovered.execute(
                update(Run).where(Run.id == run_id).values(updated_at=func.now())
            )
    finally:
        await waiter_engine.dispose()
        await holder_engine.dispose()


async def test_statement_timeout_rolls_back_and_reuses_same_connection(
    database_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = create_database_engine(SecretStr(database_url), policy=_fault_policy())
    canary = "E26-statement-bound-body-canary"
    try:
        async with engine.connect() as connection:
            pid = await connection.scalar(text("SELECT pg_backend_pid()"))
            async with asyncio.timeout(10):
                with pytest.raises(DBAPIError) as raised:
                    await connection.execute(
                        text("SELECT pg_sleep(20), CAST(:body AS text)"), {"body": canary}
                    )
            assert raised.value.orig.sqlstate == "57014"
            assert classify_database_failure(raised.value) == "query_canceled"
            assert canary not in str(raised.value)
            assert canary not in caplog.text
            await connection.rollback()
            assert await connection.scalar(text("SELECT pg_backend_pid()")) == pid
            assert await connection.scalar(text("SELECT 1")) == 1
    finally:
        await engine.dispose()


async def test_idle_transaction_termination_invalidates_connection_and_pool_recovers(
    database_url: str,
) -> None:
    engine = create_database_engine(
        SecretStr(database_url), policy=_fault_policy(idle_timeout_ms=1000)
    )
    observer_engine = create_database_engine(SecretStr(database_url))
    try:
        async with observer_engine.connect() as observer, engine.connect() as idle:
            pid = await idle.scalar(text("SELECT pg_backend_pid()"))
            assert idle.in_transaction()
            # Observe termination without issuing another command on the idle transaction.
            async with asyncio.timeout(10):
                while await observer.scalar(
                    text("SELECT count(*) FROM pg_stat_activity WHERE pid = :pid"), {"pid": pid}
                ):
                    await observer.rollback()  # Do not retain a statistics snapshot.
                    await asyncio.sleep(0.05)
            with pytest.raises(DBAPIError) as raised:
                await idle.execute(text("SELECT 1"))
            assert classify_database_failure(raised.value) == "connection_unavailable"
            assert idle.invalidated
            await idle.rollback()
        assert engine.pool.checkedout() == 0
        async with engine.connect() as recovered:
            assert await recovered.scalar(text("SELECT pg_backend_pid()")) != pid
            assert await recovered.scalar(text("SELECT 1")) == 1
    finally:
        await engine.dispose()
        await observer_engine.dispose()


async def test_real_recorder_lock_timeout_is_not_membership_revocation(
    migrated_database_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    holder_engine = create_database_engine(SecretStr(migrated_database_url))
    engine = create_database_engine(SecretStr(migrated_database_url), policy=_fault_policy())
    sessions = create_session_factory(engine)
    try:
        tenant, run_id = await _running_run(create_session_factory(holder_engine))
        recorder = SqlAlchemyToolInvocationRecorder(sessions)

        async def reserve():
            return await recorder.reserve(
                invocation_id=uuid4(),
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                run_id=run_id,
                tool_name="search_web",
                effect=ToolEffect.READ_ONLY,
                args_digest=f"sha256:{'a' * 64}",
                call_limit=8,
            )

        async with holder_engine.begin() as holder:
            await holder.execute(select(Run.id).where(Run.id == run_id).with_for_update())
            async with asyncio.timeout(10):
                with pytest.raises(DomainUnavailableError) as raised:
                    await reserve()
            assert not raised.value.commit_outcome_unknown
        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 0
            assert await session.scalar(select(Run.status).where(Run.id == run_id)) == "running"
        await reserve()
        async with sessions.begin() as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=datetime.now(UTC))
            )
        with pytest.raises(ToolInvocationAuthorizationError):
            await reserve()
        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 1
        assert "E26-private-query-canary" not in caplog.text
        assert migrated_database_url not in caplog.text
        assert engine.pool.checkedout() == 0
    finally:
        await engine.dispose()
        await holder_engine.dispose()
