from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from psycopg.pq import TransactionStatus
from pydantic import SecretStr
from sqlalchemy import select, update

from app.config import Settings
from app.db.checkpoints import CHECKPOINT_SCHEMA, open_postgres_checkpointer
from app.db.models import LLMInvocation, Run, ToolInvocation
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.runtime_policy import DatabaseComponent, DatabaseSessionPolicy
from app.db.session import create_database_engine, create_session_factory
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.runs import RunService, RunStatus
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationLimitError
from tests.integration.support import connect_database
from tests.legacy_runtime import SqlAlchemyRunStore
from tests.legacy_worker import run_worker

pytestmark = pytest.mark.integration
_CHECKPOINT_TABLES = {
    "checkpoint_migrations",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
}


def _schema_tables(database_url: str, schema: str) -> set[str]:
    with connect_database(database_url) as connection:
        rows = connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            """,
            (schema,),
        ).fetchall()
    return {row[0] for row in rows}


async def test_checkpoint_bootstrap_is_separate_and_idempotent(
    migrated_database_url: str,
) -> None:
    assert _schema_tables(migrated_database_url, CHECKPOINT_SCHEMA) == set()
    database_url = SecretStr(migrated_database_url)
    async with open_postgres_checkpointer(database_url):
        assert _schema_tables(migrated_database_url, CHECKPOINT_SCHEMA) == _CHECKPOINT_TABLES
    async with open_postgres_checkpointer(database_url):
        assert _schema_tables(migrated_database_url, CHECKPOINT_SCHEMA) == _CHECKPOINT_TABLES


@pytest.mark.parametrize("custom", [False, True])
async def test_checkpoint_setup_and_runtime_session_parameters(
    migrated_database_url: str, monkeypatch, custom: bool
) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from sqlalchemy.engine import make_url

    settings = (
        Settings(
            db_statement_timeout_ms=9000,
            db_lock_timeout_ms=2000,
            db_idle_in_transaction_timeout_ms=12000,
            db_connect_timeout_seconds=7,
        )
        if custom
        else Settings()
    )
    runtime = DatabaseSessionPolicy.from_settings(settings, DatabaseComponent.CHECKPOINT)
    observed = []
    original_setup = AsyncPostgresSaver.setup

    async def parameters(connection):
        cursor = await connection.execute(
            "SELECT name, setting FROM pg_settings WHERE name IN "
            "('statement_timeout', 'lock_timeout', 'idle_in_transaction_session_timeout', "
            "'application_name', 'search_path')"
        )
        return {row["name"]: row["setting"] for row in await cursor.fetchall()}

    async def observed_setup(saver):
        observed.append(await parameters(saver.conn))
        await original_setup(saver)

    monkeypatch.setattr(AsyncPostgresSaver, "setup", observed_setup)
    # Explicit connect keywords must override DSN values; other DSN settings survive.
    url = make_url(migrated_database_url).update_query_dict(
        {"application_name": "untrusted-name", "connect_timeout": "59"}
    )
    kwargs = {"policy": runtime} if custom else {}
    for _attempt in range(2):
        async with open_postgres_checkpointer(
            SecretStr(url.render_as_string(hide_password=False)), **kwargs
        ) as saver:
            connection = saver.conn
            actual = await parameters(connection)
            assert actual == {
                "statement_timeout": str(runtime.statement_timeout_ms),
                "lock_timeout": str(runtime.lock_timeout_ms),
                "idle_in_transaction_session_timeout": str(runtime.idle_in_transaction_timeout_ms),
                "application_name": "pathfinder-checkpoint",
                "search_path": "pathfinder_checkpoint",
            }
            assert connection.autocommit is True
            assert connection.prepare_threshold == 0
            assert connection.row_factory is dict_row
            assert connection.info.transaction_status == TransactionStatus.IDLE
            assert connection.info.get_parameters()["connect_timeout"] == str(
                runtime.connect_timeout_seconds
            )
        assert connection.closed
    assert len(observed) == 2
    for setup in observed:
        assert setup == {
            "statement_timeout": "30000",
            "lock_timeout": "5000",
            "idle_in_transaction_session_timeout": str(runtime.idle_in_transaction_timeout_ms),
            "application_name": "pathfinder-checkpoint",
            "search_path": "pathfinder_checkpoint",
        }


async def test_real_worker_fake_providers_complete_run_and_persist_checkpoint(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(session_factory)
        ).provision_personal_workspace("step-44-worker-user")
        tenant = TenantContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
            role=WorkspaceRole.ADMIN,
        )
        accepted = await RunService(SqlAlchemyRunStore(session_factory)).create_research_run(
            tenant=tenant,
            query="Research a synthetic backend developer role",
        )

        worker = asyncio.create_task(
            run_worker(Settings(database_url=migrated_database_url, log_level="ERROR"))
        )
        terminal_status: str | None = None
        for _attempt in range(100):
            async with session_factory() as session:
                terminal_status = await session.scalar(
                    select(Run.status).where(Run.id == accepted.run_id)
                )
            if terminal_status in {
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                break
            await asyncio.sleep(0.05)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

        async with session_factory() as session:
            run = await session.get(Run, accepted.run_id)
            llm_rows = list(
                await session.scalars(
                    select(LLMInvocation).where(LLMInvocation.run_id == accepted.run_id)
                )
            )
            tool_rows = list(
                await session.scalars(
                    select(ToolInvocation).where(ToolInvocation.run_id == accepted.run_id)
                )
            )
        assert terminal_status == RunStatus.COMPLETED.value, (
            run.error_category if run is not None else None,
            [(row.graph_node, row.status, row.error_category) for row in llm_rows],
            [(row.tool_name, row.status, row.error_category) for row in tool_rows],
        )
        assert run is not None and run.result_json is not None
        assert run.result_json["evidence_sufficient"] is False
        assert [row.graph_node for row in llm_rows] == [
            "plan",
            "research_agent",
            "research_agent",
            "research_agent",
            "research_agent",
            "write_report",
        ]
        assert len(tool_rows) == 2
        assert all(row.status == "succeeded" for row in tool_rows)
        assert all(
            set(row.result_summary or ()) == {"schema_version", "output_digest", "output_bytes"}
            for row in tool_rows
        )
        assert _schema_tables(migrated_database_url, CHECKPOINT_SCHEMA) == _CHECKPOINT_TABLES
        with connect_database(migrated_database_url) as connection:
            checkpoint_count = connection.execute(
                f'SELECT count(*) FROM "{CHECKPOINT_SCHEMA}".checkpoints WHERE thread_id = %s',
                (str(accepted.run_id),),
            ).fetchone()
        assert checkpoint_count is not None and checkpoint_count[0] > 0
    finally:
        await engine.dispose()


async def test_tool_call_limit_survives_new_recorder_instances(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(session_factory)
        ).provision_personal_workspace("step-44-persistent-tool-limit")
        tenant = TenantContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
            role=WorkspaceRole.ADMIN,
        )
        accepted = await RunService(SqlAlchemyRunStore(session_factory)).create_research_run(
            tenant=tenant,
            query="persistent tool limit",
        )
        async with session_factory.begin() as session:
            await session.execute(
                update(Run)
                .where(Run.id == accepted.run_id)
                .values(status=RunStatus.RUNNING.value, started_at=datetime.now(UTC))
            )

        for _call_number in range(8):
            await SqlAlchemyToolInvocationRecorder(session_factory).reserve(
                invocation_id=uuid4(),
                workspace_id=identity.workspace_id,
                actor_user_id=identity.user_id,
                run_id=accepted.run_id,
                tool_name="search_web",
                effect=ToolEffect.READ_ONLY,
                args_digest=f"sha256:{'a' * 64}",
                call_limit=8,
            )
        recorder = SqlAlchemyToolInvocationRecorder(session_factory)
        assert (
            await recorder.consumed_call_count(
                workspace_id=identity.workspace_id, run_id=accepted.run_id
            )
            == 8
        )
        assert await recorder.consumed_call_count(workspace_id=uuid4(), run_id=accepted.run_id) == 0
        assert (
            await recorder.consumed_call_count(workspace_id=identity.workspace_id, run_id=uuid4())
            == 0
        )
        with pytest.raises(ToolInvocationLimitError):
            await SqlAlchemyToolInvocationRecorder(session_factory).reserve(
                invocation_id=uuid4(),
                workspace_id=identity.workspace_id,
                actor_user_id=identity.user_id,
                run_id=accepted.run_id,
                tool_name="search_web",
                effect=ToolEffect.READ_ONLY,
                args_digest=f"sha256:{'b' * 64}",
                call_limit=8,
            )

        async with session_factory() as session:
            rows = list(
                await session.scalars(
                    select(ToolInvocation).where(ToolInvocation.run_id == accepted.run_id)
                )
            )
        assert len(rows) == 8
        assert all(row.status == "prepared" and row.attempt == 0 for row in rows)
    finally:
        await engine.dispose()
