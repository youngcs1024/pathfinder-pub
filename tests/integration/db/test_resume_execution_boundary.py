from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text, update

from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import Run, RunJob, ToolInvocation
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.runs import SqlAlchemyRunStore
from app.domain.errors import DomainInvariantError
from app.domain.run_execution import RunExecutionInvalidError
from app.domain.runs import RunService
from tests.integration.db.test_worker_jobs import _tenant_and_run
from tests.integration.db.test_worker_jobs import runtime as runtime

pytestmark = pytest.mark.integration


async def snapshot(sessions):
    async with sessions() as session:
        return [
            list(
                (
                    await session.execute(text(f"SELECT to_jsonb(t) FROM {table} t ORDER BY id"))
                ).scalars()
            )
            for table in ("runs", "run_jobs", "run_events", "tool_invocations")
        ]


@pytest.mark.parametrize("leased", [False, True])
async def test_production_never_claims_reclaims_or_rewrites_old_work(runtime, leased):
    tenant, run_id = await _tenant_and_run(runtime, "r11-pending")
    now = datetime.now(UTC)
    if leased:
        assert await runtime.jobs.claim_due_job(
            worker_id="old-worker", now=now, lease_duration=timedelta(seconds=1)
        )
    before = await snapshot(runtime.session_factory)
    reader = SqlAlchemyRunExecutionReader(runtime.session_factory)
    jobs = SqlAlchemyWorkerJobStore(runtime.session_factory, lambda _: timedelta(0))
    assert await reader.has_unsupported_pending_work()
    assert (
        await jobs.claim_due_job(
            worker_id="new-worker",
            now=now + timedelta(minutes=1),
            lease_duration=timedelta(seconds=30),
        )
        is None
    )
    assert (await jobs.reclaim_stale_leases(now=now + timedelta(minutes=1), limit=10)).scanned == 0
    with pytest.raises(RunExecutionInvalidError, match="invalid"):
        await reader.read_for_execution(
            run_id=run_id,
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            graph_version="pathfinder-research-v6",
        )
    with pytest.raises(DomainInvariantError, match="unsupported"):
        await RunService(SqlAlchemyRunStore(runtime.session_factory)).create_research_run(
            tenant=tenant, query="cannot enqueue"
        )
    assert await snapshot(runtime.session_factory) == before


@pytest.mark.parametrize("pending_job", [False, True])
async def test_terminal_runs_do_not_hide_pending_jobs(runtime, pending_job):
    tenant, run_id = await _tenant_and_run(runtime, "r11-terminal")
    await runtime.runs.cancel_run(tenant=tenant, run_id=run_id)
    if pending_job:
        async with runtime.session_factory.begin() as session:
            await session.execute(
                update(RunJob).where(RunJob.run_id == run_id).values(status="queued")
            )
    reader = SqlAlchemyRunExecutionReader(runtime.session_factory)
    before = await snapshot(runtime.session_factory)
    assert await reader.has_unsupported_pending_work() is pending_job
    assert await snapshot(runtime.session_factory) == before
    assert (
        await SqlAlchemyRunStore(runtime.session_factory).get_run(tenant=tenant, run_id=run_id)
    ).status.value == "cancelled"


@pytest.mark.parametrize("status", ["executing", "outcome_unknown"])
async def test_terminal_run_does_not_hide_executing_tool_and_unknown_is_preserved(runtime, status):
    tenant, run_id = await _tenant_and_run(runtime, "r11-tool")
    await runtime.runs.cancel_run(tenant=tenant, run_id=run_id)
    # Historical read-only tool fixture; executing work must block regardless of run status.
    async with runtime.session_factory.begin() as session:
        session.add(
            ToolInvocation(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                run_id=run_id,
                originating_actor_user_id=tenant.actor_user_id,
                tool_name="search_web",
                effect="read_only",
                status=status,
                attempt=1,
                latency_ms=1 if status == "outcome_unknown" else None,
                args_digest="sha256:" + "a" * 64,
                result_summary=None,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC) if status == "outcome_unknown" else None,
                error_category="external_outcome_unknown" if status == "outcome_unknown" else None,
            )
        )
    before = await snapshot(runtime.session_factory)
    assert await SqlAlchemyRunExecutionReader(
        runtime.session_factory
    ).has_unsupported_pending_work() is (status == "executing")
    assert await snapshot(runtime.session_factory) == before


async def test_fresh_database_has_no_unsupported_work(runtime):
    assert not await SqlAlchemyRunExecutionReader(
        runtime.session_factory
    ).has_unsupported_pending_work()


async def test_executing_action_blocks_even_if_tool_and_run_are_terminal(migrated_database_url):
    from app.db.action_execution import SqlAlchemyActionExecutionStore
    from app.db.models import ActionIntent
    from tests.integration.db.test_action_execution import NOW, _approved

    engine, old, identity = await _approved(migrated_database_url, "r11-action")
    try:
        actions = SqlAlchemyActionExecutionStore(old.sessions)
        await actions.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        assert (await actions.begin_send(identity, now=NOW + timedelta(minutes=3))).allowed
        async with old.sessions.begin() as session:
            await session.execute(
                update(Run)
                .where(Run.id == old.run_id)
                .values(
                    status="failed",
                    error_category="synthetic",
                    finished_at=NOW + timedelta(minutes=4),
                )
            )
            await session.execute(
                update(RunJob).where(RunJob.run_id == old.run_id).values(status="done")
            )
            await session.execute(
                update(ToolInvocation)
                .where(ToolInvocation.run_id == old.run_id)
                .values(
                    status="outcome_unknown",
                    finished_at=NOW + timedelta(minutes=4),
                    latency_ms=1,
                    error_category="external_outcome_unknown",
                )
            )
        reader = SqlAlchemyRunExecutionReader(old.sessions)
        assert await reader.has_unsupported_pending_work()
        async with old.sessions.begin() as session:
            await session.execute(
                update(ActionIntent)
                .where(ActionIntent.id == identity.action_intent_id)
                .values(status="outcome_unknown")
            )
        before = await snapshot(old.sessions)
        assert not await reader.has_unsupported_pending_work()
        assert await snapshot(old.sessions) == before
    finally:
        await engine.dispose()
