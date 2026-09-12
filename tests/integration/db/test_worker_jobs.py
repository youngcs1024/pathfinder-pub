from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

from app.config import Settings
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import Run, RunEvent, RunJob, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.runs import SqlAlchemyRunStore
from app.db.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
    transaction,
)
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainInvariantError, DomainNotFoundError
from app.domain.jobs import ClaimedJob, JobStatus
from app.domain.provisioning import ProvisioningService
from app.domain.research import ResearchLimitationV1, ResearchOutputV1
from app.domain.runs import RunService, RunStatus
from app.domain.tenancy import TenantContext, TenantService
from app.worker.contracts import RunExecutionResult
from app.worker.main import run_worker
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.support import connect_database

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Runtime:
    session_factory: AsyncSessionFactory
    provisioning: ProvisioningService
    tenants: TenantService
    runs: RunService
    jobs: SqlAlchemyWorkerJobStore


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        yield _Runtime(
            session_factory=session_factory,
            provisioning=ProvisioningService(SqlAlchemyProvisioningStore(session_factory)),
            tenants=TenantService(SqlAlchemyTenantResolver(session_factory)),
            runs=RunService(SqlAlchemyRunStore(session_factory)),
            jobs=SqlAlchemyWorkerJobStore(
                session_factory,
                lambda attempt: timedelta(seconds=attempt),
            ),
        )
    finally:
        await engine.dispose()


async def _tenant_and_run(
    runtime: _Runtime,
    subject: str,
) -> tuple[TenantContext, UUID]:
    provisioned = await runtime.provisioning.provision_personal_workspace(subject)
    tenant = await runtime.tenants.resolve_tenant(
        workspace_id=provisioned.workspace_id,
        actor_user_id=provisioned.user_id,
    )
    accepted = await runtime.runs.create_research_run(tenant=tenant, query="Research")
    return tenant, accepted.run_id


def _output(detail: str = "Contract fake did not execute research.") -> ResearchOutputV1:
    return ResearchOutputV1(
        evidence_sufficient=False,
        limitations=(ResearchLimitationV1(code="insufficient_evidence", detail=detail),),
    )


async def _prepare(
    runtime: _Runtime,
    job: ClaimedJob,
    tenant: TenantContext,
    now: datetime,
) -> TenantContext:
    prepared = await runtime.jobs.prepare_claimed_job(
        job=job,
        resolved_tenant=tenant,
        now=now,
    )
    assert prepared.disposition == "execute"
    assert prepared.tenant is not None
    return prepared.tenant


async def _job_and_events(
    runtime: _Runtime,
    run_id: UUID,
) -> tuple[RunJob, Run, list[RunEvent]]:
    async with runtime.session_factory() as session:
        job = await session.scalar(select(RunJob).where(RunJob.run_id == run_id))
        run = await session.get(Run, run_id)
        events = list(
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
        )
    assert job is not None and run is not None
    return job, run, events


async def test_concurrent_claim_has_one_owner_and_completed_state_is_atomic(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-concurrent-claim")
    now = datetime.now(UTC)

    first, second = await asyncio.gather(
        runtime.jobs.claim_due_job(
            worker_id="worker-a",
            now=now,
            lease_duration=timedelta(seconds=30),
        ),
        runtime.jobs.claim_due_job(
            worker_id="worker-b",
            now=now,
            lease_duration=timedelta(seconds=30),
        ),
    )

    claims = [claim for claim in (first, second) if claim is not None]
    assert len(claims) == 1
    claim = claims[0]
    assert claim.run_id == run_id and claim.attempt == 1
    prepared_tenant = await _prepare(runtime, claim, tenant, now)
    assert prepared_tenant.role is tenant.role
    assert await runtime.jobs.complete(job=claim, result=_output(), now=now) is True

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value
    assert job.owner_token is None and job.lease_expires_at is None and job.leased_by is None
    assert run.status == RunStatus.COMPLETED.value
    assert run.result_json == _output().model_dump(mode="json", round_trip=True)
    record = await runtime.runs.get_run(tenant=tenant, run_id=run_id)
    assert record.status is RunStatus.COMPLETED
    assert record.result == _output()
    assert [(event.seq, event.type) for event in events] == [
        (1, "run.created"),
        (2, "run.status_changed"),
        (3, "run.completed"),
    ]


async def test_due_order_and_future_jobs_are_not_claimed(runtime: _Runtime) -> None:
    _first_tenant, first_run = await _tenant_and_run(runtime, "worker-due-first")
    _second_tenant, second_run = await _tenant_and_run(runtime, "worker-due-second")
    now = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(RunJob)
            .where(RunJob.run_id == first_run)
            .values(available_at=now + timedelta(minutes=5))
        )
        await session.execute(
            update(RunJob)
            .where(RunJob.run_id == second_run)
            .values(available_at=now - timedelta(seconds=1))
        )

    claimed = await runtime.jobs.claim_due_job(
        worker_id="worker-order",
        now=now,
        lease_duration=timedelta(seconds=30),
    )

    assert claimed is not None and claimed.run_id == second_run
    second_claim = await runtime.jobs.claim_due_job(
        worker_id="worker-order",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert second_claim is None


async def test_stale_reclaim_invalidates_old_token_and_new_owner_completes(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-stale-token")
    started_at = datetime.now(UTC)
    first = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=started_at,
        lease_duration=timedelta(seconds=30),
    )
    assert first is not None
    await _prepare(runtime, first, tenant, started_at)

    expired_at = started_at + timedelta(seconds=31)
    summary = await runtime.jobs.reclaim_stale_leases(now=expired_at, limit=10)
    assert summary.requeued == 1
    second = await runtime.jobs.claim_due_job(
        worker_id="worker-b",
        now=expired_at + timedelta(seconds=2),
        lease_duration=timedelta(seconds=30),
    )
    assert second is not None and second.owner_token != first.owner_token and second.attempt == 2

    assert (
        await runtime.jobs.heartbeat(
            job=first,
            now=expired_at + timedelta(seconds=2),
            lease_duration=timedelta(seconds=30),
        )
        is False
    )
    assert (
        await runtime.jobs.complete(
            job=first,
            result=_output("Old worker result."),
            now=expired_at + timedelta(seconds=2),
        )
        is False
    )
    await _prepare(runtime, second, tenant, expired_at + timedelta(seconds=2))
    assert (
        await runtime.jobs.complete(
            job=second,
            result=_output("New worker result."),
            now=expired_at + timedelta(seconds=2),
        )
        is True
    )

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value and job.attempt == 2
    assert run.result_json == _output("New worker result.").model_dump(mode="json", round_trip=True)
    assert [event.type for event in events] == [
        "run.created",
        "run.status_changed",
        "job.lease_expired",
        "run.completed",
    ]


async def test_abrupt_death_after_claim_recovers_only_after_lease_expiry(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-crash-after-claim")
    claimed_at = datetime.now(UTC)
    first = await runtime.jobs.claim_due_job(
        worker_id="crashed-worker",
        now=claimed_at,
        lease_duration=timedelta(seconds=30),
    )
    assert first is not None and first.attempt == 1

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.LEASED.value
    assert job.owner_token == first.owner_token
    assert run.status == RunStatus.QUEUED.value
    assert [(event.seq, event.type) for event in events] == [(1, "run.created")]

    expired_at = claimed_at + timedelta(seconds=31)
    summary = await runtime.jobs.reclaim_stale_leases(now=expired_at, limit=10)
    assert summary.requeued == 1
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.QUEUED.value
    assert job.owner_token is None and job.leased_by is None and job.lease_expires_at is None
    assert job.error_summary == "lease_expired"
    assert run.status == RunStatus.QUEUED.value
    assert [(event.seq, event.type) for event in events] == [
        (1, "run.created"),
        (2, "job.lease_expired"),
    ]

    second_at = expired_at + timedelta(seconds=2)
    second = await runtime.jobs.claim_due_job(
        worker_id="recovery-worker",
        now=second_at,
        lease_duration=timedelta(seconds=30),
    )
    assert second is not None
    assert second.attempt == 2 and second.owner_token != first.owner_token
    assert (
        await runtime.jobs.heartbeat(
            job=first,
            now=second_at,
            lease_duration=timedelta(seconds=30),
        )
        is False
    )
    assert await runtime.jobs.complete(job=first, result=_output(), now=second_at) is False
    assert (
        await runtime.jobs.requeue(
            job=first,
            error_category="provider_timeout",
            now=second_at,
        )
        is False
    )
    assert (
        await runtime.jobs.fail(
            job=first,
            error_category="provider_timeout",
            now=second_at,
        )
        is False
    )
    assert (
        await runtime.jobs.cancel(
            job=first,
            reason="executor_cancelled",
            now=second_at,
        )
        is False
    )
    assert (
        await runtime.jobs.release(
            job=first,
            error_category="worker_shutdown",
            now=second_at,
        )
        is False
    )

    await _prepare(runtime, second, tenant, second_at)
    assert await runtime.jobs.complete(job=second, result=_output(), now=second_at) is True
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value and job.attempt == 2
    assert run.status == RunStatus.COMPLETED.value
    assert [event.seq for event in events] == list(range(1, len(events) + 1))
    assert [event.type for event in events].count("run.completed") == 1


async def test_exhausted_stale_lease_fails_queued_run_without_second_claim(
    runtime: _Runtime,
) -> None:
    _tenant, run_id = await _tenant_and_run(runtime, "worker-exhausted-stale")
    async with transaction(runtime.session_factory) as session:
        await session.execute(update(RunJob).where(RunJob.run_id == run_id).values(max_attempts=1))
    now = datetime.now(UTC)
    claim = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None
    assert (
        await runtime.jobs.release(
            job=claim,
            error_category="worker_shutdown",
            now=now,
        )
        is False
    )

    summary = await runtime.jobs.reclaim_stale_leases(
        now=now + timedelta(seconds=31),
        limit=10,
    )

    assert summary.dead == 1
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DEAD.value
    assert job.error_summary == "job_attempts_exhausted"
    assert run.status == RunStatus.FAILED.value
    assert run.error_category == "job_attempts_exhausted"
    assert [event.type for event in events] == [
        "run.created",
        "job.lease_expired",
        "job.dead",
        "run.failed",
    ]


async def test_membership_revoked_after_claim_cancels_without_execution(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-revoked-after-claim")
    now = datetime.now(UTC)
    claim = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == tenant.workspace_id,
                WorkspaceMembership.user_id == tenant.actor_user_id,
            )
            .values(revoked_at=now)
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.tenants.resolve_tenant(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
        )

    prepared = await runtime.jobs.prepare_claimed_job(
        job=claim,
        resolved_tenant=None,
        now=now,
    )

    assert prepared.disposition == "finished"
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value
    assert run.status == RunStatus.CANCELLED.value
    assert events[-1].type == "run.cancelled"
    assert events[-1].payload["reason"] == "authorization_revoked"


async def test_cancel_between_claim_and_prepare_invalidates_lease_without_execution(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-cancel-before-prepare")
    now = datetime.now(UTC)
    claim = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None

    cancellation = await runtime.runs.cancel_run(tenant=tenant, run_id=run_id)
    prepared = await runtime.jobs.prepare_claimed_job(
        job=claim,
        resolved_tenant=tenant,
        now=now,
    )

    assert cancellation.status is RunStatus.CANCELLED
    assert prepared.disposition == "lease_lost"
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value
    assert job.leased_by is None and job.owner_token is None and job.lease_expires_at is None
    assert run.status == RunStatus.CANCELLED.value
    assert [event.type for event in events] == ["run.created", "run.cancelled"]


async def test_wrong_owner_cannot_use_any_lease_cas_and_retry_does_not_repeat_start_event(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-cas-matrix")
    now = datetime.now(UTC)
    first = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert first is not None
    await _prepare(runtime, first, tenant, now)
    wrong_owner = replace(first, owner_token=uuid4())

    assert (
        await runtime.jobs.heartbeat(
            job=wrong_owner,
            now=now,
            lease_duration=timedelta(seconds=30),
        )
        is False
    )
    assert await runtime.jobs.complete(job=wrong_owner, result=_output(), now=now) is False
    assert (
        await runtime.jobs.requeue(
            job=wrong_owner,
            error_category="provider_timeout",
            now=now,
        )
        is False
    )
    assert (
        await runtime.jobs.fail(
            job=wrong_owner,
            error_category="provider_timeout",
            now=now,
        )
        is False
    )
    assert (
        await runtime.jobs.cancel(
            job=wrong_owner,
            reason="executor_cancelled",
            now=now,
        )
        is False
    )
    assert (
        await runtime.jobs.release(
            job=wrong_owner,
            error_category="worker_shutdown",
            now=now,
        )
        is False
    )

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.LEASED.value and job.owner_token == first.owner_token
    assert run.status == RunStatus.RUNNING.value
    assert [event.type for event in events] == ["run.created", "run.status_changed"]

    assert (
        await runtime.jobs.requeue(
            job=first,
            error_category="provider_timeout",
            now=now,
        )
        is True
    )
    second_now = now + timedelta(seconds=2)
    second = await runtime.jobs.claim_due_job(
        worker_id="worker-b",
        now=second_now,
        lease_duration=timedelta(seconds=30),
    )
    assert second is not None and second.attempt == 2
    await _prepare(runtime, second, tenant, second_now)
    assert (
        await runtime.jobs.fail(
            job=second,
            error_category="provider_timeout",
            now=second_now,
        )
        is True
    )

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DEAD.value and job.error_summary == "provider_timeout"
    assert run.status == RunStatus.FAILED.value and run.error_category == "provider_timeout"
    assert [event.type for event in events] == [
        "run.created",
        "run.status_changed",
        "job.dead",
        "run.failed",
    ]


async def test_waiting_approval_without_resume_identity_fails_closed(runtime: _Runtime) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-waiting-invariant")
    now = datetime.now(UTC)
    claim = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == run_id)
            .values(status=RunStatus.WAITING_APPROVAL.value, started_at=now)
        )

    with pytest.raises(DomainInvariantError, match="missing resume identity"):
        await runtime.jobs.prepare_claimed_job(
            job=claim,
            resolved_tenant=tenant,
            now=now,
        )

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.LEASED.value
    assert run.status == RunStatus.WAITING_APPROVAL.value
    assert job.error_summary is None and run.error_category is None
    assert [event.type for event in events] == ["run.created"]


async def test_stale_job_for_terminal_run_only_converges_job_to_done(runtime: _Runtime) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-terminal-stale")
    cancellation = await runtime.runs.cancel_run(tenant=tenant, run_id=run_id)
    assert cancellation.status is RunStatus.CANCELLED
    now = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(RunJob)
            .where(RunJob.run_id == run_id)
            .values(
                status=JobStatus.LEASED.value,
                attempt=1,
                leased_by="stale-worker",
                owner_token=uuid4(),
                lease_expires_at=now - timedelta(seconds=1),
            )
        )

    summary = await runtime.jobs.reclaim_stale_leases(now=now, limit=10)

    assert summary.finished == 1
    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value
    assert job.leased_by is None and job.owner_token is None and job.lease_expires_at is None
    assert run.status == RunStatus.CANCELLED.value
    assert [event.type for event in events] == ["run.created", "run.cancelled"]


class _BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        self.started.set()
        await self.finish.wait()
        return RunExecutionResult(status=RunStatus.COMPLETED, result=_output())


async def test_executor_wait_does_not_hold_rows_and_cancel_converges_at_finalize(
    runtime: _Runtime,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-no-long-transaction")
    executor = _BlockingExecutor()
    runner = WorkerRunner(
        worker_id="worker-blocking",
        store=runtime.jobs,
        tenant_service=runtime.tenants,
        executor=executor,
        settings=WorkerRuntimeSettings(),
    )
    worker_task = asyncio.create_task(runner.run_once(asyncio.Event()))
    await asyncio.wait_for(executor.started.wait(), timeout=1)

    cancellation = await asyncio.wait_for(
        runtime.runs.cancel_run(tenant=tenant, run_id=run_id),
        timeout=1,
    )
    assert cancellation.status is RunStatus.RUNNING
    executor.finish.set()
    assert await asyncio.wait_for(worker_task, timeout=1) is True

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.DONE.value
    assert run.status == RunStatus.CANCELLED.value
    assert [event.type for event in events] == [
        "run.created",
        "run.status_changed",
        "run.cancelled",
    ]


async def test_event_insert_failure_rolls_back_dead_job_and_failed_run(
    runtime: _Runtime,
    migrated_database_url: str,
) -> None:
    tenant, run_id = await _tenant_and_run(runtime, "worker-event-rollback")
    now = datetime.now(UTC)
    claim = await runtime.jobs.claim_due_job(
        worker_id="worker-a",
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None
    await _prepare(runtime, claim, tenant, now)
    with connect_database(migrated_database_url) as connection:
        connection.execute(
            """
            CREATE FUNCTION fail_worker_event_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected worker event failure';
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER fail_worker_event_insert_trigger
            BEFORE INSERT ON run_events
            FOR EACH ROW EXECUTE FUNCTION fail_worker_event_insert()
            """
        )

    with pytest.raises(DBAPIError):
        await runtime.jobs.fail(
            job=claim,
            error_category="executor_unhandled_error",
            now=now,
        )

    job, run, events = await _job_and_events(runtime, run_id)
    assert job.status == JobStatus.LEASED.value and job.owner_token == claim.owner_token
    assert run.status == RunStatus.RUNNING.value and run.error_category is None
    assert [event.type for event in events] == ["run.created", "run.status_changed"]


async def test_worker_startup_requires_exact_database_revision(database_url: str) -> None:
    with pytest.raises(RuntimeError, match="required revision"):
        await run_worker(Settings(database_url=database_url))
