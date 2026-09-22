from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.db.actions import SqlAlchemyActionStore
from app.db.approval_expiry import SqlAlchemyApprovalRequestExpirySweeper
from app.db.models import ApprovalRequest, Document, Run, RunEvent, RunJob, ToolInvocation
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.actions import ACTION_KEY, PrepareActionCommand, SubmitApplicationArgsV1
from app.domain.approvals import ApprovalStatus
from app.domain.errors import DomainInvariantError
from app.domain.jobs import ClaimedJob, JobStatus
from app.domain.provisioning import ProvisioningService
from app.domain.runs import RunMode, RunService, RunStatus
from app.domain.tenancy import TenantContext, TenantService
from tests.legacy_runtime import SqlAlchemyRunStore, SqlAlchemyWorkerJobStore

pytestmark = pytest.mark.integration

_CREATED = datetime(2030, 1, 1, 12, tzinfo=UTC)
_EXPIRES = _CREATED + timedelta(hours=1)


@dataclass(frozen=True, slots=True)
class _Paused:
    engine: object
    sessions: object
    tenant: TenantContext
    run_id: UUID
    request_id: UUID
    action_id: UUID
    claim: ClaimedJob


async def _paused(
    database_url: str,
    suffix: str,
    *,
    finalize: bool = True,
    lease_duration: timedelta = timedelta(minutes=10),
) -> _Paused:
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(f"gate6-expiry-{suffix}")
    tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
    )
    document_id = uuid4()
    content = f"Synthetic resume {suffix}"
    async with sessions.begin() as session:
        session.add(
            Document(
                id=document_id,
                workspace_id=identity.workspace_id,
                created_by_user_id=identity.user_id,
                title="Synthetic resume",
                source_type="text",
                source_name=f"{suffix}.txt",
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                normalization_version="text-normalization-v1",
                chunking_version="document-chunking-v1",
                embedding_model="qwen-beijing-text-embedding-v4-1536-v1",
            )
        )
    run_id = (
        await RunService(SqlAlchemyRunStore(sessions)).create_run(
            tenant=tenant,
            mode=RunMode.APPLICATION,
            query="Prepare a synthetic application",
            resume_document_id=document_id,
        )
    ).run_id
    jobs = SqlAlchemyWorkerJobStore(sessions, lambda _attempt: timedelta(0))
    claim = await jobs.claim_due_job(
        worker_id="gate6-worker",
        now=_CREATED,
        lease_duration=lease_duration,
    )
    assert claim is not None
    prepared_claim = await jobs.prepare_claimed_job(
        job=claim,
        resolved_tenant=tenant,
        now=_CREATED,
    )
    assert prepared_claim.disposition == "execute"
    action_id = uuid4()
    prepared = await SqlAlchemyActionStore(sessions).prepare_action(
        PrepareActionCommand(
            tenant=tenant,
            run_id=run_id,
            action_proposal_id=action_id,
            action_key=ACTION_KEY,
            action_revision=1,
            args=SubmitApplicationArgsV1(
                job_ref=f"pathfinder-mock-job-v1:{run_id}",
                resume_document_id=document_id,
                answers={},
                cover_letter="Synthetic cover letter",
            ),
            now=_CREATED,
            expires_at=_EXPIRES,
        )
    )
    if finalize:
        assert await jobs.wait_for_approval(
            job=claim,
            approval_request_id=prepared.approval_request.request_id,
            now=_CREATED + timedelta(minutes=1),
        )
    return _Paused(
        engine=engine,
        sessions=sessions,
        tenant=tenant,
        run_id=run_id,
        request_id=prepared.approval_request.request_id,
        action_id=action_id,
        claim=claim,
    )


async def _close(paused: _Paused) -> None:
    await paused.engine.dispose()


async def test_waiting_transaction_releases_lease_and_emits_safe_event(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "waiting")
    try:
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            event = await session.scalar(
                select(RunEvent)
                .where(RunEvent.run_id == paused.run_id)
                .order_by(RunEvent.seq.desc())
                .limit(1)
            )
        assert job is not None and run is not None and event is not None
        assert job.status == JobStatus.DONE.value
        assert job.leased_by is None and job.owner_token is None and job.lease_expires_at is None
        assert run.status == RunStatus.WAITING_APPROVAL.value
        assert event.type == "run.status_changed"
        assert event.payload == {
            "previous_status": "running",
            "status": "waiting_approval",
            "reason": "approval_required",
            "approval_request_id": str(paused.request_id),
            "action_intent_id": str(paused.action_id),
        }
        assert all(term not in str(event.payload) for term in ("cover", "answers", "resume"))
    finally:
        await _close(paused)


async def test_initial_pause_after_ttl_converges_through_sweeper(
    migrated_database_url: str,
) -> None:
    paused = await _paused(
        migrated_database_url,
        "initial-pause-after-ttl",
        finalize=False,
        lease_duration=timedelta(hours=2),
    )
    now = _EXPIRES + timedelta(seconds=1)
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        assert await jobs.wait_for_approval(
            job=paused.claim,
            approval_request_id=paused.request_id,
            now=now,
        )
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            event = await session.scalar(
                select(RunEvent)
                .where(RunEvent.run_id == paused.run_id)
                .order_by(RunEvent.seq.desc())
                .limit(1)
            )
        assert request is not None and job is not None and run is not None and event is not None
        assert request.status == ApprovalStatus.PENDING.value and request.version == 1
        assert job.status == JobStatus.DONE.value
        assert job.leased_by is None and job.owner_token is None and job.lease_expires_at is None
        assert run.status == RunStatus.WAITING_APPROVAL.value
        assert event.payload["reason"] == "approval_required"

        summary = await SqlAlchemyApprovalRequestExpirySweeper(
            paused.sessions
        ).sweep_due_approval_requests(now=now, limit=10)
        assert summary.expired == summary.requeued == 1
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            expiry_events = list(
                await session.scalars(
                    select(RunEvent).where(
                        RunEvent.run_id == paused.run_id,
                        RunEvent.type == "approval.expired",
                    )
                )
            )
        assert request is not None and job is not None and run is not None
        assert request.status == ApprovalStatus.EXPIRED.value and request.version == 2
        assert job.status == JobStatus.QUEUED.value and job.attempt == 0
        assert job.resume_approval_request_id == paused.request_id
        assert run.status == RunStatus.WAITING_APPROVAL.value
        assert len(expiry_events) == 1
        assert expiry_events[0].payload["reason"] == "natural_expiry"
    finally:
        await _close(paused)


async def test_wrong_owner_waiting_finalize_returns_false(migrated_database_url: str) -> None:
    paused = await _paused(migrated_database_url, "wrong-owner", finalize=False)
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        assert not await jobs.wait_for_approval(
            job=replace(paused.claim, owner_token=uuid4()),
            approval_request_id=paused.request_id,
            now=_CREATED + timedelta(minutes=2),
        )
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
        assert job is not None and run is not None
        assert job.status == JobStatus.LEASED.value
        assert job.owner_token == paused.claim.owner_token
        assert run.status == RunStatus.RUNNING.value
    finally:
        await _close(paused)


async def test_waiting_event_failure_rolls_back_run_job_and_sequence(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "waiting-rollback", finalize=False)
    try:
        async with paused.sessions.begin() as session:
            before_seq = await session.scalar(
                select(Run.next_event_seq).where(Run.id == paused.run_id)
            )
            before_count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
            await session.execute(
                text(
                    "CREATE FUNCTION reject_waiting_event() RETURNS trigger LANGUAGE plpgsql AS $$ "
                    "BEGIN IF NEW.type = 'run.status_changed' AND "
                    "NEW.payload ->> 'reason' = 'approval_required' "
                    "THEN RAISE EXCEPTION 'injected'; END IF; RETURN NEW; END $$"
                )
            )
            await session.execute(
                text(
                    "CREATE TRIGGER reject_waiting_event BEFORE INSERT ON run_events "
                    "FOR EACH ROW EXECUTE FUNCTION reject_waiting_event()"
                )
            )
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        with pytest.raises(Exception, match="injected"):
            await jobs.wait_for_approval(
                job=paused.claim,
                approval_request_id=paused.request_id,
                now=_CREATED + timedelta(minutes=1),
            )
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
        assert job is not None and run is not None
        assert job.status == JobStatus.LEASED.value
        assert job.owner_token == paused.claim.owner_token
        assert job.lease_expires_at == paused.claim.lease_expires_at
        assert run.status == RunStatus.RUNNING.value
        assert run.next_event_seq == before_seq and count == before_count
    finally:
        await _close(paused)


async def test_legal_resume_claim_and_stale_recovery_preserve_request(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "resume-stale")
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        async with paused.sessions.begin() as session:
            job = await session.scalar(
                select(RunJob).where(RunJob.run_id == paused.run_id).with_for_update()
            )
            assert job is not None
            job.status = JobStatus.QUEUED.value
            job.attempt = 0
            job.available_at = _EXPIRES
            job.resume_approval_request_id = paused.request_id
        claim = await jobs.claim_due_job(
            worker_id="resume-worker",
            now=_EXPIRES,
            lease_duration=timedelta(minutes=1),
        )
        assert claim is not None and claim.resume_approval_request_id == paused.request_id
        summary = await jobs.reclaim_stale_leases(
            now=_EXPIRES + timedelta(minutes=2),
            limit=10,
        )
        assert summary.requeued == 1
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
        assert job is not None and run is not None
        assert job.status == JobStatus.QUEUED.value
        assert job.resume_approval_request_id == paused.request_id
        assert run.status == RunStatus.WAITING_APPROVAL.value

        second = await jobs.claim_due_job(
            worker_id="resume-worker",
            now=_EXPIRES + timedelta(minutes=2),
            lease_duration=timedelta(minutes=1),
        )
        assert second is not None
        prepared = await jobs.prepare_claimed_job(
            job=second,
            resolved_tenant=paused.tenant,
            now=_EXPIRES + timedelta(minutes=2),
        )
        assert prepared.disposition == "execute"
        async with paused.sessions() as session:
            run = await session.get(Run, paused.run_id)
        assert run is not None and run.status == RunStatus.RUNNING.value
    finally:
        await _close(paused)


@pytest.mark.parametrize("status", [ApprovalStatus.CONSUMED])
async def test_gate6_resume_rejects_consumption_state(
    migrated_database_url: str,
    status: ApprovalStatus,
) -> None:
    paused = await _paused(migrated_database_url, f"invalid-resume-{status.value}")
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        resume_at = _CREATED + timedelta(minutes=2)
        async with paused.sessions.begin() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(
                select(RunJob).where(RunJob.run_id == paused.run_id).with_for_update()
            )
            assert request is not None and job is not None
            request.status = status.value
            request.consumed_at = resume_at if status is ApprovalStatus.CONSUMED else None
            job.status = JobStatus.QUEUED.value
            job.attempt = 0
            job.available_at = resume_at
            job.resume_approval_request_id = paused.request_id
        claim = await jobs.claim_due_job(
            worker_id="invalid-resume-worker",
            now=resume_at,
            lease_duration=timedelta(minutes=5),
        )
        assert claim is not None
        with pytest.raises(DomainInvariantError, match="resume approval request status"):
            await jobs.prepare_claimed_job(
                job=claim,
                resolved_tenant=paused.tenant,
                now=resume_at,
            )
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
        assert job is not None and run is not None
        assert job.status == JobStatus.LEASED.value and job.owner_token == claim.owner_token
        assert run.status == RunStatus.WAITING_APPROVAL.value
    finally:
        await _close(paused)


@pytest.mark.parametrize("initial_status", [ApprovalStatus.PENDING, ApprovalStatus.APPROVED])
async def test_due_request_expires_and_done_job_requeues_once(
    migrated_database_url: str,
    initial_status: ApprovalStatus,
) -> None:
    paused = await _paused(migrated_database_url, f"due-{initial_status.value}")
    try:
        async with paused.sessions.begin() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            assert request is not None
            request.status = initial_status.value
        sweeper = SqlAlchemyApprovalRequestExpirySweeper(paused.sessions)
        first, second = await asyncio.gather(
            sweeper.sweep_due_approval_requests(now=_EXPIRES, limit=10),
            sweeper.sweep_due_approval_requests(now=_EXPIRES, limit=10),
        )
        assert first.expired + second.expired == 1
        assert first.requeued + second.requeued == 1
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            events = list(
                await session.scalars(
                    select(RunEvent).where(
                        RunEvent.run_id == paused.run_id,
                        RunEvent.type == "approval.expired",
                    )
                )
            )
        assert request is not None and job is not None
        assert request.status == ApprovalStatus.EXPIRED.value and request.version == 2
        assert job.status == JobStatus.QUEUED.value and job.attempt == 0
        assert job.resume_approval_request_id == paused.request_id
        assert len(events) == 1 and events[0].payload["reason"] == "natural_expiry"
    finally:
        await _close(paused)


async def test_natural_expiry_claim_resumes_and_repauses_without_authorizing_effect(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "expiry-resume-repause")
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        summary = await SqlAlchemyApprovalRequestExpirySweeper(
            paused.sessions
        ).sweep_due_approval_requests(now=_EXPIRES, limit=10)
        assert summary.expired == summary.requeued == 1
        claim = await jobs.claim_due_job(
            worker_id="expiry-resume-worker",
            now=_EXPIRES,
            lease_duration=timedelta(minutes=5),
        )
        assert claim is not None and claim.resume_approval_request_id == paused.request_id
        prepared = await jobs.prepare_claimed_job(
            job=claim,
            resolved_tenant=paused.tenant,
            now=_EXPIRES,
        )
        assert prepared.disposition == "execute"
        assert await jobs.wait_for_approval(
            job=claim,
            approval_request_id=paused.request_id,
            now=_EXPIRES,
        )

        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            events = list(
                await session.scalars(
                    select(RunEvent).where(RunEvent.run_id == paused.run_id).order_by(RunEvent.seq)
                )
            )
            tool_count = await session.scalar(
                select(func.count())
                .select_from(ToolInvocation)
                .where(ToolInvocation.run_id == paused.run_id)
            )
        assert request is not None and job is not None and run is not None
        assert request.status == ApprovalStatus.EXPIRED.value and request.version == 2
        assert job.status == JobStatus.DONE.value
        assert job.leased_by is None and job.owner_token is None and job.lease_expires_at is None
        assert job.resume_approval_request_id == paused.request_id
        assert run.status == RunStatus.WAITING_APPROVAL.value
        reasons = [event.payload.get("reason") for event in events]
        assert reasons.count("natural_expiry") == 1
        assert reasons[-2:] == ["approval_resume_claimed", "approval_resume_paused"]
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert all(
            term not in str(event.payload).lower()
            for event in events
            for term in ("cover letter", "answers", "resume text", "credential", "secret")
        )
        assert tool_count == 0
    finally:
        await _close(paused)


async def test_expiry_preserves_queued_and_leased_job_ownership(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "leased")
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        async with paused.sessions.begin() as session:
            job = await session.scalar(
                select(RunJob).where(RunJob.run_id == paused.run_id).with_for_update()
            )
            assert job is not None
            job.status = JobStatus.QUEUED.value
            job.resume_approval_request_id = paused.request_id
            job.available_at = _EXPIRES
        claim = await jobs.claim_due_job(
            worker_id="leased-worker",
            now=_EXPIRES,
            lease_duration=timedelta(minutes=5),
        )
        assert claim is not None
        await SqlAlchemyApprovalRequestExpirySweeper(paused.sessions).sweep_due_approval_requests(
            now=_EXPIRES, limit=10
        )
        async with paused.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
        assert job is not None
        assert job.status == JobStatus.LEASED.value
        assert job.leased_by == "leased-worker"
        assert job.owner_token == claim.owner_token
        assert job.lease_expires_at == claim.lease_expires_at
        assert job.resume_approval_request_id == paused.request_id

        prepared = await jobs.prepare_claimed_job(
            job=claim,
            resolved_tenant=paused.tenant,
            now=_EXPIRES,
        )
        assert prepared.disposition == "execute"
        assert await jobs.wait_for_approval(
            job=claim,
            approval_request_id=paused.request_id,
            now=_EXPIRES,
        )
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            tool_count = await session.scalar(
                select(func.count())
                .select_from(ToolInvocation)
                .where(ToolInvocation.run_id == paused.run_id)
            )
        assert request is not None and job is not None and run is not None
        assert request.status == ApprovalStatus.EXPIRED.value
        assert job.status == JobStatus.DONE.value and job.owner_token is None
        assert job.resume_approval_request_id == paused.request_id
        assert run.status == RunStatus.WAITING_APPROVAL.value
        assert tool_count == 0
    finally:
        await _close(paused)


async def test_resume_pause_event_failure_rolls_back_run_job_request_and_sequence(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "resume-pause-rollback")
    try:
        jobs = SqlAlchemyWorkerJobStore(paused.sessions, lambda _attempt: timedelta(0))
        resume_at = _CREATED + timedelta(minutes=2)
        async with paused.sessions.begin() as session:
            job = await session.scalar(
                select(RunJob).where(RunJob.run_id == paused.run_id).with_for_update()
            )
            assert job is not None
            job.status = JobStatus.QUEUED.value
            job.attempt = 0
            job.available_at = resume_at
            job.resume_approval_request_id = paused.request_id
        claim = await jobs.claim_due_job(
            worker_id="resume-rollback-worker",
            now=resume_at,
            lease_duration=timedelta(minutes=5),
        )
        assert claim is not None
        prepared = await jobs.prepare_claimed_job(
            job=claim,
            resolved_tenant=paused.tenant,
            now=resume_at,
        )
        assert prepared.disposition == "execute"
        async with paused.sessions.begin() as session:
            before_seq = await session.scalar(
                select(Run.next_event_seq).where(Run.id == paused.run_id)
            )
            before_count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
            await session.execute(
                text(
                    "CREATE FUNCTION reject_resume_pause_event() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN IF NEW.type = 'run.status_changed' AND "
                    "NEW.payload ->> 'reason' = 'approval_resume_paused' "
                    "THEN RAISE EXCEPTION 'injected'; END IF; RETURN NEW; END $$"
                )
            )
            await session.execute(
                text(
                    "CREATE TRIGGER reject_resume_pause_event BEFORE INSERT ON run_events "
                    "FOR EACH ROW EXECUTE FUNCTION reject_resume_pause_event()"
                )
            )
        with pytest.raises(Exception, match="injected"):
            await jobs.wait_for_approval(
                job=claim,
                approval_request_id=paused.request_id,
                now=resume_at,
            )
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
        assert request is not None and job is not None and run is not None
        assert request.status == ApprovalStatus.PENDING.value and request.version == 1
        assert job.status == JobStatus.LEASED.value
        assert job.owner_token == claim.owner_token
        assert job.lease_expires_at == claim.lease_expires_at
        assert job.resume_approval_request_id == paused.request_id
        assert run.status == RunStatus.RUNNING.value
        assert run.next_event_seq == before_seq and count == before_count
    finally:
        await _close(paused)


async def test_expiry_event_failure_rolls_back_request_job_and_sequence(
    migrated_database_url: str,
) -> None:
    paused = await _paused(migrated_database_url, "rollback")
    try:
        async with paused.sessions.begin() as session:
            before_seq = await session.scalar(
                select(Run.next_event_seq).where(Run.id == paused.run_id)
            )
            before_count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
            await session.execute(
                text(
                    "CREATE FUNCTION reject_expiry_event() RETURNS trigger LANGUAGE plpgsql AS $$ "
                    "BEGIN IF NEW.type = 'approval.expired' THEN RAISE EXCEPTION 'injected'; "
                    "END IF; RETURN NEW; END $$"
                )
            )
            await session.execute(
                text(
                    "CREATE TRIGGER reject_expiry_event BEFORE INSERT ON run_events "
                    "FOR EACH ROW EXECUTE FUNCTION reject_expiry_event()"
                )
            )
        with pytest.raises(Exception, match="injected"):
            await SqlAlchemyApprovalRequestExpirySweeper(
                paused.sessions
            ).sweep_due_approval_requests(now=_EXPIRES, limit=10)
        async with paused.sessions() as session:
            request = await session.get(ApprovalRequest, paused.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == paused.run_id))
            run = await session.get(Run, paused.run_id)
            count = await session.scalar(
                select(func.count()).select_from(RunEvent).where(RunEvent.run_id == paused.run_id)
            )
        assert request is not None and job is not None and run is not None
        assert request.status == ApprovalStatus.PENDING.value and request.version == 1
        assert job.status == JobStatus.DONE.value and job.resume_approval_request_id is None
        assert run.status == RunStatus.WAITING_APPROVAL.value
        assert run.next_event_seq == before_seq and count == before_count
    finally:
        await _close(paused)
