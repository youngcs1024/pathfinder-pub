from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.actions import SqlAlchemyActionStore
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Document,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.runs import SqlAlchemyRunStore
from app.db.session import AsyncSessionFactory, create_database_engine, create_session_factory
from app.domain.action_execution import ActionExecutionIdentity, ConfirmedActionResult
from app.domain.actions import ACTION_KEY, PrepareActionCommand, SubmitApplicationArgsV1
from app.domain.approvals import ApprovalDecisionCommand
from app.domain.errors import DomainInvariantError, DomainNotFoundError
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.runs import RunMode, RunService
from app.domain.tenancy import TenantContext

pytestmark = pytest.mark.integration
NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)


@dataclass(frozen=True)
class _Runtime:
    sessions: AsyncSessionFactory
    tenant: TenantContext
    run_id: UUID
    document_id: UUID


async def _approved(database_url: str, suffix: str):
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(f"gate6-execution-{suffix}")
    tenant = TenantContext(identity.workspace_id, identity.user_id, WorkspaceRole.ADMIN)
    document_id = uuid4()
    content = f"execution resume {suffix}"
    async with sessions.begin() as session:
        session.add(
            Document(
                id=document_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                title="Resume",
                source_type="text",
                source_name="resume.txt",
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                normalization_version="text-normalization-v1",
                chunking_version="document-chunking-v1",
                embedding_model="qwen-beijing-text-embedding-v4-1536-v1",
            )
        )
    accepted = await RunService(SqlAlchemyRunStore(sessions)).create_run(
        tenant=tenant,
        mode=RunMode.APPLICATION,
        query="prepare application",
        resume_document_id=document_id,
    )
    async with sessions.begin() as session:
        run = await session.get(Run, accepted.run_id)
        assert run is not None
        run.status = "running"
        run.started_at = NOW
    prepared = await SqlAlchemyActionStore(sessions).prepare_action(
        PrepareActionCommand(
            tenant=tenant,
            run_id=accepted.run_id,
            action_proposal_id=uuid4(),
            action_key=ACTION_KEY,
            action_revision=1,
            args=SubmitApplicationArgsV1(
                job_ref="persisted-job",
                resume_document_id=document_id,
                answers={"availability": "two weeks"},
                cover_letter="Exact persisted draft",
            ),
            now=NOW,
            expires_at=NOW + timedelta(hours=1),
        )
    )
    async with sessions.begin() as session:
        run = await session.get(Run, accepted.run_id)
        job = await session.scalar(select(RunJob).where(RunJob.run_id == accepted.run_id))
        assert run is not None and job is not None
        run.status = "waiting_approval"
        job.status = "done"
    await SqlAlchemyApprovalStore(sessions).decide(
        ApprovalDecisionCommand(
            tenant=tenant,
            action_intent_id=prepared.intent.action_intent_id,
            decision="approve",
            expected_version=1,
            reason="approved",
            now=NOW + timedelta(minutes=1),
        )
    )
    async with sessions.begin() as session:
        run = await session.get(Run, accepted.run_id)
        assert run is not None
        run.status = "running"
    runtime = _Runtime(sessions, tenant, accepted.run_id, document_id)
    execution_id = ActionExecutionIdentity(
        tenant.workspace_id,
        accepted.run_id,
        prepared.intent.action_intent_id,
        prepared.approval_request.request_id,
    )
    return engine, runtime, execution_id


async def test_prepare_is_atomic_idempotent_and_begin_send_is_a_second_transaction(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "normal")
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        first = await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        second = await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        assert second == first
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, identity.approval_request_id)
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocations = list(
                await session.scalars(
                    select(ToolInvocation).where(
                        ToolInvocation.action_intent_id == identity.action_intent_id
                    )
                )
            )
            assert request is not None and request.status == "consumed"
            assert request.consumed_at == NOW + timedelta(minutes=2)
            assert action is not None and action.status == "authorized"
            assert len(invocations) == 1 and invocations[0].status == "prepared"
            assert invocations[0].attempt == 0

        send = await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        assert send.allowed is True
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "executing"
            assert invocation is not None and (invocation.status, invocation.attempt) == (
                "executing",
                1,
            )
        await store.confirm_success(
            identity,
            result=ConfirmedActionResult("mock-submission:confirmed"),
            latency_ms=7,
            now=NOW + timedelta(minutes=4),
        )
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "succeeded"
            assert action.result == {"external_ref": "mock-submission:confirmed"}
            assert invocation is not None and invocation.status == "succeeded"
    finally:
        await engine.dispose()


@pytest.mark.parametrize("action_status", ["authorized", "executing", "succeeded"])
async def test_consumed_resume_preserves_original_execution(
    migrated_database_url: str, action_status: str
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "resume-" + action_status)
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        prepared = await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        if action_status != "authorized":
            assert (await store.begin_send(identity, now=NOW + timedelta(minutes=3))).allowed
        if action_status == "succeeded":
            await store.confirm_success(
                identity,
                result=ConfirmedActionResult("mock-submission:resume"),
                latency_ms=7,
                now=NOW + timedelta(minutes=4),
            )
        jobs = SqlAlchemyWorkerJobStore(runtime.sessions, retry_delay=lambda _attempt: timedelta(0))
        now = NOW + timedelta(minutes=5)
        claim = await jobs.claim_due_job(
            worker_id="resume-worker", now=now, lease_duration=timedelta(seconds=30)
        )
        assert (
            claim is not None and claim.resume_approval_request_id == identity.approval_request_id
        )
        result = await jobs.prepare_claimed_job(job=claim, resolved_tenant=runtime.tenant, now=now)
        assert result.disposition == "execute" and result.tenant == runtime.tenant
        recovered = await store.prepare_execution(identity, now=now)
        assert recovered.invocation_id == prepared.invocation_id
        assert recovered.idempotency_key == prepared.idempotency_key
        assert recovered.args == prepared.args and recovered.target == prepared.target
        assert recovered.status == action_status
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, identity.approval_request_id)
            assert request.status == "consumed"
            assert request.consumed_at == NOW + timedelta(minutes=2)
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    "conflict", ["binding", "cross_run", "missing_invocation", "status", "first_resume"]
)
async def test_consumed_resume_rejects_conflicting_persisted_facts(
    migrated_database_url: str, conflict: str
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "resume-bad-" + conflict)
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        if conflict != "missing_invocation":
            await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        async with runtime.sessions.begin() as session:
            request = await session.get(ApprovalRequest, identity.approval_request_id)
            intent = await session.get(ActionIntent, identity.action_intent_id)
            if conflict == "binding":
                request.approval_binding_digest = "sha256:" + "0" * 64
            elif conflict == "missing_invocation":
                request.status = "consumed"
                request.consumed_at = NOW + timedelta(minutes=2)
                intent.status = "authorized"
            elif conflict == "status":
                intent.status = "executing"
            elif conflict == "first_resume":
                run = await session.get(Run, runtime.run_id)
                run.status = "waiting_approval"
        jobs = SqlAlchemyWorkerJobStore(runtime.sessions, retry_delay=lambda _attempt: timedelta(0))
        now = NOW + timedelta(minutes=5)
        claim = await jobs.claim_due_job(
            worker_id="resume-worker", now=now, lease_duration=timedelta(seconds=30)
        )
        assert claim is not None
        with pytest.raises(DomainInvariantError):
            if conflict == "cross_run":
                async with runtime.sessions() as session:
                    await jobs._require_resume_request(
                        session, job=replace(claim, run_id=uuid4()), allow_consumed=True
                    )
            else:
                await jobs.prepare_claimed_job(job=claim, resolved_tenant=runtime.tenant, now=now)
        async with runtime.sessions() as session:
            run = await session.get(Run, runtime.run_id)
            job = await session.get(RunJob, claim.job_id)
            assert run.status == ("waiting_approval" if conflict == "first_resume" else "running")
            assert job.status == "leased" and job.owner_token == claim.owner_token
    finally:
        await engine.dispose()


async def test_consume_crash_rolls_back_and_retry_reuses_one_invocation(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "consume-crash")

    async def fail(point: str) -> None:
        if point == "consume_before_commit":
            raise RuntimeError("crash")

    try:
        with pytest.raises(RuntimeError, match="crash"):
            await SqlAlchemyActionExecutionStore(
                runtime.sessions, fault_injector=fail
            ).prepare_execution(identity, now=NOW + timedelta(minutes=2))
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, identity.approval_request_id)
            action = await session.get(ActionIntent, identity.action_intent_id)
            count = await session.scalar(select(func.count()).select_from(ToolInvocation))
            assert request is not None and request.status == "approved"
            assert action is not None and action.status == "proposed"
            assert count == 0
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        first = await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        second = await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        assert first.invocation_id == second.invocation_id
    finally:
        await engine.dispose()


async def test_cancellation_wins_before_send_and_converges_without_executing(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "cancel")
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        cancelled = await RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
            tenant=runtime.tenant, run_id=runtime.run_id
        )
        assert cancelled.cancel_requested_at is not None
        send = await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        assert send.allowed is False
        assert send.error_category == "cancelled_before_send"
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "cancelled"
            assert invocation is not None and invocation.status == "failed"
            assert invocation.error_category == "cancelled_before_send"
    finally:
        await engine.dispose()


async def test_send_cas_wins_race_and_later_cancellation_cannot_rewrite_action(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "send-wins")
    transition_entered = asyncio.Event()
    release_transition = asyncio.Event()

    async def barrier(point: str) -> None:
        if point == "send_transition_before_commit":
            transition_entered.set()
            await release_transition.wait()

    try:
        await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
            identity, now=NOW + timedelta(minutes=2)
        )
        send_task = asyncio.create_task(
            SqlAlchemyActionExecutionStore(runtime.sessions, fault_injector=barrier).begin_send(
                identity, now=NOW + timedelta(minutes=3)
            )
        )
        await transition_entered.wait()
        cancel_task = asyncio.create_task(
            RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
                tenant=runtime.tenant, run_id=runtime.run_id
            )
        )
        release_transition.set()
        send, cancellation = await asyncio.gather(send_task, cancel_task)
        assert send.allowed is True
        assert cancellation.cancel_requested_at is not None
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "executing"
            assert invocation is not None and invocation.status == "executing"
            assert invocation.error_category is None
    finally:
        await engine.dispose()


async def test_send_transition_crash_rolls_back_to_authorized_and_prepared(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "send-crash")

    async def fail(point: str) -> None:
        if point == "send_transition_before_commit":
            raise RuntimeError("crash")

    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        with pytest.raises(RuntimeError, match="crash"):
            await SqlAlchemyActionExecutionStore(runtime.sessions, fault_injector=fail).begin_send(
                identity, now=NOW + timedelta(minutes=3)
            )
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "authorized"
            assert invocation is not None and invocation.status == "prepared"
        assert (await store.begin_send(identity, now=NOW + timedelta(minutes=3))).allowed
    finally:
        await engine.dispose()


async def test_exact_binding_tamper_fails_closed_before_invocation(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "tamper")
    try:
        async with runtime.sessions.begin() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            assert action is not None
            action.tool_name = "tampered_tool"
        with pytest.raises(DomainInvariantError, match="binding"):
            await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
                identity, now=NOW + timedelta(minutes=2)
            )
        async with runtime.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 0
    finally:
        await engine.dispose()


async def test_database_rejects_unbound_or_duplicate_irreversible_invocations(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "constraints")
    try:
        with pytest.raises(IntegrityError):
            async with runtime.sessions.begin() as session:
                session.add(
                    ToolInvocation(
                        id=uuid4(),
                        workspace_id=identity.workspace_id,
                        originating_actor_user_id=runtime.tenant.actor_user_id,
                        run_id=identity.run_id,
                        action_intent_id=None,
                        tool_name="submit_mock_application",
                        effect="irreversible",
                        args_digest=f"sha256:{'a' * 64}",
                        status="prepared",
                        attempt=0,
                    )
                )
        await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
            identity, now=NOW + timedelta(minutes=2)
        )
        with pytest.raises(IntegrityError):
            async with runtime.sessions.begin() as session:
                session.add(
                    ToolInvocation(
                        id=uuid4(),
                        workspace_id=identity.workspace_id,
                        originating_actor_user_id=runtime.tenant.actor_user_id,
                        run_id=identity.run_id,
                        action_intent_id=identity.action_intent_id,
                        tool_name="submit_mock_application",
                        effect="irreversible",
                        args_digest=f"sha256:{'a' * 64}",
                        status="prepared",
                        attempt=0,
                    )
                )
    finally:
        await engine.dispose()


async def test_originating_membership_revocation_expires_and_cancels_before_send(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "revoked")
    try:
        async with runtime.sessions.begin() as session:
            membership = await session.scalar(
                select(WorkspaceMembership).where(
                    WorkspaceMembership.workspace_id == identity.workspace_id,
                    WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                )
            )
            assert membership is not None
            membership.revoked_at = NOW + timedelta(minutes=2)
        with pytest.raises(DomainNotFoundError):
            await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
                identity, now=NOW + timedelta(minutes=3)
            )
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, identity.approval_request_id)
            action = await session.get(ActionIntent, identity.action_intent_id)
            assert request is not None and request.status == "expired"
            assert action is not None and action.status == "cancelled"
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 0
    finally:
        await engine.dispose()


async def test_reconciliation_attempt_is_durable_before_network_and_exhaustion_is_unknown(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "recovery-budget")
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        first = await store.begin_reconciliation(
            identity, max_attempts=1, now=NOW + timedelta(minutes=4)
        )
        assert first.network_allowed is True
        assert first.recovery_attempt == 1
        exhausted = await store.begin_reconciliation(
            identity, max_attempts=1, now=NOW + timedelta(minutes=5)
        )
        assert exhausted.network_allowed is False
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            events = list(
                await session.scalars(
                    select(RunEvent)
                    .where(RunEvent.run_id == identity.run_id)
                    .order_by(RunEvent.seq)
                )
            )
            assert action is not None and action.status == "outcome_unknown"
            assert action.recovery_attempts == 1
            assert invocation is not None and invocation.status == "outcome_unknown"
            assert invocation.error_category == "external_outcome_unknown"
            assert [event.type for event in events].count("action.started") == 1
            assert [event.type for event in events].count("action.outcome_unknown") == 1
    finally:
        await engine.dispose()


async def test_explicit_absence_resend_cas_increments_same_invocation_attempt(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "recovery-resend")
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        recovery = await store.begin_reconciliation(
            identity, max_attempts=3, now=NOW + timedelta(minutes=4)
        )
        resend = await store.authorize_resend(
            identity,
            expected_recovery_attempt=recovery.recovery_attempt,
            now=NOW + timedelta(minutes=5),
        )
        assert resend.allowed is True
        async with runtime.sessions() as session:
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert invocation is not None and invocation.attempt == 2
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 1
    finally:
        await engine.dispose()


async def test_confirmed_absence_plus_cancellation_prevents_recovery_resend(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "recovery-cancel")
    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        recovery = await store.begin_reconciliation(
            identity, max_attempts=3, now=NOW + timedelta(minutes=4)
        )
        await RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
            tenant=runtime.tenant, run_id=runtime.run_id
        )
        resend = await store.authorize_resend(
            identity,
            expected_recovery_attempt=recovery.recovery_attempt,
            now=NOW + timedelta(minutes=5),
        )
        assert resend.allowed is False
        assert resend.error_category == "cancelled_confirmed_absent"
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "failed"
            assert action.evidence is not None
            assert action.evidence["classification"] == "confirmed_absent"
            assert invocation is not None and invocation.attempt == 1
            assert invocation.status == "failed"
    finally:
        await engine.dispose()


async def test_stale_job_attempt_exhaustion_yields_to_executing_action_recovery_budget(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "stale-recovery")
    stale_at = NOW + timedelta(minutes=4)
    try:
        action_store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await action_store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await action_store.begin_send(identity, now=NOW + timedelta(minutes=3))
        async with runtime.sessions.begin() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            assert job is not None
            job.status = "leased"
            job.attempt = job.max_attempts
            job.leased_by = "crashed-worker"
            job.owner_token = uuid4()
            job.lease_expires_at = stale_at
        jobs = SqlAlchemyWorkerJobStore(runtime.sessions, lambda _attempt: timedelta(0))
        summary = await jobs.reclaim_stale_leases(now=stale_at, limit=10)
        assert summary.requeued == 1
        assert summary.dead == 0
        async with runtime.sessions() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            run = await session.get(Run, identity.run_id)
            assert job is not None and job.status == "queued"
            assert job.attempt == job.max_attempts
            assert run is not None and run.status == "running"
    finally:
        await engine.dispose()


async def test_stale_recovery_exhaustion_terminalizes_unknown_without_delivery(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "stale-exhausted")
    stale_at = NOW + timedelta(minutes=4)
    try:
        action_store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await action_store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await action_store.begin_send(identity, now=NOW + timedelta(minutes=3))
        async with runtime.sessions.begin() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            assert action is not None and job is not None
            action.recovery_attempts = 3
            job.status = "leased"
            job.attempt = job.max_attempts
            job.leased_by = "crashed-worker"
            job.owner_token = uuid4()
            job.lease_expires_at = stale_at
        jobs = SqlAlchemyWorkerJobStore(runtime.sessions, lambda _attempt: timedelta(0))
        summary = await jobs.reclaim_stale_leases(now=stale_at, limit=10)
        assert summary.finished == 1
        assert summary.dead == 0
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            run = await session.get(Run, identity.run_id)
            assert action is not None and action.status == "outcome_unknown"
            assert job is not None and job.status == "done"
            assert run is not None and run.status == "failed"
            assert run.error_category == "external_outcome_unknown"
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("terminal_case", "expected_run_status", "expected_reason", "expected_error"),
    [
        ("cancelled_absent", "cancelled", "user_requested", None),
        ("revoked_absent", "cancelled", "authorization_revoked", None),
        ("explicit_rejection", "failed", None, "external_action_failed"),
        ("unknown_with_cancel_and_revoke", "failed", None, "external_outcome_unknown"),
    ],
)
async def test_stale_terminal_action_convergence_preserves_external_truth(
    migrated_database_url: str,
    terminal_case: str,
    expected_run_status: str,
    expected_reason: str | None,
    expected_error: str | None,
) -> None:
    engine, runtime, identity = await _approved(
        migrated_database_url, f"terminal-convergence-{terminal_case}"
    )
    stale_at = NOW + timedelta(minutes=10)
    try:
        action_store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await action_store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await action_store.begin_send(identity, now=NOW + timedelta(minutes=3))

        if terminal_case in {"cancelled_absent", "revoked_absent"}:
            recovery = await action_store.begin_reconciliation(
                identity,
                max_attempts=3,
                now=NOW + timedelta(minutes=4),
            )
            if terminal_case == "cancelled_absent":
                await RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
                    tenant=runtime.tenant,
                    run_id=runtime.run_id,
                )
            else:
                async with runtime.sessions.begin() as session:
                    membership = await session.scalar(
                        select(WorkspaceMembership).where(
                            WorkspaceMembership.workspace_id == runtime.tenant.workspace_id,
                            WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                        )
                    )
                    assert membership is not None
                    membership.revoked_at = NOW + timedelta(minutes=5)
            resend = await action_store.authorize_resend(
                identity,
                expected_recovery_attempt=recovery.recovery_attempt,
                now=NOW + timedelta(minutes=6),
            )
            assert resend.allowed is False
        elif terminal_case == "explicit_rejection":
            await action_store.confirm_failure(
                identity,
                error_category="mock_portal_rejected",
                evidence={"classification": "explicit_no_effect_rejection"},
                latency_ms=3,
                now=NOW + timedelta(minutes=4),
            )
        else:
            await RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
                tenant=runtime.tenant,
                run_id=runtime.run_id,
            )
            async with runtime.sessions.begin() as session:
                membership = await session.scalar(
                    select(WorkspaceMembership).where(
                        WorkspaceMembership.workspace_id == runtime.tenant.workspace_id,
                        WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                    )
                )
                assert membership is not None
                membership.revoked_at = NOW + timedelta(minutes=5)
            await action_store.confirm_outcome_unknown(
                identity,
                evidence={"classification": "reconciliation_query_unavailable"},
                latency_ms=3,
                now=NOW + timedelta(minutes=4),
            )

        async with runtime.sessions.begin() as session:
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            assert job is not None
            job.status = "leased"
            job.attempt = 1
            job.leased_by = "crashed-before-worker-finalization"
            job.owner_token = uuid4()
            job.lease_expires_at = stale_at

        jobs = SqlAlchemyWorkerJobStore(runtime.sessions, lambda _attempt: timedelta(0))
        summary = await jobs.reclaim_stale_leases(now=stale_at, limit=10)
        assert summary.finished == 1 and summary.dead == 0 and summary.requeued == 0

        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            job = await session.scalar(select(RunJob).where(RunJob.run_id == identity.run_id))
            run = await session.get(Run, identity.run_id)
            events = list(
                await session.scalars(
                    select(RunEvent)
                    .where(RunEvent.run_id == identity.run_id)
                    .order_by(RunEvent.seq)
                )
            )
            assert action is not None and invocation is not None
            expected_action_status = (
                "outcome_unknown" if terminal_case == "unknown_with_cancel_and_revoke" else "failed"
            )
            assert action.status == expected_action_status
            assert invocation.status == expected_action_status
            assert job is not None and job.status == "done"
            assert job.error_summary == expected_error
            assert run is not None and run.status == expected_run_status
            assert run.error_category == expected_error
            terminal_events = [
                event for event in events if event.type in {"run.cancelled", "run.failed"}
            ]
            assert len(terminal_events) == 1
            if expected_reason is not None:
                assert terminal_events[0].payload["reason"] == expected_reason
    finally:
        await engine.dispose()


async def test_recovery_resend_cas_wins_cancellation_race_without_rewriting_action(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "resend-race")
    transition_entered = asyncio.Event()
    release_transition = asyncio.Event()

    async def barrier(point: str) -> None:
        if point == "resend_transition_before_commit":
            transition_entered.set()
            await release_transition.wait()

    try:
        store = SqlAlchemyActionExecutionStore(runtime.sessions)
        await store.prepare_execution(identity, now=NOW + timedelta(minutes=2))
        await store.begin_send(identity, now=NOW + timedelta(minutes=3))
        recovery = await store.begin_reconciliation(
            identity, max_attempts=3, now=NOW + timedelta(minutes=4)
        )
        resend_task = asyncio.create_task(
            SqlAlchemyActionExecutionStore(
                runtime.sessions, fault_injector=barrier
            ).authorize_resend(
                identity,
                expected_recovery_attempt=recovery.recovery_attempt,
                now=NOW + timedelta(minutes=5),
            )
        )
        await transition_entered.wait()
        cancel_task = asyncio.create_task(
            RunService(SqlAlchemyRunStore(runtime.sessions)).cancel_run(
                tenant=runtime.tenant, run_id=runtime.run_id
            )
        )
        release_transition.set()
        resend, cancellation = await asyncio.gather(resend_task, cancel_task)
        assert resend.allowed is True
        assert cancellation.cancel_requested_at is not None
        await store.confirm_success(
            identity,
            result=ConfirmedActionResult("mock-submission:resend-race"),
            latency_ms=1,
            now=NOW + timedelta(minutes=6),
        )
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, identity.action_intent_id)
            invocation = await session.scalar(
                select(ToolInvocation).where(
                    ToolInvocation.action_intent_id == identity.action_intent_id
                )
            )
            assert action is not None and action.status == "succeeded"
            assert invocation is not None and invocation.status == "succeeded"
            assert invocation.attempt == 2
    finally:
        await engine.dispose()
