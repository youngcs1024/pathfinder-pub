from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select

from app.db.actions import SqlAlchemyActionStore
from app.db.approval_expiry import SqlAlchemyApprovalRequestExpirySweeper
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.events import SqlAlchemyRunEventReader
from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    Document,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
    User,
    Workspace,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.runs import SqlAlchemyRunStore
from app.db.session import AsyncSessionFactory, create_database_engine, create_session_factory
from app.domain.actions import (
    ACTION_KEY,
    CancelActionCommand,
    PrepareActionCommand,
    SubmitApplicationArgsV1,
    SupersedeActionCommand,
)
from app.domain.approvals import ApprovalDecisionCommand
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
)
from app.domain.provisioning import ProvisioningService, WorkspaceKind, WorkspaceRole
from app.domain.runs import RunMode, RunService
from app.domain.tenancy import TenantContext
from app.events.contracts import RunEventType

pytestmark = pytest.mark.integration
NOW = datetime(2026, 8, 24, 12, tzinfo=UTC)


@dataclass(frozen=True)
class Runtime:
    sessions: AsyncSessionFactory
    tenant: TenantContext
    run_id: UUID
    document_id: UUID


async def _seed(database_url: str, suffix: str) -> tuple[object, Runtime, object]:
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(f"gate6-decision-{suffix}")
    tenant = TenantContext(identity.workspace_id, identity.user_id, WorkspaceRole.ADMIN)
    document_id = uuid4()
    content = f"Synthetic resume {suffix}"
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
        query="Prepare application",
        resume_document_id=document_id,
    )
    async with sessions.begin() as session:
        run = await session.get(Run, accepted.run_id)
        assert run is not None
        run.status = "running"
        run.started_at = NOW
    runtime = Runtime(sessions, tenant, accepted.run_id, document_id)
    prepared = await SqlAlchemyActionStore(sessions).prepare_action(
        PrepareActionCommand(
            tenant=tenant,
            run_id=accepted.run_id,
            action_proposal_id=uuid4(),
            action_key=ACTION_KEY,
            action_revision=1,
            args=SubmitApplicationArgsV1(
                job_ref="job",
                resume_document_id=document_id,
                answers={},
                cover_letter="Exact application draft",
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
        job.leased_by = None
        job.owner_token = None
        job.lease_expires_at = None
    return engine, runtime, prepared


def _command(runtime: Runtime, prepared: object, decision: str = "reject"):
    return ApprovalDecisionCommand(
        tenant=runtime.tenant,
        action_intent_id=prepared.intent.action_intent_id,
        decision=decision,
        expected_version=1,
        reason="reviewed",
        now=NOW + timedelta(minutes=1),
    )


async def test_action_review_workspace_members_can_read_without_cross_tenant_leak(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "review-read-matrix")
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(runtime.sessions))
    other = await provisioning.provision_personal_workspace("review-read-other")
    other_personal = TenantContext(other.workspace_id, other.user_id, WorkspaceRole.ADMIN)
    store = SqlAlchemyApprovalStore(runtime.sessions)
    member_membership_id = uuid4()
    second_workspace_id = uuid4()
    try:
        async with runtime.sessions.begin() as session:
            session.add_all(
                (
                    WorkspaceMembership(
                        id=member_membership_id,
                        workspace_id=runtime.tenant.workspace_id,
                        user_id=other.user_id,
                        role=WorkspaceRole.MEMBER.value,
                    ),
                    Workspace(
                        id=second_workspace_id,
                        kind=WorkspaceKind.TEAM.value,
                        name="Same actor alternate workspace",
                        created_by_user_id=runtime.tenant.actor_user_id,
                    ),
                )
            )
            await session.flush()
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=second_workspace_id,
                    user_id=runtime.tenant.actor_user_id,
                    role=WorkspaceRole.MEMBER.value,
                )
            )

        for role in (WorkspaceRole.MEMBER, WorkspaceRole.REVIEWER, WorkspaceRole.ADMIN):
            async with runtime.sessions.begin() as session:
                membership = await session.get(WorkspaceMembership, member_membership_id)
                assert membership is not None
                membership.role = role.value
            review = await store.get_action_review(
                tenant=TenantContext(runtime.tenant.workspace_id, other.user_id, role),
                action_intent_id=prepared.intent.action_intent_id,
            )
            assert review.intent.action_intent_id == prepared.intent.action_intent_id

        denied = (
            other_personal,
            TenantContext(second_workspace_id, runtime.tenant.actor_user_id, WorkspaceRole.MEMBER),
        )
        for tenant in denied:
            with pytest.raises(DomainNotFoundError):
                await store.get_action_review(
                    tenant=tenant,
                    action_intent_id=prepared.intent.action_intent_id,
                )
        with pytest.raises(DomainNotFoundError):
            await store.get_action_review(
                tenant=runtime.tenant,
                action_intent_id=uuid4(),
            )

        stale_member = TenantContext(
            runtime.tenant.workspace_id,
            other.user_id,
            WorkspaceRole.ADMIN,
        )
        async with runtime.sessions.begin() as session:
            membership = await session.get(WorkspaceMembership, member_membership_id)
            assert membership is not None
            membership.revoked_at = NOW
        with pytest.raises(DomainNotFoundError):
            await store.get_action_review(
                tenant=stale_member,
                action_intent_id=prepared.intent.action_intent_id,
            )
    finally:
        await engine.dispose()


async def test_decision_is_atomic_idempotent_and_wakes_the_existing_job(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "idempotent")
    try:
        store = SqlAlchemyApprovalStore(runtime.sessions)
        first = await store.decide(_command(runtime, prepared))
        second = await store.decide(_command(runtime, prepared))
        assert second == first
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            run = await session.get(Run, runtime.run_id)
            assert request is not None and job is not None and run is not None
            assert (request.status, request.version) == ("rejected", 2)
            assert (job.status, job.attempt, job.resume_approval_request_id) == (
                "queued",
                0,
                request.id,
            )
            assert run.status == "waiting_approval"
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 1
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(RunEvent.type == RunEventType.APPROVAL_DECIDED.value)
                )
                == 1
            )
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 0
        replay = await SqlAlchemyRunEventReader(runtime.sessions).read_after(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            after_seq=2,
            limit=100,
        )
        decided = [event for event in replay.events if event.type is RunEventType.APPROVAL_DECIDED]
        assert len(decided) == 1
        assert decided[0].payload == {
            "approval_request_id": str(prepared.approval_request.request_id),
            "approval_decision_id": str(first.approval_decision_id),
            "action_intent_id": str(prepared.intent.action_intent_id),
            "decision": "reject",
            "request_version": 2,
        }
        with pytest.raises(DomainConflictError):
            await store.decide(_command(runtime, prepared, "approve"))
    finally:
        await engine.dispose()


async def test_decision_fault_rolls_back_every_fact_then_retry_converges(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "rollback")
    try:

        async def fail(_point: str) -> None:
            raise RuntimeError("decision fault")

        with pytest.raises(RuntimeError, match="decision fault"):
            await SqlAlchemyApprovalStore(runtime.sessions, fault_injector=fail).decide(
                _command(runtime, prepared)
            )
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert request is not None and job is not None
            assert (request.status, request.version, job.status) == ("pending", 1, "done")
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 0
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(RunEvent.type == RunEventType.APPROVAL_DECIDED.value)
                )
                == 0
            )
        resolved = await SqlAlchemyApprovalStore(runtime.sessions).resolve_approval_resume(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            action_intent_id=prepared.intent.action_intent_id,
            approval_request_id=prepared.approval_request.request_id,
        )
        assert resolved.decision is None
        await SqlAlchemyApprovalStore(runtime.sessions).decide(_command(runtime, prepared))
    finally:
        await engine.dispose()


async def test_rejected_action_cancellation_is_atomic_and_idempotent(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "cancel")
    try:
        await SqlAlchemyApprovalStore(runtime.sessions).decide(_command(runtime, prepared))
        command = CancelActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            action_intent_id=prepared.intent.action_intent_id,
            approval_request_id=prepared.approval_request.request_id,
            reason="approval_rejected",
            now=NOW + timedelta(minutes=2),
        )
        store = SqlAlchemyActionStore(runtime.sessions)
        assert (await store.cancel_action(command)).changed is True
        assert (await store.cancel_action(command)).changed is False
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, prepared.intent.action_intent_id)
            assert action is not None and action.status == "cancelled"
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(RunEvent.type == RunEventType.ACTION_CANCELLED.value)
                )
                == 1
            )
            assert await session.scalar(select(func.count()).select_from(ToolInvocation)) == 0
    finally:
        await engine.dispose()


async def test_action_cancellation_fault_rolls_back_action_event_and_sequence(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "cancel-rollback")
    try:
        await SqlAlchemyApprovalStore(runtime.sessions).decide(_command(runtime, prepared))
        command = CancelActionCommand(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            action_intent_id=prepared.intent.action_intent_id,
            approval_request_id=prepared.approval_request.request_id,
            reason="approval_rejected",
            now=NOW + timedelta(minutes=2),
        )

        async def fail(point: str) -> None:
            if point == "cancel_before_commit":
                raise RuntimeError("cancel fault")

        with pytest.raises(RuntimeError, match="cancel fault"):
            await SqlAlchemyActionStore(runtime.sessions, fault_injector=fail).cancel_action(
                command
            )
        async with runtime.sessions() as session:
            action = await session.get(ActionIntent, prepared.intent.action_intent_id)
            assert action is not None and action.status == "proposed"
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(RunEvent.type == RunEventType.ACTION_CANCELLED.value)
                )
                == 0
            )
        await SqlAlchemyActionStore(runtime.sessions).cancel_action(command)
        async with runtime.sessions() as session:
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(RunEvent.type == RunEventType.ACTION_CANCELLED.value)
                )
                == 1
            )
    finally:
        await engine.dispose()


async def test_persisted_membership_role_defeats_forged_tenant_context(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "forged-role")
    try:
        async with runtime.sessions.begin() as session:
            membership = await session.scalar(
                select(WorkspaceMembership).where(
                    WorkspaceMembership.workspace_id == runtime.tenant.workspace_id,
                    WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                )
            )
            assert membership is not None
            membership.role = "member"
        forged = TenantContext(
            runtime.tenant.workspace_id,
            runtime.tenant.actor_user_id,
            WorkspaceRole.REVIEWER,
        )
        with pytest.raises(DomainConflictError):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(
                ApprovalDecisionCommand(
                    tenant=forged,
                    action_intent_id=prepared.intent.action_intent_id,
                    decision="approve",
                    expected_version=1,
                    reason=None,
                    now=NOW + timedelta(minutes=1),
                )
            )
        member = TenantContext(
            runtime.tenant.workspace_id,
            runtime.tenant.actor_user_id,
            WorkspaceRole.MEMBER,
        )
        with pytest.raises(DomainConflictError):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(
                ApprovalDecisionCommand(
                    tenant=member,
                    action_intent_id=prepared.intent.action_intent_id,
                    decision="approve",
                    expected_version=1,
                    reason=None,
                    now=NOW + timedelta(minutes=1),
                )
            )
    finally:
        await engine.dispose()


async def test_decision_uses_committed_persisted_role_after_promotion(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "role-promotion")
    try:
        stale_member = TenantContext(
            runtime.tenant.workspace_id,
            runtime.tenant.actor_user_id,
            WorkspaceRole.MEMBER,
        )
        async with runtime.sessions.begin() as session:
            membership = await session.scalar(
                select(WorkspaceMembership).where(
                    WorkspaceMembership.workspace_id == runtime.tenant.workspace_id,
                    WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                )
            )
            assert membership is not None
            membership.role = WorkspaceRole.REVIEWER.value

        decision = await SqlAlchemyApprovalStore(runtime.sessions).decide(
            ApprovalDecisionCommand(
                tenant=stale_member,
                action_intent_id=prepared.intent.action_intent_id,
                decision="approve",
                expected_version=1,
                reason=None,
                now=NOW + timedelta(minutes=1),
            )
        )

        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert request is not None and job is not None
            assert (request.status, request.version) == ("approved", 2)
            assert job.status == "queued"
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 1
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(
                        RunEvent.run_id == runtime.run_id,
                        RunEvent.type == RunEventType.APPROVAL_DECIDED.value,
                    )
                )
                == 1
            )
            persisted = await session.get(ApprovalDecision, decision.approval_decision_id)
            assert persisted is not None and persisted.actor_user_id == stale_member.actor_user_id
    finally:
        await engine.dispose()


async def test_decision_uses_committed_persisted_role_after_demotion(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "role-demotion")
    try:
        stale_reviewer = TenantContext(
            runtime.tenant.workspace_id,
            runtime.tenant.actor_user_id,
            WorkspaceRole.REVIEWER,
        )
        async with runtime.sessions.begin() as session:
            membership = await session.scalar(
                select(WorkspaceMembership).where(
                    WorkspaceMembership.workspace_id == runtime.tenant.workspace_id,
                    WorkspaceMembership.user_id == runtime.tenant.actor_user_id,
                )
            )
            assert membership is not None
            membership.role = WorkspaceRole.MEMBER.value

        with pytest.raises(DomainConflictError):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(
                ApprovalDecisionCommand(
                    tenant=stale_reviewer,
                    action_intent_id=prepared.intent.action_intent_id,
                    decision="approve",
                    expected_version=1,
                    reason=None,
                    now=NOW + timedelta(minutes=1),
                )
            )

        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert request is not None and job is not None
            assert (request.status, request.version) == ("pending", 1)
            assert job.status == "done"
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 0
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(RunEvent)
                    .where(
                        RunEvent.run_id == runtime.run_id,
                        RunEvent.type == RunEventType.APPROVAL_DECIDED.value,
                    )
                )
                == 0
            )
    finally:
        await engine.dispose()


async def test_waiting_approval_user_cancel_converges_request_action_job_and_run(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "user-cancel")
    try:
        cancellation = await SqlAlchemyRunStore(runtime.sessions).cancel_run(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            allow_other_creator=True,
        )
        assert cancellation.status.value == "cancelled"
        with pytest.raises(DomainConflictError):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(
                _command(runtime, prepared, "approve")
            )
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            action = await session.get(ActionIntent, prepared.intent.action_intent_id)
            run = await session.get(Run, runtime.run_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert request is not None and action is not None
            assert run is not None and job is not None
            assert (request.status, request.version) == ("expired", 2)
            assert action.status == "cancelled"
            assert run.status == "cancelled"
            assert job.status == "done" and job.owner_token is None
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 0
            event_types = list(
                await session.scalars(
                    select(RunEvent.type)
                    .where(RunEvent.run_id == runtime.run_id)
                    .order_by(RunEvent.seq)
                )
            )
            assert event_types[-3:] == [
                RunEventType.APPROVAL_EXPIRED.value,
                RunEventType.ACTION_CANCELLED.value,
                RunEventType.RUN_CANCELLED.value,
            ]
    finally:
        await engine.dispose()


async def test_waiting_cancel_targets_current_pair_after_action_supersede(
    migrated_database_url: str,
) -> None:
    engine, runtime, old = await _seed(migrated_database_url, "superseded-cancel")
    try:
        async with runtime.sessions.begin() as session:
            run = await session.get(Run, runtime.run_id)
            assert run is not None
            run.status = "running"

        new = await SqlAlchemyActionStore(runtime.sessions).supersede_action(
            SupersedeActionCommand(
                tenant=runtime.tenant,
                run_id=runtime.run_id,
                old_action_intent_id=old.intent.action_intent_id,
                new_action_proposal_id=uuid4(),
                action_key=ACTION_KEY,
                action_revision=2,
                args=SubmitApplicationArgsV1(
                    job_ref="job-revision-2",
                    resume_document_id=runtime.document_id,
                    answers={},
                    cover_letter="Revised exact application draft",
                ),
                now=NOW + timedelta(minutes=2),
                expires_at=NOW + timedelta(hours=2),
            )
        )
        async with runtime.sessions.begin() as session:
            run = await session.get(Run, runtime.run_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert run is not None and job is not None
            run.status = "waiting_approval"
            job.status = "done"

        cancellation = await SqlAlchemyRunStore(runtime.sessions).cancel_run(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            allow_other_creator=True,
        )
        assert cancellation.status.value == "cancelled"

        async with runtime.sessions() as session:
            old_request = await session.get(ApprovalRequest, old.approval_request.request_id)
            old_action = await session.get(ActionIntent, old.intent.action_intent_id)
            new_request = await session.get(ApprovalRequest, new.approval_request.request_id)
            new_action = await session.get(ActionIntent, new.intent.action_intent_id)
            run = await session.get(Run, runtime.run_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert old_request is not None and old_action is not None
            assert new_request is not None and new_action is not None
            assert run is not None and job is not None
            assert (old_request.status, old_request.version) == ("expired", 2)
            assert old_action.status == "cancelled"
            assert (new_request.status, new_request.version) == ("expired", 2)
            assert new_action.status == "cancelled"
            assert run.status == "cancelled"
            assert job.status == "done"
            assert (
                job.leased_by,
                job.owner_token,
                job.lease_expires_at,
            ) == (None, None, None)
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(ActionIntent)
                    .where(
                        ActionIntent.run_id == runtime.run_id,
                        ActionIntent.status.in_(("proposed", "authorized")),
                    )
                )
                == 0
            )
            user_cancel_events = list(
                await session.scalars(
                    select(RunEvent)
                    .where(
                        RunEvent.run_id == runtime.run_id,
                        RunEvent.payload["reason"].as_string() == "user_requested",
                    )
                    .order_by(RunEvent.seq)
                )
            )
            assert [event.type for event in user_cancel_events] == [
                RunEventType.APPROVAL_EXPIRED.value,
                RunEventType.ACTION_CANCELLED.value,
                RunEventType.RUN_CANCELLED.value,
            ]
            assert user_cancel_events[0].payload["approval_request_id"] == str(new_request.id)
            assert user_cancel_events[0].payload["action_intent_id"] == str(new_action.id)
            assert user_cancel_events[1].payload["approval_request_id"] == str(new_request.id)
            assert user_cancel_events[1].payload["action_intent_id"] == str(new_action.id)
            assert all(
                event.payload.get("approval_request_id") != str(old_request.id)
                for event in user_cancel_events
            )
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("action", "args_snapshot", {"job_ref": "tampered"}),
        ("action", "args_digest", f"sha256:{'b' * 64}"),
        ("action", "target_snapshot", {"provider": "tampered"}),
        ("action", "target_digest", f"sha256:{'b' * 64}"),
        ("action", "action_key", "tampered_action"),
        ("action", "action_revision", 2),
        ("action", "tool_name", "tampered_tool"),
        ("action", "effect", "read_only"),
        ("request", "policy_version", 2),
        ("request", "policy_snapshot", {"eligible_roles": ["admin"]}),
        ("request", "approval_binding_version", 2),
        ("request", "approval_binding_digest", f"sha256:{'b' * 64}"),
    ],
)
async def test_decision_fails_closed_for_persisted_binding_tamper(
    migrated_database_url: str,
    target: str,
    field: str,
    value: object,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, f"tamper-{target}-{field}")
    try:
        async with runtime.sessions.begin() as session:
            row = await session.get(
                ActionIntent if target == "action" else ApprovalRequest,
                prepared.intent.action_intent_id
                if target == "action"
                else prepared.approval_request.request_id,
            )
            assert row is not None
            setattr(row, field, value)
        with pytest.raises((DomainInvariantError, DomainValidationError)):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(_command(runtime, prepared))
    finally:
        await engine.dispose()


async def test_concurrent_reviewers_converge_to_one_decision(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "concurrent")
    try:
        second_user_id = uuid4()
        async with runtime.sessions.begin() as session:
            session.add(User(id=second_user_id, auth_subject="gate6-second-reviewer"))
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=runtime.tenant.workspace_id,
                    user_id=second_user_id,
                    role="reviewer",
                    revoked_at=None,
                )
            )
        second = TenantContext(runtime.tenant.workspace_id, second_user_id, WorkspaceRole.REVIEWER)
        first_command = _command(runtime, prepared, "approve")
        second_command = ApprovalDecisionCommand(
            tenant=second,
            action_intent_id=prepared.intent.action_intent_id,
            decision="reject",
            expected_version=1,
            reason=None,
            now=NOW + timedelta(minutes=1),
        )
        results = await asyncio.gather(
            SqlAlchemyApprovalStore(runtime.sessions).decide(first_command),
            SqlAlchemyApprovalStore(runtime.sessions).decide(second_command),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, Exception) for result in results) == 1
        assert sum(isinstance(result, DomainConflictError) for result in results) == 1
        async with runtime.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 1
    finally:
        await engine.dispose()


async def test_expiry_winner_rejects_late_decision_without_history(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "expiry-wins")
    try:
        summary = await SqlAlchemyApprovalRequestExpirySweeper(
            runtime.sessions
        ).sweep_due_approval_requests(now=NOW + timedelta(hours=2), limit=10)
        assert summary.expired == summary.requeued == 1
        with pytest.raises(DomainConflictError):
            await SqlAlchemyApprovalStore(runtime.sessions).decide(
                _command(runtime, prepared, "approve")
            )
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            assert request is not None and request.status == "expired"
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 0
    finally:
        await engine.dispose()


async def test_approved_decision_then_user_cancel_preserves_decision_and_converges(
    migrated_database_url: str,
) -> None:
    engine, runtime, prepared = await _seed(migrated_database_url, "decision-cancel")
    try:
        decision = await SqlAlchemyApprovalStore(runtime.sessions).decide(
            _command(runtime, prepared, "approve")
        )
        cancellation = await SqlAlchemyRunStore(runtime.sessions).cancel_run(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            allow_other_creator=True,
        )
        assert cancellation.status.value == "cancelled"
        async with runtime.sessions() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            persisted = await session.get(ApprovalDecision, decision.approval_decision_id)
            action = await session.get(ActionIntent, prepared.intent.action_intent_id)
            run = await session.get(Run, runtime.run_id)
            job = await session.scalar(select(RunJob).where(RunJob.run_id == runtime.run_id))
            assert request is not None and (request.status, request.version) == ("expired", 3)
            assert persisted is not None
            assert (
                persisted.id,
                persisted.approval_request_id,
                persisted.actor_user_id,
                persisted.decision,
                persisted.reason,
                persisted.decided_at,
            ) == (
                decision.approval_decision_id,
                decision.approval_request_id,
                runtime.tenant.actor_user_id,
                decision.decision,
                decision.reason,
                decision.decided_at,
            )
            assert action is not None and action.status == "cancelled"
            assert run is not None and run.status == "cancelled"
            assert job is not None and job.status == "done"
            assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 1
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        ("pending", None),
        ("approved", "approve"),
        ("rejected", "reject"),
        ("expired", None),
        ("expired", "approve"),
    ],
)
async def test_resume_resolver_returns_only_persisted_decision(
    migrated_database_url, status, decision
):
    engine, runtime, prepared = await _seed(migrated_database_url, f"resume-{status}-{decision}")
    store = SqlAlchemyApprovalStore(runtime.sessions)
    try:
        if decision:
            await store.decide(_command(runtime, prepared, decision))
        if status == "expired":
            await SqlAlchemyApprovalRequestExpirySweeper(
                runtime.sessions
            ).sweep_due_approval_requests(now=NOW + timedelta(hours=2), limit=10)
        resolved = await store.resolve_approval_resume(
            tenant=runtime.tenant,
            run_id=runtime.run_id,
            action_intent_id=prepared.intent.action_intent_id,
            approval_request_id=prepared.approval_request.request_id,
        )
        assert resolved.approval_request.status.value == status
        assert resolved.decision == decision
        assert not hasattr(resolved, "reason") and not hasattr(resolved, "actor_user_id")
        other = await ProvisioningService(
            SqlAlchemyProvisioningStore(runtime.sessions)
        ).provision_personal_workspace("resolver-other")
        with pytest.raises(DomainNotFoundError):
            await store.resolve_approval_resume(
                tenant=TenantContext(other.workspace_id, other.user_id, WorkspaceRole.ADMIN),
                run_id=runtime.run_id,
                action_intent_id=prepared.intent.action_intent_id,
                approval_request_id=prepared.approval_request.request_id,
            )
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("status", "decision"),
    [
        ("approved", None),
        ("rejected", None),
        ("pending", "approve"),
        ("approved", "reject"),
        ("rejected", "approve"),
        ("expired", "reject"),
    ],
)
async def test_resume_resolver_rejects_inconsistent_persisted_facts(
    migrated_database_url, status, decision
):
    engine, runtime, prepared = await _seed(migrated_database_url, f"corrupt-{status}-{decision}")
    store = SqlAlchemyApprovalStore(runtime.sessions)
    try:
        if decision:
            await store.decide(_command(runtime, prepared, decision))
        async with runtime.sessions.begin() as session:
            request = await session.get(ApprovalRequest, prepared.approval_request.request_id)
            request.status = status
        with pytest.raises(DomainInvariantError, match="status and decision disagree"):
            await store.resolve_approval_resume(
                tenant=runtime.tenant,
                run_id=runtime.run_id,
                action_intent_id=prepared.intent.action_intent_id,
                approval_request_id=prepared.approval_request.request_id,
            )
    finally:
        await engine.dispose()


async def test_resume_resolver_rejects_multiple_persisted_decisions(migrated_database_url):
    engine, runtime, prepared = await _seed(migrated_database_url, "multiple-resume-decisions")
    store = SqlAlchemyApprovalStore(runtime.sessions)
    try:
        await store.decide(_command(runtime, prepared, "approve"))
        # Uniqueness is per actor; inject another actor's row without disabling any constraint.
        other = await ProvisioningService(
            SqlAlchemyProvisioningStore(runtime.sessions)
        ).provision_personal_workspace("corrupt-second-reviewer")
        async with runtime.sessions.begin() as session:
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=runtime.tenant.workspace_id,
                    user_id=other.user_id,
                    role="reviewer",
                )
            )
            await session.flush()
            session.add(
                ApprovalDecision(
                    id=uuid4(),
                    workspace_id=runtime.tenant.workspace_id,
                    approval_request_id=prepared.approval_request.request_id,
                    actor_user_id=other.user_id,
                    decision="approve",
                    reason="PRIVATE-REASON",
                    decided_at=NOW + timedelta(minutes=2),
                )
            )
        with pytest.raises(DomainInvariantError, match="multiple decisions"):
            await store.resolve_approval_resume(
                tenant=runtime.tenant,
                run_id=runtime.run_id,
                action_intent_id=prepared.intent.action_intent_id,
                approval_request_id=prepared.approval_request.request_id,
            )
    finally:
        await engine.dispose()
