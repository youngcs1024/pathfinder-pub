from __future__ import annotations

from inspect import isawaitable
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    Run,
    RunEvent,
    RunJob,
    WorkspaceMembership,
)
from app.db.session import AsyncSessionFactory, database_session, transaction
from app.domain.actions import (
    ActionIntentRecord,
    persisted_action_intent_status,
    validate_exact_approval_binding,
)
from app.domain.approvals import (
    ApprovalDecisionCommand,
    ApprovalDecisionRecord,
    ApprovalRequestRecord,
    ApprovalResumeRecord,
    ApprovalReviewRecord,
    ApprovalStatus,
    ApprovalStoreFaultInjector,
    ApprovalStoreFaultPoint,
    persisted_approval_status,
    validate_approval_transition,
)
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
)
from app.domain.jobs import JobStatus
from app.domain.provisioning import WorkspaceRole
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


def _intent_record(row: ActionIntent) -> ActionIntentRecord:
    try:
        effect = ToolEffect(row.effect)
    except ValueError:
        raise DomainInvariantError("persisted action effect is invalid") from None
    return ActionIntentRecord(
        action_intent_id=row.id,
        workspace_id=row.workspace_id,
        originating_actor_user_id=row.originating_actor_user_id,
        run_id=row.run_id,
        action_key=row.action_key,
        action_revision=row.action_revision,
        tool_name=row.tool_name,
        effect=effect,
        args_snapshot=dict(row.args_snapshot),
        canonicalization_version=row.canonicalization_version,
        args_digest=row.args_digest,
        target_snapshot=dict(row.target_snapshot),
        target_canonicalization_version=row.target_canonicalization_version,
        target_digest=row.target_digest,
        approval_binding_version=row.approval_binding_version,
        approval_binding_digest=row.approval_binding_digest,
        status=persisted_action_intent_status(row.status),
        idempotency_key=row.idempotency_key,
        recovery_attempts=row.recovery_attempts,
        result=dict(row.result) if row.result is not None else None,
        evidence=dict(row.evidence) if row.evidence is not None else None,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _request_record(row: ApprovalRequest) -> ApprovalRequestRecord:
    return ApprovalRequestRecord(
        request_id=row.id,
        workspace_id=row.workspace_id,
        run_id=row.run_id,
        action_intent_id=row.action_intent_id,
        status=persisted_approval_status(row.status),
        args_digest=row.args_digest,
        target_digest=row.target_digest,
        approval_binding_version=row.approval_binding_version,
        approval_binding_digest=row.approval_binding_digest,
        policy_version=row.policy_version,
        policy_snapshot=dict(row.policy_snapshot),
        version=row.version,
        expires_at=row.expires_at,
        consumed_at=row.consumed_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _decision_record(row: ApprovalDecision, action_intent_id: UUID) -> ApprovalDecisionRecord:
    if row.decision not in {"approve", "reject"}:
        raise DomainInvariantError("persisted approval decision is invalid")
    return ApprovalDecisionRecord(
        approval_decision_id=row.id,
        workspace_id=row.workspace_id,
        approval_request_id=row.approval_request_id,
        action_intent_id=action_intent_id,
        actor_user_id=row.actor_user_id,
        decision=row.decision,  # type: ignore[arg-type]
        reason=row.reason,
        decided_at=row.decided_at,
    )


async def _membership(
    session: AsyncSession, tenant: TenantContext, *, lock: bool
) -> WorkspaceMembership:
    query = select(WorkspaceMembership).where(
        WorkspaceMembership.workspace_id == tenant.workspace_id,
        WorkspaceMembership.user_id == tenant.actor_user_id,
    )
    if lock:
        query = query.with_for_update()
    row = await session.scalar(query)
    if row is None or row.revoked_at is not None or row.role != tenant.role.value:
        raise DomainNotFoundError
    return row


async def _current_membership_for_decision(
    session: AsyncSession,
    tenant: TenantContext,
) -> WorkspaceMembership:
    row = await session.scalar(
        select(WorkspaceMembership)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
        )
        .with_for_update()
    )
    if row is None or row.revoked_at is not None:
        raise DomainNotFoundError
    return row


async def _bound_rows(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    action_intent_id: UUID,
    lock: bool,
) -> tuple[ApprovalRequest, ActionIntent]:
    query = select(ApprovalRequest).where(
        ApprovalRequest.workspace_id == workspace_id,
        ApprovalRequest.action_intent_id == action_intent_id,
    )
    if lock:
        query = query.with_for_update()
    request = await session.scalar(query)
    if request is None:
        raise DomainNotFoundError
    action_query = select(ActionIntent).where(
        ActionIntent.workspace_id == workspace_id,
        ActionIntent.run_id == request.run_id,
        ActionIntent.id == action_intent_id,
    )
    if lock:
        action_query = action_query.with_for_update()
    intent = await session.scalar(action_query)
    if intent is None:
        raise DomainInvariantError("approval action binding is missing")
    validate_exact_approval_binding(intent=_intent_record(intent), request=_request_record(request))
    return request, intent


class SqlAlchemyApprovalStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        fault_injector: ApprovalStoreFaultInjector | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._fault_injector = fault_injector

    async def _inject(self, point: ApprovalStoreFaultPoint) -> None:
        if self._fault_injector is None:
            return
        result = self._fault_injector(point)
        if isawaitable(result):
            await result

    async def get_action_review(
        self, *, tenant: TenantContext, action_intent_id: UUID
    ) -> ApprovalReviewRecord:
        async with database_session(self._session_factory) as session:
            await _membership(session, tenant, lock=False)
            request, intent = await _bound_rows(
                session,
                workspace_id=tenant.workspace_id,
                action_intent_id=action_intent_id,
                lock=False,
            )
            decisions = list(
                await session.scalars(
                    select(ApprovalDecision).where(
                        ApprovalDecision.workspace_id == tenant.workspace_id,
                        ApprovalDecision.approval_request_id == request.id,
                    )
                )
            )
            if len(decisions) > 1:
                raise DomainInvariantError("approval request has multiple decisions")
            return ApprovalReviewRecord(
                intent=_intent_record(intent),
                approval_request=_request_record(request),
                decision=(_decision_record(decisions[0], intent.id) if decisions else None),
            )

    async def decide(self, command: ApprovalDecisionCommand) -> ApprovalDecisionRecord:
        async with transaction(self._session_factory) as session:
            membership = await _current_membership_for_decision(session, command.tenant)
            request = await session.scalar(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.workspace_id == command.tenant.workspace_id,
                    ApprovalRequest.action_intent_id == command.action_intent_id,
                )
                .with_for_update()
            )
            if request is None:
                raise DomainNotFoundError
            own = await session.scalar(
                select(ApprovalDecision).where(
                    ApprovalDecision.workspace_id == command.tenant.workspace_id,
                    ApprovalDecision.approval_request_id == request.id,
                    ApprovalDecision.actor_user_id == command.tenant.actor_user_id,
                )
            )
            if own is not None:
                if own.decision != command.decision:
                    raise DomainConflictError("approval decision retry conflicts")
                return _decision_record(own, request.action_intent_id)
            if membership.role not in {
                WorkspaceRole.REVIEWER.value,
                WorkspaceRole.ADMIN.value,
            }:
                raise DomainConflictError("workspace role cannot decide approval")
            if (
                await session.scalar(
                    select(ApprovalDecision.id).where(
                        ApprovalDecision.workspace_id == command.tenant.workspace_id,
                        ApprovalDecision.approval_request_id == request.id,
                    )
                )
                is not None
            ):
                raise DomainConflictError("approval request was already decided")
            if persisted_approval_status(request.status) is not ApprovalStatus.PENDING:
                raise DomainConflictError("approval request is not pending")
            if command.now >= request.expires_at:
                raise DomainConflictError("approval request has expired")
            if request.version != command.expected_version:
                raise DomainConflictError("approval request version is stale")
            job = await session.scalar(
                select(RunJob)
                .where(
                    RunJob.workspace_id == command.tenant.workspace_id,
                    RunJob.run_id == request.run_id,
                )
                .with_for_update()
            )
            run = await session.scalar(
                select(Run)
                .where(
                    Run.workspace_id == command.tenant.workspace_id,
                    Run.id == request.run_id,
                )
                .with_for_update()
            )
            if job is None or run is None:
                raise DomainInvariantError("approval run or job is missing")
            if run.status != RunStatus.WAITING_APPROVAL.value:
                raise DomainConflictError("approval run is not waiting")
            if job.status != JobStatus.DONE.value:
                raise DomainConflictError("approval job is not dormant")
            intent = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == command.tenant.workspace_id,
                    ActionIntent.run_id == request.run_id,
                    ActionIntent.id == command.action_intent_id,
                )
                .with_for_update()
            )
            if intent is None:
                raise DomainInvariantError("approval action binding is missing")
            validate_exact_approval_binding(
                intent=_intent_record(intent), request=_request_record(request)
            )
            target = (
                ApprovalStatus.APPROVED
                if command.decision == "approve"
                else ApprovalStatus.REJECTED
            )
            validate_approval_transition(ApprovalStatus.PENDING, target)
            decision = ApprovalDecision(
                id=uuid4(),
                workspace_id=command.tenant.workspace_id,
                approval_request_id=request.id,
                actor_user_id=command.tenant.actor_user_id,
                decision=command.decision,
                reason=command.reason,
                decided_at=command.now,
            )
            session.add(decision)
            request.status = target.value
            request.version += 1
            request.updated_at = command.now
            job.status = JobStatus.QUEUED.value
            job.attempt = 0
            job.available_at = command.now
            job.leased_by = None
            job.owner_token = None
            job.lease_expires_at = None
            job.error_summary = None
            job.resume_approval_request_id = request.id
            sequence = run.next_event_seq
            if not isinstance(sequence, int) or sequence < 1:
                raise DomainInvariantError("persisted run event sequence is invalid")
            run.next_event_seq = sequence + 1
            session.add(
                RunEvent(
                    id=uuid4(),
                    workspace_id=run.workspace_id,
                    run_id=run.id,
                    actor_user_id=command.tenant.actor_user_id,
                    seq=sequence,
                    type=RunEventType.APPROVAL_DECIDED.value,
                    version=CURRENT_RUN_EVENT_VERSION,
                    payload={
                        "approval_request_id": str(request.id),
                        "approval_decision_id": str(decision.id),
                        "action_intent_id": str(intent.id),
                        "decision": command.decision,
                        "request_version": request.version,
                    },
                )
            )
            await session.flush()
            await self._inject("decision_before_commit")
            record = _decision_record(decision, intent.id)
        return record

    async def resolve_approval_resume(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        action_intent_id: UUID,
        approval_request_id: UUID,
    ) -> ApprovalResumeRecord:
        async with database_session(self._session_factory) as session:
            await _membership(session, tenant, lock=False)
            request, intent = await _bound_rows(
                session,
                workspace_id=tenant.workspace_id,
                action_intent_id=action_intent_id,
                lock=False,
            )
            if request.id != approval_request_id or request.run_id != run_id:
                raise DomainInvariantError("approval resume identity does not match binding")
            decisions = (
                await session.scalars(
                    select(ApprovalDecision.decision)
                    .where(
                        ApprovalDecision.workspace_id == tenant.workspace_id,
                        ApprovalDecision.approval_request_id == request.id,
                    )
                    .limit(2)
                )
            ).all()
            if len(decisions) > 1:
                raise DomainInvariantError("persisted approval has multiple decisions")
            decision = decisions[0] if decisions else None
            if decisions and decision not in {"approve", "reject"}:
                raise DomainInvariantError("persisted approval decision is invalid")
            record = _request_record(request)
            allowed_decisions = {
                ApprovalStatus.PENDING: {None},
                ApprovalStatus.APPROVED: {"approve"},
                ApprovalStatus.REJECTED: {"reject"},
                ApprovalStatus.CONSUMED: {"approve"},
                ApprovalStatus.EXPIRED: {None, "approve"},
            }
            if decision not in allowed_decisions[record.status]:
                raise DomainInvariantError("persisted approval status and decision disagree")
            return ApprovalResumeRecord(
                approval_request=record,
                action_intent_id=intent.id,
                decision=decision,
            )
