from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from inspect import isawaitable
from uuid import UUID, uuid4

from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Run,
    RunEvent,
    WorkspaceMembership,
)
from app.db.session import AsyncSessionFactory, transaction
from app.domain.actions import (
    ACTION_KEY,
    APPROVAL_BINDING_VERSION,
    CANONICALIZATION_VERSION,
    INITIAL_ACTION_REVISION,
    POLICY_VERSION,
    SUBMIT_APPLICATION_TOOL_NAME,
    TARGET_CANONICALIZATION_VERSION,
    ActionIntentRecord,
    ActionIntentStatus,
    ActionStoreFaultInjector,
    ActionStoreFaultPoint,
    ApprovalBindingV1,
    CancelActionCommand,
    CancelledAction,
    PrepareActionCommand,
    PreparedAction,
    SupersedeActionCommand,
    approval_binding_digest,
    canonicalize_action_args,
    canonicalize_trusted_target,
    persisted_action_intent_status,
    validate_action_intent_transition,
    validate_exact_approval_binding,
)
from app.domain.approvals import (
    ApprovalRequestRecord,
    ApprovalStatus,
    fixed_approval_policy,
    persisted_approval_status,
    validate_approval_transition,
)
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
)
from app.domain.runs import CURRENT_GRAPH_VERSION, RunMode, RunStatus
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


@dataclass(frozen=True, slots=True)
class _ProposalFacts:
    workspace_id: UUID
    actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID
    action_key: str
    action_revision: int
    args_snapshot: dict[str, object]
    args_digest: str
    target_snapshot: dict[str, object]
    target_digest: str
    approval_binding_digest: str
    policy_snapshot: dict[str, object]
    expires_at: datetime


async def _require_current_tenant(
    session: AsyncSession,
    tenant: TenantContext,
) -> None:
    role = await session.scalar(
        select(WorkspaceMembership.role)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
            WorkspaceMembership.revoked_at.is_(None),
            WorkspaceMembership.role == tenant.role.value,
        )
        .with_for_update(read=True)
    )
    if role is None:
        raise DomainNotFoundError


async def _lock_application_run(
    session: AsyncSession,
    command: PrepareActionCommand,
) -> Run:
    run = await session.scalar(
        select(Run)
        .where(
            Run.workspace_id == command.tenant.workspace_id,
            Run.id == command.run_id,
        )
        .with_for_update()
    )
    if run is None or run.created_by_user_id != command.tenant.actor_user_id:
        raise DomainNotFoundError
    if run.mode != RunMode.APPLICATION.value:
        raise DomainConflictError("action preparation requires an application run")
    if run.status != RunStatus.RUNNING.value:
        raise DomainConflictError("action preparation requires a running run")
    if run.graph_version != CURRENT_GRAPH_VERSION:
        raise DomainConflictError("action preparation requires the current graph version")
    if run.resume_document_id != command.args.resume_document_id:
        raise DomainInvariantError("action resume document does not match its run")
    return run


def _derive_facts(command: PrepareActionCommand) -> _ProposalFacts:
    args_snapshot, _args_bytes, args_digest = canonicalize_action_args(command.args)
    target_snapshot, _target_bytes, target_digest = canonicalize_trusted_target()
    policy_snapshot = fixed_approval_policy().model_dump(mode="json", round_trip=True)
    binding = ApprovalBindingV1(
        workspace_id=command.tenant.workspace_id,
        run_id=command.run_id,
        action_intent_id=command.action_proposal_id,
        action_key=command.action_key,
        action_revision=command.action_revision,
        tool_name=SUBMIT_APPLICATION_TOOL_NAME,
        effect=ToolEffect.IRREVERSIBLE,
        args_digest=args_digest,
        target_digest=target_digest,
        policy_version=POLICY_VERSION,
    )
    _binding_bytes, binding_digest = approval_binding_digest(binding)
    return _ProposalFacts(
        workspace_id=command.tenant.workspace_id,
        actor_user_id=command.tenant.actor_user_id,
        run_id=command.run_id,
        action_intent_id=command.action_proposal_id,
        action_key=command.action_key,
        action_revision=command.action_revision,
        args_snapshot=args_snapshot,
        args_digest=args_digest,
        target_snapshot=target_snapshot,
        target_digest=target_digest,
        approval_binding_digest=binding_digest,
        policy_snapshot=policy_snapshot,
        expires_at=command.expires_at,
    )


def _append_event(
    session: AsyncSession,
    *,
    run: Run,
    actor_user_id: UUID,
    event_type: RunEventType,
    payload: dict[str, object],
) -> None:
    sequence = run.next_event_seq
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise DomainInvariantError("persisted run event sequence is invalid")
    run.next_event_seq = sequence + 1
    session.add(
        RunEvent(
            id=uuid4(),
            workspace_id=run.workspace_id,
            run_id=run.id,
            actor_user_id=actor_user_id,
            seq=sequence,
            type=event_type.value,
            version=CURRENT_RUN_EVENT_VERSION,
            payload=payload,
        )
    )


def _action_proposed_payload(facts: _ProposalFacts) -> dict[str, object]:
    return {
        "action_intent_id": str(facts.action_intent_id),
        "action_key": facts.action_key,
        "action_revision": facts.action_revision,
        "tool_name": SUBMIT_APPLICATION_TOOL_NAME,
        "effect": ToolEffect.IRREVERSIBLE.value,
        "args_digest": facts.args_digest,
        "target_digest": facts.target_digest,
        "approval_binding_digest": facts.approval_binding_digest,
    }


async def _proposal_events(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    run_id: UUID,
    action_intent_id: UUID,
) -> list[RunEvent]:
    events = list(
        await session.scalars(
            select(RunEvent).where(
                RunEvent.workspace_id == workspace_id,
                RunEvent.run_id == run_id,
                RunEvent.type == RunEventType.ACTION_PROPOSED.value,
            )
        )
    )
    return [
        event for event in events if event.payload.get("action_intent_id") == str(action_intent_id)
    ]


async def _superseded_events(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    run_id: UUID,
    action_intent_id: UUID,
) -> list[RunEvent]:
    events = list(
        await session.scalars(
            select(RunEvent).where(
                RunEvent.workspace_id == workspace_id,
                RunEvent.run_id == run_id,
                RunEvent.type == RunEventType.APPROVAL_EXPIRED.value,
            )
        )
    )
    return [
        event
        for event in events
        if event.payload.get("action_intent_id") == str(action_intent_id)
        and event.payload.get("reason") == "superseded"
    ]


def _intent_matches(row: ActionIntent, facts: _ProposalFacts) -> bool:
    return (
        row.id == facts.action_intent_id
        and row.workspace_id == facts.workspace_id
        and row.originating_actor_user_id == facts.actor_user_id
        and row.run_id == facts.run_id
        and row.action_key == facts.action_key
        and row.action_revision == facts.action_revision
        and row.tool_name == SUBMIT_APPLICATION_TOOL_NAME
        and row.effect == ToolEffect.IRREVERSIBLE.value
        and row.args_snapshot == facts.args_snapshot
        and row.canonicalization_version == CANONICALIZATION_VERSION
        and row.args_digest == facts.args_digest
        and row.target_snapshot == facts.target_snapshot
        and row.target_canonicalization_version == TARGET_CANONICALIZATION_VERSION
        and row.target_digest == facts.target_digest
        and row.approval_binding_version == APPROVAL_BINDING_VERSION
        and row.approval_binding_digest == facts.approval_binding_digest
        and row.idempotency_key == str(facts.action_intent_id)
    )


def _request_matches(row: ApprovalRequest, facts: _ProposalFacts) -> bool:
    return (
        row.workspace_id == facts.workspace_id
        and row.run_id == facts.run_id
        and row.action_intent_id == facts.action_intent_id
        and row.args_digest == facts.args_digest
        and row.target_digest == facts.target_digest
        and row.approval_binding_version == APPROVAL_BINDING_VERSION
        and row.approval_binding_digest == facts.approval_binding_digest
        and row.policy_version == POLICY_VERSION
        and row.policy_snapshot == facts.policy_snapshot
        and row.expires_at == facts.expires_at
    )


async def _read_intent_candidate(
    session: AsyncSession,
    facts: _ProposalFacts,
) -> ActionIntent | None:
    rows = list(
        await session.scalars(
            select(ActionIntent)
            .where(
                ActionIntent.workspace_id == facts.workspace_id,
                or_(
                    ActionIntent.id == facts.action_intent_id,
                    and_(
                        ActionIntent.run_id == facts.run_id,
                        ActionIntent.action_key == facts.action_key,
                        ActionIntent.action_revision == facts.action_revision,
                    ),
                ),
            )
            .with_for_update()
        )
    )
    if len(rows) > 1:
        raise DomainInvariantError("action proposal identities resolve to different rows")
    return rows[0] if rows else None


async def _read_request(
    session: AsyncSession,
    facts: _ProposalFacts,
) -> ApprovalRequest | None:
    return await session.scalar(
        select(ApprovalRequest)
        .where(
            ApprovalRequest.workspace_id == facts.workspace_id,
            ApprovalRequest.action_intent_id == facts.action_intent_id,
        )
        .with_for_update()
    )


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


class SqlAlchemyActionStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        fault_injector: ActionStoreFaultInjector | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._fault_injector = fault_injector

    async def _inject(self, point: ActionStoreFaultPoint) -> None:
        if self._fault_injector is None:
            return
        result = self._fault_injector(point)
        if isawaitable(result):
            await result

    async def _insert_proposal(
        self,
        session: AsyncSession,
        *,
        run: Run,
        command: PrepareActionCommand,
        allow_create: bool,
    ) -> PreparedAction:
        facts = _derive_facts(command)
        inserted_intent_id = None
        if allow_create:
            inserted_intent_id = await session.scalar(
                insert(ActionIntent)
                .values(
                    id=facts.action_intent_id,
                    workspace_id=facts.workspace_id,
                    originating_actor_user_id=facts.actor_user_id,
                    run_id=facts.run_id,
                    action_key=facts.action_key,
                    action_revision=facts.action_revision,
                    tool_name=SUBMIT_APPLICATION_TOOL_NAME,
                    effect=ToolEffect.IRREVERSIBLE.value,
                    args_snapshot=facts.args_snapshot,
                    canonicalization_version=CANONICALIZATION_VERSION,
                    args_digest=facts.args_digest,
                    target_snapshot=facts.target_snapshot,
                    target_canonicalization_version=TARGET_CANONICALIZATION_VERSION,
                    target_digest=facts.target_digest,
                    approval_binding_version=APPROVAL_BINDING_VERSION,
                    approval_binding_digest=facts.approval_binding_digest,
                    status=ActionIntentStatus.PROPOSED.value,
                    idempotency_key=str(facts.action_intent_id),
                    recovery_attempts=0,
                    result=None,
                    evidence=None,
                )
                .on_conflict_do_nothing()
                .returning(ActionIntent.id)
            )
        created_intent = inserted_intent_id is not None
        intent = await _read_intent_candidate(session, facts)
        if intent is None:
            raise DomainInvariantError("action proposal conflict has no authoritative row")
        if not _intent_matches(intent, facts):
            raise DomainInvariantError("action proposal immutable facts conflict")
        persisted_action_intent_status(intent.status)

        request_id = uuid4()
        if allow_create:
            await session.scalar(
                insert(ApprovalRequest)
                .values(
                    id=request_id,
                    workspace_id=facts.workspace_id,
                    run_id=facts.run_id,
                    action_intent_id=facts.action_intent_id,
                    status=ApprovalStatus.PENDING.value,
                    args_digest=facts.args_digest,
                    target_digest=facts.target_digest,
                    approval_binding_version=APPROVAL_BINDING_VERSION,
                    approval_binding_digest=facts.approval_binding_digest,
                    policy_version=POLICY_VERSION,
                    policy_snapshot=facts.policy_snapshot,
                    version=1,
                    expires_at=facts.expires_at,
                    consumed_at=None,
                )
                .on_conflict_do_nothing()
                .returning(ApprovalRequest.id)
            )
        request = await _read_request(session, facts)
        if request is None:
            raise DomainInvariantError("action proposal has no approval request")
        if not _request_matches(request, facts):
            raise DomainInvariantError("approval request immutable facts conflict")
        persisted_approval_status(request.status)

        existing_events = await _proposal_events(
            session,
            workspace_id=facts.workspace_id,
            run_id=facts.run_id,
            action_intent_id=facts.action_intent_id,
        )
        if created_intent:
            if existing_events:
                raise DomainInvariantError("new action proposal already has an event")
            _append_event(
                session,
                run=run,
                actor_user_id=facts.actor_user_id,
                event_type=RunEventType.ACTION_PROPOSED,
                payload=_action_proposed_payload(facts),
            )
        elif len(existing_events) != 1 or existing_events[0].payload != (
            _action_proposed_payload(facts)
        ):
            raise DomainInvariantError("persisted action proposal event is inconsistent")

        await session.flush()
        return PreparedAction(
            intent=_intent_record(intent),
            approval_request=_request_record(request),
        )

    async def prepare_action(self, command: PrepareActionCommand) -> PreparedAction:
        if not isinstance(command, PrepareActionCommand):
            raise DomainValidationError("prepare action command is invalid")
        if command.action_key != ACTION_KEY:
            raise DomainConflictError("initial action proposal key is invalid")
        if command.action_revision != INITIAL_ACTION_REVISION:
            raise DomainConflictError("initial action proposal revision must be one")
        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, command.tenant)
            run = await _lock_application_run(session, command)
            prepared = await self._insert_proposal(
                session,
                run=run,
                command=command,
                allow_create=True,
            )
            await self._inject("prepare_before_commit")
        return prepared

    async def supersede_action(self, command: SupersedeActionCommand) -> PreparedAction:
        if not isinstance(command, SupersedeActionCommand):
            raise DomainValidationError("supersede action command is invalid")
        if not isinstance(command.old_action_intent_id, UUID):
            raise DomainValidationError("old action intent identity is invalid")
        prepare = command.as_prepare_command()
        if command.old_action_intent_id == command.new_action_proposal_id:
            raise DomainConflictError("supersede requires a new proposal identity")

        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, command.tenant)
            run = await _lock_application_run(session, prepare)
            old_intent = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == command.tenant.workspace_id,
                    ActionIntent.run_id == command.run_id,
                    ActionIntent.id == command.old_action_intent_id,
                )
                .with_for_update()
            )
            if (
                old_intent is None
                or old_intent.originating_actor_user_id != command.tenant.actor_user_id
            ):
                raise DomainNotFoundError
            old_request = await session.scalar(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.workspace_id == command.tenant.workspace_id,
                    ApprovalRequest.run_id == command.run_id,
                    ApprovalRequest.action_intent_id == command.old_action_intent_id,
                )
                .with_for_update()
            )
            if old_request is None:
                raise DomainInvariantError("action intent has no approval request")
            if (
                command.action_key != old_intent.action_key
                or command.action_revision != old_intent.action_revision + 1
            ):
                raise DomainConflictError("supersede requires the same key and next revision")

            intent_status = persisted_action_intent_status(old_intent.status)
            request_status = persisted_approval_status(old_request.status)
            if request_status is ApprovalStatus.CONSUMED:
                raise DomainConflictError("consumed approval cannot be superseded")

            if intent_status is ActionIntentStatus.CANCELLED:
                if request_status not in {
                    ApprovalStatus.REJECTED,
                    ApprovalStatus.EXPIRED,
                }:
                    raise DomainConflictError("cancelled action has no superseded approval")
                events = await _superseded_events(
                    session,
                    workspace_id=command.tenant.workspace_id,
                    run_id=command.run_id,
                    action_intent_id=command.old_action_intent_id,
                )
                expected_event_payload = {
                    "approval_request_id": str(old_request.id),
                    "action_intent_id": str(old_intent.id),
                    "action_key": old_intent.action_key,
                    "action_revision": old_intent.action_revision,
                    "reason": "superseded",
                }
                if (
                    len(events) > 1
                    or (request_status is ApprovalStatus.REJECTED and events)
                    or (events and events[0].payload != expected_event_payload)
                ):
                    raise DomainInvariantError("superseded action event is inconsistent")
                return await self._insert_proposal(
                    session,
                    run=run,
                    command=prepare,
                    allow_create=False,
                )

            if intent_status not in {
                ActionIntentStatus.PROPOSED,
                ActionIntentStatus.AUTHORIZED,
            }:
                raise DomainConflictError("action state cannot be superseded")
            validate_action_intent_transition(intent_status, ActionIntentStatus.CANCELLED)
            old_intent.status = ActionIntentStatus.CANCELLED.value

            expired_by_supersede = request_status in {
                ApprovalStatus.PENDING,
                ApprovalStatus.APPROVED,
            }
            if expired_by_supersede:
                validate_approval_transition(request_status, ApprovalStatus.EXPIRED)
                old_request.status = ApprovalStatus.EXPIRED.value
                old_request.version += 1
                old_request.consumed_at = None
                _append_event(
                    session,
                    run=run,
                    actor_user_id=command.tenant.actor_user_id,
                    event_type=RunEventType.APPROVAL_EXPIRED,
                    payload={
                        "approval_request_id": str(old_request.id),
                        "action_intent_id": str(old_intent.id),
                        "action_key": old_intent.action_key,
                        "action_revision": old_intent.action_revision,
                        "reason": "superseded",
                    },
                )
            elif request_status not in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
                raise DomainInvariantError("persisted approval status cannot be superseded")

            await session.flush()
            await self._inject("supersede_after_expiry_before_new_proposal")
            return await self._insert_proposal(
                session,
                run=run,
                command=prepare,
                allow_create=True,
            )

    async def cancel_action(self, command: CancelActionCommand) -> CancelledAction:
        if not isinstance(command, CancelActionCommand):
            raise DomainValidationError("cancel action command is invalid")
        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, command.tenant)
            request = await session.scalar(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.workspace_id == command.tenant.workspace_id,
                    ApprovalRequest.run_id == command.run_id,
                    ApprovalRequest.id == command.approval_request_id,
                    ApprovalRequest.action_intent_id == command.action_intent_id,
                )
                .with_for_update()
            )
            if request is None:
                raise DomainNotFoundError
            run = await session.scalar(
                select(Run)
                .where(
                    Run.workspace_id == command.tenant.workspace_id,
                    Run.id == command.run_id,
                )
                .with_for_update()
            )
            intent = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == command.tenant.workspace_id,
                    ActionIntent.run_id == command.run_id,
                    ActionIntent.id == command.action_intent_id,
                )
                .with_for_update()
            )
            if run is None or intent is None:
                raise DomainInvariantError("cancel action binding is missing")
            validate_exact_approval_binding(
                intent=_intent_record(intent), request=_request_record(request)
            )
            expected_status = (
                ApprovalStatus.REJECTED
                if command.reason == "approval_rejected"
                else ApprovalStatus.EXPIRED
            )
            if persisted_approval_status(request.status) is not expected_status:
                raise DomainConflictError("approval status cannot cancel action")
            status = persisted_action_intent_status(intent.status)
            events = list(
                await session.scalars(
                    select(RunEvent).where(
                        RunEvent.workspace_id == command.tenant.workspace_id,
                        RunEvent.run_id == command.run_id,
                        RunEvent.type == RunEventType.ACTION_CANCELLED.value,
                    )
                )
            )
            matching = [
                event for event in events if event.payload.get("action_intent_id") == str(intent.id)
            ]
            payload = {
                "action_intent_id": str(intent.id),
                "approval_request_id": str(request.id),
                "reason": command.reason,
            }
            if status is ActionIntentStatus.CANCELLED:
                if len(matching) != 1 or matching[0].payload != payload:
                    raise DomainInvariantError("cancelled action event is inconsistent")
                return CancelledAction(intent=_intent_record(intent), changed=False)
            if status not in {ActionIntentStatus.PROPOSED, ActionIntentStatus.AUTHORIZED}:
                raise DomainConflictError("action state cannot be cancelled")
            validate_action_intent_transition(status, ActionIntentStatus.CANCELLED)
            intent.status = ActionIntentStatus.CANCELLED.value
            _append_event(
                session,
                run=run,
                actor_user_id=command.tenant.actor_user_id,
                event_type=RunEventType.ACTION_CANCELLED,
                payload=payload,
            )
            await session.flush()
            await self._inject("cancel_before_commit")
            result = CancelledAction(intent=_intent_record(intent), changed=True)
        return result
