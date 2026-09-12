from __future__ import annotations

import json
from datetime import datetime
from inspect import isawaitable
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import func, select

from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    Run,
    RunEvent,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.session import AsyncSessionFactory, transaction
from app.domain.action_execution import (
    ActionApprovalExpiredError,
    ActionExecutionIdentity,
    ActionExecutionStoreFaultInjector,
    ActionExecutionStoreFaultPoint,
    ConfirmedActionResult,
    PreparedActionExecution,
    ReconciliationAuthorization,
    ResendAuthorization,
    SendAuthorization,
)
from app.domain.actions import (
    SUBMIT_APPLICATION_TOOL_NAME,
    ActionIntentRecord,
    ActionIntentStatus,
    SubmitApplicationArgsV1,
    TrustedActionTargetV1,
    persisted_action_intent_status,
    validate_action_intent_transition,
    validate_exact_approval_binding,
)
from app.domain.approvals import (
    ApprovalRequestRecord,
    ApprovalStatus,
    persisted_approval_status,
    validate_approval_transition,
)
from app.domain.errors import DomainConflictError, DomainInvariantError, DomainNotFoundError
from app.domain.runs import CURRENT_GRAPH_VERSION, RunMode, RunStatus
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationStatus
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


def _append_action_event(
    session: object,
    *,
    run: Run,
    event_type: RunEventType,
    intent: ActionIntent,
    invocation: ToolInvocation,
    error_category: str | None = None,
) -> None:
    sequence = run.next_event_seq
    if not isinstance(sequence, int) or sequence < 1:
        raise DomainInvariantError("persisted run event sequence is invalid")
    run.next_event_seq = sequence + 1
    payload: dict[str, object] = {
        "action_intent_id": str(intent.id),
        "invocation_id": str(invocation.id),
        "status": intent.status,
        "recovery_attempts": intent.recovery_attempts,
    }
    if error_category is not None:
        payload["error_category"] = error_category
    session.add(  # type: ignore[attr-defined]
        RunEvent(
            id=uuid4(),
            workspace_id=run.workspace_id,
            run_id=run.id,
            actor_user_id=None,
            seq=sequence,
            type=event_type.value,
            version=CURRENT_RUN_EVENT_VERSION,
            payload=payload,
        )
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


def _prepared(
    intent: ActionIntent,
    request: ApprovalRequest,
    invocation: ToolInvocation,
) -> PreparedActionExecution:
    try:
        args = SubmitApplicationArgsV1.model_validate_json(
            json.dumps(intent.args_snapshot, allow_nan=False, separators=(",", ":")),
            strict=True,
        )
        target = TrustedActionTargetV1.model_validate_json(
            json.dumps(intent.target_snapshot, allow_nan=False, separators=(",", ":")),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError):
        raise DomainInvariantError("persisted approved action payload is invalid") from None
    return PreparedActionExecution(
        workspace_id=intent.workspace_id,
        originating_actor_user_id=intent.originating_actor_user_id,
        run_id=intent.run_id,
        action_intent_id=intent.id,
        approval_request_id=request.id,
        invocation_id=invocation.id,
        tool_name=intent.tool_name,
        args=args,
        target=target,
        idempotency_key=intent.idempotency_key,
        status=intent.status,
        recovery_attempts=intent.recovery_attempts,
        result=dict(intent.result) if intent.result is not None else None,
        evidence=dict(intent.evidence) if intent.evidence is not None else None,
    )


class SqlAlchemyActionExecutionStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        fault_injector: ActionExecutionStoreFaultInjector | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._fault_injector = fault_injector

    async def _inject(self, point: ActionExecutionStoreFaultPoint) -> None:
        if self._fault_injector is None:
            return
        result = self._fault_injector(point)
        if isawaitable(result):
            await result

    @staticmethod
    async def _originating_actor(session: object, identity: ActionExecutionIdentity) -> UUID:
        actor = await session.scalar(  # type: ignore[attr-defined]
            select(Run.created_by_user_id).where(
                Run.workspace_id == identity.workspace_id,
                Run.id == identity.run_id,
            )
        )
        if actor is None:
            raise DomainNotFoundError
        return actor

    @staticmethod
    async def _lock_membership(session: object, workspace_id: UUID, actor_user_id: UUID) -> bool:
        row = await session.scalar(  # type: ignore[attr-defined]
            select(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == workspace_id,
                WorkspaceMembership.user_id == actor_user_id,
            )
            .with_for_update()
        )
        return row is not None and row.revoked_at is None

    @staticmethod
    async def _lock_bound_rows(session: object, identity: ActionExecutionIdentity):
        request = await session.scalar(  # type: ignore[attr-defined]
            select(ApprovalRequest)
            .where(
                ApprovalRequest.workspace_id == identity.workspace_id,
                ApprovalRequest.run_id == identity.run_id,
                ApprovalRequest.id == identity.approval_request_id,
                ApprovalRequest.action_intent_id == identity.action_intent_id,
            )
            .with_for_update()
        )
        if request is None:
            raise DomainNotFoundError
        run = await session.scalar(  # type: ignore[attr-defined]
            select(Run)
            .where(Run.workspace_id == identity.workspace_id, Run.id == identity.run_id)
            .with_for_update()
        )
        intent = await session.scalar(  # type: ignore[attr-defined]
            select(ActionIntent)
            .where(
                ActionIntent.workspace_id == identity.workspace_id,
                ActionIntent.run_id == identity.run_id,
                ActionIntent.id == identity.action_intent_id,
            )
            .with_for_update()
        )
        if run is None or intent is None:
            raise DomainInvariantError("approved action execution binding is missing")
        if (
            run.created_by_user_id != intent.originating_actor_user_id
            or run.mode != RunMode.APPLICATION.value
            or run.graph_version != CURRENT_GRAPH_VERSION
        ):
            raise DomainInvariantError("approved action execution run facts conflict")
        validate_exact_approval_binding(
            intent=_intent_record(intent), request=_request_record(request)
        )
        return request, run, intent

    @staticmethod
    def _validate_invocation(invocation: ToolInvocation, intent: ActionIntent) -> None:
        if (
            invocation.workspace_id != intent.workspace_id
            or invocation.run_id != intent.run_id
            or invocation.originating_actor_user_id != intent.originating_actor_user_id
            or invocation.action_intent_id != intent.id
            or invocation.tool_name != SUBMIT_APPLICATION_TOOL_NAME
            or invocation.effect != ToolEffect.IRREVERSIBLE.value
            or invocation.args_digest != intent.args_digest
        ):
            raise DomainInvariantError("irreversible invocation facts conflict")

    async def prepare_execution(
        self,
        identity: ActionExecutionIdentity,
        *,
        now: datetime,
    ) -> PreparedActionExecution:
        if not isinstance(identity, ActionExecutionIdentity) or now.tzinfo is None:
            raise TypeError("action execution preparation input is invalid")
        expired = False
        revoked = False
        result: PreparedActionExecution | None = None
        async with transaction(self._session_factory) as session:
            actor_user_id = await self._originating_actor(session, identity)
            membership_active = await self._lock_membership(
                session, identity.workspace_id, actor_user_id
            )
            request, run, intent = await self._lock_bound_rows(session, identity)
            request_status = persisted_approval_status(request.status)
            intent_status = persisted_action_intent_status(intent.status)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if intent_status in {
                ActionIntentStatus.EXECUTING,
                ActionIntentStatus.SUCCEEDED,
                ActionIntentStatus.FAILED,
                ActionIntentStatus.OUTCOME_UNKNOWN,
            }:
                if invocation is None:
                    raise DomainInvariantError("irreversible invocation is missing")
                self._validate_invocation(invocation, intent)
                expected_invocation_status = {
                    ActionIntentStatus.EXECUTING: ToolInvocationStatus.EXECUTING,
                    ActionIntentStatus.SUCCEEDED: ToolInvocationStatus.SUCCEEDED,
                    ActionIntentStatus.FAILED: ToolInvocationStatus.FAILED,
                    ActionIntentStatus.OUTCOME_UNKNOWN: ToolInvocationStatus.OUTCOME_UNKNOWN,
                }[intent_status]
                if (
                    request_status is not ApprovalStatus.CONSUMED
                    or invocation.status != expected_invocation_status.value
                    or invocation.attempt < 1
                ):
                    raise DomainInvariantError("persisted action recovery facts conflict")
                result = _prepared(intent, request, invocation)
            elif not membership_active:
                if request_status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
                    validate_approval_transition(request_status, ApprovalStatus.EXPIRED)
                    request.status = ApprovalStatus.EXPIRED.value
                    request.version += 1
                    request.consumed_at = None
                if intent_status in {
                    ActionIntentStatus.PROPOSED,
                    ActionIntentStatus.AUTHORIZED,
                }:
                    validate_action_intent_transition(intent_status, ActionIntentStatus.CANCELLED)
                    intent.status = ActionIntentStatus.CANCELLED.value
                if invocation is not None:
                    self._validate_invocation(invocation, intent)
                    if invocation.status != ToolInvocationStatus.PREPARED.value:
                        raise DomainInvariantError(
                            "revoked action invocation may already have been sent"
                        )
                    invocation.status = ToolInvocationStatus.FAILED.value
                    invocation.latency_ms = 0
                    invocation.error_category = "aborted_before_send"
                    invocation.finished_at = now
                revoked = True
            elif run.status != RunStatus.RUNNING.value or run.cancel_requested_at is not None:
                raise DomainConflictError("run does not authorize action preparation")
            elif request_status is ApprovalStatus.APPROVED and request.expires_at <= now:
                validate_approval_transition(request_status, ApprovalStatus.EXPIRED)
                request.status = ApprovalStatus.EXPIRED.value
                request.version += 1
                request.consumed_at = None
                expired = True
            elif (
                request_status is ApprovalStatus.CONSUMED
                and intent_status is ActionIntentStatus.AUTHORIZED
                and invocation is not None
            ):
                self._validate_invocation(invocation, intent)
                if (
                    invocation.status != ToolInvocationStatus.PREPARED.value
                    or invocation.attempt != 0
                ):
                    raise DomainInvariantError("prepared execution retry facts conflict")
                result = _prepared(intent, request, invocation)
            else:
                if request_status is not ApprovalStatus.APPROVED:
                    raise DomainConflictError("approval request is not executable")
                if intent_status is not ActionIntentStatus.PROPOSED:
                    raise DomainConflictError("action intent is not executable")
                if invocation is not None:
                    raise DomainInvariantError("action already has a conflicting invocation")
                decision_count = await session.scalar(
                    select(func.count())
                    .select_from(ApprovalDecision)
                    .where(
                        ApprovalDecision.workspace_id == identity.workspace_id,
                        ApprovalDecision.approval_request_id == request.id,
                        ApprovalDecision.decision == "approve",
                    )
                )
                if decision_count != 1:
                    raise DomainInvariantError("approved request decision facts conflict")
                validate_approval_transition(request_status, ApprovalStatus.CONSUMED)
                validate_action_intent_transition(intent_status, ActionIntentStatus.AUTHORIZED)
                request.status = ApprovalStatus.CONSUMED.value
                request.consumed_at = now
                request.version += 1
                intent.status = ActionIntentStatus.AUTHORIZED.value
                invocation = ToolInvocation(
                    id=uuid4(),
                    workspace_id=intent.workspace_id,
                    originating_actor_user_id=intent.originating_actor_user_id,
                    run_id=intent.run_id,
                    action_intent_id=intent.id,
                    tool_name=intent.tool_name,
                    effect=intent.effect,
                    args_digest=intent.args_digest,
                    status=ToolInvocationStatus.PREPARED.value,
                    attempt=0,
                    latency_ms=None,
                    result_summary=None,
                    error_category=None,
                    started_at=None,
                    finished_at=None,
                )
                session.add(invocation)
                await session.flush()
                await self._inject("consume_before_commit")
                result = _prepared(intent, request, invocation)
        if expired:
            raise ActionApprovalExpiredError
        if revoked:
            raise DomainNotFoundError
        if result is None:
            raise DomainInvariantError("action preparation produced no execution")
        return result

    async def begin_send(
        self,
        identity: ActionExecutionIdentity,
        *,
        now: datetime,
    ) -> SendAuthorization:
        if not isinstance(identity, ActionExecutionIdentity) or now.tzinfo is None:
            raise TypeError("action send authorization input is invalid")
        async with transaction(self._session_factory) as session:
            actor_user_id = await self._originating_actor(session, identity)
            membership_active = await self._lock_membership(
                session, identity.workspace_id, actor_user_id
            )
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("prepared irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status)
                is not ActionIntentStatus.AUTHORIZED
                or invocation.status != ToolInvocationStatus.PREPARED.value
                or invocation.attempt != 0
            ):
                raise DomainConflictError("pre-send action state changed")
            if (
                not membership_active
                or run.status != RunStatus.RUNNING.value
                or run.cancel_requested_at is not None
            ):
                category = (
                    "aborted_before_send" if not membership_active else "cancelled_before_send"
                )
                validate_action_intent_transition(
                    ActionIntentStatus.AUTHORIZED, ActionIntentStatus.CANCELLED
                )
                intent.status = ActionIntentStatus.CANCELLED.value
                invocation.status = ToolInvocationStatus.FAILED.value
                invocation.latency_ms = 0
                invocation.error_category = category
                invocation.finished_at = now
                return SendAuthorization(
                    execution=_prepared(intent, request, invocation),
                    allowed=False,
                    error_category=category,
                )
            validate_action_intent_transition(
                ActionIntentStatus.AUTHORIZED, ActionIntentStatus.EXECUTING
            )
            intent.status = ActionIntentStatus.EXECUTING.value
            invocation.status = ToolInvocationStatus.EXECUTING.value
            invocation.attempt = 1
            invocation.started_at = now
            _append_action_event(
                session,
                run=run,
                event_type=RunEventType.ACTION_STARTED,
                intent=intent,
                invocation=invocation,
            )
            await session.flush()
            await self._inject("send_transition_before_commit")
            return SendAuthorization(execution=_prepared(intent, request, invocation), allowed=True)

    async def begin_reconciliation(
        self,
        identity: ActionExecutionIdentity,
        *,
        max_attempts: int,
        now: datetime,
    ) -> ReconciliationAuthorization:
        if (
            not isinstance(identity, ActionExecutionIdentity)
            or isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
            or now.tzinfo is None
        ):
            raise TypeError("action reconciliation input is invalid")
        async with transaction(self._session_factory) as session:
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("executing irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status) is not ActionIntentStatus.EXECUTING
                or invocation.status != ToolInvocationStatus.EXECUTING.value
                or invocation.attempt < 1
            ):
                raise DomainConflictError("action is not reconcilable")
            if intent.recovery_attempts >= max_attempts:
                self._mark_unknown(
                    session,
                    run=run,
                    intent=intent,
                    invocation=invocation,
                    evidence={"classification": "recovery_exhausted"},
                    latency_ms=0,
                    now=now,
                )
                return ReconciliationAuthorization(
                    execution=_prepared(intent, request, invocation),
                    recovery_attempt=intent.recovery_attempts,
                    network_allowed=False,
                )
            intent.recovery_attempts += 1
            await session.flush()
            await self._inject("recovery_attempt_before_commit")
            return ReconciliationAuthorization(
                execution=_prepared(intent, request, invocation),
                recovery_attempt=intent.recovery_attempts,
                network_allowed=True,
            )

    async def authorize_resend(
        self,
        identity: ActionExecutionIdentity,
        *,
        expected_recovery_attempt: int,
        now: datetime,
    ) -> ResendAuthorization:
        if (
            not isinstance(identity, ActionExecutionIdentity)
            or isinstance(expected_recovery_attempt, bool)
            or expected_recovery_attempt < 1
            or now.tzinfo is None
        ):
            raise TypeError("recovery resend authorization input is invalid")
        async with transaction(self._session_factory) as session:
            actor_user_id = await self._originating_actor(session, identity)
            membership_active = await self._lock_membership(
                session, identity.workspace_id, actor_user_id
            )
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("executing irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status) is not ActionIntentStatus.EXECUTING
                or invocation.status != ToolInvocationStatus.EXECUTING.value
                or intent.recovery_attempts != expected_recovery_attempt
            ):
                raise DomainConflictError("recovery resend state changed")
            category: Literal["cancelled_confirmed_absent", "authorization_revoked"] | None = None
            if not membership_active:
                category = "authorization_revoked"
            elif run.cancel_requested_at is not None or run.status != RunStatus.RUNNING.value:
                category = "cancelled_confirmed_absent"
            if category is not None:
                self._mark_failed(
                    session,
                    run=run,
                    intent=intent,
                    invocation=invocation,
                    error_category=category,
                    evidence={
                        "classification": "confirmed_absent",
                        "reason": category,
                        "recovery_attempt": expected_recovery_attempt,
                    },
                    latency_ms=0,
                    now=now,
                )
                return ResendAuthorization(
                    execution=_prepared(intent, request, invocation),
                    recovery_attempt=expected_recovery_attempt,
                    allowed=False,
                    error_category=category,
                )
            invocation.attempt += 1
            await session.flush()
            await self._inject("resend_transition_before_commit")
            return ResendAuthorization(
                execution=_prepared(intent, request, invocation),
                recovery_attempt=expected_recovery_attempt,
                allowed=True,
            )

    @staticmethod
    def _mark_failed(
        session: object,
        *,
        run: Run,
        intent: ActionIntent,
        invocation: ToolInvocation,
        error_category: str,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None:
        validate_action_intent_transition(ActionIntentStatus.EXECUTING, ActionIntentStatus.FAILED)
        intent.status = ActionIntentStatus.FAILED.value
        intent.result = None
        intent.evidence = evidence
        invocation.status = ToolInvocationStatus.FAILED.value
        invocation.latency_ms = latency_ms
        invocation.result_summary = None
        invocation.error_category = error_category
        invocation.finished_at = now
        _append_action_event(
            session,
            run=run,
            event_type=RunEventType.ACTION_FAILED,
            intent=intent,
            invocation=invocation,
            error_category=error_category,
        )

    @staticmethod
    def _mark_unknown(
        session: object,
        *,
        run: Run,
        intent: ActionIntent,
        invocation: ToolInvocation,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None:
        validate_action_intent_transition(
            ActionIntentStatus.EXECUTING, ActionIntentStatus.OUTCOME_UNKNOWN
        )
        intent.status = ActionIntentStatus.OUTCOME_UNKNOWN.value
        intent.result = None
        intent.evidence = evidence
        invocation.status = ToolInvocationStatus.OUTCOME_UNKNOWN.value
        invocation.latency_ms = latency_ms
        invocation.result_summary = None
        invocation.error_category = "external_outcome_unknown"
        invocation.finished_at = now
        _append_action_event(
            session,
            run=run,
            event_type=RunEventType.ACTION_OUTCOME_UNKNOWN,
            intent=intent,
            invocation=invocation,
            error_category="external_outcome_unknown",
        )

    async def confirm_success(
        self,
        identity: ActionExecutionIdentity,
        *,
        result: ConfirmedActionResult,
        latency_ms: int,
        now: datetime,
    ) -> None:
        if (
            not isinstance(identity, ActionExecutionIdentity)
            or not isinstance(result, ConfirmedActionResult)
            or isinstance(latency_ms, bool)
            or latency_ms < 0
            or now.tzinfo is None
        ):
            raise TypeError("confirmed action result input is invalid")
        async with transaction(self._session_factory) as session:
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("executing irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status) is not ActionIntentStatus.EXECUTING
                or invocation.status != ToolInvocationStatus.EXECUTING.value
                or invocation.attempt < 1
            ):
                raise DomainConflictError("confirmed action state changed")
            validate_action_intent_transition(
                ActionIntentStatus.EXECUTING, ActionIntentStatus.SUCCEEDED
            )
            intent.status = ActionIntentStatus.SUCCEEDED.value
            intent.result = {"external_ref": result.external_ref}
            intent.evidence = {
                "classification": "confirmed_success",
                "idempotency_key": intent.idempotency_key,
                **(
                    {
                        "payload_digest": result.payload_digest,
                        "source": result.source,
                    }
                    if result.payload_digest is not None and result.source is not None
                    else {}
                ),
            }
            invocation.status = ToolInvocationStatus.SUCCEEDED.value
            invocation.latency_ms = latency_ms
            invocation.result_summary = {"external_ref": result.external_ref}
            invocation.finished_at = now
            _append_action_event(
                session,
                run=run,
                event_type=RunEventType.ACTION_COMPLETED,
                intent=intent,
                invocation=invocation,
            )

    async def confirm_failure(
        self,
        identity: ActionExecutionIdentity,
        *,
        error_category: str,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None:
        if (
            not isinstance(identity, ActionExecutionIdentity)
            or not isinstance(error_category, str)
            or not error_category
            or not isinstance(evidence, dict)
            or isinstance(latency_ms, bool)
            or latency_ms < 0
            or now.tzinfo is None
        ):
            raise TypeError("confirmed action failure input is invalid")
        async with transaction(self._session_factory) as session:
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("executing irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status) is not ActionIntentStatus.EXECUTING
                or invocation.status != ToolInvocationStatus.EXECUTING.value
            ):
                raise DomainConflictError("confirmed action failure state changed")
            self._mark_failed(
                session,
                run=run,
                intent=intent,
                invocation=invocation,
                error_category=error_category,
                evidence=evidence,
                latency_ms=latency_ms,
                now=now,
            )

    async def confirm_outcome_unknown(
        self,
        identity: ActionExecutionIdentity,
        *,
        evidence: dict[str, object],
        latency_ms: int,
        now: datetime,
    ) -> None:
        if (
            not isinstance(identity, ActionExecutionIdentity)
            or not isinstance(evidence, dict)
            or isinstance(latency_ms, bool)
            or latency_ms < 0
            or now.tzinfo is None
        ):
            raise TypeError("unknown action outcome input is invalid")
        async with transaction(self._session_factory) as session:
            request, run, intent = await self._lock_bound_rows(session, identity)
            invocation = await session.scalar(
                select(ToolInvocation)
                .where(ToolInvocation.action_intent_id == intent.id)
                .with_for_update()
            )
            if invocation is None:
                raise DomainInvariantError("executing irreversible invocation is missing")
            self._validate_invocation(invocation, intent)
            if (
                persisted_approval_status(request.status) is not ApprovalStatus.CONSUMED
                or persisted_action_intent_status(intent.status) is not ActionIntentStatus.EXECUTING
                or invocation.status != ToolInvocationStatus.EXECUTING.value
            ):
                raise DomainConflictError("unknown action outcome state changed")
            self._mark_unknown(
                session,
                run=run,
                intent=intent,
                invocation=invocation,
                evidence=evidence,
                latency_ms=latency_ms,
                now=now,
            )
