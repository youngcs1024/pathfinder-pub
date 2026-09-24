from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.action_execution import _intent_record, _request_record
from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.resume_generation import ResumeGenerationPublisher
from app.db.resume_revision import ResumeRevisionPublisher
from app.db.session import AsyncSessionFactory, transaction
from app.domain.action_execution import ActionExecutionIdentity
from app.domain.actions import validate_exact_approval_binding
from app.domain.approvals import ApprovalStatus, persisted_approval_status
from app.domain.errors import DomainInvariantError
from app.domain.jobs import (
    ClaimedJob,
    JobStatus,
    PrepareClaimResult,
    ReclaimSummary,
    RetryDelayPolicy,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.run_payloads import (
    EXECUTION_CONTRACTS,
    LEGACY_RUN_MODES,
    ResumeGenerationCandidateOutputV1,
    ResumeRevisionCandidateOutputV1,
    RunContractV1,
    RunOutput,
    find_run_contract,
)
from app.domain.runs import RunMode, RunStatus
from app.domain.tenancy import TenantContext
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType

_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value}
)
_GATE6_RESUME_APPROVAL_STATUSES = frozenset(
    {
        ApprovalStatus.PENDING,
        ApprovalStatus.APPROVED,
        ApprovalStatus.REJECTED,
        ApprovalStatus.EXPIRED,
    }
)


def _valid_error_category(value: str) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 100
        and value[0].islower()
        and all(
            character.islower() or character.isdigit() or character == "_" for character in value
        )
    )


def _clear_lease(job: RunJob, *, status: JobStatus) -> None:
    job.status = status.value
    job.leased_by = None
    job.owner_token = None
    job.lease_expires_at = None


def _append_event(
    session: AsyncSession,
    *,
    run: Run,
    event_type: RunEventType,
    payload: dict[str, object],
) -> None:
    sequence = run.next_event_seq
    if not isinstance(sequence, int) or sequence < 1:
        raise DomainInvariantError("persisted run event sequence is invalid")
    run.next_event_seq = sequence + 1
    session.add(
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


def _terminal_time(run: Run, now: datetime) -> datetime:
    boundaries = tuple(
        value for value in (now, run.started_at, run.cancel_requested_at) if value is not None
    )
    return max(boundaries)


def _mark_run_cancelled(
    session: AsyncSession,
    *,
    run: Run,
    reason: str,
    now: datetime,
) -> None:
    if run.status in _TERMINAL_RUN_STATUSES:
        return
    previous_status = run.status
    run.status = RunStatus.CANCELLED.value
    run.finished_at = _terminal_time(run, now)
    _append_event(
        session,
        run=run,
        event_type=RunEventType.RUN_CANCELLED,
        payload={
            "previous_status": previous_status,
            "status": RunStatus.CANCELLED.value,
            "reason": reason,
        },
    )


def _mark_run_failed(
    session: AsyncSession,
    *,
    run: Run,
    error_category: str,
    now: datetime,
) -> None:
    if run.status in _TERMINAL_RUN_STATUSES:
        return
    if run.started_at is None:
        run.started_at = now
    run.status = RunStatus.FAILED.value
    run.error_category = error_category
    run.finished_at = _terminal_time(run, now)
    _append_event(
        session,
        run=run,
        event_type=RunEventType.RUN_FAILED,
        payload={
            "status": RunStatus.FAILED.value,
            "error_category": error_category,
        },
    )


def _mark_job_dead(
    session: AsyncSession,
    *,
    job: RunJob,
    run: Run,
    error_category: str,
) -> None:
    _clear_lease(job, status=JobStatus.DEAD)
    job.error_summary = error_category
    _append_event(
        session,
        run=run,
        event_type=RunEventType.JOB_DEAD,
        payload={
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "error_category": error_category,
        },
    )


class SqlAlchemyWorkerJobStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        retry_delay: RetryDelayPolicy,
        action_recovery_max_attempts: int = 3,
        *,
        execution_contracts: tuple[RunContractV1, ...] = EXECUTION_CONTRACTS,
        generation_publisher: ResumeGenerationPublisher | None = None,
        revision_publisher: ResumeRevisionPublisher | None = None,
    ) -> None:
        if (
            isinstance(action_recovery_max_attempts, bool)
            or not isinstance(action_recovery_max_attempts, int)
            or action_recovery_max_attempts < 1
        ):
            raise ValueError("action recovery max attempts must be positive")
        self._session_factory = session_factory
        self._execution_contracts = execution_contracts
        self._executable_versions = frozenset(c.graph_version for c in execution_contracts)
        self._retry_delay = retry_delay
        self._action_recovery_max_attempts = action_recovery_max_attempts
        self._generation_publisher = generation_publisher
        self._revision_publisher = revision_publisher

    def _executable_run_predicate(self):
        return (
            select(Run.id)
            .where(
                Run.workspace_id == RunJob.workspace_id,
                Run.id == RunJob.run_id,
                Run.graph_version.in_(self._executable_versions),
            )
            .exists()
        )

    async def _converge_terminal_action(
        self,
        session: AsyncSession,
        *,
        job: RunJob,
        run: Run,
        action: ActionIntent,
        now: datetime,
    ) -> None:
        invocation = await session.scalar(
            select(ToolInvocation)
            .where(ToolInvocation.action_intent_id == action.id)
            .with_for_update()
        )
        if invocation is None or invocation.status != action.status:
            raise DomainInvariantError("terminal action invocation is inconsistent")

        if action.status == "outcome_unknown":
            job.error_summary = "external_outcome_unknown"
            _mark_run_failed(
                session,
                run=run,
                error_category="external_outcome_unknown",
                now=now,
            )
            return
        if action.status != "failed":
            raise DomainInvariantError("terminal action convergence requires failure")

        evidence = action.evidence if isinstance(action.evidence, dict) else {}
        confirmed_absent_reason = evidence.get("reason")
        confirmed_absent = (
            evidence.get("classification") == "confirmed_absent"
            and confirmed_absent_reason == invocation.error_category
        )
        if (
            confirmed_absent
            and invocation.error_category == "cancelled_confirmed_absent"
            and run.cancel_requested_at is not None
        ):
            job.error_summary = None
            _mark_run_cancelled(session, run=run, reason="user_requested", now=now)
            return
        if confirmed_absent and invocation.error_category == "authorization_revoked":
            active_membership = await session.scalar(
                select(WorkspaceMembership.id)
                .where(
                    WorkspaceMembership.workspace_id == action.workspace_id,
                    WorkspaceMembership.user_id == action.originating_actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
                .with_for_update()
            )
            if active_membership is None:
                job.error_summary = None
                _mark_run_cancelled(
                    session,
                    run=run,
                    reason="authorization_revoked",
                    now=now,
                )
                return

        job.error_summary = "external_action_failed"
        _mark_run_failed(
            session,
            run=run,
            error_category="external_action_failed",
            now=now,
        )

    async def claim_due_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedJob | None:
        if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id) > 200:
            raise ValueError("worker_id must be a bounded non-blank string")
        if now.tzinfo is None or lease_duration <= timedelta(0):
            raise ValueError("claim timing is invalid")
        async with transaction(self._session_factory) as session:
            job = await session.scalar(
                select(RunJob)
                .where(
                    RunJob.status == JobStatus.QUEUED.value,
                    RunJob.available_at <= now,
                    self._executable_run_predicate(),
                    or_(
                        RunJob.attempt < RunJob.max_attempts,
                        select(ActionIntent.id)
                        .where(
                            ActionIntent.workspace_id == RunJob.workspace_id,
                            ActionIntent.run_id == RunJob.run_id,
                            ActionIntent.status == "executing",
                            ActionIntent.recovery_attempts <= self._action_recovery_max_attempts,
                        )
                        .exists(),
                        select(ActionIntent.id)
                        .where(
                            ActionIntent.workspace_id == RunJob.workspace_id,
                            ActionIntent.run_id == RunJob.run_id,
                            ActionIntent.status == "succeeded",
                            RunJob.attempt
                            < RunJob.max_attempts + self._action_recovery_max_attempts,
                        )
                        .exists(),
                    ),
                )
                .order_by(RunJob.available_at, RunJob.created_at, RunJob.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return None
            graph_version = await session.scalar(
                select(Run.graph_version).where(
                    Run.workspace_id == job.workspace_id,
                    Run.id == job.run_id,
                )
            )
            if graph_version is None:
                raise DomainInvariantError("claimed job run is missing")
            owner_token = uuid4()
            expires_at = now + lease_duration
            job.status = JobStatus.LEASED.value
            if job.attempt < job.max_attempts:
                job.attempt += 1
            job.leased_by = worker_id
            job.owner_token = owner_token
            job.lease_expires_at = expires_at
            await session.flush()
            claimed = ClaimedJob(
                job_id=job.id,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
                originating_actor_user_id=job.originating_actor_user_id,
                graph_version=graph_version,
                attempt=job.attempt,
                max_attempts=job.max_attempts,
                owner_token=owner_token,
                lease_expires_at=expires_at,
                resume_approval_request_id=job.resume_approval_request_id,
            )
        return claimed

    async def prepare_claimed_job(
        self,
        *,
        job: ClaimedJob,
        resolved_tenant: TenantContext | None,
        now: datetime,
    ) -> PrepareClaimResult:
        if resolved_tenant is not None and (
            resolved_tenant.workspace_id != job.workspace_id
            or resolved_tenant.actor_user_id != job.originating_actor_user_id
        ):
            raise DomainInvariantError("resolved tenant does not match claimed job")
        async with transaction(self._session_factory) as session:
            membership = await session.scalar(
                select(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == job.workspace_id,
                    WorkspaceMembership.user_id == job.originating_actor_user_id,
                )
                .with_for_update()
            )
            persisted_job = await self._lock_claimed_job(session, job=job, now=now)
            if persisted_job is None:
                return PrepareClaimResult(disposition="lease_lost")
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            executing_action = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                    ActionIntent.status == "executing",
                )
                .with_for_update()
            )
            terminal_failure_action = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                    ActionIntent.status.in_(("failed", "outcome_unknown")),
                )
                .with_for_update()
            )

            if executing_action is not None and (
                membership is None
                or membership.revoked_at is not None
                or run.cancel_requested_at is not None
            ):
                request = await session.scalar(
                    select(ApprovalRequest).where(
                        ApprovalRequest.workspace_id == job.workspace_id,
                        ApprovalRequest.run_id == job.run_id,
                        ApprovalRequest.action_intent_id == executing_action.id,
                        ApprovalRequest.status == ApprovalStatus.CONSUMED.value,
                    )
                )
                if request is None:
                    raise DomainInvariantError("executing action approval binding is missing")
                return PrepareClaimResult(
                    disposition="recover_action",
                    action_identity=ActionExecutionIdentity(
                        workspace_id=job.workspace_id,
                        run_id=job.run_id,
                        action_intent_id=executing_action.id,
                        approval_request_id=request.id,
                    ),
                )

            if terminal_failure_action is not None:
                _clear_lease(persisted_job, status=JobStatus.DONE)
                await self._converge_terminal_action(
                    session,
                    job=persisted_job,
                    run=run,
                    action=terminal_failure_action,
                    now=now,
                )
                return PrepareClaimResult(disposition="finished")

            if run.status in _TERMINAL_RUN_STATUSES:
                _clear_lease(persisted_job, status=JobStatus.DONE)
                persisted_job.error_summary = None
                return PrepareClaimResult(disposition="finished")

            if membership is None or membership.revoked_at is not None:
                _clear_lease(persisted_job, status=JobStatus.DONE)
                persisted_job.error_summary = None
                _mark_run_cancelled(
                    session,
                    run=run,
                    reason="authorization_revoked",
                    now=now,
                )
                return PrepareClaimResult(disposition="finished")

            try:
                current_role = WorkspaceRole(membership.role)
            except ValueError:
                raise DomainInvariantError("persisted workspace role is invalid") from None

            if (
                run.graph_version != job.graph_version
                or run.graph_version not in self._executable_versions
            ):
                error_category = "unsupported_graph_version"
                _mark_job_dead(
                    session,
                    job=persisted_job,
                    run=run,
                    error_category=error_category,
                )
                _mark_run_failed(
                    session,
                    run=run,
                    error_category=error_category,
                    now=now,
                )
                return PrepareClaimResult(disposition="finished")

            if run.cancel_requested_at is not None:
                _clear_lease(persisted_job, status=JobStatus.DONE)
                persisted_job.error_summary = None
                _mark_run_cancelled(
                    session,
                    run=run,
                    reason="user_requested",
                    now=now,
                )
                return PrepareClaimResult(disposition="finished")

            if persisted_job.resume_approval_request_id != job.resume_approval_request_id:
                raise DomainInvariantError("claimed job resume approval identity changed")

            if run.status == RunStatus.WAITING_APPROVAL.value:
                request = await self._require_resume_request(
                    session,
                    job=job,
                )
                run.status = RunStatus.RUNNING.value
                _append_event(
                    session,
                    run=run,
                    event_type=RunEventType.RUN_STATUS_CHANGED,
                    payload={
                        "previous_status": RunStatus.WAITING_APPROVAL.value,
                        "status": RunStatus.RUNNING.value,
                        "reason": "approval_resume_claimed",
                        "approval_request_id": str(request.id),
                        "action_intent_id": str(request.action_intent_id),
                    },
                )

            if run.status == RunStatus.QUEUED.value:
                if job.resume_approval_request_id is not None:
                    raise DomainInvariantError("initial queued run has a resume approval identity")
                run.status = RunStatus.RUNNING.value
                run.started_at = now
                _append_event(
                    session,
                    run=run,
                    event_type=RunEventType.RUN_STATUS_CHANGED,
                    payload={
                        "previous_status": RunStatus.QUEUED.value,
                        "status": RunStatus.RUNNING.value,
                        "reason": "worker_claimed",
                    },
                )
            elif run.status != RunStatus.RUNNING.value:
                raise DomainInvariantError("claimed job run status is invalid")
            elif job.resume_approval_request_id is not None:
                await self._require_resume_request(session, job=job, allow_consumed=True)

            tenant = TenantContext(
                workspace_id=job.workspace_id,
                actor_user_id=job.originating_actor_user_id,
                role=current_role,
            )
            return PrepareClaimResult(disposition="execute", tenant=tenant)

    async def heartbeat(
        self,
        *,
        job: ClaimedJob,
        now: datetime,
        lease_duration: timedelta,
    ) -> bool:
        async with transaction(self._session_factory) as session:
            updated_id = await session.scalar(
                update(RunJob)
                .where(*self._active_lease_predicates(job, now=now))
                .values(lease_expires_at=now + lease_duration)
                .returning(RunJob.id)
            )
        return updated_id is not None

    async def complete(
        self,
        *,
        job: ClaimedJob,
        result: RunOutput,
        now: datetime,
    ) -> bool:
        async with transaction(self._session_factory) as session:
            if not await self._consume_lease(
                session,
                job=job,
                now=now,
                status=JobStatus.DONE,
                error_summary=None,
            ):
                return False
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            if run.status in _TERMINAL_RUN_STATUSES:
                return True
            if run.cancel_requested_at is not None:
                _mark_run_cancelled(session, run=run, reason="user_requested", now=now)
                return True
            if run.status != RunStatus.RUNNING.value:
                raise DomainInvariantError("completed job run is not running")
            if isinstance(
                result, ResumeGenerationCandidateOutputV1 | ResumeRevisionCandidateOutputV1
            ):
                publisher = (
                    self._generation_publisher
                    if isinstance(result, ResumeGenerationCandidateOutputV1)
                    else self._revision_publisher
                )
                expected_mode = (
                    RunMode.RESUME_GENERATION.value
                    if isinstance(result, ResumeGenerationCandidateOutputV1)
                    else RunMode.RESUME_REVISION.value
                )
                if publisher is None or run.mode != expected_mode:
                    raise DomainInvariantError("resume publisher is unavailable")
                membership = await session.scalar(
                    select(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == job.workspace_id,
                        WorkspaceMembership.user_id == job.originating_actor_user_id,
                        WorkspaceMembership.revoked_at.is_(None),
                    )
                    .with_for_update(read=True)
                )
                if membership is None:
                    _mark_run_cancelled(session, run=run, reason="authorization_revoked", now=now)
                    return True
                result = await publisher.publish(
                    session,
                    TenantContext(
                        job.workspace_id,
                        job.originating_actor_user_id,
                        WorkspaceRole(membership.role),
                    ),
                    job.run_id,
                    result,
                )
            try:
                contract = find_run_contract(
                    self._execution_contracts, run.graph_version, RunMode(run.mode)
                )
                validated = contract.decode_output(
                    result.model_dump(mode="json", round_trip=True, warnings="error")
                )
            except (TypeError, ValueError, AttributeError):
                raise DomainInvariantError("completed job result contract is invalid") from None
            run.status = RunStatus.COMPLETED.value
            run.result_json = validated.model_dump(mode="json", round_trip=True)
            run.error_category = None
            run.finished_at = _terminal_time(run, now)
            _append_event(
                session,
                run=run,
                event_type=RunEventType.RUN_COMPLETED,
                payload={"status": RunStatus.COMPLETED.value},
            )
        return True

    async def wait_for_approval(
        self,
        *,
        job: ClaimedJob,
        approval_request_id: UUID,
        now: datetime,
    ) -> bool:
        if not isinstance(approval_request_id, UUID):
            raise TypeError("approval request identity must be a UUID")
        async with transaction(self._session_factory) as session:
            persisted_job = await self._lock_claimed_job(session, job=job, now=now)
            if persisted_job is None:
                return False
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            if RunMode(run.mode) not in LEGACY_RUN_MODES:
                raise DomainInvariantError("resume execution cannot wait for legacy approval")
            request = await session.scalar(
                select(ApprovalRequest)
                .where(
                    ApprovalRequest.id == approval_request_id,
                    ApprovalRequest.workspace_id == job.workspace_id,
                    ApprovalRequest.run_id == job.run_id,
                )
                .with_for_update()
            )
            if request is None:
                raise DomainInvariantError("approval pause request is missing or cross-tenant")
            intent = await session.scalar(
                select(ActionIntent).where(
                    ActionIntent.id == request.action_intent_id,
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                )
            )
            if intent is None:
                raise DomainInvariantError("approval pause action binding is invalid")
            is_resume_pause = persisted_job.resume_approval_request_id is not None
            if is_resume_pause:
                if (
                    persisted_job.resume_approval_request_id != approval_request_id
                    or job.resume_approval_request_id != approval_request_id
                ):
                    raise DomainInvariantError("approval resume pause identity is invalid")
                if persisted_approval_status(request.status) not in _GATE6_RESUME_APPROVAL_STATUSES:
                    raise DomainInvariantError("approval resume pause request status is invalid")
                event_reason = "approval_resume_paused"
            else:
                if job.resume_approval_request_id is not None:
                    raise DomainInvariantError("initial approval pause has a resume identity")
                if persisted_approval_status(request.status) is not ApprovalStatus.PENDING:
                    raise DomainInvariantError("approval pause request is not pending")
                event_reason = "approval_required"
            if run.status != RunStatus.RUNNING.value:
                raise DomainInvariantError("approval pause run is not running")
            _clear_lease(persisted_job, status=JobStatus.DONE)
            persisted_job.error_summary = None
            run.status = RunStatus.WAITING_APPROVAL.value
            _append_event(
                session,
                run=run,
                event_type=RunEventType.RUN_STATUS_CHANGED,
                payload={
                    "previous_status": RunStatus.RUNNING.value,
                    "status": RunStatus.WAITING_APPROVAL.value,
                    "reason": event_reason,
                    "approval_request_id": str(request.id),
                    "action_intent_id": str(request.action_intent_id),
                },
            )
        return True

    async def requeue(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self._require_error_category(error_category)
        async with transaction(self._session_factory) as session:
            if not await self._consume_lease(
                session,
                job=job,
                now=now,
                status=JobStatus.QUEUED,
                error_summary=error_category,
                available_at=now + self._retry_delay(job.attempt),
            ):
                return False
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            executing_action = await session.scalar(
                select(ActionIntent.id).where(
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                    ActionIntent.status == "executing",
                )
            )
            if (
                run.cancel_requested_at is not None
                and run.status == RunStatus.RUNNING.value
                and executing_action is None
            ):
                await session.execute(
                    update(RunJob)
                    .where(RunJob.id == job.job_id, RunJob.status == JobStatus.QUEUED.value)
                    .values(status=JobStatus.DONE.value, error_summary=None)
                )
                _mark_run_cancelled(session, run=run, reason="user_requested", now=now)
        return True

    async def fail(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self._require_error_category(error_category)
        async with transaction(self._session_factory) as session:
            action = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                    ActionIntent.status.in_(
                        ("executing", "succeeded", "failed", "outcome_unknown")
                    ),
                )
                .with_for_update()
            )
            if (
                action is not None
                and action.status == "executing"
                and action.recovery_attempts < self._action_recovery_max_attempts
            ):
                return await self._consume_lease(
                    session,
                    job=job,
                    now=now,
                    status=JobStatus.QUEUED,
                    error_summary="action_recovery_required",
                    available_at=now + self._retry_delay(job.attempt),
                )
            if action is not None and action.status == "executing":
                if not await self._consume_lease(
                    session,
                    job=job,
                    now=now,
                    status=JobStatus.DONE,
                    error_summary="external_outcome_unknown",
                ):
                    return False
                run = await self._lock_run(
                    session,
                    workspace_id=job.workspace_id,
                    run_id=job.run_id,
                )
                invocation = await session.scalar(
                    select(ToolInvocation)
                    .where(ToolInvocation.action_intent_id == action.id)
                    .with_for_update()
                )
                if invocation is None or invocation.status != "executing":
                    raise DomainInvariantError("executing action invocation is inconsistent")
                action.status = "outcome_unknown"
                action.result = None
                action.evidence = {"classification": "recovery_exhausted"}
                invocation.status = "outcome_unknown"
                invocation.latency_ms = 0
                invocation.result_summary = None
                invocation.error_category = "external_outcome_unknown"
                invocation.finished_at = now
                _append_event(
                    session,
                    run=run,
                    event_type=RunEventType.ACTION_OUTCOME_UNKNOWN,
                    payload={
                        "action_intent_id": str(action.id),
                        "invocation_id": str(invocation.id),
                        "status": "outcome_unknown",
                        "error_category": "external_outcome_unknown",
                        "recovery_attempts": action.recovery_attempts,
                    },
                )
                _mark_run_failed(
                    session,
                    run=run,
                    error_category="external_outcome_unknown",
                    now=now,
                )
                return True
            terminal_action_failure = action is not None and action.status in {
                "failed",
                "outcome_unknown",
            }
            if not await self._consume_lease(
                session,
                job=job,
                now=now,
                status=(JobStatus.DONE if terminal_action_failure else JobStatus.DEAD),
                error_summary=error_category,
            ):
                return False
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            if run.status in _TERMINAL_RUN_STATUSES:
                return True
            if terminal_action_failure:
                if action is None:
                    raise DomainInvariantError("terminal action failure is missing")
                persisted_job = await session.get(RunJob, job.job_id)
                if persisted_job is None:
                    raise DomainInvariantError("claimed job is missing")
                await self._converge_terminal_action(
                    session,
                    job=persisted_job,
                    run=run,
                    action=action,
                    now=now,
                )
                return True
            if run.cancel_requested_at is not None:
                await session.execute(
                    update(RunJob)
                    .where(RunJob.id == job.job_id, RunJob.status == JobStatus.DEAD.value)
                    .values(status=JobStatus.DONE.value, error_summary=None)
                )
                _mark_run_cancelled(session, run=run, reason="user_requested", now=now)
                return True
            _append_event(
                session,
                run=run,
                event_type=RunEventType.JOB_DEAD,
                payload={
                    "attempt": job.attempt,
                    "max_attempts": job.max_attempts,
                    "error_category": error_category,
                },
            )
            _mark_run_failed(session, run=run, error_category=error_category, now=now)
        return True

    async def cancel(
        self,
        *,
        job: ClaimedJob,
        reason: str,
        now: datetime,
    ) -> bool:
        self._require_error_category(reason)
        async with transaction(self._session_factory) as session:
            action = await session.scalar(
                select(ActionIntent)
                .where(
                    ActionIntent.workspace_id == job.workspace_id,
                    ActionIntent.run_id == job.run_id,
                    ActionIntent.status.in_(("executing", "failed", "outcome_unknown")),
                )
                .with_for_update()
            )
            if action is not None and action.status == "executing":
                return await self._consume_lease(
                    session,
                    job=job,
                    now=now,
                    status=JobStatus.QUEUED,
                    error_summary="action_recovery_required",
                    available_at=now + self._retry_delay(job.attempt),
                )
            if not await self._consume_lease(
                session,
                job=job,
                now=now,
                status=JobStatus.DONE,
                error_summary=None,
            ):
                return False
            run = await self._lock_run(
                session,
                workspace_id=job.workspace_id,
                run_id=job.run_id,
            )
            if action is not None:
                persisted_job = await session.get(RunJob, job.job_id)
                if persisted_job is None:
                    raise DomainInvariantError("claimed job is missing")
                await self._converge_terminal_action(
                    session,
                    job=persisted_job,
                    run=run,
                    action=action,
                    now=now,
                )
                return True
            active_membership = await session.scalar(
                select(WorkspaceMembership.id).where(
                    WorkspaceMembership.workspace_id == job.workspace_id,
                    WorkspaceMembership.user_id == job.originating_actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
            )
            persisted_reason = reason
            if run.cancel_requested_at is not None:
                persisted_reason = "user_requested"
            elif active_membership is None:
                persisted_reason = "authorization_revoked"
            _mark_run_cancelled(session, run=run, reason=persisted_reason, now=now)
        return True

    async def release(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self._require_error_category(error_category)
        async with transaction(self._session_factory) as session:
            updated_id = await session.scalar(
                update(RunJob)
                .where(
                    *self._active_lease_predicates(job, now=now),
                    RunJob.attempt < RunJob.max_attempts,
                )
                .values(
                    status=JobStatus.QUEUED.value,
                    leased_by=None,
                    owner_token=None,
                    lease_expires_at=None,
                    error_summary=error_category,
                    available_at=now + self._retry_delay(job.attempt),
                )
                .returning(RunJob.id)
            )
        return updated_id is not None

    async def reclaim_stale_leases(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> ReclaimSummary:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("stale lease reclaim limit must be positive")
        requeued = 0
        finished = 0
        dead = 0
        async with transaction(self._session_factory) as session:
            jobs = list(
                await session.scalars(
                    select(RunJob)
                    .where(
                        RunJob.status == JobStatus.LEASED.value,
                        RunJob.lease_expires_at <= now,
                        self._executable_run_predicate(),
                    )
                    .order_by(RunJob.lease_expires_at, RunJob.created_at, RunJob.id)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for persisted_job in jobs:
                run = await self._lock_run(
                    session,
                    workspace_id=persisted_job.workspace_id,
                    run_id=persisted_job.run_id,
                )
                action = await session.scalar(
                    select(ActionIntent)
                    .where(
                        ActionIntent.workspace_id == persisted_job.workspace_id,
                        ActionIntent.run_id == persisted_job.run_id,
                        ActionIntent.status.in_(
                            ("executing", "succeeded", "failed", "outcome_unknown")
                        ),
                    )
                    .with_for_update()
                )
                if action is not None and action.status == "executing":
                    invocation = await session.scalar(
                        select(ToolInvocation)
                        .where(ToolInvocation.action_intent_id == action.id)
                        .with_for_update()
                    )
                    if invocation is None or invocation.status != "executing":
                        raise DomainInvariantError("executing action invocation is inconsistent")
                    _append_event(
                        session,
                        run=run,
                        event_type=RunEventType.JOB_LEASE_EXPIRED,
                        payload={
                            "attempt": persisted_job.attempt,
                            "max_attempts": persisted_job.max_attempts,
                            "action_recovery": True,
                        },
                    )
                    if action.recovery_attempts < self._action_recovery_max_attempts:
                        _clear_lease(persisted_job, status=JobStatus.QUEUED)
                        persisted_job.available_at = now + self._retry_delay(persisted_job.attempt)
                        persisted_job.error_summary = "action_recovery_required"
                        requeued += 1
                        continue
                    action.status = "outcome_unknown"
                    action.result = None
                    action.evidence = {"classification": "recovery_exhausted"}
                    invocation.status = "outcome_unknown"
                    invocation.latency_ms = 0
                    invocation.result_summary = None
                    invocation.error_category = "external_outcome_unknown"
                    invocation.finished_at = now
                    _append_event(
                        session,
                        run=run,
                        event_type=RunEventType.ACTION_OUTCOME_UNKNOWN,
                        payload={
                            "action_intent_id": str(action.id),
                            "invocation_id": str(invocation.id),
                            "status": "outcome_unknown",
                            "error_category": "external_outcome_unknown",
                            "recovery_attempts": action.recovery_attempts,
                        },
                    )
                    _clear_lease(persisted_job, status=JobStatus.DONE)
                    persisted_job.error_summary = "external_outcome_unknown"
                    _mark_run_failed(
                        session,
                        run=run,
                        error_category="external_outcome_unknown",
                        now=now,
                    )
                    finished += 1
                    continue
                if action is not None and action.status in {"failed", "outcome_unknown"}:
                    _clear_lease(persisted_job, status=JobStatus.DONE)
                    await self._converge_terminal_action(
                        session,
                        job=persisted_job,
                        run=run,
                        action=action,
                        now=now,
                    )
                    finished += 1
                    continue
                if (
                    action is not None
                    and action.status == "succeeded"
                    and run.status not in _TERMINAL_RUN_STATUSES
                    and persisted_job.attempt
                    < persisted_job.max_attempts + self._action_recovery_max_attempts
                ):
                    _append_event(
                        session,
                        run=run,
                        event_type=RunEventType.JOB_LEASE_EXPIRED,
                        payload={
                            "attempt": persisted_job.attempt,
                            "max_attempts": persisted_job.max_attempts,
                            "terminal_action_convergence": True,
                        },
                    )
                    _clear_lease(persisted_job, status=JobStatus.QUEUED)
                    persisted_job.available_at = now + self._retry_delay(persisted_job.attempt)
                    persisted_job.error_summary = "terminal_action_convergence"
                    requeued += 1
                    continue
                if run.status in _TERMINAL_RUN_STATUSES:
                    _clear_lease(persisted_job, status=JobStatus.DONE)
                    persisted_job.error_summary = None
                    finished += 1
                    continue
                if run.cancel_requested_at is not None:
                    _clear_lease(persisted_job, status=JobStatus.DONE)
                    persisted_job.error_summary = None
                    _mark_run_cancelled(session, run=run, reason="user_requested", now=now)
                    finished += 1
                    continue

                _append_event(
                    session,
                    run=run,
                    event_type=RunEventType.JOB_LEASE_EXPIRED,
                    payload={
                        "attempt": persisted_job.attempt,
                        "max_attempts": persisted_job.max_attempts,
                    },
                )
                legal_waiting_resume = (
                    run.status == RunStatus.WAITING_APPROVAL.value
                    and persisted_job.resume_approval_request_id is not None
                )
                if persisted_job.attempt < persisted_job.max_attempts and (
                    run.status
                    in {
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    }
                    or legal_waiting_resume
                ):
                    if legal_waiting_resume:
                        await self._require_persisted_resume_request(
                            session,
                            persisted_job=persisted_job,
                        )
                    _clear_lease(persisted_job, status=JobStatus.QUEUED)
                    persisted_job.available_at = now + self._retry_delay(persisted_job.attempt)
                    persisted_job.error_summary = "lease_expired"
                    requeued += 1
                    continue

                error_category = (
                    "unexpected_waiting_approval"
                    if run.status == RunStatus.WAITING_APPROVAL.value
                    else "job_attempts_exhausted"
                )
                _mark_job_dead(
                    session,
                    job=persisted_job,
                    run=run,
                    error_category=error_category,
                )
                _mark_run_failed(
                    session,
                    run=run,
                    error_category=error_category,
                    now=now,
                )
                dead += 1

        return ReclaimSummary(
            scanned=len(jobs),
            requeued=requeued,
            finished=finished,
            dead=dead,
        )

    async def _lock_claimed_job(
        self,
        session: AsyncSession,
        *,
        job: ClaimedJob,
        now: datetime,
    ) -> RunJob | None:
        return await session.scalar(
            select(RunJob).where(*self._active_lease_predicates(job, now=now)).with_for_update()
        )

    @staticmethod
    async def _require_resume_request(
        session: AsyncSession,
        *,
        job: ClaimedJob,
        allow_consumed: bool = False,
    ) -> ApprovalRequest:
        if job.resume_approval_request_id is None:
            raise DomainInvariantError("waiting approval job is missing resume identity")
        request = await session.scalar(
            select(ApprovalRequest).where(
                ApprovalRequest.id == job.resume_approval_request_id,
                ApprovalRequest.workspace_id == job.workspace_id,
                ApprovalRequest.run_id == job.run_id,
            )
        )
        if request is None:
            raise DomainInvariantError("resume approval request is missing or cross-tenant")
        status = persisted_approval_status(request.status)
        if status is ApprovalStatus.CONSUMED and allow_consumed:
            await SqlAlchemyWorkerJobStore._require_consumed_resume(session, job, request)
        elif status not in _GATE6_RESUME_APPROVAL_STATUSES:
            raise DomainInvariantError("resume approval request status is invalid")
        return request

    @staticmethod
    async def _require_consumed_resume(
        session: AsyncSession, job: ClaimedJob, request: ApprovalRequest
    ) -> None:
        # Consumption authorizes recovery of the original action, never a new approval.
        intent = await session.scalar(
            select(ActionIntent)
            .where(
                ActionIntent.workspace_id == job.workspace_id,
                ActionIntent.run_id == job.run_id,
                ActionIntent.id == request.action_intent_id,
                ActionIntent.originating_actor_user_id == job.originating_actor_user_id,
            )
            .with_for_update()
        )
        if intent is None or request.consumed_at is None:
            raise DomainInvariantError("consumed resume action is missing or conflicting")
        validate_exact_approval_binding(
            intent=_intent_record(intent), request=_request_record(request)
        )
        invocation = await session.scalar(
            select(ToolInvocation)
            .where(
                ToolInvocation.workspace_id == job.workspace_id,
                ToolInvocation.run_id == job.run_id,
                ToolInvocation.action_intent_id == intent.id,
            )
            .with_for_update()
        )
        expected_status = {
            "authorized": "prepared",
            "executing": "executing",
            "succeeded": "succeeded",
        }.get(intent.status)
        if (
            invocation is None
            or expected_status is None
            or invocation.status != expected_status
            or invocation.originating_actor_user_id != job.originating_actor_user_id
            or invocation.tool_name != intent.tool_name
            or invocation.effect != intent.effect
            or invocation.args_digest != intent.args_digest
            or (expected_status == "prepared" and invocation.attempt != 0)
            or (expected_status != "prepared" and invocation.attempt < 1)
        ):
            raise DomainInvariantError("consumed resume invocation facts conflict")

    @staticmethod
    async def _require_persisted_resume_request(
        session: AsyncSession,
        *,
        persisted_job: RunJob,
    ) -> ApprovalRequest:
        request_id = persisted_job.resume_approval_request_id
        if request_id is None:
            raise DomainInvariantError("waiting approval job is missing resume identity")
        request = await session.scalar(
            select(ApprovalRequest).where(
                ApprovalRequest.id == request_id,
                ApprovalRequest.workspace_id == persisted_job.workspace_id,
                ApprovalRequest.run_id == persisted_job.run_id,
            )
        )
        if request is None:
            raise DomainInvariantError("resume approval request is missing or cross-tenant")
        return request

    @staticmethod
    async def _lock_run(
        session: AsyncSession,
        *,
        workspace_id: UUID,
        run_id: UUID,
    ) -> Run:
        run = await session.scalar(
            select(Run).where(Run.workspace_id == workspace_id, Run.id == run_id).with_for_update()
        )
        if run is None:
            raise DomainInvariantError("claimed job run is missing")
        return run

    @staticmethod
    def _active_lease_predicates(
        job: ClaimedJob,
        *,
        now: datetime,
    ) -> tuple[object, ...]:
        return (
            RunJob.id == job.job_id,
            RunJob.workspace_id == job.workspace_id,
            RunJob.run_id == job.run_id,
            RunJob.originating_actor_user_id == job.originating_actor_user_id,
            RunJob.status == JobStatus.LEASED.value,
            RunJob.owner_token == job.owner_token,
            RunJob.attempt == job.attempt,
            RunJob.lease_expires_at > now,
        )

    async def _consume_lease(
        self,
        session: AsyncSession,
        *,
        job: ClaimedJob,
        now: datetime,
        status: JobStatus,
        error_summary: str | None,
        available_at: datetime | None = None,
    ) -> bool:
        values: dict[str, object] = {
            "status": status.value,
            "leased_by": None,
            "owner_token": None,
            "lease_expires_at": None,
            "error_summary": error_summary,
        }
        if available_at is not None:
            values["available_at"] = available_at
        updated_id = await session.scalar(
            update(RunJob)
            .where(*self._active_lease_predicates(job, now=now))
            .values(**values)
            .returning(RunJob.id)
        )
        return updated_id is not None

    @staticmethod
    def _require_error_category(value: str) -> None:
        if not _valid_error_category(value):
            raise ValueError("job error category is invalid")
