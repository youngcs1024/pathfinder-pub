from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    ActionIntent,
    ApprovalRequest,
    Conversation,
    Document,
    LLMInvocation,
    Message,
    Run,
    RunEvent,
    RunJob,
    WorkspaceMembership,
)
from app.db.session import AsyncSessionFactory, database_session, transaction
from app.domain.actions import ActionIntentStatus, persisted_action_intent_status
from app.domain.approvals import ApprovalStatus, persisted_approval_status
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
)
from app.domain.jobs import JobStatus
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchOutputV1, ResearchOutputV2, ResearchRequestV1
from app.domain.runs import (
    EXECUTABLE_GRAPH_VERSIONS,
    READABLE_GRAPH_VERSIONS,
    MessageRole,
    RunAccepted,
    RunCancellation,
    RunCreateIdentity,
    RunMode,
    RunRecord,
    RunStatus,
    RunUsageBucket,
    RunUsageSummary,
    create_request_digest_v1,
)
from app.domain.tenancy import TenantContext
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType

_CONVERSATION_TITLE = "Research request"


async def _require_current_tenant(
    session: AsyncSession,
    tenant: TenantContext,
    *,
    lock: bool,
) -> None:
    statement = select(WorkspaceMembership.role).where(
        WorkspaceMembership.workspace_id == tenant.workspace_id,
        WorkspaceMembership.user_id == tenant.actor_user_id,
        WorkspaceMembership.revoked_at.is_(None),
        WorkspaceMembership.role == tenant.role.value,
    )
    if lock:
        statement = statement.with_for_update(read=True)
    role = await session.scalar(statement)
    if role is None:
        raise DomainNotFoundError


def _usage_token(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DomainInvariantError(f"persisted invocation {field} is invalid")
    return value


def _usage_bucket(rows: list[object]) -> RunUsageBucket:
    attempt_count = len(rows)
    succeeded_rows = [row for row in rows if row.status == "succeeded"]
    input_tokens = 0
    output_tokens = 0
    reasoning_output_tokens = 0
    cached_input_tokens = 0
    cache_write_input_tokens = 0

    for row in succeeded_rows:
        usage = row.token_usage
        if not isinstance(usage, dict):
            raise DomainInvariantError("successful invocation usage is invalid")
        input_tokens += _usage_token(usage.get("input_tokens"), field="input_tokens")
        output_tokens += _usage_token(usage.get("output_tokens"), field="output_tokens")
        reasoning_output_tokens += _usage_token(
            usage.get("reasoning_output_tokens", 0),
            field="reasoning_output_tokens",
        )
        cached_input_tokens += _usage_token(
            usage.get("cached_input_tokens", 0),
            field="cached_input_tokens",
        )
        cache_write_input_tokens += _usage_token(
            usage.get("cache_write_input_tokens", 0),
            field="cache_write_input_tokens",
        )

    currencies = {row.currency for row in succeeded_rows if row.currency is not None}
    cost_available = bool(succeeded_rows) and all(
        row.estimated_cost is not None and row.currency is not None for row in succeeded_rows
    )
    cost_available = cost_available and len(currencies) == 1
    estimated_cost = (
        sum((row.estimated_cost for row in succeeded_rows), start=Decimal("0"))
        if cost_available
        else None
    )
    currency = next(iter(currencies)) if cost_available else None
    return RunUsageBucket(
        attempt_count=attempt_count,
        succeeded_count=len(succeeded_rows),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        estimated_cost=estimated_cost,
        currency=currency,
        cost_available=cost_available,
    )


def _usage_summary(rows: list[object]) -> RunUsageSummary:
    by_kind = {
        "chat": [row for row in rows if row.invocation_kind == "chat"],
        "embedding": [row for row in rows if row.invocation_kind == "embedding"],
    }
    if sum(len(items) for items in by_kind.values()) != len(rows):
        raise DomainInvariantError("persisted invocation kind is invalid")
    return RunUsageSummary(
        chat=_usage_bucket(by_kind["chat"]),
        embedding=_usage_bucket(by_kind["embedding"]),
    )


def _run_status(value: str) -> RunStatus:
    try:
        return RunStatus(value)
    except ValueError:
        raise DomainInvariantError("persisted run status is invalid") from None


def _run_mode(value: str) -> RunMode:
    try:
        return RunMode(value)
    except ValueError:
        raise DomainInvariantError("persisted run mode is invalid") from None


class SqlAlchemyRunStore:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def create_run(
        self,
        *,
        tenant: TenantContext,
        mode: RunMode,
        resume_document_id: UUID | None,
        request: ResearchRequestV1,
        limits: dict[str, int],
        graph_version: str,
        request_identity: RunCreateIdentity | None = None,
    ) -> RunAccepted:
        if graph_version not in EXECUTABLE_GRAPH_VERSIONS:
            raise DomainInvariantError("unsupported graph version")
        if request_identity is not None:
            digest = create_request_digest_v1(
                mode=mode, query=request.query, resume_document_id=resume_document_id
            )
            if request_identity.create_request_digest != digest:
                raise DomainValidationError("run creation identity does not match input")
        conversation_id = uuid4()
        message_id = uuid4()
        run_id = uuid4()
        try:
            async with transaction(self._session_factory) as session:
                await _require_current_tenant(session, tenant, lock=True)
                if request_identity is not None:
                    existing = await self._find_request(session, tenant, request_identity)
                    if existing is not None:
                        return self._replay(existing, mode, request, resume_document_id)
                if resume_document_id is not None:
                    document_id = await session.scalar(
                        select(Document.id).where(
                            Document.workspace_id == tenant.workspace_id,
                            Document.id == resume_document_id,
                        )
                    )
                    if document_id is None:
                        raise DomainNotFoundError
                session.add(
                    Conversation(
                        id=conversation_id,
                        workspace_id=tenant.workspace_id,
                        created_by_user_id=tenant.actor_user_id,
                        title=_CONVERSATION_TITLE,
                    )
                )
                session.add(
                    Message(
                        id=message_id,
                        workspace_id=tenant.workspace_id,
                        conversation_id=conversation_id,
                        actor_user_id=tenant.actor_user_id,
                        role=MessageRole.USER.value,
                        content=request.query,
                    )
                )
                session.add(
                    Run(
                        id=run_id,
                        workspace_id=tenant.workspace_id,
                        created_by_user_id=tenant.actor_user_id,
                        conversation_id=conversation_id,
                        request_message_id=message_id,
                        mode=mode.value,
                        resume_document_id=resume_document_id,
                        client_request_id=(
                            request_identity.client_request_id if request_identity else None
                        ),
                        create_request_digest=(
                            request_identity.create_request_digest if request_identity else None
                        ),
                        create_request_version=(
                            request_identity.create_request_version if request_identity else None
                        ),
                        input_json=request.model_dump(mode="json", round_trip=True),
                        limits_json=dict(limits),
                        status=RunStatus.QUEUED.value,
                        graph_version=graph_version,
                    )
                )
                session.add(
                    RunJob(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        originating_actor_user_id=tenant.actor_user_id,
                        run_id=run_id,
                        status=JobStatus.QUEUED.value,
                    )
                )
                await session.flush()
                allocated_seq = await session.scalar(
                    update(Run)
                    .where(
                        Run.workspace_id == tenant.workspace_id,
                        Run.id == run_id,
                    )
                    .values(next_event_seq=Run.next_event_seq + 1)
                    .returning(Run.next_event_seq - 1)
                )
                if allocated_seq != 1:
                    raise DomainInvariantError("new run event sequence did not start at one")
                session.add(
                    RunEvent(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        run_id=run_id,
                        actor_user_id=tenant.actor_user_id,
                        seq=allocated_seq,
                        type=RunEventType.RUN_CREATED.value,
                        version=CURRENT_RUN_EVENT_VERSION,
                        payload={
                            "mode": mode.value,
                            "status": RunStatus.QUEUED.value,
                            "graph_version": graph_version,
                        },
                    )
                )
            return RunAccepted(run_id=run_id, status=RunStatus.QUEUED)
        except IntegrityError as error:
            # transaction() has rolled back and closed the entire losing session.
            if (
                request_identity is None
                or getattr(error.orig, "sqlstate", None) != "23505"
                or getattr(getattr(error.orig, "diag", None), "constraint_name", None)
                != "uq_runs_workspace_creator_request"
            ):
                raise
        # Leave the driver exception context before emitting safe domain errors.
        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, tenant, lock=True)
            existing = await self._find_request(session, tenant, request_identity)
            if existing is None:
                raise DomainInvariantError("run creation conflict winner is missing")
            return self._replay(existing, mode, request, resume_document_id)

    async def _find_request(
        self, session: AsyncSession, tenant: TenantContext, identity: RunCreateIdentity
    ) -> Run | None:
        return await session.scalar(
            select(Run).where(
                Run.workspace_id == tenant.workspace_id,
                Run.created_by_user_id == tenant.actor_user_id,
                Run.client_request_id == identity.client_request_id,
            )
        )

    @staticmethod
    def _replay(
        existing: Run,
        mode: RunMode,
        request: ResearchRequestV1,
        resume_document_id: UUID | None,
    ) -> RunAccepted:
        if existing.create_request_version != 1:
            raise DomainInvariantError("persisted run creation identity version is unsupported")
        digest = create_request_digest_v1(
            mode=mode, query=request.query, resume_document_id=resume_document_id
        )
        if digest != existing.create_request_digest:
            raise DomainConflictError("run creation request conflicts with accepted request")
        return RunAccepted(run_id=existing.id, status=RunStatus.QUEUED, replayed=True)

    async def get_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
    ) -> RunRecord:
        async with database_session(self._session_factory) as session:
            await _require_current_tenant(session, tenant, lock=False)
            run = await session.scalar(
                select(Run).where(
                    Run.workspace_id == tenant.workspace_id,
                    Run.id == run_id,
                )
            )
            if run is None:
                raise DomainNotFoundError
            invocation_rows = list(
                (
                    await session.execute(
                        select(
                            LLMInvocation.invocation_kind,
                            LLMInvocation.status,
                            LLMInvocation.token_usage,
                            LLMInvocation.estimated_cost,
                            LLMInvocation.currency,
                        ).where(
                            LLMInvocation.workspace_id == tenant.workspace_id,
                            LLMInvocation.run_id == run_id,
                        )
                    )
                ).all()
            )

        result = None
        if run.result_json is not None:
            try:
                result_model = (
                    ResearchOutputV2
                    if run.result_json.get("schema_version") == 2
                    else ResearchOutputV1
                )
                result = result_model.model_validate_json(
                    json.dumps(
                        run.result_json,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                    strict=True,
                )
            except (TypeError, ValidationError, ValueError):
                raise DomainInvariantError("persisted run result is invalid") from None
        if run.graph_version not in READABLE_GRAPH_VERSIONS:
            raise DomainInvariantError("persisted graph version is unsupported")
        return RunRecord(
            run_id=run.id,
            mode=_run_mode(run.mode),
            status=_run_status(run.status),
            graph_version=run.graph_version,
            resume_document_id=run.resume_document_id,
            result=result,
            error_category=run.error_category,
            cancel_requested_at=run.cancel_requested_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
            updated_at=run.updated_at,
            usage=_usage_summary(invocation_rows),
        )

    async def cancel_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        allow_other_creator: bool,
    ) -> RunCancellation:
        if allow_other_creator and tenant.role is not WorkspaceRole.ADMIN:
            raise DomainInvariantError("non-admin tenant cannot cancel another actor run")
        async with transaction(self._session_factory) as session:
            await _require_current_tenant(session, tenant, lock=True)
            active_requests = list(
                await session.scalars(
                    select(ApprovalRequest)
                    .join(
                        ActionIntent,
                        (ActionIntent.workspace_id == ApprovalRequest.workspace_id)
                        & (ActionIntent.run_id == ApprovalRequest.run_id)
                        & (ActionIntent.id == ApprovalRequest.action_intent_id),
                    )
                    .where(
                        ApprovalRequest.workspace_id == tenant.workspace_id,
                        ApprovalRequest.run_id == run_id,
                        ActionIntent.status.in_(
                            (
                                ActionIntentStatus.PROPOSED.value,
                                ActionIntentStatus.AUTHORIZED.value,
                            )
                        ),
                    )
                    .with_for_update(of=ApprovalRequest)
                )
            )
            approval_request = active_requests[0] if len(active_requests) == 1 else None
            if len(active_requests) > 1:
                raise DomainInvariantError("waiting run has multiple active approval requests")
            job = await session.scalar(
                select(RunJob)
                .where(
                    RunJob.workspace_id == tenant.workspace_id,
                    RunJob.run_id == run_id,
                )
                .with_for_update()
            )
            run = await session.scalar(
                select(Run)
                .where(
                    Run.workspace_id == tenant.workspace_id,
                    Run.id == run_id,
                )
                .with_for_update()
            )
            if run is None:
                raise DomainNotFoundError
            if job is None:
                raise DomainInvariantError("run does not have one job")
            if not allow_other_creator and run.created_by_user_id != tenant.actor_user_id:
                raise DomainNotFoundError

            status = _run_status(run.status)
            if status is RunStatus.WAITING_APPROVAL:
                if approval_request is None:
                    raise DomainInvariantError("waiting run has no approval request")
                action = await session.scalar(
                    select(ActionIntent)
                    .where(
                        ActionIntent.workspace_id == tenant.workspace_id,
                        ActionIntent.run_id == run_id,
                        ActionIntent.id == approval_request.action_intent_id,
                    )
                    .with_for_update()
                )
                if action is None:
                    raise DomainInvariantError("waiting run has no action intent")
                if persisted_action_intent_status(action.status) not in {
                    ActionIntentStatus.PROPOSED,
                    ActionIntentStatus.AUTHORIZED,
                }:
                    raise DomainInvariantError("waiting run active action changed")
                now = datetime.now(UTC)
                request_status = persisted_approval_status(approval_request.status)
                if request_status in {ApprovalStatus.PENDING, ApprovalStatus.APPROVED}:
                    approval_request.status = ApprovalStatus.EXPIRED.value
                    approval_request.version += 1
                    approval_request.updated_at = now
                    sequence = run.next_event_seq
                    run.next_event_seq += 1
                    session.add(
                        RunEvent(
                            id=uuid4(),
                            workspace_id=tenant.workspace_id,
                            run_id=run_id,
                            actor_user_id=tenant.actor_user_id,
                            seq=sequence,
                            type=RunEventType.APPROVAL_EXPIRED.value,
                            version=CURRENT_RUN_EVENT_VERSION,
                            payload={
                                "approval_request_id": str(approval_request.id),
                                "action_intent_id": str(action.id),
                                "reason": "user_requested",
                            },
                        )
                    )
                elif request_status not in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
                    raise DomainConflictError("approval state cannot be cancelled")
                action_status = persisted_action_intent_status(action.status)
                if action_status in {
                    ActionIntentStatus.PROPOSED,
                    ActionIntentStatus.AUTHORIZED,
                }:
                    action.status = ActionIntentStatus.CANCELLED.value
                    sequence = run.next_event_seq
                    run.next_event_seq += 1
                    session.add(
                        RunEvent(
                            id=uuid4(),
                            workspace_id=tenant.workspace_id,
                            run_id=run_id,
                            actor_user_id=tenant.actor_user_id,
                            seq=sequence,
                            type=RunEventType.ACTION_CANCELLED.value,
                            version=CURRENT_RUN_EVENT_VERSION,
                            payload={
                                "action_intent_id": str(action.id),
                                "approval_request_id": str(approval_request.id),
                                "reason": "user_requested",
                            },
                        )
                    )
                elif action_status is not ActionIntentStatus.CANCELLED:
                    raise DomainConflictError("action state cannot be cancelled")
                if job.status not in {
                    JobStatus.DONE.value,
                    JobStatus.QUEUED.value,
                    JobStatus.LEASED.value,
                }:
                    raise DomainInvariantError("waiting run job is not cancellable")
                job.status = JobStatus.DONE.value
                job.leased_by = None
                job.owner_token = None
                job.lease_expires_at = None
                job.error_summary = None
                previous_status = run.status
                run.status = RunStatus.CANCELLED.value
                run.finished_at = max(now, run.started_at or now)
                sequence = run.next_event_seq
                run.next_event_seq += 1
                session.add(
                    RunEvent(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        run_id=run_id,
                        actor_user_id=tenant.actor_user_id,
                        seq=sequence,
                        type=RunEventType.RUN_CANCELLED.value,
                        version=CURRENT_RUN_EVENT_VERSION,
                        payload={
                            "previous_status": previous_status,
                            "status": RunStatus.CANCELLED.value,
                            "reason": "user_requested",
                        },
                    )
                )
                status = RunStatus.CANCELLED
            if status is RunStatus.QUEUED:
                if job.status not in {JobStatus.QUEUED.value, JobStatus.LEASED.value}:
                    raise DomainInvariantError("queued run does not have one cancellable job")
                now = datetime.now(UTC)
                run.status = RunStatus.CANCELLED.value
                run.finished_at = now
                job.status = JobStatus.DONE.value
                job.leased_by = None
                job.owner_token = None
                job.lease_expires_at = None
                job.error_summary = None
                await session.flush()
                allocated_seq = await session.scalar(
                    update(Run)
                    .where(
                        Run.workspace_id == tenant.workspace_id,
                        Run.id == run_id,
                    )
                    .values(next_event_seq=Run.next_event_seq + 1)
                    .returning(Run.next_event_seq - 1)
                )
                if allocated_seq is None:
                    raise DomainInvariantError("run event sequence could not be allocated")
                session.add(
                    RunEvent(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        run_id=run_id,
                        actor_user_id=tenant.actor_user_id,
                        seq=allocated_seq,
                        type=RunEventType.RUN_CANCELLED.value,
                        version=CURRENT_RUN_EVENT_VERSION,
                        payload={
                            "previous_status": RunStatus.QUEUED.value,
                            "status": RunStatus.CANCELLED.value,
                            "reason": "user_requested",
                        },
                    )
                )
                status = RunStatus.CANCELLED
            elif status is RunStatus.RUNNING and run.cancel_requested_at is None:
                run.cancel_requested_at = datetime.now(UTC)

            cancellation = RunCancellation(
                run_id=run.id,
                status=status,
                cancel_requested_at=run.cancel_requested_at,
            )
        return cancellation
