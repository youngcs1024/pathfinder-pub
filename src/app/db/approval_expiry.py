from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ApprovalRequest, Run, RunEvent, RunJob
from app.db.session import AsyncSessionFactory, transaction
from app.domain.approvals import ApprovalExpirySweepSummary, ApprovalStatus
from app.domain.errors import DomainInvariantError
from app.domain.jobs import JobStatus
from app.domain.runs import RunStatus
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


def _append_expiry_event(
    session: AsyncSession,
    *,
    run: Run,
    request: ApprovalRequest,
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
            type=RunEventType.APPROVAL_EXPIRED.value,
            version=CURRENT_RUN_EVENT_VERSION,
            payload={
                "reason": "natural_expiry",
                "approval_request_id": str(request.id),
                "action_intent_id": str(request.action_intent_id),
            },
        )
    )


class SqlAlchemyApprovalRequestExpirySweeper:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def sweep_due_approval_requests(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> ApprovalExpirySweepSummary:
        if now.tzinfo is None:
            raise ValueError("approval expiry sweep time must be timezone-aware")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("approval expiry sweep limit must be positive")
        expired = 0
        requeued = 0
        async with transaction(self._session_factory) as session:
            requests = list(
                await session.scalars(
                    select(ApprovalRequest)
                    .where(
                        ApprovalRequest.status.in_(
                            (ApprovalStatus.PENDING.value, ApprovalStatus.APPROVED.value)
                        ),
                        ApprovalRequest.expires_at <= now,
                    )
                    .order_by(ApprovalRequest.expires_at, ApprovalRequest.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            )
            for request in requests:
                run = await session.scalar(
                    select(Run)
                    .where(
                        Run.workspace_id == request.workspace_id,
                        Run.id == request.run_id,
                    )
                    .with_for_update()
                )
                if run is None:
                    raise DomainInvariantError("approval expiry run is missing")
                if run.status != RunStatus.WAITING_APPROVAL.value:
                    continue
                job = await session.scalar(
                    select(RunJob)
                    .where(
                        RunJob.workspace_id == request.workspace_id,
                        RunJob.run_id == request.run_id,
                    )
                    .with_for_update()
                )
                if job is None:
                    raise DomainInvariantError("approval expiry job is missing")

                request.status = ApprovalStatus.EXPIRED.value
                request.version += 1
                request.updated_at = now
                _append_expiry_event(session, run=run, request=request)
                expired += 1

                if job.status == JobStatus.DONE.value:
                    job.status = JobStatus.QUEUED.value
                    job.attempt = 0
                    job.available_at = now
                    job.error_summary = None
                    job.resume_approval_request_id = request.id
                    requeued += 1
                elif job.status == JobStatus.QUEUED.value:
                    if job.resume_approval_request_id not in (None, request.id):
                        raise DomainInvariantError("queued job has a conflicting resume identity")
                    job.resume_approval_request_id = request.id
                elif job.status == JobStatus.LEASED.value:
                    if job.resume_approval_request_id != request.id:
                        raise DomainInvariantError("leased job has a conflicting resume identity")
                else:
                    raise DomainInvariantError("approval expiry job status is invalid")

        return ApprovalExpirySweepSummary(
            scanned=len(requests),
            expired=expired,
            requeued=requeued,
        )
