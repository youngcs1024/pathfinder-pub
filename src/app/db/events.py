from __future__ import annotations

from itertools import pairwise
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import aliased

from app.db.models import Run, RunEvent, WorkspaceMembership
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import (
    DomainInvariantError,
    DomainNotFoundError,
    InvalidEventCursorError,
)
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.events.contracts import RunEventBatch, RunEventRecord, RunEventType

_TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)


class SqlAlchemyRunEventReader:
    """Read one bounded event page in one short, workspace-scoped session."""

    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def read_after(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        after_seq: int,
        limit: int,
    ) -> RunEventBatch:
        if not isinstance(tenant, TenantContext) or not isinstance(run_id, UUID):
            raise DomainNotFoundError
        if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
            raise InvalidEventCursorError("event cursor must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise DomainInvariantError("event replay batch limit is invalid")

        high_watermark_event = aliased(RunEvent)
        high_watermark = (
            select(func.coalesce(func.max(high_watermark_event.seq), 0))
            .where(
                high_watermark_event.workspace_id == tenant.workspace_id,
                high_watermark_event.run_id == run_id,
            )
            .scalar_subquery()
        )
        statement = (
            select(
                Run.status,
                high_watermark.label("high_watermark"),
                RunEvent.run_id,
                RunEvent.seq,
                RunEvent.type,
                RunEvent.version,
                RunEvent.payload,
                RunEvent.recorded_at,
            )
            .join(
                WorkspaceMembership,
                and_(
                    WorkspaceMembership.workspace_id == Run.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                    WorkspaceMembership.role == tenant.role.value,
                    WorkspaceMembership.revoked_at.is_(None),
                ),
            )
            .outerjoin(
                RunEvent,
                and_(
                    RunEvent.workspace_id == Run.workspace_id,
                    RunEvent.run_id == Run.id,
                    RunEvent.seq > after_seq,
                ),
            )
            .where(
                Run.workspace_id == tenant.workspace_id,
                Run.id == run_id,
            )
            .order_by(RunEvent.seq.asc().nulls_last())
            .limit(limit)
        )

        async with database_session(self._session_factory) as session:
            rows = (await session.execute(statement)).all()

        if not rows:
            raise DomainNotFoundError
        persisted_status = rows[0].status
        try:
            run_status = RunStatus(persisted_status)
        except ValueError:
            raise DomainInvariantError("persisted run status is invalid") from None
        persisted_high_watermark = rows[0].high_watermark
        if (
            isinstance(persisted_high_watermark, bool)
            or not isinstance(persisted_high_watermark, int)
            or persisted_high_watermark < 0
        ):
            raise DomainInvariantError("persisted run event high watermark is invalid")
        if after_seq > persisted_high_watermark:
            raise InvalidEventCursorError("event cursor is ahead of the committed event history")

        events: list[RunEventRecord] = []
        for row in rows:
            if row.seq is None:
                continue
            try:
                event_type = RunEventType(row.type)
            except ValueError:
                raise DomainInvariantError("persisted run event type is invalid") from None
            if row.version != 1 or not isinstance(row.payload, dict):
                raise DomainInvariantError("persisted run event data is invalid")
            events.append(
                RunEventRecord(
                    run_id=row.run_id,
                    seq=row.seq,
                    type=event_type,
                    version=row.version,
                    payload=row.payload,
                    recorded_at=row.recorded_at,
                )
            )

        if any(current.seq <= previous.seq for previous, current in pairwise(events)):
            raise DomainInvariantError("persisted run event sequence is not strictly increasing")
        return RunEventBatch(
            events=tuple(events),
            high_watermark=persisted_high_watermark,
            run_is_terminal=run_status in _TERMINAL_RUN_STATUSES,
        )
