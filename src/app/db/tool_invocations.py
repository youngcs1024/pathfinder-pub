from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy import func, select, update

from app.db.models import Run, ToolInvocation, WorkspaceMembership
from app.db.session import AsyncSessionFactory, transaction
from app.domain.runs import RunStatus
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import (
    ToolInvocationAuthorizationError,
    ToolInvocationLimitError,
    ToolInvocationRecorderError,
    ToolInvocationReservation,
)


def _now() -> datetime:
    return datetime.now(UTC)


class SqlAlchemyToolInvocationRecorder:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def consumed_call_count(self, *, workspace_id: UUID, run_id: UUID) -> int:
        async with transaction(self._session_factory) as session:
            count = await session.scalar(
                select(func.count())
                .select_from(ToolInvocation)
                .where(
                    ToolInvocation.workspace_id == workspace_id,
                    ToolInvocation.run_id == run_id,
                )
            )
            if count is None:
                raise ToolInvocationRecorderError
            return count

    async def reserve(
        self,
        *,
        invocation_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        tool_name: str,
        effect: ToolEffect,
        args_digest: str,
        call_limit: int,
    ) -> ToolInvocationReservation:
        async with transaction(self._session_factory) as session:
            run = await session.scalar(
                select(Run)
                .where(Run.workspace_id == workspace_id, Run.id == run_id)
                .with_for_update()
            )
            active_membership = await session.scalar(
                select(WorkspaceMembership.id).where(
                    WorkspaceMembership.workspace_id == workspace_id,
                    WorkspaceMembership.user_id == actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
            )
            if (
                active_membership is None
                or run is None
                or run.created_by_user_id != actor_user_id
                or run.status != RunStatus.RUNNING.value
                or run.cancel_requested_at is not None
            ):
                raise ToolInvocationAuthorizationError
            prior_count = await session.scalar(
                select(func.count())
                .select_from(ToolInvocation)
                .where(
                    ToolInvocation.workspace_id == workspace_id,
                    ToolInvocation.run_id == run_id,
                )
            )
            if prior_count is None or prior_count >= call_limit:
                raise ToolInvocationLimitError
            call_number = prior_count + 1
            session.add(
                ToolInvocation(
                    id=invocation_id,
                    workspace_id=workspace_id,
                    originating_actor_user_id=actor_user_id,
                    run_id=run_id,
                    tool_name=tool_name,
                    effect=effect.value,
                    args_digest=args_digest,
                    status="prepared",
                    attempt=0,
                    latency_ms=None,
                    result_summary=None,
                    error_category=None,
                    started_at=None,
                    finished_at=None,
                )
            )
        return ToolInvocationReservation(invocation_id, call_number)

    async def start_attempt(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        attempt: int,
    ) -> None:
        async with transaction(self._session_factory) as session:
            active_membership = await session.scalar(
                select(WorkspaceMembership.id).where(
                    WorkspaceMembership.workspace_id == workspace_id,
                    WorkspaceMembership.user_id == actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
            )
            run = await session.scalar(
                select(Run).where(Run.workspace_id == workspace_id, Run.id == run_id)
            )
            if (
                active_membership is None
                or run is None
                or run.created_by_user_id != actor_user_id
                or run.status != RunStatus.RUNNING.value
                or run.cancel_requested_at is not None
            ):
                raise ToolInvocationAuthorizationError
            expected_status = "prepared" if attempt == 1 else "executing"
            expected_attempt = 0 if attempt == 1 else attempt - 1
            values: dict[str, object] = {"status": "executing", "attempt": attempt}
            if attempt == 1:
                values["started_at"] = _now()
            result = await session.execute(
                update(ToolInvocation)
                .where(
                    ToolInvocation.id == reservation.invocation_id,
                    ToolInvocation.workspace_id == workspace_id,
                    ToolInvocation.run_id == run_id,
                    ToolInvocation.status == expected_status,
                    ToolInvocation.attempt == expected_attempt,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                current = (
                    await session.execute(
                        select(ToolInvocation.status, ToolInvocation.attempt).where(
                            ToolInvocation.id == reservation.invocation_id,
                            ToolInvocation.workspace_id == workspace_id,
                            ToolInvocation.run_id == run_id,
                        )
                    )
                ).one_or_none()
                if current != ("executing", attempt):
                    raise ToolInvocationRecorderError

    async def succeed(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        run_id: UUID,
        latency_ms: int,
        result_summary: dict[str, JsonValue],
    ) -> None:
        await self._finish(
            reservation=reservation,
            workspace_id=workspace_id,
            run_id=run_id,
            status="succeeded",
            latency_ms=latency_ms,
            result_summary=result_summary,
            error_category=None,
        )

    async def fail(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        run_id: UUID,
        latency_ms: int,
        error_category: str,
    ) -> None:
        await self._finish(
            reservation=reservation,
            workspace_id=workspace_id,
            run_id=run_id,
            status="failed",
            latency_ms=latency_ms,
            result_summary=None,
            error_category=error_category,
        )

    async def _finish(
        self,
        *,
        reservation: ToolInvocationReservation,
        workspace_id: UUID,
        run_id: UUID,
        status: str,
        latency_ms: int,
        result_summary: dict[str, JsonValue] | None,
        error_category: str | None,
    ) -> None:
        async with transaction(self._session_factory) as session:
            result = await session.execute(
                update(ToolInvocation)
                .where(
                    ToolInvocation.id == reservation.invocation_id,
                    ToolInvocation.workspace_id == workspace_id,
                    ToolInvocation.run_id == run_id,
                    ToolInvocation.status.in_(("prepared", "executing")),
                )
                .values(
                    status=status,
                    latency_ms=latency_ms,
                    result_summary=result_summary,
                    error_category=error_category,
                    finished_at=_now(),
                )
            )
            if result.rowcount != 1:
                raise ToolInvocationRecorderError
