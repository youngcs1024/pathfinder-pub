from __future__ import annotations

from sqlalchemy import select, update

from app.db.models import LLMInvocation, Run, WorkspaceMembership
from app.db.session import AsyncSessionFactory, transaction
from app.domain.runs import RunStatus
from app.llm.invocations import (
    LLMInvocationAttempt,
    LLMInvocationAuthorizationError,
    LLMInvocationInvariantError,
    LLMInvocationOutcome,
)


class SqlAlchemyInvocationRecorder:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        if not isinstance(attempt, LLMInvocationAttempt):
            raise LLMInvocationInvariantError
        async with transaction(self._session_factory) as session:
            active_membership = await session.scalar(
                select(WorkspaceMembership.id).where(
                    WorkspaceMembership.workspace_id == attempt.workspace_id,
                    WorkspaceMembership.user_id == attempt.actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
            )
            if active_membership is None:
                raise LLMInvocationAuthorizationError
            if attempt.run_id is not None:
                matching_run = await session.scalar(
                    select(Run.id).where(
                        Run.workspace_id == attempt.workspace_id,
                        Run.id == attempt.run_id,
                        Run.created_by_user_id == attempt.actor_user_id,
                        Run.status == RunStatus.RUNNING.value,
                        Run.cancel_requested_at.is_(None),
                    )
                )
                if matching_run is None:
                    raise LLMInvocationAuthorizationError
            session.add(
                LLMInvocation(
                    id=attempt.invocation_id,
                    workspace_id=attempt.workspace_id,
                    actor_user_id=attempt.actor_user_id,
                    run_id=attempt.run_id,
                    invocation_kind=attempt.invocation_kind,
                    provider=attempt.provider,
                    model=attempt.model,
                    graph_node=attempt.graph_node,
                    prompt_version=attempt.prompt_version,
                    request_hash=attempt.request_hash,
                    provider_response_id=None,
                    token_usage=None,
                    pricing_version=None,
                    currency=None,
                    estimated_cost=None,
                    trace_ids=None,
                    latency_ms=None,
                    status="started",
                    error_category=None,
                )
            )

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        if not isinstance(attempt, LLMInvocationAttempt) or not isinstance(
            outcome, LLMInvocationOutcome
        ):
            raise LLMInvocationInvariantError
        token_usage = (
            outcome.token_usage.model_dump(mode="json", round_trip=True)
            if outcome.token_usage is not None
            else None
        )
        trace_ids = (
            outcome.trace_ids.model_dump(mode="json", round_trip=True)
            if outcome.trace_ids is not None
            else None
        )
        async with transaction(self._session_factory) as session:
            result = await session.execute(
                update(LLMInvocation)
                .where(
                    LLMInvocation.id == attempt.invocation_id,
                    LLMInvocation.workspace_id == attempt.workspace_id,
                    LLMInvocation.actor_user_id == attempt.actor_user_id,
                    LLMInvocation.run_id == attempt.run_id,
                    LLMInvocation.status == "started",
                )
                .values(
                    status=outcome.status,
                    provider_response_id=outcome.provider_response_id,
                    token_usage=token_usage,
                    pricing_version=outcome.pricing_version,
                    currency=outcome.currency,
                    estimated_cost=outcome.estimated_cost,
                    trace_ids=trace_ids,
                    latency_ms=outcome.latency_ms,
                    error_category=outcome.error_category,
                )
            )
            if result.rowcount == 1:
                return

            existing = (
                await session.execute(
                    select(
                        LLMInvocation.status,
                        LLMInvocation.provider_response_id,
                        LLMInvocation.token_usage,
                        LLMInvocation.pricing_version,
                        LLMInvocation.currency,
                        LLMInvocation.estimated_cost,
                        LLMInvocation.trace_ids,
                        LLMInvocation.latency_ms,
                        LLMInvocation.error_category,
                    ).where(
                        LLMInvocation.id == attempt.invocation_id,
                        LLMInvocation.workspace_id == attempt.workspace_id,
                        LLMInvocation.actor_user_id == attempt.actor_user_id,
                        LLMInvocation.run_id == attempt.run_id,
                    )
                )
            ).one_or_none()
            if existing is None or (
                existing.status,
                existing.provider_response_id,
                existing.token_usage,
                existing.pricing_version,
                existing.currency,
                existing.estimated_cost,
                existing.trace_ids,
                existing.latency_ms,
                existing.error_category,
            ) != (
                outcome.status,
                outcome.provider_response_id,
                token_usage,
                outcome.pricing_version,
                outcome.currency,
                outcome.estimated_cost,
                trace_ids,
                outcome.latency_ms,
                outcome.error_category,
            ):
                raise LLMInvocationInvariantError
