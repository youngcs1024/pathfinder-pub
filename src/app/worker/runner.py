from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Literal

from app.domain.action_execution import (
    ActionExecutionFailedError,
    ActionExecutionIdentity,
    ActionOutcomeUnknownError,
    ActionResultUnconfirmedError,
)
from app.domain.approvals import ApprovalRequestExpirySweeper
from app.domain.errors import DomainNotFoundError
from app.domain.jobs import ClaimedJob, WorkerJobStore
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext, TenantService
from app.domain.tracing import (
    ActiveTraceScope,
    ExecutionSegmentIdentity,
    SpanStatus,
    TraceIdentity,
    TraceSinkPort,
    bind_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.obs.logging import get_logger
from app.tools.contracts import ApprovedActionExecutor, CancellationCheck
from app.worker.contracts import RunExecutionResult, RunExecutor
from app.worker.settings import WorkerRuntimeSettings

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class WorkerRunner:
    def __init__(
        self,
        *,
        worker_id: str,
        store: WorkerJobStore,
        tenant_service: TenantService,
        executor: RunExecutor,
        settings: WorkerRuntimeSettings,
        approval_expiry_sweeper: ApprovalRequestExpirySweeper | None = None,
        approved_action_executor: ApprovedActionExecutor | None = None,
        trace_sink: TraceSinkPort | None = None,
        clock: Clock = utc_now,
    ) -> None:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be non-blank")
        self._worker_id = worker_id
        self._store = store
        self._tenant_service = tenant_service
        self._executor = executor
        self._settings = settings
        self._approval_expiry_sweeper = approval_expiry_sweeper
        self._approved_action_executor = approved_action_executor
        self._trace_sink = trace_sink
        self._clock = clock
        self._logger = get_logger("pathfinder.worker")

    async def run(self, stop_requested: asyncio.Event) -> None:
        while not stop_requested.is_set():
            processed = await self.run_once(stop_requested)
            if processed or stop_requested.is_set():
                continue
            try:
                await asyncio.wait_for(
                    stop_requested.wait(),
                    timeout=self._settings.poll_seconds,
                )
            except TimeoutError:
                continue

    async def run_once(self, stop_requested: asyncio.Event) -> bool:
        now = self._clock()
        reclaimed = await self._store.reclaim_stale_leases(
            now=now,
            limit=self._settings.stale_batch_size,
        )
        if reclaimed.scanned:
            self._logger.info(
                "worker_stale_leases_reclaimed",
                scanned=reclaimed.scanned,
                requeued=reclaimed.requeued,
                finished=reclaimed.finished,
                dead=reclaimed.dead,
            )
        if stop_requested.is_set():
            return False

        if self._approval_expiry_sweeper is not None:
            swept = await self._approval_expiry_sweeper.sweep_due_approval_requests(
                now=self._clock(),
                limit=self._settings.approval_expiry_batch_size,
            )
            if swept.expired:
                self._logger.info(
                    "worker_approval_requests_expired",
                    scanned=swept.scanned,
                    expired=swept.expired,
                    requeued=swept.requeued,
                )
        if stop_requested.is_set():
            return False

        job = await self._store.claim_due_job(
            worker_id=self._worker_id,
            now=self._clock(),
            lease_duration=timedelta(seconds=self._settings.lease_seconds),
        )
        if job is None:
            return False

        scope = ActiveTraceScope(
            trace_identity=TraceIdentity(workspace_id=job.workspace_id, run_id=job.run_id),
            segment_identity=ExecutionSegmentIdentity(job_id=job.job_id, attempt=job.attempt),
            sink=self._trace_sink,
        )
        started_at = monotonic()
        segment = start_trace_span(
            scope, span_kind="execution_segment", metadata={"graph_version": job.graph_version}
        )
        scope = replace(scope, parent=segment)
        status: SpanStatus = "failed"
        error_category: str | None = "worker_unhandled_error"
        disposition = None
        # Bind before _execute_with_heartbeat creates the executor child task.
        with bind_trace_scope(scope):
            try:
                self._logger.info(
                    "worker_job_claimed",
                    job_id=str(job.job_id),
                    run_id=str(job.run_id),
                    workspace_id=str(job.workspace_id),
                    attempt=job.attempt,
                )
                resolved_tenant = None
                try:
                    resolved_tenant = await self._tenant_service.resolve_tenant(
                        workspace_id=job.workspace_id,
                        actor_user_id=job.originating_actor_user_id,
                    )
                except DomainNotFoundError:
                    pass

                prepared = await self._store.prepare_claimed_job(
                    job=job, resolved_tenant=resolved_tenant, now=self._clock()
                )
                disposition = prepared.disposition
                if disposition == "recover_action":
                    if self._approved_action_executor is None or prepared.action_identity is None:
                        raise RuntimeError("action recovery executor is unavailable")
                    status, error_category = await self._recover_action(
                        job=job, identity=prepared.action_identity, scope=scope
                    )
                    return True
                if disposition != "execute":
                    self._logger.info(
                        "worker_job_not_executed",
                        job_id=str(job.job_id),
                        run_id=str(job.run_id),
                        disposition=disposition,
                    )
                    status = "cancelled" if disposition == "lease_lost" else "skipped"
                    error_category = "lease_lost" if disposition == "lease_lost" else None
                    return True
                if prepared.tenant is None:
                    raise RuntimeError("prepared executable job did not include a tenant")

                if stop_requested.is_set():
                    released = await self._store.release(
                        job=job, error_category="worker_shutdown", now=self._clock()
                    )
                    status = "cancelled"
                    error_category = "worker_shutdown" if released else "lease_lost"
                    return True

                result, interruption = await self._execute_with_heartbeat(
                    job=job, tenant=prepared.tenant, stop_requested=stop_requested
                )
                if result is None:
                    status, error_category = "cancelled", interruption
                    return True
                finalized = await self._finalize(job=job, result=result)
                if not finalized or interruption == "lease_lost":
                    status, error_category = "cancelled", "lease_lost"
                elif result.status is RunStatus.FAILED:
                    status, error_category = "failed", result.error_category
                elif result.status is RunStatus.CANCELLED:
                    status, error_category = "cancelled", "executor_cancelled"
                else:
                    status, error_category = "succeeded", None
                return True
            except asyncio.CancelledError:
                status, error_category = "cancelled", "worker_cancelled"
                raise
            finally:
                finish_trace_span(
                    scope,
                    segment,
                    started_at=started_at,
                    status=status,
                    error_category=error_category,
                    metadata={"disposition": disposition} if disposition is not None else {},
                )

    async def _recover_action(
        self, *, job: ClaimedJob, identity: ActionExecutionIdentity, scope: ActiveTraceScope
    ) -> tuple[SpanStatus, str | None]:
        started_at = monotonic()
        child = start_trace_span(
            scope,
            span_kind="action_recovery",
            metadata={
                "action_intent_id": identity.action_intent_id,
                "approval_request_id": identity.approval_request_id,
            },
        )
        status: SpanStatus = "failed"
        error_category: str | None = "worker_unhandled_error"
        with bind_trace_scope(replace(scope, parent=child)):
            try:
                assert self._approved_action_executor is not None
                try:
                    await self._approved_action_executor.execute_approved_action(
                        identity,
                        deadline=monotonic() + self._settings.lease_seconds,
                        cancellation=_RecoveryCancellation(),
                    )
                except ActionOutcomeUnknownError:
                    error_category = "external_outcome_unknown"
                    finalized = await self._store.fail(
                        job=job, error_category=error_category, now=self._clock()
                    )
                except ActionExecutionFailedError:
                    error_category = "action_confirmed_no_effect"
                    finalized = await self._store.cancel(
                        job=job, reason=error_category, now=self._clock()
                    )
                except ActionResultUnconfirmedError:
                    error_category = "external_action_unconfirmed"
                    finalized = await self._store.fail(
                        job=job, error_category=error_category, now=self._clock()
                    )
                else:
                    finalized = await self._store.cancel(
                        job=job, reason="action_reconciled_after_cancellation", now=self._clock()
                    )
                    status, error_category = "succeeded", None
                if not finalized:
                    status, error_category = "cancelled", "lease_lost"
                return status, error_category
            except asyncio.CancelledError:
                status, error_category = "cancelled", "worker_cancelled"
                raise
            except Exception:
                status, error_category = "failed", "worker_unhandled_error"
                raise
            finally:
                finish_trace_span(
                    scope,
                    child,
                    started_at=started_at,
                    status=status,
                    error_category=error_category,
                )

    async def _execute_with_heartbeat(
        self,
        *,
        job: ClaimedJob,
        tenant: TenantContext,
        stop_requested: asyncio.Event,
    ) -> tuple[RunExecutionResult | None, Literal["lease_lost", "worker_shutdown"] | None]:
        executor_task = asyncio.create_task(
            self._executor.execute(job.run_id, tenant, job.graph_version)
        )
        heartbeat_done = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(
                job=job,
                done=heartbeat_done,
                lease_lost=lease_lost,
            )
        )
        stop_task = asyncio.create_task(stop_requested.wait())
        lease_lost_task = asyncio.create_task(lease_lost.wait())
        try:
            done, _pending = await asyncio.wait(
                {executor_task, stop_task, lease_lost_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if executor_task in done:
                return self._executor_result(executor_task), (
                    "lease_lost" if lease_lost.is_set() else None
                )
            if lease_lost_task in done and lease_lost.is_set():
                await self._cancel_task(executor_task)
                self._logger.warning(
                    "worker_lease_lost",
                    job_id=str(job.job_id),
                    run_id=str(job.run_id),
                )
                return None, "lease_lost"

            done_during_grace, _pending = await asyncio.wait(
                {executor_task, lease_lost_task},
                timeout=self._settings.shutdown_grace_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if executor_task in done_during_grace:
                return self._executor_result(executor_task), (
                    "lease_lost" if lease_lost.is_set() else None
                )
            if lease_lost_task in done_during_grace and lease_lost.is_set():
                await self._cancel_task(executor_task)
                return None, "lease_lost"

            await self._cancel_task(executor_task)
            released = await self._store.release(
                job=job,
                error_category="worker_shutdown",
                now=self._clock(),
            )
            self._logger.info(
                "worker_shutdown_release",
                job_id=str(job.job_id),
                run_id=str(job.run_id),
                released=released,
            )
            return None, "worker_shutdown" if released else "lease_lost"
        finally:
            if not executor_task.done():
                await self._cancel_task(executor_task)
            heartbeat_done.set()
            await self._cancel_task(heartbeat_task)
            await self._cancel_task(stop_task)
            await self._cancel_task(lease_lost_task)

    async def _heartbeat_loop(
        self,
        *,
        job: ClaimedJob,
        done: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        while not done.is_set():
            try:
                await asyncio.wait_for(
                    done.wait(),
                    timeout=self._settings.heartbeat_seconds,
                )
                return
            except TimeoutError:
                try:
                    renewed = await self._store.heartbeat(
                        job=job,
                        now=self._clock(),
                        lease_duration=timedelta(seconds=self._settings.lease_seconds),
                    )
                except Exception:
                    self._logger.warning(
                        "worker_heartbeat_failed",
                        job_id=str(job.job_id),
                        run_id=str(job.run_id),
                    )
                    lease_lost.set()
                    return
                if not renewed:
                    lease_lost.set()
                    return

    async def _finalize(self, *, job: ClaimedJob, result: RunExecutionResult) -> bool:
        now = self._clock()
        finalized = False
        if result.status is RunStatus.COMPLETED:
            if result.result is None:
                raise RuntimeError("completed executor result is missing its output")
            finalized = await self._store.complete(job=job, result=result.result, now=now)
        elif result.status is RunStatus.CANCELLED:
            finalized = await self._store.cancel(
                job=job,
                reason="executor_cancelled",
                now=now,
            )
        elif result.status is RunStatus.FAILED:
            if result.error_category is None:
                raise RuntimeError("failed executor result is missing its category")
            if result.retryable and job.attempt < job.max_attempts:
                finalized = await self._store.requeue(
                    job=job,
                    error_category=result.error_category,
                    now=now,
                )
            else:
                finalized = await self._store.fail(
                    job=job,
                    error_category=result.error_category,
                    now=now,
                )
        elif result.status is RunStatus.WAITING_APPROVAL:
            if result.approval_request_id is None:
                raise RuntimeError("waiting executor result is missing approval request identity")
            finalized = await self._store.wait_for_approval(
                job=job,
                approval_request_id=result.approval_request_id,
                now=now,
            )
        if not finalized:
            self._logger.warning(
                "worker_finalize_lease_lost",
                job_id=str(job.job_id),
                run_id=str(job.run_id),
            )

        return finalized

    @staticmethod
    def _executor_result(task: asyncio.Task[RunExecutionResult]) -> RunExecutionResult:
        try:
            result = task.result()
        except Exception:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="executor_unhandled_error",
                retryable=False,
            )
        if not isinstance(result, RunExecutionResult):
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="executor_contract_violation",
                retryable=False,
            )
        return result

    @staticmethod
    async def _cancel_task(task: asyncio.Task[object]) -> None:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class _RecoveryCancellation(CancellationCheck):
    def is_cancelled(self) -> bool:
        return False
