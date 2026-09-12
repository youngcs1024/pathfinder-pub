from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.domain.action_execution import (
    ActionExecutionFailedError,
    ActionExecutionIdentity,
    ActionOutcomeUnknownError,
    ActionResultUnconfirmedError,
)
from app.domain.approvals import ApprovalExpirySweepSummary
from app.domain.jobs import ClaimedJob, PrepareClaimResult, ReclaimSummary
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchLimitationV1, ResearchOutputV1
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext, TenantService
from app.domain.tracing import ExecutionSegmentIdentity, TraceIdentity, current_trace_scope
from app.worker.contracts import RunExecutionResult
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.tracing import CollectingTraceSink


def _result() -> ResearchOutputV1:
    return ResearchOutputV1(
        evidence_sufficient=False,
        limitations=(ResearchLimitationV1(code="insufficient_evidence", detail="No evidence."),),
    )


def _job(*, attempt: int = 1, max_attempts: int = 3) -> ClaimedJob:
    return ClaimedJob(
        job_id=uuid4(),
        workspace_id=uuid4(),
        run_id=uuid4(),
        originating_actor_user_id=uuid4(),
        graph_version="pathfinder-research-v1",
        attempt=attempt,
        max_attempts=max_attempts,
        owner_token=uuid4(),
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=30),
    )


class _TenantResolver:
    def __init__(self, tenant: TenantContext | None) -> None:
        self.tenant = tenant

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        if self.tenant is None:
            return None
        if workspace_id == self.tenant.workspace_id and actor_user_id == self.tenant.actor_user_id:
            return self.tenant
        return None


@dataclass
class _Store:
    claimed: ClaimedJob | None
    prepared: PrepareClaimResult
    heartbeat_result: bool = True
    finalize_result: bool = True
    calls: list[tuple[str, object]] = field(default_factory=list)

    async def reclaim_stale_leases(self, *, now: datetime, limit: int) -> ReclaimSummary:
        self.calls.append(("reclaim", limit))
        return ReclaimSummary(scanned=0, requeued=0, finished=0, dead=0)

    async def claim_due_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedJob | None:
        self.calls.append(("claim", worker_id))
        claimed, self.claimed = self.claimed, None
        return claimed

    async def prepare_claimed_job(
        self,
        *,
        job: ClaimedJob,
        resolved_tenant: TenantContext | None,
        now: datetime,
    ) -> PrepareClaimResult:
        self.calls.append(("prepare", resolved_tenant))
        return self.prepared

    async def heartbeat(
        self,
        *,
        job: ClaimedJob,
        now: datetime,
        lease_duration: timedelta,
    ) -> bool:
        self.calls.append(("heartbeat", job.owner_token))
        return self.heartbeat_result

    async def complete(
        self,
        *,
        job: ClaimedJob,
        result: ResearchOutputV1,
        now: datetime,
    ) -> bool:
        self.calls.append(("complete", result))
        return self.finalize_result

    async def wait_for_approval(
        self,
        *,
        job: ClaimedJob,
        approval_request_id: UUID,
        now: datetime,
    ) -> bool:
        self.calls.append(("wait_for_approval", approval_request_id))
        return self.finalize_result

    async def requeue(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self.calls.append(("requeue", error_category))
        return self.finalize_result

    async def fail(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self.calls.append(("fail", error_category))
        return self.finalize_result

    async def cancel(
        self,
        *,
        job: ClaimedJob,
        reason: str,
        now: datetime,
    ) -> bool:
        self.calls.append(("cancel", reason))
        return self.finalize_result

    async def release(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool:
        self.calls.append(("release", error_category))
        return self.finalize_result


class _Executor:
    def __init__(self, result: RunExecutionResult) -> None:
        self.result = result
        self.calls: list[tuple[UUID, TenantContext, str]] = []
        self.scope = None

    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        self.scope = current_trace_scope()
        self.calls.append((run_id, tenant, graph_version))
        return self.result


class _ActionExecutor:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def execute_approved_action(self, identity, **kwargs):
        self.calls.append((identity, kwargs["cancellation"].is_cancelled()))
        if self.error is not None:
            raise self.error
        return '{"external_ref":"mock-submission:recovered"}'


class _ExpirySweeper:
    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self.calls = calls

    async def sweep_due_approval_requests(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> ApprovalExpirySweepSummary:
        self.calls.append(("sweep", limit))
        return ApprovalExpirySweepSummary(scanned=0, expired=0, requeued=0)


class _BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _InvalidExecutor:
    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> object:
        return object()


class _HeartbeatErrorStore(_Store):
    async def heartbeat(
        self,
        *,
        job: ClaimedJob,
        now: datetime,
        lease_duration: timedelta,
    ) -> bool:
        self.calls.append(("heartbeat_error", job.owner_token))
        raise RuntimeError("sensitive database failure")


def _tenant(job: ClaimedJob) -> TenantContext:
    return TenantContext(
        workspace_id=job.workspace_id,
        actor_user_id=job.originating_actor_user_id,
        role=WorkspaceRole.ADMIN,
    )


def _runner(
    *,
    job: ClaimedJob,
    store: _Store,
    executor,
    tenant: TenantContext | None,
    settings: WorkerRuntimeSettings | None = None,
    expiry_sweeper: _ExpirySweeper | None = None,
    action_executor: _ActionExecutor | None = None,
    trace_sink=None,
) -> WorkerRunner:
    return WorkerRunner(
        worker_id="unit-worker",
        store=store,
        tenant_service=TenantService(_TenantResolver(tenant)),
        executor=executor,
        settings=settings or WorkerRuntimeSettings(),
        approval_expiry_sweeper=expiry_sweeper,
        approved_action_executor=action_executor,
        trace_sink=trace_sink,
    )


async def test_runner_reclaims_then_sweeps_then_claims() -> None:
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
    )
    await _runner(
        job=job,
        store=store,
        executor=_Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result())),
        tenant=tenant,
        expiry_sweeper=_ExpirySweeper(store.calls),
    ).run_once(asyncio.Event())

    assert [call[0] for call in store.calls[:3]] == ["reclaim", "sweep", "claim"]


async def test_runner_passes_current_tenant_and_finalizes_completed_result() -> None:
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
    )
    executor = _Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result()))

    processed = await _runner(
        job=job,
        store=store,
        executor=executor,
        tenant=tenant,
    ).run_once(asyncio.Event())

    assert processed is True
    assert executor.calls == [(job.run_id, tenant, job.graph_version)]
    assert [name for name, _value in store.calls] == ["reclaim", "claim", "prepare", "complete"]


async def test_runner_does_not_execute_revoked_membership() -> None:
    job = _job()
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="finished"),
    )
    executor = _Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result()))

    processed = await _runner(
        job=job,
        store=store,
        executor=executor,
        tenant=None,
    ).run_once(asyncio.Event())

    assert processed is True
    assert executor.calls == []
    assert store.calls[-1] == ("prepare", None)


async def test_runner_reconciles_executing_action_without_fabricating_revoked_tenant() -> None:
    job = _job()
    identity = ActionExecutionIdentity(job.workspace_id, job.run_id, uuid4(), uuid4())
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="recover_action", action_identity=identity),
    )
    graph_executor = _Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result()))
    action_executor = _ActionExecutor()

    processed = await _runner(
        job=job,
        store=store,
        executor=graph_executor,
        tenant=None,
        action_executor=action_executor,
    ).run_once(asyncio.Event())

    assert processed is True
    assert graph_executor.calls == []
    assert action_executor.calls == [(identity, False)]
    assert store.calls[-1] == ("cancel", "action_reconciled_after_cancellation")


async def test_runner_terminalizes_unknown_recovery_as_failed_run_done_job() -> None:
    job = _job(attempt=3, max_attempts=3)
    identity = ActionExecutionIdentity(job.workspace_id, job.run_id, uuid4(), uuid4())
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="recover_action", action_identity=identity),
    )
    action_executor = _ActionExecutor(ActionOutcomeUnknownError())

    await _runner(
        job=job,
        store=store,
        executor=_Executor(RunExecutionResult(status=RunStatus.CANCELLED)),
        tenant=None,
        action_executor=action_executor,
    ).run_once(asyncio.Event())

    assert store.calls[-1] == ("fail", "external_outcome_unknown")


async def test_retryable_and_waiting_results_use_fixed_job_paths() -> None:
    retry_job = _job(attempt=1)
    retry_tenant = _tenant(retry_job)
    retry_store = _Store(
        claimed=retry_job,
        prepared=PrepareClaimResult(disposition="execute", tenant=retry_tenant),
    )
    retry_executor = _Executor(
        RunExecutionResult(
            status=RunStatus.FAILED,
            error_category="provider_timeout",
            retryable=True,
        )
    )
    await _runner(
        job=retry_job,
        store=retry_store,
        executor=retry_executor,
        tenant=retry_tenant,
    ).run_once(asyncio.Event())
    assert retry_store.calls[-1] == ("requeue", "provider_timeout")

    waiting_job = _job()
    waiting_tenant = _tenant(waiting_job)
    waiting_store = _Store(
        claimed=waiting_job,
        prepared=PrepareClaimResult(disposition="execute", tenant=waiting_tenant),
    )
    request_id = uuid4()
    await _runner(
        job=waiting_job,
        store=waiting_store,
        executor=_Executor(
            RunExecutionResult(
                status=RunStatus.WAITING_APPROVAL,
                approval_request_id=request_id,
            )
        ),
        tenant=waiting_tenant,
    ).run_once(asyncio.Event())
    assert waiting_store.calls[-1] == ("wait_for_approval", request_id)
    assert not any(call[0] == "heartbeat" for call in waiting_store.calls)


async def test_invalid_executor_return_is_a_non_retryable_contract_failure() -> None:
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
    )

    await _runner(
        job=job,
        store=store,
        executor=_InvalidExecutor(),
        tenant=tenant,
    ).run_once(asyncio.Event())

    assert store.calls[-1] == ("fail", "executor_contract_violation")


async def test_heartbeat_lease_loss_cancels_executor_without_finalize() -> None:
    sink = CollectingTraceSink()
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
        heartbeat_result=False,
    )
    executor = _BlockingExecutor()
    settings = WorkerRuntimeSettings(
        heartbeat_seconds=0.01,
        lease_seconds=0.1,
        shutdown_grace_seconds=0.02,
    )

    await _runner(
        job=job,
        store=store,
        executor=executor,
        tenant=tenant,
        settings=settings,
        trace_sink=sink,
    ).run_once(asyncio.Event())

    assert executor.cancelled.is_set()
    assert [name for name, _value in store.calls][-1] == "heartbeat"

    _assert_segment(sink, job, "cancelled", "lease_lost", "execute")
    assert "sensitive database failure" not in sink.safe_json()


async def test_heartbeat_error_cancels_executor_without_finalize() -> None:
    sink = CollectingTraceSink()
    job = _job()
    tenant = _tenant(job)
    store = _HeartbeatErrorStore(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
    )
    executor = _BlockingExecutor()
    settings = WorkerRuntimeSettings(
        heartbeat_seconds=0.01,
        lease_seconds=0.1,
        shutdown_grace_seconds=0.02,
    )

    await _runner(
        job=job,
        store=store,
        executor=executor,
        tenant=tenant,
        settings=settings,
        trace_sink=sink,
    ).run_once(asyncio.Event())

    assert executor.cancelled.is_set()
    assert [name for name, _value in store.calls][-1] == "heartbeat_error"

    _assert_segment(sink, job, "cancelled", "lease_lost", "execute")
    assert "sensitive database failure" not in sink.safe_json()


async def test_shutdown_grace_timeout_releases_without_claiming_another_job() -> None:
    sink = CollectingTraceSink()
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        claimed=job,
        prepared=PrepareClaimResult(disposition="execute", tenant=tenant),
    )
    executor = _BlockingExecutor()
    settings = WorkerRuntimeSettings(
        heartbeat_seconds=0.01,
        lease_seconds=0.1,
        shutdown_grace_seconds=0.02,
    )
    stop_requested = asyncio.Event()
    task = asyncio.create_task(
        _runner(
            job=job,
            store=store,
            executor=executor,
            tenant=tenant,
            settings=settings,
            trace_sink=sink,
        ).run_once(stop_requested)
    )
    await executor.started.wait()
    stop_requested.set()

    assert await task is True
    assert executor.cancelled.is_set()
    assert store.calls[-1] == ("release", "worker_shutdown")

    _assert_segment(sink, job, "cancelled", "worker_shutdown", "execute")
    assert "sensitive database failure" not in sink.safe_json()


def _assert_segment(sink, job, status, category, disposition):
    segments = [
        (context, span)
        for context, span in sink.starts.values()
        if span.span_kind == "execution_segment"
    ]
    assert len(segments) == 1
    context, span = segments[0]
    assert span.trace_identity == TraceIdentity(workspace_id=job.workspace_id, run_id=job.run_id)
    assert span.segment_identity == ExecutionSegmentIdentity(job_id=job.job_id, attempt=job.attempt)
    assert span.parent is None
    assert dict(span.metadata) == {"graph_version": job.graph_version}
    finish = sink.finishes[context.context_id]
    assert (finish.status, finish.error_category) == (status, category)
    assert dict(finish.metadata) == ({"disposition": disposition} if disposition else {})
    assert finish.latency_ms >= 0
    assert sink.starts.keys() == sink.finishes.keys()
    assert current_trace_scope() is None
    return context


@pytest.mark.parametrize("case", ["completed", "waiting", "retry", "invalid", "cancelled"])
async def test_execution_segment_result_and_executor_task_inheritance(case) -> None:
    job = _job()
    tenant = _tenant(job)
    request_id = uuid4()
    results = {
        "completed": RunExecutionResult(status=RunStatus.COMPLETED, result=_result()),
        "waiting": RunExecutionResult(
            status=RunStatus.WAITING_APPROVAL, approval_request_id=request_id
        ),
        "retry": RunExecutionResult(
            status=RunStatus.FAILED, error_category="provider_timeout", retryable=True
        ),
        "cancelled": RunExecutionResult(status=RunStatus.CANCELLED),
    }
    executor = _InvalidExecutor() if case == "invalid" else _Executor(results[case])
    store = _Store(job, PrepareClaimResult(disposition="execute", tenant=tenant))
    sink = CollectingTraceSink()
    assert await _runner(
        job=job, store=store, executor=executor, tenant=tenant, trace_sink=sink
    ).run_once(asyncio.Event())
    expected = {
        "completed": ("succeeded", None, "complete"),
        "waiting": ("succeeded", None, "wait_for_approval"),
        "retry": ("failed", "provider_timeout", "requeue"),
        "invalid": ("failed", "executor_contract_violation", "fail"),
        "cancelled": ("cancelled", "executor_cancelled", "cancel"),
    }
    status, category, operation = expected[case]
    parent = _assert_segment(sink, job, status, category, "execute")
    assert len(sink.starts) == 1
    assert store.calls[-1][0] == operation
    if category is not None:
        assert store.calls[-1][1] == category
    if case != "invalid":
        assert executor.scope.parent == parent
        assert executor.scope.sink is sink


@pytest.mark.parametrize("disposition", ["finished", "lease_lost"])
async def test_nonexecute_segment(disposition) -> None:
    job = _job()
    store = _Store(job, PrepareClaimResult(disposition=disposition))
    sink = CollectingTraceSink()
    executor = _Executor(RunExecutionResult(status=RunStatus.CANCELLED))
    assert await _runner(
        job=job, store=store, executor=executor, tenant=None, trace_sink=sink
    ).run_once(asyncio.Event())
    assert executor.calls == []
    _assert_segment(
        sink,
        job,
        "skipped" if disposition == "finished" else "cancelled",
        None if disposition == "finished" else "lease_lost",
        disposition,
    )


async def test_finalize_cas_false_closes_segment_without_duplicate_finalize() -> None:
    job = _job()
    tenant = _tenant(job)
    store = _Store(
        job, PrepareClaimResult(disposition="execute", tenant=tenant), finalize_result=False
    )
    sink = CollectingTraceSink()
    await _runner(
        job=job,
        store=store,
        tenant=tenant,
        trace_sink=sink,
        executor=_Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result())),
    ).run_once(asyncio.Event())
    assert [name for name, _ in store.calls] == ["reclaim", "claim", "prepare", "complete"]
    _assert_segment(sink, job, "cancelled", "lease_lost", "execute")


@pytest.mark.parametrize(
    "error,operation,category",
    [
        (None, "cancel", "action_reconciled_after_cancellation"),
        (ActionOutcomeUnknownError(), "fail", "external_outcome_unknown"),
        (ActionExecutionFailedError(), "cancel", "action_confirmed_no_effect"),
        (ActionResultUnconfirmedError(), "fail", "external_action_unconfirmed"),
    ],
)
async def test_action_recovery_child_preserves_business_transition(error, operation, category):
    job = _job()
    identity = ActionExecutionIdentity(job.workspace_id, job.run_id, uuid4(), uuid4())
    store = _Store(job, PrepareClaimResult(disposition="recover_action", action_identity=identity))
    sink = CollectingTraceSink()
    action_executor = _ActionExecutor(error)
    await _runner(
        job=job,
        store=store,
        tenant=None,
        trace_sink=sink,
        executor=_InvalidExecutor(),
        action_executor=action_executor,
    ).run_once(asyncio.Event())
    status, trace_error = ("succeeded", None) if error is None else ("failed", category)
    parent = _assert_segment(sink, job, status, trace_error, "recover_action")
    assert len(sink.starts) == 2
    child, start = list(sink.starts.values())[1]
    assert start.parent == parent and start.span_kind == "action_recovery"
    assert dict(start.metadata) == {
        "action_intent_id": identity.action_intent_id,
        "approval_request_id": identity.approval_request_id,
    }
    finish = sink.finishes[child.context_id]
    assert (finish.status, finish.error_category) == (status, trace_error)
    assert store.calls[-1] == (operation, category)
    assert action_executor.calls == [(identity, False)]


@pytest.mark.parametrize("failure", ["start", "finish"])
async def test_broken_trace_sink_does_not_change_completed_business_result(failure):
    class BrokenSink(CollectingTraceSink):
        def start(self, span):
            if failure == "start":
                raise RuntimeError("TRACE-BUSINESS-BODY-CANARY")
            return super().start(span)

        def finish(self, context, outcome):
            raise RuntimeError("TRACE-BUSINESS-BODY-CANARY")

    job = _job()
    tenant = _tenant(job)
    store = _Store(job, PrepareClaimResult(disposition="execute", tenant=tenant))
    sink = BrokenSink()
    await _runner(
        job=job,
        store=store,
        tenant=tenant,
        trace_sink=sink,
        executor=_Executor(RunExecutionResult(status=RunStatus.COMPLETED, result=_result())),
    ).run_once(asyncio.Event())
    assert store.calls[-1] == ("complete", _result())
    assert current_trace_scope() is None
    assert "TRACE-BUSINESS-BODY-CANARY" not in sink.safe_json()


async def test_unexpected_prepare_exception_propagates_and_closes_segment():
    class BrokenStore(_Store):
        async def prepare_claimed_job(self, **kwargs):
            assert current_trace_scope() is not None
            raise RuntimeError("TRACE-BUSINESS-BODY-CANARY")

    job = _job()
    store = BrokenStore(job, PrepareClaimResult(disposition="finished"))
    sink = CollectingTraceSink()
    with pytest.raises(RuntimeError, match="TRACE-BUSINESS-BODY-CANARY"):
        await _runner(
            job=job, store=store, tenant=None, trace_sink=sink, executor=_InvalidExecutor()
        ).run_once(asyncio.Event())
    _assert_segment(sink, job, "failed", "worker_unhandled_error", None)
    assert "TRACE-BUSINESS-BODY-CANARY" not in sink.safe_json()


async def test_runner_cancellation_closes_segment_and_cancels_child_task():
    job = _job()
    tenant = _tenant(job)
    store = _Store(job, PrepareClaimResult(disposition="execute", tenant=tenant))
    sink = CollectingTraceSink()
    executor = _BlockingExecutor()
    task = asyncio.create_task(
        _runner(job=job, store=store, tenant=tenant, trace_sink=sink, executor=executor).run_once(
            asyncio.Event()
        )
    )
    await executor.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert executor.cancelled.is_set()
    assert store.calls[-1][0] == "prepare"
    _assert_segment(sink, job, "cancelled", "worker_cancelled", "execute")


async def test_completed_executor_does_not_hide_simultaneous_heartbeat_loss():
    completed = asyncio.Event()

    class CompletingExecutor(_Executor):
        async def execute(self, *args):
            await completed.wait()
            return self.result

    class RacingStore(_Store):
        async def heartbeat(self, **kwargs):
            completed.set()
            return False

    job = _job()
    tenant = _tenant(job)
    store = RacingStore(job, PrepareClaimResult(disposition="execute", tenant=tenant))
    sink = CollectingTraceSink()
    await _runner(
        job=job,
        store=store,
        tenant=tenant,
        trace_sink=sink,
        executor=CompletingExecutor(
            RunExecutionResult(status=RunStatus.COMPLETED, result=_result())
        ),
        settings=WorkerRuntimeSettings(heartbeat_seconds=0.01, lease_seconds=0.1),
    ).run_once(asyncio.Event())
    _assert_segment(sink, job, "cancelled", "lease_lost", "execute")
    # Preserve the original executor-first business branch; observability cannot override CAS.
    assert store.calls[-1] == ("complete", _result())
