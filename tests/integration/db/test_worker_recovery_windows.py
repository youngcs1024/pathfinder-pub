from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select, update

from app.db.checkpoints import open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    LLMInvocation,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.session import create_database_engine, create_session_factory, transaction
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.errors import DomainUnavailableError
from app.domain.jobs import ClaimedJob, JobStatus
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.runs import RunService, RunStatus
from app.domain.tenancy import TenantContext, TenantService
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationAuthorizationError
from app.domain.tracing import current_trace_scope
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import (
    LLMInvocationAttempt,
    LLMInvocationAuthorizationError,
)
from app.tools.fake_search import FakeSearch
from app.worker.contracts import RunExecutionResult
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.legacy_runtime import (
    SqlAlchemyRunExecutionReader,
    SqlAlchemyRunStore,
    SqlAlchemyWorkerJobStore,
)
from tests.tracing import CollectingTraceSink

pytestmark = pytest.mark.integration


class _ProviderFinishedRecorder:
    def __init__(self, delegate: SqlAlchemyInvocationRecorder) -> None:
        self._delegate = delegate
        self.provider_finished = asyncio.Event()
        self.release_to_graph = asyncio.Event()
        self.returned_to_graph = False

    async def prepare(self, attempt: object) -> None:
        await self._delegate.prepare(attempt)

    async def finalize(self, attempt: object, outcome: object) -> None:
        await self._delegate.finalize(attempt, outcome)
        if outcome.status == "succeeded":
            self.provider_finished.set()
            await self.release_to_graph.wait()
            self.returned_to_graph = True


class _BlockingChatAdapter(DeterministicResearchFakeChatAdapter):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def invoke(self, *args: object, **kwargs: object) -> object:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _FailingExecutionGuard:
    def __init__(self, delegate: SqlAlchemyRunExecutionReader) -> None:
        self._delegate = delegate
        self.fail = False

    async def read_for_execution(self, **kwargs: object) -> object:
        return await self._delegate.read_for_execution(**kwargs)

    async def assert_execution_allowed(self, **kwargs: object) -> None:
        if self.fail:
            raise DomainUnavailableError()
        await self._delegate.assert_execution_allowed(**kwargs)

    async def has_nonterminal_other_graph_versions(self, graph_version: str) -> bool:
        return await self._delegate.has_nonterminal_other_graph_versions(graph_version)


@dataclass(frozen=True)
class _RecoveryState:
    job: RunJob
    run: Run
    events: list[RunEvent]
    llm: list[LLMInvocation]
    tools: list[ToolInvocation]


async def _state(session_factory: object, run_id: object) -> _RecoveryState:
    async with session_factory() as session:
        job = await session.scalar(select(RunJob).where(RunJob.run_id == run_id))
        run = await session.scalar(select(Run).where(Run.id == run_id))
        events = list(
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
        )
        llm = list(
            await session.scalars(
                select(LLMInvocation)
                .where(LLMInvocation.run_id == run_id)
                .order_by(LLMInvocation.created_at, LLMInvocation.id)
            )
        )
        tools = list(
            await session.scalars(
                select(ToolInvocation)
                .where(ToolInvocation.run_id == run_id)
                .order_by(ToolInvocation.created_at, ToolInvocation.id)
            )
        )
    assert job is not None and run is not None
    return _RecoveryState(job=job, run=run, events=events, llm=llm, tools=tools)


def _executor(
    *,
    session_factory: object,
    checkpointer: object,
    recorder: object,
) -> LangGraphRunExecutor:
    return LangGraphRunExecutor(
        reader=SqlAlchemyRunExecutionReader(session_factory),
        checkpointer=checkpointer,
        llm_factory=LLMFactory(
            recorder=recorder,
            chat_adapter=DeterministicResearchFakeChatAdapter(),
            embedding_adapter=FakeEmbeddingModel(),
        ),
        search_port=FakeSearch({}),
        tool_recorder=SqlAlchemyToolInvocationRecorder(session_factory),
        document_repository=SqlAlchemyDocumentRepository(session_factory),
        retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(session_factory),
        execution_timeout_seconds=300.0,
        execution_guard_poll_seconds=0.01,
    )


async def _claim_and_prepare(
    store: SqlAlchemyWorkerJobStore,
    *,
    tenant: TenantContext,
    worker_id: str,
    now: datetime,
) -> ClaimedJob:
    claim = await store.claim_due_job(
        worker_id=worker_id,
        now=now,
        lease_duration=timedelta(seconds=30),
    )
    assert claim is not None
    prepared = await store.prepare_claimed_job(
        job=claim,
        resolved_tenant=tenant,
        now=now,
    )
    assert prepared.disposition == "execute" and prepared.tenant == tenant
    return claim


async def _setup_run(migrated_database_url: str, subject: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(session_factory)
    ).provision_personal_workspace(subject)
    tenant = TenantContext(
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
        role=WorkspaceRole.ADMIN,
    )
    accepted = await RunService(SqlAlchemyRunStore(session_factory)).create_research_run(
        tenant=tenant,
        query="Research deterministic checkpoint recovery",
    )
    store = SqlAlchemyWorkerJobStore(session_factory, lambda _attempt: timedelta(0))
    return engine, session_factory, tenant, accepted.run_id, store


def _blocking_executor(
    *,
    session_factory: object,
    checkpointer: object,
    adapter: _BlockingChatAdapter,
    reader: object | None = None,
) -> LangGraphRunExecutor:
    return LangGraphRunExecutor(
        reader=reader or SqlAlchemyRunExecutionReader(session_factory),
        checkpointer=checkpointer,
        llm_factory=LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(session_factory),
            chat_adapter=adapter,
            embedding_adapter=FakeEmbeddingModel(),
        ),
        search_port=FakeSearch({}),
        tool_recorder=SqlAlchemyToolInvocationRecorder(session_factory),
        document_repository=SqlAlchemyDocumentRepository(session_factory),
        retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(session_factory),
        execution_timeout_seconds=300.0,
        execution_guard_poll_seconds=0.01,
    )


def _runner(
    *,
    session_factory: object,
    store: SqlAlchemyWorkerJobStore,
    executor: LangGraphRunExecutor,
    trace_sink=None,
) -> WorkerRunner:
    return WorkerRunner(
        worker_id="step-46-cancellation-worker",
        trace_sink=trace_sink,
        store=store,
        tenant_service=TenantService(SqlAlchemyTenantResolver(session_factory)),
        executor=executor,
        settings=WorkerRuntimeSettings(
            heartbeat_seconds=1,
            lease_seconds=5,
            shutdown_grace_seconds=1,
        ),
    )


async def test_running_user_cancellation_converges_through_real_executor_and_worker(
    migrated_database_url: str,
) -> None:
    engine, session_factory, tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "step-46-running-user-cancel",
    )
    adapter = _BlockingChatAdapter()
    try:
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            worker_task = asyncio.create_task(
                _runner(
                    session_factory=session_factory,
                    store=store,
                    executor=_blocking_executor(
                        session_factory=session_factory,
                        checkpointer=checkpointer,
                        adapter=adapter,
                    ),
                ).run_once(asyncio.Event())
            )
            await asyncio.wait_for(adapter.started.wait(), timeout=5)
            cancellation = await RunService(SqlAlchemyRunStore(session_factory)).cancel_run(
                tenant=tenant,
                run_id=run_id,
            )
            assert cancellation.status is RunStatus.RUNNING
            assert cancellation.cancel_requested_at is not None
            assert await asyncio.wait_for(worker_task, timeout=5) is True

        state = await _state(session_factory, run_id)
        assert adapter.cancelled.is_set()
        assert state.job.status == JobStatus.DONE.value
        assert state.run.status == RunStatus.CANCELLED.value
        assert len(state.llm) == 1
        assert state.llm[0].status == "failed" and state.llm[0].error_category == "cancelled"
        assert state.tools == []
        assert [event.type for event in state.events] == [
            "run.created",
            "run.status_changed",
            "run.cancelled",
        ]
        assert state.events[-1].payload["reason"] == "user_requested"

        repeated = await RunService(SqlAlchemyRunStore(session_factory)).cancel_run(
            tenant=tenant,
            run_id=run_id,
        )
        assert repeated.status is RunStatus.CANCELLED
        repeated_state = await _state(session_factory, run_id)
        assert [event.type for event in repeated_state.events].count("run.cancelled") == 1
        assert not {"run.completed", "run.failed"}.intersection(
            event.type for event in repeated_state.events
        )
    finally:
        await engine.dispose()


async def test_membership_revocation_during_model_wait_cancels_without_new_attempt(
    migrated_database_url: str,
) -> None:
    engine, session_factory, tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "step-46-running-membership-revoke",
    )
    adapter = _BlockingChatAdapter()
    try:
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            worker_task = asyncio.create_task(
                _runner(
                    session_factory=session_factory,
                    store=store,
                    executor=_blocking_executor(
                        session_factory=session_factory,
                        checkpointer=checkpointer,
                        adapter=adapter,
                    ),
                ).run_once(asyncio.Event())
            )
            await asyncio.wait_for(adapter.started.wait(), timeout=5)
            async with transaction(session_factory) as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == tenant.workspace_id,
                        WorkspaceMembership.user_id == tenant.actor_user_id,
                    )
                    .values(revoked_at=datetime.now(UTC))
                )
            assert await asyncio.wait_for(worker_task, timeout=5) is True

        state = await _state(session_factory, run_id)
        assert adapter.cancelled.is_set()
        assert state.job.status == JobStatus.DONE.value
        assert state.run.status == RunStatus.CANCELLED.value
        assert len(state.llm) == 1 and state.llm[0].error_category == "cancelled"
        assert state.tools == []
        assert [event.type for event in state.events].count("run.cancelled") == 1
        assert state.events[-1].payload["reason"] == "authorization_revoked"
    finally:
        await engine.dispose()


async def test_execution_guard_infrastructure_failure_requeues_without_run_cancelled(
    migrated_database_url: str,
) -> None:
    engine, session_factory, _tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "step-46-execution-guard-failure",
    )
    adapter = _BlockingChatAdapter()
    reader = _FailingExecutionGuard(SqlAlchemyRunExecutionReader(session_factory))
    try:
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            worker_task = asyncio.create_task(
                _runner(
                    session_factory=session_factory,
                    store=store,
                    executor=_blocking_executor(
                        session_factory=session_factory,
                        checkpointer=checkpointer,
                        adapter=adapter,
                        reader=reader,
                    ),
                ).run_once(asyncio.Event())
            )
            await asyncio.wait_for(adapter.started.wait(), timeout=5)
            reader.fail = True
            assert await asyncio.wait_for(worker_task, timeout=5) is True

        state = await _state(session_factory, run_id)
        assert adapter.cancelled.is_set()
        assert state.job.status == JobStatus.QUEUED.value
        assert state.job.error_summary == "execution_guard_unavailable"
        assert state.run.status == RunStatus.RUNNING.value
        assert len(state.llm) == 1 and state.llm[0].error_category == "cancelled"
        assert "run.cancelled" not in [event.type for event in state.events]
        assert "run.failed" not in [event.type for event in state.events]
    finally:
        await engine.dispose()


async def test_persisted_cancel_rejects_new_model_and_tool_attempts_before_send(
    migrated_database_url: str,
) -> None:
    engine, session_factory, tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "step-46-cancel-before-new-attempt",
    )
    try:
        now = datetime.now(UTC)
        await _claim_and_prepare(
            store,
            tenant=tenant,
            worker_id="boundary-worker",
            now=now,
        )
        cancellation = await RunService(SqlAlchemyRunStore(session_factory)).cancel_run(
            tenant=tenant,
            run_id=run_id,
        )
        assert cancellation.status is RunStatus.RUNNING

        with pytest.raises(LLMInvocationAuthorizationError):
            await SqlAlchemyInvocationRecorder(session_factory).prepare(
                LLMInvocationAttempt(
                    invocation_id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    actor_user_id=tenant.actor_user_id,
                    run_id=run_id,
                    invocation_kind="chat",
                    provider="fake",
                    model="qwen3.6-flash-2026-04-16",
                    graph_node="plan",
                    prompt_version=f"sha256:{'a' * 64}",
                    request_hash=f"sha256:{'b' * 64}",
                )
            )
        with pytest.raises(ToolInvocationAuthorizationError):
            await SqlAlchemyToolInvocationRecorder(session_factory).reserve(
                invocation_id=uuid4(),
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                run_id=run_id,
                tool_name="search_web",
                effect=ToolEffect.READ_ONLY,
                args_digest=f"sha256:{'c' * 64}",
                call_limit=8,
            )

        state = await _state(session_factory, run_id)
        assert state.llm == [] and state.tools == []
        assert state.job.status == JobStatus.LEASED.value
        assert state.run.status == RunStatus.RUNNING.value
        assert state.run.cancel_requested_at is not None
    finally:
        await engine.dispose()


async def test_provider_finalization_completes_before_cooperative_cancellation_recovery(
    migrated_database_url: str,
) -> None:
    engine, session_factory, tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "provider-finalization-before-cancellation",
    )
    try:
        started_at = datetime.now(UTC)
        first_claim = await _claim_and_prepare(
            store,
            tenant=tenant,
            worker_id="crashed-worker",
            now=started_at,
        )
        recorder = _ProviderFinishedRecorder(SqlAlchemyInvocationRecorder(session_factory))
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            execution_task = asyncio.create_task(
                _executor(
                    session_factory=session_factory,
                    checkpointer=checkpointer,
                    recorder=recorder,
                ).execute(run_id, tenant, first_claim.graph_version)
            )
            barrier_task = asyncio.create_task(recorder.provider_finished.wait())
            done, _pending = await asyncio.wait(
                {execution_task, barrier_task},
                timeout=5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution_task in done:
                pytest.fail(f"executor ended before checkpoint barrier: {execution_task.result()}")
            assert barrier_task in done
            before_crash = await _state(session_factory, run_id)
            assert recorder.returned_to_graph is False
            assert len(before_crash.llm) == 1
            assert before_crash.llm[0].status == "succeeded"
            assert before_crash.job.status == JobStatus.LEASED.value
            assert before_crash.run.status == RunStatus.RUNNING.value

            execution_task.cancel()
            await asyncio.sleep(0)
            assert execution_task.done() is False
            recorder.release_to_graph.set()
            with pytest.raises(asyncio.CancelledError):
                await execution_task
            after_crash = await _state(session_factory, run_id)
            assert recorder.returned_to_graph is True
            assert after_crash.job.status == JobStatus.LEASED.value
            assert after_crash.job.owner_token == first_claim.owner_token
            persisted = await checkpointer.aget_tuple({"configurable": {"thread_id": str(run_id)}})
            assert persisted is not None
            plan_checkpointed_after_crash = "plan" in persisted.checkpoint["channel_values"]

            expired_at = started_at + timedelta(seconds=31)
            reclaimed = await store.reclaim_stale_leases(now=expired_at, limit=10)
            assert reclaimed.requeued == 1
            second_claim = await _claim_and_prepare(
                store,
                tenant=tenant,
                worker_id="recovery-worker",
                now=expired_at,
            )
            assert second_claim.attempt == 2
            assert second_claim.owner_token != first_claim.owner_token
            assert (
                await store.complete(
                    job=first_claim,
                    result=before_crash.run.result_json,
                    now=expired_at,
                )
                is False
            )

            recovered = await _executor(
                session_factory=session_factory,
                checkpointer=checkpointer,
                recorder=SqlAlchemyInvocationRecorder(session_factory),
            ).execute(run_id, tenant, second_claim.graph_version)
            assert recovered.status is RunStatus.COMPLETED and recovered.result is not None
            assert await store.complete(job=second_claim, result=recovered.result, now=expired_at)

        final = await _state(session_factory, run_id)
        expected_llm_count = 6 if plan_checkpointed_after_crash else 7
        assert len(final.llm) == expected_llm_count > len(before_crash.llm)
        assert len(final.tools) == 2
        assert all(row.status in {"succeeded", "failed"} for row in final.llm)
        assert final.job.status == JobStatus.DONE.value and final.job.attempt == 2
        assert final.run.status == RunStatus.COMPLETED.value
        assert [event.seq for event in final.events] == list(range(1, len(final.events) + 1))
        assert [event.type for event in final.events].count("run.completed") == 1
        assert [event.type for event in final.events].count("job.lease_expired") == 1
    finally:
        await engine.dispose()


async def test_completed_checkpoint_recovers_business_finalize_without_new_invocations(
    migrated_database_url: str,
) -> None:
    engine, session_factory, tenant, run_id, store = await _setup_run(
        migrated_database_url,
        "step-46-checkpoint-before-finalize",
    )
    try:
        started_at = datetime.now(UTC)
        first_claim = await _claim_and_prepare(
            store,
            tenant=tenant,
            worker_id="crashed-worker",
            now=started_at,
        )
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            first = await _executor(
                session_factory=session_factory,
                checkpointer=checkpointer,
                recorder=SqlAlchemyInvocationRecorder(session_factory),
            ).execute(run_id, tenant, first_claim.graph_version)
            assert first.status is RunStatus.COMPLETED and first.result is not None
            before_crash = await _state(session_factory, run_id)
            llm_count = len(before_crash.llm)
            tool_count = len(before_crash.tools)
            assert llm_count == 6 and tool_count == 2
            assert before_crash.run.status == RunStatus.RUNNING.value
            assert before_crash.run.result_json is None
            assert before_crash.job.status == JobStatus.LEASED.value

            expired_at = started_at + timedelta(seconds=31)
            reclaimed = await store.reclaim_stale_leases(now=expired_at, limit=10)
            assert reclaimed.requeued == 1
            second_claim = await _claim_and_prepare(
                store,
                tenant=tenant,
                worker_id="recovery-worker",
                now=expired_at,
            )
            assert second_claim.attempt == 2
            assert second_claim.owner_token != first_claim.owner_token
            assert (
                await store.complete(job=first_claim, result=first.result, now=expired_at) is False
            )

            second = await _executor(
                session_factory=session_factory,
                checkpointer=checkpointer,
                recorder=SqlAlchemyInvocationRecorder(session_factory),
            ).execute(run_id, tenant, second_claim.graph_version)
            assert second == first
            after_resume = await _state(session_factory, run_id)
            assert len(after_resume.llm) == llm_count
            assert len(after_resume.tools) == tool_count
            assert await store.complete(job=second_claim, result=second.result, now=expired_at)

        final = await _state(session_factory, run_id)
        assert final.job.status == JobStatus.DONE.value and final.job.attempt == 2
        assert final.run.status == RunStatus.COMPLETED.value
        assert len(final.llm) == llm_count and len(final.tools) == tool_count
        assert [event.seq for event in final.events] == list(range(1, len(final.events) + 1))
        assert [event.type for event in final.events].count("run.completed") == 1
        assert [event.type for event in final.events].count("job.lease_expired") == 1
    finally:
        await engine.dispose()


async def test_trace_retry_uses_new_segment_and_checkpoint_without_duplicate_invocations(
    migrated_database_url: str,
) -> None:
    engine, session_factory, _tenant, run_id, store = await _setup_run(
        migrated_database_url, "step-102-trace-retry"
    )
    sink = CollectingTraceSink()
    scopes = []

    class RetryAfterCheckpoint:
        def __init__(self, delegate):
            self.delegate = delegate

        async def execute(self, run_id, tenant, graph_version):
            scopes.append(current_trace_scope())
            result = await self.delegate.execute(run_id, tenant, graph_version)
            assert result.status is RunStatus.COMPLETED
            if len(scopes) == 1:
                # Controlled interruption between completed checkpoint and job finalization.
                return RunExecutionResult(
                    status=RunStatus.FAILED,
                    error_category="executor_unhandled_error",
                    retryable=True,
                )
            return result

    try:
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
            executor = RetryAfterCheckpoint(
                _executor(
                    session_factory=session_factory,
                    checkpointer=checkpointer,
                    recorder=SqlAlchemyInvocationRecorder(session_factory),
                )
            )
            runner = _runner(
                session_factory=session_factory, store=store, executor=executor, trace_sink=sink
            )
            assert await runner.run_once(asyncio.Event())
            before = await _state(session_factory, run_id)
            assert before.job.status == JobStatus.QUEUED.value
            assert before.job.attempt == 1
            assert len(before.llm) == 6 and len(before.tools) == 2
            assert current_trace_scope() is None
            assert await runner.run_once(asyncio.Event())
            after = await _state(session_factory, run_id)

        roots = [
            (context, span)
            for context, span in sink.starts.values()
            if span.span_kind == "execution_segment"
        ]
        assert len(roots) == 2
        assert [span.segment_identity.attempt for _, span in roots] == [1, 2]
        assert roots[0][1].segment_identity.job_id == roots[1][1].segment_identity.job_id
        assert roots[0][1].trace_identity == roots[1][1].trace_identity
        assert roots[0][1].trace_identity.run_id == run_id
        assert roots[0][0].context_id != roots[1][0].context_id
        assert all(span.parent is None for _, span in roots)
        assert [scope.parent for scope in scopes] == [context for context, _ in roots]
        assert [sink.finishes[context.context_id].status for context, _ in roots] == [
            "failed",
            "succeeded",
        ]
        assert sink.starts.keys() == sink.finishes.keys()
        assert current_trace_scope() is None
        assert after.job.status == JobStatus.DONE.value and after.job.attempt == 2
        assert after.run.status == RunStatus.COMPLETED.value
        assert [row.id for row in after.llm] == [row.id for row in before.llm]
        assert [row.id for row in after.tools] == [row.id for row in before.tools]
        assert [event.id for event in after.events[: len(before.events)]] == [
            event.id for event in before.events
        ]
        assert [event.type for event in after.events].count("run.completed") == 1
        assert [event.seq for event in after.events] == list(range(1, len(after.events) + 1))
    finally:
        await engine.dispose()


async def test_trace_restart_after_checkpoint_uses_only_fresh_memory(migrated_database_url):
    from tests.tracing import assert_closed_trace_tree

    engine, sessions, _tenant, run_id, store = await _setup_run(
        migrated_database_url, "step-105-fresh-memory"
    )
    old_sink = CollectingTraceSink()
    new_sink = CollectingTraceSink()

    class StopBeforeFinalize:
        def __getattr__(self, name):
            return getattr(store, name)

        async def complete(self, **kwargs):
            # Graph/checkpoint and invocation commits already succeeded. The process
            # disappears before the worker's business finalization transaction begins.
            raise ConnectionError("TRACE_SECRET_CANARY_BEFORE_FINALIZE")

    try:
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as saver:
            runner = _runner(
                session_factory=sessions,
                store=StopBeforeFinalize(),
                trace_sink=old_sink,
                executor=_executor(
                    session_factory=sessions,
                    checkpointer=saver,
                    recorder=SqlAlchemyInvocationRecorder(sessions),
                ),
            )
            with pytest.raises(ConnectionError, match="BEFORE_FINALIZE"):
                await runner.run_once(asyncio.Event())
            completed = await saver.aget_tuple({"configurable": {"thread_id": str(run_id)}})
            assert completed is not None and completed.checkpoint["channel_values"]["output"]
            checkpoint_id = completed.checkpoint["id"]
        before = await _state(sessions, run_id)
        assert before.run.status == "running" and before.run.result_json is None
        assert before.job.status == "leased" and before.job.attempt == 1
        assert len(before.llm) == 6 and len(before.tools) == 2
        assert all(r.status == "succeeded" for r in [*before.llm, *before.tools])
        assert current_trace_scope() is None
        expired_at = before.job.lease_expires_at + timedelta(seconds=1)
        reclaimed = await store.reclaim_stale_leases(now=expired_at, limit=10)
        assert reclaimed.requeued == 1
        queued = await _state(sessions, run_id)
        assert queued.job.status == "queued" and queued.run.status == "running"
        assert [r.id for r in queued.llm] == [r.id for r in before.llm]

        # Reopen the durable checkpointer; instantiate all execution/observation objects
        # afresh. No old scope, context_id or vendor object is supplied to recovery.
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as saver:
            fresh = WorkerRunner(
                worker_id="step-105-restarted",
                store=store,
                tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
                executor=_executor(
                    session_factory=sessions,
                    checkpointer=saver,
                    recorder=SqlAlchemyInvocationRecorder(sessions),
                ),
                settings=WorkerRuntimeSettings(),
                trace_sink=new_sink,
                clock=lambda: expired_at,
            )
            assert await fresh.run_once(asyncio.Event())
            recovered = await saver.aget_tuple({"configurable": {"thread_id": str(run_id)}})
            assert recovered.checkpoint["id"] == checkpoint_id
        after = await _state(sessions, run_id)
        assert after.run.status == "completed" and after.job.status == "done"
        assert after.job.attempt == 2
        assert after.job.owner_token is None and after.job.lease_expires_at is None
        assert [r.id for r in after.llm] == [r.id for r in before.llm]
        assert [r.id for r in after.tools] == [r.id for r in before.tools]
        assert [e.id for e in after.events[: len(before.events)]] == [e.id for e in before.events]
        assert [e.type for e in after.events].count("job.lease_expired") == 1
        assert [e.type for e in after.events].count("run.completed") == 1
        assert [e.seq for e in after.events] == list(range(1, len(after.events) + 1))
        assert_closed_trace_tree(old_sink, segments=1)
        assert_closed_trace_tree(new_sink, segments=1)
        old_root = next(
            c for c, s in old_sink.starts.values() if s.span_kind == "execution_segment"
        )
        [(new_root, new_start)] = new_sink.starts.values()
        assert new_root.trace_identity == old_root.trace_identity
        assert new_root.trace_identity.run_id == run_id
        assert new_root.segment_identity.job_id == old_root.segment_identity.job_id
        assert (old_root.segment_identity.attempt, new_root.segment_identity.attempt) == (1, 2)
        assert new_start.parent is None and new_root.context_id not in old_sink.starts
        for old_id in old_sink.starts:
            assert str(old_id) not in new_sink.safe_json()
            assert str(old_id) not in repr(recovered)
        assert new_sink.finishes[new_root.context_id].status == "succeeded"
        # This exception seam unwinds finally; an actual SIGKILL need not close old spans.
        assert old_sink.finishes[old_root.context_id].error_category == "worker_unhandled_error"
        assert "TRACE_SECRET_CANARY" not in old_sink.safe_json() + new_sink.safe_json()
    finally:
        await engine.dispose()


async def test_worker_retry_with_exhausted_persisted_tool_budget(migrated_database_url):
    engine, sessions, tenant, run_id, store = await _setup_run(
        migrated_database_url, "f2-tool-budget-retry"
    )

    class FailAfterTools(DeterministicResearchFakeChatAdapter):
        failed = False

        async def invoke(self, messages, tools, metadata, *, attempt=None):
            count = await SqlAlchemyToolInvocationRecorder(sessions).consumed_call_count(
                workspace_id=tenant.workspace_id,
                run_id=run_id,
            )
            if count == 8 and not self.failed:
                self.failed = True
                from app.llm.factory import LLMProviderError

                raise LLMProviderError(category="provider_unavailable")
            return await super().invoke(messages, tools, metadata, attempt=attempt)

    try:
        async with sessions.begin() as session:
            # Simulate seven reservations already committed before a worker restart.
            for _ in range(7):
                session.add(
                    ToolInvocation(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        originating_actor_user_id=tenant.actor_user_id,
                        run_id=run_id,
                        tool_name="search_web",
                        effect="read_only",
                        args_digest=f"sha256:{'a' * 64}",
                        status="prepared",
                        attempt=0,
                    )
                )
        async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as saver:
            executor = _executor(
                session_factory=sessions,
                checkpointer=saver,
                recorder=SqlAlchemyInvocationRecorder(sessions),
            )
            executor._llm_factory = LLMFactory(
                recorder=SqlAlchemyInvocationRecorder(sessions),
                chat_adapter=FailAfterTools(),
                embedding_adapter=FakeEmbeddingModel(),
            )
            runner = _runner(session_factory=sessions, store=store, executor=executor)
            assert await runner.run_once(asyncio.Event())
            before = await _state(sessions, run_id)
            assert before.job.status == "queued"
            assert len(before.tools) == 8
            assert await runner.run_once(asyncio.Event())
            after = await _state(sessions, run_id)
            assert after.job.attempt == 2
            assert after.job.status == "done"
            assert after.run.status == "completed", after.run.error_category
            assert after.run.result_json["evidence_sufficient"] is False
            assert len(after.tools) == 8
    finally:
        await engine.dispose()
