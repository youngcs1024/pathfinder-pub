from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select

from app.auth.fake import FAKE_ACTOR_SUBJECT
from app.config import Settings
from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.actions import SqlAlchemyActionStore
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.checkpoints import open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    LLMInvocation,
    MockSubmission,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.approvals import ApprovalDecisionCommand
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.tenancy import TenantContext, TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.documents import DocumentIngestionService
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.fake_search import FakeSearch
from app.tools.mock_application import create_approved_action_registry
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.main import _create_langgraph_executor, _run_with_checkpoint_supervisor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.support import connect_database
from tests.legacy_app import create_app
from tests.legacy_runtime import SqlAlchemyRunExecutionReader, SqlAlchemyWorkerJobStore

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("shutdown", ["graceful", "interrupted"])
async def test_checkpoint_backend_termination_stops_worker_and_recovers_exact_approval(
    migrated_database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    shutdown: str,
) -> None:
    application = create_app(
        Settings(
            database_url=SecretStr(migrated_database_url),
            llm_mode="fake",
            search_mode="fake",
            auth_mode="fake",
            trace_mode="off",
        )
    )
    posts = []

    async def count_post(request: httpx.Request) -> None:
        if request.method == "POST":
            posts.append(request.url.path)

    async with application.router.lifespan_context(application):
        sessions = application.state.database_session_factory
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace(FAKE_ACTOR_SUBJECT)
        tenant = TenantContext(identity.workspace_id, identity.user_id, WorkspaceRole.ADMIN)
        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=DeterministicResearchFakeChatAdapter(),
            embedding_adapter=FakeEmbeddingModel(),
        )
        canary = "E26-private-resume-canary Python PostgreSQL backend experience."
        batch = normalize_and_chunk_batch(
            ValidatedIngestionBatch(
                (
                    ValidatedIngestionSource(
                        source_name="resume.txt",
                        source_type="text",
                        title="Resume",
                        raw_text=canary,
                        character_count=len(canary),
                    ),
                )
            )
        )
        (document_id,) = await DocumentIngestionService(
            SqlAlchemyDocumentRepository(sessions),
            factory.create_embedding_model(
                LLMInvocationContext(identity.workspace_id, identity.user_id)
            ),
        ).ingest(tenant=tenant, batch=batch)
        store = SqlAlchemyWorkerJobStore(sessions, lambda _attempt: timedelta(0))
        runtime = WorkerRuntimeSettings(
            poll_seconds=0.05,
            heartbeat_seconds=0.1,
            shutdown_grace_seconds=0.1 if shutdown == "graceful" else 30,
        )
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://testserver"
            ) as api,
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://testserver",
                event_hooks={"request": [count_post]},
            ) as portal,
        ):
            response = await api.post(
                f"/api/v1/workspaces/{tenant.workspace_id}/runs",
                json={
                    "mode": "application",
                    "query": "Draft a backend application",
                    "resume_document_id": str(document_id),
                },
            )
            assert response.status_code == 202
            run_id = UUID(response.json()["run_id"])
            registry = create_approved_action_registry(
                adapter=MockPortalHTTPAdapter(portal),
                action_execution_store=SqlAlchemyActionExecutionStore(sessions),
            )

            def runner(checkpointer, worker_id, *, now=None):
                return WorkerRunner(
                    worker_id=worker_id,
                    store=store,
                    tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
                    executor=_create_langgraph_executor(
                        runtime_settings=runtime,
                        reader=SqlAlchemyRunExecutionReader(sessions),
                        checkpointer=checkpointer,
                        llm_factory=factory,
                        search_port=FakeSearch({}),
                        tool_recorder=SqlAlchemyToolInvocationRecorder(sessions),
                        document_repository=SqlAlchemyDocumentRepository(sessions),
                        retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(sessions),
                        action_store=SqlAlchemyActionStore(sessions),
                        approval_resume_resolver=SqlAlchemyApprovalStore(sessions),
                        approved_action_executor=registry,
                    ),
                    approved_action_executor=registry,
                    settings=runtime,
                    clock=lambda: now or datetime.now(UTC),
                )

            async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as saver:
                async with asyncio.timeout(30):
                    assert await runner(saver, "e26-prepare").run_once(asyncio.Event())
                async with sessions() as session:
                    request = await session.scalar(
                        select(ApprovalRequest).where(ApprovalRequest.run_id == run_id)
                    )
                    assert request is not None and request.status == "pending"
                    action = await session.get(ActionIntent, request.action_intent_id)
                    assert action is not None
                    binding = action.approval_binding_digest
                    idempotency_key = action.idempotency_key
                    llm_before = await session.scalar(
                        select(func.count()).select_from(LLMInvocation)
                    )
                await SqlAlchemyApprovalStore(sessions).decide(
                    ApprovalDecisionCommand(
                        tenant=tenant,
                        action_intent_id=request.action_intent_id,
                        decision="approve",
                        expected_version=request.version,
                        reason=None,
                        now=datetime.now(UTC),
                    )
                )
                read_entered = asyncio.Event()
                release_read = asyncio.Event()
                original_read = saver.aget_tuple

                async def gated_read(*args, **kwargs):
                    read_entered.set()
                    await release_read.wait()
                    return await original_read(*args, **kwargs)

                monkeypatch.setattr(saver, "aget_tuple", gated_read)
                stop = asyncio.Event()
                task = asyncio.create_task(
                    _run_with_checkpoint_supervisor(
                        runner=runner(saver, "e26-broken"),
                        checkpointer=saver,
                        worker_id="e26-broken",
                        heartbeat_seconds=runtime.heartbeat_seconds,
                        stop_requested=stop,
                    )
                )
                try:
                    async with asyncio.timeout(10):
                        await read_entered.wait()
                        pid = saver.conn.info.backend_pid
                        # The PID comes from this saver, never a broad application-name match.
                        with connect_database(migrated_database_url, autocommit=True) as control:
                            assert control.execute(
                                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                                "WHERE pid = %s AND datname = current_database()",
                                (pid,),
                            ).fetchone() == (True,)
                        release_read.set()  # The real saver read discovers the broken socket.
                        await stop.wait()
                        assert saver.conn.closed or saver.conn.broken
                        if shutdown == "interrupted":
                            # Simulate process cancellation during the supervisor's grace period.
                            task.cancel()
                            with pytest.raises(asyncio.CancelledError):
                                await task
                        else:
                            with pytest.raises(
                                RuntimeError, match="checkpoint connection was lost"
                            ):
                                await task
                finally:
                    release_read.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

            assert task.done()  # No second worker runs until the original has fully exited.
            async with sessions() as session:
                job = await session.scalar(select(RunJob).where(RunJob.run_id == run_id))
                run = await session.get(Run, run_id)
                assert job is not None and run is not None
                assert run.status == "running" and run.result_json is None
                assert job.status == ("queued" if shutdown == "graceful" else "leased")
                still_approved = await session.get(ApprovalRequest, request.id)
                assert still_approved.status == "approved"
                assert still_approved.approval_binding_digest == binding
                assert await session.scalar(select(func.count()).select_from(MockSubmission)) == 0
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(ToolInvocation)
                        .where(ToolInvocation.effect == "irreversible")
                    )
                    == 0
                )
                assert (
                    await session.scalar(select(func.count()).select_from(LLMInvocation))
                    == llm_before
                )
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(RunEvent)
                        .where(
                            RunEvent.run_id == run_id,
                            RunEvent.type.in_(["run.completed", "run.failed", "run.cancelled"]),
                        )
                    )
                    == 0
                )
            assert posts == []
            recovery_now = None
            if shutdown == "interrupted":
                assert job.lease_expires_at is not None
                recovery_now = job.lease_expires_at + timedelta(seconds=1)
                reclaimed = await store.reclaim_stale_leases(now=recovery_now, limit=10)
                assert reclaimed.requeued == 1 and reclaimed.dead == 0

            async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as recovered:
                assert recovered.conn.info.backend_pid != pid
                # Reclaim uses a future test clock; use that clock for claiming the due job too.
                recovered_runner = runner(recovered, "e26-recovered", now=recovery_now)
                async with asyncio.timeout(30):
                    assert await recovered_runner.run_once(asyncio.Event())
                    assert not await recovered_runner.run_once(asyncio.Event())
            async with sessions() as session:
                run = await session.get(Run, run_id)
                job = await session.scalar(select(RunJob).where(RunJob.run_id == run_id))
                action = await session.get(ActionIntent, request.action_intent_id)
                consumed = await session.get(ApprovalRequest, request.id)
                assert run.status == "completed" and job.status == "done"
                assert action.status == "succeeded" and consumed.status == "consumed"
                assert action.approval_binding_digest == binding
                assert action.idempotency_key == idempotency_key
                assert await session.scalar(select(func.count()).select_from(MockSubmission)) == 1
                assert await session.scalar(select(func.count()).select_from(ApprovalDecision)) == 1
                [invocation] = list(
                    await session.scalars(
                        select(ToolInvocation).where(ToolInvocation.effect == "irreversible")
                    )
                )
                assert invocation.status == "succeeded" and invocation.attempt == 1
                assert (
                    await session.scalar(select(func.count()).select_from(LLMInvocation))
                    == llm_before
                )
            assert len(posts) == 1
            assert "E26-private-resume-canary" not in caplog.text
            assert migrated_database_url not in caplog.text
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "worker_checkpoint_connection_lost" in output
    assert "E26-private-resume-canary" not in output
    assert migrated_database_url not in output
