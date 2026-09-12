from __future__ import annotations

import asyncio
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

import httpx

from app.config import Settings
from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.actions import SqlAlchemyActionStore
from app.db.approval_expiry import SqlAlchemyApprovalRequestExpirySweeper
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.checkpoints import open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.errors import is_checkpoint_unavailable
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.readiness import DatabaseReadinessProbe
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy, DatabaseSessionPolicy
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.llm.qwen_adapters import create_qwen_adapters
from app.obs.langfuse import agent_trace_sink, build_trace_sink
from app.obs.logging import configure_logging, get_logger
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.adapters.tavily import create_tavily_adapter
from app.tools.fake_search import FakeSearch
from app.tools.mock_application import create_approved_action_registry
from app.worker.backoff import ExponentialBackoff
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WORKER_READY_PATH, WorkerRuntimeSettings

_LOGGER = get_logger("app.worker.main")


def _clear_worker_ready_marker(path: Path) -> None:
    path.unlink(missing_ok=True)


def _create_worker_ready_marker(path: Path) -> None:
    path.touch(mode=0o600, exist_ok=False)
    try:
        path.chmod(0o600)
    except OSError:
        path.unlink(missing_ok=True)
        raise


@contextmanager
def _worker_readiness_marker(path: Path) -> Iterator[None]:
    _create_worker_ready_marker(path)
    try:
        yield
    finally:
        _clear_worker_ready_marker(path)


def _create_langgraph_executor(
    *,
    runtime_settings: WorkerRuntimeSettings,
    reader: Any,
    checkpointer: Any,
    llm_factory: LLMFactory,
    search_port: Any,
    tool_recorder: Any,
    document_repository: Any,
    retrieval_event_recorder: Any,
    action_store: Any,
    approval_resume_resolver: Any,
    approved_action_executor: Any,
) -> LangGraphRunExecutor:
    return LangGraphRunExecutor(
        reader=reader,
        checkpointer=checkpointer,
        checkpoint_error_is_unavailable=is_checkpoint_unavailable,
        llm_factory=llm_factory,
        search_port=search_port,
        tool_recorder=tool_recorder,
        document_repository=document_repository,
        retrieval_event_recorder=retrieval_event_recorder,
        action_store=action_store,
        approval_resume_resolver=approval_resume_resolver,
        approved_action_executor=approved_action_executor,
        execution_timeout_seconds=runtime_settings.execution_timeout_seconds,
    )


async def _supervise_checkpoint_connection(
    *,
    checkpointer: Any,
    worker_id: str,
    heartbeat_seconds: float,
    stop_requested: asyncio.Event,
    checkpoint_connection_lost: asyncio.Event,
) -> None:
    while not stop_requested.is_set():
        try:
            await asyncio.wait_for(stop_requested.wait(), timeout=heartbeat_seconds)
        except TimeoutError:
            pass
        if stop_requested.is_set():
            return
        connection = getattr(checkpointer, "conn", None)
        closed = getattr(connection, "closed", False)
        broken = getattr(connection, "broken", False)
        if closed is True or broken is True:
            _LOGGER.error("worker_checkpoint_connection_lost", worker_id=worker_id)
            checkpoint_connection_lost.set()
            stop_requested.set()
            return


async def _run_with_checkpoint_supervisor(
    *,
    runner: WorkerRunner,
    checkpointer: Any,
    worker_id: str,
    heartbeat_seconds: float,
    stop_requested: asyncio.Event,
) -> None:
    checkpoint_connection_lost = asyncio.Event()
    checkpoint_supervisor = asyncio.create_task(
        _supervise_checkpoint_connection(
            checkpointer=checkpointer,
            worker_id=worker_id,
            heartbeat_seconds=heartbeat_seconds,
            stop_requested=stop_requested,
            checkpoint_connection_lost=checkpoint_connection_lost,
        )
    )
    try:
        await runner.run(stop_requested)
    finally:
        if not checkpoint_supervisor.done():
            checkpoint_supervisor.cancel()
        await asyncio.gather(checkpoint_supervisor, return_exceptions=True)
    if checkpoint_connection_lost.is_set():
        raise RuntimeError("worker checkpoint connection was lost")


async def run_worker(settings: Settings | None = None) -> None:
    resolved_settings = settings or Settings()
    configure_logging(log_level=resolved_settings.log_level)
    _clear_worker_ready_marker(WORKER_READY_PATH)
    engine = create_database_engine(
        resolved_settings.database_url,
        policy=DatabasePoolPolicy.from_settings(resolved_settings, DatabaseComponent.WORKER),
    )
    readiness = DatabaseReadinessProbe(engine)
    qwen_bundle: Any | None = None
    tavily_bundle: Any | None = None
    trace_sink: Any | None = None
    mock_portal_client: httpx.AsyncClient | None = None
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    try:
        if not await readiness.is_ready():
            raise RuntimeError("worker database is unavailable or not at the required revision")
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(handled_signal, stop_requested.set)
            installed_signals.append(handled_signal)

        session_factory = create_session_factory(engine)
        run_reader = SqlAlchemyRunExecutionReader(session_factory)
        if await run_reader.has_nonterminal_other_graph_versions(CURRENT_GRAPH_VERSION):
            raise RuntimeError("nonterminal runs use an unsupported graph version")
        trace_sink = build_trace_sink(resolved_settings)
        if resolved_settings.llm_mode == "qwen":
            assert resolved_settings.qwen_api_key is not None
            assert resolved_settings.qwen_workspace_id is not None
            qwen_bundle = create_qwen_adapters(
                api_key=resolved_settings.qwen_api_key,
                workspace_id=resolved_settings.qwen_workspace_id,
            )
            chat_adapter = qwen_bundle.chat
            embedding_adapter = qwen_bundle.embedding
            provider = "qwen"
        else:
            chat_adapter = DeterministicResearchFakeChatAdapter()
            embedding_adapter = FakeEmbeddingModel()
            provider = "fake"

        if resolved_settings.search_mode == "tavily":
            assert resolved_settings.tavily_api_key is not None
            tavily_bundle = create_tavily_adapter(api_key=resolved_settings.tavily_api_key)
            search_port = tavily_bundle.search
        else:
            search_port = FakeSearch({})

        llm_factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(session_factory),
            chat_adapter=chat_adapter,
            embedding_adapter=embedding_adapter,
            trace_sink=trace_sink,
            provider=provider,
        )
        runtime_settings = WorkerRuntimeSettings()
        retry_delay = ExponentialBackoff(runtime_settings, Random())
        store = SqlAlchemyWorkerJobStore(
            session_factory,
            retry_delay,
            action_recovery_max_attempts=runtime_settings.action_recovery_max_attempts,
        )
        approval_store = SqlAlchemyApprovalStore(session_factory)
        mock_portal_client = httpx.AsyncClient(
            base_url=resolved_settings.mock_portal_base_url,
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
        )
        approved_action_registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(mock_portal_client),
            action_execution_store=SqlAlchemyActionExecutionStore(session_factory),
            action_recovery_max_attempts=runtime_settings.action_recovery_max_attempts,
        )
        async with open_postgres_checkpointer(
            resolved_settings.database_url,
            policy=DatabaseSessionPolicy.from_settings(
                resolved_settings, DatabaseComponent.CHECKPOINT
            ),
        ) as checkpointer:
            worker_id = f"pathfinder-worker-{uuid4()}"
            runner = WorkerRunner(
                worker_id=worker_id,
                store=store,
                tenant_service=TenantService(SqlAlchemyTenantResolver(session_factory)),
                executor=_create_langgraph_executor(
                    runtime_settings=runtime_settings,
                    reader=run_reader,
                    checkpointer=checkpointer,
                    llm_factory=llm_factory,
                    search_port=search_port,
                    tool_recorder=SqlAlchemyToolInvocationRecorder(session_factory),
                    document_repository=SqlAlchemyDocumentRepository(session_factory),
                    retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(session_factory),
                    action_store=SqlAlchemyActionStore(session_factory),
                    approval_resume_resolver=approval_store,
                    approved_action_executor=approved_action_registry,
                ),
                settings=runtime_settings,
                trace_sink=agent_trace_sink(trace_sink),
                approval_expiry_sweeper=SqlAlchemyApprovalRequestExpirySweeper(session_factory),
                approved_action_executor=approved_action_registry,
            )
            with _worker_readiness_marker(WORKER_READY_PATH):
                await _run_with_checkpoint_supervisor(
                    runner=runner,
                    checkpointer=checkpointer,
                    worker_id=worker_id,
                    heartbeat_seconds=runtime_settings.heartbeat_seconds,
                    stop_requested=stop_requested,
                )
    finally:
        readiness.mark_not_ready()
        for handled_signal in installed_signals:
            loop.remove_signal_handler(handled_signal)
        try:
            if qwen_bundle is not None:
                await qwen_bundle.aclose()
        finally:
            try:
                if tavily_bundle is not None:
                    await tavily_bundle.aclose()
            finally:
                try:
                    if trace_sink is not None:
                        shutdown = getattr(trace_sink, "shutdown", None)
                        if callable(shutdown):
                            shutdown()
                finally:
                    try:
                        if mock_portal_client is not None:
                            await mock_portal_client.aclose()
                    finally:
                        await engine.dispose()


def run() -> None:
    try:
        asyncio.run(run_worker())
    except Exception:
        raise SystemExit("pathfinder worker stopped before a safe running state") from None


if __name__ == "__main__":
    run()
