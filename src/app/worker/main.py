from __future__ import annotations

import asyncio
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from random import Random
from uuid import uuid4

from app.config import Settings
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.readiness import DatabaseReadinessProbe
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.llm.qwen_adapters import create_qwen_adapters
from app.material.aliases import load_aliases
from app.obs.logging import configure_logging
from app.retrieval.documents import DocumentIngestionService
from app.worker.backoff import ExponentialBackoff
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.material_executor import MaterialRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WORKER_READY_PATH, WorkerRuntimeSettings


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


async def run_worker(settings: Settings | None = None) -> None:
    resolved_settings = settings or Settings()
    configure_logging(log_level=resolved_settings.log_level)
    _clear_worker_ready_marker(WORKER_READY_PATH)
    engine = create_database_engine(
        resolved_settings.database_url,
        policy=DatabasePoolPolicy.from_settings(resolved_settings, DatabaseComponent.WORKER),
    )
    readiness = DatabaseReadinessProbe(engine)
    stop_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    qwen_bundle = None
    try:
        if not await readiness.is_ready():
            raise RuntimeError("worker database is unavailable or not at the required revision")
        sessions = create_session_factory(engine)
        reader = SqlAlchemyRunExecutionReader(sessions)
        aliases = load_aliases(resolved_settings.material_aliases_file)
        materials = SqlAlchemyMaterialStore(sessions, aliases)
        if resolved_settings.llm_mode == "qwen":
            if (
                resolved_settings.qwen_api_key is None
                or resolved_settings.qwen_workspace_id is None
            ):
                raise RuntimeError("material embedding credentials are unavailable")
            qwen_bundle = create_qwen_adapters(
                api_key=resolved_settings.qwen_api_key,
                workspace_id=resolved_settings.qwen_workspace_id,
            )
            chat_adapter, embedding_adapter, provider = (
                qwen_bundle.chat,
                qwen_bundle.embedding,
                "qwen",
            )
        else:
            chat_adapter, embedding_adapter, provider = (
                FakeChatModel(),
                FakeEmbeddingModel(),
                "fake",
            )
        llm = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=chat_adapter,
            embedding_adapter=embedding_adapter,
            provider=provider,
        )
        document_repository = SqlAlchemyDocumentRepository(sessions)
        material_executor = MaterialRunExecutor(
            reader=reader,
            materials=materials,
            aliases=aliases,
            ingestion_factory=lambda tenant, run_id: DocumentIngestionService(
                repository=document_repository,
                embedding=llm.create_embedding_model(
                    LLMInvocationContext(
                        workspace_id=tenant.workspace_id,
                        actor_user_id=tenant.actor_user_id,
                        run_id=run_id,
                    )
                ),
            ),
        )
        if await reader.has_unsupported_pending_work():
            raise RuntimeError("unsupported pending work requires the previous executor")
        for handled_signal in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(handled_signal, stop_requested.set)
            installed_signals.append(handled_signal)
        runtime = WorkerRuntimeSettings()
        runner = WorkerRunner(
            worker_id=f"pathfinder-worker-{uuid4()}",
            store=SqlAlchemyWorkerJobStore(sessions, ExponentialBackoff(runtime, Random())),
            tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
            executor=RunExecutorDispatcher({"pathfinder-resume-v1": material_executor}),
            settings=runtime,
            unsupported_work_guard=reader.has_unsupported_pending_work,
        )
        with _worker_readiness_marker(WORKER_READY_PATH):
            await runner.run(stop_requested)
    finally:
        if qwen_bundle is not None:
            await qwen_bundle.aclose()
        readiness.mark_not_ready()
        for handled_signal in installed_signals:
            loop.remove_signal_handler(handled_signal)
        await engine.dispose()


def run() -> None:
    try:
        asyncio.run(run_worker())
    except Exception:
        raise SystemExit("pathfinder worker stopped before a safe running state") from None


if __name__ == "__main__":
    run()
