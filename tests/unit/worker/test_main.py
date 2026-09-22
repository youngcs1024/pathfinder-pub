from __future__ import annotations

import asyncio
import importlib
import json
import stat
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy, DatabaseSessionPolicy
from app.obs.logging import configure_logging
from app.worker.settings import WorkerRuntimeSettings
from tests.legacy_worker import (
    _clear_worker_ready_marker,
    _create_worker_ready_marker,
    _run_with_checkpoint_supervisor,
    _worker_readiness_marker,
)


def test_worker_ready_marker_can_replace_stale_file_with_private_permissions(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "worker-ready"
    marker.write_text("stale", encoding="utf-8")

    _clear_worker_ready_marker(marker)
    assert not marker.exists()

    _create_worker_ready_marker(marker)

    assert marker.is_file()
    assert marker.read_bytes() == b""
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_worker_ready_marker_is_removed_when_runner_scope_fails(tmp_path: Path) -> None:
    marker = tmp_path / "worker-ready"

    with pytest.raises(RuntimeError, match="runner failed"):
        with _worker_readiness_marker(marker):
            assert marker.is_file()
            raise RuntimeError("runner failed")

    assert not marker.exists()


def test_production_executor_wiring_uses_runtime_execution_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = importlib.import_module("tests.legacy_worker")
    captured: dict[str, object] = {}
    executor = object()

    def construct_executor(**kwargs: object) -> object:
        captured.update(kwargs)
        return executor

    monkeypatch.setattr(main_module, "LangGraphRunExecutor", construct_executor)
    dependencies = {
        "reader": object(),
        "checkpointer": object(),
        "llm_factory": object(),
        "search_port": object(),
        "tool_recorder": object(),
        "document_repository": object(),
        "retrieval_event_recorder": object(),
        "action_store": object(),
        "approval_resume_resolver": object(),
        "approved_action_executor": object(),
    }

    result = main_module._create_langgraph_executor(
        runtime_settings=WorkerRuntimeSettings(execution_timeout_seconds=75.0),
        **dependencies,
    )

    assert result is executor
    assert captured == {
        **dependencies,
        "execution_timeout_seconds": 75.0,
        "checkpoint_error_is_unavailable": main_module.is_checkpoint_unavailable,
    }


class _FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    async def dispose(self) -> None:
        self.disposed = True


class _UnavailableProbe:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.marked_not_ready = False

    async def is_ready(self) -> bool:
        return False

    def mark_not_ready(self) -> None:
        self.marked_not_ready = True


async def test_supabase_worker_fails_at_database_readiness_without_creating_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_module = importlib.import_module("tests.legacy_worker")
    marker = tmp_path / "worker-ready"
    marker.write_text("stale", encoding="utf-8")
    engine = _FakeEngine()
    probes: list[_UnavailableProbe] = []

    def create_probe(received_engine: _FakeEngine) -> _UnavailableProbe:
        assert received_engine is engine
        probe = _UnavailableProbe(received_engine)
        probes.append(probe)
        return probe

    monkeypatch.setattr(main_module, "WORKER_READY_PATH", marker)

    def create_engine(_url, *, policy):
        assert policy == DatabasePoolPolicy.from_settings(settings, DatabaseComponent.WORKER)
        return engine

    monkeypatch.setattr(main_module, "create_database_engine", create_engine)
    monkeypatch.setattr(main_module, "DatabaseReadinessProbe", create_probe)

    settings = Settings(
        auth_mode="supabase",
        supabase_project_ref="project-ref",
        supabase_publishable_key="sb_publishable_test-public-key",
    )
    with pytest.raises(
        RuntimeError,
        match="worker database is unavailable or not at the required revision",
    ):
        await main_module.run_worker(settings)

    assert not marker.exists()
    assert len(probes) == 1
    assert probes[0].marked_not_ready is True
    assert engine.disposed is True


class _GracefulRunner:
    def __init__(self) -> None:
        self.job_started = asyncio.Event()
        self.graceful_release_completed = False

    async def run(self, stop_requested: asyncio.Event) -> None:
        self.job_started.set()
        await stop_requested.wait()
        self.graceful_release_completed = True


@pytest.mark.parametrize("lost_attribute", ["closed", "broken"])
async def test_checkpoint_connection_loss_gracefully_stops_then_raises(
    lost_attribute: str,
) -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    connection = SimpleNamespace(
        closed=lost_attribute == "closed",
        broken=lost_attribute == "broken",
        dsn="postgresql://user:connection-string-canary@db/pathfinder",
    )
    runner = _GracefulRunner()

    with pytest.raises(RuntimeError, match="checkpoint connection was lost"):
        await _run_with_checkpoint_supervisor(
            runner=runner,  # type: ignore[arg-type]
            checkpointer=SimpleNamespace(conn=connection),
            worker_id="pathfinder-worker-test",
            heartbeat_seconds=0.01,
            stop_requested=asyncio.Event(),
        )

    assert runner.job_started.is_set()
    assert runner.graceful_release_completed is True
    event = json.loads(stream.getvalue())
    assert event["event"] == "worker_checkpoint_connection_lost"
    assert event["worker_id"] == "pathfinder-worker-test"
    assert "connection-string-canary" not in stream.getvalue()


@pytest.mark.parametrize(
    "checkpointer",
    [object(), SimpleNamespace(conn=object())],
    ids=["missing-saver-conn", "missing-connection-health-fields"],
)
async def test_checkpoint_supervisor_treats_missing_attributes_as_healthy(
    checkpointer: object,
) -> None:
    runner = _GracefulRunner()
    stop_requested = asyncio.Event()
    task = asyncio.create_task(
        _run_with_checkpoint_supervisor(
            runner=runner,  # type: ignore[arg-type]
            checkpointer=checkpointer,
            worker_id="pathfinder-worker-test",
            heartbeat_seconds=0.01,
            stop_requested=stop_requested,
        )
    )
    await asyncio.wait_for(runner.job_started.wait(), timeout=1)
    await asyncio.sleep(0.03)
    assert not task.done()

    stop_requested.set()
    await asyncio.wait_for(task, timeout=1)
    assert runner.graceful_release_completed is True


@pytest.mark.parametrize("startup_failure", [None, "setup", "switch"])
async def test_worker_wires_two_adapter_views_with_one_client_lifecycle(
    monkeypatch, tmp_path, startup_failure
):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, Mock

    from app.obs.langfuse import LangfuseTraceSink

    main_module = importlib.import_module("tests.legacy_worker")
    client = Mock()
    legacy_sink = LangfuseTraceSink(client)
    build = Mock(return_value=legacy_sink)
    engine = _FakeEngine()
    readiness = SimpleNamespace(is_ready=AsyncMock(return_value=True), mark_not_ready=Mock())
    reader = SimpleNamespace(has_nonterminal_other_graph_versions=AsyncMock(return_value=False))
    factory = Mock(return_value=object())
    runner = Mock(return_value=object())
    http_client = SimpleNamespace(aclose=AsyncMock())
    supervisor = AsyncMock()
    checkpoint_closed = Mock()
    marker = tmp_path / "worker-ready"
    marker_scope = Mock(wraps=_worker_readiness_marker)

    @asynccontextmanager
    async def checkpointer(_url, *, policy):
        assert policy == DatabaseSessionPolicy.from_settings(settings, DatabaseComponent.CHECKPOINT)
        try:
            if startup_failure is not None:
                raise RuntimeError(f"checkpoint {startup_failure} failed")
            yield object()
        finally:
            checkpoint_closed()

    def create_engine(_url, *, policy):
        assert policy == DatabasePoolPolicy.from_settings(settings, DatabaseComponent.WORKER)
        return engine

    monkeypatch.setattr(main_module, "create_database_engine", create_engine)
    monkeypatch.setattr(main_module, "DatabaseReadinessProbe", lambda _engine: readiness)
    monkeypatch.setattr(main_module, "WORKER_READY_PATH", marker)
    monkeypatch.setattr(main_module, "_worker_readiness_marker", marker_scope)
    monkeypatch.setattr(main_module, "build_trace_sink", build)
    monkeypatch.setattr(main_module, "SqlAlchemyRunExecutionReader", lambda _factory: reader)
    monkeypatch.setattr(main_module, "LLMFactory", factory)
    monkeypatch.setattr(main_module, "WorkerRunner", runner)
    monkeypatch.setattr(main_module, "open_postgres_checkpointer", checkpointer)
    monkeypatch.setattr(main_module, "_run_with_checkpoint_supervisor", supervisor)
    monkeypatch.setattr(main_module, "_create_langgraph_executor", Mock(return_value=object()))
    monkeypatch.setattr(main_module.httpx, "AsyncClient", Mock(return_value=http_client))
    for name in (
        "create_session_factory",
        "SqlAlchemyInvocationRecorder",
        "SqlAlchemyWorkerJobStore",
        "SqlAlchemyApprovalStore",
        "create_approved_action_registry",
        "MockPortalHTTPAdapter",
        "SqlAlchemyActionExecutionStore",
        "SqlAlchemyTenantResolver",
        "SqlAlchemyToolInvocationRecorder",
        "SqlAlchemyDocumentRepository",
        "SqlAlchemyRetrievalEventRecorder",
        "SqlAlchemyActionStore",
        "SqlAlchemyApprovalRequestExpirySweeper",
    ):
        monkeypatch.setattr(main_module, name, Mock(return_value=object()))
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", Mock())
    monkeypatch.setattr(loop, "remove_signal_handler", Mock())

    settings = Settings(trace_mode="off", db_pool_size=3, db_statement_timeout_ms=7000)
    if startup_failure is None:
        await main_module.run_worker(settings)
        assert factory.call_args.kwargs["trace_sink"] is legacy_sink
        assert runner.call_args.kwargs["trace_sink"] is legacy_sink.agent_sink
        supervisor.assert_awaited_once()
        marker_scope.assert_called_once_with(marker)
    else:
        with pytest.raises(RuntimeError, match=f"checkpoint {startup_failure} failed"):
            await main_module.run_worker(settings)
        runner.assert_not_called()
        supervisor.assert_not_awaited()
        marker_scope.assert_not_called()
    assert not marker.exists()
    readiness.mark_not_ready.assert_called_once()
    checkpoint_closed.assert_called_once()
    build.assert_called_once()
    client.shutdown.assert_called_once()
    http_client.aclose.assert_awaited_once()
    assert engine.disposed
