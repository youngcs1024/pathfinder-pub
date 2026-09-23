import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.config import Settings
from app.domain.provisioning import WorkspaceRole
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.worker import main
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings


@pytest.mark.parametrize("version", ["pathfinder-research-v6", "unknown"])
async def test_unregistered_graph_is_rejected_without_running_any_handler(version):
    result = await RunExecutorDispatcher({"pathfinder-resume-v2": object()}).execute(
        uuid4(), TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN), version
    )
    assert result.status is RunStatus.FAILED
    assert result.error_category == "unknown_graph_version" and not result.retryable
    with pytest.raises(ValueError):
        RunExecutorDispatcher({version: object()})


@pytest.mark.parametrize("pending", [True, False])
async def test_production_startup_is_read_only_and_only_clean_state_becomes_ready(
    monkeypatch, tmp_path, pending
):
    events = []

    class Engine:
        async def dispose(self):
            events.append("dispose")

    class Probe:
        def __init__(self, engine):
            pass

        async def is_ready(self):
            return True

        def mark_not_ready(self):
            events.append("not-ready")

    class Reader:
        def __init__(self, sessions):
            pass

        async def has_unsupported_pending_work(self):
            events.append("guard")
            return pending

    class Runner:
        def __init__(self, **kwargs):
            assert {item.graph_version for item in kwargs["store"]._execution_contracts} == {
                "pathfinder-resume-v2"
            }
            assert isinstance(kwargs["executor"], RunExecutorDispatcher)
            assert "approved_action_executor" not in kwargs
            self.guard = kwargs["unsupported_work_guard"]

        async def run(self, stop):
            assert (tmp_path / "ready").exists()
            assert not await self.guard()
            events.append("idle")

    monkeypatch.setattr(main, "WORKER_READY_PATH", tmp_path / "ready")
    monkeypatch.setattr(main, "create_database_engine", lambda *a, **kw: Engine())
    monkeypatch.setattr(main, "create_session_factory", lambda engine: object())
    monkeypatch.setattr(main, "DatabaseReadinessProbe", Probe)
    monkeypatch.setattr(main, "SqlAlchemyRunExecutionReader", Reader)
    monkeypatch.setattr(main, "WorkerRunner", Runner)
    if pending:
        with pytest.raises(RuntimeError, match="previous executor"):
            await main.run_worker(Settings(log_level="ERROR"))
        assert "idle" not in events
    else:
        await main.run_worker(Settings(log_level="ERROR"))
        assert "idle" in events
    assert events[-2:] == ["not-ready", "dispose"]
    assert not (tmp_path / "ready").exists()


async def test_work_appearing_after_startup_stops_before_reclaim_or_claim():
    async def guard():
        return True

    async def forbidden(**kwargs):
        pytest.fail("guard must precede all queue writes")

    runner = WorkerRunner(
        worker_id="synthetic",
        store=SimpleNamespace(reclaim_stale_leases=forbidden, claim_due_job=forbidden),
        tenant_service=object(),
        executor=RunExecutorDispatcher({"pathfinder-resume-v2": object()}),
        settings=WorkerRuntimeSettings(),
        unsupported_work_guard=guard,
    )
    with pytest.raises(RuntimeError, match="previous executor"):
        await runner.run_once(asyncio.Event())
