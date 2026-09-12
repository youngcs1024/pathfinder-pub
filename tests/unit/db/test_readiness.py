from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

import app.db.readiness as readiness_module
from app.db.readiness import REQUIRED_DATABASE_REVISION, DatabaseReadinessProbe

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _Result:
    def __init__(self, revision: str | None, embedding_type: str | None) -> None:
        self._revision = revision
        self._embedding_type = embedding_type

    def one(self) -> SimpleNamespace:
        return SimpleNamespace(
            revision=self._revision,
            embedding_type=self._embedding_type,
        )


class _Connection:
    def __init__(
        self,
        *,
        revision: str | None = REQUIRED_DATABASE_REVISION,
        embedding_type: str | None = "vector(1536)",
        error: Exception | None = None,
        delay: float = 0,
    ) -> None:
        self._revision = revision
        self._embedding_type = embedding_type
        self._error = error
        self._delay = delay

    async def execute(self, _statement: object) -> _Result:
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return _Result(self._revision, self._embedding_type)


class _ConnectionContext:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self.closed = False

    async def __aenter__(self) -> _Connection:
        return self._connection

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class _Engine:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self.connect_calls = 0
        self.context = _ConnectionContext(connection)

    def connect(self) -> _ConnectionContext:
        self.connect_calls += 1
        return self.context


def _probe(connection: _Connection) -> tuple[DatabaseReadinessProbe, _Engine]:
    engine = _Engine(connection)
    probe = DatabaseReadinessProbe(cast(AsyncEngine, cast(Any, engine)))
    return probe, engine


async def test_probe_requires_exact_migration_revision() -> None:
    ready_probe, _ = _probe(_Connection())
    stale_probe, _ = _probe(_Connection(revision="stale_revision"))
    missing_probe, _ = _probe(_Connection(revision=None))

    assert await ready_probe.is_ready() is True
    assert await stale_probe.is_ready() is False
    assert await missing_probe.is_ready() is False


async def test_probe_requires_exact_embedding_dimension() -> None:
    wrong_probe, _ = _probe(_Connection(embedding_type="vector(3)"))
    missing_probe, _ = _probe(_Connection(embedding_type=None))

    assert await wrong_probe.is_ready() is False
    assert await missing_probe.is_ready() is False


async def test_probe_returns_false_for_sqlalchemy_errors_without_leaking_dsn() -> None:
    canary = "postgresql+psycopg://PATHFINDER-DSN-CANARY@database/pathfinder"
    probe, _ = _probe(_Connection(error=SQLAlchemyError(canary)))

    assert await probe.is_ready() is False


async def test_probe_times_out_at_the_fixed_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    assert readiness_module._READINESS_TIMEOUT_SECONDS == 1.0
    monkeypatch.setattr(readiness_module, "_READINESS_TIMEOUT_SECONDS", 0.001)
    probe, engine = _probe(_Connection(delay=1))

    assert await probe.is_ready() is False
    assert engine.context.closed is True


async def test_probe_propagates_unknown_programming_errors() -> None:
    probe, _ = _probe(_Connection(error=RuntimeError("programming error")))

    with pytest.raises(RuntimeError, match="programming error"):
        await probe.is_ready()


async def test_mark_not_ready_prevents_database_access() -> None:
    probe, engine = _probe(_Connection())

    probe.mark_not_ready()

    assert await probe.is_ready() is False
    assert engine.connect_calls == 0


def test_required_revision_matches_alembic_head() -> None:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == REQUIRED_DATABASE_REVISION


@pytest.mark.parametrize("phase", ["connect", "execute"])
async def test_readiness_one_second_deadline_covers_connection_and_query(phase: str) -> None:
    entered = asyncio.Event()
    released = asyncio.Event()

    class BlockingConnection(_Connection):
        async def execute(self, statement: object) -> _Result:
            if phase == "execute":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    released.set()
            return await super().execute(statement)

    class BlockingContext(_ConnectionContext):
        async def __aenter__(self) -> _Connection:
            if phase == "connect":
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    released.set()
            return await super().__aenter__()

    probe, engine = _probe(BlockingConnection())
    engine.context = BlockingContext(engine._connection)
    assert readiness_module._READINESS_TIMEOUT_SECONDS == 1.0
    # A five-second SQL policy cannot extend the whole probe to five seconds.
    async with asyncio.timeout(3):
        assert await probe.is_ready() is False
    assert entered.is_set()
    assert released.is_set()
    assert engine.context.closed is (phase == "execute")


async def test_readiness_external_cancellation_propagates_and_releases_connection() -> None:
    entered = asyncio.Event()

    class BlockingConnection(_Connection):
        async def execute(self, _statement: object) -> _Result:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    probe, engine = _probe(BlockingConnection())
    task = asyncio.create_task(probe.is_ready())
    try:
        async with asyncio.timeout(3):
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert engine.context.closed is True
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
