from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langgraph.types import Interrupt
from pydantic import SecretStr

from app.db.checkpoints import (
    _psycopg_connection_url,
    create_checkpoint_serializer,
    open_postgres_checkpointer,
)
from app.db.runtime_policy import DatabaseComponent, DatabaseSessionPolicy


@dataclass(frozen=True)
class _ForbiddenCheckpointValue:
    value: str


def test_checkpoint_dsn_conversion_and_serializer_are_strict() -> None:
    converted = _psycopg_connection_url(
        SecretStr("postgresql+psycopg://worker:secret@db.test:5432/pathfinder")
    )
    assert converted == "postgresql://worker:secret@db.test:5432/pathfinder"

    serializer = create_checkpoint_serializer()
    encoded = serializer.dumps_typed({"schema_version": 1, "state": ["safe", 1, True, None]})
    assert serializer.loads_typed(encoded) == {
        "schema_version": 1,
        "state": ["safe", 1, True, None],
    }
    interrupt = Interrupt(value={"version": 1, "approval_request_id": "safe-id"}, id="i-1")
    assert serializer.loads_typed(serializer.dumps_typed((interrupt,))) == [interrupt]
    with pytest.raises(TypeError, match="JSON values only"):
        serializer.dumps_typed(_ForbiddenCheckpointValue("secret-canary"))


@pytest.mark.parametrize(
    "url",
    (
        "postgresql://worker:secret@db.test:5432/pathfinder",
        "sqlite:///pathfinder.db",
    ),
)
def test_checkpoint_dsn_rejects_non_authoritative_drivers(url: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        _psycopg_connection_url(SecretStr(url))


@pytest.mark.parametrize(
    "query",
    ["options=", "options", "options=&options=secret-canary", "%6fptions=secret-canary"],
)
async def test_checkpoint_rejects_dsn_options_before_connect(monkeypatch, query) -> None:
    connect = AsyncMock()
    monkeypatch.setattr("app.db.checkpoints.AsyncConnection.connect", connect)
    with pytest.raises(ValueError, match="options are not supported") as caught:
        async with open_postgres_checkpointer(
            SecretStr(f"postgresql+psycopg://user:secret-canary@db/test?{query}")
        ):
            pytest.fail("invalid DSN must not yield")
    assert "secret-canary" not in str(caught.value)
    connect.assert_not_awaited()


@pytest.mark.parametrize("custom", [False, True])
async def test_checkpoint_applies_setup_then_runtime_before_yield(monkeypatch, custom) -> None:
    from psycopg.rows import dict_row

    from app.db.checkpoints import StrictJsonCheckpointSerializer

    cursor = SimpleNamespace(
        fetchone=AsyncMock(return_value={"current_schema": "pathfinder_checkpoint"})
    )
    connection = SimpleNamespace(execute=AsyncMock(return_value=cursor), close=AsyncMock())
    connect = AsyncMock(return_value=connection)
    runtime = (
        DatabaseSessionPolicy(
            component=DatabaseComponent.CHECKPOINT,
            statement_timeout_ms=9000,
            lock_timeout_ms=2000,
            idle_in_transaction_timeout_ms=12000,
            connect_timeout_seconds=7,
        )
        if custom
        else DatabaseSessionPolicy(component=DatabaseComponent.CHECKPOINT)
    )

    async def setup() -> None:
        assert connection.execute.await_count == 3

    saver = SimpleNamespace(setup=AsyncMock(side_effect=setup))
    construct = Mock(return_value=saver)
    monkeypatch.setattr("app.db.checkpoints.AsyncConnection.connect", connect)
    monkeypatch.setattr("app.db.checkpoints.AsyncPostgresSaver", construct)
    kwargs = {"policy": runtime} if custom else {}
    async with open_postgres_checkpointer(
        SecretStr("postgresql+psycopg://user:secret@db/test?sslmode=require"), **kwargs
    ) as result:
        assert result is saver
        saver.setup.assert_awaited_once()
        assert connection.execute.await_count == 4
        switch = connection.execute.call_args
        assert "false" in switch.args[0]
        assert switch.args[1] == (
            str(runtime.statement_timeout_ms),
            str(runtime.lock_timeout_ms),
            str(runtime.idle_in_transaction_timeout_ms),
        )
        connection.close.assert_not_awaited()
    connection.close.assert_awaited_once()
    assert connect.call_args.args == ("postgresql://user:secret@db/test?sslmode=require",)
    assert connect.call_args.kwargs == {
        "connect_timeout": runtime.connect_timeout_seconds,
        "application_name": "pathfinder-checkpoint",
        "options": (
            "-c statement_timeout=30000 -c lock_timeout=5000 "
            f"-c idle_in_transaction_session_timeout={runtime.idle_in_transaction_timeout_ms}"
        ),
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
    assert isinstance(construct.call_args.kwargs["serde"], StrictJsonCheckpointSerializer)


@pytest.mark.parametrize("stage", ["schema", "search_path", "verify", "setup", "switch", "body"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_checkpoint_failure_closes_connection_without_yield(monkeypatch, stage, cancel):
    failure = asyncio.CancelledError() if cancel else RuntimeError("controlled failure")
    cursor = SimpleNamespace(
        fetchone=AsyncMock(return_value={"current_schema": "pathfinder_checkpoint"})
    )
    connection = SimpleNamespace(close=AsyncMock())
    stages = iter(["schema", "search_path", "verify", "switch"])

    async def execute(*_args):
        if next(stages) == stage:
            raise failure
        return cursor

    connection.execute = AsyncMock(side_effect=execute)
    saver = SimpleNamespace(setup=AsyncMock(side_effect=failure if stage == "setup" else None))
    monkeypatch.setattr(
        "app.db.checkpoints.AsyncConnection.connect", AsyncMock(return_value=connection)
    )
    monkeypatch.setattr("app.db.checkpoints.AsyncPostgresSaver", Mock(return_value=saver))
    yielded = False
    with pytest.raises(type(failure)) as caught:
        async with open_postgres_checkpointer(SecretStr("postgresql+psycopg://db/test")):
            yielded = True
            raise failure
    assert caught.value is failure
    assert yielded is (stage == "body")
    connection.close.assert_awaited_once()


async def test_checkpoint_rejects_wrong_component_before_connect(monkeypatch):
    connect = AsyncMock()
    monkeypatch.setattr("app.db.checkpoints.AsyncConnection.connect", connect)
    with pytest.raises(ValueError, match="fixed checkpoint component"):
        async with open_postgres_checkpointer(
            SecretStr("postgresql+psycopg://db/test"),
            policy=DatabaseSessionPolicy(component=DatabaseComponent.API),
        ):
            pytest.fail("wrong component must not yield")
    connect.assert_not_awaited()


@pytest.mark.parametrize("row", [None, {"current_schema": "public"}])
async def test_checkpoint_schema_mismatch_closes_before_setup(monkeypatch, row):
    cursor = SimpleNamespace(fetchone=AsyncMock(return_value=row))
    connection = SimpleNamespace(execute=AsyncMock(return_value=cursor), close=AsyncMock())
    construct = Mock()
    monkeypatch.setattr(
        "app.db.checkpoints.AsyncConnection.connect", AsyncMock(return_value=connection)
    )
    monkeypatch.setattr("app.db.checkpoints.AsyncPostgresSaver", construct)
    with pytest.raises(RuntimeError, match="schema search path"):
        async with open_postgres_checkpointer(SecretStr("postgresql+psycopg://db/test")):
            pytest.fail("schema mismatch must not yield")
    construct.assert_not_called()
    connection.close.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
async def test_checkpoint_connect_failure_propagates_before_setup(monkeypatch, cancel):
    failure = asyncio.CancelledError() if cancel else RuntimeError("connect failed")
    construct = Mock()
    monkeypatch.setattr(
        "app.db.checkpoints.AsyncConnection.connect", AsyncMock(side_effect=failure)
    )
    monkeypatch.setattr("app.db.checkpoints.AsyncPostgresSaver", construct)
    with pytest.raises(type(failure)) as caught:
        async with open_postgres_checkpointer(SecretStr("postgresql+psycopg://db/test")):
            pytest.fail("failed connection must not yield")
    assert caught.value is failure
    construct.assert_not_called()
