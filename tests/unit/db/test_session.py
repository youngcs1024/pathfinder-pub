from __future__ import annotations

from types import TracebackType
from typing import Any, cast

import pytest

from app.db.session import AsyncSessionFactory, transaction


class _TransactionContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self) -> None:
        self._events.append("begin")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._events.append("rollback" if exc_type is not None else "commit")


class _SessionContext:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def __aenter__(self) -> _SessionContext:
        self._events.append("open")
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._events.append("close")

    def begin(self) -> _TransactionContext:
        return _TransactionContext(self._events)


class _SessionFactory:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __call__(self) -> _SessionContext:
        return _SessionContext(self._events)


async def test_transaction_commits_and_closes_on_success() -> None:
    events: list[str] = []
    factory = cast(AsyncSessionFactory, cast(Any, _SessionFactory(events)))

    async with transaction(factory):
        events.append("work")

    assert events == ["open", "begin", "work", "commit", "close"]


async def test_transaction_rolls_back_closes_and_propagates_failure() -> None:
    events: list[str] = []
    factory = cast(AsyncSessionFactory, cast(Any, _SessionFactory(events)))

    with pytest.raises(RuntimeError, match="injected failure"):
        async with transaction(factory):
            events.append("work")
            raise RuntimeError("injected failure")

    assert events == ["open", "begin", "work", "rollback", "close"]


@pytest.mark.parametrize(
    "query",
    ["options=", "options", "options=&options=-c+statement_timeout%3D0", "%6fptions=canary"],
)
def test_engine_rejects_dsn_options_without_exposing_input(query: str) -> None:
    import traceback

    from pydantic import SecretStr

    from app.db.session import create_database_engine

    with pytest.raises(ValueError, match="database URL options") as raised:
        create_database_engine(SecretStr(f"postgresql+psycopg://user:canary@db/test?{query}"))

    assert "canary" not in str(raised.value)
    assert "canary" not in "".join(traceback.format_exception_only(raised.value))
    assert raised.value.__context__ is None


async def test_engine_default_policy_is_bounded_async_and_does_not_load_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import SecretStr
    from sqlalchemy.pool import AsyncAdaptedQueuePool

    from app.db.session import create_database_engine

    monkeypatch.setenv("PF_DB_POOL_SIZE", "19")
    engine = create_database_engine(SecretStr("postgresql+psycopg://user:canary@db/test"))
    try:
        assert isinstance(engine.pool, AsyncAdaptedQueuePool)
        assert engine.pool.size() == 5
        assert engine.pool.timeout() == 2
        assert engine.pool._max_overflow == 0
        assert engine.pool._pre_ping is True
        assert engine.sync_engine.hide_parameters is True
        assert engine.echo is False
    finally:
        await engine.dispose()


async def test_driver_receives_policy_over_dsn_connection_parameters() -> None:
    from pydantic import SecretStr
    from sqlalchemy import event

    from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy, DatabaseSessionPolicy
    from app.db.session import create_database_engine

    policy = DatabasePoolPolicy(
        session=DatabaseSessionPolicy(
            component=DatabaseComponent.INGEST,
            statement_timeout_ms=8000,
            lock_timeout_ms=2000,
            idle_in_transaction_timeout_ms=12000,
            connect_timeout_seconds=7,
        ),
        pool_size=2,
        max_overflow=1,
        pool_timeout_seconds=0.5,
    )
    engine = create_database_engine(
        SecretStr(
            "postgresql+psycopg://user:canary@db/test"
            "?connect_timeout=0&application_name=untrusted&sslmode=require"
        ),
        policy=policy,
    )
    captured: dict[str, object] = {}

    class StopBeforeNetwork(Exception):
        pass

    @event.listens_for(engine.sync_engine, "do_connect")
    def capture(_dialect, _record, _args, params):
        captured.update(params)
        raise StopBeforeNetwork

    try:
        with pytest.raises(StopBeforeNetwork):
            async with engine.connect():
                pytest.fail("must stop before network")
        assert captured["options"] == (
            "-c statement_timeout=8000 -c lock_timeout=2000 "
            "-c idle_in_transaction_session_timeout=12000"
        )
        assert captured["connect_timeout"] == 7
        assert captured["application_name"] == "pathfinder-ingest"
        assert captured["sslmode"] == "require"
        assert engine.pool.size() == 2
        assert engine.pool._max_overflow == 1
        assert engine.pool.timeout() == 0.5
    finally:
        await engine.dispose()


@pytest.mark.parametrize("read_only", [False, True])
async def test_database_failure_is_translated_after_session_cleanup(read_only):
    import traceback

    from psycopg.errors import QueryCanceled

    from app.db.session import database_session
    from app.domain.errors import DomainUnavailableError

    events = []
    factory = _SessionFactory(events)
    boundary = database_session if read_only else transaction
    with pytest.raises(DomainUnavailableError) as raised:
        async with boundary(factory):
            raise QueryCanceled("SQL-PARAM-SECRET-CANARY")
    assert events == (["open", "close"] if read_only else ["open", "begin", "rollback", "close"])
    assert not raised.value.commit_outcome_unknown
    assert "SQL-PARAM-SECRET-CANARY" not in "".join(traceback.format_exception(raised.value))


async def test_commit_disconnect_does_not_claim_write_was_rolled_back():
    from psycopg import OperationalError

    from app.domain.errors import DomainUnavailableError

    events = []

    class CommitDisconnect(_TransactionContext):
        async def __aexit__(self, *args):
            events.append("commit_may_have_succeeded")
            raise OperationalError("COMMIT-CANARY")

    class Session(_SessionContext):
        def begin(self):
            return CommitDisconnect(events)

    with pytest.raises(DomainUnavailableError) as raised:
        async with transaction(lambda: Session(events)):
            events.append("write")
    assert raised.value.commit_outcome_unknown
    assert events == ["open", "begin", "write", "commit_may_have_succeeded", "close"]


@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_cancellation_survives_cleanup_failure(read_only, cleanup_fails):
    import asyncio

    from psycopg import OperationalError

    from app.db.session import database_session

    events = []
    cancellation = asyncio.CancelledError()

    class Session(_SessionContext):
        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            if cleanup_fails:
                raise OperationalError("CLEANUP-CANARY")

    boundary = database_session if read_only else transaction
    with pytest.raises(asyncio.CancelledError) as raised:
        async with boundary(lambda: Session(events)):
            raise cancellation
    assert raised.value is cancellation
    assert events[-1] == "close"


async def test_unique_conflict_is_propagated_after_rollback():
    from psycopg.errors import UniqueViolation
    from sqlalchemy.exc import IntegrityError

    events = []
    conflict = IntegrityError(None, None, UniqueViolation())
    with pytest.raises(IntegrityError) as raised:
        async with transaction(_SessionFactory(events)):
            raise conflict
    assert raised.value is conflict
    assert events == ["open", "begin", "rollback", "close"]
