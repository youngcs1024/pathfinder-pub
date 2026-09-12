from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from math import isfinite
from typing import Any, Final
from urllib.parse import parse_qsl

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Interrupt
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from pydantic import SecretStr
from sqlalchemy.engine import make_url

from app.db.runtime_policy import (
    DatabaseComponent,
    DatabaseSessionPolicy,
    checkpoint_setup_policy,
)

CHECKPOINT_SCHEMA: Final = "pathfinder_checkpoint"


def _require_json_value(value: object) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float and isfinite(value):
        return
    if isinstance(value, list | tuple):
        for item in value:
            _require_json_value(item)
        return
    if isinstance(value, dict) and all(type(key) is str for key in value):
        for item in value.values():
            _require_json_value(item)
        return
    if isinstance(value, Interrupt):
        if not isinstance(value.id, str) or not value.id:
            raise TypeError("checkpoint interrupt identity is invalid")
        _require_json_value(value.value)
        return
    raise TypeError("checkpoint serializer accepts JSON values only")


class StrictJsonCheckpointSerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        _require_json_value(obj)
        return super().dumps_typed(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        value = super().loads_typed(data)
        _require_json_value(value)
        return value


def _psycopg_connection_url(database_url: SecretStr) -> str:
    if not isinstance(database_url, SecretStr):
        raise TypeError("checkpoint database URL must be a SecretStr")
    raw_url = database_url.get_secret_value()
    query_parameters = parse_qsl(raw_url.partition("?")[2], keep_blank_values=True)
    if any(key == "options" for key, _ in query_parameters):
        raise ValueError("checkpoint URL options are not supported; use database policy settings")
    try:
        parsed = make_url(raw_url)
        if parsed.drivername != "postgresql+psycopg":
            raise ValueError
        return parsed.set(drivername="postgresql").render_as_string(hide_password=False)
    except Exception:
        raise ValueError("checkpoint database URL is invalid") from None


def create_checkpoint_serializer() -> StrictJsonCheckpointSerializer:
    return StrictJsonCheckpointSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=(Interrupt,),
    )


@asynccontextmanager
async def open_postgres_checkpointer(
    database_url: SecretStr,
    *,
    policy: DatabaseSessionPolicy | None = None,
) -> AsyncIterator[AsyncPostgresSaver]:
    runtime = policy or DatabaseSessionPolicy(component=DatabaseComponent.CHECKPOINT)
    if runtime.component != DatabaseComponent.CHECKPOINT:
        raise ValueError("checkpoint requires the fixed checkpoint component")
    setup = checkpoint_setup_policy(runtime)
    connection = await AsyncConnection.connect(
        _psycopg_connection_url(database_url),
        connect_timeout=setup.connect_timeout_seconds,
        application_name=setup.application_name,
        options=(
            f"-c statement_timeout={setup.statement_timeout_ms} "
            f"-c lock_timeout={setup.lock_timeout_ms} "
            f"-c idle_in_transaction_session_timeout={setup.idle_in_transaction_timeout_ms}"
        ),
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    try:
        await connection.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(CHECKPOINT_SCHEMA))
        )
        await connection.execute(
            sql.SQL("SET search_path TO {}").format(sql.Identifier(CHECKPOINT_SCHEMA))
        )
        current_schema = await connection.execute("SELECT current_schema()")
        row = await current_schema.fetchone()
        if row is None or row.get("current_schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("checkpoint schema search path could not be fixed")

        checkpointer = AsyncPostgresSaver(
            connection,
            serde=create_checkpoint_serializer(),
        )
        await checkpointer.setup()
        # Session-level settings survive autocommit without opening a transaction.
        await connection.execute(
            "SELECT set_config('statement_timeout', %s, false), "
            "set_config('lock_timeout', %s, false), "
            "set_config('idle_in_transaction_session_timeout', %s, false)",
            (
                str(runtime.statement_timeout_ms),
                str(runtime.lock_timeout_ms),
                str(runtime.idle_in_transaction_timeout_ms),
            ),
        )
        yield checkpointer
    finally:
        await connection.close()
