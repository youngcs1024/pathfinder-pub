"""Validated business and independent Checkpoint database policies."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from app.config import Settings


def _integer_input(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        try:
            return int(value)
        except ValueError:
            pass
    raise ValueError("database setting must be an integer")


def _number_input(value: object) -> object:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError("database pool timeout must be a finite number")
    return value


StatementTimeoutMs = Annotated[int, Field(ge=1, le=300_000), BeforeValidator(_integer_input)]
LockTimeoutMs = Annotated[int, Field(ge=1, le=30_000), BeforeValidator(_integer_input)]
IdleInTransactionTimeoutMs = Annotated[
    int, Field(ge=1, le=300_000), BeforeValidator(_integer_input)
]
ConnectTimeoutSeconds = Annotated[int, Field(ge=1, le=60), BeforeValidator(_integer_input)]
PoolTimeoutSeconds = Annotated[
    float, Field(gt=0, le=60, allow_inf_nan=False), BeforeValidator(_number_input)
]
PoolSize = Annotated[int, Field(ge=1, le=20), BeforeValidator(_integer_input)]
MaxOverflow = Annotated[int, Field(ge=0, le=20), BeforeValidator(_integer_input)]


def validate_timeout_order(statement_timeout_ms: int, lock_timeout_ms: int) -> None:
    if lock_timeout_ms >= statement_timeout_ms:
        raise ValueError("database lock timeout must be less than statement timeout")


class DatabaseComponent(StrEnum):
    API = "pathfinder-api"
    WORKER = "pathfinder-worker"
    INGEST = "pathfinder-ingest"
    CHECKPOINT = "pathfinder-checkpoint"
    WORKER_HEALTHCHECK = "pathfinder-worker-healthcheck"


class _PolicyValue(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
    )


class DatabaseSessionPolicy(_PolicyValue):
    component: DatabaseComponent
    statement_timeout_ms: StatementTimeoutMs = 5_000
    lock_timeout_ms: LockTimeoutMs = 1_000
    idle_in_transaction_timeout_ms: IdleInTransactionTimeoutMs = 10_000
    connect_timeout_seconds: ConnectTimeoutSeconds = 5

    @model_validator(mode="after")
    def validate_timeouts(self) -> Self:
        validate_timeout_order(self.statement_timeout_ms, self.lock_timeout_ms)
        return self

    @property
    def application_name(self) -> str:
        return self.component.value

    @classmethod
    def from_settings(
        cls, settings: Settings, component: DatabaseComponent
    ) -> DatabaseSessionPolicy:
        return cls(
            component=component,
            statement_timeout_ms=settings.db_statement_timeout_ms,
            lock_timeout_ms=settings.db_lock_timeout_ms,
            idle_in_transaction_timeout_ms=settings.db_idle_in_transaction_timeout_ms,
            connect_timeout_seconds=settings.db_connect_timeout_seconds,
        )


class DatabasePoolPolicy(_PolicyValue):
    session: DatabaseSessionPolicy
    pool_timeout_seconds: PoolTimeoutSeconds = 2.0
    pool_size: PoolSize = 5
    max_overflow: MaxOverflow = 0

    @property
    def pool_pre_ping(self) -> bool:
        return True

    @classmethod
    def from_settings(cls, settings: Settings, component: DatabaseComponent) -> DatabasePoolPolicy:
        if component == DatabaseComponent.CHECKPOINT:
            raise ValueError("checkpoint uses an independent session, not a business pool")
        return cls(
            session=DatabaseSessionPolicy.from_settings(settings, component),
            pool_timeout_seconds=settings.db_pool_timeout_seconds,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )

    @model_validator(mode="after")
    def reject_checkpoint_pool(self) -> Self:
        if self.session.component == DatabaseComponent.CHECKPOINT:
            raise ValueError("checkpoint uses an independent session, not a business pool")
        return self


def checkpoint_setup_policy(settings: Settings | DatabaseSessionPolicy) -> DatabaseSessionPolicy:
    """Finite setup DDL limits, independent of runtime overrides and pool capacity."""
    runtime = (
        settings
        if isinstance(settings, DatabaseSessionPolicy)
        else DatabaseSessionPolicy.from_settings(settings, DatabaseComponent.CHECKPOINT)
    )
    return DatabaseSessionPolicy(
        component=DatabaseComponent.CHECKPOINT,
        statement_timeout_ms=30_000,
        lock_timeout_ms=5_000,
        idle_in_transaction_timeout_ms=runtime.idle_in_transaction_timeout_ms,
        connect_timeout_seconds=runtime.connect_timeout_seconds,
    )
