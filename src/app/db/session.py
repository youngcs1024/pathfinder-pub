import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl

from pydantic import SecretStr
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.errors import classify_database_failure
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy, DatabaseSessionPolicy
from app.domain.errors import DomainUnavailableError

AsyncSessionFactory = async_sessionmaker[AsyncSession]


def create_database_engine(
    database_url: SecretStr, *, policy: DatabasePoolPolicy | None = None
) -> AsyncEngine:
    resolved = policy or DatabasePoolPolicy(
        session=DatabaseSessionPolicy(component=DatabaseComponent.API)
    )
    raw_url = database_url.get_secret_value()
    # SQLAlchemy omits empty query values, so inspect those before URL parsing.
    query_parameters = parse_qsl(raw_url.partition("?")[2], keep_blank_values=True)
    if any(key == "options" for key, _ in query_parameters):
        raise ValueError("database URL options are not supported; use database policy settings")
    try:
        url = make_url(raw_url)
    except (ArgumentError, ValueError):
        raise ValueError("database URL is invalid") from None
    session = resolved.session
    return create_async_engine(
        url,
        pool_pre_ping=resolved.pool_pre_ping,
        pool_size=resolved.pool_size,
        max_overflow=resolved.max_overflow,
        pool_timeout=resolved.pool_timeout_seconds,
        hide_parameters=True,
        echo=False,
        connect_args={
            "connect_timeout": session.connect_timeout_seconds,
            "application_name": session.application_name,
            "options": (
                f"-c statement_timeout={session.statement_timeout_ms} "
                f"-c lock_timeout={session.lock_timeout_ms} "
                f"-c idle_in_transaction_session_timeout={session.idle_in_transaction_timeout_ms}"
            ),
        },
    )


def create_session_factory(engine: AsyncEngine) -> AsyncSessionFactory:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )


@asynccontextmanager
async def database_session(
    session_factory: AsyncSessionFactory,
) -> AsyncIterator[AsyncSession]:
    cancellation = None
    try:
        async with session_factory() as session:
            try:
                yield session
            except asyncio.CancelledError as error:
                cancellation = error
                raise
    except Exception as error:
        if cancellation is not None:
            raise cancellation from None
        if classify_database_failure(error) is not None:
            raise DomainUnavailableError from None
        raise


@asynccontextmanager
async def transaction(
    session_factory: AsyncSessionFactory,
) -> AsyncIterator[AsyncSession]:
    commit_started = False
    cancellation = None
    try:
        async with session_factory() as session:
            async with session.begin():
                try:
                    yield session
                except asyncio.CancelledError as error:
                    cancellation = error
                    raise
                commit_started = True
    except Exception as error:
        if cancellation is not None:
            raise cancellation from None
        # The transaction and session have exited before callers can handle failures.
        # A failed COMMIT acknowledgement cannot prove that the write was rolled back.
        if classify_database_failure(error) is not None:
            raise DomainUnavailableError(commit_outcome_unknown=commit_started) from None
        raise
