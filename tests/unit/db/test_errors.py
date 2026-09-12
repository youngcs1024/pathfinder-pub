import asyncio

import psycopg
import pytest
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.db.errors import classify_database_failure


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (psycopg.errors.QueryCanceled("BODY-CANARY"), "query_canceled"),
        (psycopg.errors.LockNotAvailable("BODY-CANARY"), "lock_unavailable"),
        (psycopg.errors.ConnectionFailure("BODY-CANARY"), "connection_unavailable"),
        (psycopg.errors.AdminShutdown("BODY-CANARY"), "connection_unavailable"),
        (psycopg.errors.IdleInTransactionSessionTimeout(), "connection_unavailable"),
        (psycopg.errors.TooManyConnections(), "connection_unavailable"),
        (psycopg.OperationalError("BODY-CANARY"), "connection_unavailable"),
        (psycopg.errors.UniqueViolation("BODY-CANARY"), None),
        (psycopg.errors.ForeignKeyViolation(), None),
        (psycopg.errors.SyntaxError(), None),
        (psycopg.errors.InvalidPassword(), None),
        (psycopg.errors.InFailedSqlTransaction(), None),
        (psycopg.errors.SerializationFailure(), None),
        (psycopg.InterfaceError(), None),
        (psycopg.errors.PipelineAborted(), None),
        (psycopg.errors.CancellationTimeout(), None),
    ],
)
def test_classification_uses_only_driver_type_and_sqlstate(error, expected, wrapped):
    if wrapped:
        error = DBAPIError("SQL-BODY-CANARY", {"secret": "PARAM-CANARY"}, error)
    assert classify_database_failure(error) == expected


def test_pool_timeout_and_invalidated_connection_are_distinct():
    assert classify_database_failure(PoolTimeoutError()) == "pool_exhausted"
    error = DBAPIError(None, None, psycopg.InterfaceError(), connection_invalidated=True)
    assert classify_database_failure(error) == "connection_unavailable"


@pytest.mark.parametrize("error", [asyncio.CancelledError(), TimeoutError(), RuntimeError()])
def test_non_driver_failures_are_not_inferred_from_generic_types(error):
    assert classify_database_failure(error) is None


def test_exception_text_is_never_read():
    class MessageMustNotBeRead(psycopg.errors.QueryCanceled):
        def __str__(self):
            pytest.fail("classification must not format exceptions")

    assert classify_database_failure(MessageMustNotBeRead()) == "query_canceled"
