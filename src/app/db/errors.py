"""Classify driver failures without reading SQL, parameters, or error messages."""

from typing import Literal

import psycopg
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

type DatabaseFailure = Literal[
    "pool_exhausted", "connection_unavailable", "query_canceled", "lock_unavailable"
]


def classify_database_failure(error: BaseException) -> DatabaseFailure | None:
    if isinstance(error, PoolTimeoutError):
        return "pool_exhausted"
    original = error.orig if isinstance(error, DBAPIError) else error
    if not isinstance(original, psycopg.Error):
        return None
    state = original.sqlstate
    if state == "57014":
        return "query_canceled"
    if state == "55P03":
        return "lock_unavailable"
    if state is not None:
        if state.startswith("08") or state in {"57P01", "57P02", "57P03", "25P03", "53300"}:
            return "connection_unavailable"
        # Includes integrity, authentication, syntax and aborted-transaction errors.
        return None
    if type(original) is psycopg.OperationalError or isinstance(
        original, psycopg.errors.ConnectionTimeout
    ):
        return "connection_unavailable"
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return "connection_unavailable"
    return None


def is_checkpoint_unavailable(error: Exception) -> bool:
    return classify_database_failure(error) is not None
