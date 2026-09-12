from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

import structlog

from app.config import LogLevel
from app.obs.redaction import redact_event

_LOG_LEVELS: dict[LogLevel, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
_VENDOR_LOGGER_NAMES = (
    "httpcore",
    "httpx",
    "langchain_openai",
    "langfuse",
    "openai",
    "opentelemetry",
    "psycopg",
    "sqlalchemy",
)


def _json_serializer(value: Any, **kwargs: Any) -> str:
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("sort_keys", True)
    return json.dumps(value, **kwargs)


class _SafeTextWriter:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, value: str) -> int:
        try:
            return self._stream.write(value)
        except Exception:
            return len(value)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            return


def _stdlib_level_method(level: int) -> str:
    if level >= logging.CRITICAL:
        return "critical"
    if level >= logging.ERROR:
        return "error"
    if level >= logging.WARNING:
        return "warning"
    if level >= logging.INFO:
        return "info"
    return "debug"


class _SafeUvicornHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            fields: dict[str, str] = {"logger": record.name}
            if record.exc_info is not None and isinstance(record.exc_info[0], type):
                fields["error_type"] = record.exc_info[0].__name__
            method = getattr(get_logger("app.server"), _stdlib_level_method(record.levelno))
            method("server.lifecycle", **fields)
        except Exception:
            return


class _SafeVendorLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            vendor_logger = record.name.partition(".")[0]
            if vendor_logger not in _VENDOR_LOGGER_NAMES:
                vendor_logger = "external_vendor"
            fields: dict[str, str] = {"vendor_logger": vendor_logger}
            if record.exc_info is not None and isinstance(record.exc_info[0], type):
                fields["error_type"] = record.exc_info[0].__name__
            method = getattr(get_logger("app.vendor"), _stdlib_level_method(record.levelno))
            method("vendor.observability", **fields)
        except Exception:
            return


def _configure_uvicorn_logging(log_level: LogLevel) -> None:
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.addHandler(_SafeUvicornHandler(level=_LOG_LEVELS[log_level]))
    uvicorn_logger.setLevel(_LOG_LEVELS[log_level])
    uvicorn_logger.propagate = False
    uvicorn_logger.disabled = False

    for name in ("uvicorn.error", "uvicorn.access"):
        child_logger = logging.getLogger(name)
        child_logger.handlers.clear()
        child_logger.setLevel(_LOG_LEVELS[log_level])
        child_logger.propagate = True
        child_logger.disabled = False


def configure_vendor_logging() -> None:
    handler = _SafeVendorLogHandler(level=logging.WARNING)
    for name in _VENDOR_LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.disabled = False


def configure_logging(*, log_level: LogLevel, stream: TextIO | None = None) -> None:
    output = _SafeTextWriter(stream or sys.stdout)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            redact_event,
            structlog.processors.JSONRenderer(serializer=_json_serializer),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LOG_LEVELS[log_level]),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=output),
        cache_logger_on_first_use=False,
    )
    _configure_uvicorn_logging(log_level)
    configure_vendor_logging()


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


def clear_logging_context() -> None:
    structlog.contextvars.clear_contextvars()


def bind_request_context(*, request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def get_logging_context() -> dict[str, Any]:
    return structlog.contextvars.get_contextvars()
