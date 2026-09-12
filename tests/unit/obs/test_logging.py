from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from io import StringIO
from typing import Any
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy.exc import StatementError

from app.obs.logging import (
    bind_request_context,
    clear_logging_context,
    configure_logging,
    get_logger,
)
from app.obs.redaction import REDACTED, redact_event, redact_value

CANARY = "PATHFINDER-SECRET-CANARY"


class _DangerousObject:
    def __repr__(self) -> str:
        return CANARY


class _BrokenMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("payload",))

    def __len__(self) -> int:
        return 1

    def items(self) -> Any:
        raise RuntimeError(CANARY)


class _FailingStream:
    def write(self, _value: str) -> int:
        raise RuntimeError(CANARY)

    def flush(self) -> None:
        raise RuntimeError(CANARY)


def test_redaction_is_recursive_json_safe_and_non_mutating() -> None:
    original = {
        "Authorization": f"Bearer {CANARY}",
        "DASHSCOPE_API_KEY": CANARY,
        "PF_QWEN_WORKSPACE_ID": CANARY,
        "nested": [
            {"api-key": CANARY, "token_count": 42},
            (f"Bearer {CANARY}", "https://example.test/path?signature=secret"),
        ],
        "object": _DangerousObject(),
    }

    redacted = redact_value(original)

    assert redacted == {
        "Authorization": REDACTED,
        "DASHSCOPE_API_KEY": REDACTED,
        "PF_QWEN_WORKSPACE_ID": REDACTED,
        "nested": [
            {"api-key": REDACTED, "token_count": 42},
            [f"Bearer {REDACTED}", f"https://example.test/path?{REDACTED}"],
        ],
        "object": "<_DangerousObject>",
    }
    assert original["Authorization"] == f"Bearer {CANARY}"
    assert CANARY not in json.dumps(redacted)


def test_redaction_handles_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    assert redact_value(cyclic) == ["<cycle>"]


def test_redaction_covers_cookie_secret_and_credential_key_variants() -> None:
    redacted = redact_value(
        {
            "Cookie": CANARY,
            "set-cookie": CANARY,
            "client-secret": CANARY,
            "workspace_credential": CANARY,
            "wrapped": SecretStr(CANARY),
        }
    )

    assert redacted == {
        "Cookie": REDACTED,
        "set-cookie": REDACTED,
        "client-secret": REDACTED,
        "workspace_credential": REDACTED,
        "wrapped": REDACTED,
    }
    assert CANARY not in json.dumps(redacted)


def test_redaction_processor_discards_event_when_nested_mapping_fails() -> None:
    request_id = str(uuid4())

    result = redact_event(
        object(),
        "info",
        {"event": CANARY, "request_id": request_id, "nested": _BrokenMapping()},
    )

    assert result == {
        "event": "log.redaction_failed",
        "level": "error",
        "redaction_error_type": "RuntimeError",
        "request_id": request_id,
    }
    assert CANARY not in json.dumps(result)


def test_structlog_outputs_json_with_bound_request_context() -> None:
    stream = StringIO()
    request_id = str(uuid4())
    configure_logging(log_level="INFO", stream=stream)
    clear_logging_context()
    bind_request_context(request_id=request_id)

    get_logger("test").info(
        "test.event",
        password=CANARY,
        location=f"https://example.test/path?token={CANARY}",
    )
    clear_logging_context()

    event = json.loads(stream.getvalue())
    assert event["event"] == "test.event"
    assert event["level"] == "info"
    assert event["request_id"] == request_id
    assert event["password"] == REDACTED
    assert event["location"] == f"https://example.test/path?{REDACTED}"
    assert CANARY not in stream.getvalue()


def test_logging_sink_failure_is_dropped_without_raw_fallback(capsys) -> None:
    configure_logging(log_level="INFO", stream=_FailingStream())

    get_logger("test").info("test.event", password=CANARY)

    captured = capsys.readouterr()
    assert CANARY not in captured.out
    assert CANARY not in captured.err


def test_uvicorn_logging_is_json_safe_and_does_not_format_original_record() -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    logger = logging.getLogger("uvicorn.error")

    logger.warning(CANARY, CANARY)
    try:
        raise RuntimeError(CANARY)
    except RuntimeError:
        logger.exception(CANARY, CANARY)

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        "server.lifecycle",
        "server.lifecycle",
    ]
    assert [event["logger"] for event in events] == [
        "uvicorn.error",
        "uvicorn.error",
    ]
    assert [event["level"] for event in events] == ["warning", "error"]
    assert "error_type" not in events[0]
    assert events[1]["error_type"] == "RuntimeError"
    assert CANARY not in stream.getvalue()
    assert "Traceback" not in stream.getvalue()


def test_qwen_transport_vendor_logs_drop_workspace_host_key_and_exception_text() -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    logger = logging.getLogger("openai.qwen-transport")
    workspace_canary = "provider-workspace-secret-canary"
    key_canary = "dashscope-key-secret-canary"

    try:
        raise RuntimeError(key_canary)
    except RuntimeError:
        logger.error(
            "request failed at https://%s.cn-beijing.maas.aliyuncs.com with Bearer %s",
            workspace_canary,
            key_canary,
            exc_info=True,
        )

    event = json.loads(stream.getvalue())
    assert event["event"] == "vendor.observability"
    assert event["vendor_logger"] == "openai"
    assert event["error_type"] == "RuntimeError"
    assert workspace_canary not in stream.getvalue()
    assert key_canary not in stream.getvalue()
    assert "cn-beijing.maas.aliyuncs.com" not in stream.getvalue()


def test_repeated_logging_configuration_does_not_stack_uvicorn_handlers() -> None:
    stream = StringIO()

    configure_logging(log_level="INFO", stream=stream)
    configure_logging(log_level="INFO", stream=stream)
    logging.getLogger("uvicorn.error").info("ignored original message")

    assert len(logging.getLogger("uvicorn").handlers) == 1
    assert logging.getLogger("uvicorn.error").handlers == []
    assert logging.getLogger("uvicorn.error").propagate is True
    assert len(logging.getLogger("openai").handlers) == 1
    assert len(stream.getvalue().splitlines()) == 1


@pytest.mark.parametrize("logger_name", ["sqlalchemy.engine.Engine", "sqlalchemy.pool", "psycopg"])
def test_database_logs_discard_sql_parameters_dsn_and_exception_chain(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    stream = StringIO()
    configure_logging(log_level="DEBUG", stream=stream)
    configure_logging(log_level="DEBUG", stream=stream)
    logger = logging.getLogger(logger_name)
    canaries = (
        "SQL-TEXT-CANARY",
        "BOUND-PARAMETER-CANARY",
        "postgresql://user:DSN-CANARY@localhost/db",
        "USER-QUERY-CANARY",
        "MODEL-BODY-CANARY",
    )
    try:
        try:
            raise RuntimeError(canaries[2])
        except RuntimeError as cause:
            raise StatementError(canaries[3], canaries[0], {"body": canaries[1]}, cause) from cause
    except StatementError:
        logger.error("%s %s", canaries[3], canaries[4], exc_info=True, stack_info=True)
    logger.debug("%s", canaries)

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "vendor.observability"
    assert events[0]["vendor_logger"] == logger_name.partition(".")[0]
    assert events[0]["error_type"] == "StatementError"
    assert events[0]["level"] == "error"
    captured = capsys.readouterr()
    output = stream.getvalue() + captured.out + captured.err
    assert all(canary not in output for canary in canaries)
    assert "Traceback" not in output
    assert "Stack" not in output
