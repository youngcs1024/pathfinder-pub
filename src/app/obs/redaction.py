from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from pydantic import SecretBytes, SecretStr

REDACTED = "[REDACTED]"
_MAX_DEPTH = 8
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_KEY_SEPARATOR_PATTERN = re.compile(r"[^a-z0-9]+")
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credential_handle",
        "dashscope_api_key",
        "id_token",
        "qwen_workspace_id",
        "password",
        "passwd",
        "pf_qwen_workspace_id",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "session",
        "session_token",
        "set_cookie",
        "signature",
        "signed_url",
        "token",
    }
)
_SENSITIVE_SUFFIXES = (
    "_access_token",
    "_api_key",
    "_cookie",
    "_credential",
    "_password",
    "_refresh_token",
    "_secret",
)


def _normalized_key(value: str) -> str:
    return _KEY_SEPARATOR_PATTERN.sub("_", value.casefold()).strip("_")


def _is_sensitive_key(value: str) -> bool:
    normalized = _normalized_key(value)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _safe_key(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float | UUID | Enum):
        return str(value)
    return f"<{type(value).__name__}>"


def _redact_string(value: str) -> str:
    redacted = _BEARER_PATTERN.sub(r"\1[REDACTED]", value)
    try:
        parsed = urlsplit(redacted)
    except ValueError:
        return redacted

    if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc and parsed.query:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, REDACTED, parsed.fragment))
    return redacted


def _safe_value(value: Any, *, depth: int, seen: set[int]) -> Any:
    if depth > _MAX_DEPTH:
        return "<max-depth>"
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, SecretStr | SecretBytes):
        return REDACTED
    if isinstance(value, UUID | Decimal | datetime | date | time):
        return str(value)
    if isinstance(value, Enum):
        return _safe_value(value.value, depth=depth + 1, seen=seen)

    identity = id(value)
    if identity in seen:
        return "<cycle>"

    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                safe_key = _safe_key(key)
                result[safe_key] = (
                    REDACTED
                    if _is_sensitive_key(safe_key)
                    else _safe_value(item, depth=depth + 1, seen=seen)
                )
            return result
        finally:
            seen.remove(identity)

    if isinstance(value, list | tuple):
        seen.add(identity)
        try:
            return [_safe_value(item, depth=depth + 1, seen=seen) for item in value]
        finally:
            seen.remove(identity)

    return f"<{type(value).__name__}>"


def redact_value(value: Any) -> Any:
    """Return a JSON-safe redacted copy without mutating the input."""

    return _safe_value(value, depth=0, seen=set())


def redact_event(
    _logger: object,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor that fails closed without echoing the original event."""

    request_id = event_dict.get("request_id")
    try:
        redacted = redact_value(event_dict)
        if not isinstance(redacted, dict):
            raise TypeError("redacted log event must remain a mapping")
        return redacted
    except Exception as error:
        fallback: dict[str, Any] = {
            "event": "log.redaction_failed",
            "level": "error",
            "redaction_error_type": type(error).__name__,
        }
        if isinstance(request_id, str):
            try:
                UUID(request_id)
            except ValueError:
                pass
            else:
                fallback["request_id"] = request_id
        return fallback
