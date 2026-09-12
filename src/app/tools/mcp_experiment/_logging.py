"""Pinned mcp==2.1.1 logging boundary, active only during experiment sessions.

The stdio parser and JSON-RPC dispatcher can log raw validation exceptions.
Filter at the originating loggers, before any handler formats those records.
Context-local state follows SDK tasks; overlapping sessions do not share signals.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_SURVIVOR_WARNING = "MCP server process %d is still alive after the kill escalation; abandoning it"
_LOGGERS = ("mcp.client.stdio", "mcp.shared.jsonrpc_dispatcher", "client")


@dataclass
class _LogEvidence:
    survivor: bool = False


_ACTIVE: ContextVar[_LogEvidence | None] = ContextVar("mcp_log_evidence", default=None)


class _SDKFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        evidence = _ACTIVE.get()
        if evidence is None:
            return True
        # Never call getMessage(), interpolate args, or inspect exception text.
        if (
            record.name == "mcp.client.stdio"
            and record.levelno == logging.WARNING
            and type(record.msg) is str
            and record.msg == _SURVIVOR_WARNING
        ):
            evidence.survivor = True
        return False


_FILTER = _SDKFilter()
_USERS = 0
_LEVELS: dict[str, int] = {}


@contextmanager
def _sdk_logging_guard() -> Iterator[_LogEvidence]:
    global _USERS
    evidence = _LogEvidence()
    token = _ACTIVE.set(evidence)
    if _USERS == 0:
        for name in _LOGGERS:
            logger = logging.getLogger(name)
            _LEVELS[name] = logger.level
            logger.addFilter(_FILTER)
            # Ensure the fixed survivor warning reaches the filter.
            if logger.getEffectiveLevel() > logging.WARNING:
                logger.setLevel(logging.WARNING)
    _USERS += 1
    try:
        yield evidence
    finally:
        _ACTIVE.reset(token)
        _USERS -= 1
        if _USERS == 0:
            for name in _LOGGERS:
                logger = logging.getLogger(name)
                logger.removeFilter(_FILTER)
                logger.setLevel(_LEVELS.pop(name))
