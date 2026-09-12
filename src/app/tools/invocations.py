from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from uuid import UUID

from pydantic import JsonValue

from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import (
    ToolInvocationLimitError,
    ToolInvocationReservation,
)

_ARGS_DIGEST_PREFIX = b"pathfinder-tool-args-v1\x00"
_OUTPUT_DIGEST_PREFIX = b"pathfinder-tool-output-v1\x00"


def canonical_args_digest(arguments: dict[str, JsonValue]) -> str:
    serialized = json.dumps(
        {"schema_version": 1, "arguments": arguments},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(_ARGS_DIGEST_PREFIX + serialized).hexdigest()}"


def output_summary(serialized_output: str) -> dict[str, JsonValue]:
    encoded = serialized_output.encode("utf-8")
    return {
        "schema_version": 1,
        "output_digest": f"sha256:{sha256(_OUTPUT_DIGEST_PREFIX + encoded).hexdigest()}",
        "output_bytes": len(encoded),
    }


class InMemoryToolInvocationRecorder:
    """Contract recorder for isolated unit tests; worker composition uses PostgreSQL."""

    def __init__(self) -> None:
        self._counts: dict[UUID, int] = {}
        self._lock = asyncio.Lock()

    async def consumed_call_count(self, *, workspace_id: UUID, run_id: UUID) -> int:
        async with self._lock:
            return self._counts.get(run_id, 0)

    async def reserve(
        self,
        *,
        invocation_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        tool_name: str,
        effect: ToolEffect,
        args_digest: str,
        call_limit: int,
    ) -> ToolInvocationReservation:
        del workspace_id, actor_user_id, tool_name, effect, args_digest
        async with self._lock:
            key = run_id
            call_number = self._counts.get(key, 0) + 1
            if call_number > call_limit:
                raise ToolInvocationLimitError
            self._counts[key] = call_number
        return ToolInvocationReservation(invocation_id, call_number)

    async def start_attempt(self, **kwargs: object) -> None:
        return None

    async def succeed(self, **kwargs: object) -> None:
        return None

    async def fail(self, **kwargs: object) -> None:
        return None
