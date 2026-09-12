from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC
from time import monotonic
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Header, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import RunEventReaderDependency, TenantDependency
from app.domain.errors import InvalidEventCursorError
from app.domain.tenancy import TenantContext
from app.events.contracts import (
    TERMINAL_RUN_EVENT_TYPES,
    RunEventBatch,
    RunEventReader,
    RunEventRecord,
)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["runs"])

_CURSOR_PATTERN = re.compile(r"^[0-9]+$")
_MAX_CURSOR = 2**63 - 1
_EVENT_BATCH_SIZE = 100
_POLL_INTERVAL_SECONDS = 0.5
_HEARTBEAT_INTERVAL_SECONDS = 15.0
_HEARTBEAT = ": keepalive\n\n"


def parse_last_event_id(value: str | None) -> int:
    if value is None:
        return 0
    if _CURSOR_PATTERN.fullmatch(value) is None:
        raise InvalidEventCursorError("Last-Event-ID must be a non-negative integer")
    cursor = int(value)
    if cursor > _MAX_CURSOR:
        raise InvalidEventCursorError("Last-Event-ID exceeds the supported event sequence")
    return cursor


def encode_sse_event(event: RunEventRecord) -> str:
    occurred_at = event.recorded_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    data = json.dumps(
        {
            "version": event.version,
            "run_id": str(event.run_id),
            "seq": event.seq,
            "occurred_at": occurred_at,
            "payload": event.payload,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {event.seq}\nevent: {event.type.value}\ndata: {data}\n\n"


async def stream_run_events(
    *,
    reader: RunEventReader,
    tenant: TenantContext,
    run_id: UUID,
    cursor: int,
    initial_batch: RunEventBatch,
    is_disconnected: Callable[[], Awaitable[bool]],
    batch_size: int = _EVENT_BATCH_SIZE,
    poll_interval_seconds: float = _POLL_INTERVAL_SECONDS,
    heartbeat_interval_seconds: float = _HEARTBEAT_INTERVAL_SECONDS,
) -> AsyncIterator[str]:
    batch = initial_batch
    last_sent_at = monotonic()
    while True:
        for event in batch.events:
            if await is_disconnected():
                return
            yield encode_sse_event(event)
            cursor = event.seq
            last_sent_at = monotonic()
            if event.type in TERMINAL_RUN_EVENT_TYPES:
                return

        if batch.run_is_terminal and cursor >= batch.high_watermark:
            return
        if await is_disconnected():
            return

        if cursor >= batch.high_watermark:
            await asyncio.sleep(poll_interval_seconds)
        batch = await reader.read_after(
            tenant=tenant,
            run_id=run_id,
            after_seq=cursor,
            limit=batch_size,
        )
        if not batch.events and monotonic() - last_sent_at >= heartbeat_interval_seconds:
            if await is_disconnected():
                return
            yield _HEARTBEAT
            last_sent_at = monotonic()


@router.get(
    "/runs/{run_id}/events",
    status_code=status.HTTP_200_OK,
    response_class=StreamingResponse,
)
async def get_run_events(
    run_id: UUID,
    request: Request,
    tenant: TenantDependency,
    reader: RunEventReaderDependency,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    cursor = parse_last_event_id(last_event_id)
    initial_batch = await reader.read_after(
        tenant=tenant,
        run_id=run_id,
        after_seq=cursor,
        limit=_EVENT_BATCH_SIZE,
    )
    return StreamingResponse(
        stream_run_events(
            reader=reader,
            tenant=tenant,
            run_id=run_id,
            cursor=cursor,
            initial_batch=initial_batch,
            is_disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
