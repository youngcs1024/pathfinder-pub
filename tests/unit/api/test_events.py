from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest

from app.api.routes.events import stream_run_events
from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.errors import DomainNotFoundError, InvalidEventCursorError
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext, TenantService
from app.events.contracts import RunEventBatch, RunEventRecord, RunEventType
from app.main import create_app

_RECORDED_AT = datetime(2026, 8, 20, 12, 30, tzinfo=UTC)


def _event(run_id: UUID, seq: int, event_type: RunEventType) -> RunEventRecord:
    return RunEventRecord(
        run_id=run_id,
        seq=seq,
        type=event_type,
        version=1,
        payload={"status": event_type.value.rsplit(".", maxsplit=1)[-1]},
        recorded_at=_RECORDED_AT,
    )


class _ActorProvider:
    def __init__(self, actor: ActorContext) -> None:
        self.actor = actor

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        del access_token
        return self.actor


class _TenantResolver:
    def __init__(self, tenant: TenantContext | None) -> None:
        self.tenant = tenant

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        if (
            self.tenant is not None
            and workspace_id == self.tenant.workspace_id
            and actor_user_id == self.tenant.actor_user_id
        ):
            return self.tenant
        return None


class _HistoryReader:
    def __init__(self, *, run_id: UUID, events: tuple[RunEventRecord, ...]) -> None:
        self.run_id = run_id
        self.events = events
        self.calls: list[tuple[TenantContext, UUID, int, int]] = []

    async def read_after(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        after_seq: int,
        limit: int,
    ) -> RunEventBatch:
        self.calls.append((tenant, run_id, after_seq, limit))
        if run_id != self.run_id:
            raise DomainNotFoundError
        high_watermark = self.events[-1].seq
        if after_seq > high_watermark:
            raise InvalidEventCursorError
        selected = tuple(event for event in self.events if event.seq > after_seq)[:limit]
        return RunEventBatch(
            events=selected,
            high_watermark=high_watermark,
            run_is_terminal=True,
        )


class _ScriptedReader:
    def __init__(self, batches: list[RunEventBatch]) -> None:
        self.batches = batches
        self.calls = 0

    async def read_after(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        after_seq: int,
        limit: int,
    ) -> RunEventBatch:
        del tenant, run_id, after_seq, limit
        self.calls += 1
        return self.batches.pop(0)


def _application(*, member: bool = True):
    actor = ActorContext(user_id=uuid4(), subject="event-unit-actor")
    tenant = (
        TenantContext(
            workspace_id=uuid4(),
            actor_user_id=actor.user_id,
            role=WorkspaceRole.ADMIN,
        )
        if member
        else None
    )
    run_id = uuid4()
    history = (
        _event(run_id, 1, RunEventType.RUN_CREATED),
        _event(run_id, 2, RunEventType.RAG_RETRIEVED),
        _event(run_id, 3, RunEventType.RUN_COMPLETED),
    )
    reader = _HistoryReader(run_id=run_id, events=history)
    application = create_app(Settings(log_level="ERROR"))
    application.state.actor_provider = _ActorProvider(actor)
    application.state.tenant_service = TenantService(_TenantResolver(tenant))
    application.state.run_event_reader = reader
    workspace_id = tenant.workspace_id if tenant is not None else uuid4()
    return application, workspace_id, run_id, reader


async def _request(
    application,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path, headers=headers)


def _sse_events(body: str) -> list[dict[str, object]]:
    decoded: list[dict[str, object]] = []
    if not body:
        return decoded
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        decoded.append(
            {
                "id": lines[0].removeprefix("id: "),
                "event": lines[1].removeprefix("event: "),
                "data": json.loads(lines[2].removeprefix("data: ")),
            }
        )
    return decoded


async def test_first_connection_replays_ordered_persisted_events_and_closes() -> None:
    application, workspace_id, run_id, reader = _application()

    response = await _request(
        application,
        f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    UUID(response.headers["x-request-id"])
    frames = _sse_events(response.text)
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]
    assert [frame["event"] for frame in frames] == [
        "run.created",
        "rag.retrieved",
        "run.completed",
    ]
    assert frames[0]["data"] == {
        "version": 1,
        "run_id": str(run_id),
        "seq": 1,
        "occurred_at": "2026-08-20T12:30:00Z",
        "payload": {"status": "created"},
    }
    assert reader.calls[0][2] == 0


@pytest.mark.parametrize(
    ("cursor", "expected_ids"),
    [("0", ["1", "2", "3"]), ("1", ["2", "3"]), ("2", ["3"]), ("3", [])],
)
async def test_last_event_id_replays_only_unacknowledged_events(
    cursor: str,
    expected_ids: list[str],
) -> None:
    application, workspace_id, run_id, _reader = _application()

    response = await _request(
        application,
        f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
        headers={"Last-Event-ID": cursor},
    )

    assert response.status_code == 200
    assert [frame["id"] for frame in _sse_events(response.text)] == expected_ids
    if "2" in expected_ids:
        assert (
            next(frame["event"] for frame in _sse_events(response.text) if frame["id"] == "2")
            == "rag.retrieved"
        )


@pytest.mark.parametrize("cursor", ["-1", "+1", " 1", "1 ", "1.0", "abc", str(2**63)])
async def test_malformed_negative_or_unsupported_cursor_is_typed_client_error(
    cursor: str,
) -> None:
    application, workspace_id, run_id, reader = _application()

    response = await _request(
        application,
        f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
        headers={"Last-Event-ID": cursor},
    )

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:pathfinder:problem:invalid-event-cursor"
    assert reader.calls == []


async def test_future_cursor_is_rejected_without_repairing_from_another_source() -> None:
    application, workspace_id, run_id, reader = _application()

    response = await _request(
        application,
        f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
        headers={"Last-Event-ID": "4"},
    )

    assert response.status_code == 400
    assert response.json()["type"] == "urn:pathfinder:problem:invalid-event-cursor"
    assert reader.calls[0][2] == 4


async def test_nonmember_and_cross_workspace_run_are_hidden_as_not_found() -> None:
    application, workspace_id, run_id, reader = _application(member=False)
    nonmember = await _request(
        application,
        f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
    )
    assert nonmember.status_code == 404
    assert reader.calls == []

    member_application, member_workspace, _member_run, member_reader = _application()
    cross_workspace = await _request(
        member_application,
        f"/api/v1/workspaces/{member_workspace}/runs/{uuid4()}/events",
    )
    assert cross_workspace.status_code == 404
    assert member_reader.calls[0][0].workspace_id == member_workspace


async def test_replay_then_heartbeat_then_follow_terminal_event() -> None:
    run_id = uuid4()
    tenant = TenantContext(
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        role=WorkspaceRole.MEMBER,
    )
    created = _event(run_id, 1, RunEventType.RUN_CREATED)
    completed = _event(run_id, 2, RunEventType.RUN_COMPLETED)
    initial = RunEventBatch(events=(created,), high_watermark=1, run_is_terminal=False)
    reader = _ScriptedReader(
        [
            RunEventBatch(events=(), high_watermark=1, run_is_terminal=False),
            RunEventBatch(events=(completed,), high_watermark=2, run_is_terminal=True),
        ]
    )

    async def connected() -> bool:
        return False

    stream = stream_run_events(
        reader=reader,
        tenant=tenant,
        run_id=run_id,
        cursor=0,
        initial_batch=initial,
        is_disconnected=connected,
        poll_interval_seconds=0,
        heartbeat_interval_seconds=0,
    )

    assert (await anext(stream)).startswith("id: 1\nevent: run.created\n")
    assert await anext(stream) == ": keepalive\n\n"
    assert (await anext(stream)).startswith("id: 2\nevent: run.completed\n")
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert reader.calls == 2


async def test_disconnect_and_task_cancellation_cleanly_stop_polling() -> None:
    run_id = uuid4()
    tenant = TenantContext(
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        role=WorkspaceRole.MEMBER,
    )
    empty = RunEventBatch(events=(), high_watermark=0, run_is_terminal=False)
    reader = _ScriptedReader([])

    async def disconnected() -> bool:
        return True

    disconnected_stream = stream_run_events(
        reader=reader,
        tenant=tenant,
        run_id=run_id,
        cursor=0,
        initial_batch=empty,
        is_disconnected=disconnected,
        poll_interval_seconds=60,
    )
    with pytest.raises(StopAsyncIteration):
        await anext(disconnected_stream)
    assert reader.calls == 0

    async def connected() -> bool:
        return False

    cancelled_stream = stream_run_events(
        reader=reader,
        tenant=tenant,
        run_id=run_id,
        cursor=0,
        initial_batch=empty,
        is_disconnected=connected,
        poll_interval_seconds=60,
    )
    pending = asyncio.create_task(anext(cancelled_stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await cancelled_stream.aclose()
    assert reader.calls == 0
