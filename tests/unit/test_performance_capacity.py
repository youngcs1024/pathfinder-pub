"""E5.6 pure contracts/transport/failure regressions, in the existing unit CI target."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from tests.performance import capacity as c
from tests.performance.__main__ import main
from tests.performance.capacity_contracts import Health, Point, Result, load_profile, points
from tests.performance.capacity_metrics import CapacityCollector, CapacityPacket
from tests.performance.capacity_transport import CapacityHTTP, Consumer, Parser, close_consumers
from tests.performance.environment import EnvironmentError, EnvironmentProfile, IsolatedEnvironment
from tests.performance.metrics import Collector, Packet
from tests.performance.smoke import SmokeFailure
from tests.performance.workload import publish


def test_fixed_selection_budget_and_modes():
    assert len(points()) == 20
    p = load_profile("capacity-e56-v1")
    assert (p.warmup_seconds, p.measurement_seconds, p.drain_seconds) == (5, 15, 10)
    for change in ({"measurement_seconds": 14.0}, {"modes": ["qwen"]}, {"max_http": 513}):
        with pytest.raises(ValidationError):
            type(p).model_validate_json(json.dumps({**p.model_dump(), **change}))
    with pytest.raises(ValueError, match="invalid_profile"):
        load_profile("../../private")
    with pytest.raises(ValidationError):
        Point(scenario="idle", level=200)


def test_guard_fails_closed_and_stops_before_boundary():
    p = load_profile("capacity-e56-v1")
    good = Health(
        at=1.0,
        memory=1,
        tmpfs=1,
        rss=1,
        available=p.minimum_free,
        disk_free=p.minimum_free,
        connections=2,
        queue=0,
    )
    assert good.stop(p) is None
    for field, value, expected in (
        ("memory", None, "guard_failed"),
        ("memory", p.memory_limit, "resource_limit"),
        ("tmpfs", p.tmpfs_limit, "resource_limit"),
        ("rss", p.rss_limit, "resource_limit"),
        ("queue", 96, "queue_limit"),
        ("connections", 33, "resource_limit"),
        ("available", 0, "resource_limit"),
        ("disk_free", 0, "resource_limit"),
    ):
        assert good.model_copy(update={field: value}).stop(p) == expected


def test_capacity_limit_is_versioned_and_old_packet_unchanged(tmp_path):
    old = Collector("api")
    new = CapacityCollector("api", tmp_path)
    assert old.limit == 2048 and new.limit == 16384
    for _ in range(2049):
        old.end(old.begin("http"))
        new.end(new.begin("http"))
    assert old.dropped == 1 and new.dropped == 0
    assert old.packet().schema_version == 1 and new.packet().schema_version == 2
    new.finish()
    original = (tmp_path / "metrics-api.json").read_bytes()
    packet = CapacityPacket.model_validate_json(original)
    assert len(packet.samples) == 2049
    with pytest.raises(ValidationError):
        Packet.model_validate_json(original)
    new.finish()
    assert (tmp_path / "metrics-api.json").read_bytes() == original
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "metrics-api.json", packet)


def frame(run_id, seq, terminal=False):
    event = "run.completed" if terminal else "run.created"
    body = {"run_id": str(run_id), "seq": seq, "occurred_at": "2026-09-13T00:00:00Z", "payload": {}}
    return f"id: {seq}\nevent: {event}\ndata: {json.dumps(body)}\n\n".encode()


def test_parser_handles_split_utf8_and_heartbeat_and_caps():
    parser = Parser()
    raw = frame(uuid4(), 1)
    assert parser.feed(b": keepalive\n\n" + raw[:7]) == []
    assert parser.feed(raw[7:])[0]["id"] == 1
    parser.finish()
    with pytest.raises(SmokeFailure):
        parser.feed(b"x" * 65537)
    with pytest.raises(SmokeFailure):
        Parser().feed(b"x" * (2 * 1024 * 1024 + 1))


class Stream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body
        self.closed = False

    async def __aiter__(self):
        yield self.body

    async def aclose(self):
        self.closed = True


async def test_processed_cursor_not_last_received_and_cancel_closes_transport():
    run_id = uuid4()
    stream = Stream(frame(run_id, 1) + frame(run_id, 2, True))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
        base_url="http://127.0.0.1",
    ) as client:
        http = CapacityHTTP(client, CapacityCollector("driver"))
        consumer = Consumer(http, run_id, "/events", [], delay=10)
        task = asyncio.create_task(consumer.run())
        await consumer.established.wait()
        while len(consumer.received) != 2:
            await asyncio.sleep(0)
        await close_consumers([task])
        assert consumer.cursor == 0 and consumer.record.received == (1, 2)
        assert consumer.record.processed == () and stream.closed
        assert consumer.record.outcome == "disconnected"


async def test_reconnect_sends_processed_id_and_checks_suffix():
    run_id = uuid4()

    def respond(request):
        assert request.headers["Last-Event-ID"] == "1"
        return httpx.Response(200, stream=Stream(frame(run_id, 2, True)))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://127.0.0.1"
    ) as client:
        http = CapacityHTTP(client, CapacityCollector("driver"))
        consumer = Consumer(http, run_id, "/events", [], cursor=1)
        await consumer.run()
        assert consumer.record.terminal and consumer.cursor == 2
        assert consumer.record.processed == (2,)


async def test_consumer_rejects_foreign_run_or_missing_sequence():
    run_id = uuid4()
    for body in (frame(uuid4(), 1), frame(run_id, 2)):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request, body=body: httpx.Response(200, stream=Stream(body))
            ),
            base_url="http://127.0.0.1",
        ) as client:
            consumer = Consumer(
                CapacityHTTP(client, CapacityCollector("driver")), run_id, "/events", []
            )
            with pytest.raises(SmokeFailure):
                await consumer.run()
            assert consumer.record.outcome == "failed"


def test_no_capacity_ipc_for_smoke_or_unknown_commands(tmp_path):
    env = IsolatedEnvironment(EnvironmentProfile("environment-v1"), tmp_path / "output")
    with pytest.raises(EnvironmentError, match="protocol_failed"):
        env.capacity_command("health")
    env = replace(env, capacity=True)
    env._started = True
    with pytest.raises(EnvironmentError, match="protocol_failed"):
        env.capacity_command("arbitrary_exec")


def test_cli_requires_explicit_authorization_and_rejects_unknown_profile():
    with pytest.raises(SystemExit) as error:
        main(["capacity", "--profile", "capacity-e56-v1", "--output", "/tmp/unused"])
    assert error.value.code == 2
    with pytest.raises(SmokeFailure):
        c.run_point(
            None,
            point=Point(scenario="idle", level=10),
            profile=load_profile("capacity-e56-v1"),
            authorization="ci_instant",
        )


def test_suite_stops_higher_levels_and_retains_other_scenarios(tmp_path, monkeypatch):
    sha = "a" * 40
    monkeypatch.setattr(c, "source_sha", lambda: sha)
    seen = []

    def run(directory, *, point, **kwargs):
        seen.append(point)
        return Result(
            point=point,
            source_sha=sha,
            status="IN_PROGRESS",
            stop="resource_limit",
            resources_released=True,
        )

    monkeypatch.setattr(c, "run_point", run)
    suite = c.run_suite(
        tmp_path / "suite",
        authorization="e56_local_capacity_user_approved_v1",
        selected_profile="capacity-e56-v1",
    )
    assert len(seen) == 6 and len(suite.results) == 20
    assert sum(r.status == "NOT_RUN" for r in suite.results) == 14
    assert (tmp_path / "suite" / "capacity-suite.json").exists()


def test_correctness_or_cleanup_failure_stops_entire_suite(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "source_sha", lambda: "a" * 40)
    run = Mock(
        side_effect=lambda directory, point, **kw: Result(
            point=point,
            source_sha="a" * 40,
            status="IN_PROGRESS",
            stop="correctness_failed",
            resources_released=True,
        )
    )
    monkeypatch.setattr(c, "run_point", run)
    suite = c.run_suite(
        tmp_path / "suite",
        authorization="e56_local_capacity_user_approved_v1",
        selected_profile="capacity-e56-v1",
    )
    assert run.call_count == 1 and all(r.status == "NOT_RUN" for r in suite.results[1:])


def test_missing_metrics_and_cleanup_failure_are_both_reported(tmp_path):
    manifest = SimpleNamespace(
        profile=load_profile("capacity-instant-ci-v1"), point=Point(scenario="idle", level=1)
    )
    p = c.Experiment(SimpleNamespace(output_dir=tmp_path), manifest)
    result = c.summarize(p, "a" * 40, "http_failed", {"resources_released": False})
    assert result.stop == "http_failed"
    assert set(result.diagnostics) == {"metrics_incomplete", "cleanup_failed"}
    assert result.correctness is None and not result.completeness


def test_warmup_and_cross_window_samples_are_not_measurement(tmp_path):
    manifest = SimpleNamespace(
        profile=load_profile("capacity-instant-ci-v1"), point=Point(scenario="api", level=1)
    )
    p = c.Experiment(SimpleNamespace(output_dir=tmp_path), manifest)
    p.measurement_start, p.measurement_end = 10.0, 20.0
    collector = CapacityCollector("api", tmp_path, clock=lambda: 5.0)
    t = collector.begin("read_after")
    collector.clock = lambda: 6.0
    collector.end(t)
    collector.clock = lambda: 15.0
    t = collector.begin("read_after")
    collector.clock = lambda: 16.0
    collector.end(t)
    collector.clock = lambda: 19.0
    t = collector.begin("read_after")
    collector.clock = lambda: 21.0
    collector.end(t)
    collector.finish()
    result = c.summarize(p, "a" * 40, "deadline", {"resources_released": True})
    assert len(result.summaries) == 1
    assert result.summaries[0].count == 1 and result.summaries[0].calls_per_second.value == 0.1
    assert result.cross_window_samples == 1
