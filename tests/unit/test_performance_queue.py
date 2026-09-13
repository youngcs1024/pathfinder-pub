"""E5.7 bounded queue contracts, accounting and lifecycle regressions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from tests.performance import _supervisor, queue
from tests.performance.adapters import Calls
from tests.performance.capacity_contracts import Health
from tests.performance.contracts import digest
from tests.performance.environment import (
    QUEUE_PROFILE,
    EnvironmentError,
    EnvironmentProfile,
    IsolatedEnvironment,
)
from tests.performance.metrics import Collector, Packet, RunTiming, Value
from tests.performance.queue_contracts import (
    Baseline,
    Fact,
    Manifest,
    Point,
    Request,
    Result,
    Suite,
    load_profile,
    points,
    successful_suite,
)
from tests.performance.queue_facts import read_calls, sample
from tests.performance.queue_metrics import QueueCollector, QueuePacket
from tests.performance.queue_report import baseline_from, window_counts
from tests.performance.workload import (
    CallProfile,
    QueueCallRecord,
    parse_profile,
    profile,
    publish,
    queue_profile,
)

SHA = "a" * 40
NOW = datetime(2026, 9, 13, tzinfo=UTC)


def manifest(kind="baseline", mode="research", baseline=None):
    p = load_profile("queue-instant-ci-v1")
    calls = queue_profile(instant=True)
    return Manifest(
        experiment_id=uuid4(),
        source_sha=SHA,
        lock_digest="b" * 64,
        profile=p,
        profile_digest=digest(p),
        call_profile=calls,
        call_profile_digest=digest(calls),
        point=Point(kind=kind, mode=mode, factor=0.8 if kind == "load" else None),
        authorization="ci_instant",
        database_image="pgvector/pgvector:0.8.5-pg16",
        database_version="16.1",
        baseline=baseline,
        arrival_rate=0.8 / baseline.mean_seconds if baseline else None,
    )


def baseline():
    return Baseline(
        experiment_id=uuid4(),
        manifest_digest="b" * 64,
        mode="research",
        source_sha=SHA,
        lock_digest="b" * 64,
        call_profile_digest=digest(queue_profile(instant=True)),
        database_version="16.1",
        service_seconds=(2.0, 4.0),
        mean_seconds=3.0,
    )


def fact(**updates):
    data = dict(
        run_id=uuid4(),
        request_id=uuid4(),
        mode="research",
        phase="measurement",
        status="completed",
        job_status="done",
        created_at=NOW,
        started_at=NOW + timedelta(seconds=1),
        finished_at=NOW + timedelta(seconds=10),
        available_at=NOW,
        event_count=3,
        model_attempts=3,
        mock_effects=0,
    )
    return Fact(**(data | updates))


def timing(f, seconds=2.0, pending=0):
    return RunTiming(
        run_id=f.run_id,
        initial_queue_approx_seconds=Value(value=1.0),
        run_total_seconds=Value(value=10.0),
        worker_completed_segments_seconds=Value(value=seconds),
        worker_unfinished_observed_lower_bound_seconds=Value(value=0.0),
        segments=2,
        unfinished_segments=pending,
        approval_mode="synthetic_driver",
        resume_queue_seconds=(),
        resume_queue_basis=(),
    )


def test_versions_preserve_legacy_limits_and_profiles():
    assert profile().max_calls == 63 and Collector("worker").segment_limit == 64
    new = queue_profile()
    assert new.max_calls == 2047 and new.schema_version == 2
    assert new.delays["chat"].minimum == 0.4 and new.delays["chat"].maximum == 0.6
    assert parse_profile(new.model_dump(mode="json")) == new
    with pytest.raises(ValidationError):
        CallProfile.model_validate_json(new.model_dump_json())
    with pytest.raises(ValidationError):
        type(new).model_validate_json(new.model_copy(update={"max_calls": 99999}).model_dump_json())
    with pytest.raises(ValidationError):
        type(new).model_validate_json(
            new.model_copy(
                update={"faults": ({"call": "chat", "ordinal": 1, "kind": "timeout"},)}
            ).model_dump_json()
        )


def test_queue_packet_retains_claim_65_and_old_reader_rejects(tmp_path):
    c = QueueCollector("worker", tmp_path)
    for ordinal in range(1, 66):
        token = c.begin("segment", run_id=uuid4(), job_id=uuid4(), attempt=1, claim_ordinal=ordinal)
        c.segment_record(token, "started")
        c.end(token)
        c.segment_record(token, "finished")
    c.finish()
    raw = (tmp_path / "metrics-worker.json").read_bytes()
    assert len(QueuePacket.model_validate_json(raw).samples) == 65
    assert (tmp_path / "metrics-segment-065-finished.json").exists()
    with pytest.raises(ValidationError):
        Packet.model_validate_json(raw)
    with pytest.raises(EnvironmentError):
        publish(tmp_path, "metrics-worker.json", c.packet())


async def test_queue_calls_retain_65_and_cancellation(tmp_path):
    c = Calls(queue_profile(instant=True), directory=tmp_path)

    async def operation():
        return 1

    for _ in range(65):
        assert await c.invoke("chat", operation, invocation_id=uuid4()) == 1
    records, unfinished = read_calls(tmp_path)
    assert len(records) == 65 and unfinished == 0
    assert records[-1].sequence == 65
    c = Calls(queue_profile())
    task = asyncio.create_task(c.invoke("chat", operation))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert c.records[-1].outcome == "cancelled"


def test_call_reader_retains_started_and_rejects_unpaired(tmp_path):
    row = QueueCallRecord(
        process="worker",
        sequence=1,
        call="chat",
        ordinal=1,
        phase="started",
        outcome="pending",
        elapsed_seconds=0.0,
        delay_seconds=0.5,
    )
    publish(tmp_path, "calls-worker-001-started.json", row)
    records, unfinished = read_calls(tmp_path)
    assert unfinished == len(records) == 1
    with pytest.raises(EnvironmentError):
        publish(tmp_path, "calls-worker-002-started.json", row)


def test_manifest_locks_rate_and_authorization():
    m = manifest("load", baseline=baseline())
    assert m.arrival_rate == 0.8 / 3
    for changes in (
        {"arrival_rate": 1.0},
        {"authorization": "e57_local_queue_user_approved_v1"},
        {"modes": ("qwen", "fake", "fake", "off")},
    ):
        with pytest.raises(ValidationError):
            Manifest.model_validate_json(m.model_copy(update=changes).model_dump_json())
    assert len(points()) == 11
    with pytest.raises(ValueError):
        load_profile("../production")


def test_baseline_uses_worker_segments_and_requires_every_sample():
    m = manifest()
    facts = [fact(), fact()]
    times = [timing(facts[0], 2.0), timing(facts[1], 4.0)]
    value = baseline_from(m, facts, times)
    assert value.mean_seconds == 3.0  # Not the 10s total including queue/approval.
    assert baseline_from(m, facts[:1], times) is None
    assert baseline_from(m, facts, [times[0], timing(facts[1], pending=1)]) is None
    assert (
        baseline_from(m, [facts[0], facts[1].model_copy(update={"status": "failed"})], times)
        is None
    )


def test_window_rates_include_only_observed_window_and_keep_failure_counts():
    start, end = NOW, NOW + timedelta(seconds=10)
    requests = [
        Request(
            ordinal=i,
            request_id=uuid4(),
            mode="research",
            phase="measurement",
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=i),
            http_status=202 if i == 1 else None,
        )
        for i in range(2)
    ]
    facts = [
        fact(finished_at=NOW + timedelta(seconds=9)),
        fact(status="failed", finished_at=NOW + timedelta(seconds=8)),
        fact(),
    ]
    accept, finish, failed, cancelled = window_counts(requests, facts, start, end)
    assert accept.value == finish.value == 0.1 and failed == 1 and cancelled == 0
    assert window_counts(requests, facts, end, start)[0].value is None


def test_oldest_due_wait_excludes_future_jobs_and_approval_idle():
    rows = [
        dict(job_status="queued", status="queued", available_at=NOW - timedelta(seconds=4)),
        dict(job_status="queued", status="queued", available_at=NOW + timedelta(seconds=10)),
        dict(job_status="done", status="waiting_approval", available_at=NOW - timedelta(days=1)),
    ]
    value = sample(rows, NOW, 1.0)
    assert value.oldest_due_seconds.value == 4 and value.queued == 2 and value.waiting_approval == 1


def test_queue_guard_and_lifetime_do_not_change_old_environment(tmp_path):
    p = load_profile("queue-e57-v1")
    health = Health(
        at=0.0, memory=1, tmpfs=1, rss=1, available=2**32, disk_free=2**32, connections=1, queue=16
    )
    assert health.stop(p) == "queue_limit"
    assert health.model_copy(update={"memory": None}).stop(p) == "guard_failed"
    old = _supervisor.Supervisor(tmp_path, uuid4().hex, None)
    new = _supervisor.Supervisor(
        tmp_path, uuid4().hex, None, metrics=True, capacity=True, queue=True
    )
    assert old.runtime_seconds == 240 and new.runtime_seconds == 420
    env = IsolatedEnvironment(EnvironmentProfile("environment-v1"), tmp_path)
    env._started = True
    env.capacity = True
    with pytest.raises(EnvironmentError):
        env.capacity_command("stop_worker")
    bad = IsolatedEnvironment(EnvironmentProfile(QUEUE_PROFILE), tmp_path / "never-created")
    with pytest.raises(EnvironmentError):
        bad.start()
    assert not (tmp_path / "never-created").exists()


async def test_approval_delay_is_observed_and_approved_once(tmp_path, monkeypatch):
    m = manifest()
    e = queue.QueueExperiment(SimpleNamespace(output_dir=tmp_path), m)
    e.sessions = object()
    e.tenant = object()
    e.http = object()
    run_id = uuid4()
    rows = [dict(id=run_id, status="waiting_approval", job_status="done")]
    calls = []

    async def approve(*args):
        calls.append(args)
        return True

    monkeypatch.setattr(queue, "approve_synthetic", approve)
    time = [10.0]
    monkeypatch.setattr(queue, "monotonic", lambda: time[0])
    await e.approve_ready(rows)
    assert not calls
    time[0] = 10.9
    await e.approve_ready(rows)
    assert not calls
    time[0] = 11.0
    await e.approve_ready(rows)
    await e.approve_ready(rows)
    assert len(calls) == 1


async def test_submission_keeps_unknown_response_without_retry(tmp_path):
    e = queue.QueueExperiment(SimpleNamespace(output_dir=tmp_path), manifest())
    e.tenant = SimpleNamespace(workspace_id=uuid4())
    seen = []

    def handler(request):
        seen.append(request.headers["Idempotency-Key"])
        raise httpx.ReadTimeout("BODY_CANARY")

    async with httpx.AsyncClient(
        base_url="http://127.0.0.1", transport=httpx.MockTransport(handler)
    ) as client:
        e.http = queue.QueueHTTP(client, e.metrics)
        receipt = await e.submit("research", "measurement")
    assert receipt.sent is None and e.requests[0].outcome == "timeout" and len(seen) == 1
    assert str(e.requests[0].request_id) == seen[0]
    assert "BODY_CANARY" not in e.requests[0].model_dump_json()


async def test_business_drain_is_bounded_and_does_not_cancel_runs(tmp_path, monkeypatch):
    e = queue.QueueExperiment(SimpleNamespace(output_dir=tmp_path), manifest())
    now = [0.0]
    reads = []

    async def rows():
        reads.append(1)
        return [dict(status="queued", job_status="queued")], NOW

    async def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(e, "rows", rows)
    monkeypatch.setattr(queue, "monotonic", lambda: now[0])
    monkeypatch.setattr(queue.asyncio, "sleep", sleep)
    await e.drain()
    assert now[0] == 10.0 and reads and not e.requests


def test_suite_never_treats_deadline_or_incomplete_as_success():
    p = load_profile("queue-e57-v1")
    results = [
        Result(point=point, source_sha=SHA, status="PASS", stop="window_complete")
        for point in points()
    ]
    suite = Suite(
        source_sha=SHA,
        authorization="e57_local_queue_user_approved_v1",
        profile=p,
        results=tuple(results),
    )
    assert successful_suite(suite)
    results[-1] = results[-1].model_copy(update={"status": "NOT_RUN", "stop": "suite_deadline"})
    assert not successful_suite(suite.model_copy(update={"results": tuple(results)}))
    assert not successful_suite(suite.model_copy(update={"results": ()}))
