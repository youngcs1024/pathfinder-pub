"""Hand-computable E5.5 regressions; no Docker, providers or load experiments."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from tests.performance import metrics as m
from tests.performance import metrics_runtime as runtime
from tests.performance import smoke
from tests.performance.environment import EnvironmentError, EnvironmentProfile, IsolatedEnvironment
from tests.performance.workload import profile, publish

BASE = datetime(2026, 9, 13, tzinfo=UTC)
RUN = uuid4()
JOB = uuid4()
REQUEST = uuid4()


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def utc(self):
        return BASE + timedelta(seconds=self.value)


def fact(**changes):
    return m.RunFact(
        **{
            "run_id": RUN,
            "created_at": BASE,
            "started_at": BASE + timedelta(seconds=2),
            "finished_at": BASE + timedelta(seconds=105),
            "status": "completed",
            "approval_mode": "synthetic_driver",
            **changes,
        }
    )


def segment(ordinal, start, finish=None, **changes):
    return m.Sample(
        **{
            "ordinal": ordinal,
            "metric": "segment",
            "started": start,
            "recorded_at": BASE + timedelta(seconds=start),
            "finished": finish,
            "outcome": "pending" if finish is None else "succeeded",
            "run_id": RUN,
            "job_id": JOB,
            "attempt": 1,
            "claim_ordinal": ordinal,
            **changes,
        }
    )


def packet(samples=(), role="worker", **changes):
    return m.Packet(
        **{
            "role": role,
            "process_id": uuid4(),
            "started": 0.0,
            "finished": 110.0,
            "dropped": 0,
            "write_failed": False,
            "observer_seconds": 0.0,
            "samples": samples,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "start,end,reason",
    [
        (None, 1.0, "missing"),
        (1.0, None, "unfinished"),
        (2.0, 1.0, "clock_invalid"),
        (0.0, float("inf"), "clock_invalid"),
    ],
)
def test_missing_unfinished_and_bad_clocks_never_become_zero(start, end, reason):
    assert m.difference(start, end) == m.missing(reason)


def test_values_and_nearest_rank_preserve_denominator():
    values = [m.Value(value=float(v)) for v in (1, 2, 3, 4, 100)] + [m.missing("unfinished")]
    result = m.distribution(values)
    assert (result.count, result.observed, result.p50.value, result.p95.value) == (6, 5, 3, 100)
    assert result.excluded == {"unfinished": 1}
    assert m.distribution([]).p50.reason == "empty"
    assert m.utc_difference(BASE.replace(tzinfo=None), BASE).reason == "clock_invalid"
    with pytest.raises(ValidationError):
        m.Value(value=0.0, reason="missing")
    with pytest.raises(ValidationError):
        m.Value(value=-1.0)


def test_approval_wait_does_not_inflate_worker_occupancy_and_attempt_is_not_identity():
    first = segment(1, 2.0, 5.0)
    resumed = segment(2, 102.0, 105.0, approval_request_id=REQUEST)
    p = packet((first, resumed))
    result = m.run_timing(
        fact(),
        list(p.samples),
        [p],
        (m.DecisionFact(run_id=RUN, request_id=REQUEST, decided_at=BASE + timedelta(seconds=100)),),
    )
    assert result.initial_queue_approx_seconds.value == 2
    assert result.worker_completed_segments_seconds.value == 6
    assert result.run_total_seconds.value == 105
    assert result.resume_queue_seconds[0].value == 2
    assert result.resume_queue_basis == ("decision_recorded_at",)
    assert first.attempt == resumed.attempt == 1
    assert result.segments == 2
    assert result.worker_unfinished_observed_lower_bound_seconds.value == 0


def test_censored_segment_uses_worker_observation_not_driver_clock():
    pending = segment(1, 2.0)
    observation = m.Sample(ordinal=2, metric="retrieval", started=4.0, recorded_at=BASE)
    worker = packet((pending, observation))
    driver = packet(role="driver", started=10000.0, finished=20000.0)
    result = m.run_timing(
        fact(finished_at=None, status="running"), list(worker.samples), [worker, driver], ()
    )
    assert result.run_total_seconds.reason == "unfinished"
    assert result.worker_completed_segments_seconds.value == 0
    assert result.worker_unfinished_observed_lower_bound_seconds.value == 2
    assert result.unfinished_segments == 1


def test_requeue_wait_has_distinct_basis_and_does_not_reuse_old_decision():
    first = segment(1, 2.0, 3.0, approval_request_id=REQUEST)
    retry = segment(2, 10.0, 12.0, approval_request_id=REQUEST)
    requeued = m.Sample(
        ordinal=3,
        metric="requeue",
        run_id=RUN,
        started=4.0,
        finished=4.1,
        recorded_at=BASE + timedelta(seconds=4),
        outcome="succeeded",
    )
    decisions = (m.DecisionFact(run_id=RUN, request_id=REQUEST, decided_at=BASE),)
    result = m.run_timing(fact(), [first, retry, requeued], [], decisions)
    assert result.resume_queue_seconds[0].value == 6
    assert result.resume_queue_basis == ("requeue_return_observed_at",)


def test_sql_execution_is_not_transaction_or_reader_count_and_listeners_are_removed():
    engine = create_engine("sqlite://")
    collector = m.Collector("api")
    try:
        with m.count_sql(engine, collector):
            with collector.observe("read_after"):
                with engine.begin() as connection:
                    connection.execute(text("SELECT 1"))
                    connection.execute(text("SELECT 2"))
            with pytest.raises(OperationalError), collector.observe("read_after"):
                with engine.connect() as connection:
                    connection.execute(text("SELECT * FROM missing_canary_table"))
        before = collector.packet()
        with engine.connect() as connection:
            connection.execute(text("SELECT 3"))
        assert collector.packet().samples == before.samples
        good, bad = collector.samples
        assert (good.sql_started, good.sql_finished, good.sql_failed) == (2, 2, 0)
        assert (bad.sql_started, bad.sql_finished, bad.sql_failed) == (1, 0, 1)
        assert "missing_canary_table" not in before.model_dump_json()
    finally:
        engine.dispose()


async def test_overlapping_scopes_are_task_local():
    collector = m.Collector("api")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def one():
        with collector.observe("read_after") as token:
            entered.set()
            await release.wait()
            assert collector.active.get() == (token,)

    task = asyncio.create_task(one())
    await entered.wait()
    with collector.observe("read_after") as token:
        assert collector.active.get() == (token,)
        release.set()
        await task
    assert collector.active.get() == ()
    assert len(collector.samples) == 2


async def test_cancellation_is_recorded_and_propagated():
    collector = m.Collector("worker")
    with pytest.raises(asyncio.CancelledError), collector.observe("retrieval"):
        raise asyncio.CancelledError
    assert collector.samples[0].outcome == "cancelled"
    assert collector.samples[0].finished is not None


def test_sample_limit_and_write_failure_are_visible(tmp_path, monkeypatch):
    collector = m.Collector("api", tmp_path)
    monkeypatch.setattr(m, "LIMIT", 1)
    collector.end(collector.begin("read_after"))
    assert collector.begin("read_after") is None
    assert collector.dropped == 1
    monkeypatch.setattr(m, "publish", Mock(side_effect=EnvironmentError("report_failed")))
    collector.finish()
    assert collector.write_failed


def test_segment_artifact_is_create_only_and_type_bound(tmp_path):
    sample = segment(1, 2.0)
    record = m.SegmentRecord(process_id=uuid4(), phase="started", sample=sample)
    name = "metrics-segment-001-started.json"
    publish(tmp_path, name, record)
    assert (tmp_path / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, name, record)
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "metrics-segment-002-started.json", record)
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "metrics-driver.json", profile())
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "metrics-PRIVATE_CANARY.json", record)


def test_partial_worker_exit_retains_started_segment_without_fabricating_finish(tmp_path):
    sample = segment(1, 2.0)
    publish(
        tmp_path,
        "metrics-segment-001-started.json",
        m.SegmentRecord(process_id=uuid4(), phase="started", sample=sample),
    )
    result = m.build_report(
        tmp_path,
        source_commit="a" * 40,
        lock_digest="b" * 64,
        facts=(fact(status="running", finished_at=None),),
        decisions=(),
    )
    assert result.status == "IN_PROGRESS" and "worker" in result.missing_roles
    assert result.recovered_segments[0].sample.finished is None
    assert result.runs[0].unfinished_segments == 1
    assert result.recovery.status == "not_run"


def test_cross_process_segment_identity_mismatch_rejected(tmp_path):
    sample = segment(1, 2.0)
    for phase in ("started", "finished"):
        publish(
            tmp_path,
            f"metrics-segment-001-{phase}.json",
            m.SegmentRecord(
                process_id=uuid4(),
                phase=phase,
                sample=sample if phase == "started" else segment(1, 2.0, 4.0),
            ),
        )
    with pytest.raises(ValueError, match="segment_identity_mismatch"):
        m.build_report(
            tmp_path, source_commit="a" * 40, lock_digest="b" * 64, facts=(), decisions=()
        )


def test_reader_refuses_symlink(tmp_path):
    target = tmp_path / "target.json"
    target.write_text(packet().model_dump_json())
    (tmp_path / "metrics-worker.json").symlink_to(target)
    with pytest.raises(OSError):
        m.read_record(tmp_path, "metrics-worker.json", m.Packet)


@pytest.mark.parametrize("status", [202, 500])
async def test_http_records_complete_response_and_failure_latency(status):
    clock = Clock()
    collector = m.Collector("driver", clock=clock, utc=clock.utc)

    async def handler(request):
        clock.value = 0.25
        return httpx.Response(
            status, json={"run_id": str(RUN)}, headers={"Idempotency-Replayed": "true"}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1"
    ) as client:
        http = smoke.HTTP(client, collector)
        if status == 202:
            await http.request("POST", "/runs", expected=202)
        else:
            with pytest.raises(smoke.SmokeFailure):
                await http.request("POST", "/runs", expected=202)
    sample = collector.samples[0]
    assert sample.finished - sample.started == 0.25
    assert sample.http_status == status
    assert sample.outcome == ("succeeded" if status == 202 else "failed")
    counts = m.http_counts([collector.packet()], [])
    assert counts.http_202 == (1 if status == 202 else 0)
    assert counts.replayed == (1 if status == 202 else 0)


@pytest.mark.parametrize(
    "error,outcome",
    [
        (httpx.ReadTimeout("PRIVATE_CANARY"), "timeout"),
        (httpx.ConnectError("PRIVATE_CANARY"), "failed"),
        (asyncio.CancelledError(), "cancelled"),
    ],
)
async def test_http_transport_unknown_and_cancellation_keep_samples(error, outcome):
    async def handler(request):
        raise error

    collector = m.Collector("driver")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1"
    ) as client:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancelled" else smoke.SmokeFailure
        ):
            await smoke.HTTP(client, collector).request("GET", "/")
    assert collector.samples[0].outcome == outcome
    assert collector.samples[0].http_status is None
    assert "PRIVATE_CANARY" not in collector.packet().model_dump_json()


async def test_sse_records_frame_receive_before_stream_close_and_replay_cursor():
    clock = Clock()
    collector = m.Collector("driver", clock=clock, utc=clock.utc)
    body = (
        'id: 1\nevent: run.completed\ndata: {"run_id":"'
        + str(RUN)
        + '","occurred_at":"2026-09-13T00:00:00Z"}\n\n'
    ).encode()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            clock.value = 2.0
            yield body[:20]
            clock.value = 3.0
            yield body[20:]
            clock.value = 20.0

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream())),
        base_url="http://127.0.0.1",
    ) as client:
        await smoke.HTTP(client, collector).events("/events")
    connection, received = collector.samples
    assert connection.finished == 20
    assert received.recorded_at == BASE + timedelta(seconds=3)
    assert m.utc_difference(received.event_recorded_at, received.recorded_at).value == 3
    assert received.seq == 1 and received.cursor == 0


def test_resource_cpu_uses_real_counters_memory_is_not_connection_count():
    previous = {"cpu_stats": {"cpu_usage": {"total_usage": 10}, "system_cpu_usage": 100}}
    current = {
        "cpu_stats": {"cpu_usage": {"total_usage": 20}, "system_cpu_usage": 200, "online_cpus": 4},
        "memory_stats": {"usage": 1024},
    }
    value = runtime.container_resources(current, previous)
    assert value.cpu_percent.value == 40 and value.memory_bytes.value == 1024
    assert value.connections.reason == "not_applicable"
    assert runtime.container_resources(current, None).cpu_percent.reason == "missing"
    current["cpu_stats"]["system_cpu_usage"] = 50
    assert runtime.container_resources(current, previous).cpu_percent.reason == "counter_reset"


async def test_database_permission_failure_is_null_and_no_query_body_is_read():
    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def begin(self):
            return self

        async def execute(self, statement):
            error = RuntimeError("PRIVATE_CANARY")
            error.sqlstate = "42501"
            raise error

    collector = m.Collector("driver")
    await runtime.sample_database(Session, collector)
    sample = collector.samples[0]
    assert sample.resources.connections == m.missing("permission_denied")
    assert sample.outcome == "failed" and collector.observer_seconds >= 0
    assert "PRIVATE_CANARY" not in collector.packet().model_dump_json()
    assert "query" not in runtime.DB_STATE_SQL and "client_addr" not in runtime.DB_STATE_SQL


def test_unknown_outcome_can_match_expected_without_being_automatic_success():
    case = m.RecoveryCase(
        case_id="unknown",
        expected_terminal_matches=True,
        automatically_completed=False,
        needs_manual_verification=True,
        duplicate_mock_effects=None,
        injected_at=BASE,
        restarted_at=BASE + timedelta(seconds=2),
        converged_at=BASE + timedelta(seconds=5),
    )
    result = m.recovery_summary((case,))
    assert result.expected_terminal_matches.true == 1
    assert result.automatically_completed.false == 1
    assert result.needs_manual_verification.true == 1
    assert result.duplicate_mock_effects.unknown_cases == 1
    assert result.fault_to_convergence.p50.value == 5
    assert result.restart_to_convergence.p50.value == 3


def test_invalid_metrics_opt_in_rejected_before_resources(tmp_path, monkeypatch):
    create = Mock()
    monkeypatch.setattr("tests.performance.environment.create_output_directory", create)
    environment = IsolatedEnvironment(
        EnvironmentProfile("environment-v1"), tmp_path / "new", metrics="true"
    )
    with pytest.raises(EnvironmentError, match="invalid_profile"):
        environment.start()
    create.assert_not_called()


async def test_worker_wrappers_use_real_runner_call_and_restore_references(tmp_path):
    clock = Clock()
    collector = m.Collector("worker", tmp_path, clock=clock, utc=clock.utc)
    job = SimpleNamespace(job_id=JOB, run_id=RUN, attempt=1, resume_approval_request_id=None)
    calls = []

    class Store:
        async def claim_due_job(self, **kwargs):
            calls.append("claim")
            clock.value += 2
            return job

    class Runner:
        async def run_once(self, stop):
            await root.SqlAlchemyWorkerJobStore().claim_due_job()
            clock.value += 3
            calls.append("finalize")
            return True

    root = SimpleNamespace(
        create_database_engine=lambda: None, SqlAlchemyWorkerJobStore=Store, WorkerRunner=Runner
    )
    with runtime.instrument(root, collector):
        runner = root.WorkerRunner()
        await runner.run_once(None)
        clock.value += 100  # Simulated approval idle, never inside run_once.
        await runner.run_once(None)
    assert root.WorkerRunner is Runner and root.SqlAlchemyWorkerJobStore is Store
    assert calls == ["claim", "finalize", "claim", "finalize"]
    assert [s.claim_ordinal for s in collector.samples] == [1, 2]
    assert sum(s.finished - s.started for s in collector.samples) == 6
    assert len(list(tmp_path.glob("metrics-segment-*-started.json"))) == 2
    assert (tmp_path / "metrics-worker.json").is_file()


def test_scheduler_delay_is_separate_and_dropped_arrivals_remain_in_denominator():
    from tests.performance.contracts import Receipt, RequestRecord, Result, Slot, count

    records = (
        RequestRecord(
            slot=Slot(ordinal=0, fixture_seed=1, phase="measurement", scheduled_at=0.0),
            started_at=0.2,
            finished_at=0.5,
            outcome="finished",
            receipt=Receipt(sent=True, http_status=202, run_id=RUN, replayed=False),
        ),
        RequestRecord(
            slot=Slot(ordinal=1, fixture_seed=1, phase="measurement", scheduled_at=1.0),
            outcome="generator_dropped",
            receipt=Receipt(sent=False),
        ),
    )
    result = Result(
        experiment_id=uuid4(),
        manifest_digest="a" * 64,
        stop_reason="window_complete",
        drain_timed_out=False,
        tasks_released=True,
        elapsed_seconds=2.0,
        submission_stopped_at=2.0,
        records=records,
        warmup=count(()),
        measurement=count(records),
    )
    delays = m.schedule_delays(result)
    assert delays.measurement.count == 2 and delays.measurement.observed == 1
    assert delays.measurement.p50.value == 0.2
    assert delays.measurement.excluded == {"not_applicable": 1}


def test_process_summaries_use_their_own_actual_windows_and_no_hourly_sql(tmp_path):
    for role, length in (("api", 10.0), ("driver", 5.0)):
        sample = m.Sample(
            ordinal=1,
            metric="read_after" if role == "api" else "http",
            started=1.0,
            finished=2.0,
            recorded_at=BASE,
            outcome="succeeded",
            sql_started=2 if role == "api" else 0,
            sql_finished=2 if role == "api" else 0,
        )
        publish(tmp_path, f"metrics-{role}.json", packet((sample,), role, finished=length))
    result = m.build_report(
        tmp_path, source_commit="a" * 40, lock_digest="b" * 64, facts=(), decisions=()
    )
    summaries = {s.role: s for s in result.summaries}
    assert summaries["api"].calls_per_second.value == 0.1
    assert summaries["api"].sql_executions_per_second.value == 0.2
    assert summaries["driver"].calls_per_second.value == 0.2
    assert result.status == "IN_PROGRESS"


def test_empty_packets_cannot_claim_smoke_metric_coverage(tmp_path):
    for role in m.ROLES:
        publish(tmp_path, f"metrics-{role}.json", packet(role=role))
    result = m.build_report(
        tmp_path, source_commit="a" * 40, lock_digest="b" * 64, facts=(fact(),), decisions=()
    )
    assert result.status == "IN_PROGRESS"
    assert result.runs[0].worker_completed_segments_seconds.reason == "missing"


def test_inherited_secret_and_body_fields_cannot_enter_metrics():
    for field in ("query", "sql", "parameters", "credential", "target", "error"):
        raw = segment(1, 1.0).model_dump()
        raw[field] = "PRIVATE_CANARY"
        with pytest.raises(ValidationError) as caught:
            m.Sample.model_validate(raw)
        assert "PRIVATE_CANARY" not in str(caught.value)


async def test_container_ownership_is_checked_before_stats(tmp_path, monkeypatch):
    supervisor = SimpleNamespace(
        metrics=m.Collector("supervisor", tmp_path),
        owner=uuid4().hex,
        name="owned",
        container_id="a" * 64,
        previous_stats=None,
        container=Mock(),
    )
    wrapped = supervisor.container.get_wrapped_container.return_value
    monkeypatch.setattr(
        "tests.performance._supervisor.verify_container",
        Mock(side_effect=EnvironmentError("ownership_mismatch")),
    )
    runtime.sample_container(supervisor)
    wrapped.stats.assert_not_called()
    assert supervisor.metrics.samples[0].resources.cpu_percent.reason == "unavailable"


async def test_database_sampler_cancellation_closes_sample():
    entered = asyncio.Event()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def begin(self):
            return self

        async def execute(self, statement):
            entered.set()
            await asyncio.Event().wait()

    collector = m.Collector("driver")
    task = asyncio.create_task(runtime.sample_database(Session, collector))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert collector.samples[0].outcome == "cancelled"
    assert collector.samples[0].resources.events.value is None


def test_metrics_reader_rejects_nonregular_files_and_path_escape(tmp_path):
    import os

    os.mkfifo(tmp_path / "metrics-worker.json", mode=0o600)
    with pytest.raises(ValueError, match="invalid_metrics_file"):
        m.read_record(tmp_path, "metrics-worker.json", m.Packet)
    with pytest.raises(ValueError, match="invalid_metrics_name"):
        m.read_record(tmp_path, "../metrics-worker.json", m.Packet)


def test_metrics_bootstrap_is_only_sent_to_observed_roles(tmp_path, monkeypatch):
    from tests.performance import _supervisor

    launched = []

    def launch(role, **kwargs):
        launched.append((role, kwargs["bootstrap"]))
        return SimpleNamespace(process=SimpleNamespace(pid=len(launched)))

    monkeypatch.setattr(_supervisor.OwnedProcess, "launch", launch)
    supervisor = _supervisor.Supervisor(tmp_path, uuid4().hex, None, metrics=True)
    for role in ("migrate", "api", "worker"):
        supervisor.launch(role, "owned-private-capability")
    assert "metrics" not in launched[0][1]
    assert all(bootstrap["metrics"] is True for _, bootstrap in launched[1:])
    assert all("PF_QWEN_API_KEY" not in bootstrap for _, bootstrap in launched)


def test_report_publish_failure_during_cancel_does_not_replace_cancellation(tmp_path, monkeypatch):
    class Environment:
        def __init__(self, _, directory, **kwargs):
            self.directory = directory
            self.output_created = False

        def __enter__(self):
            self.directory.mkdir(mode=0o700)
            self.output_created = True
            return self

        def __exit__(self, *args):
            return None

        def close(self):
            return {"status": "PASS", "resources_released": True}

    async def cancelled(*args):
        raise asyncio.CancelledError

    real_publish = smoke.publish

    def failing(directory, name, record):
        if name == "smoke-result.json":
            raise EnvironmentError("report_failed")
        return real_publish(directory, name, record)

    monkeypatch.setattr(smoke, "IsolatedEnvironment", Environment)
    monkeypatch.setattr(smoke, "drive", cancelled)
    monkeypatch.setattr(smoke, "publish", failing)
    with pytest.raises(asyncio.CancelledError):
        smoke.run_smoke(tmp_path / "cancelled", mode="research")
    assert (tmp_path / "cancelled" / "smoke-started.json").is_file()
