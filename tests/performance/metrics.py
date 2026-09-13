"""E5.5 bounded, body-free observations; never a business or tracing authority.

Durations use process-local monotonic clocks. UTC differences are explicitly labelled
recorded/observed times on the same host, not database commit timestamps. SQLAlchemy
cursor executions exclude implicit driver BEGIN/COMMIT, psycopg checkpoint traffic and
pool pre-ping; they are not transaction counts or a claim to see every DB wire command.
"""

from __future__ import annotations

import asyncio
import os
import re
import stat
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from itertools import pairwise
from math import ceil, isfinite
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator
from sqlalchemy import event

from tests.performance.environment import EnvironmentError
from tests.performance.workload import Contract, publish

LIMIT = 2048
SEGMENT_LIMIT = 64
ROLES = ("driver", "api", "worker", "supervisor")
type Role = Literal["driver", "api", "worker", "supervisor"]
type Metric = Literal[
    "http",
    "sse_connection",
    "sse_event",
    "read_after",
    "repository",
    "retrieval",
    "segment",
    "requeue",
    "resource_db",
    "resource_container",
]
type Outcome = Literal["pending", "succeeded", "failed", "timeout", "cancelled"]
type Reason = Literal[
    "missing",
    "unfinished",
    "clock_invalid",
    "empty",
    "unavailable",
    "permission_denied",
    "not_run",
    "not_applicable",
    "counter_reset",
    "sample_limit",
]


class Value(Contract):
    value: float | None = None
    reason: Reason | None = None

    @model_validator(mode="after")
    def consistent(self):
        if (self.value is None) != (self.reason is not None):
            raise ValueError("invalid_metric_value")
        if self.value is not None and self.value < 0:
            raise ValueError("invalid_metric_value")
        return self


def missing(reason: Reason = "missing") -> Value:
    return Value(reason=reason)


def difference(start: float | None, end: float | None) -> Value:
    if start is None:
        return missing()
    if end is None:
        return missing("unfinished")
    value = end - start
    if not isfinite(value) or value < 0:
        return missing("clock_invalid")
    return Value(value=value)


def utc_difference(start: datetime | None, end: datetime | None) -> Value:
    if start is not None and (start.tzinfo is None or (end is not None and end.tzinfo is None)):
        return missing("clock_invalid")
    return difference(start.timestamp() if start else None, end.timestamp() if end else None)


class Distribution(Contract):
    count: int = Field(ge=0)
    observed: int = Field(ge=0)
    algorithm: Literal["nearest_rank"] = "nearest_rank"
    p50: Value
    p95: Value
    excluded: dict[Reason, int]


def distribution(values: list[Value]) -> Distribution:
    numbers = sorted(v.value for v in values if v.value is not None)
    return Distribution(
        count=len(values),
        observed=len(numbers),
        p50=Value(value=numbers[ceil(len(numbers) * 0.5) - 1]) if numbers else missing("empty"),
        p95=Value(value=numbers[ceil(len(numbers) * 0.95) - 1]) if numbers else missing("empty"),
        excluded=dict(Counter(v.reason for v in values if v.reason is not None)),
    )


class ScheduleDelays(Contract):
    basis: Literal["scheduled_to_execute_port_entry"] = "scheduled_to_execute_port_entry"
    warmup: Distribution
    measurement: Distribution


def schedule_delays(result) -> ScheduleDelays:
    """E5.4 offsets measure generator lag, never HTTP or worker service time."""
    from tests.performance.contracts import Result

    checked = Result.model_validate_json(result.model_dump_json())
    groups = {}
    for phase in ("warmup", "measurement"):
        groups[phase] = distribution(
            [
                difference(r.slot.scheduled_at, r.started_at)
                if r.started_at is not None
                else missing("not_applicable")
                for r in checked.records
                if r.slot.phase == phase
            ]
        )
    return ScheduleDelays(**groups)


class Resources(Contract):
    cpu_percent: Value = missing("not_applicable")
    memory_bytes: Value = missing("not_applicable")
    connections: Value = missing("not_applicable")
    active: Value = missing("not_applicable")
    waiting: Value = missing("not_applicable")
    table_bytes: Value = missing("not_applicable")
    index_bytes: Value = missing("not_applicable")
    events: Value = missing("not_applicable")


class Sample(Contract):
    ordinal: int = Field(ge=1, le=LIMIT)
    metric: Metric
    started: float
    recorded_at: datetime
    finished: float | None = None
    outcome: Outcome = "pending"
    run_id: UUID | None = None
    job_id: UUID | None = None
    approval_request_id: UUID | None = None
    attempt: int | None = Field(default=None, ge=1)
    claim_ordinal: int | None = Field(default=None, ge=1, le=SEGMENT_LIMIT)
    seq: int | None = Field(default=None, ge=1)
    cursor: int | None = Field(default=None, ge=0)
    http_status: int | None = Field(default=None, ge=100, le=599)
    expected_status: int | None = Field(default=None, ge=100, le=599)
    replayed: bool | None = None
    event_recorded_at: datetime | None = None
    sql_started: int = Field(default=0, ge=0)
    sql_finished: int = Field(default=0, ge=0)
    sql_failed: int = Field(default=0, ge=0)
    sql_seconds: float = Field(default=0.0, ge=0)
    sql_clock_invalid: bool = False
    resources: Resources | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.recorded_at.tzinfo is None or (
            self.event_recorded_at is not None and self.event_recorded_at.tzinfo is None
        ):
            raise ValueError("invalid_timestamp")
        if (self.outcome == "pending") != (self.finished is None):
            raise ValueError("invalid_finish")
        if self.sql_finished + self.sql_failed > self.sql_started:
            raise ValueError("invalid_sql_counts")
        if self.metric == "segment" and any(
            value is None for value in (self.run_id, self.job_id, self.attempt, self.claim_ordinal)
        ):
            raise ValueError("missing_segment_identity")
        return self


class Packet(Contract):
    schema_version: Literal[1] = 1
    role: Role
    process_id: UUID
    started: float
    finished: float
    dropped: int = Field(ge=0)
    write_failed: bool
    observer_seconds: float = Field(ge=0)
    samples: tuple[Sample, ...] = Field(max_length=LIMIT)

    @model_validator(mode="after")
    def unique(self):
        if self.process_id.version != 4:
            raise ValueError("invalid_process_identity")
        if any(s.metric == "segment" for s in self.samples) and self.role != "worker":
            raise ValueError("invalid_segment_role")
        claims = [s.claim_ordinal for s in self.samples if s.metric == "segment"]
        if claims != list(range(1, len(claims) + 1)):
            raise ValueError("invalid_claim_ordinals")
        if [s.ordinal for s in self.samples] != list(range(1, len(self.samples) + 1)):
            raise ValueError("invalid_ordinals")
        return self


class SegmentRecord(Contract):
    schema_version: Literal[1] = 1
    process_id: UUID
    phase: Literal["started", "finished"]
    sample: Sample

    @model_validator(mode="after")
    def segment(self):
        if self.sample.metric != "segment" or (
            (self.phase == "started") != (self.sample.outcome == "pending")
        ):
            raise ValueError("invalid_segment")
        return self


class RunFact(Contract):
    run_id: UUID
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    status: Literal["queued", "running", "waiting_approval", "completed", "failed", "cancelled"]
    approval_mode: Literal["none", "synthetic_driver", "human"]


class DecisionFact(Contract):
    run_id: UUID
    request_id: UUID
    decided_at: datetime


class RecoveryCase(Contract):
    # Logical pre-registered case ID, never an exception message.
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    expected_terminal_matches: bool | None
    automatically_completed: bool | None
    needs_manual_verification: bool | None
    duplicate_mock_effects: int | None = Field(ge=0)
    injected_at: datetime | None
    restarted_at: datetime | None
    converged_at: datetime | None


class BooleanCounts(Contract):
    true: int = Field(ge=0)
    false: int = Field(ge=0)
    unknown: int = Field(ge=0)
    denominator: int = Field(ge=0)


class EffectCounts(Contract):
    known_total: int = Field(ge=0)
    unknown_cases: int = Field(ge=0)


class RecoveryTotals(Contract):
    status: Literal["observed", "not_run"]
    cases: int = Field(ge=0)
    expected_terminal_matches: BooleanCounts
    automatically_completed: BooleanCounts
    needs_manual_verification: BooleanCounts
    duplicate_mock_effects: EffectCounts
    fault_to_convergence: Distribution
    restart_to_convergence: Distribution


def recovery_summary(cases: tuple[RecoveryCase, ...]) -> RecoveryTotals:
    result = {"status": "observed" if cases else "not_run", "cases": len(cases)}
    for field in (
        "expected_terminal_matches",
        "automatically_completed",
        "needs_manual_verification",
    ):
        values = [getattr(c, field) for c in cases]
        result[field] = BooleanCounts(
            true=values.count(True),
            false=values.count(False),
            unknown=values.count(None),
            denominator=len(values),
        )
    effects = [c.duplicate_mock_effects for c in cases]
    result["duplicate_mock_effects"] = EffectCounts(
        known_total=sum(v for v in effects if v is not None), unknown_cases=effects.count(None)
    )
    result["fault_to_convergence"] = distribution(
        [utc_difference(c.injected_at, c.converged_at) for c in cases]
    )
    result["restart_to_convergence"] = distribution(
        [utc_difference(c.restarted_at, c.converged_at) for c in cases]
    )
    return RecoveryTotals(**result)


class HttpCounts(Contract):
    client_calls: int
    http_202: int
    replayed: int
    replay_unknown: int
    observed_unique_runs: int
    observed_new_runs: int
    accepted_run_unknown: int
    persisted_runs: int
    http_status_unknown: int
    outcomes: dict[Outcome, int]


def http_counts(packets, facts):
    samples = [s for p in packets for s in p.samples if s.metric == "http"]
    accepted = [s for s in samples if s.http_status == 202]
    return HttpCounts(
        client_calls=len(samples),
        http_202=len(accepted),
        replayed=sum(s.replayed is True for s in accepted),
        replay_unknown=sum(s.replayed is None for s in accepted),
        observed_unique_runs=len({s.run_id for s in accepted if s.run_id is not None}),
        observed_new_runs=len(
            {s.run_id for s in accepted if s.run_id is not None and s.replayed is False}
        ),
        accepted_run_unknown=sum(s.run_id is None for s in accepted),
        persisted_runs=len(facts),
        http_status_unknown=sum(s.http_status is None for s in samples),
        outcomes=dict(Counter(s.outcome for s in samples)),
    )


class Collector:
    @property
    def limit(self):
        return LIMIT

    segment_limit = SEGMENT_LIMIT
    segment_type = SegmentRecord
    sample_type = Sample
    packet_type = Packet

    def __init__(self, role: Role, directory: Path | None = None, *, clock=monotonic, utc=None):
        if role not in ROLES:
            raise ValueError("invalid_role")
        self.role = role
        self.directory = directory
        self.clock = clock
        self.utc = utc or (lambda: datetime.now(UTC))
        self.process_id = uuid4()
        self.started = clock()
        self.samples: list[Sample] = []
        self.dropped = 0
        self.write_failed = False
        self.observer_seconds = 0.0
        self.claims = 0
        self.active: ContextVar[tuple[int, ...]] = ContextVar("metrics_scopes", default=())

    def begin(self, metric: Metric, **fields) -> int | None:
        if len(self.samples) >= self.limit:
            self.dropped += 1
            return None
        ordinal = len(self.samples) + 1
        self.samples.append(
            self.sample_type(
                ordinal=ordinal,
                metric=metric,
                started=self.clock(),
                recorded_at=self.utc(),
                **fields,
            )
        )
        return ordinal

    def update(self, token: int | None, **fields) -> None:
        if token is not None:
            sample = self.samples[token - 1]
            self.samples[token - 1] = self.sample_type.model_validate(
                {**sample.model_dump(), **fields}
            )

    def end(self, token: int | None, outcome: Outcome = "succeeded", **fields) -> None:
        self.update(token, finished=self.clock(), outcome=outcome, **fields)

    def write(self, name, record):
        if self.directory is None:
            return
        start = self.clock()
        try:
            publish(self.directory, name, record)
        except EnvironmentError:
            self.write_failed = True
        finally:
            elapsed = difference(start, self.clock())
            if elapsed.value is not None:
                self.observer_seconds += elapsed.value

    def segment_record(self, token, phase):
        if token is not None:
            sample = self.samples[token - 1]
            self.write(
                f"metrics-segment-{sample.claim_ordinal:03}-{phase}.json",
                self.segment_type(process_id=self.process_id, phase=phase, sample=sample),
            )

    def packet(self):
        return self.packet_type(
            role=self.role,
            process_id=self.process_id,
            started=self.started,
            finished=self.clock(),
            dropped=self.dropped,
            write_failed=self.write_failed,
            observer_seconds=self.observer_seconds,
            samples=tuple(self.samples),
        )

    def finish(self):
        if getattr(self, "_finished", False):
            return
        self._finished = True
        self.write(f"metrics-{self.role}.json", self.packet())

    @contextmanager
    def observe(self, metric: Metric, **fields):
        token = self.begin(metric, **fields)
        scope = self.active.set(
            (*self.active.get(), token) if token is not None else self.active.get()
        )
        outcome: Outcome = "failed"
        try:
            yield token
            outcome = "succeeded"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except TimeoutError:
            outcome = "timeout"
            raise
        finally:
            self.active.reset(scope)
            self.end(token, outcome)


@contextmanager
def count_sql(engine, collector: Collector):
    """Listeners belong only to this engine. Never retain statement/parameters/errors."""
    pending = {}

    def before(conn, cursor, statement, parameters, context, executemany):
        tokens = collector.active.get()
        if not tokens:
            return
        pending[id(context)] = (collector.clock(), tokens)
        for token in tokens:
            sample = collector.samples[token - 1]
            collector.update(token, sql_started=sample.sql_started + 1)

    def finish(context, failed):
        entry = pending.pop(id(context), None)
        if entry is None:
            return
        start, tokens = entry
        elapsed = difference(start, collector.clock())
        for token in tokens:
            sample = collector.samples[token - 1]
            collector.update(
                token,
                sql_finished=sample.sql_finished + (not failed),
                sql_failed=sample.sql_failed + failed,
                sql_seconds=sample.sql_seconds + (elapsed.value or 0.0),
                sql_clock_invalid=sample.sql_clock_invalid or elapsed.value is None,
            )

    def after(conn, cursor, statement, parameters, context, executemany):
        finish(context, False)

    def error(context):
        finish(context.execution_context, True)

    target = getattr(engine, "sync_engine", engine)
    handlers = (
        ("before_cursor_execute", before),
        ("after_cursor_execute", after),
        ("handle_error", error),
    )
    for name, callback in handlers:
        event.listen(target, name, callback)
    try:
        yield
    finally:
        for name, callback in reversed(handlers):
            event.remove(target, name, callback)
        pending.clear()


def validate_publication(name: str, record: Contract) -> None:
    if name in {f"metrics-{role}.json" for role in ROLES}:
        from tests.performance.capacity_metrics import CapacityPacket
        from tests.performance.queue_metrics import QueuePacket

        if (
            type(record) not in {Packet, CapacityPacket, QueuePacket}
            or name != f"metrics-{record.role}.json"
        ):
            raise ValueError("invalid_metrics_type")
    elif match := re.fullmatch(r"metrics-segment-(\d{3})-(started|finished)\.json", name):
        from tests.performance.queue_metrics import QueueSegment

        if (
            type(record) not in {SegmentRecord, QueueSegment}
            or int(match[1]) != record.sample.claim_ordinal
            or match[2] != record.phase
        ):
            raise ValueError("invalid_metrics_type")
    elif name == "metrics-result.json":
        if type(record) is not Report:
            raise ValueError("invalid_metrics_type")
    else:
        raise ValueError("invalid_metrics_name")


def read_record(directory: Path, name: str, cls):
    """Fixed filenames only, bounded reads, no symlink following (including directories)."""
    if (
        re.fullmatch(
            r"metrics-(?:(?:driver|api|worker|supervisor)|segment-\d{3}-(?:started|finished))\.json",
            name,
        )
        is None
    ):
        raise ValueError("invalid_metrics_name")
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("invalid_metrics_file")
            raw = stream.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("invalid_metrics_file")
        record = cls.model_validate_json(raw)
        validate_publication(name, record)
        return record
    finally:
        os.close(directory_fd)


class Summary(Contract):
    role: Role
    metric: Metric
    outcome: Outcome
    expected_status: int | None
    count: int
    duration_seconds: Distribution
    sql_started: int
    sql_finished: int
    sql_failed: int
    sql_duration_seconds: Value
    process_observation_window_seconds: Value
    calls_per_second: Value
    sql_executions_per_second: Value


class RunTiming(Contract):
    run_id: UUID
    initial_queue_approx_seconds: Value
    run_total_seconds: Value
    worker_completed_segments_seconds: Value
    worker_unfinished_observed_lower_bound_seconds: Value
    segments: int
    unfinished_segments: int
    approval_mode: Literal["none", "synthetic_driver", "human"]
    resume_queue_seconds: tuple[Value, ...]
    resume_queue_basis: tuple[
        Literal["decision_recorded_at", "requeue_return_observed_at", "missing"], ...
    ]


class Report(Contract):
    schema_version: Literal[1] = 1
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["PASS", "IN_PROGRESS"]
    time_basis: Literal["process_monotonic_and_same_host_utc"] = (
        "process_monotonic_and_same_host_utc"
    )
    sse_latency_basis: Literal["recorded_at_to_receive_not_commit"] = (
        "recorded_at_to_receive_not_commit"
    )
    rate_window_basis: Literal["each_process_observer_lifetime_including_idle"] = (
        "each_process_observer_lifetime_including_idle"
    )
    sql_scope: Literal["sqlalchemy_cursor_executions_excludes_checkpoint_and_driver_control"] = (
        "sqlalchemy_cursor_executions_excludes_checkpoint_and_driver_control"
    )
    window: Literal["single_run_smoke_no_warmup"] = "single_run_smoke_no_warmup"
    packets: tuple[Packet, ...]
    incomplete_reasons: tuple[
        Literal[
            "missing_role",
            "missing_sample",
            "missing_segment",
            "missing_fact",
            "write_failed",
            "sample_limit",
            "unfinished",
            "clock_invalid",
        ],
        ...,
    ] = ()
    missing_roles: tuple[Role, ...]
    recovered_segments: tuple[SegmentRecord, ...]
    summaries: tuple[Summary, ...]
    runs: tuple[RunTiming, ...]
    sse_latency_seconds: Distribution
    sse_initial_latency_seconds: Distribution
    sse_replay_latency_seconds: Distribution
    recovery_cases: tuple[RecoveryCase, ...] = ()
    recovery_evidence: Literal["NOT_RUN"] = "NOT_RUN"
    db_event_growth: Value
    facts: tuple[RunFact, ...]
    decisions: tuple[DecisionFact, ...]

    recovery: RecoveryTotals
    http_accounting: HttpCounts


def run_timing(fact: RunFact, samples: list[Sample], packets: list[Packet], decisions) -> RunTiming:
    segments = [s for s in samples if s.metric == "segment" and s.run_id == fact.run_id]
    segments.sort(key=lambda s: s.claim_ordinal)
    complete = [difference(s.started, s.finished) for s in segments if s.finished is not None]
    pending = [s for s in segments if s.finished is None]
    # One worker process per current environment; never compare monotonic values across roles.
    observed = max((s.started for s in segments), default=None)
    for packet in packets:
        if packet.role == "worker":
            observed = max(
                [s.started for s in packet.samples] + ([observed] if observed else []),
                default=observed,
            )
    lower = [difference(s.started, observed) for s in pending]
    queue = []
    bases = []
    for previous, segment in pairwise(segments):
        decision = next(
            (
                d
                for d in decisions
                if d.run_id == fact.run_id
                and d.request_id == segment.approval_request_id
                and d.decided_at >= previous.recorded_at
            ),
            None,
        )
        requeues = [
            s
            for s in samples
            if s.metric == "requeue"
            and s.outcome == "succeeded"
            and s.run_id == fact.run_id
            and previous.recorded_at <= s.recorded_at <= segment.recorded_at
        ]
        if decision is not None:
            queue.append(utc_difference(decision.decided_at, segment.recorded_at))
            bases.append("decision_recorded_at")
        elif requeues:
            queue.append(utc_difference(max(s.recorded_at for s in requeues), segment.recorded_at))
            bases.append("requeue_return_observed_at")
        else:
            queue.append(missing())
            bases.append("missing")

    def total(values):
        if any(v.value is None for v in values):
            return missing("clock_invalid")
        return Value(value=sum(v.value for v in values))

    return RunTiming(
        run_id=fact.run_id,
        initial_queue_approx_seconds=utc_difference(fact.created_at, fact.started_at),
        run_total_seconds=utc_difference(fact.created_at, fact.finished_at),
        worker_completed_segments_seconds=total(complete) if segments else missing(),
        worker_unfinished_observed_lower_bound_seconds=total(lower),
        segments=len(segments),
        unfinished_segments=len(pending),
        approval_mode=fact.approval_mode,
        resume_queue_seconds=tuple(queue),
        resume_queue_basis=tuple(bases),
    )


def build_report(
    directory: Path,
    *,
    source_commit: str,
    lock_digest: str,
    facts: tuple[RunFact, ...],
    decisions: tuple[DecisionFact, ...],
) -> Report:
    packets = []
    absent = []
    for role in ROLES:
        packet = read_record(directory, f"metrics-{role}.json", Packet)
        if packet is None:
            absent.append(role)
        else:
            packets.append(packet)
    if len({p.process_id for p in packets}) != len(packets):
        raise ValueError("duplicate_process_identity")
    samples = [s for p in packets for s in p.samples]
    recovered = []
    durable_segments = set()
    recovered_processes = set()
    worker = next((p for p in packets if p.role == "worker"), None)
    for ordinal in range(1, SEGMENT_LIMIT + 1):
        start = read_record(directory, f"metrics-segment-{ordinal:03}-started.json", SegmentRecord)
        end = read_record(directory, f"metrics-segment-{ordinal:03}-finished.json", SegmentRecord)
        if start is None:
            if end is not None:
                raise ValueError("missing_segment_start")
            continue
        if end is not None and (
            end.process_id != start.process_id
            or end.sample.model_dump(exclude={"finished", "outcome"})
            != start.sample.model_dump(exclude={"finished", "outcome"})
        ):
            raise ValueError("segment_identity_mismatch")
        record = end or start
        durable_segments.add(record.sample.claim_ordinal)
        recovered_processes.add(record.process_id)
        if worker is None:
            recovered.append(record)
            samples.append(record.sample)
        elif record.process_id != worker.process_id or record.sample not in worker.samples:
            raise ValueError("segment_packet_mismatch")
    if len(recovered_processes) > 1:
        raise ValueError("multiple_worker_processes")
    summaries = []
    for packet in packets:
        window = difference(packet.started, packet.finished)

        def rate(number, window=window):
            return Value(value=number / window.value) if window.value else missing("empty")

        groups = {(s.metric, s.outcome, s.expected_status) for s in packet.samples}
        for metric, outcome, expected_status in sorted(
            groups, key=lambda k: (k[0], k[1], k[2] or 0)
        ):
            group = [
                s
                for s in packet.samples
                if s.metric == metric
                and s.outcome == outcome
                and s.expected_status == expected_status
            ]
            summaries.append(
                Summary(
                    role=packet.role,
                    metric=metric,
                    outcome=outcome,
                    expected_status=expected_status,
                    count=len(group),
                    duration_seconds=distribution(
                        [difference(s.started, s.finished) for s in group]
                    ),
                    sql_started=sum(s.sql_started for s in group),
                    sql_finished=sum(s.sql_finished for s in group),
                    sql_failed=sum(s.sql_failed for s in group),
                    sql_duration_seconds=missing("clock_invalid")
                    if any(s.sql_clock_invalid for s in group)
                    else Value(value=sum(s.sql_seconds for s in group)),
                    process_observation_window_seconds=window,
                    calls_per_second=rate(len(group)),
                    sql_executions_per_second=rate(sum(s.sql_started for s in group)),
                )
            )
    resources = [s.resources.events for s in samples if s.metric == "resource_db" and s.resources]
    growth = (
        difference(resources[0].value, resources[-1].value) if len(resources) >= 2 else missing()
    )
    required = {"http", "sse_connection", "sse_event", "read_after", "segment"}
    missing_samples = not required.issubset({s.metric for s in samples})
    missing_segments = any(
        s.metric == "segment" and s.claim_ordinal not in durable_segments for s in samples
    )
    bad = (
        missing_samples
        or missing_segments
        or absent
        or not facts
        or any(p.dropped or p.write_failed for p in packets)
        or any(
            s.outcome == "pending" or difference(s.started, s.finished).reason == "clock_invalid"
            for s in samples
        )
        or any(difference(p.started, p.finished).value is None for p in packets)
    )
    reasons = {
        "missing_role": bool(absent),
        "missing_sample": missing_samples,
        "missing_segment": missing_segments,
        "missing_fact": not facts,
        "write_failed": any(p.write_failed for p in packets),
        "sample_limit": any(p.dropped for p in packets),
        "unfinished": any(s.outcome == "pending" for s in samples),
        "clock_invalid": any(
            difference(s.started, s.finished).reason == "clock_invalid" for s in samples
        )
        or any(difference(p.started, p.finished).value is None for p in packets),
    }
    return Report(
        incomplete_reasons=tuple(key for key, present in reasons.items() if present),
        source_commit=source_commit,
        lock_digest=lock_digest,
        status="IN_PROGRESS" if bad else "PASS",
        packets=tuple(packets),
        missing_roles=tuple(absent),
        recovered_segments=tuple(recovered),
        summaries=tuple(summaries),
        runs=tuple(run_timing(f, samples, packets, decisions) for f in facts),
        sse_latency_seconds=distribution(
            [
                utc_difference(s.event_recorded_at, s.recorded_at)
                for s in samples
                if s.metric == "sse_event"
            ]
        ),
        sse_initial_latency_seconds=distribution(
            [
                utc_difference(s.event_recorded_at, s.recorded_at)
                for s in samples
                if s.metric == "sse_event" and s.cursor == 0
            ]
        ),
        sse_replay_latency_seconds=distribution(
            [
                utc_difference(s.event_recorded_at, s.recorded_at)
                for s in samples
                if s.metric == "sse_event" and s.cursor is not None and s.cursor > 0
            ]
        ),
        db_event_growth=growth,
        facts=facts,
        decisions=decisions,
        recovery=recovery_summary(()),
        http_accounting=http_counts(packets, facts),
    )
