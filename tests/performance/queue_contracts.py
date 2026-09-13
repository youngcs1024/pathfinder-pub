"""E5.7 fixed experiment identities and bounded, body-free evidence."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from tests.performance.capacity_contracts import Health
from tests.performance.contracts import Digest, digest
from tests.performance.contracts import Result as LoadResult
from tests.performance.metrics import Distribution, RunTiming, Value
from tests.performance.workload import Contract, QueueCallProfile

type Mode = Literal["research", "application"]
type Phase = Literal["setup", "warmup", "measurement"]
type Status = Literal["queued", "running", "waiting_approval", "completed", "failed", "cancelled"]
type Stop = Literal[
    "window_complete",
    "queue_limit",
    "resource_limit",
    "request_limit",
    "guard_failed",
    "deadline",
    "cancelled",
    "environment_failed",
    "correctness_failed",
    "metrics_incomplete",
    "cleanup_failed",
    "report_failed",
    "http_failed",
    "baseline_incomplete",
    "previous_stop",
    "suite_deadline",
    "call_limit",
]
SAFE_STOPS = {"window_complete", "queue_limit", "resource_limit", "request_limit"}
TERMINAL = {"completed", "failed", "cancelled"}


class Profile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["queue-e57-v1", "queue-instant-ci-v1"]
    warmup_seconds: float
    measurement_seconds: float
    drain_seconds: float
    baseline_warmup: int
    baseline_samples: int
    approval_seconds: Literal[1.0] = 1.0
    max_runs: Literal[128] = 128
    max_http: Literal[2048] = 2048
    max_inflight: Literal[8] = 8
    max_queue: Literal[16] = 16
    max_seconds: Literal[420] = 420
    suite_seconds: Literal[2700] = 2700
    memory_limit: Literal[939524096] = 896 * 1024**2
    tmpfs_limit: Literal[234881024] = 224 * 1024**2
    rss_limit: Literal[2147483648] = 2 * 1024**3
    minimum_free: Literal[2147483648] = 2 * 1024**3
    connection_limit: Literal[32] = 32

    @model_validator(mode="after")
    def fixed(self):
        values = (
            self.warmup_seconds,
            self.measurement_seconds,
            self.drain_seconds,
            self.baseline_warmup,
            self.baseline_samples,
        )
        expected = (
            (15.0, 120.0, 30.0, 2, 10) if self.name == "queue-e57-v1" else (0.0, 2.0, 10.0, 0, 2)
        )
        if values != expected:
            raise ValueError("invalid_profile")
        return self


def load_profile(name):
    if name not in {"queue-e57-v1", "queue-instant-ci-v1"}:
        raise ValueError("invalid_profile")
    return Profile.model_validate_json(
        (Path(__file__).parent / "profiles" / f"{name}.json").read_bytes()
    )


class Point(Contract):
    kind: Literal["control", "baseline", "load"]
    mode: Mode
    factor: float | None = None

    @model_validator(mode="after")
    def fixed(self):
        if (
            (self.kind == "load" and self.factor not in {0.25, 0.5, 0.8, 1.1})
            or (self.kind != "load" and self.factor is not None)
            or (self.kind == "control" and self.mode != "application")
        ):
            raise ValueError("invalid_point")
        return self


def points():
    return (
        Point(kind="control", mode="application"),
        *(Point(kind="baseline", mode=mode) for mode in ("research", "application")),
        *(
            Point(kind="load", mode=mode, factor=factor)
            for mode in ("research", "application")
            for factor in (0.25, 0.5, 0.8, 1.1)
        ),
    )


class Baseline(Contract):
    experiment_id: UUID
    manifest_digest: Digest
    mode: Mode
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: Digest
    call_profile_digest: Digest
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    service_seconds: tuple[float, ...] = Field(min_length=1, max_length=10)
    mean_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def mean(self):
        if (
            any(s <= 0 for s in self.service_seconds)
            or abs(self.mean_seconds - sum(self.service_seconds) / len(self.service_seconds)) > 1e-9
        ):
            raise ValueError("invalid_baseline")
        return self


class Manifest(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: Digest
    profile: Profile
    profile_digest: Digest
    call_profile: QueueCallProfile
    call_profile_digest: Digest
    point: Point
    authorization: Literal["e57_local_queue_user_approved_v1", "ci_instant"]
    database_image: Literal["pgvector/pgvector:0.8.5-pg16"]
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    environment: Literal["environment-queue-v1"] = "environment-queue-v1"
    baseline: Baseline | None = None
    arrival_rate: float | None = Field(default=None, gt=0, le=100)
    worker_limit: Literal[1] = 1
    modes: tuple[Literal["fake", "off"], ...] = ("fake", "fake", "fake", "off")
    repetition: Literal[1] = 1

    @model_validator(mode="after")
    def identities(self):
        if (
            self.experiment_id.version != 4
            or self.profile_digest != digest(self.profile)
            or self.call_profile_digest != digest(self.call_profile)
        ):
            raise ValueError("identity_mismatch")
        instant = self.authorization == "ci_instant"
        if self.profile.name != (
            "queue-instant-ci-v1" if instant else "queue-e57-v1"
        ) or self.call_profile.name != ("queue-instant-ci-v1" if instant else "queue-delayed-v1"):
            raise ValueError("invalid_authorization")
        if self.modes != ("fake", "fake", "fake", "off"):
            raise ValueError("invalid_modes")
        if self.point.kind == "load":
            if (
                self.baseline is None
                or self.baseline.mode != self.point.mode
                or self.baseline.source_sha != self.source_sha
                or self.baseline.lock_digest != self.lock_digest
                or self.baseline.call_profile_digest != self.call_profile_digest
                or self.baseline.database_version != self.database_version
                or len(self.baseline.service_seconds) != self.profile.baseline_samples
                or self.arrival_rate is None
                or abs(self.arrival_rate - self.point.factor / self.baseline.mean_seconds) > 1e-9
            ):
                raise ValueError("invalid_baseline")
        elif self.baseline is not None or self.arrival_rate is not None:
            raise ValueError("invalid_baseline")
        return self


class Request(Contract):
    ordinal: int = Field(ge=0, lt=128)
    request_id: UUID
    mode: Mode
    phase: Phase
    started_at: datetime
    finished_at: datetime | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    replayed: bool | None = None
    run_id: UUID | None = None
    outcome: Literal["pending", "succeeded", "failed", "timeout", "cancelled"] = "pending"


class Fact(Contract):
    run_id: UUID
    request_id: UUID
    mode: Mode
    phase: Phase
    status: Status
    job_status: Literal["queued", "leased", "done", "dead"]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    available_at: datetime
    decision_at: datetime | None = None
    approval_request_id: UUID | None = None
    event_count: int = Field(ge=0)
    model_attempts: int = Field(ge=0)
    mock_effects: int = Field(ge=0)


class Snapshot(Contract):
    at: float
    recorded_at: datetime
    stage: Literal["submission_stopped", "drain_cutoff", "worker_stopped"]
    runs: tuple[Fact, ...] = Field(max_length=128)


class QueueSample(Contract):
    at: float
    recorded_at: datetime
    queued: int = Field(ge=0)
    leased: int = Field(ge=0, le=1)
    waiting_approval: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    oldest_due_seconds: Value


class Control(Contract):
    worker_released: bool = False
    backlog: int = Field(default=0, ge=0)
    approval_http_accepted: bool = False
    cancel_http_accepted: bool = False
    approval_run: UUID | None = None
    cancelled_run: UUID | None = None
    approval_accepted_at: datetime | None = None
    cancel_accepted_at: datetime | None = None
    converged: bool = False


class Result(Contract):
    schema_version: Literal[1] = 1
    point: Point
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    status: Literal["PASS", "IN_PROGRESS", "NOT_RUN"] = "IN_PROGRESS"
    stop: Stop
    diagnostics: tuple[Stop, ...] = ()
    completeness: bool = False
    correctness: bool | None = None
    performance_target_met: None = None
    resources_released: bool = False
    experiment_id: UUID | None = None
    manifest_digest: Digest | None = None
    baseline: Baseline | None = None
    measurement_start: datetime | None = None
    measurement_end: datetime | None = None
    actual_measurement_end: datetime | None = None
    requests: tuple[Request, ...] = Field(default=(), max_length=128)
    snapshots: tuple[Snapshot, ...] = Field(default=(), max_length=3)
    queue: tuple[QueueSample, ...] = Field(default=(), max_length=350)
    health: tuple[Health, ...] = Field(default=(), max_length=350)
    load: LoadResult | None = None
    control: Control | None = None
    timings: tuple[RunTiming, ...] = Field(default=(), max_length=128)
    accepted_rate: Value | None = None
    completed_rate: Value | None = None
    failed_in_window: int = Field(default=0, ge=0)
    cancelled_in_window: int = Field(default=0, ge=0)
    initial_queue: Distribution | None = None
    counts: dict[
        Literal[
            "new_runs",
            "http_accepted",
            "http_unknown",
            "http_failed",
            "replays",
            "unfinished_at_cutoff",
            "completed_at_cutoff",
            "failed_at_cutoff",
            "cancelled_at_cutoff",
            "changed_during_shutdown",
            "pool_remaining",
            "unfinished_calls",
        ],
        int,
    ] = Field(default_factory=dict)
    oldest_basis: Literal["due_queued_now_minus_available_at"] = "due_queued_now_minus_available_at"
    observation_basis: Literal["same_host_monotonic_and_recorded_utc_not_commit"] = (
        "same_host_monotonic_and_recorded_utc_not_commit"
    )


class Suite(Contract):
    schema_version: Literal[1] = 1
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    authorization: Literal["e57_local_queue_user_approved_v1"]
    profile: Profile
    results: tuple[Result, ...] = Field(max_length=11)


def validate_publication(name, record):
    expected = {
        "queue-manifest.json": Manifest,
        "queue-result.json": Result,
        "queue-suite.json": Suite,
    }
    if (
        name.startswith("queue-snapshot-")
        and type(record) is Snapshot
        and name == f"queue-snapshot-{record.stage}.json"
    ):
        return
    if expected.get(name) is not type(record):
        raise ValueError("invalid_queue_artifact")


def successful_suite(suite):
    return (
        len(suite.results) == 11
        and all(r.status == "PASS" for r in suite.results[:3])
        and all(
            r.status == "PASS"
            or (
                r.status == "NOT_RUN"
                and r.stop == "previous_stop"
                and any(
                    p.point.mode == r.point.mode
                    and p.status == "PASS"
                    and p.stop in SAFE_STOPS - {"window_complete"}
                    for p in suite.results[:i]
                )
            )
            for i, r in enumerate(suite.results)
        )
    )
