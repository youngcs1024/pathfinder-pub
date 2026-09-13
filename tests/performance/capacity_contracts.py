"""Fixed E5.6 experiment selection and body-free, create-only evidence contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from tests.performance.contracts import digest
from tests.performance.metrics import Distribution, Summary
from tests.performance.workload import CallProfile, Contract

type Scenario = Literal["api", "mixed", "idle", "active", "reconnect", "slow"]
type Stop = Literal[
    "window_complete",
    "resource_limit",
    "queue_limit",
    "request_limit",
    "deadline",
    "guard_failed",
    "ownership_mismatch",
    "correctness_failed",
    "http_failed",
    "cancelled",
    "environment_failed",
    "metrics_incomplete",
    "cleanup_failed",
    "report_failed",
    "previous_stop",
    "suite_deadline",
]


class Profile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["capacity-e56-v1", "capacity-instant-ci-v1"]
    warmup_seconds: float = Field(ge=0, le=5)
    measurement_seconds: float = Field(gt=0, le=15)
    drain_seconds: float = Field(gt=0, le=10)
    max_http: Literal[512] = 512
    max_runs: Literal[128] = 128
    max_queue: Literal[96] = 96
    max_seconds: Literal[240] = 240
    suite_seconds: Literal[1800] = 1800
    memory_limit: Literal[939524096] = 896 * 1024**2
    tmpfs_limit: Literal[234881024] = 224 * 1024**2
    rss_limit: Literal[2147483648] = 2 * 1024**3
    minimum_free: Literal[2147483648] = 2 * 1024**3
    connection_limit: Literal[32] = 32
    slow_seconds: Literal[0.25] = 0.25
    modes: tuple[Literal["fake", "off"], ...] = ("fake", "fake", "fake", "off")

    @model_validator(mode="after")
    def fixed(self):
        expected = (5.0, 15.0, 10.0) if self.name == "capacity-e56-v1" else (5.0, 5.0, 10.0)
        if (self.warmup_seconds, self.measurement_seconds, self.drain_seconds) != expected:
            raise ValueError("invalid_windows")
        if self.modes != ("fake", "fake", "fake", "off"):
            raise ValueError("invalid_modes")
        return self


def load_profile(name: str) -> Profile:
    if name not in {"capacity-e56-v1", "capacity-instant-ci-v1"}:
        raise ValueError("invalid_profile")
    return Profile.model_validate_json(
        (Path(__file__).parent / "profiles" / f"{name}.json").read_bytes()
    )


class Point(Contract):
    scenario: Scenario
    level: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def selection(self):
        allowed = (
            {1, 2, 4}
            if self.scenario == "api"
            else ({1} if self.scenario == "mixed" else {1, 10, 50, 100})
        )
        if self.level not in allowed:
            raise ValueError("invalid_point")
        return self


def points() -> tuple[Point, ...]:
    return tuple(
        Point(scenario=scenario, level=level)
        for scenario in ("api", "mixed", "idle", "active", "reconnect", "slow")
        for level in ({"api": (1, 2, 4), "mixed": (1,)}.get(scenario, (1, 10, 50, 100)))
    )


class Manifest(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    authorization: Literal["e56_local_capacity_user_approved_v1", "ci_instant"]
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: Profile
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    call_profile: CallProfile
    call_profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    point: Point
    database_image: Literal["pgvector/pgvector:0.8.5-pg16"]
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    environment: Literal["environment-v1-capacity-ipc-v1"] = "environment-v1-capacity-ipc-v1"
    database_cpu: Literal[1] = 1
    database_memory: Literal[1073741824] = 1024**3
    database_tmpfs: Literal[268435456] = 256 * 1024**2
    worker_limit: Literal[1] = 1
    repetition: Literal[1] = 1

    @model_validator(mode="after")
    def identities(self):
        if self.experiment_id.version != 4:
            raise ValueError("invalid_identity")
        if self.profile_digest != digest(self.profile) or self.call_profile_digest != digest(
            self.call_profile
        ):
            raise ValueError("invalid_identity")
        if self.authorization == "ci_instant" and (
            self.profile.name != "capacity-instant-ci-v1"
            or self.call_profile.name != "instant-v1"
            or self.point.level != 1
        ):
            raise ValueError("invalid_authorization")
        return self


class Connection(Contract):
    ordinal: int = Field(ge=0, lt=512)
    run_id: UUID
    cursor: int = Field(ge=0)
    established_at: float | None = None
    closed_at: float | None = None
    received: tuple[int, ...] = Field(default=(), max_length=256)
    processed: tuple[int, ...] = Field(default=(), max_length=256)
    terminal: bool = False
    outcome: Literal["pending", "closed", "disconnected", "failed", "cancelled"] = "pending"


class Health(Contract):
    at: float
    memory: int | None = Field(default=None, ge=0)
    tmpfs: int | None = Field(default=None, ge=0)
    rss: int | None = Field(default=None, ge=0)
    available: int | None = Field(default=None, ge=0)
    disk_free: int | None = Field(default=None, ge=0)
    connections: int | None = Field(default=None, ge=0)
    queue: int | None = Field(default=None, ge=0)

    def stop(self, profile: Profile) -> Stop | None:
        if any(v is None for k, v in self.model_dump().items() if k != "at"):
            return "guard_failed"
        if self.queue >= profile.max_queue:
            return "queue_limit"
        if (
            self.memory >= profile.memory_limit
            or self.tmpfs >= profile.tmpfs_limit
            or self.rss >= profile.rss_limit
            or self.available < profile.minimum_free
            or self.disk_free < profile.minimum_free
            or self.connections > profile.connection_limit
        ):
            return "resource_limit"
        return None


class RunResult(Contract):
    run_id: UUID
    mode: Literal["research", "application"]
    phase: Literal["warmup", "measurement", "setup"]
    approval_mode: Literal["none", "synthetic_driver"] | None = None
    status: Literal["queued", "running", "waiting_approval", "completed", "failed", "cancelled"]
    job_status: Literal["queued", "leased", "done", "dead"]
    events: tuple[int, ...] = Field(max_length=256)
    model_attempts: int = Field(ge=0)
    mock_effects: int = Field(ge=0)


class Result(Contract):
    schema_version: Literal[1] = 1
    point: Point
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    status: Literal["PASS", "IN_PROGRESS", "NOT_RUN"]
    stop: Stop
    diagnostics: tuple[Stop, ...] = ()
    resources_released: bool = False
    completeness: bool = False
    correctness: bool | None = None
    performance_target_met: None = None
    measurement_start: float | None = None
    measurement_end: float | None = None
    actual_observation_end: float | None = None
    offered: int = 0
    generator_dropped: int = 0
    http_requests: int = 0
    runs: tuple[RunResult, ...] = ()
    connections: tuple[Connection, ...] = ()
    health: tuple[Health, ...] = Field(default=(), max_length=240)
    established_peak: int = 0
    connection_seconds: float = 0.0
    summaries: tuple[Summary, ...] = ()
    cross_window_samples: int = 0
    missing_roles: tuple[Literal["api", "worker", "driver", "supervisor"], ...] = ()
    sse_latency: Distribution | None = None
    counts_basis: Literal["whole_point_including_setup_warmup_drain"] = (
        "whole_point_including_setup_warmup_drain"
    )
    rate_window_basis: Literal["measurement_only_complete_samples_crossings_separate"] = (
        "measurement_only_complete_samples_crossings_separate"
    )
    time_basis: Literal["same_host_monotonic_recorded_at_to_receive_not_commit"] = (
        "same_host_monotonic_recorded_at_to_receive_not_commit"
    )
    counts: dict[
        Literal[
            "http_202",
            "http_failed",
            "http_unknown",
            "replayed",
            "unique_runs",
            "unfinished_runs",
            "unprocessed_events",
            "pool_peak",
            "pool_remaining",
        ],
        int,
    ] = Field(default_factory=dict)


class Finalization(Contract):
    schema_version: Literal[1] = 1
    role: Literal["api", "worker", "driver", "supervisor"]
    category: Literal["invalid_packet", "write_failed"]
    sample_count: int = Field(ge=0, le=16384)
    pool_remaining: int = Field(ge=-32768, le=32768)


class Suite(Contract):
    schema_version: Literal[1] = 1
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    profile: Profile
    authorization: Literal["e56_local_capacity_user_approved_v1"]
    results: tuple[Result, ...] = Field(max_length=20)


def validate_publication(name, record):
    types = {
        "capacity-manifest.json": Manifest,
        "capacity-result.json": Result,
        "capacity-suite.json": Suite,
    }
    if type(record) is Finalization:
        if name != f"capacity-finalization-{record.role}.json":
            raise ValueError("invalid_capacity_artifact")
        return
    if types.get(name) is not type(record):
        raise ValueError("invalid_capacity_artifact")


def lock_digest(root):
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def safe_read(path, cls, limit=32 * 1024 * 1024):
    import os
    import stat

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("invalid_artifact")
        body = stream.read(limit + 1)
    if len(body) > limit:
        raise ValueError("artifact_limit")
    return cls.model_validate_json(body)
