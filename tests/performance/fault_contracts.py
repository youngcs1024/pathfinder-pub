"""E5.9 fixed selection and strictly body-free evidence; no production state additions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from tests.performance.capacity_contracts import Health
from tests.performance.contracts import Digest, digest
from tests.performance.workload import CallRecord, Contract

type Scenario = Literal[
    "claim",
    "provider",
    "checkpoint",
    "approval",
    "response_lost",
    "lookup_unavailable",
    "cancel_before",
    "revoke_before",
    "cancel_after",
    "revoke_after",
]
SCENARIOS = (
    "claim",
    "provider",
    "checkpoint",
    "approval",
    "response_lost",
    "lookup_unavailable",
    "cancel_before",
    "revoke_before",
    "cancel_after",
    "revoke_after",
)
CRASH = SCENARIOS[:4]
AUTHORIZATION = "e59_local_faults_user_approved_v1"


class Config(Contract):
    scenario: Scenario
    instant: bool = False


def parse_config(value):
    return Config.model_validate_json(json.dumps(value))


class Profile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["faults-e59-v1", "faults-instant-ci-v1"]
    repeats: Literal[1, 3]
    seed: Literal[59] = 59
    max_seconds: Literal[240] = 240
    suite_seconds: Literal[7200] = 7200
    cleanup_seconds: Literal[70] = 70
    barrier_seconds: Literal[60] = 60
    max_http: Literal[128] = 128
    max_calls: Literal[63] = 63
    max_queue: Literal[1] = 1
    memory_limit: Literal[939524096] = 896 * 1024**2
    tmpfs_limit: Literal[234881024] = 224 * 1024**2
    rss_limit: Literal[2147483648] = 2 * 1024**3
    minimum_free: Literal[2147483648] = 2 * 1024**3
    connection_limit: Literal[32] = 32
    modes: tuple[Literal["fake", "off"], ...] = ("fake", "fake", "fake", "off")

    @model_validator(mode="after")
    def fixed(self):
        if self.repeats != (1 if self.name == "faults-instant-ci-v1" else 3):
            raise ValueError("invalid_repeats")
        if self.modes != ("fake", "fake", "fake", "off"):
            raise ValueError("invalid_modes")
        return self


def load_profile(name):
    if name not in {"faults-e59-v1", "faults-instant-ci-v1"}:
        raise ValueError("invalid_profile")
    return Profile.model_validate_json(
        (Path(__file__).parent / "profiles" / f"{name}.json").read_bytes()
    )


class Point(Contract):
    scenario: Scenario
    repetition: int = Field(ge=1, le=3)

    @property
    def mode(self):
        return "research" if self.scenario in CRASH[:3] else "application"

    @property
    def expected_run(self):
        if self.scenario.startswith(("cancel_", "revoke_")):
            return "cancelled"
        return "failed" if self.scenario == "lookup_unavailable" else "completed"


def points(profile):
    return tuple(
        Point(scenario=s, repetition=r) for s in SCENARIOS for r in range(1, profile.repeats + 1)
    )


class Manifest(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: Digest
    profile: Profile
    profile_digest: Digest
    point: Point
    authorization: Literal["e59_local_faults_user_approved_v1", "ci_instant"]
    database_image: Literal["pgvector/pgvector:0.8.5-pg16"] = "pgvector/pgvector:0.8.5-pg16"
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    database_cpu: Literal[1] = 1
    database_memory: Literal[1073741824] = 1024**3
    database_tmpfs: Literal[268435456] = 256 * 1024**2
    worker_limit: Literal[1] = 1
    generation_limit: Literal[2] = 2
    lease_seconds: Literal[3.0, 30.0]
    heartbeat_seconds: Literal[0.5, 10.0]
    call_profile_digest: Digest
    clock: Literal["real_utc_and_monotonic"] = "real_utc_and_monotonic"

    @model_validator(mode="after")
    def identity(self):
        instant = self.authorization == "ci_instant"
        if (self.profile.name == "faults-instant-ci-v1") != instant:
            raise ValueError("invalid_authorization")
        if (self.lease_seconds, self.heartbeat_seconds) != (
            (3.0, 0.5) if instant else (30.0, 10.0)
        ):
            raise ValueError("invalid_runtime")
        if (
            self.profile_digest != digest(self.profile)
            or self.point.repetition > self.profile.repeats
        ):
            raise ValueError("identity_mismatch")
        if self.experiment_id.version != 4:
            raise ValueError("invalid_identity")
        return self


class Lifecycle(Contract):
    generation: Literal[1, 2]
    pid: int = Field(gt=0)
    kind: Literal["started", "barrier", "killed", "released"]
    at: float = Field(ge=0)
    exit_code: Literal[-9] | None = None


class FaultCall(CallRecord):
    generation: Literal[1, 2]


class ModelFact(Contract):
    id: UUID
    node: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    status: Literal["started", "succeeded", "failed"]
    cost_known: bool


class Snapshot(Contract):
    run_id: UUID
    job_id: UUID
    run_status: Literal["queued", "running", "waiting_approval", "completed", "failed", "cancelled"]
    job_status: Literal["queued", "leased", "done", "dead"]
    job_attempt: int = Field(ge=0)
    lease_expires_at: float | None = None
    events: tuple[str, ...] = Field(max_length=128)
    seq: tuple[int, ...] = Field(max_length=128)
    models: tuple[ModelFact, ...] = Field(max_length=63)
    tools: tuple[UUID, ...] = Field(max_length=63)
    action_id: UUID | None = None
    request_id: UUID | None = None
    invocation_id: UUID | None = None
    action_status: (
        Literal[
            "proposed",
            "authorized",
            "executing",
            "succeeded",
            "failed",
            "outcome_unknown",
            "cancelled",
        ]
        | None
    ) = None
    invocation_status: (
        Literal["prepared", "executing", "succeeded", "failed", "outcome_unknown"] | None
    ) = None
    approval_status: Literal["pending", "approved", "rejected", "consumed", "expired"] | None = None
    args_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    target_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    binding_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    decision_count: int = Field(ge=0, le=1)
    recovery_attempts: int = Field(ge=0, le=3)
    effects: int = Field(ge=0)

    @model_validator(mode="after")
    def events_safe(self):
        if any(re.fullmatch(r"[a-z][a-z_.]{0,63}", e) is None for e in self.events):
            raise ValueError("invalid_event")
        if self.seq != tuple(range(1, len(self.events) + 1)):
            raise ValueError("invalid_sequence")
        return self


type Stop = Literal[
    "complete",
    "previous_stop",
    "suite_deadline",
    "deadline",
    "barrier_timeout",
    "resource_limit",
    "queue_limit",
    "guard_failed",
    "environment_failed",
    "correctness_failed",
    "cleanup_failed",
    "report_failed",
    "cancelled",
    "call_limit",
    "ownership_mismatch",
]


class Result(Contract):
    schema_version: Literal[1] = 1
    point: Point
    manifest_digest: Digest | None = None
    status: Literal["PASS", "IN_PROGRESS", "NOT_STARTED"]
    stop: Stop
    stage: Literal[
        "environment", "setup", "barrier", "injection", "recovery", "verification", "complete"
    ] = "environment"
    complete: bool = False
    expected_state_correct: bool = False
    automatic_business_complete: bool = False
    manual_verification_required: bool = False
    resources_released: bool = False
    pool_remaining: int | None = Field(default=None, ge=0)
    before: Snapshot | None = None
    final: Snapshot | None = None
    lifecycle: tuple[Lifecycle, ...] = Field(default=(), max_length=6)
    calls: tuple[FaultCall, ...] = Field(default=(), max_length=126)
    health: tuple[Health, ...] = Field(default=(), max_length=180)
    injection_at: float | None = None
    restart_at: float | None = None
    converged_at: float | None = None
    fault_to_convergence: float | None = Field(default=None, ge=0)
    restart_to_convergence: float | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    observation_seconds: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def consistency(self):
        if self.status == "PASS" and not (
            self.stop == "complete"
            and self.complete
            and self.expected_state_correct
            and self.resources_released
            and self.pool_remaining == 0
            and self.final is not None
            and self.manifest_digest is not None
            and self.injection_at is not None
            and self.fault_to_convergence is not None
        ):
            raise ValueError("invalid_pass")
        if self.final is not None:
            if self.automatic_business_complete != (self.final.run_status == "completed"):
                raise ValueError("invalid_business_completion")
            if self.manual_verification_required != (self.final.action_status == "outcome_unknown"):
                raise ValueError("invalid_manual_verification")
        for start, elapsed in (
            (self.injection_at, self.fault_to_convergence),
            (self.restart_at, self.restart_to_convergence),
        ):
            if elapsed is not None and (
                start is None
                or self.converged_at is None
                or abs(elapsed - (self.converged_at - start)) > 1e-6
            ):
                raise ValueError("invalid_duration")
        return self


class Suite(Contract):
    schema_version: Literal[1] = 1
    profile: Profile
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    results: tuple[Result, ...] = Field(min_length=30, max_length=30)

    @model_validator(mode="after")
    def schedule(self):
        if tuple(r.point for r in self.results) != points(self.profile):
            raise ValueError("invalid_schedule")
        return self


def validate_publication(name, record):
    if type(record) is ChildFailure and name == f"fault-error-worker-{record.generation}.json":
        return
    fixed = {
        "fault-manifest.json": Manifest,
        "fault-result.json": Result,
        "fault-suite.json": Suite,
    }
    if name in fixed and type(record) is fixed[name]:
        return
    if type(record) is Lifecycle and name == f"fault-life-{record.generation}-{record.kind}.json":
        return
    if (
        type(record) is FaultCall
        and name == f"fault-call-{record.generation}-{record.sequence:03}-{record.phase}.json"
    ):
        return
    raise ValueError("invalid_artifact")


class ErrorFrame(Contract):
    file: str = Field(pattern=r"^(?:src/app|tests/performance)/[a-z_/]+\.py$")
    line: int = Field(gt=0)


class ChildFailure(Contract):
    generation: Literal[1, 2]
    category: Literal[
        "TypeError", "ValueError", "RuntimeError", "InvalidRequestError", "IntegrityError", "other"
    ]
    frames: tuple[ErrorFrame, ...] = Field(max_length=32)
