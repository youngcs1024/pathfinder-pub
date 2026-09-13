"""E5.4 scheduling contracts; these artifacts are not production workflow states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from tests.performance.environment import EnvironmentError
from tests.performance.workload import CallProfile, Contract

type Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
type StopReason = Literal[
    "window_complete",
    "request_limit",
    "queue_limit",
    "resource_limit",
    "guard_failed",
    "deadline",
    "cancelled",
    "driver_failed",
]


class Resources(Contract):
    workers: Literal[1] = 1
    database_cpu: Literal[1] = 1
    database_memory_bytes: Literal[1073741824] = 1073741824
    database_tmpfs_bytes: Literal[268435456] = 268435456


class LoadProfile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["scheduler-smoke-v1", "low-load-v1", "capacity-api-v1"]
    seed: int = Field(default=54, ge=0, le=2**32 - 1)
    warmup_seconds: float = Field(ge=0, le=120)
    measurement_seconds: float = Field(gt=0, le=120)
    drain_seconds: float = Field(gt=0, le=60)
    arrival_rate: float = Field(gt=0, le=100)
    connections: int = Field(ge=1, le=100)
    max_requests: int = Field(ge=1, le=128)
    max_inflight: int = Field(ge=1, le=100)
    max_run_seconds: float = Field(gt=0, le=180)
    max_queue_depth: int = Field(ge=1, le=128)
    saturation: Literal["drop"] = "drop"
    stop_policy: Literal["stop_submission_then_bounded_drain"] = (
        "stop_submission_then_bounded_drain"
    )
    resources: Resources = Resources()
    llm_mode: Literal["fake"] = "fake"
    search_mode: Literal["fake"] = "fake"
    auth_mode: Literal["fake"] = "fake"
    trace_mode: Literal["off"] = "off"

    @model_validator(mode="after")
    def bounds(self):
        if (
            self.warmup_seconds + self.measurement_seconds + self.drain_seconds
            > self.max_run_seconds
        ):
            raise ValueError("invalid_windows")
        if self.max_inflight > self.connections:
            raise ValueError("invalid_connections")
        return self


def digest(record: Contract) -> str:
    data = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def load_profile(name: str) -> LoadProfile:
    if name not in {"scheduler-smoke-v1", "low-load-v1"}:
        raise EnvironmentError("invalid_profile")
    try:
        return LoadProfile.model_validate_json(
            (Path(__file__).parent / "profiles" / f"{name}.json").read_bytes()
        )
    except (OSError, ValueError):
        raise EnvironmentError("invalid_profile") from None


class Manifest(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    lock_digest: Digest
    database_image: Literal["pgvector/pgvector:0.8.5-pg16"]
    database_version: str = Field(pattern=r"^16\.[0-9]{1,3}$")
    environment_profile: Literal["environment-v1"]
    profile: LoadProfile
    profile_digest: Digest
    call_profile: CallProfile
    call_profile_digest: Digest

    @model_validator(mode="after")
    def identities(self):
        if self.experiment_id.version != 4:
            raise ValueError("invalid_experiment_id")
        if self.profile_digest != digest(self.profile) or self.call_profile_digest != digest(
            self.call_profile
        ):
            raise ValueError("identity_mismatch")
        return self


class GuardState(Contract):
    """The runtime adapter must check actual ownership/resources before returning this."""

    queue_depth: int = Field(ge=0)
    resources_within_limits: bool


class Slot(Contract):
    ordinal: int = Field(ge=0, le=127)
    # A reproducible fixture selector, never a production ID or authorization binding.
    fixture_seed: int = Field(ge=0, le=2**32 - 1)
    phase: Literal["warmup", "measurement"]
    scheduled_at: float = Field(ge=0)


class Receipt(Contract):
    """Only facts observed by the transport; no response payload or exception text."""

    sent: bool | None
    http_status: int | None = Field(default=None, ge=100, le=599)
    replayed: bool | None = None
    run_id: UUID | None = None

    @model_validator(mode="after")
    def consistent(self):
        if self.http_status is not None and self.sent is not True:
            raise ValueError("invalid_receipt")
        if (self.replayed is not None or self.run_id is not None) and self.http_status != 202:
            raise ValueError("invalid_receipt")
        return self


class RequestRecord(Contract):
    slot: Slot
    started_at: float | None = Field(default=None, ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    outcome: Literal[
        "generator_dropped", "stopped", "finished", "failed", "cancelled", "unfinished"
    ]
    receipt: Receipt

    @model_validator(mode="after")
    def times(self):
        if self.started_at is not None and self.started_at < self.slot.scheduled_at:
            raise ValueError("invalid_time")
        if self.finished_at is not None and (
            self.started_at is None or self.finished_at < self.started_at
        ):
            raise ValueError("invalid_time")
        if self.outcome in {"generator_dropped", "stopped"} and (
            self.started_at is not None
            or self.finished_at is not None
            or self.receipt.sent is not False
        ):
            raise ValueError("invalid_unsent")
        if self.outcome in {"finished", "failed", "cancelled"} and (
            self.started_at is None or self.finished_at is None
        ):
            raise ValueError("missing_time")
        if self.outcome == "unfinished" and self.finished_at is not None:
            raise ValueError("invalid_unfinished")
        return self


class Counts(Contract):
    offered: int = Field(ge=0)
    generator_dropped: int = Field(ge=0)
    stopped_before_send: int = Field(ge=0)
    sent: int = Field(ge=0)
    sent_unknown: int = Field(ge=0)
    not_sent: int = Field(ge=0)
    http_accepted: int = Field(ge=0)
    http_failed: int = Field(ge=0)
    http_outcome_unknown: int = Field(ge=0)
    replays: int = Field(ge=0)
    replay_unknown: int = Field(ge=0)
    observed_unique_runs: int = Field(ge=0)
    observed_new_runs: int = Field(ge=0)
    accepted_run_unknown: int = Field(ge=0)
    failed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    unfinished: int = Field(ge=0)


def count(records: tuple[RequestRecord, ...]) -> Counts:
    accepted = [r for r in records if r.receipt.http_status == 202]
    return Counts(
        offered=len(records),
        generator_dropped=sum(r.outcome == "generator_dropped" for r in records),
        stopped_before_send=sum(r.outcome == "stopped" for r in records),
        sent=sum(r.receipt.sent is True for r in records),
        sent_unknown=sum(r.receipt.sent is None for r in records),
        not_sent=sum(r.receipt.sent is False for r in records),
        http_accepted=len(accepted),
        http_failed=sum(
            r.receipt.http_status is not None and r.receipt.http_status >= 400 for r in records
        ),
        http_outcome_unknown=sum(
            r.receipt.sent is not False and r.receipt.http_status is None for r in records
        ),
        replays=sum(r.receipt.replayed is True for r in accepted),
        replay_unknown=sum(r.receipt.replayed is None for r in accepted),
        observed_unique_runs=len(
            {r.receipt.run_id for r in accepted if r.receipt.run_id is not None}
        ),
        observed_new_runs=len(
            {
                r.receipt.run_id
                for r in accepted
                if r.receipt.run_id is not None and r.receipt.replayed is False
            }
        ),
        accepted_run_unknown=sum(r.receipt.run_id is None for r in accepted),
        failed=sum(r.outcome == "failed" for r in records),
        cancelled=sum(r.outcome == "cancelled" for r in records),
        unfinished=sum(r.outcome == "unfinished" for r in records),
    )


class Started(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    manifest_digest: Digest
    # Times in request records are monotonic offsets from this experiment's origin.
    time_basis: Literal["monotonic_offset_seconds"] = "monotonic_offset_seconds"


class Result(Contract):
    schema_version: Literal[1] = 1
    experiment_id: UUID
    manifest_digest: Digest
    stop_reason: StopReason
    drain_timed_out: bool
    tasks_released: bool
    elapsed_seconds: float = Field(ge=0)
    submission_stopped_at: float = Field(ge=0)
    records: tuple[RequestRecord, ...] = Field(max_length=128)
    warmup: Counts
    measurement: Counts

    @model_validator(mode="after")
    def accounting(self):
        if self.submission_stopped_at > self.elapsed_seconds:
            raise ValueError("invalid_time")
        if tuple(r.slot.ordinal for r in self.records) != tuple(range(len(self.records))):
            raise ValueError("invalid_ordinals")
        for phase in ("warmup", "measurement"):
            if getattr(self, phase) != count(
                tuple(r for r in self.records if r.slot.phase == phase)
            ):
                raise ValueError("invalid_counts")
        return self
