"""Versioned, dev-only call policy and body-free, create-only evidence."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from tests.performance.environment import EnvironmentError

type CallKind = Literal["chat", "embedding", "search", "mock_submit", "mock_lookup"]
KINDS = ("chat", "embedding", "search", "mock_submit", "mock_lookup")
type FaultKind = Literal["transient", "permanent", "timeout", "response_lost"]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class Delay(Contract):
    minimum: float = Field(ge=0, le=2)
    maximum: float = Field(ge=0, le=2)

    @model_validator(mode="after")
    def ordered(self):
        if self.minimum > self.maximum:
            raise ValueError("invalid_delay")
        return self


class Fault(Contract):
    call: CallKind
    ordinal: int = Field(ge=1, le=64)
    kind: FaultKind

    @model_validator(mode="after")
    def supported(self):
        if self.kind == "response_lost" and self.call != "mock_submit":
            raise ValueError("invalid_fault")
        if self.call == "mock_lookup" and self.kind == "permanent":
            raise ValueError("invalid_fault")
        return self


class CallProfile(Contract):
    schema_version: Literal[1] = 1
    name: Literal["instant-v1", "delayed-v1"]
    seed: int = Field(default=53, ge=0, le=2**32 - 1)
    # One of the total 64 calls is reserved for driver-side resume ingestion.
    max_calls: int = Field(default=63, ge=1, le=63)
    delays: dict[CallKind, Delay]
    faults: tuple[Fault, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def complete(self):
        if set(self.delays) != set(KINDS):
            raise ValueError("invalid_profile")
        if self.name == "instant-v1" and any(item.maximum for item in self.delays.values()):
            raise ValueError("invalid_profile")
        keys = [(item.call, item.ordinal) for item in self.faults]
        if len(keys) != len(set(keys)):
            raise ValueError("invalid_profile")
        return self


class QueueCallProfile(CallProfile):
    schema_version: Literal[2] = 2
    name: Literal["queue-delayed-v1", "queue-instant-ci-v1"]
    seed: Literal[57] = 57
    max_calls: Literal[2047] = 2047

    @model_validator(mode="after")
    def queue_policy(self):
        if self.faults:
            raise ValueError("invalid_faults")
        instant = self.name == "queue-instant-ci-v1"
        expected = {
            kind: Delay(
                minimum=0.0 if instant else (0.4 if kind == "chat" else 0.02),
                maximum=0.0 if instant else (0.6 if kind == "chat" else 0.05),
            )
            for kind in KINDS
        }
        if self.delays != expected:
            raise ValueError("invalid_delays")
        return self


type CallPolicy = QueueCallProfile | CallProfile


def queue_profile(*, instant=False):
    return QueueCallProfile(
        name="queue-instant-ci-v1" if instant else "queue-delayed-v1",
        delays={
            kind: Delay(
                minimum=0.0 if instant else (0.4 if kind == "chat" else 0.02),
                maximum=0.0 if instant else (0.6 if kind == "chat" else 0.05),
            )
            for kind in KINDS
        },
    )


def profile(name: Literal["instant-v1", "delayed-v1"] = "instant-v1") -> CallProfile:
    return CallProfile(
        name=name,
        delays={
            kind: Delay(
                minimum=0.0 if name == "instant-v1" else (0.05 if kind == "chat" else 0.01),
                maximum=0.0 if name == "instant-v1" else (0.15 if kind == "chat" else 0.05),
            )
            for kind in KINDS
        },
    )


def parse_profile(value: object) -> CallPolicy:
    try:
        # Round-trip validates tuple fields at the JSON IPC boundary, and rejects non-JSON data.
        cls = (
            QueueCallProfile
            if isinstance(value, dict) and value.get("schema_version") == 2
            else CallProfile
        )
        return cls.model_validate_json(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError, ValidationError):
        raise EnvironmentError("invalid_profile") from None


class CallRecord(Contract):
    schema_version: Literal[1] = 1
    process: Literal["worker", "ingest"]
    sequence: int = Field(ge=1, le=64)
    call: CallKind
    ordinal: int = Field(ge=1, le=64)
    invocation_id: UUID | None = None
    action_id: UUID | None = None
    delay_seconds: float = Field(ge=0, le=2)
    phase: Literal["started", "finished"]
    outcome: Literal["pending", "succeeded", "failed", "cancelled", "timeout", "response_lost"]
    elapsed_seconds: float = Field(ge=0)


class QueueCallRecord(CallRecord):
    schema_version: Literal[2] = 2
    sequence: int = Field(ge=1, le=2048)
    ordinal: int = Field(ge=1, le=2048)


def publish(directory: Path, name: str, record: Contract) -> None:
    """No free-form exception, payload, path or target string enters an artifact."""
    try:
        if not name.startswith(("metrics-", "capacity-", "queue-")) and (
            re.fullmatch(
                r"(?:calls-(?:worker|ingest)-\d{3,4}-(?:started|finished)|smoke-(?:started|result)|call-profile|load-(?:manifest|started|result))\.json",
                name,
            )
            is None
        ):
            raise ValueError
        if name.startswith("calls-"):
            if (
                type(record) not in {CallRecord, QueueCallRecord}
                or name != f"calls-{record.process}-{record.sequence:03}-{record.phase}.json"
            ):
                raise ValueError("invalid_call_record")
        if name == "call-profile.json" and type(record) not in {CallProfile, QueueCallProfile}:
            raise ValueError("invalid_call_profile")
        if name.startswith("queue-"):
            from tests.performance.queue_contracts import validate_publication

            validate_publication(name, record)
        if name.startswith("metrics-"):
            from tests.performance.metrics import validate_publication

            validate_publication(name, record)
        if name.startswith("load-"):
            from tests.performance.contracts import Manifest, Result, Started

            expected = {
                "load-manifest.json": Manifest,
                "load-started.json": Started,
                "load-result.json": Result,
            }
            if type(record) is not expected[name]:
                raise ValueError
        if name.startswith("capacity-"):
            from tests.performance.capacity_contracts import validate_publication

            validate_publication(name, record)
        # Validate even model_copy/model_construct callers before creating a file.
        payload = type(record).model_validate_json(record.model_dump_json()).model_dump(mode="json")
        if len(json.dumps(payload, allow_nan=False).encode()) > 32 * 1024 * 1024:
            raise ValueError("artifact_limit")
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, allow_nan=False, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(directory_fd)
    except (OSError, TypeError, ValueError):
        raise EnvironmentError("report_failed") from None
