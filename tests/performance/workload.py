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


def parse_profile(value: object) -> CallProfile:
    try:
        # Round-trip validates tuple fields at the JSON IPC boundary, and rejects non-JSON data.
        return CallProfile.model_validate_json(json.dumps(value, allow_nan=False))
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


def publish(directory: Path, name: str, record: Contract) -> None:
    """No free-form exception, payload, path or target string enters an artifact."""
    try:
        if (
            re.fullmatch(
                r"(?:calls-(?:worker|ingest)-\d{3}-(?:started|finished)|smoke-(?:started|result)|call-profile|load-(?:manifest|started|result))\.json",
                name,
            )
            is None
        ):
            raise ValueError
        if name.startswith("load-"):
            from tests.performance.contracts import Manifest, Result, Started

            expected = {
                "load-manifest.json": Manifest,
                "load-started.json": Started,
                "load-result.json": Result,
            }
            if type(record) is not expected[name]:
                raise ValueError
        # Validate even model_copy/model_construct callers before creating a file.
        payload = type(record).model_validate_json(record.model_dump_json()).model_dump(mode="json")
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
