"""Identity and concurrency rules for accepted resume business commands."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.errors import DomainConflictError, DomainValidationError


class CommandReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    command_id: UUID
    run_id: UUID | None = None
    resource_id: UUID | None = None
    status: Literal["queued", "completed"]


@dataclass(frozen=True, slots=True)
class SessionWritePrecondition:
    expected_session_revision: int
    base_version_id: UUID | None

    def __post_init__(self) -> None:
        if (
            type(self.expected_session_revision) is not int
            or self.expected_session_revision < 0
            or (self.base_version_id is not None and not isinstance(self.base_version_id, UUID))
        ):
            raise DomainValidationError("session write precondition is invalid")


@dataclass(frozen=True, slots=True)
class SessionWriteState:
    revision: int
    current_version_id: UUID | None
    has_active_modification: bool

    def __post_init__(self) -> None:
        if (
            type(self.revision) is not int
            or self.revision < 0
            or (
                self.current_version_id is not None
                and not isinstance(self.current_version_id, UUID)
            )
            or type(self.has_active_modification) is not bool
        ):
            raise DomainValidationError("session write state is invalid")


def require_current_session_write(
    expected: SessionWritePrecondition, current: SessionWriteState
) -> None:
    if (
        expected.expected_session_revision != current.revision
        or expected.base_version_id != current.current_version_id
        or current.has_active_modification
    ):
        raise DomainConflictError("session write conflicts with current state")


@dataclass(frozen=True, slots=True)
class ResumeCommandRequest:
    client_request_id: UUID
    kind: str
    target_id: UUID | None
    payload_version: int
    payload: BaseModel
    session_write: SessionWritePrecondition | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.client_request_id, UUID)
            or self.client_request_id.version != 4
            or not isinstance(self.kind, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.kind) is None
            or (self.target_id is not None and not isinstance(self.target_id, UUID))
            or type(self.payload_version) is not int
            or self.payload_version < 1
            or not isinstance(self.payload, BaseModel)
            or (
                self.session_write is not None
                and not isinstance(self.session_write, SessionWritePrecondition)
            )
        ):
            raise DomainValidationError("resume command request is invalid")

    def digest_v1(self) -> str:
        """Digest the entire typed request, without changing E3's Run v1 digest."""
        try:
            encoded = json.dumps(
                {
                    "command_identity_version": 1,
                    "kind": self.kind,
                    "target_id": str(self.target_id) if self.target_id is not None else None,
                    "payload_version": self.payload_version,
                    "payload": self.payload.model_dump(mode="json", round_trip=True),
                    "expected_session_revision": (
                        self.session_write.expected_session_revision
                        if self.session_write is not None
                        else None
                    ),
                    "base_version_id": (
                        str(self.session_write.base_version_id)
                        if self.session_write is not None
                        and self.session_write.base_version_id is not None
                        else None
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, ValidationError, UnicodeError):
            raise DomainValidationError("resume command request is invalid") from None
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CommandAccepted:
    receipt: CommandReceiptV1
    replayed: bool = False
