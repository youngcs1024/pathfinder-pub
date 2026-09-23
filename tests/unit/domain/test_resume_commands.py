"""Resume command identity and stale-write rules stay independent of E3 v1."""

from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from app.domain.errors import DomainConflictError, DomainValidationError
from app.domain.resume_commands import (
    ResumeCommandRequest,
    SessionWritePrecondition,
    SessionWriteState,
    require_current_session_write,
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    selected: tuple[str, ...]


def _request(**changes):
    values = {
        "client_request_id": uuid4(),
        "kind": "resume_revision",
        "target_id": uuid4(),
        "payload_version": 1,
        "payload": _Payload(text="Synthetic draft", selected=("a", "b")),
        "session_write": SessionWritePrecondition(2, uuid4()),
    }
    values.update(changes)
    return ResumeCommandRequest(**values)


def test_digest_covers_every_write_affecting_field_and_preserves_e3() -> None:
    request = _request()
    digest = request.digest_v1()
    assert len(digest) == 64
    assert replace(request, client_request_id=uuid4()).digest_v1() == digest
    for changed in (
        {"kind": "resume_generation"},
        {"target_id": uuid4()},
        {"payload_version": 2},
        {"payload": _Payload(text="Changed", selected=("a", "b"))},
        {"payload": _Payload(text="Synthetic draft", selected=("b", "a"))},
        {"session_write": SessionWritePrecondition(3, request.session_write.base_version_id)},
        {"session_write": SessionWritePrecondition(2, uuid4())},
    ):
        assert replace(request, **changed).digest_v1() != digest


def test_session_write_checks_revision_base_and_active_task() -> None:
    base = uuid4()
    expected = SessionWritePrecondition(4, base)
    require_current_session_write(expected, SessionWriteState(4, base, False))
    for actual in (
        SessionWriteState(5, base, False),
        SessionWriteState(4, uuid4(), False),
        SessionWriteState(4, base, True),
    ):
        with pytest.raises(DomainConflictError):
            require_current_session_write(expected, actual)


def test_invalid_identity_and_precondition_fail_safely() -> None:
    with pytest.raises(DomainValidationError):
        _request(kind="Invalid Kind")
    with pytest.raises(DomainValidationError):
        _request(payload={"text": "unvalidated"})
    with pytest.raises(DomainValidationError):
        SessionWritePrecondition(True, None)
