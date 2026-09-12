import hashlib
import json
from dataclasses import FrozenInstanceError
from uuid import UUID, uuid4

import pytest

from app.domain.errors import DomainValidationError
from app.domain.research import normalize_research_query
from app.domain.runs import RunCreateIdentity, RunMode, create_request_digest_v1

RESUME = UUID("12345678-1234-4234-9234-123456789abc")
QUERY = "AI 工程师 café"
DIGEST = "9321d0f0692ffb9f1776b5c54243cf23d00f83868a393cb8eecaa370d8e8e136"


@pytest.mark.parametrize(
    "query", [QUERY, "  \uff21\uff29\t工程师\u3000cafe\u0301\n", "AI\n工程师  café"]
)
def test_v1_normalization_has_frozen_unicode_vector(query: str) -> None:
    assert normalize_research_query(query) == QUERY
    assert create_request_digest_v1(mode=RunMode.RESEARCH, query=query) == DIGEST


def test_v1_application_vector_and_resume_changes() -> None:
    application = create_request_digest_v1(
        mode=RunMode.APPLICATION, query=QUERY, resume_document_id=RESUME
    )
    assert application == "23e1d3a8d547d3e693097e400c0164c470581c08aba3021ba10a21cc9c002805"
    research_with_resume = create_request_digest_v1(
        mode=RunMode.RESEARCH, query=QUERY, resume_document_id=RESUME
    )
    other_resume = create_request_digest_v1(
        mode=RunMode.APPLICATION, query=QUERY, resume_document_id=uuid4()
    )
    assert len({DIGEST, application, research_with_resume, other_resume}) == 4
    assert create_request_digest_v1(mode=RunMode.RESEARCH, query="another query") != DIGEST


def test_digest_contains_only_declared_fields_with_order_independent_json() -> None:
    fields = {"version": 1, "resume_document_id": None, "query": QUERY, "mode": "research"}
    for payload in (fields, dict(reversed(list(fields.items())))):
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == DIGEST
    assert create_request_digest_v1(**{"query": QUERY, "mode": RunMode.RESEARCH}) == DIGEST
    assert create_request_digest_v1(**{"mode": RunMode.RESEARCH, "query": QUERY}) == DIGEST


@pytest.mark.parametrize(
    "name",
    [
        "run_id",
        "client_request_id",
        "workspace_id",
        "actor_user_id",
        "created_at",
        "graph_version",
        "model",
        "limits",
        "include_application_draft",
        "version",
        "secret",
    ],
)
def test_digest_rejects_undeclared_input(name: str) -> None:
    with pytest.raises(TypeError):
        create_request_digest_v1(mode=RunMode.RESEARCH, query=QUERY, **{name: "CANARY"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "research"},
        {"mode": RunMode.APPLICATION},
        {"resume_document_id": str(RESUME)},
        {"query": None},
        {"query": 123},
        {"query": ""},
        {"query": " \t\u3000"},
        {"query": "CANARY" * 400},
        {"query": "CANARY\ud800"},
    ],
)
def test_invalid_content_has_safe_error(overrides: dict[str, object]) -> None:
    fields = {"mode": RunMode.RESEARCH, "query": QUERY, **overrides}
    with pytest.raises(DomainValidationError) as caught:
        create_request_digest_v1(**fields)
    assert str(caught.value) == "run creation input is invalid"
    assert "CANARY" not in str(caught.value)


def test_identity_is_immutable_and_key_does_not_change_digest() -> None:
    first = RunCreateIdentity(uuid4(), DIGEST)
    second = RunCreateIdentity(uuid4(), DIGEST)
    assert first.client_request_id != second.client_request_id
    assert first.create_request_digest == second.create_request_digest
    assert first.create_request_version == 1
    with pytest.raises(FrozenInstanceError):
        first.create_request_digest = "0" * 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"client_request_id": str(RESUME)},
        {"client_request_id": UUID("12345678-1234-1234-9234-123456789abc")},
        {"create_request_digest": DIGEST.upper()},
        {"create_request_digest": "CANARY"},
        {"create_request_digest": "a" * 63},
        {"create_request_digest": "a" * 65},
        {"create_request_digest": "g" * 64},
        {"create_request_digest": None},
        {"create_request_version": True},
        {"create_request_version": 1.0},
        {"create_request_version": 2},
    ],
)
def test_invalid_identity_is_rejected(overrides: dict[str, object]) -> None:
    fields = {"client_request_id": RESUME, "create_request_digest": DIGEST, **overrides}
    with pytest.raises(DomainValidationError) as caught:
        RunCreateIdentity(**fields)
    assert str(caught.value) == "run creation identity is invalid"
