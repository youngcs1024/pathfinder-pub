from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.domain.actions import (
    ACTION_KEY,
    SUBMIT_APPLICATION_TOOL_NAME,
    ActionIntentStatus,
    ApprovalBindingV1,
    SubmitApplicationArgsV1,
    approval_binding_digest,
    canonical_json_bytes,
    canonicalize_action_args,
    canonicalize_trusted_target,
    is_valid_action_intent_transition,
    persisted_action_intent_status,
    trusted_action_target,
    validate_action_intent_transition,
)
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainValidationError,
)
from app.domain.tool_effects import ToolEffect

_ACTION_TRANSITIONS = {
    ActionIntentStatus.PROPOSED: {
        ActionIntentStatus.AUTHORIZED,
        ActionIntentStatus.CANCELLED,
    },
    ActionIntentStatus.AUTHORIZED: {
        ActionIntentStatus.EXECUTING,
        ActionIntentStatus.CANCELLED,
    },
    ActionIntentStatus.EXECUTING: {
        ActionIntentStatus.SUCCEEDED,
        ActionIntentStatus.FAILED,
        ActionIntentStatus.OUTCOME_UNKNOWN,
    },
    ActionIntentStatus.SUCCEEDED: set(),
    ActionIntentStatus.FAILED: set(),
    ActionIntentStatus.OUTCOME_UNKNOWN: set(),
    ActionIntentStatus.CANCELLED: set(),
}


@pytest.mark.parametrize("current", tuple(ActionIntentStatus))
@pytest.mark.parametrize("target", tuple(ActionIntentStatus))
def test_action_intent_transition_matrix(
    current: ActionIntentStatus,
    target: ActionIntentStatus,
) -> None:
    expected = target in _ACTION_TRANSITIONS[current]

    assert is_valid_action_intent_transition(current, target) is expected
    if expected:
        validate_action_intent_transition(current, target)
    else:
        with pytest.raises(DomainConflictError):
            validate_action_intent_transition(current, target)


def test_persisted_unknown_action_state_is_an_invariant_failure() -> None:
    with pytest.raises(DomainInvariantError):
        persisted_action_intent_status("superseded")


def _args(
    *,
    resume_document_id: UUID | None = None,
    answers: dict[str, str] | None = None,
    cover_letter: str = "Exact cover letter",
) -> SubmitApplicationArgsV1:
    return SubmitApplicationArgsV1(
        job_ref="job:backend:001",
        resume_document_id=resume_document_id or UUID("AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"),
        answers=answers or {"availability": "Two weeks", "location": "Remote"},
        cover_letter=cover_letter,
    )


def test_action_args_canonicalization_is_stable_across_object_and_nested_mapping_order() -> None:
    first = _args(answers={"availability": "Two weeks", "location": "Remote"})
    second = SubmitApplicationArgsV1(
        cover_letter=first.cover_letter,
        answers={"location": "Remote", "availability": "Two weeks"},
        resume_document_id=first.resume_document_id,
        job_ref=first.job_ref,
    )

    first_snapshot, first_bytes, first_digest = canonicalize_action_args(first)
    second_snapshot, second_bytes, second_digest = canonicalize_action_args(second)

    assert first_snapshot == second_snapshot
    assert first_bytes == second_bytes
    assert first_digest == second_digest
    assert first_bytes == (
        b'{"answers":{"availability":"Two weeks","location":"Remote"},'
        b'"cover_letter":"Exact cover letter","job_ref":"job:backend:001",'
        b'"resume_document_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}'
    )


def test_action_args_preserve_exact_unicode_without_normalization() -> None:
    composed = _args(cover_letter="caf\N{LATIN SMALL LETTER E WITH ACUTE}")
    decomposed = _args(cover_letter="cafe\N{COMBINING ACUTE ACCENT}")

    _snapshot_a, bytes_a, digest_a = canonicalize_action_args(composed)
    _snapshot_b, bytes_b, digest_b = canonicalize_action_args(decomposed)

    assert bytes_a != bytes_b
    assert digest_a != digest_b
    assert "caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode() in bytes_a
    assert "cafe\N{COMBINING ACUTE ACCENT}".encode() in bytes_b


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf"), 1.5])
def test_action_canonicalization_rejects_floats_nan_and_infinity(number: float) -> None:
    with pytest.raises(DomainValidationError):
        canonical_json_bytes({"number": number})


@pytest.mark.parametrize("invalid", [True, 1])
def test_action_payload_does_not_treat_bool_or_int_as_string(invalid: object) -> None:
    with pytest.raises(ValidationError):
        SubmitApplicationArgsV1(
            job_ref="job",
            resume_document_id=uuid4(),
            answers={"question": invalid},
            cover_letter="letter",
        )


def test_action_payload_rejects_extra_fields_and_unbounded_json_tree() -> None:
    payload = _args().model_dump(mode="python") | {"workspace_id": uuid4()}
    with pytest.raises(ValidationError):
        SubmitApplicationArgsV1.model_validate(payload, strict=True)
    with pytest.raises(ValidationError):
        SubmitApplicationArgsV1(
            job_ref="job",
            resume_document_id=uuid4(),
            answers={"question": {"nested": "not allowed"}},
            cover_letter="letter",
        )


def test_uuid_from_json_is_serialized_as_canonical_lowercase() -> None:
    payload = json.dumps(
        {
            "job_ref": "job",
            "resume_document_id": "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
            "answers": {},
            "cover_letter": "letter",
        }
    )
    validated = SubmitApplicationArgsV1.model_validate_json(payload, strict=True)
    snapshot, _canonical, _digest = canonicalize_action_args(validated)

    assert snapshot["resume_document_id"] == "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def test_single_character_business_content_mutation_changes_args_digest() -> None:
    _snapshot_a, _bytes_a, digest_a = canonicalize_action_args(_args())
    _snapshot_b, _bytes_b, digest_b = canonicalize_action_args(
        _args(cover_letter="Exact cover letteS")
    )

    assert digest_a != digest_b


def test_trusted_target_is_stable_fixed_and_has_no_secret_canary() -> None:
    first_snapshot, first_bytes, first_digest = canonicalize_trusted_target()
    _second_snapshot, second_bytes, second_digest = canonicalize_trusted_target()

    assert first_snapshot == {
        "provider": "mock_portal",
        "target_type": "internal_mock",
        "target_ref": "default",
        "resource_scope": "application_submission",
    }
    assert (first_bytes, first_digest) == (second_bytes, second_digest)
    assert "secret-canary" not in first_bytes.decode()
    with pytest.raises(TypeError):
        trusted_action_target(provider="secret-canary")  # type: ignore[call-arg]


def _binding() -> ApprovalBindingV1:
    return ApprovalBindingV1(
        workspace_id=UUID("10000000-0000-4000-8000-000000000001"),
        run_id=UUID("20000000-0000-4000-8000-000000000002"),
        action_intent_id=UUID("30000000-0000-4000-8000-000000000003"),
        action_key=ACTION_KEY,
        action_revision=1,
        tool_name=SUBMIT_APPLICATION_TOOL_NAME,
        effect=ToolEffect.IRREVERSIBLE,
        args_digest=f"sha256:{'a' * 64}",
        target_digest=f"sha256:{'b' * 64}",
        policy_version=1,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("workspace_id", UUID("10000000-0000-4000-8000-000000000009")),
        ("run_id", UUID("20000000-0000-4000-8000-000000000009")),
        ("action_intent_id", UUID("30000000-0000-4000-8000-000000000009")),
        ("action_key", "another_application"),
        ("action_revision", 2),
        ("tool_name", "another_mock_tool"),
        ("effect", ToolEffect.REVERSIBLE),
        ("args_digest", f"sha256:{'c' * 64}"),
        ("target_digest", f"sha256:{'d' * 64}"),
        ("policy_version", 2),
    ],
)
def test_each_approval_binding_field_changes_the_digest(field: str, replacement: object) -> None:
    baseline = _binding()
    _baseline_bytes, baseline_digest = approval_binding_digest(baseline)
    mutated = baseline.model_copy(update={field: replacement})
    _mutated_bytes, mutated_digest = approval_binding_digest(mutated)

    assert mutated_digest != baseline_digest


def test_approval_binding_digest_is_stable_and_domain_prefixed_format() -> None:
    canonical_a, digest_a = approval_binding_digest(_binding())
    canonical_b, digest_b = approval_binding_digest(_binding())

    assert canonical_a == canonical_b
    assert digest_a == digest_b
    assert digest_a.startswith("sha256:") and len(digest_a) == 71
