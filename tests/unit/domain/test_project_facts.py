from __future__ import annotations

from uuid import uuid4

import pytest

from app.domain.project_facts import (
    CandidateFactV1,
    FactConditionsV1,
    FactEvidenceError,
    FactEvidenceV1,
    candidate_issues,
    check_evidence,
)


def test_fixed_snapshot_quote_and_line_range_must_match() -> None:
    file_id = uuid4()
    allowed = {file_id: b"planned release\nmeasured 12 requests in a local test\n"}
    valid = FactEvidenceV1(
        snapshot_file_id=file_id,
        start_line=2,
        end_line=2,
        quote="measured 12 requests in a local test",
    )
    check_evidence(valid, allowed)
    for invalid in (
        valid.model_copy(update={"snapshot_file_id": uuid4()}),
        valid.model_copy(update={"start_line": 3, "end_line": 3}),
        valid.model_copy(update={"quote": "measured 120 requests"}),
    ):
        with pytest.raises(FactEvidenceError):
            check_evidence(invalid, allowed)


def test_classification_preserves_personal_role_and_metric_questions() -> None:
    evidence = FactEvidenceV1(
        snapshot_file_id=uuid4(),
        start_line=1,
        end_line=1,
        quote="Local test measured 12 requests",
    )
    unsupported = CandidateFactV1(
        claim="I led a production rollout serving 120 requests",
        kind="implementation",
        evidence=(evidence,),
    )
    assert set(candidate_issues(unsupported)) == {
        "metric_basis_missing",
        "metric_not_in_evidence",
        "personal_role_unverified",
    }
    experiment = CandidateFactV1(
        claim="Local test measured 12 requests",
        kind="experiment",
        conditions=FactConditionsV1(environment="local", metric_basis="test log"),
        evidence=(evidence,),
    )
    assert candidate_issues(experiment) == ()
    personal = CandidateFactV1(claim="I coordinated the project", kind="personal_statement")
    assert candidate_issues(personal) == ("evidence_missing", "personal_role_requires_attestation")
