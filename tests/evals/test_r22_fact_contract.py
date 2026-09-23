"""R2.2 deterministic cases; these do not measure real semantic quality."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.domain.project_facts import (
    CandidateFactV1,
    FactConditionsV1,
    FactEvidenceV1,
    candidate_issues,
)


@pytest.mark.parametrize(
    ("claim", "kind", "environment", "basis", "quote", "required_issue"),
    [
        ("Planned a rollout", "plan", None, None, "Planned a rollout", None),
        (
            "Measured 12 requests",
            "experiment",
            None,
            "local log",
            "Measured 12 requests",
            "experiment_environment_missing",
        ),
        (
            "Measured 120 requests",
            "experiment",
            "local",
            "local log",
            "Measured 12 requests",
            "metric_not_in_evidence",
        ),
        ("I led a release", "implementation", None, None, "release()", "personal_role_unverified"),
        ("Implemented parser", "implementation", None, None, "Implemented parser", None),
    ],
)
def test_synthetic_claims_keep_evidence_and_kind_boundaries(
    claim, kind, environment, basis, quote, required_issue
) -> None:
    candidate = CandidateFactV1(
        claim=claim,
        kind=kind,
        conditions=FactConditionsV1(environment=environment, metric_basis=basis),
        evidence=(FactEvidenceV1(snapshot_file_id=uuid4(), start_line=1, end_line=1, quote=quote),),
    )
    issues = candidate_issues(candidate)
    if required_issue is None:
        assert issues == ()
    else:
        assert required_issue in issues
