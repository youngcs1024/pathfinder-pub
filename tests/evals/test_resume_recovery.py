"""D fixed denominators, fail-closed reports and prerequisite identities."""

from collections import Counter

import pytest

from scripts.ci_evidence import FULL_JOBS
from tests.evals.product_acceptance_contracts import AcceptanceError
from tests.evals.resume_recovery import validate_ci
from tests.evals.resume_recovery_contracts import cases, summarize


def test_d_fixed_matrix_and_no_duplicate_cases():
    planned = cases()
    assert len(planned) == len({c["case_id"] for c in planned}) == 1060
    assert Counter(c["kind"] for c in planned) == {"controlled": 900, "real": 100, "blocking": 60}
    cells = Counter((c["mode"], c["fault"], c["kind"]) for c in planned)
    assert all(n == (90 if kind == "controlled" else 10) for (_, _, kind), n in cells.items())


def test_missing_and_failed_results_keep_fixed_denominators():
    planned = cases()
    result = dict(case=planned[0], status="FAIL", verified=False, injected=True)
    report = summarize([result])
    assert report["planned_recovery"] == 1000 and report["planned_blocking"] == 60
    assert report["recorded"] == 1 and len(report["missing_cases"]) == 1059
    assert report["unverified_cases"] == 1 and report["status"] == "PARTIAL"
    assert report["real_latency"]["samples"] == 0
    with pytest.raises(AcceptanceError, match="invalid_result_identity"):
        summarize([result, result])


@pytest.mark.parametrize(
    "mutation",
    [{"verified": False}, {"injected": False}, {"clock": "injected"}, {"recovery_seconds": 121}],
)
def test_real_pass_requires_verified_fault_and_real_window(mutation):
    case = next(c for c in cases() if c["kind"] == "real")
    result = dict(
        case=case,
        status="PASS",
        verified=True,
        injected=True,
        clock="monotonic",
        recovery_seconds=30,
    )
    assert summarize([result])["real_latency"]["p50_seconds"] == 30
    result.update(mutation)
    with pytest.raises(AcceptanceError):
        summarize([result])


def test_ci_requires_same_commit_all_nine_jobs_and_full_attempt():
    evidence = dict(
        repo="youngcs1024/pathfinder-pub",
        sha="a" * 40,
        category="full_success",
        attempt=1,
        jobs=[dict(name=n, status="completed", conclusion="success") for n in FULL_JOBS],
    )
    validate_ci(evidence, "a" * 40)
    for wrong in (
        {**evidence, "sha": "b" * 40},
        {**evidence, "jobs": evidence["jobs"][:-1]},
        {**evidence, "category": "pending"},
    ):
        with pytest.raises(AcceptanceError, match="full_ci_required"):
            validate_ci(wrong, "a" * 40)
