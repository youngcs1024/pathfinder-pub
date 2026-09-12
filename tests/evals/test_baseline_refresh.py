from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evals.baseline_refresh import (
    BaselineRefreshError,
    prepare_baseline_refresh,
    refresh_baseline,
)
from tests.evals.contracts import AcceptedEvalBaselineV1, EvalReportV3
from tests.evals.harness import TRUSTED_CONTEXT_CANARY, run_evaluation_sync
from tests.evals.regression import load_accepted_baseline, run_regression


@pytest.fixture(scope="module")
def full_report() -> EvalReportV3:
    report, exit_code = run_evaluation_sync()
    assert exit_code == 0
    return report


def test_missing_reason_is_refused(tmp_path: Path, full_report: EvalReportV3) -> None:
    with pytest.raises(BaselineRefreshError, match="invalid_refresh_input"):
        prepare_baseline_refresh(
            reason="   ",
            baseline_path=tmp_path / "baseline.json",
            current_report=full_report,
        )


def test_current_quality_failure_is_refused(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    failed_report = full_report.model_copy(update={"passed": False})

    with pytest.raises(BaselineRefreshError, match="current_eval_quality_failure") as caught:
        prepare_baseline_refresh(
            reason="reviewed",
            baseline_path=tmp_path / "baseline.json",
            current_report=failed_report,
        )

    assert caught.value.exit_code == 1


def test_missing_baseline_allows_initial_creation_and_strict_round_trip(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "nested" / "baseline.json"

    accepted, comparison = refresh_baseline(
        reason="Initial reviewed baseline.",
        baseline_path=baseline_path,
        current_report=full_report,
    )

    assert comparison is None
    assert load_accepted_baseline(baseline_path) == accepted
    assert baseline_path.read_bytes().endswith(b"\n")


def test_valid_existing_baseline_allows_explicit_refresh(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    refresh_baseline(
        reason="Initial reviewed baseline.",
        baseline_path=baseline_path,
        current_report=full_report,
    )

    accepted, comparison = refresh_baseline(
        reason="Second explicit review.",
        baseline_path=baseline_path,
        current_report=full_report,
    )

    assert comparison is not None
    assert comparison.passed is True
    assert accepted.acceptance_reason == "Second explicit review."
    assert load_accepted_baseline(baseline_path) == accepted


def test_planned_version_change_can_be_explicitly_accepted_after_review(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    refresh_baseline(
        reason="Initial reviewed baseline.",
        baseline_path=baseline_path,
        current_report=full_report,
    )
    old_bytes = baseline_path.read_bytes()
    payload = full_report.model_dump(mode="json", round_trip=True)
    payload["artifact_identity"]["graph_version"] = "pathfinder-research-v999"
    changed_report = EvalReportV3.model_validate_json(json.dumps(payload), strict=True)
    review_observations: list[bytes] = []

    accepted, comparison = refresh_baseline(
        reason="Reviewed planned graph contract change.",
        baseline_path=baseline_path,
        current_report=changed_report,
        review_callback=lambda _comparison: review_observations.append(baseline_path.read_bytes()),
    )

    assert comparison is not None
    assert comparison.passed is False
    assert tuple(change.field for change in comparison.version_changes) == ("graph_version",)
    assert review_observations == [old_bytes]
    assert load_accepted_baseline(baseline_path) == accepted


def test_malformed_existing_baseline_is_never_overwritten(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    old_bytes = b"{malformed accepted evidence\n"
    baseline_path.write_bytes(old_bytes)

    with pytest.raises(BaselineRefreshError, match="baseline_malformed_json"):
        refresh_baseline(
            reason="must not overwrite",
            baseline_path=baseline_path,
            current_report=full_report,
        )

    assert baseline_path.read_bytes() == old_bytes


def test_ordinary_eval_and_regression_are_read_only_for_baseline(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    refresh_baseline(
        reason="Read-only boundary test.",
        baseline_path=baseline_path,
        current_report=full_report,
    )
    before = baseline_path.read_bytes()

    ordinary_report, ordinary_exit = run_evaluation_sync()
    after_ordinary = baseline_path.read_bytes()
    regression_report, regression_exit = run_regression(
        baseline_path=baseline_path,
        current_report=ordinary_report,
    )

    assert ordinary_exit == regression_exit == 0
    assert regression_report.passed is True
    assert after_ordinary == before
    assert baseline_path.read_bytes() == before


def test_injected_replace_failure_preserves_exact_old_baseline_bytes(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    old = AcceptedEvalBaselineV1(
        acceptance_reason="Known old evidence.",
        report=full_report,
    )
    old_bytes = (old.model_dump_json(indent=2) + "\n").encode()
    baseline_path.write_bytes(old_bytes)

    def fail_replace(_source: str, _destination: Path) -> None:
        raise OSError("injected replace failure")

    with pytest.raises(BaselineRefreshError, match="baseline_write_failed"):
        refresh_baseline(
            reason="New reviewed evidence.",
            baseline_path=baseline_path,
            current_report=full_report,
            replace_operation=fail_replace,
        )

    assert baseline_path.read_bytes() == old_bytes


def test_accepted_baseline_excludes_canaries_and_runtime_fields(
    full_report: EvalReportV3,
) -> None:
    accepted = AcceptedEvalBaselineV1(
        acceptance_reason="Canary-reviewed evidence.",
        report=full_report,
    )
    serialized = accepted.model_dump_json()
    payload = json.loads(serialized)
    forbidden = {
        "generated_at",
        "timestamp",
        "hostname",
        "baseline_path",
        "temporary_path",
        "process_id",
        "runtime_invocation_id",
        "duration",
        "duration_seconds",
    }

    def fields(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {item for nested in value.values() for item in fields(nested)}
        if isinstance(value, list):
            return {item for nested in value for item in fields(nested)}
        return set()

    assert TRUSTED_CONTEXT_CANARY not in serialized
    assert "pathfinder-credential-canary" not in serialized
    assert fields(payload).isdisjoint(forbidden)
