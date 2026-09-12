from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import ValidationError

from tests.evals.contracts import AcceptedEvalBaselineV1, EvalRegressionReportV1, EvalReportV3
from tests.evals.harness import run_evaluation_sync
from tests.evals.regression import (
    DEFAULT_BASELINE_PATH,
    BaselineLoadError,
    compare_eval_reports,
    load_accepted_baseline,
)


class BaselineRefreshError(Exception):
    def __init__(self, category: str, exit_code: int = 2) -> None:
        self.category = category
        self.exit_code = exit_code
        super().__init__(category)


def prepare_baseline_refresh(
    *,
    reason: str,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    current_report: EvalReportV3 | None = None,
) -> tuple[AcceptedEvalBaselineV1, EvalRegressionReportV1 | None]:
    old_baseline: AcceptedEvalBaselineV1 | None = None
    if baseline_path.exists():
        try:
            old_baseline = load_accepted_baseline(baseline_path)
        except BaselineLoadError as error:
            raise BaselineRefreshError(error.category) from None

    if current_report is None:
        current_report, current_exit_code = run_evaluation_sync()
        if current_exit_code == 2:
            raise BaselineRefreshError("current_eval_configuration_error")
    if current_report.error_category is not None:
        raise BaselineRefreshError("current_eval_configuration_error")
    if not current_report.passed:
        raise BaselineRefreshError("current_eval_quality_failure", 1)

    try:
        accepted = AcceptedEvalBaselineV1(
            acceptance_reason=reason,
            report=current_report,
        )
    except ValidationError:
        raise BaselineRefreshError("invalid_refresh_input") from None

    comparison = (
        compare_eval_reports(old_baseline, current_report) if old_baseline is not None else None
    )
    return accepted, comparison


def atomic_replace_baseline(
    accepted: AcceptedEvalBaselineV1,
    path: Path = DEFAULT_BASELINE_PATH,
    *,
    replace_operation: Callable[[str, Path], None] = os.replace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = accepted.model_dump_json(indent=2) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            errors="strict",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        replace_operation(temporary_name, path)
    except OSError:
        raise BaselineRefreshError("baseline_write_failed") from None


def refresh_baseline(
    *,
    reason: str,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    current_report: EvalReportV3 | None = None,
    replace_operation: Callable[[str, Path], None] = os.replace,
    review_callback: Callable[[EvalRegressionReportV1 | None], None] | None = None,
) -> tuple[AcceptedEvalBaselineV1, EvalRegressionReportV1 | None]:
    accepted, comparison = prepare_baseline_refresh(
        reason=reason,
        baseline_path=baseline_path,
        current_report=current_report,
    )
    if review_callback is not None:
        review_callback(comparison)
    atomic_replace_baseline(
        accepted,
        baseline_path,
        replace_operation=replace_operation,
    )
    return accepted, comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly refresh the accepted deterministic eval baseline."
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Human-reviewed reason for accepting the complete passing result.",
    )
    return parser


def _write_review_summary(comparison: EvalRegressionReportV1 | None) -> None:
    if comparison is None:
        sys.stderr.write("BASELINE REFRESH REVIEW initial_creation\n")
        return
    sys.stderr.write(
        "BASELINE REFRESH REVIEW "
        f"added={len(comparison.added_case_ids)} "
        f"removed={len(comparison.removed_case_ids)} "
        f"versions={len(comparison.version_changes)} "
        f"contracts={len(comparison.contract_regressions)} "
        f"metrics={len(comparison.metric_regressions)}\n"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        accepted, _ = refresh_baseline(
            reason=parsed.reason,
            review_callback=_write_review_summary,
        )
    except BaselineRefreshError as error:
        sys.stderr.write(f"BASELINE REFRESH ERROR {error.category}\n")
        return error.exit_code
    sys.stdout.write(accepted.model_dump_json(indent=2) + "\n")
    sys.stderr.write("BASELINE REFRESH PASS\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
