from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from tests.evals.contracts import (
    EVAL_GRADER_NAMES,
    AcceptedEvalBaselineV1,
    EvalArtifactIdentityV1,
    EvalContractRegressionV1,
    EvalMetricRegressionV1,
    EvalRegressionConfigurationErrorCategory,
    EvalRegressionMetric,
    EvalRegressionReportV1,
    EvalReportV3,
    EvalVersionChangeV1,
    EvalVersionMetadataV1,
)
from tests.evals.harness import run_evaluation_sync

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "evals" / "baselines" / "research_v3.json"

_ARTIFACT_FIELDS = (
    "dataset_digest",
    "case_set_digest",
    "graph_version",
    "embedding_profile",
    "grader_contract_version",
    "grader_contract_digest",
)
_VERSION_FIELDS = (
    "chat_model",
    "embedding_model",
    "embedding_dimension",
    "reasoning_effort",
    "plan_prompt_version",
    "research_prompt_version",
    "writer_prompt_version",
    "research_tool_policy",
    "research_tool_schema_version",
    "research_output_schema_version",
    "pricing_version",
)
_LOWER_IS_BETTER: tuple[EvalRegressionMetric, ...] = (
    "input_tokens",
    "output_tokens",
    "model_call_count",
    "total_tool_call_count",
    "irrelevant_context_count",
)


class BaselineLoadError(Exception):
    def __init__(self, category: EvalRegressionConfigurationErrorCategory) -> None:
        self.category = category
        super().__init__(category)


def load_accepted_baseline(path: Path = DEFAULT_BASELINE_PATH) -> AcceptedEvalBaselineV1:
    try:
        serialized = path.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        raise BaselineLoadError("baseline_missing") from None
    except (OSError, UnicodeError):
        raise BaselineLoadError("baseline_unreadable") from None
    try:
        json.loads(serialized)
    except (json.JSONDecodeError, UnicodeError):
        raise BaselineLoadError("baseline_malformed_json") from None
    try:
        return AcceptedEvalBaselineV1.model_validate_json(serialized, strict=True)
    except ValidationError:
        raise BaselineLoadError("baseline_schema_incompatible") from None


def _version_changes(
    baseline_identity: EvalArtifactIdentityV1,
    current_identity: EvalArtifactIdentityV1,
    baseline_versions: EvalVersionMetadataV1,
    current_versions: EvalVersionMetadataV1,
) -> tuple[EvalVersionChangeV1, ...]:
    changes: list[EvalVersionChangeV1] = []
    for field in _ARTIFACT_FIELDS:
        baseline_value = getattr(baseline_identity, field)
        current_value = getattr(current_identity, field)
        if baseline_value != current_value:
            changes.append(
                EvalVersionChangeV1(
                    field=field,
                    baseline_value=baseline_value,
                    current_value=current_value,
                )
            )
    for field in _VERSION_FIELDS:
        baseline_value = getattr(baseline_versions, field)
        current_value = getattr(current_versions, field)
        if baseline_value != current_value:
            changes.append(
                EvalVersionChangeV1(
                    field=field,
                    baseline_value=baseline_value,
                    current_value=current_value,
                )
            )
    return tuple(changes)


def _contract_regressions(
    baseline: EvalReportV3,
    current: EvalReportV3,
) -> tuple[EvalContractRegressionV1, ...]:
    baseline_cases = {case.case_id: case for case in baseline.cases}
    current_cases = {case.case_id: case for case in current.cases}
    regressions: list[EvalContractRegressionV1] = []
    for case_id in sorted(baseline_cases.keys() & current_cases.keys()):
        baseline_case = baseline_cases[case_id]
        current_case = current_cases[case_id]
        if baseline_case.error_category is None and current_case.error_category is not None:
            regressions.append(
                EvalContractRegressionV1(
                    case_id=case_id,
                    kind="case_execution",
                    baseline_status="PASS",
                    current_status="FAIL",
                )
            )
        baseline_graders = {grader.name: grader for grader in baseline_case.graders}
        current_graders = {grader.name: grader for grader in current_case.graders}
        for grader_name in EVAL_GRADER_NAMES:
            if baseline_graders[grader_name].passed and not current_graders[grader_name].passed:
                regressions.append(
                    EvalContractRegressionV1(
                        case_id=case_id,
                        kind="grader",
                        grader=grader_name,
                        baseline_status="PASS",
                        current_status="FAIL",
                    )
                )
    return tuple(regressions)


def _metric_regressions(
    baseline: EvalReportV3,
    current: EvalReportV3,
) -> tuple[EvalMetricRegressionV1, ...]:
    baseline_cases = {case.case_id: case for case in baseline.cases}
    current_cases = {case.case_id: case for case in current.cases}
    regressions: list[EvalMetricRegressionV1] = []
    for case_id in sorted(baseline_cases.keys() & current_cases.keys()):
        baseline_metrics = baseline_cases[case_id].metrics
        current_metrics = current_cases[case_id].metrics
        for metric in _LOWER_IS_BETTER:
            baseline_value = getattr(baseline_metrics, metric)
            current_value = getattr(current_metrics, metric)
            if current_value > baseline_value:
                regressions.append(
                    EvalMetricRegressionV1(
                        case_id=case_id,
                        metric=metric,
                        direction="lower_is_better",
                        baseline_value=baseline_value,
                        current_value=current_value,
                    )
                )
        baseline_recall = baseline_metrics.recall_at_5
        current_recall = current_metrics.recall_at_5
        if baseline_recall is not None and (
            current_recall is None or current_recall < baseline_recall
        ):
            regressions.append(
                EvalMetricRegressionV1(
                    case_id=case_id,
                    metric="recall_at_5",
                    direction="higher_is_better",
                    baseline_value=baseline_recall,
                    current_value=current_recall,
                )
            )
    return tuple(regressions)


def compare_eval_reports(
    accepted: AcceptedEvalBaselineV1,
    current: EvalReportV3,
) -> EvalRegressionReportV1:
    baseline = accepted.report
    if (
        current.error_category is not None
        or current.selected_case is not None
        or current.comparison_metadata is None
        or current.comparison_metadata.scope != "full_dataset"
        or current.artifact_identity is None
        or current.version_metadata is None
    ):
        return EvalRegressionReportV1(
            passed=False,
            error_category="current_eval_configuration_error",
        )

    baseline_identity = baseline.artifact_identity
    baseline_versions = baseline.version_metadata
    baseline_comparison = baseline.comparison_metadata
    if baseline_identity is None or baseline_versions is None or baseline_comparison is None:
        raise ValueError("strict accepted baseline lost required comparison evidence")

    baseline_case_ids = set(baseline_comparison.dataset_case_ids)
    current_case_ids = set(current.comparison_metadata.dataset_case_ids)
    added_case_ids = tuple(sorted(current_case_ids - baseline_case_ids))
    removed_case_ids = tuple(sorted(baseline_case_ids - current_case_ids))
    version_changes = _version_changes(
        baseline_identity,
        current.artifact_identity,
        baseline_versions,
        current.version_metadata,
    )
    contract_regressions = _contract_regressions(baseline, current)
    metric_regressions = _metric_regressions(baseline, current)
    current_quality_failure = not current.passed
    passed = (
        not any(
            (
                added_case_ids,
                removed_case_ids,
                version_changes,
                contract_regressions,
                metric_regressions,
            )
        )
        and not current_quality_failure
    )
    return EvalRegressionReportV1(
        passed=passed,
        baseline_artifact_identity=baseline_identity,
        current_artifact_identity=current.artifact_identity,
        baseline_version_metadata=baseline_versions,
        current_version_metadata=current.version_metadata,
        added_case_ids=added_case_ids,
        removed_case_ids=removed_case_ids,
        version_changes=version_changes,
        contract_regressions=contract_regressions,
        metric_regressions=metric_regressions,
        current_quality_failure=current_quality_failure,
    )


def run_regression(
    *,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    current_report: EvalReportV3 | None = None,
) -> tuple[EvalRegressionReportV1, Literal[0, 1, 2]]:
    try:
        accepted = load_accepted_baseline(baseline_path)
    except BaselineLoadError as error:
        return EvalRegressionReportV1(passed=False, error_category=error.category), 2

    if current_report is None:
        current_report, current_exit_code = run_evaluation_sync()
        if current_exit_code == 2:
            return (
                EvalRegressionReportV1(
                    passed=False,
                    error_category="current_eval_configuration_error",
                ),
                2,
            )
    comparison = compare_eval_reports(accepted, current_report)
    if comparison.error_category is not None:
        return comparison, 2
    return comparison, 0 if comparison.passed else 1


def main() -> int:
    report, exit_code = run_regression()
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    if report.error_category is not None:
        sys.stderr.write(f"EVAL REGRESSION CONFIGURATION_ERROR {report.error_category}\n")
    else:
        status = "PASS" if report.passed else "FAIL"
        sys.stderr.write(
            f"EVAL REGRESSION {status} "
            f"added={len(report.added_case_ids)} removed={len(report.removed_case_ids)} "
            f"versions={len(report.version_changes)} "
            f"contracts={len(report.contract_regressions)} "
            f"metrics={len(report.metric_regressions)}\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
