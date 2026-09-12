from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from tests.evals.retrieval_contracts import (
    RetrievalBenchmarkReportV1,
    RetrievalHardInvariantRegressionV1,
    RetrievalIdentityDifferenceV1,
    RetrievalMetricRegressionV1,
    RetrievalRankingRegressionV2,
    RetrievalRegressionConfigurationErrorCategory,
    RetrievalRegressionPolicyV2,
    RetrievalRegressionReportV2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_POLICY_PATH = PROJECT_ROOT / "evals" / "baselines" / "retrieval_v1_policy.json"

_IDENTITY_FIELDS = (
    "dataset_version",
    "dataset_digest",
    "case_set_digest",
    "normalization_version",
    "chunking_version",
    "embedding_profile",
    "embedding_dimension",
    "retrieval_top_k",
)
_HIGHER_IS_BETTER = (
    ("macro_recall_at_1", "minimum_macro_recall_at_1"),
    ("macro_recall_at_3", "minimum_macro_recall_at_3"),
    ("macro_recall_at_5", "minimum_macro_recall_at_5"),
    ("mean_reciprocal_rank", "minimum_mean_reciprocal_rank"),
)


class RetrievalPolicyLoadError(Exception):
    def __init__(self, category: RetrievalRegressionConfigurationErrorCategory) -> None:
        self.category = category
        super().__init__(category)


def load_retrieval_policy(
    path: Path = DEFAULT_RETRIEVAL_POLICY_PATH,
) -> RetrievalRegressionPolicyV2:
    try:
        serialized = path.read_text(encoding="utf-8", errors="strict")
    except FileNotFoundError:
        raise RetrievalPolicyLoadError("policy_missing") from None
    except (OSError, UnicodeError):
        raise RetrievalPolicyLoadError("policy_unreadable") from None
    try:
        json.loads(serialized)
    except (json.JSONDecodeError, UnicodeError):
        raise RetrievalPolicyLoadError("policy_malformed_json") from None
    try:
        return RetrievalRegressionPolicyV2.model_validate_json(serialized, strict=True)
    except ValidationError:
        raise RetrievalPolicyLoadError("policy_schema_incompatible") from None


def load_current_benchmark(path: Path) -> RetrievalBenchmarkReportV1:
    try:
        serialized = path.read_text(encoding="utf-8", errors="strict")
        json.loads(serialized)
        return RetrievalBenchmarkReportV1.model_validate_json(serialized, strict=True)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        raise RetrievalPolicyLoadError("current_benchmark_configuration_error") from None


def compare_retrieval_report(
    policy: RetrievalRegressionPolicyV2,
    current: RetrievalBenchmarkReportV1,
) -> RetrievalRegressionReportV2:
    identity_differences: list[RetrievalIdentityDifferenceV1] = []
    for field in _IDENTITY_FIELDS:
        expected_value = getattr(policy.expected_identity, field)
        current_value = getattr(current, field)
        if expected_value != current_value:
            identity_differences.append(
                RetrievalIdentityDifferenceV1(
                    field=field,
                    expected_value=expected_value,
                    current_value=current_value,
                )
            )
    for field in ("evidence_scope", "semantic_quality_claim"):
        expected_value = getattr(policy, field)
        current_value = getattr(current, field)
        if expected_value != current_value:
            identity_differences.append(
                RetrievalIdentityDifferenceV1(
                    field=field,
                    expected_value=expected_value,
                    current_value=current_value,
                )
            )

    hard_regressions: list[RetrievalHardInvariantRegressionV1] = []
    for invariant in ("workspace_leakage_count", "allowlist_leakage_count"):
        expected_value = getattr(policy.hard_invariants, invariant)
        current_value = getattr(current.aggregate, invariant)
        if current_value != expected_value:
            hard_regressions.append(
                RetrievalHardInvariantRegressionV1(
                    invariant=invariant,
                    expected_value=expected_value,
                    current_value=current_value,
                )
            )
    if not current.passed:
        hard_regressions.append(
            RetrievalHardInvariantRegressionV1(
                invariant="current_benchmark_passed",
                expected_value=True,
                current_value=False,
            )
        )

    metric_regressions: list[RetrievalMetricRegressionV1] = []
    for metric, threshold_field in _HIGHER_IS_BETTER:
        threshold = getattr(policy.thresholds, threshold_field)
        current_value = getattr(current.aggregate, metric)
        if current_value < threshold:
            metric_regressions.append(
                RetrievalMetricRegressionV1(
                    metric=metric,
                    direction="higher_is_better",
                    threshold_value=threshold,
                    current_value=current_value,
                )
            )
    irrelevant_threshold = policy.thresholds.maximum_total_irrelevant_context_count
    irrelevant_current = current.aggregate.total_irrelevant_context_count
    if irrelevant_current > irrelevant_threshold:
        metric_regressions.append(
            RetrievalMetricRegressionV1(
                metric="total_irrelevant_context_count",
                direction="lower_is_better",
                threshold_value=irrelevant_threshold,
                current_value=irrelevant_current,
            )
        )

    expected_by_case = {
        item.case_id: item.ranked_retrieved_chunks for item in policy.expected_rankings
    }
    current_by_case = {item.case_id: item.ranked_retrieved_chunks for item in current.cases}
    ranking_regressions: list[RetrievalRankingRegressionV2] = []
    for case_id in sorted(expected_by_case.keys() - current_by_case.keys()):
        ranking_regressions.append(
            RetrievalRankingRegressionV2(
                case_id=case_id,
                regression="missing_case",
                expected_ranking=expected_by_case[case_id],
            )
        )
    for case_id in sorted(current_by_case.keys() - expected_by_case.keys()):
        ranking_regressions.append(
            RetrievalRankingRegressionV2(
                case_id=case_id,
                regression="extra_case",
                current_ranking=current_by_case[case_id],
            )
        )
    for case_id in sorted(expected_by_case.keys() & current_by_case.keys()):
        expected_ranking = expected_by_case[case_id]
        current_ranking = current_by_case[case_id]
        if current_ranking != expected_ranking:
            ranking_regressions.append(
                RetrievalRankingRegressionV2(
                    case_id=case_id,
                    regression="ranking_tuple_drift",
                    expected_ranking=expected_ranking,
                    current_ranking=current_ranking,
                )
            )

    passed = not any(
        (identity_differences, hard_regressions, metric_regressions, ranking_regressions)
    )
    return RetrievalRegressionReportV2(
        policy_version=policy.policy_version,
        passed=passed,
        identity_differences=tuple(identity_differences),
        hard_invariant_regressions=tuple(hard_regressions),
        metric_regressions=tuple(metric_regressions),
        ranking_regressions=tuple(ranking_regressions),
    )


def run_retrieval_regression(
    *,
    policy_path: Path = DEFAULT_RETRIEVAL_POLICY_PATH,
    current_report: RetrievalBenchmarkReportV1 | None = None,
    current_report_path: Path | None = None,
) -> tuple[RetrievalRegressionReportV2, Literal[0, 1, 2]]:
    try:
        policy = load_retrieval_policy(policy_path)
        if current_report is None:
            if current_report_path is None:
                raise RetrievalPolicyLoadError("current_benchmark_configuration_error")
            current_report = load_current_benchmark(current_report_path)
    except RetrievalPolicyLoadError as error:
        return RetrievalRegressionReportV2(passed=False, error_category=error.category), 2
    comparison = compare_retrieval_report(policy, current_report)
    return comparison, 0 if comparison.passed else 1


def main() -> int:
    if len(sys.argv) != 2:
        report = RetrievalRegressionReportV2(
            passed=False,
            error_category="current_benchmark_configuration_error",
        )
        exit_code = 2
    else:
        report, exit_code = run_retrieval_regression(current_report_path=Path(sys.argv[1]))
    sys.stdout.write(report.model_dump_json(indent=2) + "\n")
    if report.error_category is not None:
        sys.stderr.write(f"RETRIEVAL REGRESSION CONFIGURATION_ERROR {report.error_category}\n")
    else:
        status = "PASS" if report.passed else "FAIL"
        sys.stderr.write(
            f"RETRIEVAL REGRESSION {status} "
            f"identity={len(report.identity_differences)} "
            f"hard={len(report.hard_invariant_regressions)} "
            f"metrics={len(report.metric_regressions)} "
            f"rankings={len(report.ranking_regressions)}\n"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
