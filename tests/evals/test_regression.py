from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.evals.contracts import (
    EVAL_GRADER_NAMES,
    AcceptedEvalBaselineV1,
    EvalRegressionReportV1,
    EvalReportV3,
)
from tests.evals.harness import TRUSTED_CONTEXT_CANARY, run_evaluation_sync
from tests.evals.regression import DEFAULT_BASELINE_PATH, compare_eval_reports, run_regression

_CHANGED_DIGEST = "sha256:" + "f" * 64


@pytest.fixture(scope="module")
def full_report() -> EvalReportV3:
    report, exit_code = run_evaluation_sync()
    assert exit_code == 0
    return report


@pytest.fixture(scope="module")
def accepted(full_report: EvalReportV3) -> AcceptedEvalBaselineV1:
    return AcceptedEvalBaselineV1(
        acceptance_reason="Reviewed deterministic baseline.",
        report=full_report,
    )


def _payload(report: EvalReportV3) -> dict[str, Any]:
    return report.model_dump(mode="json", round_trip=True)


def _validated(payload: dict[str, Any]) -> EvalReportV3:
    return EvalReportV3.model_validate_json(json.dumps(payload), strict=True)


def _write_baseline(path: Path, accepted: AcceptedEvalBaselineV1) -> bytes:
    serialized = (accepted.model_dump_json(indent=2) + "\n").encode()
    path.write_bytes(serialized)
    return serialized


def _change_version(report: EvalReportV3, field: str, value: str) -> EvalReportV3:
    payload = _payload(report)
    target = (
        payload["artifact_identity"]
        if field
        in {
            "dataset_digest",
            "case_set_digest",
            "graph_version",
            "embedding_profile",
            "grader_contract_version",
            "grader_contract_digest",
        }
        else payload["version_metadata"]
    )
    target[field] = value
    return _validated(payload)


def _change_coverage(report: EvalReportV3, *, add: bool) -> EvalReportV3:
    payload = _payload(report)
    if add:
        changed_case = deepcopy(payload["cases"][0])
        changed_case["case_id"] = "zz_added_case"
        payload["cases"].append(changed_case)
        for aggregate in payload["grader_aggregates"]:
            aggregate["passed_case_count"] += 1
        payload["input_tokens"] += changed_case["metrics"]["input_tokens"]
        payload["output_tokens"] += changed_case["metrics"]["output_tokens"]
    else:
        changed_case = payload["cases"].pop()
        for aggregate in payload["grader_aggregates"]:
            aggregate["passed_case_count"] -= 1
        payload["input_tokens"] -= changed_case["metrics"]["input_tokens"]
        payload["output_tokens"] -= changed_case["metrics"]["output_tokens"]
    case_ids = sorted(case["case_id"] for case in payload["cases"])
    payload["comparison_metadata"]["dataset_case_ids"] = case_ids
    payload["comparison_metadata"]["evaluated_case_ids"] = case_ids
    payload["artifact_identity"]["dataset_digest"] = _CHANGED_DIGEST
    payload["artifact_identity"]["case_set_digest"] = _CHANGED_DIGEST
    return _validated(payload)


def _fail_grader(report: EvalReportV3, grader_name: str) -> EvalReportV3:
    payload = _payload(report)
    case = payload["cases"][0]
    grader = next(item for item in case["graders"] if item["name"] == grader_name)
    grader.update(passed=False, failure_count=1)
    case["passed"] = False
    payload["passed"] = False
    aggregate = next(item for item in payload["grader_aggregates"] if item["name"] == grader_name)
    aggregate["passed_case_count"] -= 1
    aggregate["failed_case_count"] += 1
    aggregate["failure_count"] += 1
    return _validated(payload)


def _fail_execution(report: EvalReportV3) -> EvalReportV3:
    payload = _payload(report)
    case = payload["cases"][0]
    case.update(passed=False, output=None, error_category="graph_execution_failed")
    for grader in case["graders"]:
        grader.update(passed=False, failure_count=1)
    for aggregate in payload["grader_aggregates"]:
        aggregate["passed_case_count"] -= 1
        aggregate["failed_case_count"] += 1
        aggregate["failure_count"] += 1
    payload["passed"] = False
    return _validated(payload)


def _change_metric(
    report: EvalReportV3,
    metric: str,
    value: int | float | None,
    *,
    case_index: int = 0,
) -> EvalReportV3:
    payload = _payload(report)
    metrics = payload["cases"][case_index]["metrics"]
    old_value = metrics[metric]
    metrics[metric] = value
    if metric in {"input_tokens", "output_tokens"}:
        payload[metric] += value - old_value
    return _validated(payload)


def test_passing_full_report_can_be_accepted_and_strictly_round_tripped(
    accepted: AcceptedEvalBaselineV1,
) -> None:
    serialized = accepted.model_dump_json()

    assert accepted.baseline_schema_version == 1
    assert AcceptedEvalBaselineV1.model_validate_json(serialized, strict=True) == accepted


def test_selected_or_failed_report_cannot_be_accepted(full_report: EvalReportV3) -> None:
    selected_payload = _payload(full_report)
    selected_payload["selected_case"] = selected_payload["cases"][0]["case_id"]
    selected_payload["comparison_metadata"]["scope"] = "selected_case"
    selected_payload["comparison_metadata"]["evaluated_case_ids"] = [
        selected_payload["cases"][0]["case_id"]
    ]
    selected_payload["cases"] = selected_payload["cases"][:1]
    selected_payload["grader_aggregates"] = [
        {
            "name": name,
            "passed_case_count": 1,
            "failed_case_count": 0,
            "failure_count": 0,
        }
        for name in EVAL_GRADER_NAMES
    ]
    selected_payload["input_tokens"] = selected_payload["cases"][0]["metrics"]["input_tokens"]
    selected_payload["output_tokens"] = selected_payload["cases"][0]["metrics"]["output_tokens"]
    selected = _validated(selected_payload)

    with pytest.raises(ValidationError, match="passing full-dataset"):
        AcceptedEvalBaselineV1(acceptance_reason="reviewed", report=selected)
    with pytest.raises(ValidationError, match="passing full-dataset"):
        AcceptedEvalBaselineV1(
            acceptance_reason="reviewed",
            report=_fail_grader(full_report, "unsupported_claim"),
        )


def test_identical_comparison_passes_without_regressions_and_is_deterministic(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    first = compare_eval_reports(accepted, full_report)
    second = compare_eval_reports(accepted, full_report)

    assert first.passed is True
    assert first.added_case_ids == first.removed_case_ids == ()
    assert first.version_changes == ()
    assert first.contract_regressions == ()
    assert first.metric_regressions == ()
    assert first.model_dump_json(indent=2) == second.model_dump_json(indent=2)
    assert EvalRegressionReportV1.model_validate_json(first.model_dump_json(), strict=True) == first


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_digest", _CHANGED_DIGEST),
        ("graph_version", "pathfinder-research-v999"),
        ("plan_prompt_version", _CHANGED_DIGEST),
        ("research_tool_schema_version", _CHANGED_DIGEST),
        ("grader_contract_version", "research-graders-v999"),
    ),
)
def test_version_changes_are_structured_and_fail(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
    field: str,
    value: str,
) -> None:
    result = compare_eval_reports(accepted, _change_version(full_report, field, value))

    assert result.passed is False
    assert tuple(change.field for change in result.version_changes) == (field,)


def test_old_grader_semantic_version_is_a_structured_change_not_schema_incompatibility(
    full_report: EvalReportV3,
) -> None:
    old_report = _change_version(full_report, "grader_contract_version", "research-graders-v1")
    accepted = AcceptedEvalBaselineV1(
        acceptance_reason="Previously reviewed grader semantics.", report=old_report
    )

    result = compare_eval_reports(accepted, full_report)

    assert result.passed is False
    assert tuple(
        (item.field, item.baseline_value, item.current_value) for item in result.version_changes
    ) == (("grader_contract_version", "research-graders-v1", "research-graders-v3"),)
    assert result.contract_regressions == ()


@pytest.mark.parametrize(("add", "field"), ((True, "added_case_ids"), (False, "removed_case_ids")))
def test_case_coverage_drift_fails_closed(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
    add: bool,
    field: str,
) -> None:
    result = compare_eval_reports(accepted, _change_coverage(full_report, add=add))

    assert result.passed is False
    assert getattr(result, field)


@pytest.mark.parametrize("grader_name", ("unsupported_claim", "citation_resolvable"))
def test_grader_pass_to_fail_is_a_contract_regression(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
    grader_name: str,
) -> None:
    result = compare_eval_reports(accepted, _fail_grader(full_report, grader_name))

    assert result.passed is False
    assert any(
        item.kind == "grader" and item.grader == grader_name for item in result.contract_regressions
    )


def test_case_execution_failure_is_a_contract_regression(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    result = compare_eval_reports(accepted, _fail_execution(full_report))

    assert result.passed is False
    assert result.contract_regressions[0].kind == "case_execution"


@pytest.mark.parametrize(
    "metric",
    (
        "irrelevant_context_count",
        "input_tokens",
        "output_tokens",
        "total_tool_call_count",
    ),
)
def test_lower_is_better_metric_increase_fails(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
    metric: str,
) -> None:
    baseline_value = getattr(full_report.cases[0].metrics, metric)
    result = compare_eval_reports(
        accepted,
        _change_metric(full_report, metric, baseline_value + 1),
    )

    assert result.passed is False
    assert any(item.metric == metric for item in result.metric_regressions)


def test_recall_decrease_and_missing_current_value_fail(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    case_index = next(
        index
        for index, case in enumerate(full_report.cases)
        if case.metrics.recall_at_5 is not None
    )
    lower = compare_eval_reports(
        accepted,
        _change_metric(full_report, "recall_at_5", 0.5, case_index=case_index),
    )
    missing = compare_eval_reports(
        accepted,
        _change_metric(full_report, "recall_at_5", None, case_index=case_index),
    )

    assert lower.passed is missing.passed is False
    assert lower.metric_regressions[0].metric == "recall_at_5"
    assert missing.metric_regressions[0].current_value is None


def test_metric_improvements_do_not_regress(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    improved = full_report
    for metric in ("input_tokens", "output_tokens", "total_tool_call_count"):
        baseline_value = getattr(improved.cases[0].metrics, metric)
        improved = _change_metric(improved, metric, max(0, baseline_value - 1))

    result = compare_eval_reports(accepted, improved)

    assert result.passed is True
    assert result.metric_regressions == ()


def test_recall_improvement_does_not_regress(full_report: EvalReportV3) -> None:
    case_index = next(
        index
        for index, case in enumerate(full_report.cases)
        if case.metrics.recall_at_5 is not None
    )
    lower_baseline_report = _change_metric(
        full_report,
        "recall_at_5",
        0.5,
        case_index=case_index,
    )
    lower_baseline = AcceptedEvalBaselineV1(
        acceptance_reason="Reviewed lower recall fixture.",
        report=lower_baseline_report,
    )

    result = compare_eval_reports(lower_baseline, full_report)

    assert result.passed is True
    assert result.metric_regressions == ()


def test_missing_malformed_and_incompatible_baselines_exit_two(
    tmp_path: Path,
    full_report: EvalReportV3,
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    wrong_wrapper = tmp_path / "wrong-wrapper.json"
    wrong_wrapper.write_text("{}", encoding="utf-8")
    nested_incompatible = tmp_path / "nested.json"
    nested_incompatible.write_text(
        json.dumps(
            {
                "baseline_schema_version": 1,
                "acceptance_reason": "reviewed",
                "report": {"schema_version": 999},
            }
        ),
        encoding="utf-8",
    )
    unreadable = tmp_path / "baseline-directory"
    unreadable.mkdir()

    for path, category in (
        (tmp_path / "missing.json", "baseline_missing"),
        (unreadable, "baseline_unreadable"),
        (malformed, "baseline_malformed_json"),
        (wrong_wrapper, "baseline_schema_incompatible"),
        (nested_incompatible, "baseline_schema_incompatible"),
    ):
        report, exit_code = run_regression(
            baseline_path=path,
            current_report=full_report,
        )
        assert exit_code == 2
        assert report.error_category == category


def test_current_configuration_failure_exits_two(
    tmp_path: Path,
    accepted: AcceptedEvalBaselineV1,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path, accepted)
    current = EvalReportV3(passed=False, error_category="invalid_dataset")

    report, exit_code = run_regression(
        baseline_path=baseline_path,
        current_report=current,
    )

    assert exit_code == 2
    assert report.error_category == "current_eval_configuration_error"


def test_current_quality_failure_exits_one(
    tmp_path: Path,
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    _write_baseline(baseline_path, accepted)

    report, exit_code = run_regression(
        baseline_path=baseline_path,
        current_report=_fail_grader(full_report, "unsupported_claim"),
    )

    assert exit_code == 1
    assert report.current_quality_failure is True


def test_repository_accepted_baseline_is_read_only_for_ordinary_regression(
    full_report: EvalReportV3,
) -> None:
    before = DEFAULT_BASELINE_PATH.read_bytes()

    report, exit_code = run_regression(current_report=full_report)

    assert exit_code == 0
    assert report.passed is True
    assert DEFAULT_BASELINE_PATH.read_bytes() == before


def test_hard_grader_regression_does_not_modify_repository_baseline(
    full_report: EvalReportV3,
) -> None:
    before = DEFAULT_BASELINE_PATH.read_bytes()

    report, exit_code = run_regression(
        current_report=_fail_grader(full_report, "unsupported_claim")
    )

    assert exit_code == 1
    assert report.contract_regressions[0].grader == "unsupported_claim"
    assert DEFAULT_BASELINE_PATH.read_bytes() == before


def test_regression_artifact_excludes_canaries_and_runtime_fields(
    accepted: AcceptedEvalBaselineV1,
    full_report: EvalReportV3,
) -> None:
    serialized = compare_eval_reports(accepted, full_report).model_dump_json()
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
