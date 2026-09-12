from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evals.retrieval_contracts import (
    RetrievalBenchmarkAggregateV1,
    RetrievalBenchmarkCaseReportV1,
    RetrievalBenchmarkReportV1,
    RetrievalChunkRefV1,
    RetrievalExpectedRankingV2,
    RetrievalHardInvariantsV1,
    RetrievalPolicyIdentityV1,
    RetrievalRegressionPolicyV2,
    RetrievalRegressionReportV2,
    RetrievalThresholdsV1,
)
from tests.evals.retrieval_regression import (
    RetrievalPolicyLoadError,
    compare_retrieval_report,
    load_retrieval_policy,
    run_retrieval_regression,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64


@pytest.fixture
def current_report() -> RetrievalBenchmarkReportV1:
    reference = RetrievalChunkRefV1(document_alias="backend_profile", ordinal=0)
    return RetrievalBenchmarkReportV1(
        evidence_scope="deterministic_fake_embedding_regression",
        dataset_version="retrieval-v1",
        dataset_digest=_DIGEST_A,
        case_set_digest=_DIGEST_B,
        normalization_version="normalization-v1",
        chunking_version="chunking-v1",
        embedding_profile="fake-embedding-v1",
        embedding_dimension=1536,
        retrieval_top_k=5,
        cases=(
            RetrievalBenchmarkCaseReportV1(
                case_id="example",
                relevant_chunks=(reference,),
                ranked_retrieved_chunks=(reference,),
                recall_at_1=0.4,
                recall_at_3=0.7,
                recall_at_5=0.8,
                reciprocal_rank=0.6,
                irrelevant_context_count=3,
                workspace_leakage_count=0,
                allowlist_leakage_count=0,
                passed=True,
            ),
        ),
        aggregate=RetrievalBenchmarkAggregateV1(
            macro_recall_at_1=0.4,
            macro_recall_at_3=0.7,
            macro_recall_at_5=0.8,
            mean_reciprocal_rank=0.6,
            total_irrelevant_context_count=3,
            workspace_leakage_count=0,
            allowlist_leakage_count=0,
        ),
        passed=True,
    )


@pytest.fixture
def policy(current_report: RetrievalBenchmarkReportV1) -> RetrievalRegressionPolicyV2:
    return RetrievalRegressionPolicyV2(
        policy_version="retrieval-regression-v2",
        acceptance_reason="Reviewed deterministic fake-embedding evidence.",
        evidence_scope="deterministic_fake_embedding_regression",
        semantic_quality_claim=False,
        expected_identity=RetrievalPolicyIdentityV1(
            dataset_version=current_report.dataset_version,
            dataset_digest=current_report.dataset_digest,
            case_set_digest=current_report.case_set_digest,
            normalization_version=current_report.normalization_version,
            chunking_version=current_report.chunking_version,
            embedding_profile=current_report.embedding_profile,
            embedding_dimension=current_report.embedding_dimension,
            retrieval_top_k=current_report.retrieval_top_k,
        ),
        expected_rankings=tuple(
            RetrievalExpectedRankingV2(
                case_id=case.case_id,
                ranked_retrieved_chunks=case.ranked_retrieved_chunks,
            )
            for case in current_report.cases
        ),
        thresholds=RetrievalThresholdsV1(
            minimum_macro_recall_at_1=current_report.aggregate.macro_recall_at_1,
            minimum_macro_recall_at_3=current_report.aggregate.macro_recall_at_3,
            minimum_macro_recall_at_5=current_report.aggregate.macro_recall_at_5,
            minimum_mean_reciprocal_rank=current_report.aggregate.mean_reciprocal_rank,
            maximum_total_irrelevant_context_count=(
                current_report.aggregate.total_irrelevant_context_count
            ),
        ),
        hard_invariants=RetrievalHardInvariantsV1(
            workspace_leakage_count=0,
            allowlist_leakage_count=0,
        ),
    )


def _write_policy(path: Path, policy: RetrievalRegressionPolicyV2) -> bytes:
    serialized = (policy.model_dump_json(indent=2) + "\n").encode()
    path.write_bytes(serialized)
    return serialized


def test_policy_loader_is_strict_and_round_trips(
    tmp_path: Path,
    policy: RetrievalRegressionPolicyV2,
) -> None:
    path = tmp_path / "policy.json"
    _write_policy(path, policy)

    assert load_retrieval_policy(path) == policy


def test_policy_rejects_duplicate_expected_ranking_case_ids(
    policy: RetrievalRegressionPolicyV2,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        RetrievalRegressionPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "expected_rankings": (
                    policy.expected_rankings[0],
                    policy.expected_rankings[0],
                ),
            }
        )


def test_missing_unreadable_malformed_and_incompatible_policy_fail_closed(
    tmp_path: Path,
) -> None:
    unreadable = tmp_path / "directory"
    unreadable.mkdir()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")

    for path, category in (
        (tmp_path / "missing.json", "policy_missing"),
        (unreadable, "policy_unreadable"),
        (malformed, "policy_malformed_json"),
        (incompatible, "policy_schema_incompatible"),
    ):
        with pytest.raises(RetrievalPolicyLoadError) as raised:
            load_retrieval_policy(path)
        assert raised.value.category == category
        report, exit_code = run_retrieval_regression(policy_path=path)
        assert exit_code == 2
        assert report.error_category == category


def test_malformed_current_benchmark_is_a_configuration_error(
    tmp_path: Path,
    policy: RetrievalRegressionPolicyV2,
) -> None:
    policy_path = tmp_path / "policy.json"
    current_path = tmp_path / "current.json"
    _write_policy(policy_path, policy)
    current_path.write_text("{", encoding="utf-8")

    report, exit_code = run_retrieval_regression(
        policy_path=policy_path,
        current_report_path=current_path,
    )

    assert exit_code == 2
    assert report.error_category == "current_benchmark_configuration_error"


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("dataset_digest", "sha256:" + "c" * 64),
        ("case_set_digest", "sha256:" + "d" * 64),
        ("embedding_profile", "fake-embedding-v2"),
        ("retrieval_top_k", 4),
    ),
)
def test_identity_changes_are_valid_regressions(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
    field: str,
    changed: str | int,
) -> None:
    current = current_report.model_copy(update={field: changed})

    result = compare_retrieval_report(policy, current)

    assert result.passed is False
    assert tuple(item.field for item in result.identity_differences) == (field,)


@pytest.mark.parametrize(
    ("metric", "changed"),
    (
        ("macro_recall_at_1", 0.39),
        ("macro_recall_at_3", 0.69),
        ("macro_recall_at_5", 0.79),
        ("mean_reciprocal_rank", 0.59),
        ("total_irrelevant_context_count", 4),
    ),
)
def test_each_metric_regression_fails(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
    metric: str,
    changed: float | int,
) -> None:
    aggregate = current_report.aggregate.model_copy(update={metric: changed})
    current = current_report.model_copy(update={"aggregate": aggregate})

    result = compare_retrieval_report(policy, current)

    assert result.passed is False
    assert tuple(item.metric for item in result.metric_regressions) == (metric,)


def test_exact_threshold_boundary_passes(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
) -> None:
    result = compare_retrieval_report(policy, current_report)

    assert result.passed is True
    assert result.identity_differences == ()
    assert result.hard_invariant_regressions == ()
    assert result.metric_regressions == ()
    assert result.ranking_regressions == ()


def test_metric_improvements_pass(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
) -> None:
    aggregate = current_report.aggregate.model_copy(
        update={
            "macro_recall_at_1": 0.5,
            "macro_recall_at_3": 0.8,
            "macro_recall_at_5": 0.9,
            "mean_reciprocal_rank": 0.7,
            "total_irrelevant_context_count": 2,
        }
    )
    current = current_report.model_copy(update={"aggregate": aggregate})

    assert compare_retrieval_report(policy, current).passed is True


@pytest.mark.parametrize("invariant", ("workspace_leakage_count", "allowlist_leakage_count"))
def test_leakage_fails_even_when_metrics_improve(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
    invariant: str,
) -> None:
    aggregate = current_report.aggregate.model_copy(
        update={
            "macro_recall_at_1": 1.0,
            "macro_recall_at_3": 1.0,
            "macro_recall_at_5": 1.0,
            "mean_reciprocal_rank": 1.0,
            "total_irrelevant_context_count": 0,
            invariant: 1,
        }
    )
    current = current_report.model_copy(update={"aggregate": aggregate})

    result = compare_retrieval_report(policy, current)

    assert result.passed is False
    assert tuple(item.invariant for item in result.hard_invariant_regressions) == (invariant,)


def test_current_hard_failed_report_is_not_allowed_to_pass(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
) -> None:
    current = current_report.model_copy(update={"passed": False})

    result = compare_retrieval_report(policy, current)

    assert result.passed is False
    assert result.hard_invariant_regressions[0].invariant == "current_benchmark_passed"


def test_missing_extra_and_ranking_drift_are_typed_regressions(
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
) -> None:
    reference = RetrievalChunkRefV1(document_alias="backend_profile", ordinal=1)
    original = current_report.cases[0]
    drifted = original.model_copy(update={"ranked_retrieved_chunks": (reference,)})
    extra = original.model_copy(update={"case_id": "extra_case"})
    current = current_report.model_copy(update={"cases": (drifted, extra)})
    expanded_policy = policy.model_copy(
        update={
            "expected_rankings": (
                *policy.expected_rankings,
                RetrievalExpectedRankingV2(
                    case_id="missing_case",
                    ranked_retrieved_chunks=(reference,),
                ),
            )
        }
    )

    result = compare_retrieval_report(expanded_policy, current)

    assert result.passed is False
    assert tuple((item.case_id, item.regression) for item in result.ranking_regressions) == (
        ("missing_case", "missing_case"),
        ("extra_case", "extra_case"),
        ("example", "ranking_tuple_drift"),
    )


def test_regression_report_status_validator_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="status"):
        RetrievalRegressionReportV2(passed=True, error_category="policy_missing")


def test_run_exit_semantics_and_policy_remain_read_only(
    tmp_path: Path,
    policy: RetrievalRegressionPolicyV2,
    current_report: RetrievalBenchmarkReportV1,
) -> None:
    path = tmp_path / "policy.json"
    before = _write_policy(path, policy)

    passed, pass_code = run_retrieval_regression(
        policy_path=path,
        current_report=current_report,
    )
    lower = current_report.aggregate.model_copy(update={"macro_recall_at_5": 0.79})
    failed, fail_code = run_retrieval_regression(
        policy_path=path,
        current_report=current_report.model_copy(update={"aggregate": lower}),
    )

    assert passed.passed is True and pass_code == 0
    assert failed.passed is False and fail_code == 1
    assert path.read_bytes() == before
