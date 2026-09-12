"""E4.8 pairing and denominator regression cases; no statistical significance."""

from __future__ import annotations

import pytest

from tests.evals.quality_compare import compare_quality
from tests.evals.quality_score import QualityScoreError, write_quality_score
from tests.evals.quality_score_contracts import QualityScoredReportV1
from tests.evals.test_quality_score import changed_review, inputs, score


def test_pairs_align_case_repeat_family_and_preserve_missing_outputs():
    left = score(inputs(("succeeded", "failed")))
    dataset, execution, reviews = inputs(
        ("succeeded", "succeeded", "not_run"), experiment="candidate"
    )
    reviews = (changed_review(reviews[0], business_result="failure"), reviews[1])
    right = score((dataset, execution, reviews))
    result = compare_quality(left, right)
    assert (result.matched_pairs, result.left_only, result.right_only) == (2, 0, 1)
    assert result.independent_family_count == 1
    assert [p.delta_business for p in result.pairs] == [-1, 1, None]
    assert result.pairs[1].left.observation.output_digest is None
    assert result.pairs[1].delta_fact_support is None
    assert result.pairs[2].right.observation.status == "not_run"
    family = next(g for g in result.groups if g.dimension == "family")
    assert family.business_changes == {
        "improved": 1,
        "regressed": 1,
        "unchanged": 0,
        "unassessed": 1,
    }
    assert result.significance_claim is False
    assert result.left_summary.coverage.planned == 2
    assert result.right_summary.coverage.planned == 3


def test_per_case_denominator_changes_remain_visible():
    left = score(inputs())
    dataset, execution, reviews = inputs(experiment="candidate")
    review = changed_review(
        reviews[0],
        facts=[
            {"fact_id": "claim_one", "judgment": "supported"},
            {"fact_id": "claim_two", "judgment": "unsupported"},
        ],
    )
    right = score((dataset, execution, (review,)))
    pair = compare_quality(left, right).pairs[0]
    assert pair.delta_fact_support == -0.5
    assert pair.left.facts.supported == pair.right.facts.supported == 1
    assert pair.right.facts.unsupported == 1


def test_unknown_business_result_is_not_improvement():
    result = compare_quality(
        score(inputs(annotations=False)), score(inputs(experiment="candidate"))
    )
    assert result.pairs[0].delta_business is None
    assert result.groups[0].business_changes["unassessed"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_digest", "sha256:" + "c" * 64),
        ("rubric_digest", "sha256:" + "c" * 64),
        ("measurement_scope", "generation"),
        ("web_mode", "none"),
    ],
)
def test_incompatible_comparisons_fail_closed(field, value):
    dataset, execution, reviews = inputs(experiment="candidate")
    if field in {"dataset_digest", "rubric_digest"}:
        # Even a contract-shaped report with forged manifest metadata must not compare.
        left = score(inputs())
        right = score((dataset, execution, reviews))
        payload = right.model_dump(mode="json")
        payload["manifest"][field] = value
        import json

        right = QualityScoredReportV1.model_validate_json(json.dumps(payload))
        with pytest.raises(QualityScoreError):
            compare_quality(left, right)
    else:
        payload = execution.model_dump(mode="json")
        payload["start"]["manifest"][field] = value
        if field == "measurement_scope":
            for case in payload["cases"]:
                case["observation"][field] = value
        import json

        changed = type(execution).model_validate_json(json.dumps(payload))
        right = score((dataset, changed, reviews))
        with pytest.raises(QualityScoreError, match="comparison_identity_mismatch"):
            compare_quality(score(inputs()), right)


def test_duplicate_scored_slot_and_tampered_summary_rejected(tmp_path):
    left = score(inputs())
    duplicate = left.model_copy(update={"cases": left.cases + left.cases})
    with pytest.raises(QualityScoreError, match="invalid_comparison_input"):
        compare_quality(left, duplicate)
    forged = left.model_copy(
        update={"summary": left.summary.model_copy(update={"model_calls": 999})}
    )
    with pytest.raises(QualityScoreError, match="score_aggregate_mismatch"):
        compare_quality(left, forged)
    with pytest.raises(QualityScoreError, match="score_publication_failed"):
        write_quality_score(tmp_path / "forged.json", forged)


def test_group_identity_mismatch_rejected():
    from dataclasses import replace

    left = score(inputs())
    dataset, execution, reviews = inputs(experiment="candidate")
    dataset = replace(
        dataset,
        cases=tuple(c.model_copy(update={"family_id": "other_family"}) for c in dataset.cases),
    )
    with pytest.raises(QualityScoreError, match="comparison_group_mismatch"):
        compare_quality(left, score((dataset, execution, reviews)))


def test_comparison_publication_roundtrip(tmp_path):
    result = compare_quality(score(inputs()), score(inputs(experiment="candidate")))
    path = tmp_path / "compare.json"
    write_quality_score(path, result)
    assert '"significance_claim":false' in path.read_text()
    assert "private body" not in path.read_text()
    assert result.pairs[0].delta_fact_support == 0


def test_forged_comparison_delta_cannot_be_published(tmp_path):
    result = compare_quality(score(inputs()), score(inputs(experiment="candidate")))
    forged = result.model_copy(
        update={"pairs": (result.pairs[0].model_copy(update={"delta_business": 1}),)}
    )
    with pytest.raises(QualityScoreError, match="score_publication_failed"):
        write_quality_score(tmp_path / "forged.json", forged)
