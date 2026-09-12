"""E4.8 paired descriptive comparisons; no independent-sample or significance claim."""

from __future__ import annotations

from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_score import (
    QualityScoreError,
    group_members,
    samples,
    support,
    validate_scored_report,
)
from tests.evals.quality_score_contracts import (
    PairedCaseV1,
    PairedGroupV1,
    QualityComparisonV1,
    QualityScoredReportV1,
)

# Runtime/model/retrieval changes are visible in provenance and may be compared.
# Dataset, rubric, suite, measurement layer and data source modes must remain aligned.
COMPATIBILITY = (
    "dataset_version",
    "dataset_digest",
    "split_digest",
    "rubric_version",
    "rubric_digest",
    "suite_version",
    "measurement_scope",
    "llm_mode",
    "web_mode",
    "document_mode",
)


def _delta(left, right):
    return None if left is None or right is None else right - left


def _time(case):
    return next((t.seconds for t in case.observation.stage_times if t.stage == "total"), None)


def _pair(left, right):
    case = left or right
    both = left is not None and right is not None
    executed = (
        both and left.observation.status != "not_run" and right.observation.status != "not_run"
    )
    reviewed = both and left.assessment_complete and right.assessment_complete
    retrieval = {}
    for name in (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "required_unit_coverage",
        "reciprocal_rank",
    ):

        def value(c, name=name):
            if c is None or c.retrieval is None:
                return None
            result = getattr(c.retrieval, name)
            return result if name == "reciprocal_rank" else result.value

        retrieval[name] = _delta(value(left), value(right))
    return PairedCaseV1(
        case_id=case.observation.case_id,
        repeat_index=case.observation.repeat_index,
        family_id=case.family_id,
        stratum=case.stratum,
        split=case.split,
        tags=case.tags,
        left=left,
        right=right,
        delta_business=_delta(left.business_success, right.business_success) if both else None,
        delta_fact_support=_delta(support(left.facts).value, support(right.facts).value)
        if reviewed
        else None,
        delta_citation_support=_delta(support(left.citations).value, support(right.citations).value)
        if reviewed
        else None,
        delta_required_evidence=_delta(left.required_evidence.value, right.required_evidence.value)
        if reviewed
        else None,
        delta_known_cost_cny=_delta(
            left.observation.cost.known_cost_cny, right.observation.cost.known_cost_cny
        )
        if executed
        else None,
        delta_total_seconds=_delta(_time(left), _time(right)) if both else None,
        delta_retrieval=retrieval,
    )


def _group(dimension, key, pairs):
    business = {"improved": 0, "regressed": 0, "unchanged": 0, "unassessed": 0}
    for p in pairs:
        label = (
            "unassessed"
            if p.delta_business is None
            else "improved"
            if p.delta_business > 0
            else "regressed"
            if p.delta_business < 0
            else "unchanged"
        )
        business[label] += 1
    differences = {}
    for name in (
        "fact_support",
        "citation_support",
        "required_evidence",
        "known_cost_cny",
        "total_seconds",
    ):
        values = [getattr(p, "delta_" + name) for p in pairs]
        differences[name] = samples(
            [v for v in values if v is not None], missing=values.count(None)
        )
    return PairedGroupV1(
        dimension=dimension,
        group_id=key,
        overlapping=dimension == "tag",
        planned_pairs=len(pairs),
        matched_pairs=sum(p.left is not None and p.right is not None for p in pairs),
        left_only=sum(p.right is None for p in pairs),
        right_only=sum(p.left is None for p in pairs),
        family_count=len({p.family_id for p in pairs}),
        business_changes=business,
        differences=differences,
    )


def compare_quality(
    left: QualityScoredReportV1, right: QualityScoredReportV1
) -> QualityComparisonV1:
    """Align the union of slots; missing/unknown values never become improvements."""
    try:
        left = QualityScoredReportV1.model_validate_json(left.model_dump_json())
        right = QualityScoredReportV1.model_validate_json(right.model_dump_json())
        validate_scored_report(left)
        validate_scored_report(right)
        if (
            left.execution_kind != right.execution_kind
            or left.scorer_version != right.scorer_version
            or any(
                getattr(left.manifest, key) != getattr(right.manifest, key) for key in COMPATIBILITY
            )
        ):
            raise QualityScoreError("comparison_identity_mismatch")
        indexed = [
            {(c.observation.case_id, c.observation.repeat_index): c for c in r.cases}
            for r in (left, right)
        ]
        pairs = []
        for key in sorted(indexed[0].keys() | indexed[1].keys()):
            a, b = (index.get(key) for index in indexed)
            if (
                a is not None
                and b is not None
                and any(
                    getattr(a, field) != getattr(b, field)
                    for field in ("family_id", "stratum", "split", "tags", "answer_class")
                )
            ):
                raise QualityScoreError("comparison_group_mismatch")
            pairs.append(_pair(a, b))
        pairs = tuple(pairs)
        return QualityComparisonV1(
            left_manifest=left.manifest,
            right_manifest=right.manifest,
            left_scorer_source_sha=left.scorer_source_sha,
            right_scorer_source_sha=right.scorer_source_sha,
            left_digest=quality_identity_digest(left.model_dump(mode="json")),
            right_digest=quality_identity_digest(right.model_dump(mode="json")),
            pairs=pairs,
            groups=tuple(_group(d, k, members) for d, k, members in group_members(pairs)),
            matched_pairs=sum(p.left is not None and p.right is not None for p in pairs),
            left_only=sum(p.right is None for p in pairs),
            right_only=sum(p.left is None for p in pairs),
            independent_family_count=len({p.family_id for p in pairs}),
            left_summary=left.summary,
            right_summary=right.summary,
            evidence_valid=left.evidence_valid and right.evidence_valid,
        )
    except QualityScoreError:
        raise
    except Exception:
        raise QualityScoreError("invalid_comparison_input") from None


def validate_comparison(report: QualityComparisonV1) -> None:
    """Verify exported deltas and group counts against the retained paired rows."""
    keys = [(p.case_id, p.repeat_index) for p in report.pairs]
    expected_pairs = tuple(_pair(p.left, p.right) for p in report.pairs)
    left = tuple(p.left for p in report.pairs if p.left is not None)
    right = tuple(p.right for p in report.pairs if p.right is not None)
    from tests.evals.quality_score import summarize

    if (
        len(keys) != len(set(keys))
        or report.pairs != expected_pairs
        or report.groups != tuple(_group(d, k, m) for d, k, m in group_members(report.pairs))
        or report.matched_pairs
        != sum(p.left is not None and p.right is not None for p in report.pairs)
        or report.left_only != sum(p.right is None for p in report.pairs)
        or report.right_only != sum(p.left is None for p in report.pairs)
        or report.independent_family_count != len({p.family_id for p in report.pairs})
        or report.left_summary != summarize(left)
        or report.right_summary != summarize(right)
        or any(
            getattr(report.left_manifest, k) != getattr(report.right_manifest, k)
            for k in COMPATIBILITY
        )
    ):
        raise QualityScoreError("comparison_aggregate_mismatch")
