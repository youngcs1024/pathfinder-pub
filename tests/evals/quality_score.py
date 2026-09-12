"""Offline deterministic E4.8 scoring; no provider calls or baseline acceptance."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from decimal import Decimal
from pathlib import Path

from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityCostV1,
    QualityCoverageV1,
    QualityGenerationReportV1,
    QualityRatioV1,
    QualityRetrievalReportV1,
)
from tests.evals.quality_dataset import (
    QualityDataset,
    quality_digest,
    quality_identity_digest,
    validate_quality_annotations,
)
from tests.evals.quality_generation_support import checked_directory, secret_markers
from tests.evals.quality_score_contracts import (
    CostSummaryV1,
    JudgmentCountsV1,
    QualityComparisonV1,
    QualityScoredReportV1,
    SampleSummaryV1,
    ScoredCaseV1,
    ScoredGroupV1,
    ScoreSummaryV1,
)

# Explicit v1 policy, not inferred from output or arbitrary expected_behavior strings.
ANSWERABLE = frozenset(
    {
        "preserve_exact_technical_versions",
        "preserve_responsibility_strength",
        "distinguish_schema_from_fact_validation",
        "combine_method_result_with_limits",
        "separate_candidate_from_other_contributors",
        "use_explicit_superseding_source",
    }
)
UNANSWERABLE = frozenset(
    {
        "state_answer_not_in_material",
        "acknowledge_gap_without_inventing_experience",
    }
)
STAGES = ("ingestion", "retrieval", "generation", "approval", "submission", "total")
RETRIEVAL_RATIOS = ("recall_at_1", "recall_at_3", "recall_at_5", "required_unit_coverage")


class QualityScoreError(ValueError):
    """Only fixed codes cross the scoring/publication boundary."""


def ratio(numerator: int, denominator: int, *, unassessed: int = 0) -> QualityRatioV1:
    return QualityRatioV1(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else None,
        unassessed_count=unassessed,
        not_applicable_reason=None if denominator else "zero_denominator",
    )


def samples(values, *, missing: int = 0) -> SampleSummaryV1:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    return SampleSummaryV1(
        sample_count=n,
        missing_count=missing,
        mean=math.fsum(ordered) / n if n else None,
        p50=ordered[math.ceil(n * 0.5) - 1] if n else None,
        p95=ordered[math.ceil(n * 0.95) - 1] if n else None,
        null_reason=None if n else "no_samples",
    )


def support(counts: JudgmentCountsV1) -> QualityRatioV1:
    return ratio(
        counts.supported,
        counts.supported + counts.contradicted + counts.unsupported,
        unassessed=counts.unassessed,
    )


def _counts(judgments, unassessed=0):
    counts = Counter(j.judgment for j in judgments)
    return JudgmentCountsV1(**counts, unassessed=unassessed)


def _sum_counts(cases, field):
    return JudgmentCountsV1(
        **{
            key: sum(getattr(getattr(c, field), key) for c in cases)
            for key in JudgmentCountsV1.model_fields
        }
    )


def _cost(costs):
    known = sum((c.known_cost_cny for c in costs), Decimal(0))
    unknown = sum(c.unknown_cost_attempts for c in costs)
    return QualityCostV1(
        scope="all",
        known_cost_cny=known,
        priced_attempts=sum(c.priced_attempts for c in costs),
        unknown_cost_attempts=unknown,
        total_cost_cny=None if unknown else known,
    )


def summarize(cases: tuple[ScoredCaseV1, ...]) -> ScoreSummaryV1:
    observations = [c.observation for c in cases]
    executed = sum(o.status != "not_run" for o in observations)
    required = sum(o.assessment_required for o in observations)
    assessed = sum(c.assessment_complete for c in cases)
    successes = sum(c.business_success is True for c in cases)
    coverage = QualityCoverageV1(
        planned=len(cases),
        executed=executed,
        failed=sum(o.status == "failed" for o in observations),
        not_run=sum(o.status == "not_run" for o in observations),
        generated=sum(o.output_digest is not None for o in observations),
        assessment_required=required,
        assessed=assessed,
        unassessed=required - assessed,
    )
    facts, citations = _sum_counts(cases, "facts"), _sum_counts(cases, "citations")
    cost = _cost([o.cost for o in observations])
    stage_times = {}
    for stage in STAGES:
        values = [t.seconds for o in observations for t in o.stage_times if t.stage == stage]
        stage_times[stage] = samples(values, missing=executed - len(values))
    macro, micro = {}, {}
    for name in (*RETRIEVAL_RATIOS, "reciprocal_rank"):
        metrics = [getattr(c.retrieval, name) for c in cases if c.retrieval is not None]
        values = [v.value if isinstance(v, QualityRatioV1) else v for v in metrics]
        applicable = [v for v in values if v is not None]
        macro[name] = samples(applicable, missing=len(cases) - len(applicable))
        if name in RETRIEVAL_RATIOS:
            micro[name] = ratio(
                sum(v.numerator for v in metrics),
                sum(v.denominator for v in metrics),
                unassessed=sum(v.unassessed_count for v in metrics),
            )
    answer_metrics = {}
    for label, result in (("unanswerable", "appropriate"), ("answerable", "unnecessary_refusal")):
        eligible = [
            c for c in cases if c.answer_class == label and c.observation.status != "not_run"
        ]
        answer_metrics[label] = ratio(
            sum(c.insufficiency == result and c.assessment_complete for c in eligible),
            len(eligible),
            unassessed=sum(
                c.observation.status == "succeeded" and not c.assessment_complete for c in eligible
            ),
        )
    return ScoreSummaryV1(
        coverage=coverage,
        independent_family_count=len({c.family_id for c in cases}),
        business_success=ratio(
            successes,
            executed,
            unassessed=sum(
                c.business_success is None
                and not c.business_not_assessable
                and c.observation.status != "not_run"
                for c in cases
            ),
        ),
        business_not_assessable=sum(c.business_not_assessable for c in cases),
        annotation_coverage=ratio(assessed, required, unassessed=required - assessed),
        facts=facts,
        fact_support=support(facts),
        citations=citations,
        citation_support=support(citations),
        unknown_fact_inventory_outputs=sum(
            c.observation.assessment_required and not c.assessment_complete for c in cases
        ),
        required_evidence=ratio(
            sum(c.required_evidence.numerator for c in cases),
            sum(c.required_evidence.denominator for c in cases),
            unassessed=sum(c.required_evidence.unassessed_count for c in cases),
        ),
        no_answer_handling=answer_metrics["unanswerable"],
        unnecessary_refusal=answer_metrics["answerable"],
        unmapped_answer_cases=sum(c.answer_class == "unmapped" for c in cases),
        draft_distribution=dict(Counter(c.draft_usability or "unassessed" for c in cases)),
        insufficiency_distribution=dict(Counter(c.insufficiency or "unassessed" for c in cases)),
        failure_distribution=dict(Counter(o.failure_type for o in observations if o.failure_type)),
        cost=CostSummaryV1(
            cost=cost,
            per_executed_known_cny=cost.known_cost_cny / executed if executed else None,
            per_success_known_cny=cost.known_cost_cny / successes if successes else None,
            executed_denominator=executed,
            success_denominator=successes,
            null_reason_executed=None if executed else "zero_denominator",
            null_reason_success=None if successes else "zero_denominator",
        ),
        model_calls=sum(o.model_calls for o in observations),
        tool_calls=sum(o.tool_calls for o in observations),
        provider_attempts=sum(o.provider_attempts for o in observations),
        stage_times=stage_times,
        retrieval_macro=macro,
        retrieval_micro=micro,
    )


def group_members(cases):
    for dimension, attribute in (
        ("family", "family_id"),
        ("stratum", "stratum"),
        ("split", "split"),
        ("tag", "tags"),
    ):
        groups = {}
        for case in cases:
            keys = case.tags if dimension == "tag" else (getattr(case, attribute),)
            for key in keys:
                groups.setdefault(key, []).append(case)
        for key in sorted(groups):
            yield dimension, key, tuple(groups[key])


def _review(reviews, required_reviewers):
    if not reviews:
        return None, False

    # Compare scoring content; reason prose and reviewer identities never enter scores.
    def content(a):
        return (
            sorted((f.fact_id, f.judgment) for f in a.facts),
            sorted((c.fact_id, c.citation_id, c.judgment) for c in a.citations),
            sorted(a.covered_unit_ids),
            sorted(a.missing_unit_ids),
            a.unassessed_fact_count,
            a.unassessed_citation_count,
            a.business_result,
            a.insufficiency,
            a.draft_usability,
        )

    if any(content(a) != content(reviews[0]) for a in reviews[1:]):
        raise QualityScoreError("conflicting_annotations")
    complete = len(reviews) >= required_reviewers and all(
        a.assessment_complete and not any(d.status == "unresolved" for d in a.disagreements)
        for a in reviews
    )
    # Unresolved disagreements and missing independent reviews are not consensus evidence.
    if len(reviews) < required_reviewers or any(
        d.status == "unresolved" for a in reviews for d in a.disagreements
    ):
        return None, False
    return reviews[0], complete


def score_quality(
    dataset: QualityDataset,
    report,
    annotations: tuple[HumanAnnotationV1, ...] = (),
    *,
    scorer_source_sha: str,
    scorer_change_reason: str | None = None,
) -> QualityScoredReportV1:
    """Score a validated retrieval/generation report against its original rubric identity."""
    try:
        if type(report) not in (QualityGenerationReportV1, QualityRetrievalReportV1):
            raise QualityScoreError("unsupported_execution_report")
        report = type(report).model_validate_json(report.model_dump_json())
        annotations = tuple(
            HumanAnnotationV1.model_validate_json(a.model_dump_json()) for a in annotations
        )
        manifest = (
            report.start.manifest
            if isinstance(report, QualityGenerationReportV1)
            else report.manifest
        )
        observations = tuple(c.observation for c in report.cases)
        validate_quality_annotations(dataset, manifest, observations, annotations)
        cases_by_id = {c.case_id: c for c in dataset.cases}
        scored = []
        required_reviewers = (
            2 if dataset.rubric.review_policy == "independent_review_with_adjudication" else 1
        )
        for item in report.cases:
            o = item.observation
            case = cases_by_id[o.case_id]
            reviews = [
                a for a in annotations if (a.case_id, a.repeat_index) == (o.case_id, o.repeat_index)
            ]
            annotation, complete = _review(reviews, required_reviewers)
            required_units = set(case.required_unit_ids)
            covered = required_units & set(annotation.covered_unit_ids) if annotation else set()
            reviewed = (
                required_units & set(annotation.covered_unit_ids + annotation.missing_unit_ids)
                if annotation
                else set()
            )
            if complete and reviewed != required_units:
                raise QualityScoreError("incomplete_required_evidence")
            business = False if o.status == "failed" else None
            if (
                o.status == "succeeded"
                and complete
                and annotation.business_result != "not_assessable"
            ):
                business = annotation.business_result == "success"
            scored.append(
                ScoredCaseV1(
                    observation=o,
                    family_id=case.family_id,
                    stratum=case.stratum,
                    split=case.split,
                    tags=case.tags,
                    assessment_complete=complete,
                    business_success=business,
                    business_not_assessable=complete
                    and annotation.business_result == "not_assessable"
                    and o.status == "succeeded",
                    facts=_counts(annotation.facts, annotation.unassessed_fact_count)
                    if annotation
                    else JudgmentCountsV1(),
                    citations=_counts(annotation.citations, annotation.unassessed_citation_count)
                    if annotation
                    else JudgmentCountsV1(),
                    required_evidence=ratio(
                        len(covered), len(required_units), unassessed=len(required_units - reviewed)
                    ),
                    answer_class="answerable"
                    if case.expected_behavior in ANSWERABLE
                    else "unanswerable"
                    if case.expected_behavior in UNANSWERABLE
                    else "unmapped",
                    insufficiency=annotation.insufficiency if annotation else None,
                    draft_usability=annotation.draft_usability if annotation else None,
                    retrieval=item.metrics
                    if isinstance(report, QualityRetrievalReportV1)
                    else None,
                )
            )
        scored = tuple(scored)
        summary = summarize(scored)
        complete = report.measurement_complete and summary.coverage.unassessed == 0
        return QualityScoredReportV1(
            manifest=manifest,
            execution_kind="generation"
            if isinstance(report, QualityGenerationReportV1)
            else "retrieval",
            execution_source_sha=manifest.execution_source_sha,
            scorer_source_sha=scorer_source_sha,
            scorer_change_reason=scorer_change_reason,
            manifest_digest=quality_digest(manifest.model_dump_json().encode()),
            input_report_digest=quality_identity_digest(report.model_dump(mode="json")),
            annotation_digest=quality_identity_digest(
                [a.model_dump(mode="json") for a in annotations]
            )
            if annotations
            else None,
            cases=scored,
            summary=summary,
            groups=tuple(
                ScoredGroupV1(dimension=d, group_id=k, overlapping=d == "tag", summary=summarize(m))
                for d, k, m in group_members(scored)
            ),
            ingestion_cost=report.ingestion_usage.cost,
            ingestion_seconds=report.ingestion_seconds,
            total_cost=report.total_usage.cost,
            evidence_valid=report.evidence_valid,
            execution_measurement_complete=report.measurement_complete,
            stop_reason=report.stop_reason,
            measurement_complete=complete,
            rubric_frozen=dataset.rubric.status == "frozen",
            semantic_quality_claim=manifest.llm_mode == "qwen"
            and report.evidence_valid
            and complete
            and dataset.rubric.status == "frozen"
            and any(o.status == "succeeded" for o in observations),
        )
    except QualityScoreError:
        raise
    except Exception:
        raise QualityScoreError("invalid_scoring_input") from None


def write_quality_score(path: Path, report, *, forbidden_markers: tuple[str, ...] = ()) -> str:
    """Publish only typed metadata, scanned before create-only write; retain partial files."""
    try:
        if type(report) not in (QualityScoredReportV1, QualityComparisonV1):
            raise ValueError("unsupported")
        checked = type(report).model_validate_json(report.model_dump_json())
        if isinstance(checked, QualityScoredReportV1):
            validate_scored_report(checked)
        else:
            from tests.evals.quality_compare import validate_comparison

            validate_comparison(checked)
        data = (
            json.dumps(
                checked.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        if any(marker and marker in data.decode() for marker in secret_markers(forbidden_markers)):
            raise ValueError("unsafe")
        parent = checked_directory(path.parent, private=False)
        if path.name in {"", ".", ".."}:
            raise ValueError("invalid")
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(directory)
        finally:
            os.close(directory)
        return quality_digest(data)
    except Exception:
        raise QualityScoreError("score_publication_failed") from None


def validate_scored_report(report: QualityScoredReportV1) -> None:
    """Recompute public aggregates before comparison/publication, rejecting forged summaries."""
    expected = summarize(report.cases)
    groups = tuple(
        ScoredGroupV1(dimension=d, group_id=k, overlapping=d == "tag", summary=summarize(m))
        for d, k, m in group_members(report.cases)
    )
    if (
        report.summary != expected
        or report.groups != groups
        or report.manifest_digest != quality_digest(report.manifest.model_dump_json().encode())
        or report.total_cost != _cost([report.ingestion_cost, expected.cost.cost])
        or any(
            c.observation.experiment_id != report.manifest.experiment_id
            or c.observation.measurement_scope != report.manifest.measurement_scope
            for c in report.cases
        )
        or (expected.coverage.assessed and report.annotation_digest is None)
        or report.measurement_complete
        != (report.execution_measurement_complete and expected.coverage.unassessed == 0)
        or (
            report.execution_measurement_complete
            and (expected.coverage.not_run or report.stop_reason is not None)
        )
        or (
            report.evidence_valid
            and (
                report.stop_reason in {"configuration", "integrity", "safety"}
                or any(
                    c.observation.failure_type in {"configuration", "integrity", "safety"}
                    for c in report.cases
                )
            )
        )
    ):
        raise QualityScoreError("score_aggregate_mismatch")


def load_quality_annotations(path: Path) -> tuple[HumanAnnotationV1, ...]:
    """Read bounded UTF-8 JSONL labels; identity/scope validation happens in score_quality."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("oversized")
        rows = tuple(
            HumanAnnotationV1.model_validate_json(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        )
        keys = [(a.experiment_id, a.case_id, a.repeat_index, a.reviewer_id) for a in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate")
        return rows
    except Exception:
        raise QualityScoreError("invalid_annotation_file") from None
