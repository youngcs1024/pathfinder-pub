"""E4.8 additive scoring contracts; existing execution artifacts remain unchanged."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.quality_contracts import (
    Count,
    Failure,
    QualityCostV1,
    QualityCoverageV1,
    QualityObservationV1,
    QualityRatioV1,
    QualityRetrievalMetricsV1,
    QualityRunManifestV1,
    Seconds,
    SourceSHA,
    require_unique,
)


class JudgmentCountsV1(EvalContractModel):
    supported: Count = 0
    contradicted: Count = 0
    unsupported: Count = 0
    not_assessable: Count = 0
    unassessed: Count = 0


class SampleSummaryV1(EvalContractModel):
    sample_count: Count
    missing_count: Count
    mean: float | None
    p50: float | None
    p95: float | None
    algorithm: Literal["nearest_rank"] = "nearest_rank"
    null_reason: Literal["no_samples"] | None

    @model_validator(mode="after")
    def empty(self):
        if (self.sample_count == 0) != (self.null_reason == "no_samples") or any(
            (value is None) != (self.sample_count == 0) for value in (self.mean, self.p50, self.p95)
        ):
            raise ValueError("sample coverage mismatch")
        return self


class CostSummaryV1(EvalContractModel):
    cost: QualityCostV1
    per_executed_known_cny: Annotated[Decimal, Field(ge=0)] | None
    per_success_known_cny: Annotated[Decimal, Field(ge=0)] | None
    executed_denominator: Count
    success_denominator: Count
    null_reason_executed: Literal["zero_denominator"] | None
    null_reason_success: Literal["zero_denominator"] | None
    component_costs: None = None
    component_reason: Literal["source_not_disaggregated"] = "source_not_disaggregated"


class ScoredCaseV1(EvalContractModel):
    observation: QualityObservationV1
    family_id: EvalIdentifier
    stratum: EvalIdentifier
    split: Literal["dev", "validation"]
    tags: tuple[EvalIdentifier, ...]
    assessment_complete: bool
    business_success: bool | None
    business_not_assessable: bool
    facts: JudgmentCountsV1
    citations: JudgmentCountsV1
    required_evidence: QualityRatioV1
    answer_class: Literal["answerable", "unanswerable", "unmapped"]
    insufficiency: (
        Literal["appropriate", "unnecessary_refusal", "overclaim", "not_applicable"] | None
    )
    draft_usability: (
        Literal["usable", "minor_edits", "major_edits", "unusable", "not_applicable"] | None
    )
    retrieval: QualityRetrievalMetricsV1 | None

    @model_validator(mode="after")
    def assessment_binding(self):
        require_unique(self.tags)
        o = self.observation
        if (
            (self.assessment_complete and not o.assessment_required)
            or (
                self.business_success is True
                and (o.status != "succeeded" or not self.assessment_complete)
            )
            or (o.status == "failed" and self.business_success is not False)
            or (o.status == "not_run" and self.business_success is not None)
            or (
                self.business_not_assessable
                and (not self.assessment_complete or self.business_success is not None)
            )
            or (self.retrieval is not None and (o.status != "succeeded" or o.assessment_required))
        ):
            raise ValueError("case assessment mismatch")
        return self


class ScoreSummaryV1(EvalContractModel):
    coverage: QualityCoverageV1
    independent_family_count: Count
    business_success: QualityRatioV1
    annotation_coverage: QualityRatioV1
    business_not_assessable: Count
    facts: JudgmentCountsV1
    fact_support: QualityRatioV1
    citations: JudgmentCountsV1
    citation_support: QualityRatioV1
    unknown_fact_inventory_outputs: Count
    required_evidence: QualityRatioV1
    no_answer_handling: QualityRatioV1
    unnecessary_refusal: QualityRatioV1
    unmapped_answer_cases: Count
    draft_distribution: dict[
        Literal["usable", "minor_edits", "major_edits", "unusable", "not_applicable", "unassessed"],
        Count,
    ]
    insufficiency_distribution: dict[
        Literal["appropriate", "unnecessary_refusal", "overclaim", "not_applicable", "unassessed"],
        Count,
    ]
    failure_distribution: dict[
        Literal[
            "configuration",
            "provider",
            "tool",
            "budget",
            "cancelled",
            "safety",
            "business",
            "integrity",
        ],
        Count,
    ]
    cost: CostSummaryV1
    model_calls: Count
    tool_calls: Count
    provider_attempts: Count
    stage_times: dict[
        Literal["ingestion", "retrieval", "generation", "approval", "submission", "total"],
        SampleSummaryV1,
    ]
    retrieval_macro: dict[
        Literal[
            "recall_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank", "required_unit_coverage"
        ],
        SampleSummaryV1,
    ]
    retrieval_micro: dict[
        Literal["recall_at_1", "recall_at_3", "recall_at_5", "required_unit_coverage"],
        QualityRatioV1,
    ]


class ScoredGroupV1(EvalContractModel):
    dimension: Literal["family", "stratum", "split", "tag"]
    group_id: EvalIdentifier
    overlapping: bool
    summary: ScoreSummaryV1


class QualityScoredReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    scorer_version: Literal["quality-score-v1"] = "quality-score-v1"
    manifest: QualityRunManifestV1
    execution_kind: Literal["generation", "retrieval"]
    execution_source_sha: SourceSHA
    scorer_source_sha: SourceSHA
    scorer_change_reason: EvalIdentifier | None
    manifest_digest: EvalDigest
    input_report_digest: EvalDigest
    annotation_digest: EvalDigest | None
    cases: tuple[ScoredCaseV1, ...]
    summary: ScoreSummaryV1
    groups: tuple[ScoredGroupV1, ...]
    ingestion_cost: QualityCostV1
    ingestion_seconds: Seconds
    total_cost: QualityCostV1
    evidence_valid: bool
    execution_measurement_complete: bool
    stop_reason: Failure | None
    measurement_complete: bool
    rubric_frozen: bool
    semantic_quality_claim: bool
    meets_quality_target: None = None
    statistics: Literal["descriptive_family_dependent"] = "descriptive_family_dependent"

    @model_validator(mode="after")
    def identity(self):
        require_unique(
            tuple((c.observation.case_id, c.observation.repeat_index) for c in self.cases)
        )
        require_unique(tuple((g.dimension, g.group_id) for g in self.groups))
        if self.execution_source_sha != self.manifest.execution_source_sha or (
            self.execution_source_sha != self.scorer_source_sha
            and self.scorer_change_reason is None
        ):
            raise ValueError("scorer provenance mismatch")
        if tuple((c.observation.case_id, c.observation.repeat_index) for c in self.cases) != tuple(
            (s.case_id, s.repeat_index) for s in self.manifest.execution_order
        ):
            raise ValueError("scored slots mismatch")
        if self.semantic_quality_claim and (
            self.manifest.llm_mode != "qwen"
            or not self.evidence_valid
            or not self.measurement_complete
            or not self.rubric_frozen
        ):
            raise ValueError("semantic claim requires complete live evidence")
        return self


class PairedCaseV1(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count
    family_id: EvalIdentifier
    stratum: EvalIdentifier
    split: Literal["dev", "validation"]
    tags: tuple[EvalIdentifier, ...]
    left: ScoredCaseV1 | None
    right: ScoredCaseV1 | None
    delta_business: int | None
    delta_fact_support: float | None
    delta_citation_support: float | None
    delta_required_evidence: float | None
    delta_known_cost_cny: Decimal | None
    delta_total_seconds: float | None
    delta_retrieval: dict[
        Literal[
            "recall_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank", "required_unit_coverage"
        ],
        float | None,
    ]


class PairedGroupV1(EvalContractModel):
    dimension: Literal["family", "stratum", "split", "tag"]
    group_id: EvalIdentifier
    overlapping: bool
    planned_pairs: Count
    matched_pairs: Count
    left_only: Count
    right_only: Count
    family_count: Count
    business_changes: dict[Literal["improved", "regressed", "unchanged", "unassessed"], Count]
    differences: dict[
        Literal[
            "fact_support",
            "citation_support",
            "required_evidence",
            "known_cost_cny",
            "total_seconds",
        ],
        SampleSummaryV1,
    ]


class QualityComparisonV1(EvalContractModel):
    schema_version: Literal[1] = 1
    scorer_version: Literal["quality-score-v1"] = "quality-score-v1"
    left_manifest: QualityRunManifestV1
    right_manifest: QualityRunManifestV1
    left_scorer_source_sha: SourceSHA
    right_scorer_source_sha: SourceSHA
    left_digest: EvalDigest
    right_digest: EvalDigest
    pairs: tuple[PairedCaseV1, ...]
    groups: tuple[PairedGroupV1, ...]
    matched_pairs: Count
    left_only: Count
    right_only: Count
    independent_family_count: Count
    left_summary: ScoreSummaryV1
    right_summary: ScoreSummaryV1
    evidence_valid: bool
    statistics: Literal["descriptive_family_dependent"] = "descriptive_family_dependent"
    significance_claim: Literal[False] = False
