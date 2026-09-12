"""E4 quality artifacts. These contracts do not execute or score experiments."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier, EvalQuery, EvalText

QualityVersion = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]{0,127}$")]
SourceSHA = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Count = Annotated[int, Field(ge=0)]
Money = Annotated[Decimal, Field(ge=0, max_digits=20, decimal_places=12)]
Seconds = Annotated[float, Field(ge=0)]
Split = Literal["dev", "validation"]
Layer = Literal["contract", "retrieval", "generation", "product"]
Judgment = Literal["supported", "contradicted", "unsupported", "not_assessable"]
Failure = Literal[
    "configuration", "provider", "tool", "budget", "cancelled", "safety", "business", "integrity"
]


def require_unique(values: tuple[object, ...]) -> None:
    if len(set(values)) != len(values):
        raise ValueError("identities must be unique")


class QualityFileV1(EvalContractModel):
    path: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_./-]+$", max_length=256)]
    digest: EvalDigest
    role: Literal["cases", "source_units", "rubric", "document", "web_fixture"]

    @model_validator(mode="after")
    def relative_path(self) -> QualityFileV1:
        if self.path.startswith("/") or any(p in ("", ".", "..") for p in self.path.split("/")):
            raise ValueError("file path must be a canonical relative path")
        return self


class QualityFamilyV1(EvalContractModel):
    family_id: EvalIdentifier
    split: Split


class QualitySourceV1(EvalContractModel):
    alias: EvalIdentifier
    kind: Literal["resume", "web"]
    path: str
    normalization_version: Literal["nfc-lf-v1"] = "nfc-lf-v1"


class QualityDatasetManifestV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_version: QualityVersion
    purpose: Literal["contract_example", "pilot", "baseline"]
    files: tuple[QualityFileV1, ...] = Field(min_length=1)
    strata: tuple[EvalIdentifier, ...] = Field(min_length=1)
    families: tuple[QualityFamilyV1, ...] = Field(min_length=1)
    split_rule: Literal["family_disjoint"] = "family_disjoint"
    sources: tuple[QualitySourceV1, ...]
    author_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)
    review_status: Literal["not_reviewed", "reviewed"]
    reviewer_ids: tuple[EvalIdentifier, ...]
    review_notes: EvalText
    license_category: Literal["synthetic", "public_licensed", "private_authorized"]
    license_notes: EvalText

    @model_validator(mode="after")
    def identities(self) -> QualityDatasetManifestV1:
        for values in (
            self.strata,
            self.author_ids,
            self.reviewer_ids,
            tuple(f.path for f in self.files),
            tuple(f.family_id for f in self.families),
            tuple(s.alias for s in self.sources),
        ):
            require_unique(values)
        if (self.review_status == "reviewed") != bool(self.reviewer_ids):
            raise ValueError("review status requires matching reviewer identities")
        for role in ("cases", "source_units", "rubric"):
            if sum(f.role == role for f in self.files) != 1:
                raise ValueError("dataset requires one file per structured role")
        roles = {f.path: f.role for f in self.files}
        for source in self.sources:
            expected = "document" if source.kind == "resume" else "web_fixture"
            if roles.get(source.path) != expected:
                raise ValueError("source must reference the matching declared file role")
        return self


class QualityCaseV1(EvalContractModel):
    schema_version: Literal[1] = 1
    case_id: EvalIdentifier
    family_id: EvalIdentifier
    stratum: EvalIdentifier
    tags: tuple[EvalIdentifier, ...]
    split: Split
    mode: Literal["research", "application"]
    query: EvalQuery
    resume_alias: EvalIdentifier | None
    web_scenario_alias: EvalIdentifier | None
    required_unit_ids: tuple[EvalIdentifier, ...]
    expected_behavior: EvalIdentifier
    scope_expectation: Literal["allowed", "reject"] = "allowed"

    @model_validator(mode="after")
    def unique_labels(self) -> QualityCaseV1:
        require_unique(self.tags)
        require_unique(self.required_unit_ids)
        return self


class SourceEvidenceUnitV1(EvalContractModel):
    schema_version: Literal[1] = 1
    unit_id: EvalIdentifier
    source_alias: EvalIdentifier
    source_kind: Literal["resume", "web"]
    normalized_text_digest: EvalDigest
    start: Count
    end: Count
    quote: EvalText
    fact: EvalText
    alternative_unit_ids: tuple[EvalIdentifier, ...] = ()

    @model_validator(mode="after")
    def evidence_range(self) -> SourceEvidenceUnitV1:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("evidence range must match quote code point length")
        require_unique(self.alternative_unit_ids)
        if self.unit_id in self.alternative_unit_ids:
            raise ValueError("evidence cannot substitute itself")
        return self


class QualityRubricDimensionV1(EvalContractModel):
    dimension_id: Literal[
        "fact_support", "citation_support", "completeness", "insufficiency", "draft", "safety"
    ]
    labels: tuple[EvalIdentifier, ...] = Field(min_length=1)
    instructions: EvalText

    @model_validator(mode="after")
    def labels_unique(self) -> QualityRubricDimensionV1:
        require_unique(self.labels)
        return self


class QualityRubricV1(EvalContractModel):
    schema_version: Literal[1] = 1
    rubric_version: QualityVersion
    status: Literal["draft", "frozen"]
    dimensions: tuple[QualityRubricDimensionV1, ...] = Field(min_length=6, max_length=6)
    review_policy: Literal["single_reviewer", "independent_review_with_adjudication"]

    @model_validator(mode="after")
    def dimensions_unique(self) -> QualityRubricV1:
        require_unique(tuple(d.dimension_id for d in self.dimensions))
        support = {"supported", "contradicted", "unsupported", "not_assessable"}
        labels = {
            "fact_support": support,
            "citation_support": support,
            "completeness": {"covered", "missing", "not_assessed"},
            "insufficiency": {"appropriate", "unnecessary_refusal", "overclaim", "not_applicable"},
            "draft": {"usable", "minor_edits", "major_edits", "unusable", "not_applicable"},
            "safety": {
                "unauthorized_access",
                "secret_leak",
                "unapproved_action",
                "gold_contamination",
            },
        }
        if any(set(d.labels) != labels[d.dimension_id] for d in self.dimensions):
            raise ValueError("rubric labels must match the v1 annotation vocabulary")
        return self


class QualityExecutionSlotV1(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count


class QualityRunManifestV1(EvalContractModel):
    schema_version: Literal[1] = 1
    experiment_id: EvalIdentifier
    execution_source_sha: SourceSHA
    suite_version: QualityVersion
    dataset_version: QualityVersion
    dataset_digest: EvalDigest
    case_set_digest: EvalDigest
    split_digest: EvalDigest
    rubric_version: QualityVersion
    rubric_digest: EvalDigest
    model: QualityVersion
    prompt_digest: EvalDigest
    graph_version: QualityVersion
    embedding_profile: QualityVersion | None
    retrieval_policy_digest: EvalDigest | None
    configuration_digest: EvalDigest
    measurement_scope: Layer
    llm_mode: Literal["fake", "qwen"]
    web_mode: Literal["none", "frozen_fixture", "tavily"]
    document_mode: Literal["none", "fixture", "fake_embedding_db", "real_embedding_db"]
    selected_case_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)
    repeat_count: int = Field(gt=0)
    execution_order: tuple[QualityExecutionSlotV1, ...] = Field(min_length=1)
    cost_admission_budget_cny: Decimal = Field(gt=0)
    provider_attempt_cap: int = Field(gt=0)
    input_token_cap: int = Field(gt=0)
    output_token_cap: int = Field(gt=0)

    @model_validator(mode="after")
    def exact_order(self) -> QualityRunManifestV1:
        require_unique(self.selected_case_ids)
        actual = tuple((s.case_id, s.repeat_index) for s in self.execution_order)
        require_unique(actual)
        expected = {
            (case_id, repeat)
            for case_id in self.selected_case_ids
            for repeat in range(self.repeat_count)
        }
        if set(actual) != expected:
            raise ValueError("execution order must cover all case/repeat slots exactly once")
        if (self.embedding_profile is None) != (self.retrieval_policy_digest is None):
            raise ValueError("embedding and retrieval identities must be paired")
        if self.document_mode.endswith("embedding_db") and self.embedding_profile is None:
            raise ValueError("database retrieval requires its effective profile")
        if self.measurement_scope == "contract" and (
            self.llm_mode != "fake"
            or self.web_mode == "tavily"
            or self.document_mode == "real_embedding_db"
        ):
            raise ValueError("contract measurement must remain offline")
        return self


class QualityCostV1(EvalContractModel):
    scope: Literal["chat", "embedding", "tools", "all"]
    known_cost_cny: Money
    priced_attempts: Count
    unknown_cost_attempts: Count
    total_cost_cny: Money | None

    @model_validator(mode="after")
    def honest_total(self) -> QualityCostV1:
        if self.unknown_cost_attempts:
            if self.total_cost_cny is not None:
                raise ValueError("unknown cost requires a null total")
        elif self.total_cost_cny != self.known_cost_cny:
            raise ValueError("fully priced total must equal known cost")
        if not self.priced_attempts and self.known_cost_cny:
            raise ValueError("known cost requires priced attempts")
        return self


class QualityStageTimeV1(EvalContractModel):
    stage: Literal["ingestion", "retrieval", "generation", "approval", "submission", "total"]
    seconds: Seconds


class QualitySafetyCountsV1(EvalContractModel):
    unauthorized_access: Count = 0
    secret_leak: Count = 0
    unapproved_action: Count = 0
    gold_contamination: Count = 0


class QualityObservationV1(EvalContractModel):
    schema_version: Literal[1] = 1
    experiment_id: EvalIdentifier
    case_id: EvalIdentifier
    repeat_index: Count
    measurement_scope: Layer
    status: Literal["succeeded", "failed", "not_run"]
    failure_type: Failure | None
    safety: QualitySafetyCountsV1
    tool_calls: Count
    model_calls: Count
    provider_attempts: Count
    input_tokens: Count
    output_tokens: Count
    cost: QualityCostV1
    stage_times: tuple[QualityStageTimeV1, ...]
    output_digest: EvalDigest | None
    assessment_required: bool

    @model_validator(mode="after")
    def consistent_observation(self) -> QualityObservationV1:
        require_unique(tuple(t.stage for t in self.stage_times))
        if self.cost.scope != "all":
            raise ValueError("observation cost must cover all attempts")
        if self.provider_attempts != self.cost.priced_attempts + self.cost.unknown_cost_attempts:
            raise ValueError("cost coverage must account for every provider attempt")
        if self.model_calls > self.provider_attempts:
            raise ValueError("model calls cannot exceed provider attempts")
        if self.status == "succeeded" and self.failure_type is not None:
            raise ValueError("successful observation cannot carry a failure")
        if self.status == "failed" and self.failure_type is None:
            raise ValueError("failed observation requires a failure category")
        if self.assessment_required and self.output_digest is None:
            raise ValueError("assessment requires an actual output")
        if self.status == "not_run" and (
            self.output_digest is not None
            or self.tool_calls
            or self.model_calls
            or self.provider_attempts
            or self.input_tokens
            or self.output_tokens
            or self.stage_times
            or any(self.safety.model_dump().values())
        ):
            raise ValueError("unexecuted slots cannot carry execution evidence")
        if any(self.safety.model_dump().values()) and self.failure_type != "safety":
            raise ValueError("safety violations require a safety failure")
        return self


class QualityFactJudgmentV1(EvalContractModel):
    fact_id: EvalIdentifier
    judgment: Judgment


class QualityCitationJudgmentV1(EvalContractModel):
    fact_id: EvalIdentifier
    citation_id: EvalIdentifier
    judgment: Judgment


class QualityAdjudicationV1(EvalContractModel):
    disagreement_id: EvalIdentifier
    reviewer_ids: tuple[EvalIdentifier, ...] = Field(min_length=2)
    status: Literal["unresolved", "resolved"]
    adjudicator_id: EvalIdentifier | None
    resolution_code: EvalIdentifier | None

    @model_validator(mode="after")
    def resolution(self) -> QualityAdjudicationV1:
        require_unique(self.reviewer_ids)
        resolved = self.status == "resolved"
        if (self.adjudicator_id is not None) != resolved or (
            self.resolution_code is not None
        ) != resolved:
            raise ValueError("resolved disagreements require an adjudicator and resolution")
        return self


class HumanAnnotationV1(EvalContractModel):
    schema_version: Literal[1] = 1
    experiment_id: EvalIdentifier
    case_id: EvalIdentifier
    repeat_index: Count
    output_digest: EvalDigest
    reviewer_id: EvalIdentifier
    rubric_version: QualityVersion
    facts: tuple[QualityFactJudgmentV1, ...]
    citations: tuple[QualityCitationJudgmentV1, ...]
    covered_unit_ids: tuple[EvalIdentifier, ...]
    missing_unit_ids: tuple[EvalIdentifier, ...]
    unassessed_fact_count: Count
    unassessed_citation_count: Count
    business_result: Literal["success", "failure", "not_assessable"]
    insufficiency: Literal["appropriate", "unnecessary_refusal", "overclaim", "not_applicable"]
    draft_usability: Literal["usable", "minor_edits", "major_edits", "unusable", "not_applicable"]
    reason_codes: tuple[EvalIdentifier, ...]
    disagreements: tuple[QualityAdjudicationV1, ...]
    assessment_complete: bool

    @model_validator(mode="after")
    def annotation_consistency(self) -> HumanAnnotationV1:
        fact_ids = tuple(f.fact_id for f in self.facts)
        for values in (
            fact_ids,
            tuple((c.fact_id, c.citation_id) for c in self.citations),
            self.covered_unit_ids + self.missing_unit_ids,
            self.reason_codes,
            tuple(d.disagreement_id for d in self.disagreements),
        ):
            require_unique(values)
        if any(c.fact_id not in fact_ids for c in self.citations):
            raise ValueError("citation judgment requires a declared fact")
        if self.assessment_complete and (
            self.unassessed_fact_count
            or self.unassessed_citation_count
            or any(d.status == "unresolved" for d in self.disagreements)
        ):
            raise ValueError("incomplete judgments cannot be marked assessed")
        return self


class QualityRatioV1(EvalContractModel):
    numerator: Count
    denominator: Count
    value: Annotated[float, Field(ge=0, le=1)] | None
    unassessed_count: Count
    not_applicable_reason: Literal["zero_denominator"] | None

    @model_validator(mode="after")
    def denominator_semantics(self) -> QualityRatioV1:
        if self.numerator > self.denominator:
            raise ValueError("numerator exceeds denominator")
        if self.denominator == 0:
            if self.value is not None or self.not_applicable_reason != "zero_denominator":
                raise ValueError("zero denominator requires null and a reason")
        elif (
            self.value != self.numerator / self.denominator
            or self.not_applicable_reason is not None
        ):
            raise ValueError("ratio must agree with its counts")
        return self


class QualityCoverageV1(EvalContractModel):
    planned: Count
    executed: Count
    failed: Count
    not_run: Count
    generated: Count
    assessment_required: Count
    assessed: Count
    unassessed: Count

    @model_validator(mode="after")
    def coverage_counts(self) -> QualityCoverageV1:
        if (
            self.planned != self.executed + self.not_run
            or self.failed > self.executed
            or self.generated > self.executed
            or self.assessment_required > self.generated
            or self.assessed + self.unassessed != self.assessment_required
        ):
            raise ValueError("coverage counts are inconsistent")
        return self


class QualityGroupResultV1(EvalContractModel):
    dimension: Literal["stratum", "family", "split"]
    group_id: EvalIdentifier
    coverage: QualityCoverageV1
    business_success: QualityRatioV1

    @model_validator(mode="after")
    def business_denominator(self) -> QualityGroupResultV1:
        if self.business_success.denominator != self.coverage.executed:
            raise ValueError("business denominator must include all executed cases")
        return self


class QualityReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    experiment_id: EvalIdentifier
    execution_source_sha: SourceSHA
    scorer_source_sha: SourceSHA
    scorer_change_reason: EvalIdentifier | None
    manifest_digest: EvalDigest
    annotation_digest: EvalDigest | None
    observations: tuple[QualityObservationV1, ...]
    coverage: QualityCoverageV1
    groups: tuple[QualityGroupResultV1, ...]
    evidence_valid: bool
    measurement_complete: bool
    target_policy_digest: EvalDigest | None
    meets_quality_target: bool | None
    stop_reason: Failure | None

    @model_validator(mode="after")
    def report_consistency(self) -> QualityReportV1:
        require_unique(tuple((o.case_id, o.repeat_index) for o in self.observations))
        require_unique(tuple((g.dimension, g.group_id) for g in self.groups))
        if any(o.experiment_id != self.experiment_id for o in self.observations):
            raise ValueError("observations belong to another experiment")
        if (
            self.execution_source_sha != self.scorer_source_sha
            and self.scorer_change_reason is None
        ):
            raise ValueError("changed scorer requires an explanation code")
        c = self.coverage
        if (
            len(self.observations) != c.planned
            or sum(o.status != "not_run" for o in self.observations) != c.executed
            or sum(o.status == "failed" for o in self.observations) != c.failed
            or sum(o.output_digest is not None for o in self.observations) != c.generated
            or sum(o.assessment_required for o in self.observations) != c.assessment_required
        ):
            raise ValueError("coverage must match the raw observation index")
        if c.assessed and self.annotation_digest is None:
            raise ValueError("assessed outputs require annotation identity")
        if self.measurement_complete != (c.not_run == 0 and c.unassessed == 0):
            raise ValueError("completeness must match execution and assessment coverage")
        if self.meets_quality_target is not None and (
            not self.evidence_valid
            or not self.measurement_complete
            or self.target_policy_digest is None
        ):
            raise ValueError("target judgment requires valid complete evidence and a locked policy")
        if any(any(o.safety.model_dump().values()) for o in self.observations):
            if self.stop_reason != "safety" or self.meets_quality_target is True:
                raise ValueError("safety violations block normal acceptance")
        if any(o.safety.gold_contamination for o in self.observations) and self.evidence_valid:
            raise ValueError("gold contamination invalidates quality evidence")
        if self.evidence_valid and any(
            o.failure_type in ("configuration", "integrity") for o in self.observations
        ):
            raise ValueError("configuration or integrity failure invalidates quality evidence")
        for dimension in {g.dimension for g in self.groups}:
            groups = [g for g in self.groups if g.dimension == dimension]
            for field in QualityCoverageV1.model_fields:
                if sum(getattr(g.coverage, field) for g in groups) != getattr(c, field):
                    raise ValueError("each grouping must partition the overall coverage")
        return self


class QualityModelPayloadV1(EvalContractModel):
    """Only case-owned model input; runtime identities are supplied by trusted code."""

    mode: Literal["research", "application"]
    query: EvalQuery


class QualityChunkSpanV1(EvalContractModel):
    """An exact, equal-length fragment; offsets are Unicode code points."""

    source_start: Count
    source_end: Count
    chunk_start: Count
    chunk_end: Count

    @model_validator(mode="after")
    def valid_range(self) -> QualityChunkSpanV1:
        if (
            self.source_end <= self.source_start
            or self.chunk_end <= self.chunk_start
            or self.source_end - self.source_start != self.chunk_end - self.chunk_start
        ):
            raise ValueError("invalid mapping range")
        return self


class QualityChunkMappingV1(EvalContractModel):
    ordinal: Count
    content_digest: EvalDigest
    codepoint_count: Annotated[int, Field(gt=0)]
    candidate_spans: tuple[QualityChunkSpanV1, ...]
    spans: tuple[QualityChunkSpanV1, ...]
    resolution: Literal["unique_exact", "needs_review", "reviewed"]
    reviewer_id: EvalIdentifier | None = None

    @model_validator(mode="after")
    def review_binding(self) -> QualityChunkMappingV1:
        if (self.resolution == "reviewed") != (self.reviewer_id is not None):
            raise ValueError("mapping review identity mismatch")
        if self.resolution == "needs_review" and self.spans:
            raise ValueError("unresolved candidates cannot provide coverage")
        if self.resolution != "needs_review" and not self.spans:
            raise ValueError("resolved mapping requires fragments")
        return self


class QualitySourceMappingV1(EvalContractModel):
    source_alias: EvalIdentifier
    normalized_text_digest: EvalDigest
    chunks: tuple[QualityChunkMappingV1, ...] = Field(min_length=1)


class QualityUnitMappingV1(EvalContractModel):
    unit_id: EvalIdentifier
    source_alias: EvalIdentifier
    status: Literal["complete", "partial", "unmapped", "not_applicable"]
    covered_codepoints: Count
    chunk_ordinals: tuple[Count, ...]


class QualityChunkJudgmentV1(EvalContractModel):
    case_id: EvalIdentifier
    source_alias: EvalIdentifier
    chunk_ordinal: Count
    relevance: Literal["relevant", "irrelevant", "unjudged"]
    reason: Literal["required_evidence", "task_context", "off_topic", "not_judged"]
    reviewer_id: EvalIdentifier | None = None

    @model_validator(mode="after")
    def judgment_binding(self) -> QualityChunkJudgmentV1:
        reasons = {
            "relevant": {"required_evidence", "task_context"},
            "irrelevant": {"off_topic"},
            "unjudged": {"not_judged"},
        }
        if self.reason not in reasons[self.relevance] or (
            (self.relevance == "unjudged") != (self.reviewer_id is None)
        ):
            raise ValueError("invalid relevance judgment")
        return self


class QualityMappingV1(EvalContractModel):
    schema_version: Literal[1] = 1
    mapping_version: QualityVersion
    execution_source_sha: SourceSHA
    dataset_version: QualityVersion
    dataset_digest: EvalDigest
    normalization_version: QualityVersion
    chunking_version: QualityVersion
    rules_digest: EvalDigest
    sources: tuple[QualitySourceMappingV1, ...]
    units: tuple[QualityUnitMappingV1, ...]
    judgments: tuple[QualityChunkJudgmentV1, ...]

    @model_validator(mode="after")
    def mapping_identities(self) -> QualityMappingV1:
        require_unique(tuple(s.source_alias for s in self.sources))
        require_unique(tuple(u.unit_id for u in self.units))
        require_unique(tuple((j.case_id, j.source_alias, j.chunk_ordinal) for j in self.judgments))
        for source in self.sources:
            require_unique(tuple(c.ordinal for c in source.chunks))
        return self


class QualityMappingReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    mapping_digest: EvalDigest
    total_units: Count
    complete_units: Count
    partial_units: Count
    unmapped_units: Count
    not_applicable_units: Count
    total_chunks: Count
    ambiguous_chunks: Count
    unresolved_chunks: Count
    pending_review_chunks: Count
    relevant_contexts: Count
    irrelevant_contexts: Count
    unjudged_contexts: Count
    mapping_complete: bool
    review_complete: bool


class QualityRetrievalPolicyV1(EvalContractModel):
    policy_version: Literal["quality-retrieval-v1"] = "quality-retrieval-v1"
    normalization_version: Literal["nfc-lf-v1"] = "nfc-lf-v1"
    chunking_version: Literal["heading-paragraph-utf8-budget-800-v1"] = (
        "heading-paragraph-utf8-budget-800-v1"
    )
    embedding_profile: Literal["qwen-beijing-text-embedding-v4-1536-v1"] = (
        "qwen-beijing-text-embedding-v4-1536-v1"
    )
    embedding_dimension: Literal[1536] = 1536
    top_k: Literal[5] = 5
    chunk_byte_limit: Literal[800] = 800
    context_byte_limit: Literal[4000] = 4000
    unknown_attempt_reserve_cny: Annotated[Decimal, Field(gt=0)]


class QualityRetrievalAdmissionV1(QualityRunManifestV1):
    """Internal view for the existing attempt recorder; not a new billing authority."""

    unknown_attempt_reserve_cny: Annotated[Decimal, Field(gt=0)]


class QualityRetrievalStartV1(EvalContractModel):
    manifest: QualityRunManifestV1
    policy: QualityRetrievalPolicyV1
    mapping_digest: EvalDigest


class QualityRetrievalRefV1(EvalContractModel):
    source_alias: EvalIdentifier
    chunk_ordinal: Count
    exposed_digest: EvalDigest
    exposed_codepoints: Annotated[int, Field(gt=0)]
    exposed_bytes: Annotated[int, Field(gt=0, le=800)]


class QualityRetrievalMetricsV1(EvalContractModel):
    recall_at_1: QualityRatioV1
    recall_at_3: QualityRatioV1
    recall_at_5: QualityRatioV1
    reciprocal_rank: Annotated[float, Field(ge=0, le=1)] | None
    required_unit_coverage: QualityRatioV1
    covered_unit_ids: tuple[EvalIdentifier, ...]
    web_required_units: Count
    relevant_contexts: Count
    irrelevant_contexts: Count
    unjudged_contexts: Count
    returned_context_bytes: Annotated[int, Field(ge=0, le=4000)]
    no_relevant_evidence: bool

    @model_validator(mode="after")
    def consistent_metrics(self) -> QualityRetrievalMetricsV1:
        require_unique(self.covered_unit_ids)
        recalls = (self.recall_at_1, self.recall_at_3, self.recall_at_5)
        if (
            len({r.denominator for r in recalls}) != 1
            or not recalls[0].numerator <= recalls[1].numerator <= recalls[2].numerator
            or any(r.numerator > k for r, k in zip(recalls, (1, 3, 5), strict=True))
            or (self.reciprocal_rank is None) != (recalls[0].denominator == 0)
            or self.no_relevant_evidence != (recalls[0].denominator == 0)
            or len(self.covered_unit_ids) != self.required_unit_coverage.numerator
        ):
            raise ValueError("inconsistent retrieval metrics")
        return self


class QualityRetrievalCaseV1(EvalContractModel):
    observation: QualityObservationV1
    refs: tuple[QualityRetrievalRefV1, ...]
    metrics: QualityRetrievalMetricsV1 | None

    @model_validator(mode="after")
    def measured_success_only(self) -> QualityRetrievalCaseV1:
        require_unique(tuple((r.source_alias, r.chunk_ordinal) for r in self.refs))
        if (self.metrics is not None) != (self.observation.status == "succeeded"):
            raise ValueError("only successful retrieval has measured metrics")
        if self.metrics is None and self.refs:
            raise ValueError("failed retrieval must not publish unverified references")
        if self.metrics is not None and (
            self.metrics.returned_context_bytes != sum(r.exposed_bytes for r in self.refs)
            or len(self.refs)
            != self.metrics.relevant_contexts
            + self.metrics.irrelevant_contexts
            + self.metrics.unjudged_contexts
        ):
            raise ValueError("context counts must match exposed references")
        if self.observation.assessment_required or self.observation.tool_calls:
            raise ValueError("retrieval-only evidence has no output review or model tool calls")
        return self


class QualityRetrievalRepresentationV1(EvalContractModel):
    source_alias: EvalIdentifier
    normalized_text_digest: EvalDigest
    chunk_digests: tuple[EvalDigest, ...] = Field(min_length=1)
    chunk_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def chunk_count_matches(self) -> QualityRetrievalRepresentationV1:
        if self.chunk_count != len(self.chunk_digests):
            raise ValueError("representation count must match actual chunks")
        return self


class QualityRetrievalUsageV1(EvalContractModel):
    provider_attempts: Count
    input_tokens: Count
    output_tokens: Count
    unknown_usage_attempts: Count
    cost: QualityCostV1
    accounting_complete: bool

    @model_validator(mode="after")
    def attempt_coverage(self) -> QualityRetrievalUsageV1:
        if (
            self.provider_attempts != self.cost.priced_attempts + self.cost.unknown_cost_attempts
            or self.unknown_usage_attempts > self.provider_attempts
        ):
            raise ValueError("attempt accounting coverage mismatch")
        return self


class QualityRetrievalReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    manifest: QualityRunManifestV1
    mapping_digest: EvalDigest
    policy: QualityRetrievalPolicyV1
    semantic_quality_claim: bool
    representations: tuple[QualityRetrievalRepresentationV1, ...]
    representation_complete: bool
    ingestion_usage: QualityRetrievalUsageV1
    total_usage: QualityRetrievalUsageV1
    ingestion_seconds: Seconds
    cases: tuple[QualityRetrievalCaseV1, ...]
    full_coverage_cases: QualityRatioV1
    executed: Count
    failed: Count
    not_run: Count
    scope_leakage_count: Count
    evidence_valid: bool
    measurement_complete: bool
    meets_quality_target: None = None
    stop_reason: Failure | None

    @model_validator(mode="after")
    def report_invariants(self) -> QualityRetrievalReportV1:
        observations = tuple(c.observation for c in self.cases)
        if (
            tuple((o.case_id, o.repeat_index) for o in observations)
            != tuple((s.case_id, s.repeat_index) for s in self.manifest.execution_order)
            or any(o.experiment_id != self.manifest.experiment_id for o in observations)
            or self.executed != sum(o.status != "not_run" for o in observations)
            or self.failed != sum(o.status == "failed" for o in observations)
            or self.not_run != sum(o.status == "not_run" for o in observations)
            or self.scope_leakage_count != sum(o.safety.unauthorized_access for o in observations)
        ):
            raise ValueError("retrieval report does not match planned observations")
        require_unique(tuple(r.source_alias for r in self.representations))
        if self.semantic_quality_claim and (
            self.manifest.llm_mode != "qwen"
            or not self.evidence_valid
            or not self.measurement_complete
            or not any(o.status == "succeeded" and o.provider_attempts for o in observations)
        ):
            raise ValueError("semantic evidence requires a valid complete live measurement")
        if self.measurement_complete != (
            self.representation_complete and self.not_run == 0 and self.stop_reason is None
        ):
            raise ValueError("retrieval completeness mismatch")
        if self.evidence_valid and (
            self.stop_reason in ("configuration", "integrity", "safety")
            or not self.total_usage.accounting_complete
            or any(o.failure_type in ("configuration", "integrity", "safety") for o in observations)
        ):
            raise ValueError("invalid evidence cannot claim validity")
        if self.scope_leakage_count and self.stop_reason != "safety":
            raise ValueError("scope leakage must stop the experiment")
        if self.total_usage.provider_attempts != (
            self.ingestion_usage.provider_attempts + sum(o.provider_attempts for o in observations)
        ):
            raise ValueError("total attempts must include ingestion and all query slots")
        for field in ("input_tokens", "output_tokens"):
            if getattr(self.total_usage, field) != (
                getattr(self.ingestion_usage, field) + sum(getattr(o, field) for o in observations)
            ):
                raise ValueError("total tokens must include ingestion and all query slots")
        for field in ("known_cost_cny", "priced_attempts", "unknown_cost_attempts"):
            if getattr(self.total_usage.cost, field) != (
                getattr(self.ingestion_usage.cost, field)
                + sum(getattr(o.cost, field) for o in observations)
            ):
                raise ValueError("total costs must include ingestion and all query slots")
        applicable = [
            c.metrics.required_unit_coverage
            for c in self.cases
            if c.metrics is not None and c.metrics.required_unit_coverage.denominator
        ]
        if (
            self.full_coverage_cases.denominator != len(applicable)
            or self.full_coverage_cases.numerator
            != sum(r.numerator == r.denominator for r in applicable)
            or self.full_coverage_cases.unassessed_count != self.failed + self.not_run
        ):
            raise ValueError("full coverage denominator mismatch")
        return self


class QualityGenerationPolicyV1(EvalContractModel):
    policy_version: Literal["quality-generation-v1"] = "quality-generation-v1"
    retrieval: QualityRetrievalPolicyV1
    case_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 600.0
    max_model_calls: Literal[12] = 12
    max_tool_calls: Literal[8] = 8
    max_tool_results: Literal[8] = 8
    max_iterations: Literal[24] = 24
    web_resolution: Literal["case_bound_source_v1"] = "case_bound_source_v1"


class QualityGenerationStartV1(EvalContractModel):
    manifest: QualityRunManifestV1
    policy: QualityGenerationPolicyV1
    mapping_digest: EvalDigest
    plan_prompt_digest: EvalDigest
    research_prompt_digest: EvalDigest
    writer_prompt_digest: EvalDigest
    reasoning_effort: Literal["medium"] = "medium"
    max_output_tokens: Literal[4096] = 4096
    seed: None = None
    temperature: None = None
    provider_deterministic: Literal[False] = False


class QualityPrivateFileV1(EvalContractModel):
    name: Annotated[
        str,
        StringConstraints(pattern=r"^(sources|manifest|case-[0-9]{4,}|output-[0-9]{4,})\.json$"),
    ]
    digest: EvalDigest
    byte_count: Count


class QualityGenerationCaseV1(EvalContractModel):
    observation: QualityObservationV1
    scope_rejected: bool = False
    private_case_digest: EvalDigest | None = None
    tool_accounting_complete: bool = True

    @model_validator(mode="after")
    def generation_output_binding(self) -> QualityGenerationCaseV1:
        o = self.observation
        if o.status == "succeeded" and o.output_digest is None:
            raise ValueError("successful generation requires an output")
        if o.assessment_required != (o.output_digest is not None):
            raise ValueError("every saved generation output requires review")
        if o.status == "not_run" and self.private_case_digest is not None:
            raise ValueError("unexecuted slot has no private case")
        if self.scope_rejected and (
            o.status != "failed"
            or o.failure_type != "business"
            or o.provider_attempts
            or o.tool_calls
        ):
            raise ValueError("scope rejection must precede execution")
        return self


class QualityPrivateSourceV1(EvalContractModel):
    source_alias: EvalIdentifier
    kind: Literal["resume", "web"]
    text: str
    digest: EvalDigest


class QualityPrivateSourcesV1(EvalContractModel):
    sources: tuple[QualityPrivateSourceV1, ...]


class QualityPrivateToolV1(EvalContractModel):
    tool_name: Literal["search_web", "retrieve_documents"]
    arguments: dict[str, JsonValue]
    result: str | None
    delivered: bool

    @model_validator(mode="after")
    def delivery(self) -> QualityPrivateToolV1:
        if self.delivered != (self.result is not None):
            raise ValueError("only delivered evidence has a result")
        return self


class QualityPrivateCaseV1(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count
    input: QualityModelPayloadV1
    resume_alias: EvalIdentifier | None
    web_scenario_alias: EvalIdentifier | None
    tools: tuple[QualityPrivateToolV1, ...]
    output_digest: EvalDigest | None
    failure_type: Failure | None


class QualityPrivateOutputV1(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count
    output: dict[str, JsonValue]


class QualityGenerationReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    start: QualityGenerationStartV1
    representations: tuple[QualityRetrievalRepresentationV1, ...]
    representation_complete: bool
    ingestion_usage: QualityRetrievalUsageV1
    total_usage: QualityRetrievalUsageV1
    ingestion_seconds: Seconds
    cases: tuple[QualityGenerationCaseV1, ...]
    private_files: tuple[QualityPrivateFileV1, ...]
    executed: Count
    failed: Count
    not_run: Count
    evidence_valid: bool
    measurement_complete: bool
    annotation_complete: Literal[False] = False
    semantic_quality_claim: Literal[False] = False
    meets_quality_target: None = None
    stop_reason: Failure | None

    @model_validator(mode="after")
    def generation_report_invariants(self) -> QualityGenerationReportV1:
        manifest = self.start.manifest
        observations = tuple(c.observation for c in self.cases)
        if (
            tuple((o.case_id, o.repeat_index) for o in observations)
            != tuple((s.case_id, s.repeat_index) for s in manifest.execution_order)
            or any(o.experiment_id != manifest.experiment_id for o in observations)
            or any(o.measurement_scope != manifest.measurement_scope for o in observations)
            or self.executed != sum(o.status != "not_run" for o in observations)
            or self.failed != sum(o.status == "failed" for o in observations)
            or self.not_run != sum(o.status == "not_run" for o in observations)
        ):
            raise ValueError("generation observations must match the frozen plan")
        require_unique(tuple(f.name for f in self.private_files))
        require_unique(tuple(r.source_alias for r in self.representations))
        files = {f.name: f.digest for f in self.private_files}
        for index, case in enumerate(self.cases):
            if case.observation.output_digest != files.get(f"output-{index:04d}.json"):
                raise ValueError("output digest must reference the saved private file")
            if case.private_case_digest != files.get(f"case-{index:04d}.json"):
                raise ValueError("case digest must reference the saved private file")
        if self.measurement_complete != (
            self.representation_complete and not self.not_run and self.stop_reason is None
        ):
            raise ValueError("generation completeness mismatch")
        if self.evidence_valid and (
            self.stop_reason in {"configuration", "integrity", "safety"}
            or not self.total_usage.accounting_complete
            or any(not c.tool_accounting_complete for c in self.cases)
            or any(o.failure_type in {"configuration", "integrity", "safety"} for o in observations)
        ):
            raise ValueError("invalid generation evidence cannot be accepted")
        for field in ("provider_attempts", "input_tokens", "output_tokens"):
            if getattr(self.total_usage, field) != getattr(self.ingestion_usage, field) + sum(
                getattr(o, field) for o in observations
            ):
                raise ValueError("generation totals must cover ingestion and all slots")
        for field in ("known_cost_cny", "priced_attempts", "unknown_cost_attempts"):
            if getattr(self.total_usage.cost, field) != getattr(
                self.ingestion_usage.cost, field
            ) + sum(getattr(o.cost, field) for o in observations):
                raise ValueError("generation costs must cover ingestion and all slots")
        return self
