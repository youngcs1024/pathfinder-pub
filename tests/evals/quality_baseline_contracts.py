"""E4.10 offline evidence contracts, separate from historical Gate 11 acceptance."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.quality_contracts import QualityRunManifestV1, SourceSHA, require_unique
from tests.evals.quality_score_contracts import QualityScoredReportV1

Layer = Literal["retrieval", "generation"]


class QualityTargetV1(EvalContractModel):
    layer: Layer
    metric: Literal["recall_at_5", "business_success", "fact_support", "citation_support"]
    minimum: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def applicable(self):
        if (self.layer == "retrieval") != (self.metric == "recall_at_5"):
            raise ValueError("target layer mismatch")
        return self


class QualityPreparationV1(EvalContractModel):
    schema_version: Literal[1] = 1
    preparation_id: EvalIdentifier
    preparation_source_sha: SourceSHA
    freeze_digest: EvalDigest
    mapping_digest: EvalDigest
    selected_case_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)
    validation_case_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)
    repeat_count: Annotated[int, Field(gt=0)]
    manifests: tuple[QualityRunManifestV1, QualityRunManifestV1]
    targets: tuple[QualityTargetV1, ...] = ()
    live_authorized: Literal[False] = False
    measurement_complete: Literal[False] = False
    baseline_accepted: Literal[False] = False

    @model_validator(mode="after")
    def scope(self):
        require_unique(self.selected_case_ids)
        require_unique(self.validation_case_ids)
        require_unique(tuple((t.layer, t.metric) for t in self.targets))
        if not set(self.validation_case_ids) <= set(self.selected_case_ids):
            raise ValueError("validation selection mismatch")
        if any(
            m.measurement_scope not in (layer, "contract")
            for m, layer in zip(self.manifests, ("retrieval", "generation"), strict=True)
        ):
            raise ValueError("both ordered layers required")
        require_unique(tuple(m.experiment_id for m in self.manifests))
        first = self.manifests[0]
        for m in self.manifests:
            if m.selected_case_ids != self.selected_case_ids or m.repeat_count != self.repeat_count:
                raise ValueError("selection mismatch")
            if m.execution_order != first.execution_order:
                raise ValueError("layer order mismatch")
            for field in ("dataset_digest", "rubric_digest", "split_digest"):
                if getattr(m, field) != getattr(first, field):
                    raise ValueError("layer identity mismatch")
        return self


class QualityEvidencePathsV1(EvalContractModel):
    """Private operator input only: never included in a public artifact or diagnostic."""

    retrieval_report: str
    generation_report: str
    generation_private_dir: str
    annotations: str
    retrieval_score: str
    generation_score: str


Blocker = Literal[
    "validation_exposed",
    "not_live",
    "invalid_evidence",
    "incomplete_measurement",
    "missing_assessment",
    "safety_violation",
    "missing_live_attempts",
]


class QualityCandidateV1(EvalContractModel):
    schema_version: Literal[1] = 1
    preparation: QualityPreparationV1
    preparation_digest: EvalDigest
    private_artifacts_digest: EvalDigest
    scores: tuple[QualityScoredReportV1, QualityScoredReportV1]
    blockers: tuple[Blocker, ...]
    evidence_valid: bool
    measurement_complete: bool
    meets_quality_target: bool | None
    acceptance_ready: bool
    baseline_accepted: Literal[False] = False


class QualityBaselineDiffV1(EvalContractModel):
    schema_version: Literal[1] = 1
    candidate_digest: EvalDigest
    previous_baseline_digest: EvalDigest | None
    first_acceptance: bool
    scope_changed: bool
    versions_changed: bool
    results_changed: bool
    previous: QualityCandidateV1 | None
    current: QualityCandidateV1


class QualityAcceptedBaselineV1(EvalContractModel):
    schema_version: Literal[1] = 1
    baseline_id: EvalIdentifier
    accepting_source_sha: SourceSHA
    candidate: QualityCandidateV1
    candidate_digest: EvalDigest
    diff_digest: EvalDigest
    previous_baseline_digest: EvalDigest | None
    human_output_review_confirmed: Literal[True]
    safety_review_confirmed: Literal[True]
    preparation_preceded_execution_confirmed: Literal[True]
    confirmation_kind: Literal["operator_attestation"] = "operator_attestation"
    baseline_accepted: Literal[True] = True
    stage: Literal["BASELINE_COMPLETE"] = "BASELINE_COMPLETE"
