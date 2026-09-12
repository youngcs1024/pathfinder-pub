"""E4.7 calibration identities are separate from production and baseline labels."""

from typing import Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier, EvalText
from tests.evals.quality_contracts import (
    Count,
    HumanAnnotationV1,
    QualityRubricV1,
    SourceSHA,
    require_unique,
)


class ReviewSlotV1(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count
    output_digest: EvalDigest
    form_name: str = Field(pattern=r"^review-[0-9]{4}\.md$")
    prefix_digest: EvalDigest


class ReviewPackageV1(EvalContractModel):
    schema_version: Literal[1] = 1
    purpose: Literal["rubric_calibration_not_baseline"] = "rubric_calibration_not_baseline"
    experiment_id: EvalIdentifier
    execution_source_sha: SourceSHA
    source_manifest_digest: EvalDigest
    source_report_digest: EvalDigest
    source_rubric_version: str
    candidate_rubric: QualityRubricV1
    candidate_rubric_digest: EvalDigest
    dataset_digest: EvalDigest
    planned: Count
    failed: Count
    not_run: Count
    slots: tuple[ReviewSlotV1, ...]

    @model_validator(mode="after")
    def unique(self):
        require_unique(tuple((s.case_id, s.repeat_index) for s in self.slots))
        require_unique(tuple(s.form_name for s in self.slots))
        if len(self.slots) > self.planned or self.failed + self.not_run > self.planned:
            raise ValueError("invalid review coverage")
        return self


class PrivateFactV1(EvalContractModel):
    fact_id: EvalIdentifier
    text: EvalText


class PrivateCitationV1(EvalContractModel):
    fact_id: EvalIdentifier
    citation_id: EvalIdentifier
    location: EvalText


class CalibrationCaseV1(EvalContractModel):
    annotation: HumanAnnotationV1
    fact_texts: tuple[PrivateFactV1, ...]
    citation_locations: tuple[PrivateCitationV1, ...]
    facts_inventory_complete: bool
    citations_inventory_complete: bool
    safety_checked: bool
    safety_issue: bool
    rule_questions: EvalText
    form_digest: EvalDigest


class CalibrationV1(EvalContractModel):
    schema_version: Literal[1] = 1
    purpose: Literal["rubric_calibration_not_baseline"] = "rubric_calibration_not_baseline"
    package_digest: EvalDigest
    source_report_digest: EvalDigest
    candidate_rubric_digest: EvalDigest
    reviewer_id: EvalIdentifier
    cases: tuple[CalibrationCaseV1, ...]
    complete: bool


class FreezeRecordV1(EvalContractModel):
    schema_version: Literal[1] = 1
    purpose: Literal["rubric_freeze_not_baseline_acceptance"] = (
        "rubric_freeze_not_baseline_acceptance"
    )
    package_digest: EvalDigest
    calibration_digest: EvalDigest
    source_report_digest: EvalDigest
    candidate_rubric_digest: EvalDigest
    frozen_rubric_digest: EvalDigest
    reviewer_id: EvalIdentifier
    reviewed_outputs: int = Field(ge=10, le=15)
    confirmed_stable: Literal[True]
    semantic_quality_claim: Literal[False] = False
    baseline_accepted: Literal[False] = False
