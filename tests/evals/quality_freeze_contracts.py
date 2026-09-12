"""E4.9 sidecar contracts; no changes to historical dataset or business contracts."""

from typing import Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier, EvalText
from tests.evals.quality_contracts import Count, SourceSHA, Split, require_unique


class CoverageCaseV1(EvalContractModel):
    case_id: EvalIdentifier
    family_id: EvalIdentifier
    template_id: EvalIdentifier
    split: Split
    source_aliases: tuple[EvalIdentifier, ...]


class CoverageV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_digest: EvalDigest
    rubric_digest: EvalDigest
    cases: tuple[CoverageCaseV1, ...]

    @model_validator(mode="after")
    def unique_cases(self):
        require_unique(tuple(c.case_id for c in self.cases))
        for case in self.cases:
            require_unique(case.source_aliases)
        return self


class CaseReviewV1(EvalContractModel):
    case_id: EvalIdentifier
    case_digest: EvalDigest
    reviewer_id: Literal["codex_agent"]
    decision: Literal["reviewed", "needs_review"]
    rationale: EvalText


class ReviewsV1(EvalContractModel):
    schema_version: Literal[1] = 1
    authorization_id: Literal["e49_user_authorized_agent_review_v1"]
    reviewer_kind: Literal["agent_self_review"] = "agent_self_review"
    human_reviewed: Literal[False] = False
    independent_review: Literal[False] = False
    dataset_digest: EvalDigest
    coverage_digest: EvalDigest
    cases: tuple[CaseReviewV1, ...]

    @model_validator(mode="after")
    def unique_cases(self):
        require_unique(tuple(c.case_id for c in self.cases))
        return self


class LeakPairV1(EvalContractModel):
    pair_id: EvalDigest
    kind: Literal["query", "source", "unit"]
    left: EvalIdentifier
    right: EvalIdentifier
    cross_split: bool
    exact: bool
    similarity: float = Field(ge=0, le=1)


class LeakReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    algorithm: Literal["nfkc-casefold-space-alias-number-char5-jaccard-v1"]
    threshold: Literal[0.8] = 0.8
    dataset_digest: EvalDigest
    coverage_digest: EvalDigest
    pairs: tuple[LeakPairV1, ...]


class LeakDecisionV1(EvalContractModel):
    pair_id: EvalDigest
    disposition: Literal["same_family_variant", "distinct_context", "needs_review"]
    reviewer_id: Literal["codex_agent"]
    rationale: EvalText


class LeakReviewV1(EvalContractModel):
    schema_version: Literal[1] = 1
    report_digest: EvalDigest
    decisions: tuple[LeakDecisionV1, ...]

    @model_validator(mode="after")
    def unique_pairs(self):
        require_unique(tuple(d.pair_id for d in self.decisions))
        return self


class FrozenSetV1(EvalContractModel):
    schema_version: Literal[1] = 1
    freeze_version: Literal["quality-expanded-freeze-v1"]
    execution_source_sha: SourceSHA
    dataset_digest: EvalDigest
    rubric_digest: EvalDigest
    split_digest: EvalDigest
    coverage_digest: EvalDigest
    reviews_digest: EvalDigest
    leakage_digest: EvalDigest
    leakage_review_digest: EvalDigest
    mapping_digest: EvalDigest
    mapping_report_digest: EvalDigest
    mapping_rules_digest: EvalDigest
    dev_count: Count
    validation_count: Count
    family_count: Count
    validation_case_ids: tuple[EvalIdentifier, ...]
    visibility: Literal["public_frozen_not_for_tuning"] = "public_frozen_not_for_tuning"
    blind: Literal[False] = False
    human_reviewed: Literal[False] = False
    baseline_accepted: Literal[False] = False
    measurement_complete: Literal[False] = False


class ExposureV1(EvalContractModel):
    schema_version: Literal[1] = 1
    exposure_id: EvalIdentifier
    freeze_digest: EvalDigest
    case_id: EvalIdentifier
    execution_source_sha: SourceSHA
    reason: Literal["prompt_tuning", "code_debugging"]
    replacement_required: Literal[True] = True
