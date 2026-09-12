from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalChunkRefV1(_StrictModel):
    document_alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    ordinal: int = Field(ge=0)


class RetrievalDocumentV1(_StrictModel):
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    workspace: Literal["primary", "foreign"]
    source_name: str = Field(min_length=1, max_length=255)
    source_type: Literal["markdown", "text"]
    title: str = Field(min_length=1, max_length=255)
    raw_text: str = Field(min_length=1, max_length=100_000)


class RetrievalQueryCaseV1(_StrictModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    query: str = Field(min_length=1, max_length=2_000)
    allowed_document_aliases: tuple[str, ...] = Field(min_length=1)
    relevant_chunks: tuple[RetrievalChunkRefV1, ...] = Field(min_length=1)
    include_foreign_canary_in_allowlist: bool = False

    @model_validator(mode="after")
    def validate_case(self) -> RetrievalQueryCaseV1:
        if len(set(self.allowed_document_aliases)) != len(self.allowed_document_aliases):
            raise ValueError("allowed document aliases must be unique")
        if len(set(self.relevant_chunks)) != len(self.relevant_chunks):
            raise ValueError("relevant chunk refs must be unique")
        return self


class RetrievalDatasetV1(_StrictModel):
    dataset_version: Literal["retrieval-v1"]
    documents: tuple[RetrievalDocumentV1, ...] = Field(min_length=1, max_length=10)
    cases: tuple[RetrievalQueryCaseV1, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_references(self) -> RetrievalDatasetV1:
        aliases = [document.alias for document in self.documents]
        case_ids = [case.case_id for case in self.cases]
        if len(set(aliases)) != len(aliases):
            raise ValueError("document aliases must be unique")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("case IDs must be unique")
        known = set(aliases)
        primary = {document.alias for document in self.documents if document.workspace == "primary"}
        foreign = [document.alias for document in self.documents if document.workspace == "foreign"]
        if len(foreign) != 1:
            raise ValueError("exactly one foreign workspace canary is required")
        for case in self.cases:
            if not set(case.allowed_document_aliases) <= primary:
                raise ValueError("allowed document alias is unresolved or not primary")
            if any(reference.document_alias not in known for reference in case.relevant_chunks):
                raise ValueError("relevant document alias is unresolved")
            if any(
                reference.document_alias not in case.allowed_document_aliases
                for reference in case.relevant_chunks
            ):
                raise ValueError("relevant chunks must belong to the primary allowlist")
        return self


class RetrievalBenchmarkCaseReportV1(_StrictModel):
    case_id: str
    relevant_chunks: tuple[RetrievalChunkRefV1, ...]
    ranked_retrieved_chunks: tuple[RetrievalChunkRefV1, ...]
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    irrelevant_context_count: int = Field(ge=0)
    workspace_leakage_count: int = Field(ge=0)
    allowlist_leakage_count: int = Field(ge=0)
    passed: bool
    error_category: Literal["retrieval_invariant_failed"] | None = None

    @model_validator(mode="after")
    def status_matches_error_category(self) -> RetrievalBenchmarkCaseReportV1:
        if self.passed != (self.error_category is None):
            raise ValueError("retrieval case status must match its error category")
        return self


class RetrievalBenchmarkAggregateV1(_StrictModel):
    macro_recall_at_1: float = Field(ge=0, le=1)
    macro_recall_at_3: float = Field(ge=0, le=1)
    macro_recall_at_5: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    total_irrelevant_context_count: int = Field(ge=0)
    workspace_leakage_count: int = Field(ge=0)
    allowlist_leakage_count: int = Field(ge=0)


class RetrievalBenchmarkReportV1(_StrictModel):
    schema_version: Literal[1] = 1
    evidence_scope: Literal["deterministic_fake_embedding_regression"]
    semantic_quality_claim: Literal[False] = False
    dataset_version: Literal["retrieval-v1"]
    dataset_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    case_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    normalization_version: str
    chunking_version: str
    embedding_profile: str
    embedding_dimension: int = Field(gt=0)
    retrieval_top_k: Literal[5]
    cases: tuple[RetrievalBenchmarkCaseReportV1, ...] = Field(min_length=1, max_length=20)
    aggregate: RetrievalBenchmarkAggregateV1
    passed: bool

    @model_validator(mode="after")
    def case_ids_are_unique(self) -> RetrievalBenchmarkReportV1:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("retrieval benchmark case IDs must be unique")
        return self


class RetrievalPolicyIdentityV1(_StrictModel):
    dataset_version: Literal["retrieval-v1"]
    dataset_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    case_set_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    normalization_version: str = Field(min_length=1)
    chunking_version: str = Field(min_length=1)
    embedding_profile: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    retrieval_top_k: Literal[5]


class RetrievalThresholdsV1(_StrictModel):
    minimum_macro_recall_at_1: float = Field(ge=0, le=1)
    minimum_macro_recall_at_3: float = Field(ge=0, le=1)
    minimum_macro_recall_at_5: float = Field(ge=0, le=1)
    minimum_mean_reciprocal_rank: float = Field(ge=0, le=1)
    maximum_total_irrelevant_context_count: int = Field(ge=0)


class RetrievalHardInvariantsV1(_StrictModel):
    workspace_leakage_count: Literal[0]
    allowlist_leakage_count: Literal[0]


class RetrievalExpectedRankingV2(_StrictModel):
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    ranked_retrieved_chunks: tuple[RetrievalChunkRefV1, ...]


class RetrievalRegressionPolicyV2(_StrictModel):
    schema_version: Literal[2] = 2
    policy_version: Literal["retrieval-regression-v2"]
    acceptance_reason: str = Field(min_length=1, max_length=1_000)
    evidence_scope: Literal["deterministic_fake_embedding_regression"]
    semantic_quality_claim: Literal[False]
    expected_identity: RetrievalPolicyIdentityV1
    expected_rankings: tuple[RetrievalExpectedRankingV2, ...] = Field(min_length=1, max_length=20)
    thresholds: RetrievalThresholdsV1
    hard_invariants: RetrievalHardInvariantsV1

    @model_validator(mode="after")
    def expected_case_ids_are_unique(self) -> RetrievalRegressionPolicyV2:
        case_ids = tuple(item.case_id for item in self.expected_rankings)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("expected retrieval ranking case IDs must be unique")
        return self


RetrievalRegressionConfigurationErrorCategory = Literal[
    "policy_missing",
    "policy_unreadable",
    "policy_malformed_json",
    "policy_schema_incompatible",
    "current_benchmark_configuration_error",
]


class RetrievalIdentityDifferenceV1(_StrictModel):
    field: str
    expected_value: str | int | bool
    current_value: str | int | bool


class RetrievalHardInvariantRegressionV1(_StrictModel):
    invariant: Literal[
        "workspace_leakage_count",
        "allowlist_leakage_count",
        "current_benchmark_passed",
    ]
    expected_value: int | bool
    current_value: int | bool


class RetrievalMetricRegressionV1(_StrictModel):
    metric: Literal[
        "macro_recall_at_1",
        "macro_recall_at_3",
        "macro_recall_at_5",
        "mean_reciprocal_rank",
        "total_irrelevant_context_count",
    ]
    direction: Literal["higher_is_better", "lower_is_better"]
    threshold_value: float | int
    current_value: float | int


class RetrievalRankingRegressionV2(_StrictModel):
    case_id: str
    regression: Literal["missing_case", "extra_case", "ranking_tuple_drift"]
    expected_ranking: tuple[RetrievalChunkRefV1, ...] | None = None
    current_ranking: tuple[RetrievalChunkRefV1, ...] | None = None

    @model_validator(mode="after")
    def rankings_match_regression_kind(self) -> RetrievalRankingRegressionV2:
        expected_present = self.expected_ranking is not None
        current_present = self.current_ranking is not None
        valid = {
            "missing_case": expected_present and not current_present,
            "extra_case": not expected_present and current_present,
            "ranking_tuple_drift": expected_present and current_present,
        }[self.regression]
        if not valid:
            raise ValueError("ranking regression kind must match expected/current rankings")
        return self


class RetrievalRegressionReportV2(_StrictModel):
    schema_version: Literal[2] = 2
    policy_version: str | None = None
    passed: bool
    error_category: RetrievalRegressionConfigurationErrorCategory | None = None
    identity_differences: tuple[RetrievalIdentityDifferenceV1, ...] = ()
    hard_invariant_regressions: tuple[RetrievalHardInvariantRegressionV1, ...] = ()
    metric_regressions: tuple[RetrievalMetricRegressionV1, ...] = ()
    ranking_regressions: tuple[RetrievalRankingRegressionV2, ...] = ()

    @model_validator(mode="after")
    def pass_status_matches_all_regressions(self) -> RetrievalRegressionReportV2:
        has_failure = self.error_category is not None or any(
            (
                self.identity_differences,
                self.hard_invariant_regressions,
                self.metric_regressions,
                self.ranking_regressions,
            )
        )
        if self.passed == has_failure:
            raise ValueError("retrieval regression status must match all failure categories")
        if self.error_category is not None and any(
            (
                self.identity_differences,
                self.hard_invariant_regressions,
                self.metric_regressions,
                self.ranking_regressions,
            )
        ):
            raise ValueError("configuration errors cannot contain comparison regressions")
        return self
