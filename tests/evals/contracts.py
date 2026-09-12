from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic.types import JsonValue

from app.agents.research_contracts import (
    DocumentRetrievalCallTraceV1,
    ResearchLimitationCode,
    ResearchLimitationV1,
    ResearchOutputV1,
    ResearchRequestV1,
    SearchCallTraceV1,
)
from app.domain.research import ResearchOutputV2

TRUSTED_CONTEXT_CANARY = "pathfinder-eval-trusted-context-canary"


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


EvalIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
EvalToolIdentifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
EvalText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=4_000),
    AfterValidator(_non_blank),
]
EvalQuery = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=2_000),
    AfterValidator(_non_blank),
]
EvalUrl = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=2_000),
    AfterValidator(_non_blank),
]


class EvalContractModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class EvalSearchResultFixtureV1(EvalContractModel):
    source_alias: EvalIdentifier
    evidence_alias: EvalIdentifier | None
    title: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=500)]
    url: EvalUrl
    snippet: Annotated[str, StringConstraints(strict=True, max_length=8_000)]
    published_at: (
        Annotated[
            str,
            StringConstraints(strict=True, min_length=1, max_length=64),
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def text_and_evidence_alias_must_be_canonical(self) -> EvalSearchResultFixtureV1:
        if self.title != self.title.strip() or not self.title:
            raise ValueError("search fixture title must be normalized")
        if self.snippet != self.snippet.strip():
            raise ValueError("search fixture snippet must be normalized")
        if self.published_at is not None and self.published_at != self.published_at.strip():
            raise ValueError("search fixture publication time must be normalized")
        if bool(self.snippet) != (self.evidence_alias is not None):
            raise ValueError("non-empty search evidence requires exactly one evidence alias")
        return self


class EvalSearchFixtureV1(EvalContractModel):
    query: EvalQuery
    results: tuple[EvalSearchResultFixtureV1, ...] = Field(default=(), max_length=20)


class EvalSearchCallFixtureV1(EvalContractModel):
    query: EvalQuery
    max_results: int = Field(ge=1, le=8)


class EvalOrderedToolCallFixtureV3(EvalContractModel):
    tool_name: EvalToolIdentifier
    query: EvalQuery
    max_results: int | None = Field(default=None, ge=1, le=8)
    model_extra_arguments: dict[str, JsonValue] = Field(default_factory=dict, max_length=8)

    @model_validator(mode="after")
    def arguments_match_tool(self) -> EvalOrderedToolCallFixtureV3:
        if (self.tool_name == "search_web") != (self.max_results is not None):
            raise ValueError("ordered eval tool arguments do not match the tool")
        if {"query", "max_results"} & self.model_extra_arguments.keys():
            raise ValueError("model extra arguments cannot override canonical tool arguments")
        if any(
            not key or len(key) > 64 or not key.replace("_", "a").isalnum() or not key[0].isalpha()
            for key in self.model_extra_arguments
        ):
            raise ValueError("model extra argument keys must be bounded identifiers")
        if len(self.model_dump_json(include={"model_extra_arguments"}).encode("utf-8")) > 1_024:
            raise ValueError("model extra arguments exceed the eval fixture size limit")
        return self


class EvalResearchPassFixtureV1(EvalContractModel):
    calls: tuple[EvalSearchCallFixtureV1, ...] = Field(default=(), max_length=8)
    ordered_tool_calls: tuple[EvalOrderedToolCallFixtureV3, ...] = Field(default=(), max_length=8)
    expected_rejection: (
        Literal[
            "tool_not_allowed",
            "invalid_tool_arguments",
            "budget_exhausted",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def exactly_one_call_shape(self) -> EvalResearchPassFixtureV1:
        if bool(self.calls) == bool(self.ordered_tool_calls):
            raise ValueError("eval pass requires exactly one tool-call shape")
        has_negative_arguments = any(call.model_extra_arguments for call in self.ordered_tool_calls)
        known_tool_names = {"search_web", "retrieve_documents"}
        has_unknown_tool = any(
            call.tool_name not in known_tool_names for call in self.ordered_tool_calls
        )
        if self.expected_rejection == "invalid_tool_arguments":
            if self.calls or not has_negative_arguments or has_unknown_tool:
                raise ValueError(
                    "invalid-argument proposals require known ordered calls with negative arguments"
                )
        elif self.expected_rejection == "tool_not_allowed":
            if self.calls or not has_unknown_tool or has_negative_arguments:
                raise ValueError("tool-not-allowed proposals require an unknown bounded tool name")
        elif self.expected_rejection == "budget_exhausted":
            if self.calls or has_unknown_tool or has_negative_arguments:
                raise ValueError("budget-exhausted proposals require valid ordered tool calls")
        elif has_negative_arguments or has_unknown_tool:
            raise ValueError("negative arguments require an explicitly rejected proposal")
        return self


class EvalDocumentResultFixtureV3(EvalContractModel):
    source_alias: EvalIdentifier
    evidence_alias: EvalIdentifier
    document_id: UUID
    chunk_id: UUID
    source_name: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=255)]
    section: Annotated[str, StringConstraints(strict=True, max_length=800)] | None = None
    ordinal: int = Field(ge=0)
    cosine_distance: float
    untrusted_text: EvalText
    relevant: bool


class EvalDocumentRetrievalFixtureV3(EvalContractModel):
    query: EvalQuery
    results: tuple[EvalDocumentResultFixtureV3, ...] = Field(default=(), max_length=5)


class EvalCitationFixtureV1(EvalContractModel):
    evidence_alias: EvalIdentifier


class EvalClaimFixtureV1(EvalContractModel):
    claim_id: EvalIdentifier
    text: EvalText
    citations: tuple[EvalCitationFixtureV1, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def citations_must_be_unique(self) -> EvalClaimFixtureV1:
        aliases = tuple(citation.evidence_alias for citation in self.citations)
        if len(set(aliases)) != len(aliases):
            raise ValueError("eval claim citations must be unique")
        return self


class EvalApplicationDraftFixtureV1(EvalContractModel):
    paragraphs: tuple[EvalClaimFixtureV1, ...] = Field(min_length=1, max_length=32)


class EvalWriterFixtureV1(EvalContractModel):
    summary: tuple[EvalClaimFixtureV1, ...] = Field(default=(), max_length=8)
    findings: tuple[EvalClaimFixtureV1, ...] = Field(default=(), max_length=32)
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    application_draft: EvalApplicationDraftFixtureV1 | None = None


class EvalClaimSupportRuleV1(EvalContractModel):
    claim_id: EvalIdentifier
    exact_text: EvalText
    allowed_evidence_aliases: tuple[EvalIdentifier, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def evidence_aliases_must_be_unique(self) -> EvalClaimSupportRuleV1:
        if len(set(self.allowed_evidence_aliases)) != len(self.allowed_evidence_aliases):
            raise ValueError("support-rule evidence aliases must be unique")
        return self


class EvalExpectationsV1(EvalContractModel):
    evidence_sufficient: bool
    application_draft_present: bool
    limitation_codes: tuple[ResearchLimitationCode, ...] = Field(default=(), max_length=8)
    minimum_cited_sources: int = Field(ge=0, le=64)
    max_model_calls: int = Field(ge=1, le=32)
    max_search_calls: int = Field(ge=0, le=8)
    max_search_results: int = Field(ge=0, le=64)
    max_research_passes: int = Field(ge=1, le=2)

    @model_validator(mode="after")
    def limitation_codes_must_be_unique(self) -> EvalExpectationsV1:
        if len(set(self.limitation_codes)) != len(self.limitation_codes):
            raise ValueError("expected limitation codes must be unique")
        return self


EvalResearchPassNumber = Annotated[int, Field(strict=True, ge=1, le=2)]


class EvalExpectedToolBehaviorV1(EvalContractModel):
    search_pass_numbers: tuple[EvalResearchPassNumber, ...] = Field(default=(), max_length=8)
    document_retrieval_pass_numbers: tuple[EvalResearchPassNumber, ...] = Field(
        default=(), max_length=8
    )
    tool_invocation_reservation_count: int = Field(ge=0, le=8)

    @model_validator(mode="after")
    def reservation_count_matches_executed_traces(self) -> EvalExpectedToolBehaviorV1:
        if self.tool_invocation_reservation_count != (
            len(self.search_pass_numbers) + len(self.document_retrieval_pass_numbers)
        ):
            raise ValueError("expected reservations must match expected executed tool traces")
        return self


class EvalExpectationsV2(EvalExpectationsV1):
    max_input_tokens: int = Field(ge=0, le=1_000_000)
    max_output_tokens: int = Field(ge=0, le=1_000_000)
    max_document_retrieval_calls: int = Field(default=0, ge=0, le=8)
    max_total_tool_calls: int = Field(default=8, ge=0, le=8)
    min_recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_relevant_chunk_count: int = Field(default=0, ge=0, le=40)
    max_irrelevant_context_count: int = Field(default=0, ge=0, le=40)
    max_context_bytes: int = Field(default=4_000, ge=0, le=4_000)
    expected_tool_behavior: EvalExpectedToolBehaviorV1 | None = None


def _writer_claims(writer: EvalWriterFixtureV1) -> tuple[EvalClaimFixtureV1, ...]:
    draft_claims = writer.application_draft.paragraphs if writer.application_draft else ()
    return (*writer.summary, *writer.findings, *draft_claims)


class ResearchEvalCaseV1(EvalContractModel):
    schema_version: Literal[1] = 1
    case_id: EvalIdentifier
    request: ResearchRequestV1
    plan_queries: tuple[EvalQuery, ...] = Field(min_length=1, max_length=8)
    searches: tuple[EvalSearchFixtureV1, ...] = Field(default=(), max_length=8)
    resume_document_id: UUID | None = None
    document_retrievals: tuple[EvalDocumentRetrievalFixtureV3, ...] = Field(
        default=(), max_length=8
    )
    research_passes: tuple[EvalResearchPassFixtureV1, ...] = Field(min_length=1, max_length=2)
    writer: EvalWriterFixtureV1
    support_rules: tuple[EvalClaimSupportRuleV1, ...] = Field(default=(), max_length=72)
    expectations: EvalExpectationsV1

    @field_validator("plan_queries")
    @classmethod
    def plan_queries_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("eval plan queries must be unique")
        return value

    @model_validator(mode="after")
    def aliases_calls_and_support_rules_must_resolve(self) -> ResearchEvalCaseV1:
        search_queries = tuple(search.query for search in self.searches)
        if len(set(search_queries)) != len(search_queries):
            raise ValueError("eval search fixture queries must be unique")

        source_aliases: set[str] = set()
        evidence_aliases: set[str] = set()
        for search in self.searches:
            for result in search.results:
                if result.source_alias in source_aliases:
                    raise ValueError("eval source aliases must be unique")
                source_aliases.add(result.source_alias)
                if result.evidence_alias is not None:
                    if result.evidence_alias in evidence_aliases:
                        raise ValueError("eval evidence aliases must be unique")
                    evidence_aliases.add(result.evidence_alias)

        calls = tuple(call for item in self.research_passes for call in item.calls)
        ordered_calls = tuple(
            call for item in self.research_passes for call in item.ordered_tool_calls
        )
        document_queries = tuple(item.query for item in self.document_retrievals)
        if len(set(document_queries)) != len(document_queries):
            raise ValueError("eval document fixture queries must be unique")
        if len(calls) + len(ordered_calls) > 16 or any(
            call.query not in search_queries for call in calls
        ):
            raise ValueError("eval research calls must reference bounded search fixtures")
        if any(
            call.query
            not in (
                search_queries
                if call.tool_name == "search_web"
                else document_queries
                if call.tool_name == "retrieve_documents"
                else (call.query,)
            )
            for call in ordered_calls
        ):
            raise ValueError("ordered eval calls must reference matching fixtures")
        proposed_before = 0
        for research_pass in self.research_passes:
            proposed_now = len(research_pass.calls) + len(research_pass.ordered_tool_calls)
            if research_pass.expected_rejection == "budget_exhausted" and proposed_before < 8:
                raise ValueError("budget-exhausted proposal requires eight prior proposed calls")
            proposed_before += proposed_now

        for retrieval in self.document_retrievals:
            for result in retrieval.results:
                if result.source_alias in source_aliases:
                    raise ValueError("eval source aliases must be unique")
                source_aliases.add(result.source_alias)
                if result.evidence_alias in evidence_aliases:
                    raise ValueError("eval evidence aliases must be unique")
                evidence_aliases.add(result.evidence_alias)

        writer_claims = _writer_claims(self.writer)
        claim_ids = tuple(claim.claim_id for claim in writer_claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("eval writer claim identifiers must be unique")
        writer_evidence_aliases = {
            citation.evidence_alias for claim in writer_claims for citation in claim.citations
        }
        if not writer_evidence_aliases.issubset(evidence_aliases):
            raise ValueError("eval writer citations must reference fixture evidence aliases")

        rule_ids = tuple(rule.claim_id for rule in self.support_rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("eval support-rule claim identifiers must be unique")
        if any(
            not set(rule.allowed_evidence_aliases).issubset(evidence_aliases)
            for rule in self.support_rules
        ):
            raise ValueError("eval support rules must reference fixture evidence aliases")
        return self


class ResearchEvalCaseV2(ResearchEvalCaseV1):
    schema_version: Literal[2] = 2
    expectations: EvalExpectationsV2


class EvalManifestV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_version: Literal["research-v3"] = "research-v3"
    chat_model: str
    embedding_model: str
    embedding_dimension: Literal[1536]
    reasoning_effort: Literal["medium"]
    plan_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    writer_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_tool_policy: Literal["research_agent"]
    research_tool_schema_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_output_schema_version: Literal[2]
    pricing_version: str


EvalGraderName = Literal[
    "schema_valid",
    "citation_resolvable",
    "unsupported_claim",
    "source_diversity",
    "budget",
    "output_contract",
    "trusted_context_isolated",
    "retrieval_quality",
]
EVAL_GRADER_NAMES: tuple[EvalGraderName, ...] = (
    "schema_valid",
    "citation_resolvable",
    "unsupported_claim",
    "source_diversity",
    "budget",
    "output_contract",
    "trusted_context_isolated",
    "retrieval_quality",
)


class EvalGraderResultV1(EvalContractModel):
    name: EvalGraderName
    passed: bool
    failure_count: int = Field(ge=0)

    @model_validator(mode="after")
    def pass_flag_must_match_failures(self) -> EvalGraderResultV1:
        if self.passed != (self.failure_count == 0):
            raise ValueError("grader pass flag must match its failure count")
        return self


class EvalCitationResolutionV1(EvalContractModel):
    claim_id: str
    source_id: str
    evidence_id: str
    evidence_alias: str | None
    resolvable: bool


class EvalMetricsV1(EvalContractModel):
    schema_valid: bool
    claim_count: int = Field(ge=0)
    citation_count: int = Field(ge=0)
    unresolved_citation_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    cited_source_count: int = Field(ge=0)
    model_call_count: int = Field(ge=0)
    search_call_count: int = Field(ge=0)
    search_result_count: int = Field(ge=0)
    research_pass_count: int = Field(ge=0, le=2)


class EvalMetricsV2(EvalMetricsV1):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    document_retrieval_call_count: int = Field(ge=0)
    total_tool_call_count: int = Field(ge=0)
    retrieved_relevant_chunk_count: int = Field(ge=0)
    irrelevant_context_count: int = Field(ge=0)
    context_bytes: int = Field(ge=0)
    recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_source_type_mismatch_count: int = Field(ge=0)


class EvalCaseReportV1(EvalContractModel):
    case_id: EvalIdentifier
    passed: bool
    output: ResearchOutputV1 | None
    search_calls: tuple[SearchCallTraceV1, ...] = Field(default=(), max_length=8)
    citation_resolutions: tuple[EvalCitationResolutionV1, ...] = ()
    metrics: EvalMetricsV1
    graders: tuple[EvalGraderResultV1, ...]
    error_category: (
        Literal[
            "invalid_output",
            "provider_failure",
            "graph_execution_failed",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def status_must_match_complete_grader_set(self) -> EvalCaseReportV1:
        if tuple(grader.name for grader in self.graders) != EVAL_GRADER_NAMES:
            raise ValueError("case report must contain the complete ordered grader set")
        expected_passed = self.error_category is None and all(
            grader.passed for grader in self.graders
        )
        if self.passed != expected_passed:
            raise ValueError("case report status must match grader results")
        if (self.output is None) != (self.error_category is not None):
            raise ValueError("case output presence must match execution status")
        return self


class EvalCaseReportV2(EvalCaseReportV1):
    output: ResearchOutputV2 | None
    metrics: EvalMetricsV2


class EvalCaseReportV3(EvalCaseReportV2):
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = Field(
        default=(), max_length=8
    )


EvalConfigurationErrorCategory = Literal[
    "manifest_unavailable",
    "invalid_manifest",
    "dataset_unavailable",
    "invalid_dataset",
    "unknown_case",
]


class EvalReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_version: Literal["research-v1"] = "research-v1"
    selected_case: Annotated[str, StringConstraints(strict=True, max_length=2_000)] | None = None
    passed: bool
    cases: tuple[EvalCaseReportV1, ...] = ()
    error_category: EvalConfigurationErrorCategory | None = None

    @model_validator(mode="after")
    def report_status_must_match_cases_or_configuration_error(self) -> EvalReportV1:
        if self.error_category is not None:
            if self.passed or self.cases:
                raise ValueError("configuration failure report cannot contain case results")
            return self
        if not self.cases or self.passed != all(case.passed for case in self.cases):
            raise ValueError("eval report status must match non-empty case results")
        return self


class EvalVersionMetadataV1(EvalContractModel):
    chat_model: str
    embedding_model: str
    embedding_dimension: Literal[1536]
    reasoning_effort: Literal["medium"]
    plan_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    writer_prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_tool_policy: Literal["research_agent"]
    research_tool_schema_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    research_output_schema_version: Literal[2]
    pricing_version: str


class EvalReportV2(EvalContractModel):
    schema_version: Literal[2] = 2
    dataset_version: Literal["research-v3"] = "research-v3"
    selected_case: Annotated[str, StringConstraints(strict=True, max_length=2_000)] | None = None
    passed: bool
    cases: tuple[EvalCaseReportV2, ...] = ()
    version_metadata: EvalVersionMetadataV1 | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_available: Literal[False] = False
    estimated_cost_cny: None = None
    error_category: EvalConfigurationErrorCategory | None = None

    @model_validator(mode="after")
    def report_status_must_match_cases_or_configuration_error(self) -> EvalReportV2:
        if self.error_category is not None:
            if (
                self.passed
                or self.cases
                or self.version_metadata is not None
                or self.input_tokens
                or self.output_tokens
            ):
                raise ValueError("configuration failure report cannot contain case results")
            return self
        if (
            self.version_metadata is None
            or not self.cases
            or self.passed != all(case.passed for case in self.cases)
            or self.input_tokens != sum(case.metrics.input_tokens for case in self.cases)
            or self.output_tokens != sum(case.metrics.output_tokens for case in self.cases)
        ):
            raise ValueError("eval report status or token totals are inconsistent")
        return self


EVAL_GRADER_CONTRACT_VERSION = "research-graders-v3"
EvalDigest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$"),
]


class EvalArtifactIdentityV1(EvalContractModel):
    dataset_digest: EvalDigest
    case_set_digest: EvalDigest
    graph_version: str = Field(strict=True, min_length=1, max_length=128)
    embedding_profile: str = Field(strict=True, min_length=1, max_length=128)
    grader_contract_version: str = Field(strict=True, min_length=1, max_length=128)
    grader_contract_digest: EvalDigest


class EvalComparisonMetadataV1(EvalContractModel):
    scope: Literal["full_dataset", "selected_case"]
    dataset_case_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)
    evaluated_case_ids: tuple[EvalIdentifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def scope_and_case_ids_must_be_canonical(self) -> EvalComparisonMetadataV1:
        if self.dataset_case_ids != tuple(sorted(set(self.dataset_case_ids))):
            raise ValueError("dataset case ids must be unique and sorted")
        if self.evaluated_case_ids != tuple(sorted(set(self.evaluated_case_ids))):
            raise ValueError("evaluated case ids must be unique and sorted")
        if not set(self.evaluated_case_ids).issubset(self.dataset_case_ids):
            raise ValueError("evaluated case ids must belong to the dataset")
        if self.scope == "full_dataset" and self.evaluated_case_ids != self.dataset_case_ids:
            raise ValueError("full-dataset comparison must evaluate every dataset case")
        if self.scope == "selected_case" and len(self.evaluated_case_ids) != 1:
            raise ValueError("selected-case comparison must evaluate exactly one case")
        return self


class EvalGraderAggregateV1(EvalContractModel):
    name: EvalGraderName
    passed_case_count: int = Field(ge=0)
    failed_case_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)


class EvalReportV3(EvalReportV2):
    schema_version: Literal[3] = 3
    cases: tuple[EvalCaseReportV3, ...] = ()
    artifact_identity: EvalArtifactIdentityV1 | None = None
    comparison_metadata: EvalComparisonMetadataV1 | None = None
    grader_aggregates: tuple[EvalGraderAggregateV1, ...] = ()

    @model_validator(mode="after")
    def artifact_evidence_must_be_complete_and_consistent(self) -> EvalReportV3:
        if self.error_category is not None:
            if (
                self.artifact_identity is not None
                or self.comparison_metadata is not None
                or self.grader_aggregates
            ):
                raise ValueError("configuration failure cannot contain artifact evidence")
            return self

        if self.artifact_identity is None or self.comparison_metadata is None:
            raise ValueError("valid evaluation must contain artifact and comparison metadata")
        case_ids = tuple(sorted(case.case_id for case in self.cases))
        if case_ids != self.comparison_metadata.evaluated_case_ids:
            raise ValueError("comparison metadata must identify every evaluated case")
        if self.selected_case is None:
            if self.comparison_metadata.scope != "full_dataset":
                raise ValueError("full evaluation must use full-dataset scope")
        elif (
            self.comparison_metadata.scope != "selected_case"
            or self.comparison_metadata.evaluated_case_ids != (self.selected_case,)
        ):
            raise ValueError("selected case must match selected comparison scope")

        if tuple(item.name for item in self.grader_aggregates) != EVAL_GRADER_NAMES:
            raise ValueError("report must contain the complete ordered grader aggregate set")
        graders_by_case = ({grader.name: grader for grader in case.graders} for case in self.cases)
        expected = {name: [0, 0, 0] for name in EVAL_GRADER_NAMES}
        for graders in graders_by_case:
            for name in EVAL_GRADER_NAMES:
                grader = graders[name]
                expected[name][0 if grader.passed else 1] += 1
                expected[name][2] += grader.failure_count
        if any(
            (
                item.passed_case_count,
                item.failed_case_count,
                item.failure_count,
            )
            != tuple(expected[item.name])
            for item in self.grader_aggregates
        ):
            raise ValueError("grader aggregates must match per-case grader evidence")
        return self


EvalAcceptanceReason = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=1_000),
    AfterValidator(_non_blank),
]


class AcceptedEvalBaselineV1(EvalContractModel):
    baseline_schema_version: Literal[1] = 1
    acceptance_reason: EvalAcceptanceReason
    report: EvalReportV3

    @model_validator(mode="after")
    def accepted_report_must_be_full_and_passing(self) -> AcceptedEvalBaselineV1:
        report = self.report
        if (
            report.schema_version != 3
            or report.selected_case is not None
            or report.error_category is not None
            or not report.passed
            or report.comparison_metadata is None
            or report.comparison_metadata.scope != "full_dataset"
            or report.comparison_metadata.dataset_case_ids
            != report.comparison_metadata.evaluated_case_ids
            or report.artifact_identity is None
            or tuple(item.name for item in report.grader_aggregates) != EVAL_GRADER_NAMES
        ):
            raise ValueError(
                "accepted baseline must contain complete passing full-dataset evidence"
            )
        return self


EvalVersionField = Literal[
    "dataset_digest",
    "case_set_digest",
    "graph_version",
    "embedding_profile",
    "grader_contract_version",
    "grader_contract_digest",
    "chat_model",
    "embedding_model",
    "embedding_dimension",
    "reasoning_effort",
    "plan_prompt_version",
    "research_prompt_version",
    "writer_prompt_version",
    "research_tool_policy",
    "research_tool_schema_version",
    "research_output_schema_version",
    "pricing_version",
]


class EvalVersionChangeV1(EvalContractModel):
    field: EvalVersionField
    baseline_value: str | int
    current_value: str | int


class EvalContractRegressionV1(EvalContractModel):
    case_id: EvalIdentifier
    kind: Literal["case_execution", "grader"]
    grader: EvalGraderName | None = None
    baseline_status: Literal["PASS", "FAIL"]
    current_status: Literal["PASS", "FAIL"]

    @model_validator(mode="after")
    def grader_presence_must_match_kind(self) -> EvalContractRegressionV1:
        if (self.kind == "grader") != (self.grader is not None):
            raise ValueError("contract regression grader must match regression kind")
        return self


EvalRegressionMetric = Literal[
    "input_tokens",
    "output_tokens",
    "model_call_count",
    "total_tool_call_count",
    "irrelevant_context_count",
    "recall_at_5",
]


class EvalMetricRegressionV1(EvalContractModel):
    case_id: EvalIdentifier
    metric: EvalRegressionMetric
    direction: Literal["lower_is_better", "higher_is_better"]
    baseline_value: int | float
    current_value: int | float | None


EvalRegressionConfigurationErrorCategory = Literal[
    "baseline_missing",
    "baseline_unreadable",
    "baseline_malformed_json",
    "baseline_schema_incompatible",
    "current_eval_configuration_error",
]


class EvalRegressionReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    passed: bool
    baseline_artifact_identity: EvalArtifactIdentityV1 | None = None
    current_artifact_identity: EvalArtifactIdentityV1 | None = None
    baseline_version_metadata: EvalVersionMetadataV1 | None = None
    current_version_metadata: EvalVersionMetadataV1 | None = None
    added_case_ids: tuple[EvalIdentifier, ...] = ()
    removed_case_ids: tuple[EvalIdentifier, ...] = ()
    version_changes: tuple[EvalVersionChangeV1, ...] = ()
    contract_regressions: tuple[EvalContractRegressionV1, ...] = ()
    metric_regressions: tuple[EvalMetricRegressionV1, ...] = ()
    current_quality_failure: bool = False
    error_category: EvalRegressionConfigurationErrorCategory | None = None

    @model_validator(mode="after")
    def status_and_ordering_must_be_consistent(self) -> EvalRegressionReportV1:
        if self.added_case_ids != tuple(sorted(set(self.added_case_ids))):
            raise ValueError("added case ids must be unique and sorted")
        if self.removed_case_ids != tuple(sorted(set(self.removed_case_ids))):
            raise ValueError("removed case ids must be unique and sorted")
        if self.error_category is not None:
            if self.passed:
                raise ValueError("configuration failure cannot pass")
            return self
        complete_identity = (
            self.baseline_artifact_identity is not None
            and self.current_artifact_identity is not None
            and self.baseline_version_metadata is not None
            and self.current_version_metadata is not None
        )
        has_regression = (
            any(
                (
                    self.added_case_ids,
                    self.removed_case_ids,
                    self.version_changes,
                    self.contract_regressions,
                    self.metric_regressions,
                )
            )
            or self.current_quality_failure
        )
        if not complete_identity or self.passed == has_regression:
            raise ValueError("regression status must match complete comparison evidence")
        return self
