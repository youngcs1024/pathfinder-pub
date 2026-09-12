from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    Field,
    model_validator,
)

from app.domain.research import (
    ApplicationDraftV1,
    BodyText,
    Identifier,
    QueryText,
    ResearchCitationV1,
    ResearchClaimV1,
    ResearchContractModel,
    ResearchEvidenceV1,
    ResearchEvidenceV2,
    ResearchLimitationCode,
    ResearchLimitationV1,
    ResearchOutputV1,
    ResearchOutputV2,
    ResearchRequestV1,
    ResearchSourceV1,
    ResearchSourceV2,
    _validate_evidence_sources,
)
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.tools.search import SearchResponseError, canonicalize_search_url, search_source_id

__all__ = [
    "ApplicationDraftV1",
    "BodyText",
    "EvidenceValidationNodeInputV1",
    "EvidenceValidationNodeOutputV1",
    "Identifier",
    "PlanNodeInputV1",
    "PlanNodeOutputV1",
    "QueryText",
    "ResearchCitationV1",
    "ResearchClaimV1",
    "ResearchContractModel",
    "ResearchEvidenceV1",
    "ResearchGraphInputV1",
    "ResearchGraphOutputStateV1",
    "ResearchGraphStateV1",
    "ResearchLimitationCode",
    "ResearchLimitationV1",
    "ResearchNodeInputV1",
    "ResearchNodeOutputV1",
    "ResearchOutputV1",
    "ResearchOutputV2",
    "ResearchPlanV1",
    "ResearchRequestV1",
    "ResearchSourceV1",
    "SearchCallTraceV1",
    "SearchTraceResultV1",
    "WriteReportNodeInputV1",
    "WriteReportNodeOutputV1",
]


class ResearchPlanV1(ResearchContractModel):
    queries: tuple[QueryText, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def queries_must_be_unique(self) -> ResearchPlanV1:
        if len(set(self.queries)) != len(self.queries):
            raise ValueError("research plan queries must be unique")
        return self


class SearchTraceResultV1(ResearchContractModel):
    rank: int = Field(ge=1, le=8)
    canonical_url: AnyHttpUrl
    source_id: Identifier
    truncated: bool

    @model_validator(mode="after")
    def source_identity_must_match_url(self) -> SearchTraceResultV1:
        try:
            canonical_url = canonicalize_search_url(str(self.canonical_url))
            identity_matches = self.source_id == search_source_id(canonical_url)
        except (SearchResponseError, ValueError):
            identity_matches = False
            canonical_url = ""
        if str(self.canonical_url) != canonical_url or not identity_matches:
            raise ValueError("trace source identity must match its canonical URL")
        return self


class SearchCallTraceV1(ResearchContractModel):
    research_pass_number: int = Field(ge=1, le=2)
    call_ordinal: int = Field(ge=1, le=8)
    query: QueryText
    max_results: int = Field(ge=1, le=8)
    result_count: int = Field(ge=0, le=8)
    results: tuple[SearchTraceResultV1, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def result_summary_must_be_consistent(self) -> SearchCallTraceV1:
        if self.result_count != len(self.results):
            raise ValueError("trace result count must match its results")
        if self.result_count > self.max_results:
            raise ValueError("trace results must not exceed the requested maximum")
        if tuple(result.rank for result in self.results) != tuple(range(1, self.result_count + 1)):
            raise ValueError("trace result ranks must be contiguous")
        source_ids = tuple(result.source_id for result in self.results)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("trace result sources must be unique")
        return self


class DocumentRetrievalCallTraceV1(ResearchContractModel):
    research_pass_number: int = Field(ge=1, le=2)
    call_ordinal: int = Field(ge=1, le=8)
    result_count: int = Field(ge=0, le=5)
    document_ids: tuple[UUID, ...] = Field(default=(), max_length=5)
    chunk_ids: tuple[UUID, ...] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def results_are_consistent(self) -> DocumentRetrievalCallTraceV1:
        if self.result_count != len(self.chunk_ids):
            raise ValueError("document retrieval trace count must match chunk ids")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("document retrieval chunk ids must be unique")
        return self


class PlanNodeInputV1(ResearchContractModel):
    normalized_query: QueryText
    include_application_draft: bool


class PlanNodeOutputV1(ResearchContractModel):
    plan: ResearchPlanV1


class ResearchNodeInputV1(ResearchContractModel):
    request: ResearchRequestV1
    normalized_query: QueryText
    plan: ResearchPlanV1
    existing_sources: tuple[ResearchSourceV1, ...] = Field(default=(), max_length=64)
    existing_evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    existing_search_calls: tuple[SearchCallTraceV1, ...] = Field(default=(), max_length=8)
    existing_document_sources: tuple[ResearchSourceV2, ...] = Field(default=(), max_length=8)
    existing_document_evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    existing_document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = Field(
        default=(), max_length=8
    )
    research_pass_number: int = Field(ge=1, le=2)
    document_scope_available: bool = False


class ResearchNodeOutputV1(ResearchContractModel):
    sources: tuple[ResearchSourceV1, ...] = Field(default=(), max_length=64)
    evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    search_calls: tuple[SearchCallTraceV1, ...] = Field(default=(), max_length=8)
    document_sources: tuple[ResearchSourceV2, ...] = Field(default=(), max_length=8)
    document_evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = Field(
        default=(), max_length=8
    )


class EvidenceValidationNodeInputV1(ResearchContractModel):
    request: ResearchRequestV1
    plan: ResearchPlanV1
    sources: tuple[ResearchSourceV1, ...] = Field(default=(), max_length=64)
    evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    document_evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    research_pass_count: int = Field(ge=1, le=2)

    @model_validator(mode="after")
    def evidence_must_resolve_to_sources(self) -> EvidenceValidationNodeInputV1:
        _validate_evidence_sources(self.sources, self.evidence)
        return self


class EvidenceValidationNodeOutputV1(ResearchContractModel):
    evidence_sufficient: bool
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def insufficient_result_requires_limitation(self) -> EvidenceValidationNodeOutputV1:
        codes = {limitation.code for limitation in self.limitations}
        if self.evidence_sufficient and "insufficient_evidence" in codes:
            raise ValueError("sufficient validation cannot claim insufficient evidence")
        if not self.evidence_sufficient and "insufficient_evidence" not in codes:
            raise ValueError("insufficient validation requires an insufficient-evidence limitation")
        return self


class WriteReportNodeInputV1(ResearchContractModel):
    request: ResearchRequestV1
    evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    document_evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    evidence_sufficient: bool
    validation_limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)


class WriteReportNodeOutputV1(ResearchContractModel):
    summary: tuple[ResearchClaimV1, ...] = Field(default=(), max_length=8)
    findings: tuple[ResearchClaimV1, ...] = Field(default=(), max_length=32)
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    application_draft: ApplicationDraftV1 | None = None


class ResearchGraphInputV1(ResearchContractModel):
    schema_version: Literal[1, 2, 3] = 1
    # Deterministic non-production defaults keep isolated graph/eval fixtures concise.
    # The worker executor always supplies all five persisted identity fields explicitly.
    run_id: UUID = UUID("00000000-0000-0000-0000-000000000001")
    workspace_id: UUID = UUID("00000000-0000-0000-0000-000000000002")
    actor_user_id: UUID = UUID("00000000-0000-0000-0000-000000000003")
    conversation_id: UUID = UUID("00000000-0000-0000-0000-000000000004")
    graph_version: Literal[CURRENT_GRAPH_VERSION] = CURRENT_GRAPH_VERSION
    mode: Literal["research", "application"] = "research"
    resume_document_id: UUID | None = None
    request: ResearchRequestV1


class ResearchGraphStateV1(ResearchContractModel):
    schema_version: Literal[1, 2, 3] = 1
    run_id: UUID = UUID("00000000-0000-0000-0000-000000000001")
    workspace_id: UUID = UUID("00000000-0000-0000-0000-000000000002")
    actor_user_id: UUID = UUID("00000000-0000-0000-0000-000000000003")
    conversation_id: UUID = UUID("00000000-0000-0000-0000-000000000004")
    graph_version: Literal[CURRENT_GRAPH_VERSION] = CURRENT_GRAPH_VERSION
    mode: Literal["research", "application"] = "research"
    resume_document_id: UUID | None = None
    request: ResearchRequestV1
    normalized_query: QueryText | None = None
    plan: ResearchPlanV1 | None = None
    sources: tuple[ResearchSourceV1, ...] = Field(default=(), max_length=64)
    evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    search_calls: tuple[SearchCallTraceV1, ...] = Field(default=(), max_length=8)
    document_sources: tuple[ResearchSourceV2, ...] = Field(default=(), max_length=8)
    document_evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = Field(
        default=(), max_length=8
    )
    research_pass_count: int = Field(default=0, ge=0, le=2)
    evidence_sufficient: bool | None = None
    validation_limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    writer_output: WriteReportNodeOutputV1 | None = None
    action_proposal_id: UUID | None = None
    action_key: Literal["submit_application"] | None = None
    action_revision: int | None = Field(default=None, ge=1)
    approval_expires_at: datetime | None = None
    approval_request_id: UUID | None = None
    output: ResearchOutputV1 | ResearchOutputV2 | None = None

    @model_validator(mode="after")
    def action_proposal_identity_is_all_or_none(self) -> ResearchGraphStateV1:
        values = (
            self.action_proposal_id,
            self.action_key,
            self.action_revision,
            self.approval_expires_at,
        )
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError("action proposal identity must be all present or all absent")
        if self.approval_request_id is not None and self.action_proposal_id is None:
            raise ValueError("approval request identity requires an action proposal identity")
        return self


class ResearchGraphOutputStateV1(ResearchContractModel):
    output: ResearchOutputV1 | ResearchOutputV2 | None = None
    action_proposal_id: UUID | None = None
    action_key: Literal["submit_application"] | None = None
    action_revision: int | None = Field(default=None, ge=1)
    approval_expires_at: datetime | None = None
    approval_request_id: UUID | None = None
    search_calls: tuple[SearchCallTraceV1, ...] = Field(default=(), max_length=8)
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = Field(
        default=(), max_length=8
    )

    @model_validator(mode="after")
    def action_proposal_identity_is_all_or_none(self) -> ResearchGraphOutputStateV1:
        values = (
            self.action_proposal_id,
            self.action_key,
            self.action_revision,
            self.approval_expires_at,
        )
        if any(value is None for value in values) and any(value is not None for value in values):
            raise ValueError("action proposal identity must be all present or all absent")
        if self.approval_request_id is not None and self.action_proposal_id is None:
            raise ValueError("approval request identity requires an action proposal identity")
        return self
