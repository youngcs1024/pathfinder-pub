from __future__ import annotations

from typing import Annotated, Literal
from unicodedata import normalize as _unicode_normalize
from uuid import UUID

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


def normalize_research_query(value: str) -> str:
    return " ".join(_unicode_normalize("NFKC", value).split())


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("text must not be blank")
    return value


Identifier = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
QueryText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=2_000),
    AfterValidator(_non_blank),
]
TitleText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=500),
    AfterValidator(_non_blank),
]
BodyText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=4_000),
    AfterValidator(_non_blank),
]
SnippetText = Annotated[str, StringConstraints(strict=True, max_length=4_000)]
PublishedAtText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=64),
    AfterValidator(_non_blank),
]
LimitationDetailText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=2_000),
    AfterValidator(_non_blank),
]


class ResearchContractModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


class ResearchRequestV1(ResearchContractModel):
    schema_version: Literal[1] = 1
    query: QueryText
    include_application_draft: bool = False


class ResearchSourceV1(ResearchContractModel):
    source_id: Identifier
    title: TitleText
    url: AnyHttpUrl
    snippet: SnippetText
    published_at: PublishedAtText | None = None


class ResearchEvidenceV1(ResearchContractModel):
    evidence_id: Identifier
    source_id: Identifier
    text: BodyText


class ResearchCitationV1(ResearchContractModel):
    source_id: Identifier
    evidence_id: Identifier


class ResearchClaimV1(ResearchContractModel):
    claim_id: Identifier
    text: BodyText
    citations: tuple[ResearchCitationV1, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def citations_must_be_unique(self) -> ResearchClaimV1:
        citation_pairs = tuple(
            (citation.source_id, citation.evidence_id) for citation in self.citations
        )
        if len(set(citation_pairs)) != len(citation_pairs):
            raise ValueError("claim citations must be unique")
        return self


class ApplicationDraftV1(ResearchContractModel):
    paragraphs: tuple[ResearchClaimV1, ...] = Field(min_length=1, max_length=32)


ResearchLimitationCode = Literal[
    "insufficient_evidence",
    "conflicting_evidence",
]


class ResearchLimitationV1(ResearchContractModel):
    code: ResearchLimitationCode
    detail: LimitationDetailText


def _require_unique_ids(values: tuple[object, ...], attribute: str, label: str) -> None:
    identifiers = tuple(getattr(value, attribute) for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} identifiers must be unique")


def _validate_evidence_sources(
    sources: tuple[ResearchSourceV1, ...],
    evidence: tuple[ResearchEvidenceV1, ...],
) -> dict[str, ResearchEvidenceV1]:
    _require_unique_ids(sources, "source_id", "source")
    _require_unique_ids(evidence, "evidence_id", "evidence")

    source_ids = {source.source_id for source in sources}
    evidence_by_id: dict[str, ResearchEvidenceV1] = {}
    for item in evidence:
        if item.source_id not in source_ids:
            raise ValueError("evidence must reference an existing source")
        evidence_by_id[item.evidence_id] = item
    return evidence_by_id


def _all_claims(
    summary: tuple[ResearchClaimV1, ...],
    findings: tuple[ResearchClaimV1, ...],
    application_draft: ApplicationDraftV1 | None,
) -> tuple[ResearchClaimV1, ...]:
    draft_claims = application_draft.paragraphs if application_draft is not None else ()
    return (*summary, *findings, *draft_claims)


class ResearchOutputV1(ResearchContractModel):
    schema_version: Literal[1] = 1
    evidence_sufficient: bool
    summary: tuple[ResearchClaimV1, ...] = Field(default=(), max_length=8)
    findings: tuple[ResearchClaimV1, ...] = Field(default=(), max_length=32)
    evidence: tuple[ResearchEvidenceV1, ...] = Field(default=(), max_length=128)
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    sources: tuple[ResearchSourceV1, ...] = Field(default=(), max_length=64)
    application_draft: ApplicationDraftV1 | None = None

    @model_validator(mode="after")
    def output_must_be_internally_consistent(self) -> ResearchOutputV1:
        evidence_by_id = _validate_evidence_sources(self.sources, self.evidence)
        claims = _all_claims(self.summary, self.findings, self.application_draft)
        _require_unique_ids(claims, "claim_id", "claim")

        for claim in claims:
            for citation in claim.citations:
                cited_evidence = evidence_by_id.get(citation.evidence_id)
                if cited_evidence is None:
                    raise ValueError("citation must reference existing evidence")
                if cited_evidence.source_id != citation.source_id:
                    raise ValueError("citation source must match its evidence source")

        limitation_codes = {limitation.code for limitation in self.limitations}
        report_claim_count = len(self.summary) + len(self.findings)
        if self.evidence_sufficient:
            if not self.sources or not self.evidence or report_claim_count == 0:
                raise ValueError("sufficient output requires sources, evidence, and a report claim")
            if "insufficient_evidence" in limitation_codes:
                raise ValueError("sufficient output cannot claim insufficient evidence")
        else:
            if self.summary or self.findings or self.application_draft is not None:
                raise ValueError(
                    "insufficient output must not contain claims or an application draft"
                )
            if "insufficient_evidence" not in limitation_codes:
                raise ValueError("insufficient output requires an insufficient-evidence limitation")
        return self


ResearchSourceTypeV2 = Literal["web", "workspace_document"]


class ResearchSourceV2(ResearchContractModel):
    source_type: ResearchSourceTypeV2
    source_id: Identifier
    title: TitleText
    url: AnyHttpUrl | None = None
    snippet: SnippetText = ""
    published_at: PublishedAtText | None = None
    document_id: UUID | None = None
    source_name: TitleText | None = None

    @model_validator(mode="after")
    def identity_matches_type(self) -> ResearchSourceV2:
        if self.source_type == "web":
            if self.url is None or self.document_id is not None or self.source_name is not None:
                raise ValueError("web source fields are invalid")
        else:
            expected = (
                f"workspace-document-v1:{self.document_id}" if self.document_id is not None else ""
            )
            if (
                self.url is not None
                or self.document_id is None
                or self.source_name is None
                or self.source_id != expected
            ):
                raise ValueError("workspace document source fields are invalid")
        return self


class ResearchEvidenceV2(ResearchContractModel):
    source_type: ResearchSourceTypeV2
    evidence_id: Identifier
    source_id: Identifier
    text: BodyText
    document_id: UUID | None = None
    chunk_id: UUID | None = None
    section: str | None = Field(default=None, max_length=800)
    ordinal: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def identity_matches_type(self) -> ResearchEvidenceV2:
        if self.source_type == "web":
            if any(
                value is not None
                for value in (self.document_id, self.chunk_id, self.section, self.ordinal)
            ):
                raise ValueError("web evidence fields are invalid")
        else:
            expected_source = (
                f"workspace-document-v1:{self.document_id}" if self.document_id is not None else ""
            )
            expected_evidence = (
                f"workspace-chunk-v1:{self.chunk_id}" if self.chunk_id is not None else ""
            )
            if (
                self.document_id is None
                or self.chunk_id is None
                or self.ordinal is None
                or self.source_id != expected_source
                or self.evidence_id != expected_evidence
            ):
                raise ValueError("workspace document evidence fields are invalid")
        return self


class ResearchCitationV2(ResearchContractModel):
    source_type: ResearchSourceTypeV2
    source_id: Identifier
    evidence_id: Identifier


class ResearchClaimV2(ResearchContractModel):
    claim_id: Identifier
    text: BodyText
    citations: tuple[ResearchCitationV2, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def citations_must_be_unique(self) -> ResearchClaimV2:
        identities = tuple(
            (item.source_type, item.source_id, item.evidence_id) for item in self.citations
        )
        if len(set(identities)) != len(identities):
            raise ValueError("claim citations must be unique")
        return self


class ApplicationDraftV2(ResearchContractModel):
    paragraphs: tuple[ResearchClaimV2, ...] = Field(min_length=1, max_length=32)


class ResearchOutputV2(ResearchContractModel):
    schema_version: Literal[2] = 2
    evidence_sufficient: bool
    summary: tuple[ResearchClaimV2, ...] = Field(default=(), max_length=8)
    findings: tuple[ResearchClaimV2, ...] = Field(default=(), max_length=32)
    evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    sources: tuple[ResearchSourceV2, ...] = Field(default=(), max_length=64)
    application_draft: ApplicationDraftV2 | None = None

    @model_validator(mode="after")
    def output_must_be_internally_consistent(self) -> ResearchOutputV2:
        _require_unique_ids(self.sources, "source_id", "source")
        _require_unique_ids(self.evidence, "evidence_id", "evidence")
        source_by_id = {item.source_id: item for item in self.sources}
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        for item in self.evidence:
            source = source_by_id.get(item.source_id)
            if source is None or source.source_type != item.source_type:
                raise ValueError("evidence must match an existing source")
            if item.source_type == "workspace_document" and source.document_id != item.document_id:
                raise ValueError("document evidence must match its source document")

        draft_claims = self.application_draft.paragraphs if self.application_draft else ()
        claims = (*self.summary, *self.findings, *draft_claims)
        _require_unique_ids(claims, "claim_id", "claim")
        for claim in claims:
            for citation in claim.citations:
                source = source_by_id.get(citation.source_id)
                evidence = evidence_by_id.get(citation.evidence_id)
                if source is None or evidence is None:
                    raise ValueError("citation must resolve to source and evidence")
                if (
                    source.source_type != citation.source_type
                    or evidence.source_type != citation.source_type
                    or evidence.source_id != citation.source_id
                ):
                    raise ValueError("citation source type and evidence must match")

        limitation_codes = {item.code for item in self.limitations}
        if self.evidence_sufficient:
            if not self.sources or not self.evidence or not (self.summary or self.findings):
                raise ValueError("sufficient output requires grounded report evidence")
            if "insufficient_evidence" in limitation_codes:
                raise ValueError("sufficient output cannot claim insufficient evidence")
        else:
            if self.summary or self.findings or self.application_draft is not None:
                raise ValueError("insufficient output must not contain claims or a draft")
            if "insufficient_evidence" not in limitation_codes:
                raise ValueError("insufficient output requires an insufficient-evidence limitation")
        return self


type ResearchOutput = ResearchOutputV1 | ResearchOutputV2
