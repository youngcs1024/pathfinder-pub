from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.project_facts import CandidateFactV1, FactConditionsV1, FactKind


class FactApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactAddInput(FactApiModel):
    import_id: UUID
    candidate: CandidateFactV1


class FactReviseInput(FactAddInput):
    expected_version: int = Field(ge=1)


class FactReviewInput(FactApiModel):
    import_id: UUID
    expected_version: int = Field(ge=1)
    decision: Literal["confirm", "reject"]
    attested: bool = False


class FactCommandResponse(FactApiModel):
    command_id: UUID
    resource_id: UUID
    status: Literal["completed"]
    replayed: bool


class FactEvidenceResponse(FactApiModel):
    snapshot_file_id: UUID
    start_line: int
    end_line: int
    quote: str
    path: str
    source_revision: str


class FactResponse(FactApiModel):
    id: UUID
    version_id: UUID
    version: int
    claim: str
    kind: FactKind
    conditions: FactConditionsV1
    review_status: Literal["pending", "confirmed", "rejected"]
    issues: list[dict[str, str]]
    evidence: list[FactEvidenceResponse]


class FactListResponse(FactApiModel):
    fact_set_id: UUID | None
    import_id: UUID | None
    complete: bool
    issues: list[dict[str, str]]
    facts: list[FactResponse]


class FactSearchResponse(FactApiModel):
    project_id: UUID
    fact_id: UUID
    version_id: UUID
    claim: str
    kind: FactKind
