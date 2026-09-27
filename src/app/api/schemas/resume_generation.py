"""Public v2 first-draft request and review responses."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict

from app.domain.resume_generation import GenerationBudgetV1, JobRequirementV1
from app.domain.resume_generation import SessionCreateV1 as DomainSessionCreateV1
from app.domain.resume_profile import JobPreferenceOverrideV1, ResumeContentV1
from app.domain.run_payloads import ResumeGenerationResultV1, ResumeRevisionResultV1


def _session_from_json(value):
    # FastAPI has already decoded HTTP JSON into Python objects. Restore JSON
    # semantics for UUID/tuple/Decimal without relaxing strict scalar validation.
    if isinstance(value, dict):
        return DomainSessionCreateV1.model_validate_json(json.dumps(value, allow_nan=False))
    return value


SessionCreateV1 = Annotated[DomainSessionCreateV1, BeforeValidator(_session_from_json)]


class GenerationApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class SessionReceiptResponse(GenerationApiModel):
    command_id: UUID
    session_id: UUID
    run_id: UUID
    status: Literal["queued"] = "queued"
    replayed: bool


class JobSnapshotResponse(GenerationApiModel):
    source: Literal["paste", "upload"]
    filename: str | None
    text: str
    sha256: str


class RequirementResponse(JobRequirementV1):
    id: UUID
    ordinal: int


class SessionListItemResponse(GenerationApiModel):
    session_id: UUID
    run_id: UUID
    run_status: Literal["queued", "running", "completed", "failed", "cancelled"]
    current_version_id: UUID | None
    created_at: datetime
    job_label: str


class SessionDetailResponse(GenerationApiModel):
    session_id: UUID
    initial_run_id: UUID
    run_id: UUID
    run_status: Literal["queued", "running", "completed", "failed", "cancelled"]
    error_category: str | None
    result: ResumeGenerationResultV1 | ResumeRevisionResultV1 | None
    revision: int
    current_version_id: UUID | None
    profile_version_id: UUID
    preference_version: int
    project_ids: list[UUID]
    override: JobPreferenceOverrideV1
    locked_item_ids: list[UUID]
    budget: GenerationBudgetV1
    job: JobSnapshotResponse
    requirements: list[RequirementResponse]


class CoverageResponse(GenerationApiModel):
    requirement_id: UUID
    support: Literal["supported", "partial", "no_support_found"]
    verification: Literal[
        "needs_human_review", "confirmed_gap", "material_insufficient", "unchecked"
    ]
    reason: str
    fact_version_ids: list[UUID]
    item_ids: list[UUID]


class FactEvidenceResponse(GenerationApiModel):
    snapshot_file_id: UUID
    path: str
    source_revision: str
    start_line: int
    end_line: int
    quote: str


class SessionFactResponse(GenerationApiModel):
    version_id: UUID
    project_id: UUID
    claim: str
    kind: Literal["implementation", "plan", "experiment", "personal_statement"]
    conditions: dict[str, object]
    source: Literal["material_snapshot", "user_attestation"]
    attested_at: datetime | None
    evidence: list[FactEvidenceResponse]


class VersionDetailResponse(GenerationApiModel):
    version_id: UUID
    session_id: UUID
    version: int
    artifact_id: UUID
    parent_version_id: UUID | None
    feedback_id: UUID | None
    diff: list[dict[str, object]]
    impact: list[str]
    content: ResumeContentV1
    validation: dict[str, object]
    coverage: list[CoverageResponse]
    facts: list[SessionFactResponse]


class SessionCancelResponse(GenerationApiModel):
    run_id: UUID
    status: str


__all__ = (
    "SessionCancelResponse",
    "SessionCreateV1",
    "SessionDetailResponse",
    "SessionReceiptResponse",
    "VersionDetailResponse",
)
