from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.resume_profile import ResumeContentV1, ResumePreferencesV1, SourceLocationV1


class ProfileApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ProfileImportInput(ProfileApiModel):
    source_tex: str = Field(min_length=1, max_length=131072)


class ContactEditInput(ProfileApiModel):
    expected_version: int = Field(ge=1)
    item_id: UUID
    value: str = Field(min_length=1, max_length=300)


class ItemReviewInput(ProfileApiModel):
    expected_version: int = Field(ge=1)
    item_id: UUID


class PreferenceUpdateInput(ProfileApiModel):
    expected_version: int = Field(ge=1)
    preferences: ResumePreferencesV1


class ClaimReviewInput(ProfileApiModel):
    claim_id: UUID
    expected_review_version: int = Field(ge=0)
    decision: Literal["linked", "needs_evidence", "excluded"]
    project_id: UUID | None = None
    fact_version_ids: tuple[UUID, ...] = ()


class ProfileCommandResponse(ProfileApiModel):
    command_id: UUID
    resource_id: UUID
    status: Literal["completed"]
    replayed: bool


class ClaimResponse(ProfileApiModel):
    id: UUID
    project_item_id: UUID
    item_id: UUID
    field: Literal["title", "period", "technologies", "summary", "bullet"]
    text: str
    source: SourceLocationV1
    review_version: int
    decision: Literal["pending", "linked", "needs_evidence", "excluded"]
    project_id: UUID | None
    fact_version_ids: list[UUID]


class ProfileDetailResponse(ProfileApiModel):
    profile_id: UUID
    owner_user_id: UUID
    version_id: UUID
    version: int
    content: ResumeContentV1
    preference_version: int
    preferences: ResumePreferencesV1
    claims: list[ClaimResponse]
