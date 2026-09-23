"""Fixed job inputs and evidence-bound first draft contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.errors import DomainValidationError
from app.domain.project_facts import MaterialRetrievalScope
from app.domain.resume_profile import JobPreferenceOverrideV1, ResumeContentV1, ResumePreferencesV1
from app.domain.tenancy import TenantContext

MAX_JOB_BYTES = 32 * 1024


@dataclass(frozen=True, slots=True)
class FixedFact:
    version_id: UUID
    project_id: UUID
    claim: str
    kind: str
    conditions: dict[str, object]


@dataclass(frozen=True, slots=True)
class GenerationInputs:
    session_id: UUID
    run_id: UUID
    job_text: str
    profile_content: ResumeContentV1
    preferences: ResumePreferencesV1
    budget: GenerationBudgetV1
    facts: tuple[FixedFact, ...]
    project_map: dict[UUID, UUID]
    retrieval_scope: MaterialRetrievalScope


class GenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class JobInputV1(GenerationModel):
    source: Literal["paste", "upload"]
    text: str = Field(min_length=1)
    filename: str | None = None

    @model_validator(mode="after")
    def check_source(self) -> JobInputV1:
        if (
            not self.text.strip()
            or len(self.text.encode("utf-8")) > MAX_JOB_BYTES
            or (self.source == "paste" and self.filename is not None)
            or (
                self.source == "upload"
                and (
                    self.filename is None
                    or "/" in self.filename
                    or "\\" in self.filename
                    or not self.filename.lower().endswith((".txt", ".md"))
                    or len(self.filename) > 120
                )
            )
        ):
            raise ValueError("job input is invalid or exceeds its byte limit")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class GenerationBudgetV1(GenerationModel):
    max_model_calls: int = Field(ge=1, le=12)
    max_tool_calls: int = Field(ge=0, le=8)
    max_cost_cny: Decimal = Field(gt=0, max_digits=12, decimal_places=6)


class SessionCreateV1(GenerationModel):
    profile_version_id: UUID
    preference_version: int = Field(ge=1)
    project_ids: tuple[UUID, ...] = Field(min_length=1, max_length=20)
    job: JobInputV1
    override: JobPreferenceOverrideV1 = JobPreferenceOverrideV1()
    budget: GenerationBudgetV1

    @model_validator(mode="after")
    def unique_projects(self) -> SessionCreateV1:
        if len(set(self.project_ids)) != len(self.project_ids):
            raise ValueError("project identities must be distinct")
        return self


class JobRequirementV1(GenerationModel):
    kind: Literal["explicit", "preferred", "inferred"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=2000)
    inference_basis: str | None = Field(default=None, max_length=2000)


class RequirementExtractionV1(GenerationModel):
    requirements: tuple[JobRequirementV1, ...] = Field(max_length=40)
    questions: tuple[str, ...] = Field(default=(), max_length=20)


class DraftBulletV1(GenerationModel):
    project_item_id: UUID
    fact_version_id: UUID
    requirement_ordinals: tuple[int, ...] = ()


class DraftSelectionV1(GenerationModel):
    bullets: tuple[DraftBulletV1, ...] = Field(max_length=40)
    omitted_fact_version_ids: tuple[UUID, ...] = ()
    questions: tuple[str, ...] = Field(default=(), max_length=20)


class CoverageV1(GenerationModel):
    requirement_ordinal: int = Field(ge=0)
    support: Literal["supported", "partial", "no_support_found"]
    verification: Literal[
        "needs_human_review", "confirmed_gap", "material_insufficient", "unchecked"
    ]
    fact_version_ids: tuple[UUID, ...] = ()
    item_ids: tuple[UUID, ...] = ()
    reason: str = Field(min_length=1, max_length=2000)


class GenerationCandidateV1(GenerationModel):
    content: ResumeContentV1 | None
    requirements: tuple[JobRequirementV1, ...]
    coverage: tuple[CoverageV1, ...]
    questions: tuple[str, ...]
    omitted_fact_version_ids: tuple[UUID, ...] = ()
    correction_count: int = Field(ge=0, le=1)
    prompt_version: str
    model_id: str


def validate_requirement_positions(jd: str, requirements: tuple[JobRequirementV1, ...]) -> None:
    for requirement in requirements:
        if (
            requirement.end <= requirement.start
            or requirement.end > len(jd)
            or jd[requirement.start : requirement.end] != requirement.quote
            or (requirement.kind == "inferred") != (requirement.inference_basis is not None)
        ):
            raise DomainValidationError("job requirement reference is invalid")


class ResumeGenerationPort(Protocol):
    async def create(self, tenant: TenantContext, request: SessionCreateV1, request_id: UUID): ...
    async def get_session(self, tenant: TenantContext, session_id: UUID): ...
    async def get_version(self, tenant: TenantContext, session_id: UUID, version_id: UUID): ...
    async def cancel(self, tenant: TenantContext, session_id: UUID): ...


class ResumeGenerationExecutionPort(Protocol):
    async def execution_inputs(
        self, tenant: TenantContext, session_id: UUID
    ) -> GenerationInputs: ...
    async def spend_allowed(self, tenant: TenantContext, session_id: UUID) -> bool: ...
    async def reserve_repair(self, tenant: TenantContext, session_id: UUID) -> bool: ...


class ResumeGenerationService:
    def __init__(self, port: ResumeGenerationPort) -> None:
        self.port = port

    async def create(self, tenant: TenantContext, request: SessionCreateV1, request_id: UUID):
        return await self.port.create(tenant, request, request_id)

    async def get_session(self, tenant: TenantContext, session_id: UUID):
        return await self.port.get_session(tenant, session_id)

    async def get_version(self, tenant: TenantContext, session_id: UUID, version_id: UUID):
        return await self.port.get_version(tenant, session_id, version_id)

    async def cancel(self, tenant: TenantContext, session_id: UUID):
        return await self.port.cancel(tenant, session_id)
