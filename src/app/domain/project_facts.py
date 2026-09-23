"""Strict candidate facts and evidence checks; model output is never a confirmation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

FactKind = Literal["implementation", "plan", "experiment", "personal_statement"]
ReviewStatus = Literal["pending", "confirmed", "rejected"]


@dataclass(frozen=True, slots=True, repr=False)
class ScopedMaterialFile:
    id: UUID
    path: str
    content: bytes
    document_id: UUID | None
    source_revision: str


@dataclass(frozen=True, slots=True, repr=False)
class MaterialRetrievalScope:
    files: tuple[ScopedMaterialFile, ...]


class FactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class FactConditionsV1(FactModel):
    environment: str | None = Field(default=None, max_length=500)
    scope: str | None = Field(default=None, max_length=500)
    metric_basis: str | None = Field(default=None, max_length=500)


class FactEvidenceV1(FactModel):
    snapshot_file_id: UUID
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=4000)


class CandidateFactV1(FactModel):
    claim: str = Field(min_length=1, max_length=2000)
    kind: FactKind
    conditions: FactConditionsV1 = Field(default_factory=FactConditionsV1)
    evidence: tuple[FactEvidenceV1, ...] = Field(default=(), max_length=8)


class FactExtractionV1(FactModel):
    facts: tuple[CandidateFactV1, ...] = ()
    questions: tuple[str, ...] = Field(default=(), max_length=32)


class ProjectFactPort(Protocol):
    async def current_facts(self, tenant, project_id: UUID) -> dict[str, object]: ...
    async def command(
        self,
        tenant,
        *,
        kind: str,
        project_id: UUID,
        import_id: UUID,
        request_id: UUID,
        fact_id: UUID | None = None,
        expected_version: int | None = None,
        candidate: CandidateFactV1 | None = None,
        decision: str | None = None,
        attested: bool = False,
    ): ...
    async def search_confirmed(
        self, tenant, project_ids: tuple[UUID, ...], query: str
    ) -> list[dict[str, object]]: ...


class ProjectFactService:
    def __init__(self, port: ProjectFactPort) -> None:
        self.port = port

    async def current_facts(self, tenant, project_id: UUID) -> dict[str, object]:
        return await self.port.current_facts(tenant, project_id)

    async def command(self, tenant, **kwargs):
        return await self.port.command(tenant, **kwargs)

    async def search_confirmed(
        self, tenant, project_ids: tuple[UUID, ...], query: str
    ) -> list[dict[str, object]]:
        return await self.port.search_confirmed(tenant, project_ids, query)


class FactEvidenceError(ValueError):
    pass


def check_evidence(evidence: FactEvidenceV1, allowed_files: dict[UUID, bytes]) -> None:
    content = allowed_files.get(evidence.snapshot_file_id)
    if (
        content is None
        or evidence.end_line < evidence.start_line
        or evidence.end_line - evidence.start_line >= 80
    ):
        raise FactEvidenceError("fact evidence is outside the authorized snapshot")
    try:
        lines = content.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        raise FactEvidenceError("fact evidence is invalid") from None
    if evidence.end_line > len(lines):
        raise FactEvidenceError("fact evidence line range is invalid")
    excerpt = "\n".join(lines[evidence.start_line - 1 : evidence.end_line])
    if evidence.quote not in excerpt:
        raise FactEvidenceError("fact evidence quote is not present")


_DIGIT = re.compile(r"\d+(?:[.,%]\d+)*")
_PERSONAL_ROLE = re.compile(
    r"\b(?:led|owned|managed|designed by me)\b|主导|负责|本人", re.IGNORECASE
)


def candidate_issues(candidate: CandidateFactV1) -> tuple[str, ...]:
    issues: list[str] = []
    if not candidate.evidence:
        issues.append("evidence_missing")
    if candidate.kind == "personal_statement":
        issues.append("personal_role_requires_attestation")
    if _DIGIT.search(candidate.claim) and not candidate.conditions.metric_basis:
        issues.append("metric_basis_missing")
    numbers = _DIGIT.findall(candidate.claim)
    quotes = " ".join(item.quote for item in candidate.evidence)
    if numbers and any(number not in quotes for number in numbers):
        issues.append("metric_not_in_evidence")
    if candidate.kind != "personal_statement" and _PERSONAL_ROLE.search(candidate.claim):
        issues.append("personal_role_unverified")
    if candidate.kind == "experiment" and not candidate.conditions.environment:
        issues.append("experiment_environment_missing")
    return tuple(issues)
