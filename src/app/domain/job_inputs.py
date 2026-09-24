"""Provider-neutral job discovery and immutable input contracts.

Discovery ports are not registered in production. Callers own authorization,
network permission and budgets; a summary is never a selected full job.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from pydantic import Field, model_validator

from app.domain.errors import DomainValidationError
from app.domain.resume_generation import MAX_JOB_BYTES, GenerationModel, JobInputV1


class JobSnapshotV1(GenerationModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=120)
    source_version: str = Field(min_length=1, max_length=200)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    complete: bool

    @model_validator(mode="after")
    def validate_text(self) -> JobSnapshotV1:
        if (
            not self.text.strip()
            or len(self.text.encode("utf-8")) > MAX_JOB_BYTES
            or hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.sha256
        ):
            raise ValueError("job snapshot text or digest is invalid")
        return self

    def require_complete(self) -> JobSnapshotV1:
        if not self.complete:
            raise DomainValidationError("job detail needs supplementary input")
        return self


class JobInputAdapter(Protocol):
    def snapshot(self, value: JobInputV1) -> JobSnapshotV1: ...


class ProvidedJobInputAdapter:
    def snapshot(self, value: JobInputV1) -> JobSnapshotV1:
        value = JobInputV1.model_validate(value.model_dump())
        return JobSnapshotV1(
            text=value.text,
            source=value.source,
            source_version=value.digest,
            sha256=value.digest,
            complete=True,
        )


class JobCandidateV1(GenerationModel):
    provider: str = Field(min_length=1, max_length=120)
    source_id: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=2000)


class CandidateSearchPort(Protocol):
    async def search(self, query: str) -> tuple[JobCandidateV1, ...]: ...


class JobDetailPort(Protocol):
    async def detail(self, candidate: JobCandidateV1) -> JobSnapshotV1: ...


def unique_candidates(candidates: tuple[JobCandidateV1, ...]) -> tuple[JobCandidateV1, ...]:
    seen: set[tuple[str, str]] = set()
    result = []
    for candidate in candidates:
        key = (candidate.provider, candidate.source_id)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


async def selected_job(
    candidates: tuple[JobCandidateV1, ...],
    selected: tuple[str, str] | None,
    details: JobDetailPort,
) -> JobSnapshotV1:
    for candidate in unique_candidates(candidates):
        if (candidate.provider, candidate.source_id) == selected:
            return (await details.detail(candidate)).require_complete()
    raise DomainValidationError("job selection is required")
