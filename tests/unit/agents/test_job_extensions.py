"""Discovery replacement feeds the real generation graph, preserving JD offsets."""

import hashlib
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.domain.errors import DomainValidationError
from app.domain.job_inputs import (
    JobCandidateV1,
    JobSnapshotV1,
    ProvidedJobInputAdapter,
    selected_job,
    unique_candidates,
)
from app.domain.resume_generation import JobInputV1
from tests.unit.agents.test_resume_generation import _generate, _inputs, _requirements, _selection


class Search:
    async def search(self, query):
        candidate = JobCandidateV1(provider="test", source_id="one", summary=query)
        return (candidate, candidate)


class Details:
    def __init__(self, *, complete=True, failed=False):
        self.complete, self.failed = complete, failed
        self.calls = 0

    async def detail(self, candidate):
        self.calls += 1
        if self.failed:
            raise DomainValidationError("job detail unavailable")
        text = "Build a synthetic service"
        return JobSnapshotV1(
            text=text,
            source=candidate.provider,
            source_version=candidate.source_id,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            complete=self.complete,
        )


@pytest.mark.parametrize("source", ["paste", "upload", "search"])
async def test_input_replacement_preserves_generation_and_original_positions(source):
    _, _, inputs, project, fact = _inputs()
    if source == "search":
        candidates = await Search().search("Summary only")
        assert len(unique_candidates(candidates)) == 1
        snapshot = await selected_job(candidates, ("test", "one"), Details())
    else:
        snapshot = ProvidedJobInputAdapter().snapshot(
            JobInputV1(
                source=source,
                text=inputs.job_text,
                filename="job.md" if source == "upload" else None,
            )
        )
    candidate, _ = await _generate(
        replace(inputs, job_text=snapshot.require_complete().text),
        [_requirements(), _selection(project, fact)],
    )
    assert candidate.content is not None
    reference = candidate.requirements[0]
    assert snapshot.text[reference.start : reference.end] == reference.quote
    assert candidate.coverage[0].fact_version_ids == (fact,)


async def test_discovery_requires_selection_complete_detail_and_reports_failure():
    candidates = await Search().search("Summary only")
    detail = Details()
    for selected in (None, ("test", "missing")):
        with pytest.raises(DomainValidationError, match="selection"):
            await selected_job(candidates, selected, detail)
    assert detail.calls == 0
    with pytest.raises(DomainValidationError, match="supplementary"):
        await selected_job(candidates, ("test", "one"), Details(complete=False))
    with pytest.raises(DomainValidationError, match="unavailable"):
        await selected_job(candidates, ("test", "one"), Details(failed=True))
    with pytest.raises(ValidationError):
        JobInputV1(source="search", text="Cannot enable discovery through HTTP")


@pytest.mark.parametrize("text", ["", " ", "x" * 32769], ids=["empty", "blank", "oversized"])
def test_snapshot_rejects_invalid_text(text):
    with pytest.raises(ValidationError):
        JobSnapshotV1(
            text=text,
            source="test",
            source_version="v1",
            complete=True,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )


def test_snapshot_rejects_false_digest():
    with pytest.raises(ValidationError):
        JobSnapshotV1(
            text="Synthetic", source="test", source_version="v1", complete=True, sha256="0" * 64
        )
