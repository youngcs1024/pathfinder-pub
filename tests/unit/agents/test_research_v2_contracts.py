from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.research_nodes import web_evidence_id
from app.domain.research import (
    ResearchCitationV2,
    ResearchClaimV2,
    ResearchEvidenceV2,
    ResearchOutputV2,
    ResearchSourceV2,
)
from app.tools.search import normalize_search_result


def _document_output() -> ResearchOutputV2:
    document_id, chunk_id = uuid4(), uuid4()
    source_id = f"workspace-document-v1:{document_id}"
    evidence_id = f"workspace-chunk-v1:{chunk_id}"
    return ResearchOutputV2(
        evidence_sufficient=True,
        sources=(
            ResearchSourceV2(
                source_type="workspace_document",
                source_id=source_id,
                title="resume.md",
                document_id=document_id,
                source_name="resume.md",
            ),
        ),
        evidence=(
            ResearchEvidenceV2(
                source_type="workspace_document",
                evidence_id=evidence_id,
                source_id=source_id,
                document_id=document_id,
                chunk_id=chunk_id,
                ordinal=0,
                text="Grounded immutable resume evidence.",
            ),
        ),
        summary=(
            ResearchClaimV2(
                claim_id="summary-1",
                text="Grounded claim.",
                citations=(
                    ResearchCitationV2(
                        source_type="workspace_document",
                        source_id=source_id,
                        evidence_id=evidence_id,
                    ),
                ),
            ),
        ),
    )


def test_valid_workspace_document_citation_uses_immutable_uuid_identities() -> None:
    output = _document_output()
    assert output.sources[0].source_id.startswith("workspace-document-v1:")
    assert output.evidence[0].evidence_id.startswith("workspace-chunk-v1:")


def test_valid_web_citation_preserves_canonical_search_identity() -> None:
    result = normalize_search_result(
        title="Canonical role",
        url="HTTPS://EXAMPLE.TEST:443/jobs/backend#details",
        snippet="The role requires Python.",
    )
    evidence_id = web_evidence_id(source_id=result.source_id, snippet=result.snippet)
    output = ResearchOutputV2(
        evidence_sufficient=True,
        sources=(
            ResearchSourceV2(
                source_type="web",
                source_id=result.source_id,
                title=result.title,
                url=result.url,
                snippet=result.snippet,
            ),
        ),
        evidence=(
            ResearchEvidenceV2(
                source_type="web",
                evidence_id=evidence_id,
                source_id=result.source_id,
                text=result.snippet,
            ),
        ),
        summary=(
            ResearchClaimV2(
                claim_id="web-claim",
                text="The role requires Python.",
                citations=(
                    ResearchCitationV2(
                        source_type="web",
                        source_id=result.source_id,
                        evidence_id=evidence_id,
                    ),
                ),
            ),
        ),
    )
    assert str(output.sources[0].url) == "https://example.test/jobs/backend"
    assert output.sources[0].source_id == result.source_id


@pytest.mark.parametrize(
    "url",
    (
        "javascript:window.__pathfinder_xss=1",
        "data:text/html,<script>window.__pathfinder_xss=1</script>",
        "file:///tmp/evidence",
    ),
)
def test_v2_web_source_rejects_non_http_url_schemes(url: str) -> None:
    with pytest.raises(ValidationError):
        ResearchSourceV2(
            source_type="web",
            source_id="web-source-v1:invalid-scheme",
            title="Untrusted source",
            url=url,
            snippet="Untrusted snippet",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_source",
        "missing_evidence",
        "wrong_chunk",
        "web_to_document",
        "document_to_web",
        "duplicate",
    ],
)
def test_v2_rejects_unresolvable_mismatched_or_duplicate_citations(mutation: str) -> None:
    output = _document_output().model_dump(mode="python")
    citation = output["summary"][0]["citations"][0]
    if mutation == "missing_source":
        citation["source_id"] = f"workspace-document-v1:{uuid4()}"
    elif mutation in {"missing_evidence", "wrong_chunk"}:
        citation["evidence_id"] = f"workspace-chunk-v1:{uuid4()}"
    elif mutation == "web_to_document":
        citation["source_type"] = "web"
    elif mutation == "document_to_web":
        output["sources"][0]["source_type"] = "web"
        output["sources"][0]["url"] = "https://example.test/resume"
        output["sources"][0]["document_id"] = None
        output["sources"][0]["source_name"] = None
    else:
        output["summary"][0]["citations"] = (citation, dict(citation))
    with pytest.raises(ValidationError):
        ResearchOutputV2.model_validate(output)
