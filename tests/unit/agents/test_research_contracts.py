from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.research_contracts import (
    ApplicationDraftV1,
    EvidenceValidationNodeOutputV1,
    ResearchCitationV1,
    ResearchClaimV1,
    ResearchEvidenceV1,
    ResearchGraphStateV1,
    ResearchLimitationV1,
    ResearchNodeOutputV1,
    ResearchOutputV1,
    ResearchPlanV1,
    ResearchRequestV1,
    ResearchSourceV1,
    WriteReportNodeInputV1,
    WriteReportNodeOutputV1,
)


def _source(
    source_id: str = "S1",
    *,
    url: str = "https://example.test/source",
    snippet: str = "Recorded evidence snippet.",
) -> ResearchSourceV1:
    return ResearchSourceV1(
        source_id=source_id,
        title="Synthetic source",
        url=url,
        snippet=snippet,
    )


def _evidence(
    evidence_id: str = "E1",
    *,
    source_id: str = "S1",
    text: str = "A checkable synthetic fact.",
) -> ResearchEvidenceV1:
    return ResearchEvidenceV1(
        evidence_id=evidence_id,
        source_id=source_id,
        text=text,
    )


def _claim(
    claim_id: str = "C1",
    *,
    source_id: str = "S1",
    evidence_id: str = "E1",
) -> ResearchClaimV1:
    return ResearchClaimV1(
        claim_id=claim_id,
        text="The role lists a synthetic requirement.",
        citations=(
            ResearchCitationV1(
                source_id=source_id,
                evidence_id=evidence_id,
            ),
        ),
    )


def _sufficient_output() -> ResearchOutputV1:
    return ResearchOutputV1(
        evidence_sufficient=True,
        summary=(_claim(),),
        findings=(),
        evidence=(_evidence(),),
        limitations=(),
        sources=(_source(),),
    )


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(item) for item in value), set())
    return set()


def test_research_output_is_frozen_strict_and_json_round_trips() -> None:
    output = _sufficient_output()
    serialized = output.model_dump_json()

    assert ResearchOutputV1.model_validate_json(serialized) == output
    assert json.loads(serialized)["summary"][0]["citations"] == [
        {"source_id": "S1", "evidence_id": "E1"}
    ]
    with pytest.raises(ValidationError, match="frozen"):
        output.evidence_sufficient = False
    with pytest.raises(ValidationError, match="Input should be a valid boolean"):
        ResearchRequestV1(query="role", include_application_draft=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reserved_field",
    [
        "workspace_id",
        "actor_user_id",
        "run_id",
        "invocation_id",
        "action_intent_id",
        "deadline",
        "budget",
        "cancellation",
        "credential",
        "credentials",
        "jwt",
        "provider_key",
        "role",
        "target",
        "trusted_target",
        "tool_execution_context",
    ],
)
def test_model_writable_node_output_rejects_trusted_context_fields(
    reserved_field: str,
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResearchNodeOutputV1.model_validate(
            {
                "sources": (),
                "evidence": (),
                reserved_field: "context-canary",
            }
        )


def test_graph_state_and_output_serialization_exclude_trusted_context() -> None:
    state = ResearchGraphStateV1(request=ResearchRequestV1(query="synthetic role"))
    state_payload = json.loads(state.model_dump_json())
    output_payload = json.loads(_sufficient_output().model_dump_json())
    forbidden = {
        "invocation_id",
        "action_intent_id",
        "deadline",
        "budget",
        "cancellation",
        "credential",
        "credentials",
        "target",
        "trusted_target",
        "tool_execution_context",
        "agent_loop_control",
        "tool_runtime",
    }

    assert _all_mapping_keys(state_payload).isdisjoint(forbidden)
    assert state_payload["workspace_id"]
    assert state_payload["actor_user_id"]
    assert state_payload["run_id"]
    assert state_payload["approval_request_id"] is None
    assert _all_mapping_keys(output_payload).isdisjoint(forbidden)
    assert "context-canary" not in json.dumps(state_payload)
    assert "context-canary" not in json.dumps(output_payload)


@pytest.mark.parametrize(
    "partial_identity",
    [
        {"action_proposal_id": uuid4()},
        {"action_key": "submit_application"},
        {"action_revision": 1},
        {"action_proposal_id": uuid4(), "action_key": "submit_application"},
        {"action_key": "submit_application", "action_revision": 1},
    ],
)
def test_graph_action_proposal_identity_is_strictly_all_or_none(
    partial_identity: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="all present or all absent"):
        ResearchGraphStateV1(
            request=ResearchRequestV1(query="synthetic role"),
            **partial_identity,
        )


def test_graph_action_proposal_identity_round_trips_without_entering_public_output() -> None:
    proposal_id = uuid4()
    state = ResearchGraphStateV1(
        request=ResearchRequestV1(query="synthetic role"),
        action_proposal_id=proposal_id,
        action_key="submit_application",
        action_revision=1,
        approval_expires_at=datetime(2026, 8, 24, tzinfo=UTC),
    )

    assert ResearchGraphStateV1.model_validate_json(state.model_dump_json(), strict=True) == state
    assert "action_proposal_id" not in _sufficient_output().model_dump(mode="json")


def test_research_output_rejects_duplicate_source_and_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="source identifiers must be unique"):
        ResearchOutputV1(
            evidence_sufficient=True,
            summary=(_claim(),),
            evidence=(_evidence(),),
            sources=(_source(), _source()),
        )

    with pytest.raises(ValidationError, match="evidence identifiers must be unique"):
        ResearchOutputV1(
            evidence_sufficient=True,
            summary=(_claim(),),
            evidence=(_evidence(), _evidence()),
            sources=(_source(),),
        )


def test_research_output_rejects_orphan_evidence_and_citations() -> None:
    with pytest.raises(ValidationError, match="evidence must reference an existing source"):
        ResearchOutputV1(
            evidence_sufficient=True,
            summary=(_claim(source_id="S-MISSING"),),
            evidence=(_evidence(source_id="S-MISSING"),),
            sources=(_source(),),
        )

    with pytest.raises(ValidationError, match="citation must reference existing evidence"):
        ResearchOutputV1(
            evidence_sufficient=True,
            summary=(_claim(evidence_id="E-MISSING"),),
            evidence=(_evidence(),),
            sources=(_source(),),
        )

    with pytest.raises(ValidationError, match="citation source must match its evidence source"):
        ResearchOutputV1(
            evidence_sufficient=True,
            summary=(_claim(source_id="S2"),),
            evidence=(_evidence(),),
            sources=(_source(), _source("S2", url="https://example.test/two")),
        )


def test_claim_requires_at_least_one_unique_citation() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        ResearchClaimV1(claim_id="C1", text="Unsupported claim", citations=())

    citation = ResearchCitationV1(source_id="S1", evidence_id="E1")
    with pytest.raises(ValidationError, match="claim citations must be unique"):
        ResearchClaimV1(
            claim_id="C1",
            text="Duplicated citation",
            citations=(citation, citation),
        )


def test_structured_refusal_contains_no_claims_or_application_draft() -> None:
    limitation = ResearchLimitationV1(
        code="insufficient_evidence",
        detail="Two bounded research passes found no checkable evidence.",
    )
    refusal = ResearchOutputV1(
        evidence_sufficient=False,
        limitations=(limitation,),
    )

    assert refusal.summary == ()
    assert refusal.findings == ()
    assert refusal.application_draft is None

    with pytest.raises(ValidationError, match="must not contain claims"):
        ResearchOutputV1(
            evidence_sufficient=False,
            summary=(_claim(),),
            evidence=(_evidence(),),
            sources=(_source(),),
            limitations=(limitation,),
        )
    with pytest.raises(ValidationError, match="requires an insufficient-evidence limitation"):
        ResearchOutputV1(evidence_sufficient=False)


def test_sufficient_output_requires_report_evidence_and_respects_draft_citations() -> None:
    with pytest.raises(ValidationError, match="requires sources, evidence, and a report claim"):
        ResearchOutputV1(evidence_sufficient=True)

    output = ResearchOutputV1(
        evidence_sufficient=True,
        summary=(_claim(),),
        evidence=(_evidence(),),
        sources=(_source(),),
        application_draft=ApplicationDraftV1(
            paragraphs=(_claim("D1"),),
        ),
    )

    assert output.application_draft is not None
    assert output.application_draft.paragraphs[0].citations[0].source_id == "S1"


def test_plan_and_validation_contracts_are_bounded_and_consistent() -> None:
    with pytest.raises(ValidationError, match="research plan queries must be unique"):
        ResearchPlanV1(queries=("one query", "one query"))
    with pytest.raises(ValidationError, match="at most 8 items"):
        ResearchPlanV1(queries=tuple(f"query-{index}" for index in range(9)))
    with pytest.raises(ValidationError, match="requires an insufficient-evidence limitation"):
        EvidenceValidationNodeOutputV1(evidence_sufficient=False)


@pytest.mark.parametrize(
    "url",
    (
        "javascript:window.__pathfinder_xss=1",
        "data:text/html,<script>window.__pathfinder_xss=1</script>",
        "file:///tmp/evidence",
    ),
)
def test_source_accepts_only_bounded_http_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        _source(url=url)


def test_source_snippet_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 4000 characters"):
        _source(snippet="x" * 4_001)


def test_writer_output_cannot_embed_authoritative_sources_or_context() -> None:
    valid = WriteReportNodeOutputV1(summary=(_claim(),))

    assert not hasattr(valid, "sources")
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WriteReportNodeOutputV1.model_validate(
            {
                "summary": (),
                "sources": (),
                "workspace_id": "forged",
            }
        )

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        WriteReportNodeInputV1.model_validate(
            {
                "request": ResearchRequestV1(query="synthetic role"),
                "evidence": (),
                "evidence_sufficient": False,
                "validation_limitations": (),
                "sources": (),
            }
        )
