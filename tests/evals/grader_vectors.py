from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

from app.agents.research_contracts import (
    DocumentRetrievalCallTraceV1,
    ResearchRequestV1,
    SearchCallTraceV1,
)
from app.domain.research import (
    ResearchCitationV2,
    ResearchClaimV2,
    ResearchEvidenceV2,
    ResearchOutputV2,
    ResearchSourceV2,
)
from tests.evals.contracts import (
    EVAL_GRADER_NAMES,
    TRUSTED_CONTEXT_CANARY,
    EvalCitationFixtureV1,
    EvalClaimFixtureV1,
    EvalClaimSupportRuleV1,
    EvalDocumentResultFixtureV3,
    EvalDocumentRetrievalFixtureV3,
    EvalExpectationsV2,
    EvalGraderName,
    EvalResearchPassFixtureV1,
    EvalSearchCallFixtureV1,
    EvalSearchFixtureV1,
    EvalSearchResultFixtureV1,
    EvalWriterFixtureV1,
    ResearchEvalCaseV2,
)

type ExpectedGraderTuple = tuple[EvalGraderName, bool, int]


@dataclass(frozen=True, slots=True)
class GraderBehaviorVector:
    vector_id: str
    case: ResearchEvalCaseV2
    output: ResearchOutputV2
    search_calls: tuple[SearchCallTraceV1, ...]
    model_call_count: int
    evidence_alias_by_id: dict[str, str]
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...] = ()
    relevant_chunk_ids: frozenset[UUID] = frozenset()
    irrelevant_chunk_ids: frozenset[UUID] = frozenset()
    input_tokens: int = 0
    output_tokens: int = 0
    tool_invocation_reservation_count: int = 0
    expected: tuple[ExpectedGraderTuple, ...] = ()

    def __post_init__(self) -> None:
        if not self.vector_id or tuple(item[0] for item in self.expected) != EVAL_GRADER_NAMES:
            raise ValueError("grader vector requires an id and the complete ordered grader set")


_WEB_SOURCE_ID = "web-source-v1"
_WEB_EVIDENCE_ID = "web-evidence-v1"
_WEB_ALIAS = "web_evidence"
_CLAIM_TEXT = "The fixture supports this exact claim."


def _expected(**failures: int) -> tuple[ExpectedGraderTuple, ...]:
    return tuple(
        (name, failures.get(name, 0) == 0, failures.get(name, 0)) for name in EVAL_GRADER_NAMES
    )


def _base_case() -> ResearchEvalCaseV2:
    return ResearchEvalCaseV2(
        case_id="grader_vector",
        request=ResearchRequestV1(query="Evaluate the stable grader contract."),
        plan_queries=("stable grader contract",),
        searches=(
            EvalSearchFixtureV1(
                query="stable grader contract",
                results=(
                    EvalSearchResultFixtureV1(
                        source_alias="web_source",
                        evidence_alias=_WEB_ALIAS,
                        title="Stable source",
                        url="https://example.com/stable",
                        snippet="Stable evidence.",
                    ),
                ),
            ),
        ),
        research_passes=(
            EvalResearchPassFixtureV1(
                calls=(
                    EvalSearchCallFixtureV1(
                        query="stable grader contract",
                        max_results=1,
                    ),
                ),
            ),
        ),
        writer=EvalWriterFixtureV1(
            summary=(
                EvalClaimFixtureV1(
                    claim_id="supported_claim",
                    text=_CLAIM_TEXT,
                    citations=(EvalCitationFixtureV1(evidence_alias=_WEB_ALIAS),),
                ),
            ),
        ),
        support_rules=(
            EvalClaimSupportRuleV1(
                claim_id="supported_claim",
                exact_text=_CLAIM_TEXT,
                allowed_evidence_aliases=(_WEB_ALIAS,),
            ),
        ),
        expectations=EvalExpectationsV2(
            evidence_sufficient=True,
            application_draft_present=False,
            minimum_cited_sources=1,
            max_model_calls=4,
            max_search_calls=1,
            max_search_results=1,
            max_research_passes=1,
            max_input_tokens=10,
            max_output_tokens=10,
            max_total_tool_calls=1,
        ),
    )


def _web_output(*, evidence_text: str = "Stable evidence.") -> ResearchOutputV2:
    citation = ResearchCitationV2(
        source_type="web",
        source_id=_WEB_SOURCE_ID,
        evidence_id=_WEB_EVIDENCE_ID,
    )
    return ResearchOutputV2(
        evidence_sufficient=True,
        summary=(
            ResearchClaimV2(
                claim_id="supported_claim",
                text=_CLAIM_TEXT,
                citations=(citation,),
            ),
        ),
        evidence=(
            ResearchEvidenceV2(
                source_type="web",
                source_id=_WEB_SOURCE_ID,
                evidence_id=_WEB_EVIDENCE_ID,
                text=evidence_text,
            ),
        ),
        sources=(
            ResearchSourceV2(
                source_type="web",
                source_id=_WEB_SOURCE_ID,
                title="Stable source",
                url="https://example.com/stable",
                snippet="Stable evidence.",
            ),
        ),
    )


def _base_vector() -> GraderBehaviorVector:
    return GraderBehaviorVector(
        vector_id="all_pass",
        case=_base_case(),
        output=_web_output(),
        search_calls=(
            SearchCallTraceV1(
                research_pass_number=1,
                call_ordinal=1,
                query="stable grader contract",
                max_results=1,
                result_count=0,
            ),
        ),
        model_call_count=4,
        evidence_alias_by_id={_WEB_EVIDENCE_ID: _WEB_ALIAS},
        input_tokens=4,
        output_tokens=4,
        tool_invocation_reservation_count=1,
        expected=_expected(),
    )


def _retrieval_vector(base: GraderBehaviorVector) -> GraderBehaviorVector:
    relevant_document_id = UUID(int=101)
    relevant_chunk_id = UUID(int=201)
    missing_relevant_chunk_id = UUID(int=202)
    irrelevant_document_id = UUID(int=102)
    irrelevant_chunk_id = UUID(int=203)
    document_items = (
        (relevant_document_id, relevant_chunk_id, "Relevant context."),
        (irrelevant_document_id, irrelevant_chunk_id, "Irrelevant context."),
    )
    document_sources = tuple(
        ResearchSourceV2(
            source_type="workspace_document",
            source_id=f"workspace-document-v1:{document_id}",
            title=f"Document {index}",
            document_id=document_id,
            source_name=f"document-{index}.txt",
        )
        for index, (document_id, _, _) in enumerate(document_items, start=1)
    )
    document_evidence = tuple(
        ResearchEvidenceV2(
            source_type="workspace_document",
            source_id=f"workspace-document-v1:{document_id}",
            evidence_id=f"workspace-chunk-v1:{chunk_id}",
            text=text,
            document_id=document_id,
            chunk_id=chunk_id,
            ordinal=0,
        )
        for document_id, chunk_id, text in document_items
    )
    retrieval_case = base.case.model_copy(
        update={
            "document_retrievals": (
                EvalDocumentRetrievalFixtureV3(
                    query="document evidence",
                    results=tuple(
                        EvalDocumentResultFixtureV3(
                            source_alias=f"document_source_{index}",
                            evidence_alias=f"document_evidence_{index}",
                            document_id=document_id,
                            chunk_id=chunk_id,
                            source_name=f"document-{index}.txt",
                            ordinal=0,
                            cosine_distance=0.1 * index,
                            untrusted_text=text,
                            relevant=index == 1,
                        )
                        for index, (document_id, chunk_id, text) in enumerate(
                            document_items, start=1
                        )
                    ),
                ),
            ),
            "expectations": base.case.expectations.model_copy(
                update={
                    "min_recall_at_5": 1.0,
                    "expected_relevant_chunk_count": 2,
                    "max_irrelevant_context_count": 0,
                    "max_context_bytes": 1,
                    "max_document_retrieval_calls": 1,
                    "max_total_tool_calls": 2,
                }
            ),
        }
    )
    return replace(
        base,
        vector_id="retrieval_recall_irrelevant_context_size_failure",
        case=retrieval_case,
        output=base.output.model_copy(
            update={
                "sources": (*base.output.sources, *document_sources),
                "evidence": (*base.output.evidence, *document_evidence),
            }
        ),
        document_retrieval_calls=(
            DocumentRetrievalCallTraceV1(
                research_pass_number=1,
                call_ordinal=1,
                result_count=2,
                document_ids=(relevant_document_id, irrelevant_document_id),
                chunk_ids=(relevant_chunk_id, irrelevant_chunk_id),
            ),
        ),
        relevant_chunk_ids=frozenset((relevant_chunk_id, missing_relevant_chunk_id)),
        irrelevant_chunk_ids=frozenset((irrelevant_chunk_id,)),
        expected=_expected(retrieval_quality=4),
    )


def _vectors() -> tuple[GraderBehaviorVector, ...]:
    base = _base_vector()
    orphan_evidence_id = "orphan-evidence-v1"
    orphan_output = base.output.model_copy(
        update={
            "summary": (
                ResearchClaimV2.model_construct(
                    claim_id="supported_claim",
                    text=_CLAIM_TEXT,
                    citations=(
                        ResearchCitationV2.model_construct(
                            source_type="web",
                            source_id="orphan-source-v1",
                            evidence_id=orphan_evidence_id,
                        ),
                    ),
                ),
            )
        }
    )
    unsupported_output = base.output.model_copy(
        update={
            "summary": (
                base.output.summary[0].model_copy(update={"text": "An unsupported claim."}),
            )
        }
    )
    output_contract_case = base.case.model_copy(
        update={
            "expectations": base.case.expectations.model_copy(
                update={"application_draft_present": True}
            )
        }
    )
    diversity_case = base.case.model_copy(
        update={
            "expectations": base.case.expectations.model_copy(update={"minimum_cited_sources": 2})
        }
    )
    return (
        base,
        replace(
            base,
            vector_id="unresolved_citation",
            output=orphan_output,
            evidence_alias_by_id={
                **base.evidence_alias_by_id,
                orphan_evidence_id: _WEB_ALIAS,
            },
            expected=_expected(
                schema_valid=1,
                citation_resolvable=1,
                source_diversity=1,
            ),
        ),
        replace(
            base,
            vector_id="unsupported_claim",
            output=unsupported_output,
            expected=_expected(unsupported_claim=1),
        ),
        replace(
            base,
            vector_id="output_contract_failure",
            case=output_contract_case,
            expected=_expected(output_contract=1),
        ),
        replace(
            base,
            vector_id="source_diversity_failure",
            case=diversity_case,
            expected=_expected(source_diversity=1),
        ),
        replace(
            base,
            vector_id="budget_failure",
            model_call_count=5,
            expected=_expected(budget=1),
        ),
        replace(
            base,
            vector_id="trusted_context_canary",
            output=_web_output(evidence_text=TRUSTED_CONTEXT_CANARY),
            expected=_expected(trusted_context_isolated=1),
        ),
        _retrieval_vector(base),
    )


GRADER_BEHAVIOR_VECTORS = _vectors()
if len({vector.vector_id for vector in GRADER_BEHAVIOR_VECTORS}) != len(GRADER_BEHAVIOR_VECTORS):
    raise ValueError("grader vector identifiers must be unique")


def grader_expected_results_payload(
    vectors: tuple[GraderBehaviorVector, ...] = GRADER_BEHAVIOR_VECTORS,
) -> list[dict[str, Any]]:
    return [
        {
            "vector_id": vector.vector_id,
            "expected_results": [
                {"name": name, "passed": passed, "failure_count": failure_count}
                for name, passed, failure_count in vector.expected
            ],
        }
        for vector in sorted(vectors, key=lambda item: item.vector_id)
    ]
