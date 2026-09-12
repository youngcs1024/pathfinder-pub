from __future__ import annotations

from app.agents.research_contracts import ResearchLimitationV1
from app.domain.research import ResearchCitationV2, ResearchClaimV2
from tests.evals.contracts import EvalCaseReportV2
from tests.evals.harness import (
    TRUSTED_CONTEXT_CANARY,
    execute_eval_case,
    grade_eval_case,
    load_eval_dataset,
)


def _grader(report: EvalCaseReportV2, name: str) -> bool:
    return next(grader.passed for grader in report.graders if grader.name == name)


async def test_graders_accept_the_real_offline_graph_output() -> None:
    case = load_eval_dataset()[0]
    executed = await execute_eval_case(case)

    report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        input_tokens=executed.input_tokens,
        output_tokens=executed.output_tokens,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )

    assert report.passed is True
    assert all(grader.passed for grader in report.graders)
    assert report.metrics.model_call_count == 4
    assert report.metrics.input_tokens == 4
    assert report.metrics.output_tokens == 4
    assert report.metrics.search_call_count == 2
    assert report.metrics.cited_source_count == 2


async def test_graders_fail_closed_for_orphan_unknown_and_wrong_evidence_claims() -> None:
    case = load_eval_dataset()[0]
    executed = await execute_eval_case(case)
    output = executed.output
    original_claim = output.summary[0]

    orphan_claim = ResearchClaimV2.model_construct(
        claim_id=original_claim.claim_id,
        text=original_claim.text,
        citations=(
            ResearchCitationV2.model_construct(
                source_type="web",
                source_id="web-v1:" + "0" * 64,
                evidence_id="web-evidence-v1:" + "0" * 64,
            ),
        ),
    )
    orphan_output = output.model_copy(update={"summary": (orphan_claim,)})
    orphan_report = grade_eval_case(
        case,
        orphan_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(orphan_report, "schema_valid") is False
    assert _grader(orphan_report, "citation_resolvable") is False

    unknown_claim = ResearchClaimV2(
        claim_id="invented_claim",
        text="This claim is not present in the versioned support rules.",
        citations=original_claim.citations,
    )
    unknown_output = output.model_copy(update={"summary": (unknown_claim,)})
    unknown_report = grade_eval_case(
        case,
        unknown_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(unknown_report, "schema_valid") is True
    assert _grader(unknown_report, "unsupported_claim") is False

    wrong_citation = output.findings[0].citations[0]
    wrong_evidence_claim = ResearchClaimV2(
        claim_id=original_claim.claim_id,
        text=original_claim.text,
        citations=(wrong_citation,),
    )
    wrong_output = output.model_copy(update={"summary": (wrong_evidence_claim,)})
    wrong_report = grade_eval_case(
        case,
        wrong_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(wrong_report, "schema_valid") is True
    assert _grader(wrong_report, "citation_resolvable") is True
    assert _grader(wrong_report, "unsupported_claim") is False


async def test_diversity_budget_and_trusted_context_failures_are_independent() -> None:
    case = load_eval_dataset()[0]
    executed = await execute_eval_case(case)

    diversity_case = case.model_copy(
        update={"expectations": case.expectations.model_copy(update={"minimum_cited_sources": 3})}
    )
    diversity_report = grade_eval_case(
        diversity_case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(diversity_report, "source_diversity") is False
    assert _grader(diversity_report, "budget") is True

    budget_report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=case.expectations.max_model_calls + 1,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(budget_report, "budget") is False
    assert _grader(budget_report, "source_diversity") is True

    token_budget_report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        input_tokens=case.expectations.max_input_tokens + 1,
        output_tokens=executed.output_tokens,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(token_budget_report, "budget") is False

    canary_output = executed.output.model_copy(
        update={
            "limitations": (
                ResearchLimitationV1(
                    code="conflicting_evidence",
                    detail=TRUSTED_CONTEXT_CANARY,
                ),
            )
        }
    )
    canary_report = grade_eval_case(
        case,
        canary_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
    )
    assert _grader(canary_report, "trusted_context_isolated") is False


async def test_refusal_and_application_draft_expectations_are_checked() -> None:
    cases = {case.case_id: case for case in load_eval_dataset()}
    insufficient = await execute_eval_case(cases["insufficient"])
    refusal_report = grade_eval_case(
        insufficient.case,
        insufficient.output,
        insufficient.search_calls,
        model_call_count=insufficient.model_call_count,
        evidence_alias_by_id=insufficient.evidence_alias_by_id,
    )

    assert refusal_report.passed is True
    assert insufficient.output.evidence_sufficient is False
    assert insufficient.output.application_draft is None
    assert _grader(refusal_report, "output_contract") is True

    wrong_expectation = insufficient.case.model_copy(
        update={
            "expectations": insufficient.case.expectations.model_copy(
                update={"application_draft_present": True}
            )
        }
    )
    wrong_report = grade_eval_case(
        wrong_expectation,
        insufficient.output,
        insufficient.search_calls,
        model_call_count=insufficient.model_call_count,
        evidence_alias_by_id=insufficient.evidence_alias_by_id,
    )
    assert _grader(wrong_report, "output_contract") is False


async def test_document_citation_grader_probes_reject_wrong_chunk_and_source_type() -> None:
    case = next(item for item in load_eval_dataset() if item.case_id == "document_only_resume")
    executed = await execute_eval_case(case)
    claim = executed.output.summary[0]
    citation = claim.citations[0]

    wrong_chunk = ResearchCitationV2.model_construct(
        source_type="workspace_document",
        source_id=citation.source_id,
        evidence_id="workspace-chunk-v1:00000000-0000-0000-0000-000000000099",
    )
    wrong_chunk_output = executed.output.model_copy(
        update={"summary": (claim.model_copy(update={"citations": (wrong_chunk,)}),)}
    )
    wrong_chunk_report = grade_eval_case(
        case,
        wrong_chunk_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        document_retrieval_calls=executed.document_retrieval_calls,
        relevant_chunk_ids=executed.relevant_chunk_ids,
        irrelevant_chunk_ids=executed.irrelevant_chunk_ids,
    )
    assert _grader(wrong_chunk_report, "citation_resolvable") is False

    wrong_type = ResearchCitationV2.model_construct(
        source_type="web",
        source_id=citation.source_id,
        evidence_id=citation.evidence_id,
    )
    wrong_type_output = executed.output.model_copy(
        update={"summary": (claim.model_copy(update={"citations": (wrong_type,)}),)}
    )
    wrong_type_report = grade_eval_case(
        case,
        wrong_type_output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        document_retrieval_calls=executed.document_retrieval_calls,
        relevant_chunk_ids=executed.relevant_chunk_ids,
        irrelevant_chunk_ids=executed.irrelevant_chunk_ids,
    )
    assert _grader(wrong_type_report, "citation_resolvable") is False
    assert wrong_type_report.metrics.citation_source_type_mismatch_count == 1


async def test_reserved_context_proposal_is_rejected_before_tool_accounting() -> None:
    case = next(
        item for item in load_eval_dataset() if item.case_id == "reserved_tool_context_injection"
    )
    executed = await execute_eval_case(case)

    report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        input_tokens=executed.input_tokens,
        output_tokens=executed.output_tokens,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        document_retrieval_calls=executed.document_retrieval_calls,
        relevant_chunk_ids=executed.relevant_chunk_ids,
        irrelevant_chunk_ids=executed.irrelevant_chunk_ids,
        tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
    )

    assert report.passed is True
    assert _grader(report, "output_contract") is True
    assert executed.tool_invocation_reservation_count == 1
    assert len(executed.search_calls) == 1
    assert executed.search_calls[0].research_pass_number == 2
    assert executed.search_calls[0].call_ordinal == 1
    assert executed.document_retrieval_calls == ()
    assert executed.output.evidence_sufficient is True
    cited_evidence_ids = {
        citation.evidence_id for claim in executed.output.summary for citation in claim.citations
    }
    assert cited_evidence_ids == {executed.output.evidence[0].evidence_id}

    wrong_pass = case.model_copy(
        update={
            "expectations": case.expectations.model_copy(
                update={
                    "expected_tool_behavior": case.expectations.expected_tool_behavior.model_copy(
                        update={"search_pass_numbers": (1,)}
                    )
                }
            )
        }
    )
    wrong_pass_report = grade_eval_case(
        wrong_pass,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
    )
    assert _grader(wrong_pass_report, "output_contract") is False

    for wrong_count in (0, 2):
        wrong_count_case = case.model_copy(
            update={
                "expectations": case.expectations.model_copy(
                    update={
                        "expected_tool_behavior": (
                            case.expectations.expected_tool_behavior.model_copy(
                                update={"tool_invocation_reservation_count": wrong_count}
                            )
                        )
                    }
                )
            }
        )
        wrong_count_report = grade_eval_case(
            wrong_count_case,
            executed.output,
            executed.search_calls,
            model_call_count=executed.model_call_count,
            evidence_alias_by_id=executed.evidence_alias_by_id,
            tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
        )
        assert _grader(wrong_count_report, "output_contract") is False


async def test_unknown_tool_proposal_has_zero_rejected_reservations_then_recovers() -> None:
    case = next(item for item in load_eval_dataset() if item.case_id == "unknown_tool_proposal")
    executed = await execute_eval_case(case)
    report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        input_tokens=executed.input_tokens,
        output_tokens=executed.output_tokens,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        document_retrieval_calls=executed.document_retrieval_calls,
        relevant_chunk_ids=executed.relevant_chunk_ids,
        irrelevant_chunk_ids=executed.irrelevant_chunk_ids,
        tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
    )

    assert report.passed is True
    assert executed.tool_invocation_reservation_count == 1
    assert tuple(call.research_pass_number for call in executed.search_calls) == (2,)
    assert tuple(call.call_ordinal for call in executed.search_calls) == (1,)


async def test_ninth_tool_proposal_is_rejected_without_reservation_or_execution() -> None:
    case = next(item for item in load_eval_dataset() if item.case_id == "tool_budget_exhausted")
    executed = await execute_eval_case(case)
    report = grade_eval_case(
        case,
        executed.output,
        executed.search_calls,
        model_call_count=executed.model_call_count,
        input_tokens=executed.input_tokens,
        output_tokens=executed.output_tokens,
        evidence_alias_by_id=executed.evidence_alias_by_id,
        document_retrieval_calls=executed.document_retrieval_calls,
        relevant_chunk_ids=executed.relevant_chunk_ids,
        irrelevant_chunk_ids=executed.irrelevant_chunk_ids,
        tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
    )

    assert report.passed is True
    assert executed.tool_invocation_reservation_count == 8
    assert len(executed.search_calls) == 8
    assert tuple(call.research_pass_number for call in executed.search_calls) == (1,) * 8
    assert tuple(call.call_ordinal for call in executed.search_calls) == tuple(range(1, 9))
    assert executed.output.evidence_sufficient is False
