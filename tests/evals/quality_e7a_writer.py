"""Experiment Writer: bounded generation, reference checks, no action authorization.

The caller owns admitted logical usage; Factory owns provider-attempt accounting.
Reference checks cannot establish semantic entailment. No business persistence or
durable recovery is implemented here.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from app.agents.contracts import AgentLoopControl
from app.agents.prompting import (
    AgentPromptBundleError,
    PromptResourceReader,
    _load_standalone_prompt,
)
from app.agents.research_contracts import WriteReportNodeOutputV1
from app.agents.research_nodes import (
    _await_single_model_call,
    _check_single_model_boundary,
    _control_clock,
    _SingleModelControlError,
)
from app.domain.research import (
    ApplicationDraftV1,
    ApplicationDraftV2,
    ResearchCitationV2,
    ResearchClaimV1,
    ResearchClaimV2,
    ResearchContractModel,
    ResearchLimitationV1,
)
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.invocations import LLMInvocationAuthorizationError, LLMInvocationContext
from app.llm.ports import ChatMessage, ChatModelResult
from tests.evals.quality_e7a_contracts import (
    ERROR_CATEGORIES,
    MAX_OUTPUT_BYTES,
    AssessmentOutcome,
    E7ABudgetUsageV1,
    E7AContractError,
    E7AResearchOutputV1,
    EvidenceAssessmentV1,
    EvidenceContext,
    _read_json,
    allocate_budget,
    assessment_input,
    build_output,
    validate_assessment,
    validate_output,
    validate_usage_progress,
)

PROMPT_FILE = "e7a-writer-v1.txt"
ASSEMBLY_ID = b"pathfinder-e7a-writer-prompt-v1"
NODE_NAME = "write_report"
WRITER_ERRORS = ERROR_CATEGORIES | {
    "cancelled",
    "deadline_exceeded",
    "provider_timeout",
    "provider_unavailable",
    "model_invocation_failed",
    "invalid_model_output",
    "model_output_incomplete",
}
REPAIRABLE = frozenset(
    {
        "invalid_json",
        "invalid_schema",
        "invalid_evidence_reference",
        "invalid_source_scope",
        "contradictory_output",
        "model_output_incomplete",
    }
)


class E7AWriterError(Exception):
    def __init__(self, category: str):
        self.category = (
            category
            if type(category) is str and category in WRITER_ERRORS
            else "configuration_error"
        )
        super().__init__(self.category)


def _read_prompt(name: str) -> bytes:
    path = Path(__file__).with_name("prompts") / name
    if path.is_symlink() or path.parent.is_symlink():
        raise AgentPromptBundleError("writer prompt is unavailable")
    return path.read_bytes()


def load_writer_prompt(read_resource: PromptResourceReader = _read_prompt):
    try:
        return _load_standalone_prompt(
            assembly_id=ASSEMBLY_ID,
            prompt_file=PROMPT_FILE,
            resource_label="e7a writer",
            read_resource=read_resource,
        )
    except AgentPromptBundleError:
        raise E7AWriterError("configuration_error") from None


class _WriterReportV1(WriteReportNodeOutputV1):
    """Reuse only the structural contract; all four model fields are required."""

    summary: tuple[ResearchClaimV1, ...] = Field(max_length=8)
    findings: tuple[ResearchClaimV1, ...] = Field(max_length=32)
    limitations: tuple[ResearchLimitationV1, ...] = Field(max_length=8)
    application_draft: ApplicationDraftV1 | None = Field(...)


class WriterSummaryV1(ResearchContractModel):
    outcome: AssessmentOutcome
    prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    model_call_count: int = Field(ge=1, le=2)
    claim_count: int = Field(ge=0, le=72)
    limitation_count: int = Field(ge=0, le=8)
    draft_eligible: bool
    duration_ms: float = Field(ge=0, allow_inf_nan=False)


@dataclass(frozen=True, slots=True, repr=False)
class WriterResult:
    output: E7AResearchOutputV1
    usage: E7ABudgetUsageV1
    summary: WriterSummaryV1


def draft_eligible(output: E7AResearchOutputV1, context: EvidenceContext) -> bool:
    """Quality eligibility only; never creates an intent, binding or permission."""
    output = validate_output(output, context)
    return (
        output.assessment.outcome == "sufficient"
        and context.request.include_application_draft
        and output.application_draft is not None
    )


def _writer_output(result, context, assessment):
    if not isinstance(result, ChatModelResult):
        raise E7AWriterError("invalid_model_output")
    if result.finish_status != "completed":
        raise E7AWriterError("model_output_incomplete")
    if result.tool_calls:
        raise E7AWriterError("invalid_model_output")
    report = _read_json(_WriterReportV1, result.content, MAX_OUTPUT_BYTES)
    evidence = {item.evidence_id: item for item in context.evidence}

    def claim(value):
        citations = []
        for citation in value.citations:
            item = evidence.get(citation.evidence_id)
            if item is None or item.source_id != citation.source_id:
                raise E7AWriterError("invalid_evidence_reference")
            citations.append(
                ResearchCitationV2(source_type=item.source_type, **citation.model_dump())
            )
        return ResearchClaimV2(claim_id=value.claim_id, text=value.text, citations=tuple(citations))

    draft = report.application_draft
    output = build_output(
        context=context,
        assessment=assessment,
        sources=context.sources,
        evidence=context.evidence,
        summary=tuple(claim(x) for x in report.summary),
        findings=tuple(claim(x) for x in report.findings),
        limitations=report.limitations,
        application_draft=ApplicationDraftV2(paragraphs=tuple(claim(x) for x in draft.paragraphs))
        if draft is not None
        else None,
    )
    if (
        assessment.outcome == "sufficient"
        and context.request.include_application_draft
        and output.application_draft is None
    ):
        raise E7AWriterError("contradictory_output")
    return output


@dataclass(frozen=True, slots=True, repr=False)
class E7AWriterNode:
    factory: LLMFactory
    invocation_context: LLMInvocationContext

    def __post_init__(self):
        if (
            not isinstance(self.factory, LLMFactory)
            or type(self.invocation_context) is not LLMInvocationContext
            or self.invocation_context.run_id is None
        ):
            raise E7AWriterError("configuration_error")

    async def __call__(
        self,
        context: EvidenceContext,
        assessment: EvidenceAssessmentV1,
        *,
        usage: E7ABudgetUsageV1,
        control: AgentLoopControl,
        on_admitted: Callable[[E7ABudgetUsageV1], None],
    ) -> WriterResult:
        try:
            if type(control) is not AgentLoopControl or not callable(on_admitted):
                raise E7AWriterError("configuration_error")
            _check_single_model_boundary(control)
            started = _control_clock(control)
            inputs = assessment_input(context)
            assessment = validate_assessment(assessment, context)
            usage = validate_usage_progress(usage, usage)
            allowance = allocate_budget(usage, stage="writer", assessment=assessment)
            prompt = load_writer_prompt()
            payload = inputs.model_dump(mode="json")
            payload["assessment"] = assessment.model_dump(mode="json")
            payload["resume_evidence_ids"] = [
                item.evidence_id
                for item in inputs.evidence
                if item.source_type == "workspace_document"
                and item.document_id == context.resume_document_id
            ]
            base = (
                ChatMessage(role="system", content=prompt.system_prompt),
                ChatMessage(
                    role="user", content=json.dumps(payload, ensure_ascii=False, allow_nan=False)
                ),
            )
            model = self.factory.create_chat_model(self.invocation_context)
            retry_category = None
            for index in range(allowance.model_calls):
                allocate_budget(usage, stage="writer", assessment=assessment)
                updated = validate_usage_progress(
                    usage, usage.model_copy(update={"writer_calls": usage.writer_calls + 1})
                )
                messages = base
                if retry_category is not None:
                    messages += (
                        ChatMessage(
                            role="system",
                            content="Regenerate the complete four-field report. Previous output "
                            "failed application validation: " + retry_category,
                        ),
                    )

                def invoke(updated=updated, messages=messages):
                    if on_admitted(updated) is not None:
                        raise E7AWriterError("configuration_error")
                    return model.invoke(
                        messages, (), {"graph_node": NODE_NAME, "prompt_version": prompt.version}
                    )

                result = await _await_single_model_call(invoke, control=control)
                usage = updated
                try:
                    output = _writer_output(result, context, assessment)
                except (E7AWriterError, E7AContractError) as error:
                    if index + 1 < allowance.model_calls and error.category in REPAIRABLE:
                        retry_category = error.category
                        continue
                    raise
                _check_single_model_boundary(control)
                elapsed = _control_clock(control) - started
                if elapsed < 0:
                    raise E7AWriterError("configuration_error")
                return WriterResult(
                    output,
                    usage,
                    WriterSummaryV1(
                        outcome=assessment.outcome,
                        prompt_version=prompt.version,
                        model_call_count=index + 1,
                        claim_count=len(output.summary)
                        + len(output.findings)
                        + (
                            len(output.application_draft.paragraphs)
                            if output.application_draft
                            else 0
                        ),
                        limitation_count=len(output.limitations),
                        draft_eligible=draft_eligible(output, context),
                        duration_ms=elapsed * 1000,
                    ),
                )
            raise E7AWriterError("budget_exhausted")
        except asyncio.CancelledError:
            raise
        except (E7AWriterError, E7AContractError, _SingleModelControlError) as error:
            raise E7AWriterError(error.category) from None
        except LLMInvocationAuthorizationError:
            raise E7AWriterError("cancelled") from None
        except LLMProviderError as error:
            raise E7AWriterError(
                "provider_timeout"
                if error.category == "provider_timeout"
                else "provider_unavailable"
            ) from None
        except LLMAccountingError as error:
            raise E7AWriterError(
                "provider_unavailable" if error.retryable else "model_invocation_failed"
            ) from None
        except Exception:
            raise E7AWriterError("model_invocation_failed") from None
