"""Experiment-only assessor; caller owns usage, Factory owns attempt accounting."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.agents.contracts import AgentLoopControl
from app.agents.prompting import (
    AgentPromptBundleError,
    PromptResourceReader,
    _load_standalone_prompt,
)
from app.agents.research_nodes import (
    _await_single_model_call,
    _check_single_model_boundary,
    _control_clock,
    _SingleModelControlError,
)
from app.domain.research import ResearchContractModel
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.invocations import LLMInvocationAuthorizationError, LLMInvocationContext
from app.llm.ports import ChatMessage, ChatModelResult
from tests.evals.quality_e7a_contracts import (
    ASSESSMENT_POLICY_VERSION,
    ERROR_CATEGORIES,
    AssessmentOutcome,
    E7ABudgetUsageV1,
    E7AContractError,
    EvidenceAssessmentV1,
    EvidenceContext,
    EvidenceGapV1,
    allocate_budget,
    assessment_input,
    parse_assessment,
    validate_assessment,
    validate_usage_progress,
)

PROMPT_FILE = "e7a-assessment-v1.txt"
ASSEMBLY_ID = b"pathfinder-e7a-assessment-prompt-v1"
NODE_NAME = "evidence_assessment"
NODE_ERRORS = ERROR_CATEGORIES | {
    "cancelled",
    "deadline_exceeded",
    "model_invocation_failed",
    "model_output_incomplete",
    "invalid_model_output",
    "provider_timeout",
    "provider_unavailable",
}


class E7AAssessmentError(Exception):
    """No model text, provider exception, or validation input crosses this boundary."""

    def __init__(self, category: str):
        self.category = (
            category if type(category) is str and category in NODE_ERRORS else "configuration_error"
        )
        super().__init__(self.category)


def _read_prompt(name: str) -> bytes:
    path = Path(__file__).with_name("prompts") / name
    if path.is_symlink() or path.parent.is_symlink():
        raise AgentPromptBundleError("assessment prompt is unavailable")
    return path.read_bytes()


def load_assessment_prompt(read_resource: PromptResourceReader = _read_prompt):
    try:
        return _load_standalone_prompt(
            assembly_id=ASSEMBLY_ID,
            prompt_file=PROMPT_FILE,
            resource_label="evidence assessment",
            read_resource=read_resource,
        )
    except AgentPromptBundleError:
        raise E7AAssessmentError("configuration_error") from None


class AssessmentSummaryV1(ResearchContractModel):
    """Content-free experiment result, not new production trace metadata."""

    outcome: AssessmentOutcome
    assessment_policy_version: Literal["e7a-evidence-assessment-v1"] = ASSESSMENT_POLICY_VERSION
    prompt_version: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    pass_number: int = Field(ge=1, le=2)
    evidence_count: int = Field(ge=0, le=128)
    gap_count: int = Field(ge=0, le=8)
    model_call_count: int = Field(ge=0, le=1)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)


@dataclass(frozen=True, slots=True, repr=False)
class AssessmentResult:
    assessment: EvidenceAssessmentV1
    usage: E7ABudgetUsageV1
    summary: AssessmentSummaryV1


def _checked_usage(usage, pass_number):
    usage = validate_usage_progress(usage, usage)
    if type(pass_number) is not int or pass_number not in (1, 2) or usage.writer_calls:
        raise E7AAssessmentError("configuration_error")
    if pass_number == 1 and (
        usage.research_calls[1] or usage.tool_calls[1] or usage.assessment_calls[1]
    ):
        raise E7AAssessmentError("configuration_error")
    if usage.assessment_calls[pass_number - 1]:
        raise E7AAssessmentError("budget_exhausted")
    return usage


@dataclass(frozen=True, slots=True, repr=False)
class EvidenceAssessmentNode:
    factory: LLMFactory
    invocation_context: LLMInvocationContext

    def __post_init__(self):
        if not isinstance(self.factory, LLMFactory) or type(self.invocation_context) is not (
            LLMInvocationContext
        ):
            raise E7AAssessmentError("configuration_error")

    async def __call__(
        self,
        context: EvidenceContext,
        *,
        usage: E7ABudgetUsageV1,
        pass_number: int,
        control: AgentLoopControl,
        on_admitted: Callable[[E7ABudgetUsageV1], None],
    ) -> AssessmentResult:
        """Publish cumulative usage synchronously before awaiting a logical call.

        The caller must retain the published snapshot even on failure/cancellation,
        and supply its latest actual usage on the next admission. This is a narrow
        notification, not persistence or cross-recovery budget enforcement (A.6/A.7).
        Empty-evidence evaluation publishes nothing and consumes no model allowance.
        """
        try:
            if type(control) is not AgentLoopControl or not callable(on_admitted):
                raise E7AAssessmentError("configuration_error")
            _check_single_model_boundary(control)
            started = _control_clock(control)
            node_input = assessment_input(context)
            usage = _checked_usage(usage, pass_number)
            prompt = load_assessment_prompt()
            consumed = 0
            if not node_input.evidence:
                assessment = validate_assessment(
                    EvidenceAssessmentV1(
                        outcome="insufficient",
                        gaps=(
                            EvidenceGapV1(
                                code="missing_resume_fact"
                                if node_input.request.include_application_draft
                                else "missing_task_fact",
                                topic="Requested facts have no delivered evidence",
                            ),
                        ),
                    ),
                    context,
                )
                # No assessor slot is needed, but a bounded report still needs its slots.
                allocate_budget(usage, stage="writer", assessment=assessment)
            else:
                allocate_budget(usage, stage="assessment", pass_number=pass_number)
                counts = list(usage.assessment_calls)
                counts[pass_number - 1] += 1
                updated = validate_usage_progress(
                    usage, usage.model_copy(update={"assessment_calls": tuple(counts)})
                )
                messages = (
                    ChatMessage(role="system", content=prompt.system_prompt),
                    ChatMessage(role="user", content=node_input.model_dump_json()),
                )
                model = self.factory.create_chat_model(self.invocation_context)

                def invoke():
                    # The helper checks cancellation/deadline immediately before this call.
                    # A failing notification prevents provider submission; never roll it back.
                    if on_admitted(updated) is not None:
                        raise E7AAssessmentError("configuration_error")
                    return model.invoke(
                        messages, (), {"graph_node": NODE_NAME, "prompt_version": prompt.version}
                    )

                result = await _await_single_model_call(invoke, control=control)
                usage, consumed = updated, 1
                if not isinstance(result, ChatModelResult):
                    raise E7AAssessmentError("invalid_model_output")
                if result.finish_status != "completed":
                    raise E7AAssessmentError("model_output_incomplete")
                if result.tool_calls or result.content is None:
                    raise E7AAssessmentError("invalid_model_output")
                assessment = parse_assessment(result.content, context)
            _check_single_model_boundary(control)
            elapsed = _control_clock(control) - started
            if elapsed < 0:
                raise E7AAssessmentError("configuration_error")
            return AssessmentResult(
                assessment,
                usage,
                AssessmentSummaryV1(
                    outcome=assessment.outcome,
                    prompt_version=prompt.version,
                    pass_number=pass_number,
                    evidence_count=len(node_input.evidence),
                    gap_count=len(assessment.gaps),
                    model_call_count=consumed,
                    duration_ms=elapsed * 1000,
                ),
            )
        except asyncio.CancelledError:
            raise
        except E7AAssessmentError as error:
            raise E7AAssessmentError(error.category) from None
        except (E7AContractError, _SingleModelControlError) as error:
            raise E7AAssessmentError(error.category) from None
        except LLMInvocationAuthorizationError:
            raise E7AAssessmentError("cancelled") from None
        except LLMProviderError as error:
            raise E7AAssessmentError(
                "provider_timeout"
                if error.category == "provider_timeout"
                else "provider_unavailable"
            ) from None
        except LLMAccountingError as error:
            raise E7AAssessmentError(
                "provider_unavailable" if error.retryable else "model_invocation_failed"
            ) from None
        except Exception:
            raise E7AAssessmentError("model_invocation_failed") from None
