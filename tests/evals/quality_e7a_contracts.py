"""E7-A.2 test-only contracts and pure admission, not a graph or accounting ledger.

Use the checked entrypoints at untrusted boundaries; never log raw Pydantic errors.
EvidenceContext is constructed from already authorized, delivered runtime data,
not model JSON. Reference validation cannot prove entailment or grant approval.
BudgetUsage is supplied from actual execution usage, including failed logical
calls. Factory retries/embedding remain in the existing provider-attempt ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, StringConstraints, ValidationError, model_validator

from app.domain.research import (
    ApplicationDraftV2,
    Identifier,
    ResearchClaimV2,
    ResearchContractModel,
    ResearchEvidenceV2,
    ResearchLimitationV1,
    ResearchRequestV1,
    ResearchSourceV2,
)

ASSESSMENT_POLICY_VERSION = "e7a-evidence-assessment-v1"
OUTPUT_CONTRACT = "e7a_research_output_v1"
MAX_MODEL_CALLS = 12
MAX_TOOL_CALLS = 8
MAX_ASSESSMENT_BYTES = 32_768
MAX_OUTPUT_BYTES = 2_097_152

type AssessmentOutcome = Literal["sufficient", "partial", "insufficient", "conflicting"]
type GapCode = Literal[
    "missing_resume_fact",
    "missing_job_fact",
    "missing_task_fact",
    "unsupported_claim_strength",
    "source_conflict",
]
GAP_CODES = frozenset(
    {
        "missing_resume_fact",
        "missing_job_fact",
        "missing_task_fact",
        "unsupported_claim_strength",
        "source_conflict",
    }
)
ERROR_CATEGORIES = frozenset(
    {
        "invalid_json",
        "invalid_schema",
        "invalid_evidence_reference",
        "invalid_source_scope",
        "contradictory_output",
        "budget_exhausted",
        "configuration_error",
    }
)


class E7AContractError(ValueError):
    """Only a fixed category crosses the application boundary."""

    def __init__(self, category: str):
        self.category = (
            category
            if type(category) is str and category in ERROR_CATEGORIES
            else "configuration_error"
        )
        super().__init__(self.category)


def _fail(category: str) -> None:
    raise E7AContractError(category)


def _topic(value: str) -> str:
    if not value.strip() or len(value.encode("utf-8")) > 800:
        _fail("invalid_schema")
    return value


GapTopic = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=200),
    AfterValidator(_topic),
]


class EvidenceGapV1(ResearchContractModel):
    code: GapCode
    topic: GapTopic


class EvidenceAssessmentV1(ResearchContractModel):
    assessment_policy_version: Literal["e7a-evidence-assessment-v1"] = ASSESSMENT_POLICY_VERSION
    outcome: AssessmentOutcome
    evidence_ids: tuple[Identifier, ...] = Field(default=(), max_length=128)
    gaps: tuple[EvidenceGapV1, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            _fail("invalid_evidence_reference")
        gaps = tuple((gap.code, gap.topic.casefold().strip()) for gap in self.gaps)
        if len(set(gaps)) != len(gaps):
            _fail("invalid_schema")
        codes = {gap.code for gap in self.gaps}
        if self.outcome == "sufficient":
            if not self.evidence_ids or self.gaps:
                _fail("contradictory_output")
        elif not self.gaps:
            _fail("contradictory_output")
        if self.outcome == "partial" and not self.evidence_ids:
            _fail("contradictory_output")
        if self.outcome == "conflicting":
            # One delivered snippet can contain two conflicting announcements.
            if not self.evidence_ids or "source_conflict" not in codes:
                _fail("contradictory_output")
        elif "source_conflict" in codes:
            _fail("contradictory_output")
        return self


def _source_relations(sources, evidence):
    source_map = {item.source_id: item for item in sources}
    evidence_map = {item.evidence_id: item for item in evidence}
    if len(source_map) != len(sources) or len(evidence_map) != len(evidence):
        _fail("invalid_evidence_reference")
    for item in evidence:
        source = source_map.get(item.source_id)
        if source is None or (
            source.source_type != item.source_type or source.document_id != item.document_id
        ):
            _fail("invalid_evidence_reference")
    return evidence_map


class _EvidenceBundleV1(ResearchContractModel):
    sources: tuple[ResearchSourceV2, ...] = Field(max_length=64)
    evidence: tuple[ResearchEvidenceV2, ...] = Field(max_length=128)

    @model_validator(mode="after")
    def consistent(self):
        _source_relations(self.sources, self.evidence)
        return self


class AssessmentInputV1(ResearchContractModel):
    request: ResearchRequestV1
    evidence: tuple[ResearchEvidenceV2, ...] = Field(max_length=128)


@dataclass(frozen=True, slots=True, repr=False)
class EvidenceContext:
    """Caller-owned scope; never deserialize this from model or request fields."""

    request: ResearchRequestV1
    sources: tuple[ResearchSourceV2, ...]
    evidence: tuple[ResearchEvidenceV2, ...]
    resume_document_id: UUID | None = None


def _validation_category(error: ValidationError) -> str:
    for detail in error.errors(include_input=False, include_url=False):
        cause = detail.get("ctx", {}).get("error")
        if isinstance(cause, E7AContractError):
            return cause.category
    return "invalid_schema"


def _fresh(model, value):
    """Revalidate even model_copy/model_construct instances at checked boundaries."""
    if type(value) is not model:
        _fail("invalid_schema")
    try:
        payload = value.model_dump_json(warnings="error")
        return model.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise E7AContractError(_validation_category(error)) from None
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise E7AContractError("invalid_schema") from None


def _context(value: EvidenceContext) -> EvidenceContext:
    if type(value) is not EvidenceContext or (
        value.resume_document_id is not None and type(value.resume_document_id) is not UUID
    ):
        _fail("invalid_source_scope")
    request = _fresh(ResearchRequestV1, value.request)
    try:
        bundle = _EvidenceBundleV1(sources=value.sources, evidence=value.evidence)
    except ValidationError as error:
        raise E7AContractError(_validation_category(error)) from None
    bundle = _fresh(_EvidenceBundleV1, bundle)
    if request.include_application_draft and value.resume_document_id is None:
        _fail("invalid_source_scope")
    return EvidenceContext(request, bundle.sources, bundle.evidence, value.resume_document_id)


def assessment_input(context: EvidenceContext) -> AssessmentInputV1:
    context = _context(context)
    return AssessmentInputV1(request=context.request, evidence=context.evidence)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("invalid_json")
        result[key] = value
    return result


def _nonfinite(_value):
    _fail("invalid_json")


def _read_json(model, raw: str | bytes, maximum: int):
    try:
        if type(raw) not in {str, bytes}:
            _fail("invalid_json")
        payload = raw.encode("utf-8") if type(raw) is str else raw
        if len(payload) > maximum:
            _fail("invalid_json")
        decoded = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_nonfinite
        )
        normalized = json.dumps(decoded, allow_nan=False, ensure_ascii=False)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise E7AContractError("invalid_json") from None
    try:
        return model.model_validate_json(normalized, strict=True)
    except ValidationError as error:
        raise E7AContractError(_validation_category(error)) from None
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise E7AContractError("invalid_schema") from None


def validate_assessment(value: EvidenceAssessmentV1, context: EvidenceContext):
    context = _context(context)
    value = _fresh(EvidenceAssessmentV1, value)
    actual = {item.evidence_id: item for item in context.evidence}
    if not set(value.evidence_ids) <= actual.keys():
        _fail("invalid_evidence_reference")
    if value.outcome == "sufficient" and context.request.include_application_draft:
        if not any(
            actual[key].source_type == "workspace_document"
            and actual[key].document_id == context.resume_document_id
            for key in value.evidence_ids
        ):
            _fail("invalid_source_scope")
    return value


def parse_assessment(raw: str | bytes, context: EvidenceContext) -> EvidenceAssessmentV1:
    return validate_assessment(_read_json(EvidenceAssessmentV1, raw, MAX_ASSESSMENT_BYTES), context)


class E7AResearchOutputV1(ResearchContractModel):
    output_contract: Literal["e7a_research_output_v1"] = OUTPUT_CONTRACT
    assessment: EvidenceAssessmentV1
    evidence_sufficient: bool
    summary: tuple[ResearchClaimV2, ...] = Field(default=(), max_length=8)
    findings: tuple[ResearchClaimV2, ...] = Field(default=(), max_length=32)
    evidence: tuple[ResearchEvidenceV2, ...] = Field(default=(), max_length=128)
    limitations: tuple[ResearchLimitationV1, ...] = Field(default=(), max_length=8)
    sources: tuple[ResearchSourceV2, ...] = Field(default=(), max_length=64)
    application_draft: ApplicationDraftV2 | None = None

    @model_validator(mode="after")
    def consistent(self):
        evidence = _source_relations(self.sources, self.evidence)
        if not set(self.assessment.evidence_ids) <= evidence.keys():
            _fail("invalid_evidence_reference")
        draft = self.application_draft.paragraphs if self.application_draft else ()
        claims = (*self.summary, *self.findings, *draft)
        if len({claim.claim_id for claim in claims}) != len(claims):
            _fail("invalid_evidence_reference")
        cited_ids = set()
        for claim in claims:
            for citation in claim.citations:
                item = evidence.get(citation.evidence_id)
                if item is None or (
                    item.source_type != citation.source_type or item.source_id != citation.source_id
                ):
                    _fail("invalid_evidence_reference")
                cited_ids.add(citation.evidence_id)
        outcome = self.assessment.outcome
        if self.evidence_sufficient != (outcome == "sufficient"):
            _fail("contradictory_output")
        codes = {item.code for item in self.limitations}
        if outcome == "sufficient":
            if not (self.summary or self.findings) or codes:
                _fail("contradictory_output")
        else:
            if self.application_draft is not None:
                _fail("contradictory_output")
            if outcome == "conflicting":
                if (
                    "conflicting_evidence" not in codes
                    or not set(self.assessment.evidence_ids) <= cited_ids
                ):
                    _fail("contradictory_output")
            elif "insufficient_evidence" not in codes or "conflicting_evidence" in codes:
                _fail("contradictory_output")
            if outcome == "partial" and not (self.summary or self.findings):
                _fail("contradictory_output")
            if outcome == "insufficient" and (self.summary or self.findings):
                _fail("contradictory_output")
        return self


def validate_output(value: E7AResearchOutputV1, context: EvidenceContext) -> E7AResearchOutputV1:
    context = _context(context)
    value = _fresh(E7AResearchOutputV1, value)
    validate_assessment(value.assessment, context)
    sources = {item.source_id: item for item in context.sources}
    evidence = {item.evidence_id: item for item in context.evidence}
    if any(sources.get(item.source_id) != item for item in value.sources) or any(
        evidence.get(item.evidence_id) != item for item in value.evidence
    ):
        _fail("invalid_source_scope")
    if value.application_draft is not None:
        if not context.request.include_application_draft:
            _fail("contradictory_output")
        if not any(
            evidence[citation.evidence_id].source_type == "workspace_document"
            and evidence[citation.evidence_id].document_id == context.resume_document_id
            for paragraph in value.application_draft.paragraphs
            for citation in paragraph.citations
        ):
            _fail("invalid_source_scope")
    return value


def parse_output(raw: str | bytes, context: EvidenceContext) -> E7AResearchOutputV1:
    return validate_output(_read_json(E7AResearchOutputV1, raw, MAX_OUTPUT_BYTES), context)


def build_output(*, assessment: EvidenceAssessmentV1, context: EvidenceContext, **report):
    """Application derives the compatibility boolean, never accepts it from a writer."""
    assessment = validate_assessment(assessment, context)
    if {"assessment", "evidence_sufficient", "output_contract"} & report.keys():
        _fail("invalid_schema")
    try:
        output = E7AResearchOutputV1(
            assessment=assessment, evidence_sufficient=assessment.outcome == "sufficient", **report
        )
    except ValidationError as error:
        raise E7AContractError(_validation_category(error)) from None
    return validate_output(output, context)


Count = Annotated[int, Field(strict=True, ge=0, le=12)]
AssessmentCount = Annotated[int, Field(strict=True, ge=0, le=1)]


class E7ABudgetUsageV1(ResearchContractModel):
    """Trusted cumulative usage; reservations are not consumed calls or DB facts."""

    plan_calls: int = Field(default=0, ge=0, le=2)
    research_calls: tuple[Count, Count] = (0, 0)
    assessment_calls: tuple[AssessmentCount, AssessmentCount] = (0, 0)
    writer_calls: int = Field(default=0, ge=0, le=2)
    tool_calls: tuple[Count, Count] = (0, 0)

    @property
    def model_calls(self) -> int:
        return (
            self.plan_calls
            + sum(self.research_calls)
            + sum(self.assessment_calls)
            + self.writer_calls
        )

    @model_validator(mode="after")
    def within_run_limits(self):
        if self.model_calls > MAX_MODEL_CALLS or sum(self.tool_calls) > MAX_TOOL_CALLS:
            _fail("budget_exhausted")
        return self


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    model_calls: int
    tool_calls: int = 0
    reserved_model_calls: int = 0
    reserved_tool_calls: int = 0


def validate_usage_progress(previous: E7ABudgetUsageV1, current: E7ABudgetUsageV1):
    """Check successive trusted snapshots without owning or persisting usage."""
    previous = _fresh(E7ABudgetUsageV1, previous)
    current = _fresh(E7ABudgetUsageV1, current)
    for old, new in (
        (previous.plan_calls, current.plan_calls),
        (previous.writer_calls, current.writer_calls),
        *zip(previous.research_calls, current.research_calls, strict=True),
        *zip(previous.assessment_calls, current.assessment_calls, strict=True),
        *zip(previous.tool_calls, current.tool_calls, strict=True),
    ):
        if new < old:
            _fail("configuration_error")
    return current


def _followup_semantics(assessment, retrievable_gap_codes):
    if type(retrievable_gap_codes) is not tuple or any(
        type(code) is not str or code not in GAP_CODES for code in retrievable_gap_codes
    ):
        _fail("configuration_error")
    if len(set(retrievable_gap_codes)) != len(retrievable_gap_codes):
        _fail("configuration_error")
    if assessment is None:
        return False
    assessment = _fresh(EvidenceAssessmentV1, assessment)
    return assessment.outcome in {"partial", "insufficient"} and any(
        gap.code in retrievable_gap_codes for gap in assessment.gaps
    )


def followup_allowed(
    usage: E7ABudgetUsageV1,
    assessment: EvidenceAssessmentV1,
    *,
    retrievable_gap_codes: tuple[GapCode, ...],
) -> bool:
    usage = _fresh(E7ABudgetUsageV1, usage)
    eligible = _followup_semantics(assessment, retrievable_gap_codes)
    return (
        eligible
        and not usage.writer_calls
        and not usage.research_calls[1]
        and not usage.tool_calls[1]
        and not usage.assessment_calls[1]
        and MAX_MODEL_CALLS - usage.model_calls >= 5
        and MAX_TOOL_CALLS - sum(usage.tool_calls) >= 1
    )


def allocate_budget(
    usage: E7ABudgetUsageV1,
    *,
    stage: Literal["plan", "research", "assessment", "writer"],
    pass_number: int = 1,
    assessment: EvidenceAssessmentV1 | None = None,
    retrievable_gap_codes: tuple[GapCode, ...] = (),
) -> BudgetAllocation:
    """Return new-call allowance, preserving assessment/writer and optional followup.

    Full two-pass example: plan 2 + research 4/2 + assess 1/1 + writer 2 = 12.
    First-pass reservation is based on pass-entry usage, so recomputing after a
    consumed call cannot accidentally release promised followup slots. A caller
    must stop at zero and supply updated actual usage before admitting more work.
    """
    usage = _fresh(E7ABudgetUsageV1, usage)
    if (
        type(stage) is not str
        or stage not in {"plan", "research", "assessment", "writer"}
        or (type(pass_number) is not int or pass_number not in {1, 2})
    ):
        _fail("configuration_error")
    remaining = MAX_MODEL_CALLS - usage.model_calls
    tools = MAX_TOOL_CALLS - sum(usage.tool_calls)
    if stage == "plan":
        if usage.model_calls != usage.plan_calls or sum(usage.tool_calls):
            _fail("configuration_error")
        if usage.plan_calls == 2:
            _fail("budget_exhausted")
        return BudgetAllocation(2 - usage.plan_calls)
    if stage == "writer":
        if assessment is None:
            _fail("configuration_error")
        _fresh(EvidenceAssessmentV1, assessment)
        needed = 2 - usage.writer_calls
        if needed <= 0 or remaining < needed:
            _fail("budget_exhausted")
        return BudgetAllocation(needed)
    if usage.writer_calls:
        _fail("configuration_error")
    index = pass_number - 1
    if pass_number == 1 and (
        usage.research_calls[1] or usage.assessment_calls[1] or usage.tool_calls[1]
    ):
        _fail("configuration_error")
    if usage.assessment_calls[index]:
        _fail("budget_exhausted")
    if remaining < 3:
        _fail("budget_exhausted")
    if stage == "assessment":
        return BudgetAllocation(1, reserved_model_calls=2)
    reserved_models, reserved_tools = 3, 0
    if pass_number == 1:
        entry_models = remaining + usage.research_calls[0]
        entry_tools = tools + usage.tool_calls[0]
        if entry_models >= 8 and entry_tools >= 2:
            reserved_models += 3
            reserved_tools = 1
    elif (
        not _followup_semantics(assessment, retrievable_gap_codes)
        or remaining + usage.research_calls[1] < 5
        or tools + usage.tool_calls[1] < 1
    ):
        return BudgetAllocation(0, reserved_model_calls=3)
    models = remaining - reserved_models
    tools -= reserved_tools
    if models < 0 or tools < 0:
        _fail("budget_exhausted")
    if not usage.research_calls[index] and (models < 2 or tools == 0):
        return BudgetAllocation(
            0, reserved_model_calls=reserved_models, reserved_tool_calls=reserved_tools
        )
    return BudgetAllocation(models, tools, reserved_models, reserved_tools)
