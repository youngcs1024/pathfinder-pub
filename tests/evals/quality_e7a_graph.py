"""E7-A research and generation slices, without actions or durable recovery.

The caller supplies an already bound Registry runtime and authorized source scope.
Usage notifications precede calls, including failures. Factory/Registry retain their
existing accounting authority; this slice only enforces logical-call admission.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import Field, model_validator

from app.agents.contracts import AgentLoopControl, AgentLoopLimitsV1, AgentLoopObserver
from app.agents.create_agent_loop import (
    AgentLoopCancelled,
    AgentLoopDeadlineExceeded,
    run_create_agent_tool_loop,
)
from app.agents.research_contracts import (
    PlanNodeInputV1,
    ResearchNodeInputV1,
    ResearchNodeOutputV1,
    ResearchPlanV1,
)
from app.agents.research_graph import (
    _merge_document_calls,
    _merge_evidence,
    _merge_search_calls,
    _merge_sources,
)
from app.agents.research_nodes import (
    StructuredResearchPlanNode,
    _check_single_model_boundary,
    _research_outputs_from_transcript,
    _research_user_message,
    _validated_document_output,
)
from app.domain.research import (
    ResearchContractModel,
    ResearchEvidenceV2,
    ResearchRequestV1,
    ResearchSourceV2,
    normalize_research_query,
)
from app.domain.runs import DEFAULT_RUN_LIMITS
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.invocations import LLMInvocationAuthorizationError, LLMInvocationContext
from app.llm.ports import ChatMessage, ChatModelResult
from app.retrieval.documents import EMBEDDING_PROFILE
from app.tools.contracts import ToolRuntime
from app.tools.document_retrieval import RetrieveDocumentsInputV1
from app.tools.registry import (
    ToolCancelledError,
    ToolInputValidationError,
    ToolNotAllowedError,
    ToolTimeoutError,
    ToolUnavailableError,
)
from app.tools.web_search import SearchWebInputV1
from tests.evals.quality_e7a_assessment import EvidenceAssessmentNode
from tests.evals.quality_e7a_contracts import (
    GAP_CODES,
    E7ABudgetUsageV1,
    E7AResearchOutputV1,
    EvidenceAssessmentV1,
    EvidenceContext,
    GapCode,
    allocate_budget,
    assessment_input,
    followup_allowed,
    validate_assessment,
    validate_usage_progress,
)
from tests.evals.quality_e7a_writer import E7AWriterNode, WriterSummaryV1, draft_eligible

GRAPH_VERSION = "pathfinder-research-e7a-exp-v1"
TOOLS = frozenset({"search_web", "retrieve_documents"})
ERRORS = frozenset(
    {
        "configuration_error",
        "invalid_schema",
        "invalid_json",
        "invalid_evidence_reference",
        "invalid_source_scope",
        "contradictory_output",
        "budget_exhausted",
        "cancelled",
        "deadline_exceeded",
        "provider_timeout",
        "provider_unavailable",
        "model_invocation_failed",
        "invalid_model_output",
        "model_output_incomplete",
        "source_identity_conflict",
        "evidence_identity_conflict",
        "invalid_tool_output",
        "research_failed",
    }
)
type ToolName = Literal["search_web", "retrieve_documents"]
type StopReason = Literal[
    "sufficient",
    "conflicting",
    "no_retrievable_gap",
    "budget_exhausted",
    "no_executable_query",
    "no_new_evidence",
    "pass_limit",
]


class E7AGraphError(Exception):
    def __init__(self, category: str):
        self.category = (
            category if type(category) is str and category in ERRORS else "research_failed"
        )
        super().__init__(self.category)


def _checked(model, value):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return model.model_validate_json(raw, strict=True)
    except Exception:
        raise E7AGraphError("invalid_schema") from None


@dataclass(frozen=True, slots=True, repr=False)
class E7ASourceScope:
    """Trusted restrictions, matching the caller's existing Registry binding.

    Missing generic/task and job source policies mean no followup for that code.
    The adapter can narrow a binding; it never creates a new document binding.
    """

    allowed_document_ids: tuple[UUID, ...] = ()
    resume_document_id: UUID | None = None
    web_available: bool = False
    job_tools: tuple[ToolName, ...] = ()
    task_tools: tuple[ToolName, ...] = ()
    embedding_profile: str = EMBEDDING_PROFILE

    def __post_init__(self):
        if (
            type(self.allowed_document_ids) is not tuple
            or any(type(x) is not UUID for x in self.allowed_document_ids)
            or len(set(self.allowed_document_ids)) != len(self.allowed_document_ids)
            or len(self.allowed_document_ids) > 8
            or type(self.web_available) is not bool
            or self.embedding_profile != EMBEDDING_PROFILE
            or (
                self.resume_document_id is not None
                and (
                    type(self.resume_document_id) is not UUID
                    or self.resume_document_id not in self.allowed_document_ids
                )
            )
        ):
            raise E7AGraphError("invalid_source_scope")
        for names in (self.job_tools, self.task_tools):
            if (
                type(names) is not tuple
                or any(type(x) is not str or x not in TOOLS for x in names)
                or len(set(names)) != len(names)
                or ("search_web" in names and not self.web_available)
                or ("retrieve_documents" in names and not self.allowed_document_ids)
            ):
                raise E7AGraphError("invalid_source_scope")

    def tools_for(self, code: GapCode) -> tuple[ToolName, ...]:
        if code in {"missing_resume_fact", "unsupported_claim_strength"}:
            return ("retrieve_documents",) if self.resume_document_id is not None else ()
        if code == "missing_job_fact":
            return self.job_tools
        if code == "missing_task_fact":
            return self.task_tools
        return ()


@dataclass(slots=True, repr=False)
class E7AUsageOwner:
    """Single-invocation owner; callers retain this object even when the graph fails."""

    on_admitted: Callable[[E7ABudgetUsageV1], None]
    usage: E7ABudgetUsageV1 = field(default_factory=E7ABudgetUsageV1)
    started: bool = False

    def publish(self, value):
        self.usage = validate_usage_progress(self.usage, value)
        if self.on_admitted(self.usage) is not None:
            raise E7AGraphError("configuration_error")

    def admit(self, stage, pass_number):
        previous = self.usage
        if stage == "plan":
            changes = {"plan_calls": previous.plan_calls + 1}
        else:
            field_name = "tool_calls" if stage == "tool" else "research_calls"
            counts = list(getattr(previous, field_name))
            counts[pass_number - 1] += 1
            changes = {field_name: tuple(counts)}
        self.publish(previous.model_copy(update=changes))


@dataclass(frozen=True, slots=True, repr=False)
class E7AGraphRuntime:
    factory: LLMFactory
    invocation_context: LLMInvocationContext
    tool_runtime: ToolRuntime
    scope: E7ASourceScope
    control: AgentLoopControl
    observer: AgentLoopObserver
    usage_owner: E7AUsageOwner

    def __post_init__(self):
        if (
            not isinstance(self.factory, LLMFactory)
            or type(self.invocation_context) is not LLMInvocationContext
            or self.invocation_context.run_id is None
            or not isinstance(self.tool_runtime, ToolRuntime)
            or type(self.scope) is not E7ASourceScope
            or type(self.control) is not AgentLoopControl
            or not isinstance(self.observer, AgentLoopObserver)
            or type(self.usage_owner) is not E7AUsageOwner
            or not callable(self.usage_owner.on_admitted)
            or self.control.limits != AgentLoopLimitsV1(**dict(DEFAULT_RUN_LIMITS))
        ):
            raise E7AGraphError("configuration_error")


class FollowupQueryV1(ResearchContractModel):
    gap_code: GapCode
    tool_name: ToolName
    query: str = Field(min_length=1, max_length=200)
    identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class E7APassSummaryV1(ResearchContractModel):
    pass_number: int = Field(ge=1, le=2)
    gap_codes: tuple[GapCode, ...] = Field(default=(), max_length=5)
    model_call_count: int = Field(ge=0, le=12)
    tool_call_count: int = Field(ge=0, le=8)
    duplicate_query_count: int = Field(ge=0)
    rejected_proposal_count: int = Field(ge=0)
    new_content_count: int = Field(ge=0, le=128)
    duplicate_content_count: int = Field(ge=0, le=128)
    obtained_new_evidence: bool


class E7AResearchStateV1(ResearchContractModel):
    graph_version: Literal["pathfinder-research-e7a-exp-v1"] = GRAPH_VERSION
    request: ResearchRequestV1
    plan: ResearchPlanV1 | None = None
    research: ResearchNodeOutputV1 = Field(default_factory=ResearchNodeOutputV1)
    assessment: EvidenceAssessmentV1 | None = None
    usage: E7ABudgetUsageV1 = Field(default_factory=E7ABudgetUsageV1)
    pass_count: int = Field(default=0, ge=0, le=2)
    followup_plan: tuple[FollowupQueryV1, ...] = Field(default=(), max_length=8)
    executed_query_ids: tuple[str, ...] = Field(default=(), max_length=8)
    summaries: tuple[E7APassSummaryV1, ...] = Field(default=(), max_length=2)
    stop_reason: StopReason | None = None


class E7AResearchInputV1(ResearchContractModel):
    request: ResearchRequestV1


class GraphEnvelope(TypedDict):
    payload: dict[str, object]


def _envelope(state):
    return {"payload": state.model_dump(mode="json")}


def evidence_context(state: E7AResearchStateV1, scope: E7ASourceScope) -> EvidenceContext:
    data = state.research
    sources = (
        tuple(ResearchSourceV2(source_type="web", **x.model_dump()) for x in data.sources)
        + data.document_sources
    )
    evidence = (
        tuple(ResearchEvidenceV2(source_type="web", **x.model_dump()) for x in data.evidence)
        + data.document_evidence
    )
    context = EvidenceContext(state.request, sources, evidence, scope.resume_document_id)
    assessment_input(context)
    return context


def _content_id(text):
    # Preserve case/punctuation differences: this is duplicate content, not entailment.
    return sha256(normalize_research_query(text).encode("utf-8")).hexdigest()


def query_identity(runtime: E7AGraphRuntime, name: str, query: str) -> str:
    scope = runtime.scope
    parts = [name, str(runtime.invocation_context.workspace_id)]
    if name == "retrieve_documents":
        parts += [scope.embedding_profile, *sorted(str(x) for x in scope.allowed_document_ids)]
    parts.append(normalize_research_query(query).casefold())
    return "sha256:" + sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def _available_tools(runtime):
    names = {x.name for x in runtime.tool_runtime.model_tools()}
    return (
        names
        & (
            ({"search_web"} if runtime.scope.web_available else set())
            | ({"retrieve_documents"} if runtime.scope.allowed_document_ids else set())
        )
        & TOOLS
    )


def _retrievable_codes(runtime):
    available = _available_tools(runtime)
    return tuple(
        sorted(code for code in GAP_CODES if available.intersection(runtime.scope.tools_for(code)))
    )


def build_gap_followup_plan(state, runtime):
    context = evidence_context(state, runtime.scope)
    assessment = validate_assessment(state.assessment, context)
    if state.pass_count != 1 or not followup_allowed(
        runtime.usage_owner.usage, assessment, retrievable_gap_codes=_retrievable_codes(runtime)
    ):
        return ()
    allocation = allocate_budget(
        runtime.usage_owner.usage,
        stage="research",
        pass_number=2,
        assessment=assessment,
        retrievable_gap_codes=_retrievable_codes(runtime),
    )
    seen = set(state.executed_query_ids)
    available = _available_tools(runtime)
    queries = []
    for gap in assessment.gaps:
        query = normalize_research_query(gap.topic)
        if not query or len(query) > 200 or len(query.encode("utf-8")) > 800:
            raise E7AGraphError("invalid_schema")
        for name in runtime.scope.tools_for(gap.code):
            identity = query_identity(runtime, name, query)
            if name in available and identity not in seen:
                queries.append(
                    FollowupQueryV1(
                        gap_code=gap.code,
                        tool_name=name,
                        query=query,
                        identity=identity,
                    )
                )
                seen.add(identity)
                if len(queries) == allocation.tool_calls:
                    return tuple(queries)
    return tuple(queries)


@dataclass(slots=True, repr=False)
class _ResearchTools:
    runtime: E7AGraphRuntime
    state: E7AResearchStateV1
    pass_number: int
    allowed_models: int
    allowed_calls: int
    identities: list[str] = field(default_factory=list)
    model_calls: int = 0
    calls: int = 0
    duplicate_queries: int = 0
    rejected_proposals: int = 0
    document_hits: dict = field(default_factory=dict)

    def model_tools(self):
        available = _available_tools(self.runtime)
        if self.pass_number == 2:
            available &= {x.tool_name for x in self.state.followup_plan}
        return tuple(x for x in self.runtime.tool_runtime.model_tools() if x.name in available)

    def identity(self, call):
        model = SearchWebInputV1 if call.name == "search_web" else RetrieveDocumentsInputV1
        try:
            value = model.model_validate(call.arguments, strict=True)
        except Exception:
            raise ToolInputValidationError("invalid query") from None
        if not normalize_research_query(value.query):
            raise ToolInputValidationError("invalid query")
        return query_identity(self.runtime, call.name, value.query)

    def validate_call(self, call):
        if call.name not in {x.name for x in self.model_tools()}:
            raise ToolNotAllowedError("tool outside source scope")
        self.runtime.tool_runtime.validate_call(call)
        identity = self.identity(call)
        if self.pass_number == 2 and identity not in {x.identity for x in self.state.followup_plan}:
            raise ToolInputValidationError("query outside gap plan")

    def filter_response(self, response):
        if not isinstance(response, ChatModelResult):
            raise E7AGraphError("invalid_model_output")
        if response.finish_status != "completed":
            raise E7AGraphError("model_output_incomplete")
        if not response.tool_calls:
            return response
        accepted = []
        seen = set(self.state.executed_query_ids) | set(self.identities)
        try:
            for call in response.tool_calls:
                self.validate_call(call)
                key = self.identity(call)
                if key in seen:
                    self.duplicate_queries += 1
                    continue
                accepted.append(call)
                seen.add(key)
        except (ToolNotAllowedError, ToolInputValidationError):
            self.rejected_proposals += len(response.tool_calls)
            accepted = []
        # Leave room for the loop's closing model call; do not manufacture a call.
        if self.model_calls >= self.allowed_models:
            accepted = []
        if not accepted:
            return response.model_copy(
                update={"content": "Bounded research pass complete.", "tool_calls": ()}
            )
        return response.model_copy(update={"tool_calls": tuple(accepted)})

    async def execute(self, call):
        self.validate_call(call)
        _check_single_model_boundary(self.runtime.control)
        identity = self.identity(call)
        if identity in self.state.executed_query_ids or identity in self.identities:
            raise E7AGraphError("configuration_error")
        if self.calls >= self.allowed_calls:
            raise E7AGraphError("budget_exhausted")
        self.runtime.usage_owner.admit("tool", self.pass_number)
        self.calls += 1
        self.identities.append(identity)
        result = await self.runtime.tool_runtime.execute(call)
        _check_single_model_boundary(self.runtime.control)
        if call.name == "retrieve_documents":
            output = _validated_document_output(result)
            if output.result_count != len(output.results):
                raise E7AGraphError("invalid_tool_output")
            existing = {x.chunk_id: x for x in self.state.research.document_evidence}
            chunks = set()
            for hit in output.results:
                if hit.document_id not in self.runtime.scope.allowed_document_ids:
                    raise E7AGraphError("invalid_source_scope")
                previous = existing.get(hit.chunk_id)
                if previous is not None and (
                    previous.document_id != hit.document_id
                    or previous.text != hit.untrusted_text
                    or previous.section != hit.section
                    or previous.ordinal != hit.ordinal
                ):
                    raise E7AGraphError("evidence_identity_conflict")
                if hit.chunk_id in chunks:
                    raise E7AGraphError("evidence_identity_conflict")
                identity = (hit.document_id, hit.untrusted_text, hit.section, hit.ordinal)
                if (
                    hit.chunk_id in self.document_hits
                    and self.document_hits[hit.chunk_id] != identity
                ):
                    raise E7AGraphError("evidence_identity_conflict")
                self.document_hits[hit.chunk_id] = identity
                chunks.add(hit.chunk_id)
        return result


@dataclass(slots=True, repr=False)
class _AdmittedModel:
    runtime: E7AGraphRuntime
    stage: str
    research_tools: _ResearchTools | None = None

    async def invoke(self, messages, tools, metadata):
        runtime = self.runtime
        _check_single_model_boundary(runtime.control)
        if self.stage == "plan":
            allocate_budget(runtime.usage_owner.usage, stage="plan")
            runtime.usage_owner.admit("plan", 1)
        else:
            research = self.research_tools
            if research.model_calls >= research.allowed_models:
                raise E7AGraphError("budget_exhausted")
            allocate_budget(
                runtime.usage_owner.usage,
                stage="research",
                pass_number=research.pass_number,
                assessment=research.state.assessment,
                retrievable_gap_codes=_retrievable_codes(runtime),
            )
            runtime.usage_owner.admit("research", research.pass_number)
            research.model_calls += 1
        result = await runtime.factory.create_chat_model(runtime.invocation_context).invoke(
            messages,
            tools,
            metadata,
        )
        _check_single_model_boundary(runtime.control)
        return self.research_tools.filter_response(result) if self.research_tools else result


def _node_input(state, runtime, pass_number):
    previous = state.research
    queries = (
        tuple(dict.fromkeys(x.query for x in state.followup_plan))
        if pass_number == 2
        else state.plan.queries
    )
    return ResearchNodeInputV1(
        request=state.request,
        normalized_query=normalize_research_query(state.request.query),
        plan=ResearchPlanV1(queries=queries),
        existing_sources=previous.sources,
        existing_evidence=previous.evidence,
        existing_search_calls=previous.search_calls,
        existing_document_sources=previous.document_sources,
        existing_document_evidence=previous.document_evidence,
        existing_document_retrieval_calls=previous.document_retrieval_calls,
        research_pass_number=pass_number,
        document_scope_available=bool(runtime.scope.allowed_document_ids),
    )


def _merge_research(previous, additions, pass_number):
    return ResearchNodeOutputV1(
        sources=_merge_sources(previous.sources, additions.sources),
        evidence=_merge_evidence(previous.evidence, additions.evidence),
        search_calls=_merge_search_calls(
            previous.search_calls, additions.search_calls, research_pass_number=pass_number
        ),
        document_sources=_merge_sources(previous.document_sources, additions.document_sources),
        document_evidence=_merge_evidence(previous.document_evidence, additions.document_evidence),
        document_retrieval_calls=_merge_document_calls(
            previous.document_retrieval_calls, additions.document_retrieval_calls
        ),
    )


def _assessment_context(state, scope):
    context = evidence_context(state, scope)
    # Keep provenance types separate and prefer the designated resume's identity.
    # Identical copies within a source type are not additional assessor support.
    selected = {}
    for item in context.evidence:
        key = (item.source_type, _content_id(item.text))
        if key not in selected or (
            scope.resume_document_id is not None
            and item.document_id == scope.resume_document_id
            and selected[key].document_id != scope.resume_document_id
        ):
            selected[key] = item
    return EvidenceContext(
        context.request, context.sources, tuple(selected.values()), context.resume_document_id
    )


def _guarded(function):
    async def node(envelope, runtime: Runtime[E7AGraphRuntime]):
        try:
            if type(runtime.context) is not E7AGraphRuntime:
                raise E7AGraphError("configuration_error")
            _check_single_model_boundary(runtime.context.control)
            result = await function(envelope, runtime.context)
            _check_single_model_boundary(runtime.context.control)
            return result
        except asyncio.CancelledError:
            raise
        except (LLMInvocationAuthorizationError, ToolCancelledError, AgentLoopCancelled):
            raise E7AGraphError("cancelled") from None
        except AgentLoopDeadlineExceeded:
            raise E7AGraphError("deadline_exceeded") from None
        except ToolTimeoutError:
            raise E7AGraphError("provider_timeout") from None
        except ToolUnavailableError:
            raise E7AGraphError("provider_unavailable") from None
        except LLMProviderError as error:
            raise E7AGraphError(
                "provider_timeout"
                if error.category == "provider_timeout"
                else "provider_unavailable"
            ) from None
        except LLMAccountingError as error:
            raise E7AGraphError(
                "provider_unavailable" if error.retryable else "model_invocation_failed"
            ) from None
        except Exception as error:
            raise E7AGraphError(getattr(error, "category", "research_failed")) from None

    return node


def build_e7a_research_graph():
    """Compile a fresh-run research slice. Checkpoint/DB integration belongs to A.6."""

    async def normalize(envelope, runtime):
        request = _checked(E7AResearchInputV1, envelope["payload"]).request
        owner = runtime.usage_owner
        if owner.started or owner.usage != E7ABudgetUsageV1():
            raise E7AGraphError("configuration_error")
        owner.started = True
        state = E7AResearchStateV1(request=request)
        evidence_context(state, runtime.scope)
        return _envelope(state)

    async def plan(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        result = await StructuredResearchPlanNode(_AdmittedModel(runtime, "plan"))(
            PlanNodeInputV1(
                normalized_query=normalize_research_query(state.request.query),
                include_application_draft=state.request.include_application_draft,
            ),
            runtime.control,
        )
        return _envelope(
            state.model_copy(update={"plan": result.plan, "usage": runtime.usage_owner.usage})
        )

    async def research(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        number = state.pass_count + 1
        if number > 2 or (number == 2 and not state.followup_plan):
            raise E7AGraphError("configuration_error")
        allowance = allocate_budget(
            runtime.usage_owner.usage,
            stage="research",
            pass_number=number,
            assessment=state.assessment,
            retrievable_gap_codes=_retrievable_codes(runtime),
        )
        adapter = _ResearchTools(
            runtime, state, number, allowance.model_calls, allowance.tool_calls
        )
        node_input = _node_input(state, runtime, number)
        merged = state.research
        if allowance.model_calls and allowance.tool_calls and adapter.model_tools():
            control = AgentLoopControl(
                limits=AgentLoopLimitsV1(
                    max_model_calls=allowance.model_calls,
                    max_tool_calls=allowance.tool_calls,
                    max_tool_results=allowance.tool_calls,
                    max_iterations=runtime.control.limits.max_iterations,
                ),
                deadline=runtime.control.deadline,
                cancellation=runtime.control.cancellation,
                clock=runtime.control.clock,
            )
            result = await run_create_agent_tool_loop(
                model=_AdmittedModel(runtime, "research", adapter),
                messages=(_research_message(node_input, state, number),),
                metadata={"graph_node": "research_agent", "research_pass_number": str(number)},
                tool_runtime=adapter,
                control=control,
                observer=runtime.observer,
                prompt_profile="research",
            )
            additions = _research_outputs_from_transcript(
                node_input=node_input,
                transcript=result.transcript,
                reported_tool_call_count=result.tool_call_count,
            )
            merged = _merge_research(state.research, additions, number)
        old_content = {_content_id(x.text) for x in evidence_context(state, runtime.scope).evidence}
        updated = state.model_copy(update={"research": merged})
        all_evidence = evidence_context(updated, runtime.scope).evidence
        contents = {_content_id(x.text) for x in all_evidence}
        new_count = len(contents - old_content)
        summary = E7APassSummaryV1(
            pass_number=number,
            gap_codes=tuple(dict.fromkeys(x.gap_code for x in state.followup_plan)),
            model_call_count=adapter.model_calls,
            tool_call_count=adapter.calls,
            duplicate_query_count=adapter.duplicate_queries,
            rejected_proposal_count=adapter.rejected_proposals,
            new_content_count=new_count,
            duplicate_content_count=len(all_evidence) - len(contents),
            obtained_new_evidence=bool(new_count),
        )
        return _envelope(
            updated.model_copy(
                update={
                    "pass_count": number,
                    "usage": runtime.usage_owner.usage,
                    "executed_query_ids": (*state.executed_query_ids, *adapter.identities),
                    "summaries": (*state.summaries, summary),
                    "stop_reason": "no_new_evidence" if number == 2 and not new_count else None,
                }
            )
        )

    async def assess(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        result = await EvidenceAssessmentNode(runtime.factory, runtime.invocation_context)(
            _assessment_context(state, runtime.scope),
            usage=runtime.usage_owner.usage,
            pass_number=state.pass_count,
            control=runtime.control,
            on_admitted=runtime.usage_owner.publish,
        )
        return _envelope(
            state.model_copy(update={"assessment": result.assessment, "usage": result.usage})
        )

    async def followup(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        outcome = state.assessment.outcome
        reason = None
        queries = ()
        if outcome in {"sufficient", "conflicting"}:
            reason = outcome
        elif state.pass_count == 2:
            reason = "pass_limit"
        elif not any(x.code in _retrievable_codes(runtime) for x in state.assessment.gaps):
            reason = "no_retrievable_gap"
        elif not followup_allowed(
            runtime.usage_owner.usage,
            state.assessment,
            retrievable_gap_codes=_retrievable_codes(runtime),
        ):
            reason = "budget_exhausted"
        else:
            queries = build_gap_followup_plan(state, runtime)
            if not queries:
                reason = "no_executable_query"
        return _envelope(state.model_copy(update={"followup_plan": queries, "stop_reason": reason}))

    async def finish(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        validate_assessment(state.assessment, evidence_context(state, runtime.scope))
        validate_usage_progress(state.usage, runtime.usage_owner.usage)
        allocate_budget(runtime.usage_owner.usage, stage="writer", assessment=state.assessment)
        if state.stop_reason is None:
            raise E7AGraphError("configuration_error")
        return _envelope(state.model_copy(update={"usage": runtime.usage_owner.usage}))

    graph = StateGraph(GraphEnvelope, context_schema=E7AGraphRuntime)
    for name, function in (
        ("normalize", normalize),
        ("plan", plan),
        ("research", research),
        ("assess", assess),
        ("followup", followup),
        ("finish", finish),
    ):
        graph.add_node(name, _guarded(function))
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "plan")
    graph.add_edge("plan", "research")
    graph.add_conditional_edges(
        "research",
        lambda value: "finish" if value["payload"]["stop_reason"] else "assess",
        {"finish": "finish", "assess": "assess"},
    )
    graph.add_edge("assess", "followup")
    graph.add_conditional_edges(
        "followup",
        lambda value: "finish" if value["payload"]["stop_reason"] else "research",
        {"finish": "finish", "research": "research"},
    )
    graph.add_edge("finish", END)
    return graph.compile(name=GRAPH_VERSION, checkpointer=False)


def _research_message(node_input, state, number):
    content = _research_user_message(node_input)
    if number == 2:
        # Data projection only; code validates every resulting proposal against it.
        content += (
            "\n<untrusted_gap_queries>\n"
            + json.dumps(
                [
                    {"gap_code": x.gap_code, "tool_name": x.tool_name, "query": x.query}
                    for x in state.followup_plan
                ],
                ensure_ascii=False,
            )
            + "\n</untrusted_gap_queries>"
        )
    return ChatMessage(role="user", content=content)


class E7AGenerationResultV1(ResearchContractModel):
    """Completed generation only, not a persisted Run or application submission."""

    research_state: E7AResearchStateV1
    output: E7AResearchOutputV1
    writer_summary: WriterSummaryV1
    draft_eligible: bool
    status: Literal["completed"] = "completed"

    @model_validator(mode="after")
    def consistent(self):
        if (
            self.research_state.assessment != self.output.assessment
            or self.research_state.stop_reason is None
            or self.writer_summary.outcome != self.output.assessment.outcome
            or self.writer_summary.draft_eligible != self.draft_eligible
            or self.research_state.usage.writer_calls != self.writer_summary.model_call_count
            or self.draft_eligible
            != (
                self.output.assessment.outcome == "sufficient"
                and self.research_state.request.include_application_draft
                and self.output.application_draft is not None
            )
        ):
            raise ValueError("inconsistent generation result")
        return self


def build_e7a_generation_graph():
    """Compose the unchanged A.4 research slice with Writer and strict finalization.

    This fresh-run seam accepts the same request envelope and trusted runtime.
    It has no action store, approval interrupt, DB root or persistent checkpointer.
    """
    research_graph = build_e7a_research_graph()

    async def research(envelope, runtime):
        return await research_graph.ainvoke(envelope, context=runtime)

    async def write(envelope, runtime):
        state = _checked(E7AResearchStateV1, envelope["payload"])
        if (
            not runtime.usage_owner.started
            or state.usage != runtime.usage_owner.usage
            or state.stop_reason is None
            or state.usage.writer_calls
        ):
            raise E7AGraphError("configuration_error")
        result = await E7AWriterNode(runtime.factory, runtime.invocation_context)(
            evidence_context(state, runtime.scope),
            state.assessment,
            usage=runtime.usage_owner.usage,
            control=runtime.control,
            on_admitted=runtime.usage_owner.publish,
        )
        return _envelope(
            E7AGenerationResultV1(
                research_state=state.model_copy(update={"usage": result.usage}),
                output=result.output,
                writer_summary=result.summary,
                draft_eligible=result.summary.draft_eligible,
            )
        )

    async def finalize(envelope, runtime):
        result = _checked(E7AGenerationResultV1, envelope["payload"])
        context = evidence_context(result.research_state, runtime.scope)
        if (
            result.research_state.usage != runtime.usage_owner.usage
            or result.draft_eligible != draft_eligible(result.output, context)
        ):
            raise E7AGraphError("configuration_error")
        return _envelope(result)

    graph = StateGraph(GraphEnvelope, context_schema=E7AGraphRuntime)
    for name, function in (("research", research), ("write_report", write), ("finalize", finalize)):
        graph.add_node(name, _guarded(function))
    graph.add_edge(START, "research")
    graph.add_edge("research", "write_report")
    graph.add_edge("write_report", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(name=GRAPH_VERSION, checkpointer=False)
