from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Literal, Protocol, TypedDict, get_args
from uuid import UUID, uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from app.agents.contracts import AgentLoopControl
from app.agents.research_contracts import (
    DocumentRetrievalCallTraceV1,
    EvidenceValidationNodeInputV1,
    EvidenceValidationNodeOutputV1,
    PlanNodeInputV1,
    PlanNodeOutputV1,
    ResearchContractModel,
    ResearchEvidenceV1,
    ResearchEvidenceV2,
    ResearchGraphInputV1,
    ResearchGraphOutputStateV1,
    ResearchGraphStateV1,
    ResearchLimitationV1,
    ResearchNodeInputV1,
    ResearchNodeOutputV1,
    ResearchOutputV1,
    ResearchOutputV2,
    ResearchRequestV1,
    ResearchSourceV1,
    ResearchSourceV2,
    SearchCallTraceV1,
    WriteReportNodeInputV1,
    WriteReportNodeOutputV1,
)
from app.domain.action_execution import ActionApprovalExpiredError, ActionExecutionIdentity
from app.domain.actions import (
    ACTION_KEY,
    INITIAL_ACTION_REVISION,
    ActionStore,
    CancelActionCommand,
    PrepareActionCommand,
    SubmitApplicationArgsV1,
)
from app.domain.approvals import ApprovalRequestRecord, ApprovalResumeResolver, ApprovalStatus
from app.domain.research import (
    ApplicationDraftV2,
    ResearchCitationV2,
    ResearchClaimV2,
    normalize_research_query,
)
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantContext
from app.domain.tracing import (
    SpanStatus,
    TraceMetadataValue,
    bind_trace_scope,
    current_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.tools.contracts import ApprovedActionExecutor, ToolRuntime

ResearchGraphErrorCategory = Literal[
    "invalid_node_output",
    "node_execution_failed",
    "missing_graph_context",
    "missing_state_value",
    "invalid_research_pass_count",
    "source_identity_conflict",
    "evidence_identity_conflict",
    "evidence_source_missing",
    "search_trace_conflict",
    "unexpected_application_draft",
    "invalid_final_output",
    "invalid_action_payload",
    "action_identity_mismatch",
    "invalid_approval_resume",
]


def build_approval_resume_input(approval_request_id: UUID) -> object:
    if not isinstance(approval_request_id, UUID):
        raise TypeError("approval request identity must be a UUID")
    return Command(resume={"approval_request_id": str(approval_request_id)})


def approval_resume_was_applied(checkpoint_tuple: object) -> bool:
    """Return whether LangGraph durably accepted an interrupt resume input."""
    missing = object()
    pending_writes = getattr(checkpoint_tuple, "pending_writes", missing)
    if pending_writes is missing:
        raise TypeError("checkpoint tuple has no pending writes")
    if pending_writes is None:
        return False
    if not isinstance(pending_writes, list):
        raise TypeError("checkpoint pending writes are invalid")
    resume_was_applied = False
    for write in pending_writes:
        if (
            not isinstance(write, tuple)
            or len(write) != 3
            or not isinstance(write[0], str)
            or not isinstance(write[1], str)
        ):
            raise TypeError("checkpoint pending write is invalid")
        if write[1] == "__resume__":
            resume_was_applied = True
    return resume_was_applied


ResearchRoute = Literal["research_again", "write_report"]
ActionRoute = Literal["finalize", "prepare_action"]
ApprovalRoute = Literal["execute_mock_action", "cancel_action"]

APPROVAL_TTL = timedelta(hours=1)
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class ResearchGraphJsonInputV1(TypedDict):
    schema_version: int
    run_id: str
    workspace_id: str
    actor_user_id: str
    conversation_id: str
    graph_version: str
    mode: str
    resume_document_id: str | None
    request: dict[str, object]


class ResearchGraphJsonStateV1(ResearchGraphJsonInputV1, total=False):
    normalized_query: str | None
    plan: dict[str, object] | None
    sources: list[dict[str, object]]
    evidence: list[dict[str, object]]
    search_calls: list[dict[str, object]]
    document_sources: list[dict[str, object]]
    document_evidence: list[dict[str, object]]
    document_retrieval_calls: list[dict[str, object]]
    research_pass_count: int
    evidence_sufficient: bool | None
    validation_limitations: list[dict[str, object]]
    writer_output: dict[str, object] | None
    action_proposal_id: str | None
    action_key: str | None
    action_revision: int | None
    approval_expires_at: str | None
    approval_request_id: str | None
    output: dict[str, object] | None


class ResearchGraphProtocolError(Exception):
    def __init__(
        self,
        *,
        category: ResearchGraphErrorCategory,
        node_name: str,
        cause_category: str | None = None,
        limit_kind: str | None = None,
        limit: int | None = None,
        current_count: int | None = None,
        requested_count: int | None = None,
        schema_error_type: str | None = None,
        schema_error_path: str | None = None,
    ) -> None:
        self.category = category
        self.node_name = node_name
        self.cause_category = cause_category
        is_agent_limit = cause_category == "agent_limit_exceeded"
        self.limit_kind = (
            limit_kind
            if is_agent_limit
            and limit_kind
            in {"model_calls", "tool_calls", "tool_results", "iterations", "recursion"}
            else None
        )
        self.limit = limit if is_agent_limit and type(limit) is int and limit >= 0 else None
        self.current_count = (
            current_count
            if is_agent_limit and type(current_count) is int and current_count >= 0
            else None
        )
        self.requested_count = (
            requested_count
            if is_agent_limit and type(requested_count) is int and requested_count >= 0
            else None
        )
        self.schema_error_type = (
            schema_error_type
            if isinstance(schema_error_type, str)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", schema_error_type) is not None
            else None
        )
        self.schema_error_path = (
            schema_error_path
            if isinstance(schema_error_path, str)
            and 0 < len(schema_error_path) <= 256
            and all(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", segment) is not None
                for segment in schema_error_path.split(".")
            )
            else None
        )
        super().__init__(f"research graph protocol failure: {category} at {node_name}")


class PlanNode(Protocol):
    async def __call__(
        self,
        node_input: PlanNodeInputV1,
        control: AgentLoopControl,
    ) -> PlanNodeOutputV1: ...


class ResearchNode(Protocol):
    async def __call__(
        self,
        node_input: ResearchNodeInputV1,
        runtime_context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1: ...


class EvidenceValidationNode(Protocol):
    async def __call__(
        self, node_input: EvidenceValidationNodeInputV1
    ) -> EvidenceValidationNodeOutputV1: ...


class WriteReportNode(Protocol):
    async def __call__(
        self,
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1: ...


@dataclass(frozen=True, slots=True, repr=False)
class ResearchGraphRuntimeContext:
    tool_runtime: ToolRuntime
    agent_loop_control: AgentLoopControl
    after_research_node: Callable[[ResearchNodeOutputV1], Awaitable[None]] | None = None
    tenant: TenantContext | None = None
    action_store: ActionStore | None = None
    approval_resume_resolver: ApprovalResumeResolver | None = None
    approved_action_executor: ApprovedActionExecutor | None = None
    clock: Clock = utc_now

    def __post_init__(self) -> None:
        if not isinstance(self.tool_runtime, ToolRuntime):
            raise ValueError("research graph tool runtime must implement ToolRuntime")
        if not isinstance(self.agent_loop_control, AgentLoopControl):
            raise ValueError("research graph control must use AgentLoopControl")
        if self.tenant is not None and not isinstance(self.tenant, TenantContext):
            raise ValueError("research graph tenant must use TenantContext")
        if not callable(self.clock):
            raise ValueError("research graph clock must be callable")


def assemble_submit_application_args(
    *,
    run_id: UUID,
    resume_document_id: UUID,
    writer_output: WriteReportNodeOutputV1,
) -> SubmitApplicationArgsV1:
    draft = writer_output.application_draft
    if draft is None:
        raise ResearchGraphProtocolError(
            category="invalid_action_payload",
            node_name="prepare_action",
        )
    try:
        return SubmitApplicationArgsV1.model_validate(
            {
                "job_ref": f"pathfinder-mock-job-v1:{run_id}",
                "resume_document_id": resume_document_id,
                "answers": {},
                "cover_letter": "\n\n".join(paragraph.text for paragraph in draft.paragraphs),
            },
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        raise ResearchGraphProtocolError(
            category="invalid_action_payload",
            node_name="prepare_action",
        ) from None


@dataclass(frozen=True, slots=True, repr=False)
class ResearchGraphNodes:
    plan: PlanNode
    research_agent: ResearchNode
    validate_evidence: EvidenceValidationNode
    write_report: WriteReportNode

    def __post_init__(self) -> None:
        for name in ("plan", "research_agent", "validate_evidence", "write_report"):
            if not callable(getattr(self, name)):
                raise ValueError(f"research graph node must be callable: {name}")


def _validated_node_output[NodeOutputT: ResearchContractModel](
    model_type: type[NodeOutputT],
    value: object,
    *,
    node_name: str,
) -> NodeOutputT:
    try:
        return model_type.model_validate(value, strict=True)
    except (TypeError, ValidationError, ValueError):
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name=node_name,
        ) from None


async def _run_node[NodeOutputT](
    awaitable: Awaitable[NodeOutputT],
    *,
    node_name: str,
) -> NodeOutputT:
    try:
        return await awaitable
    except Exception as error:
        raise ResearchGraphProtocolError(
            category="node_execution_failed",
            node_name=node_name,
            cause_category=getattr(error, "category", None),
            limit_kind=getattr(error, "limit_kind", None),
            limit=getattr(error, "limit", None),
            current_count=getattr(error, "current_count", None),
            requested_count=getattr(error, "requested_count", None),
            schema_error_type=getattr(error, "schema_error_type", None),
            schema_error_path=getattr(error, "schema_error_path", None),
        ) from None


def _required_state_value[ValueT](
    value: ValueT | None,
    *,
    node_name: str,
) -> ValueT:
    if value is None:
        raise ResearchGraphProtocolError(
            category="missing_state_value",
            node_name=node_name,
        )
    return value


def _validated_graph_state(
    state: ResearchGraphJsonStateV1,
    *,
    node_name: str,
) -> ResearchGraphStateV1:
    try:
        payload = json.dumps(
            state,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return ResearchGraphStateV1.model_validate_json(payload, strict=True)
    except (TypeError, ValidationError, ValueError):
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name=node_name,
        ) from None


def _runtime_context(
    runtime: Runtime[ResearchGraphRuntimeContext],
    *,
    node_name: str,
) -> ResearchGraphRuntimeContext:
    if not isinstance(runtime.context, ResearchGraphRuntimeContext):
        raise ResearchGraphProtocolError(
            category="missing_graph_context",
            node_name=node_name,
        )
    return runtime.context


def _merge_sources(
    existing: tuple[ResearchSourceV1, ...],
    additions: tuple[ResearchSourceV1, ...],
) -> tuple[ResearchSourceV1, ...]:
    merged = list(existing)
    by_id = {source.source_id: source for source in existing}
    for source in additions:
        previous = by_id.get(source.source_id)
        if previous is None:
            by_id[source.source_id] = source
            merged.append(source)
        elif previous != source:
            raise ResearchGraphProtocolError(
                category="source_identity_conflict",
                node_name="research_agent",
            )
    if len(merged) > 64:
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name="research_agent",
        )
    return tuple(merged)


def _merge_evidence(
    existing: tuple[ResearchEvidenceV1, ...],
    additions: tuple[ResearchEvidenceV1, ...],
) -> tuple[ResearchEvidenceV1, ...]:
    merged = list(existing)
    by_id = {item.evidence_id: item for item in existing}
    for item in additions:
        previous = by_id.get(item.evidence_id)
        if previous is None:
            by_id[item.evidence_id] = item
            merged.append(item)
        elif previous != item:
            raise ResearchGraphProtocolError(
                category="evidence_identity_conflict",
                node_name="research_agent",
            )
    if len(merged) > 128:
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name="research_agent",
        )
    return tuple(merged)


def _merge_search_calls(
    existing: tuple[SearchCallTraceV1, ...],
    additions: tuple[SearchCallTraceV1, ...],
    *,
    research_pass_number: int,
) -> tuple[SearchCallTraceV1, ...]:
    if tuple(call.call_ordinal for call in existing) != tuple(range(1, len(existing) + 1)):
        raise ResearchGraphProtocolError(
            category="search_trace_conflict",
            node_name="research_agent",
        )
    expected_first_ordinal = len(existing) + 1
    for offset, call in enumerate(additions):
        if (
            call.call_ordinal != expected_first_ordinal + offset
            or call.research_pass_number != research_pass_number
        ):
            raise ResearchGraphProtocolError(
                category="search_trace_conflict",
                node_name="research_agent",
            )
    merged = (*existing, *additions)
    if len(merged) > 8:
        raise ResearchGraphProtocolError(
            category="search_trace_conflict",
            node_name="research_agent",
        )
    return merged


def _merge_document_calls(
    existing: tuple[DocumentRetrievalCallTraceV1, ...],
    additions: tuple[DocumentRetrievalCallTraceV1, ...],
) -> tuple[DocumentRetrievalCallTraceV1, ...]:
    merged = (*existing, *additions)
    if len(merged) > 8 or tuple(item.call_ordinal for item in merged) != tuple(
        range(1, len(merged) + 1)
    ):
        raise ResearchGraphProtocolError(
            category="search_trace_conflict", node_name="research_agent"
        )
    return merged


def _require_evidence_sources(
    sources: tuple[ResearchSourceV1, ...],
    evidence: tuple[ResearchEvidenceV1, ...],
) -> None:
    source_ids = {source.source_id for source in sources}
    if any(item.source_id not in source_ids for item in evidence):
        raise ResearchGraphProtocolError(
            category="evidence_source_missing",
            node_name="research_agent",
        )


def _merge_limitations(
    first: tuple[ResearchLimitationV1, ...],
    second: tuple[ResearchLimitationV1, ...],
) -> tuple[ResearchLimitationV1, ...]:
    merged = list(first)
    for limitation in second:
        if limitation not in merged:
            merged.append(limitation)
    return tuple(merged)


def _has_valid_application_draft(
    state: ResearchGraphStateV1,
    writer_output: WriteReportNodeOutputV1,
) -> bool:
    draft = writer_output.application_draft
    if (
        state.mode != "application"
        or not state.request.include_application_draft
        or state.resume_document_id is None
        or state.evidence_sufficient is not True
        or draft is None
    ):
        return False
    evidence_sources = {
        item.evidence_id: item.source_id for item in (*state.evidence, *state.document_evidence)
    }
    return all(
        evidence_sources.get(citation.evidence_id) == citation.source_id
        for paragraph in draft.paragraphs
        for citation in paragraph.citations
    )


def _normalize_request(state: ResearchGraphJsonStateV1) -> dict[str, object]:
    try:
        graph_input = ResearchGraphInputV1.model_validate_json(
            json.dumps(
                state,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name="normalize_request",
        ) from None
    normalized_query = normalize_research_query(graph_input.request.query)
    try:
        normalized_request = ResearchRequestV1(
            query=normalized_query,
            include_application_draft=graph_input.request.include_application_draft,
        )
    except ValidationError:
        raise ResearchGraphProtocolError(
            category="invalid_node_output",
            node_name="normalize_request",
        ) from None
    return {
        "schema_version": graph_input.schema_version,
        "run_id": str(graph_input.run_id),
        "workspace_id": str(graph_input.workspace_id),
        "actor_user_id": str(graph_input.actor_user_id),
        "conversation_id": str(graph_input.conversation_id),
        "graph_version": graph_input.graph_version,
        "mode": graph_input.mode,
        "resume_document_id": (
            str(graph_input.resume_document_id)
            if graph_input.resume_document_id is not None
            else None
        ),
        "request": normalized_request.model_dump(mode="json", round_trip=True),
        "normalized_query": normalized_request.query,
    }


def _validate_research_pass_count(value: int, *, node_name: str) -> int:
    if type(value) is not int or value < 0 or value > 2:
        raise ResearchGraphProtocolError(
            category="invalid_research_pass_count",
            node_name=node_name,
        )
    return value


def build_research_state_graph(
    nodes: ResearchGraphNodes,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[
    ResearchGraphJsonStateV1,
    ResearchGraphRuntimeContext,
    ResearchGraphJsonInputV1,
    ResearchGraphOutputStateV1,
]:
    if not isinstance(nodes, ResearchGraphNodes):
        raise ValueError("nodes must use ResearchGraphNodes")

    async def plan(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="plan")
        normalized_query = _required_state_value(
            validated_state.normalized_query,
            node_name="plan",
        )
        output = await _run_node(
            nodes.plan(
                PlanNodeInputV1(
                    normalized_query=normalized_query,
                    include_application_draft=validated_state.request.include_application_draft,
                ),
                _runtime_context(runtime, node_name="plan").agent_loop_control,
            ),
            node_name="plan",
        )
        validated = _validated_node_output(PlanNodeOutputV1, output, node_name="plan")
        return {"plan": validated.plan.model_dump(mode="json", round_trip=True)}

    async def research_agent(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="research_agent")
        normalized_query = _required_state_value(
            validated_state.normalized_query,
            node_name="research_agent",
        )
        research_plan = _required_state_value(
            validated_state.plan,
            node_name="research_agent",
        )
        completed_passes = _validate_research_pass_count(
            validated_state.research_pass_count,
            node_name="research_agent",
        )
        next_pass = completed_passes + 1
        if next_pass > 2:
            raise ResearchGraphProtocolError(
                category="invalid_research_pass_count",
                node_name="research_agent",
            )

        output = await _run_node(
            nodes.research_agent(
                ResearchNodeInputV1(
                    request=validated_state.request,
                    normalized_query=normalized_query,
                    plan=research_plan,
                    existing_sources=validated_state.sources,
                    existing_evidence=validated_state.evidence,
                    existing_search_calls=validated_state.search_calls,
                    existing_document_sources=validated_state.document_sources,
                    existing_document_evidence=validated_state.document_evidence,
                    existing_document_retrieval_calls=validated_state.document_retrieval_calls,
                    research_pass_number=next_pass,
                    document_scope_available=validated_state.resume_document_id is not None,
                ),
                _runtime_context(runtime, node_name="research_agent"),
            ),
            node_name="research_agent",
        )
        validated = _validated_node_output(
            ResearchNodeOutputV1,
            output,
            node_name="research_agent",
        )
        hook = _runtime_context(runtime, node_name="research_agent").after_research_node
        if hook is not None:
            await hook(validated)
        merged_sources = _merge_sources(validated_state.sources, validated.sources)
        merged_evidence = _merge_evidence(validated_state.evidence, validated.evidence)
        merged_search_calls = _merge_search_calls(
            validated_state.search_calls,
            validated.search_calls,
            research_pass_number=next_pass,
        )
        _require_evidence_sources(merged_sources, merged_evidence)
        merged_document_sources = _merge_sources(  # type: ignore[arg-type]
            validated_state.document_sources, validated.document_sources
        )
        merged_document_evidence = _merge_evidence(  # type: ignore[arg-type]
            validated_state.document_evidence, validated.document_evidence
        )
        merged_document_calls = _merge_document_calls(
            validated_state.document_retrieval_calls,
            validated.document_retrieval_calls,
        )
        return {
            "sources": [item.model_dump(mode="json", round_trip=True) for item in merged_sources],
            "evidence": [item.model_dump(mode="json", round_trip=True) for item in merged_evidence],
            "search_calls": [
                item.model_dump(mode="json", round_trip=True) for item in merged_search_calls
            ],
            "document_sources": [
                item.model_dump(mode="json", round_trip=True) for item in merged_document_sources
            ],
            "document_evidence": [
                item.model_dump(mode="json", round_trip=True) for item in merged_document_evidence
            ],
            "document_retrieval_calls": [
                item.model_dump(mode="json", round_trip=True) for item in merged_document_calls
            ],
            "research_pass_count": next_pass,
            "evidence_sufficient": None,
            "validation_limitations": [],
            "writer_output": None,
            "action_proposal_id": None,
            "action_key": None,
            "action_revision": None,
            "approval_expires_at": None,
            "approval_request_id": None,
            "output": None,
        }

    async def validate_evidence(state: ResearchGraphJsonStateV1) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="validate_evidence")
        research_plan = _required_state_value(
            validated_state.plan,
            node_name="validate_evidence",
        )
        completed_passes = _validate_research_pass_count(
            validated_state.research_pass_count,
            node_name="validate_evidence",
        )
        if completed_passes == 0:
            raise ResearchGraphProtocolError(
                category="invalid_research_pass_count",
                node_name="validate_evidence",
            )
        output = await _run_node(
            nodes.validate_evidence(
                EvidenceValidationNodeInputV1(
                    request=validated_state.request,
                    plan=research_plan,
                    sources=validated_state.sources,
                    evidence=validated_state.evidence,
                    document_evidence=validated_state.document_evidence,
                    research_pass_count=completed_passes,
                )
            ),
            node_name="validate_evidence",
        )
        validated = _validated_node_output(
            EvidenceValidationNodeOutputV1,
            output,
            node_name="validate_evidence",
        )
        return {
            "evidence_sufficient": validated.evidence_sufficient,
            "validation_limitations": [
                item.model_dump(mode="json", round_trip=True) for item in validated.limitations
            ],
        }

    def route_after_validation(state: ResearchGraphJsonStateV1) -> ResearchRoute:
        validated_state = _validated_graph_state(state, node_name="validate_evidence")
        completed_passes = _validate_research_pass_count(
            validated_state.research_pass_count,
            node_name="validate_evidence",
        )
        if validated_state.evidence_sufficient is True:
            return "write_report"
        if validated_state.evidence_sufficient is False:
            if completed_passes == 1:
                return "research_again"
            if completed_passes == 2:
                return "write_report"
        raise ResearchGraphProtocolError(
            category="invalid_research_pass_count",
            node_name="validate_evidence",
        )

    async def write_report(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="write_report")
        evidence_sufficient = _required_state_value(
            validated_state.evidence_sufficient,
            node_name="write_report",
        )
        output = await _run_node(
            nodes.write_report(
                WriteReportNodeInputV1(
                    request=validated_state.request,
                    evidence=validated_state.evidence,
                    document_evidence=validated_state.document_evidence,
                    evidence_sufficient=evidence_sufficient,
                    validation_limitations=validated_state.validation_limitations,
                ),
                _runtime_context(runtime, node_name="write_report").agent_loop_control,
            ),
            node_name="write_report",
        )
        validated = _validated_node_output(
            WriteReportNodeOutputV1,
            output,
            node_name="write_report",
        )
        proposal_id = validated_state.action_proposal_id
        action_key = validated_state.action_key
        action_revision = validated_state.action_revision
        approval_expires_at = validated_state.approval_expires_at
        context = _runtime_context(runtime, node_name="write_report")
        if _has_valid_application_draft(validated_state, validated) and context.tenant is not None:
            if proposal_id is None:
                proposal_id = uuid4()
                action_key = ACTION_KEY
                action_revision = INITIAL_ACTION_REVISION
                approval_expires_at = context.clock() + APPROVAL_TTL
        else:
            proposal_id = None
            action_key = None
            action_revision = None
            approval_expires_at = None
        return {
            "writer_output": validated.model_dump(mode="json", round_trip=True),
            "action_proposal_id": str(proposal_id) if proposal_id is not None else None,
            "action_key": action_key,
            "action_revision": action_revision,
            "approval_expires_at": (
                approval_expires_at.isoformat() if approval_expires_at is not None else None
            ),
            "approval_request_id": None,
        }

    def route_after_write(state: ResearchGraphJsonStateV1) -> ActionRoute:
        validated_state = _validated_graph_state(state, node_name="write_report")
        return "prepare_action" if validated_state.action_proposal_id is not None else "finalize"

    def observe_approval(
        name: str, request: ApprovalRequestRecord, *, decision: str | None = None
    ) -> None:
        scope = current_trace_scope()
        if scope is None:
            return
        metadata: dict[str, TraceMetadataValue] = {
            "approval_request_id": request.request_id,
            "action_intent_id": request.action_intent_id,
            "binding_version": request.approval_binding_version,
        }
        if decision is not None:
            metadata["decision_type"] = decision
        started_at = monotonic()
        observation = start_trace_span(
            scope, span_kind="approval_event", name=name, metadata=metadata
        )
        finish_trace_span(scope, observation, started_at=started_at, status="succeeded")

    async def prepare_action(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="prepare_action")
        proposal_id = _required_state_value(
            validated_state.action_proposal_id, node_name="prepare_action"
        )
        action_key = _required_state_value(validated_state.action_key, node_name="prepare_action")
        action_revision = _required_state_value(
            validated_state.action_revision, node_name="prepare_action"
        )
        expires_at = _required_state_value(
            validated_state.approval_expires_at, node_name="prepare_action"
        )
        resume_document_id = _required_state_value(
            validated_state.resume_document_id, node_name="prepare_action"
        )
        writer_output = _required_state_value(
            validated_state.writer_output, node_name="prepare_action"
        )
        context = _runtime_context(runtime, node_name="prepare_action")
        if context.action_store is None or context.tenant is None:
            raise ResearchGraphProtocolError(
                category="missing_graph_context", node_name="prepare_action"
            )
        prepared = await _run_node(
            context.action_store.prepare_action(
                PrepareActionCommand(
                    tenant=context.tenant,
                    run_id=validated_state.run_id,
                    action_proposal_id=proposal_id,
                    action_key=action_key,
                    action_revision=action_revision,
                    args=assemble_submit_application_args(
                        run_id=validated_state.run_id,
                        resume_document_id=resume_document_id,
                        writer_output=writer_output,
                    ),
                    now=expires_at - APPROVAL_TTL,
                    expires_at=expires_at,
                )
            ),
            node_name="prepare_action",
        )
        if (
            prepared.intent.action_intent_id != proposal_id
            or prepared.intent.workspace_id != validated_state.workspace_id
            or prepared.intent.run_id != validated_state.run_id
            or prepared.intent.action_key != action_key
            or prepared.intent.action_revision != action_revision
            or prepared.approval_request.workspace_id != validated_state.workspace_id
            or prepared.approval_request.run_id != validated_state.run_id
            or prepared.approval_request.action_intent_id != proposal_id
            or prepared.approval_request.expires_at != expires_at
        ):
            raise ResearchGraphProtocolError(
                category="action_identity_mismatch", node_name="prepare_action"
            )
        observe_approval("approval.request_created", prepared.approval_request)
        return {"approval_request_id": str(prepared.approval_request.request_id)}

    async def approval_interrupt(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> Command[ApprovalRoute]:
        validated_state = _validated_graph_state(state, node_name="approval_interrupt")
        proposal_id = _required_state_value(
            validated_state.action_proposal_id, node_name="approval_interrupt"
        )
        request_id = _required_state_value(
            validated_state.approval_request_id, node_name="approval_interrupt"
        )
        action_key = _required_state_value(
            validated_state.action_key, node_name="approval_interrupt"
        )
        action_revision = _required_state_value(
            validated_state.action_revision, node_name="approval_interrupt"
        )
        payload = {
            "version": 1,
            "approval_request_id": str(request_id),
            "action_intent_id": str(proposal_id),
            "action_key": action_key,
            "action_revision": action_revision,
        }
        while True:
            resumed = interrupt(payload)
            if not isinstance(resumed, dict) or resumed.get("approval_request_id") != str(
                request_id
            ):
                raise ResearchGraphProtocolError(
                    category="invalid_approval_resume", node_name="approval_interrupt"
                )
            context = _runtime_context(runtime, node_name="approval_interrupt")
            if context.approval_resume_resolver is None or context.tenant is None:
                raise ResearchGraphProtocolError(
                    category="missing_graph_context", node_name="approval_interrupt"
                )
            resolved = await _run_node(
                context.approval_resume_resolver.resolve_approval_resume(
                    tenant=context.tenant,
                    run_id=validated_state.run_id,
                    action_intent_id=proposal_id,
                    approval_request_id=request_id,
                ),
                node_name="approval_interrupt",
            )
            if resolved.action_intent_id != proposal_id:
                raise ResearchGraphProtocolError(
                    category="action_identity_mismatch", node_name="approval_interrupt"
                )
            status = resolved.approval_request.status
            if status is ApprovalStatus.PENDING:
                continue
            if status in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
                if resolved.decision is not None:
                    observe_approval(
                        "approval.decision_recorded",
                        resolved.approval_request,
                        decision=resolved.decision,
                    )
                if status is ApprovalStatus.EXPIRED:
                    observe_approval("approval.expired", resolved.approval_request)
                observe_approval("approval.resume_consumed", resolved.approval_request)
            if status is ApprovalStatus.APPROVED:
                return Command(goto="execute_mock_action")
            if status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
                return Command(goto="cancel_action")
            raise ResearchGraphProtocolError(
                category="invalid_approval_resume", node_name="approval_interrupt"
            )

    async def execute_mock_action(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="execute_mock_action")
        proposal_id = _required_state_value(
            validated_state.action_proposal_id, node_name="execute_mock_action"
        )
        request_id = _required_state_value(
            validated_state.approval_request_id, node_name="execute_mock_action"
        )
        context = _runtime_context(runtime, node_name="execute_mock_action")
        if (
            context.approved_action_executor is None
            or context.action_store is None
            or context.tenant is None
        ):
            raise ResearchGraphProtocolError(
                category="missing_graph_context", node_name="execute_mock_action"
            )
        try:
            await context.approved_action_executor.execute_approved_action(
                ActionExecutionIdentity(
                    workspace_id=validated_state.workspace_id,
                    run_id=validated_state.run_id,
                    action_intent_id=proposal_id,
                    approval_request_id=request_id,
                ),
                deadline=context.agent_loop_control.deadline,
                cancellation=context.agent_loop_control.cancellation,
            )
        except ActionApprovalExpiredError:
            await _run_node(
                context.action_store.cancel_action(
                    CancelActionCommand(
                        tenant=context.tenant,
                        run_id=validated_state.run_id,
                        action_intent_id=proposal_id,
                        approval_request_id=request_id,
                        reason="approval_expired",
                        now=context.clock(),
                    )
                ),
                node_name="execute_mock_action",
            )
        return {}

    async def cancel_action(
        state: ResearchGraphJsonStateV1,
        runtime: Runtime[ResearchGraphRuntimeContext],
    ) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="cancel_action")
        proposal_id = _required_state_value(
            validated_state.action_proposal_id, node_name="cancel_action"
        )
        request_id = _required_state_value(
            validated_state.approval_request_id, node_name="cancel_action"
        )
        context = _runtime_context(runtime, node_name="cancel_action")
        if (
            context.action_store is None
            or context.approval_resume_resolver is None
            or context.tenant is None
        ):
            raise ResearchGraphProtocolError(
                category="missing_graph_context", node_name="cancel_action"
            )
        resolved = await _run_node(
            context.approval_resume_resolver.resolve_approval_resume(
                tenant=context.tenant,
                run_id=validated_state.run_id,
                action_intent_id=proposal_id,
                approval_request_id=request_id,
            ),
            node_name="cancel_action",
        )
        if resolved.approval_request.status is ApprovalStatus.REJECTED:
            reason = "approval_rejected"
        elif resolved.approval_request.status is ApprovalStatus.EXPIRED:
            reason = "approval_expired"
        else:
            raise ResearchGraphProtocolError(
                category="invalid_approval_resume", node_name="cancel_action"
            )
        await _run_node(
            context.action_store.cancel_action(
                CancelActionCommand(
                    tenant=context.tenant,
                    run_id=validated_state.run_id,
                    action_intent_id=proposal_id,
                    approval_request_id=request_id,
                    reason=reason,
                    now=context.clock(),
                )
            ),
            node_name="cancel_action",
        )
        return {}

    def finalize(state: ResearchGraphJsonStateV1) -> dict[str, object]:
        validated_state = _validated_graph_state(state, node_name="finalize")
        evidence_sufficient = _required_state_value(
            validated_state.evidence_sufficient,
            node_name="finalize",
        )
        writer_output = _required_state_value(
            validated_state.writer_output,
            node_name="finalize",
        )
        if (
            not validated_state.request.include_application_draft
            and writer_output.application_draft is not None
        ):
            raise ResearchGraphProtocolError(
                category="unexpected_application_draft",
                node_name="finalize",
            )

        try:
            if validated_state.schema_version == 1:
                legacy_output = ResearchOutputV1(
                    evidence_sufficient=evidence_sufficient,
                    summary=writer_output.summary,
                    findings=writer_output.findings,
                    evidence=validated_state.evidence,
                    limitations=_merge_limitations(
                        validated_state.validation_limitations,
                        writer_output.limitations,
                    ),
                    sources=validated_state.sources,
                    application_draft=writer_output.application_draft,
                )
                return {"output": legacy_output.model_dump(mode="json", round_trip=True)}
            web_sources = tuple(
                ResearchSourceV2(
                    source_type="web",
                    source_id=item.source_id,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    published_at=item.published_at,
                )
                for item in validated_state.sources
            )
            web_evidence = tuple(
                ResearchEvidenceV2(
                    source_type="web",
                    evidence_id=item.evidence_id,
                    source_id=item.source_id,
                    text=item.text,
                )
                for item in validated_state.evidence
            )
            evidence_by_id = {
                item.evidence_id: item
                for item in (*web_evidence, *validated_state.document_evidence)
            }

            def convert_claim(item: object) -> ResearchClaimV2:
                claim = item
                return ResearchClaimV2(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    citations=tuple(
                        ResearchCitationV2(
                            source_type=evidence_by_id[citation.evidence_id].source_type,
                            source_id=citation.source_id,
                            evidence_id=citation.evidence_id,
                        )
                        for citation in claim.citations
                    ),
                )

            application_draft = (
                ApplicationDraftV2(
                    paragraphs=tuple(
                        convert_claim(item) for item in writer_output.application_draft.paragraphs
                    )
                )
                if writer_output.application_draft is not None
                else None
            )
            output = ResearchOutputV2(
                evidence_sufficient=evidence_sufficient,
                summary=tuple(convert_claim(item) for item in writer_output.summary),
                findings=tuple(convert_claim(item) for item in writer_output.findings),
                evidence=(*web_evidence, *validated_state.document_evidence),
                limitations=_merge_limitations(
                    validated_state.validation_limitations,
                    writer_output.limitations,
                ),
                sources=(*web_sources, *validated_state.document_sources),
                application_draft=application_draft,
            )
        except (KeyError, ValidationError):
            raise ResearchGraphProtocolError(
                category="invalid_final_output",
                node_name="finalize",
            ) from None
        return {"output": output.model_dump(mode="json", round_trip=True)}

    node_occurrences: dict[str, int] = {}

    def traced_node(node_name: str, node: Callable, *, uses_runtime: bool = True) -> Callable:
        async def traced(
            state: ResearchGraphJsonStateV1,
            runtime: Runtime[ResearchGraphRuntimeContext],
        ):
            scope = current_trace_scope()
            started_at = monotonic()
            context = None
            if scope is not None:
                node_occurrences[node_name] = node_occurrences.get(node_name, 0) + 1
                context = start_trace_span(
                    scope,
                    span_kind="graph_node",
                    metadata={
                        "graph_version": CURRENT_GRAPH_VERSION,
                        "node_name": node_name,
                        "occurrence_ordinal": node_occurrences[node_name],
                    },
                )
            status: SpanStatus = "succeeded"
            error_category = None
            with bind_trace_scope(replace(scope, parent=context) if scope is not None else None):
                try:
                    result = node(state, runtime) if uses_runtime else node(state)
                    return await result if inspect.isawaitable(result) else result
                except GraphBubbleUp:
                    # LangGraph interrupt/control flow is an expected bounded suspension.
                    raise
                except asyncio.CancelledError:
                    status, error_category = "cancelled", "node_cancelled"
                    raise
                except ResearchGraphProtocolError as error:
                    status = "failed"
                    error_category = (
                        error.category
                        if error.category in get_args(ResearchGraphErrorCategory)
                        else "node_execution_failed"
                    )
                    raise
                except Exception:
                    status, error_category = "failed", "node_execution_failed"
                    raise
                finally:
                    if scope is not None:
                        finish_trace_span(
                            scope,
                            context,
                            started_at=started_at,
                            status=status,
                            error_category=error_category,
                        )

        return traced

    graph = StateGraph(
        ResearchGraphJsonStateV1,
        context_schema=ResearchGraphRuntimeContext,
        input_schema=ResearchGraphJsonInputV1,
        output_schema=ResearchGraphOutputStateV1,
    )
    graph.add_node(
        "normalize_request",
        traced_node("normalize_request", _normalize_request, uses_runtime=False),
    )
    graph.add_node("plan", traced_node("plan", plan))
    graph.add_node("research_agent", traced_node("research_agent", research_agent))
    graph.add_node(
        "validate_evidence", traced_node("validate_evidence", validate_evidence, uses_runtime=False)
    )
    graph.add_node("write_report", traced_node("write_report", write_report))
    graph.add_node("prepare_action", traced_node("prepare_action", prepare_action))
    graph.add_node(
        "approval_interrupt",
        traced_node("approval_interrupt", approval_interrupt),
        destinations=("execute_mock_action", "cancel_action"),
    )
    graph.add_node("execute_mock_action", traced_node("execute_mock_action", execute_mock_action))
    graph.add_node("cancel_action", traced_node("cancel_action", cancel_action))
    graph.add_node("finalize", traced_node("finalize", finalize, uses_runtime=False))
    graph.add_edge(START, "normalize_request")
    graph.add_edge("normalize_request", "plan")
    graph.add_edge("plan", "research_agent")
    graph.add_edge("research_agent", "validate_evidence")
    graph.add_conditional_edges(
        "validate_evidence",
        route_after_validation,
        {
            "research_again": "research_agent",
            "write_report": "write_report",
        },
    )
    graph.add_conditional_edges(
        "write_report",
        route_after_write,
        {"finalize": "finalize", "prepare_action": "prepare_action"},
    )
    graph.add_edge("prepare_action", "approval_interrupt")
    graph.add_edge("execute_mock_action", "finalize")
    graph.add_edge("cancel_action", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(
        checkpointer=checkpointer,
        store=None,
        name=CURRENT_GRAPH_VERSION,
    )
