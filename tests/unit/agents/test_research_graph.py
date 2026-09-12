from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.contracts import AgentLoopControl, AgentLoopLimitsV1
from app.agents.research_contracts import (
    ApplicationDraftV1,
    EvidenceValidationNodeInputV1,
    EvidenceValidationNodeOutputV1,
    PlanNodeInputV1,
    PlanNodeOutputV1,
    ResearchCitationV1,
    ResearchClaimV1,
    ResearchEvidenceV2,
    ResearchGraphInputV1,
    ResearchGraphOutputStateV1,
    ResearchGraphStateV1,
    ResearchLimitationV1,
    ResearchNodeInputV1,
    ResearchNodeOutputV1,
    ResearchOutputV1,
    ResearchOutputV2,
    ResearchPlanV1,
    ResearchRequestV1,
    ResearchSourceV1,
    ResearchSourceV2,
    WriteReportNodeInputV1,
    WriteReportNodeOutputV1,
)
from app.agents.research_graph import (
    APPROVAL_TTL,
    ResearchGraphNodes,
    ResearchGraphProtocolError,
    ResearchGraphRuntimeContext,
    _normalize_request,
    assemble_submit_application_args,
    build_approval_resume_input,
    build_research_state_graph,
)
from app.agents.research_nodes import ResearchAgentNodeError, ResearchWriterNodeError
from app.domain.approvals import ApprovalStatus
from app.domain.provisioning import WorkspaceRole
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantContext
from app.llm.ports import ModelToolCall, ModelToolSchema

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "agents" / "research_cases_v1.json"
)


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class _UnusedBoundToolRuntime:
    secret_canary = "trusted-runtime-canary"

    def model_tools(self) -> tuple[ModelToolSchema, ...]:
        return ()

    def validate_call(self, call: ModelToolCall) -> None:
        raise AssertionError(f"Step 2.1 must not validate tools: {call.name}")

    async def execute(self, call: ModelToolCall) -> str:
        raise AssertionError(f"Step 2.1 must not execute tools: {call.name}")


class _ActionStore:
    def __init__(self) -> None:
        self.commands = []
        self.request_ids: dict[UUID, UUID] = {}

    async def prepare_action(self, command):
        self.commands.append(command)
        request_id = self.request_ids.setdefault(command.action_proposal_id, uuid4())
        return SimpleNamespace(
            intent=SimpleNamespace(
                action_intent_id=command.action_proposal_id,
                workspace_id=command.tenant.workspace_id,
                run_id=command.run_id,
                action_key=command.action_key,
                action_revision=command.action_revision,
            ),
            approval_request=SimpleNamespace(
                request_id=request_id,
                workspace_id=command.tenant.workspace_id,
                run_id=command.run_id,
                action_intent_id=command.action_proposal_id,
                expires_at=command.expires_at,
                approval_binding_version=1,
            ),
        )


class _ApprovalResumeResolver:
    async def resolve_approval_resume(self, **kwargs):
        return SimpleNamespace(
            action_intent_id=kwargs["action_intent_id"],
            approval_request=SimpleNamespace(status=ApprovalStatus.PENDING),
        )


def _runtime_context() -> ResearchGraphRuntimeContext:
    fixed_now = datetime(2026, 8, 24, 12, tzinfo=UTC)
    return ResearchGraphRuntimeContext(
        tool_runtime=_UnusedBoundToolRuntime(),
        agent_loop_control=AgentLoopControl(
            limits=AgentLoopLimitsV1(
                max_model_calls=3,
                max_tool_calls=2,
                max_tool_results=2,
                max_iterations=5,
            ),
            deadline=monotonic() + 60.0,
            cancellation=_NeverCancelled(),
        ),
        tenant=TenantContext(
            workspace_id=UUID("00000000-0000-0000-0000-000000000002"),
            actor_user_id=UUID("00000000-0000-0000-0000-000000000003"),
            role=WorkspaceRole.ADMIN,
        ),
        action_store=_ActionStore(),
        approval_resume_resolver=_ApprovalResumeResolver(),
        clock=lambda: fixed_now,
    )


def _cases() -> dict[str, dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    return {case["case_id"]: case for case in payload["cases"]}


def _request(case: dict[str, object]) -> ResearchRequestV1:
    return ResearchRequestV1.model_validate(case["request"], strict=True)


def _scripted_pass(case: dict[str, object], pass_number: int) -> ResearchNodeOutputV1:
    passes = cast(list[dict[str, object]], case["passes"])
    return ResearchNodeOutputV1.model_validate_json(json.dumps(passes[pass_number - 1]))


class _CaseNodes:
    def __init__(
        self, case: dict[str, object], runtime_context: ResearchGraphRuntimeContext
    ) -> None:
        self.case = case
        self.runtime_context = runtime_context
        self.events: list[str] = []
        self.planned_queries: list[str] = []

    async def plan(
        self,
        node_input: PlanNodeInputV1,
        control: AgentLoopControl,
    ) -> PlanNodeOutputV1:
        assert control is self.runtime_context.agent_loop_control
        self.events.append("plan")
        self.planned_queries.append(node_input.normalized_query)
        return PlanNodeOutputV1(
            plan=ResearchPlanV1(queries=(node_input.normalized_query,)),
        )

    async def research_agent(
        self,
        node_input: ResearchNodeInputV1,
        runtime_context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1:
        assert runtime_context is self.runtime_context
        assert node_input.request.query == node_input.normalized_query
        self.events.append(f"research:{node_input.research_pass_number}")
        return _scripted_pass(self.case, node_input.research_pass_number)

    async def validate_evidence(
        self,
        node_input: EvidenceValidationNodeInputV1,
    ) -> EvidenceValidationNodeOutputV1:
        self.events.append(f"validate:{node_input.research_pass_count}")
        validation = cast(list[bool], self.case["validation"])
        evidence_sufficient = validation[node_input.research_pass_count - 1]
        limitations: list[ResearchLimitationV1] = []
        if not evidence_sufficient:
            limitations.append(
                ResearchLimitationV1(
                    code="insufficient_evidence",
                    detail="The current bounded research pass has insufficient evidence.",
                )
            )
        if self.case["case_id"] == "conflicting_sources":
            limitations.append(
                ResearchLimitationV1(
                    code="conflicting_evidence",
                    detail="Recorded sources disagree about the remote-work policy.",
                )
            )
        return EvidenceValidationNodeOutputV1(
            evidence_sufficient=evidence_sufficient,
            limitations=tuple(limitations),
        )

    async def write_report(
        self,
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        assert control is self.runtime_context.agent_loop_control
        assert not hasattr(node_input, "sources")
        self.events.append("write")
        if not node_input.evidence_sufficient:
            return WriteReportNodeOutputV1(
                limitations=node_input.validation_limitations,
            )

        citations = tuple(
            ResearchCitationV1(
                source_id=item.source_id,
                evidence_id=item.evidence_id,
            )
            for item in node_input.evidence
        )
        report_claim = ResearchClaimV1(
            claim_id="C1",
            text="The recorded evidence supports a bounded synthetic finding.",
            citations=citations,
        )
        draft = None
        if node_input.request.include_application_draft:
            draft = ApplicationDraftV1(
                paragraphs=(
                    ResearchClaimV1(
                        claim_id="D1",
                        text="My background aligns with the cited synthetic requirement.",
                        citations=(citations[0],),
                    ),
                )
            )
        return WriteReportNodeOutputV1(
            summary=(report_claim,),
            limitations=node_input.validation_limitations,
            application_draft=draft,
        )

    def bundle(self) -> ResearchGraphNodes:
        return ResearchGraphNodes(
            plan=self.plan,
            research_agent=self.research_agent,
            validate_evidence=self.validate_evidence,
            write_report=self.write_report,
        )


async def _invoke_case(
    case_id: str,
) -> tuple[ResearchOutputV1, _CaseNodes, object]:
    case = _cases()[case_id]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())
    raw_output = await graph.ainvoke(
        _checkpoint_input(_request(case)),
        context=runtime_context,
    )
    validated_state = ResearchGraphOutputStateV1.model_validate_json(
        json.dumps(raw_output, allow_nan=False), strict=True
    )
    return validated_state.output, nodes, graph


def _checkpoint_input(
    request: ResearchRequestV1,
    *,
    mode: str = "research",
    resume_document_id: object = None,
) -> dict[str, object]:
    return ResearchGraphInputV1(
        request=request,
        mode=mode,
        resume_document_id=resume_document_id,
    ).model_dump(mode="json", round_trip=True)


@pytest.mark.parametrize("include_application_draft", [False, True])
def test_normalize_request_is_idempotent(include_application_draft: bool) -> None:
    graph_input = _checkpoint_input(
        ResearchRequestV1(
            query=" \uff33\uff45\uff4e\uff49\uff4f\uff52  Python\tEngineer \n",
            include_application_draft=include_application_draft,
        )
    )
    first = _normalize_request(graph_input)
    second = _normalize_request({**graph_input, "request": first["request"]})

    assert second == first
    assert first["normalized_query"] == "Senior Python Engineer"
    assert first["request"] == {
        "schema_version": 1,
        "query": "Senior Python Engineer",
        "include_application_draft": include_application_draft,
    }


def test_fixture_covers_required_step_2_1_cases() -> None:
    assert set(_cases()) == {
        "normal",
        "insufficient",
        "duplicate_source",
        "conflicting_sources",
        "prompt_injection",
    }


def test_oversized_deterministic_cover_letter_fails_closed_without_truncation() -> None:
    citation = ResearchCitationV1(source_id="S1", evidence_id="E1")
    writer_output = WriteReportNodeOutputV1(
        application_draft=ApplicationDraftV1(
            paragraphs=tuple(
                ResearchClaimV1(
                    claim_id=f"D{index}",
                    text="x" * 4_000,
                    citations=(citation,),
                )
                for index in range(6)
            )
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as raised:
        assemble_submit_application_args(
            run_id=uuid4(),
            resume_document_id=uuid4(),
            writer_output=writer_output,
        )

    assert raised.value.category == "invalid_action_payload"


def test_compiled_state_graph_has_only_the_fixed_topology() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    compiled = build_research_state_graph(nodes.bundle())
    drawable = compiled.get_graph()

    assert set(drawable.nodes) == {
        "__start__",
        "normalize_request",
        "plan",
        "research_agent",
        "validate_evidence",
        "write_report",
        "prepare_action",
        "approval_interrupt",
        "execute_mock_action",
        "cancel_action",
        "finalize",
        "__end__",
    }
    assert {(edge.source, edge.target, edge.conditional) for edge in drawable.edges} == {
        ("__start__", "normalize_request", False),
        ("normalize_request", "plan", False),
        ("plan", "research_agent", False),
        ("research_agent", "validate_evidence", False),
        ("validate_evidence", "research_agent", True),
        ("validate_evidence", "write_report", True),
        ("write_report", "finalize", True),
        ("write_report", "prepare_action", True),
        ("prepare_action", "approval_interrupt", False),
        ("approval_interrupt", "execute_mock_action", True),
        ("approval_interrupt", "cancel_action", True),
        ("execute_mock_action", "finalize", False),
        ("cancel_action", "finalize", False),
        ("finalize", "__end__", False),
    }


@pytest.mark.asyncio
async def test_normal_path_executes_real_graph_and_returns_resolvable_output() -> None:
    output, nodes, _graph = await _invoke_case("normal")

    assert nodes.events == ["plan", "research:1", "validate:1", "write"]
    assert nodes.planned_queries == ["Backend engineer roles"]
    assert output.evidence_sufficient is True
    assert output.sources[0].source_id == "S1"
    assert output.summary[0].citations[0].evidence_id == "E1"
    assert output.application_draft is not None
    assert ResearchOutputV1.model_validate_json(output.model_dump_json()) == output


@pytest.mark.asyncio
async def test_v2_finalizer_derives_document_source_type_from_trusted_evidence() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    web_pass = _scripted_pass(case, 1)
    document_id = UUID("00000000-0000-0000-0000-000000000010")
    chunk_id = UUID("00000000-0000-0000-0000-000000000011")
    document_source = ResearchSourceV2(
        source_type="workspace_document",
        source_id=f"workspace-document-v1:{document_id}",
        title="Synthetic resume",
        document_id=document_id,
        source_name="resume.md",
    )
    document_evidence = ResearchEvidenceV2(
        source_type="workspace_document",
        source_id=document_source.source_id,
        evidence_id=f"workspace-chunk-v1:{chunk_id}",
        text="The synthetic resume documents Python delivery experience.",
        document_id=document_id,
        chunk_id=chunk_id,
        section="Experience",
        ordinal=0,
    )

    async def mixed_research(
        node_input: ResearchNodeInputV1,
        received_context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1:
        assert node_input.research_pass_number == 1
        assert received_context is runtime_context
        return ResearchNodeOutputV1(
            sources=web_pass.sources,
            evidence=web_pass.evidence,
            document_sources=(document_source,),
            document_evidence=(document_evidence,),
        )

    async def sufficient_validation(
        node_input: EvidenceValidationNodeInputV1,
    ) -> EvidenceValidationNodeOutputV1:
        assert node_input.document_evidence == (document_evidence,)
        return EvidenceValidationNodeOutputV1(evidence_sufficient=True)

    async def mixed_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        assert control is runtime_context.agent_loop_control
        web_evidence = node_input.evidence[0]
        return WriteReportNodeOutputV1(
            summary=(
                ResearchClaimV1(
                    claim_id="web-claim",
                    text="The synthetic role has a cited requirement.",
                    citations=(
                        ResearchCitationV1(
                            source_id=web_evidence.source_id,
                            evidence_id=web_evidence.evidence_id,
                        ),
                    ),
                ),
            ),
            findings=(
                ResearchClaimV1(
                    claim_id="document-claim",
                    text="The synthetic resume has cited relevant experience.",
                    citations=(
                        ResearchCitationV1(
                            source_id=document_evidence.source_id,
                            evidence_id=document_evidence.evidence_id,
                        ),
                    ),
                ),
            ),
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=mixed_research,
            validate_evidence=sufficient_validation,
            write_report=mixed_writer,
        )
    )
    graph_input = ResearchGraphInputV1(
        schema_version=3,
        request=ResearchRequestV1(query="Mixed evidence"),
    ).model_dump(mode="json", round_trip=True)

    raw_output = await graph.ainvoke(graph_input, context=runtime_context)
    output = ResearchGraphOutputStateV1.model_validate_json(
        json.dumps(raw_output, allow_nan=False), strict=True
    ).output

    assert isinstance(output, ResearchOutputV2)
    document_citation = output.findings[0].citations[0]
    assert document_citation.source_type == "workspace_document"
    assert document_citation.source_id == document_evidence.source_id
    assert document_citation.evidence_id == document_evidence.evidence_id


@pytest.mark.asyncio
async def test_research_mode_never_gets_an_action_proposal_identity() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())

    raw_output = await graph.ainvoke(
        _checkpoint_input(_request(case), mode="research"),
        context=runtime_context,
    )
    output_state = ResearchGraphOutputStateV1.model_validate_json(
        json.dumps(raw_output, allow_nan=False), strict=True
    )

    assert output_state.output.application_draft is not None
    assert output_state.action_proposal_id is None
    assert output_state.action_key is None
    assert output_state.action_revision is None


@pytest.mark.asyncio
async def test_application_draft_gets_code_generated_stable_proposal_identity() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    checkpointer = InMemorySaver()
    graph = build_research_state_graph(nodes.bundle(), checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}
    resume_document_id = uuid4()

    raw_output = await graph.ainvoke(
        _checkpoint_input(
            _request(case),
            mode="application",
            resume_document_id=resume_document_id,
        ),
        config=config,
        context=runtime_context,
        durability="sync",
    )
    snapshot = await graph.aget_state(config)
    output_state = ResearchGraphStateV1.model_validate_json(
        json.dumps(snapshot.values, allow_nan=False), strict=True
    )

    assert raw_output["__interrupt__"][0].value["approval_request_id"] == str(
        output_state.approval_request_id
    )
    assert output_state.writer_output.application_draft is not None
    assert output_state.action_proposal_id is not None
    assert output_state.action_key == "submit_application"
    assert output_state.action_revision == 1
    assert output_state.approval_expires_at == runtime_context.clock() + APPROVAL_TTL
    command = runtime_context.action_store.commands[0]
    assert command.args.job_ref == f"pathfinder-mock-job-v1:{output_state.run_id}"
    assert command.args.resume_document_id == resume_document_id
    assert command.args.answers == {}
    assert command.args.cover_letter == (
        "My background aligns with the cited synthetic requirement."
    )


@pytest.mark.asyncio
async def test_matching_command_resume_validates_identity_and_reinterrupts_same_action() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    checkpointer = InMemorySaver()
    graph = build_research_state_graph(nodes.bundle(), checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}

    first = await graph.ainvoke(
        _checkpoint_input(
            _request(case),
            mode="application",
            resume_document_id=uuid4(),
        ),
        config=config,
        context=runtime_context,
        durability="sync",
    )
    first_payload = first["__interrupt__"][0].value
    request_id = UUID(first_payload["approval_request_id"])
    before = await graph.aget_state(config)

    second = await graph.ainvoke(
        build_approval_resume_input(request_id),
        config=config,
        context=runtime_context,
        durability="sync",
    )
    after = await graph.aget_state(config)
    second_payload = second["__interrupt__"][0].value

    assert second_payload == first_payload
    assert after.values["action_proposal_id"] == before.values["action_proposal_id"]
    assert after.values["approval_expires_at"] == before.values["approval_expires_at"]
    assert nodes.events.count("write") == 1
    assert len(runtime_context.action_store.commands) == 1


@pytest.mark.asyncio
async def test_mismatched_command_resume_fails_inside_approval_interrupt() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}

    await graph.ainvoke(
        _checkpoint_input(
            _request(case),
            mode="application",
            resume_document_id=uuid4(),
        ),
        config=config,
        context=runtime_context,
        durability="sync",
    )
    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            build_approval_resume_input(uuid4()),
            config=config,
            context=runtime_context,
            durability="sync",
        )

    assert captured.value.category == "invalid_approval_resume"
    assert captured.value.node_name == "approval_interrupt"
    assert nodes.events.count("write") == 1
    assert len(runtime_context.action_store.commands) == 1


@pytest.mark.asyncio
async def test_insufficient_application_has_no_action_proposal_identity() -> None:
    case = _cases()["insufficient"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())

    raw_output = await graph.ainvoke(
        _checkpoint_input(
            _request(case),
            mode="application",
            resume_document_id=uuid4(),
        ),
        context=runtime_context,
    )
    output_state = ResearchGraphOutputStateV1.model_validate_json(
        json.dumps(raw_output, allow_nan=False), strict=True
    )

    assert output_state.output.application_draft is None
    assert output_state.action_proposal_id is None
    assert output_state.action_key is None
    assert output_state.action_revision is None


@pytest.mark.asyncio
async def test_completed_checkpoint_preserves_the_same_proposal_identity_on_recovery() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    checkpointer = InMemorySaver()
    graph = build_research_state_graph(nodes.bundle(), checkpointer=checkpointer)
    config = {"configurable": {"thread_id": str(uuid4())}}

    await graph.ainvoke(
        _checkpoint_input(
            _request(case),
            mode="application",
            resume_document_id=uuid4(),
        ),
        config=config,
        context=runtime_context,
        durability="sync",
    )
    recovered_graph = build_research_state_graph(nodes.bundle(), checkpointer=checkpointer)
    snapshot = await recovered_graph.aget_state(config)
    recovered = ResearchGraphStateV1.model_validate_json(
        json.dumps(snapshot.values, allow_nan=False), strict=True
    )

    assert recovered.action_proposal_id is not None
    assert recovered.action_key == "submit_application"
    assert recovered.action_revision == 1
    assert recovered.approval_request_id is not None
    assert nodes.events.count("write") == 1


@pytest.mark.asyncio
async def test_every_streamed_graph_state_is_json_serializable_without_runtime_context() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())
    serialized_states: list[str] = []

    async for raw_state in graph.astream(
        _checkpoint_input(_request(case)),
        context=runtime_context,
        stream_mode="values",
    ):
        serialized_states.append(
            ResearchGraphStateV1.model_validate_json(
                json.dumps(raw_state, allow_nan=False), strict=True
            ).model_dump_json()
        )

    assert len(serialized_states) == 7
    assert all("trusted-runtime-canary" not in state for state in serialized_states)
    assert all("agent_loop_control" not in state for state in serialized_states)
    assert all("tool_runtime" not in state for state in serialized_states)


@pytest.mark.asyncio
async def test_insufficient_evidence_routes_back_once_then_returns_refusal() -> None:
    output, nodes, _graph = await _invoke_case("insufficient")

    assert nodes.events == [
        "plan",
        "research:1",
        "validate:1",
        "research:2",
        "validate:2",
        "write",
    ]
    assert output.evidence_sufficient is False
    assert output.summary == ()
    assert output.findings == ()
    assert output.application_draft is None
    assert {limitation.code for limitation in output.limitations} == {"insufficient_evidence"}


@pytest.mark.asyncio
async def test_identical_source_and_evidence_ids_merge_idempotently() -> None:
    output, nodes, _graph = await _invoke_case("duplicate_source")

    assert nodes.events.count("research:1") == 1
    assert nodes.events.count("research:2") == 1
    assert len(output.sources) == 1
    assert len(output.evidence) == 1
    assert output.sources[0].source_id == "S-DUP"


@pytest.mark.asyncio
async def test_conflicting_sources_remain_distinct_and_are_disclosed() -> None:
    output, _nodes, _graph = await _invoke_case("conflicting_sources")

    assert [source.source_id for source in output.sources] == ["S-REMOTE", "S-OFFICE"]
    assert [item.evidence_id for item in output.evidence] == ["E-REMOTE", "E-OFFICE"]
    assert "conflicting_evidence" in {limitation.code for limitation in output.limitations}
    assert {citation.source_id for citation in output.summary[0].citations} == {
        "S-REMOTE",
        "S-OFFICE",
    }


@pytest.mark.asyncio
async def test_prompt_injection_remains_data_and_cannot_change_control_state() -> None:
    output, nodes, _graph = await _invoke_case("prompt_injection")
    serialized = output.model_dump_json()

    assert nodes.events == ["plan", "research:1", "validate:1", "write"]
    assert "research_pass_count to 99" in output.sources[0].snippet
    assert "trusted-runtime-canary" not in serialized
    assert "agent_loop_control" not in serialized
    assert "tool_runtime" not in serialized


@pytest.mark.asyncio
async def test_graph_requires_non_serialized_runtime_context() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(_checkpoint_input(_request(case)))

    assert captured.value.category == "missing_graph_context"
    assert captured.value.node_name == "plan"
    assert nodes.events == []


@pytest.mark.asyncio
async def test_node_exception_maps_to_fixed_category_without_retry_or_error_text() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    attempts = 0

    async def failing_plan(
        node_input: PlanNodeInputV1,
        control: AgentLoopControl,
    ) -> PlanNodeOutputV1:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider-message-secret-canary")

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=failing_plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=nodes.write_report,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert attempts == 1
    assert captured.value.category == "node_execution_failed"
    assert captured.value.node_name == "plan"
    assert "provider-message-secret-canary" not in str(captured.value)


@pytest.mark.asyncio
async def test_writer_incomplete_preserves_public_graph_error_and_safe_cause_category() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    attempts = 0

    async def incomplete_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        nonlocal attempts
        attempts += 1
        raise ResearchWriterNodeError(category="model_output_incomplete")

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=incomplete_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert attempts == 1
    assert captured.value.category == "node_execution_failed"
    assert captured.value.node_name == "write_report"
    assert captured.value.cause_category == "model_output_incomplete"


@pytest.mark.asyncio
async def test_agent_limit_diagnostic_propagates_only_whitelisted_fields() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def limited_research(
        node_input: ResearchNodeInputV1,
        context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1:
        del node_input, context
        raise ResearchAgentNodeError(
            category="agent_limit_exceeded",
            limit_kind="model_calls",
            limit=12,
            current_count=12,
            requested_count=1,
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=limited_research,
            validate_evidence=nodes.validate_evidence,
            write_report=nodes.write_report,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "node_execution_failed"
    assert captured.value.node_name == "research_agent"
    assert captured.value.cause_category == "agent_limit_exceeded"
    assert captured.value.limit_kind == "model_calls"
    assert captured.value.limit == 12
    assert captured.value.current_count == 12
    assert captured.value.requested_count == 1


def test_agent_limit_diagnostic_rejects_untrusted_values() -> None:
    error = ResearchAgentNodeError(
        category="agent_limit_exceeded",
        limit_kind=cast(Any, "query-content-canary"),
        limit=cast(Any, True),
        current_count=cast(Any, -1),
        requested_count=cast(Any, "argument-content-canary"),
    )

    assert error.limit_kind is None
    assert error.limit is None
    assert error.current_count is None
    assert error.requested_count is None
    assert "canary" not in str(error)


@pytest.mark.asyncio
async def test_writer_schema_diagnostic_propagates_without_error_content() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def invalid_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        raise ResearchWriterNodeError(
            category="invalid_model_schema",
            schema_error_type="extra_forbidden",
            schema_error_path="application_draft.paragraphs.0.citations.0.source_type",
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=invalid_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "node_execution_failed"
    assert captured.value.node_name == "write_report"
    assert captured.value.cause_category == "invalid_model_schema"
    assert captured.value.schema_error_type == "extra_forbidden"
    assert captured.value.schema_error_path == (
        "application_draft.paragraphs.0.citations.0.source_type"
    )


@pytest.mark.asyncio
async def test_conflicting_duplicate_source_id_fails_closed_on_second_pass() -> None:
    case = _cases()["duplicate_source"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def conflicting_research(
        node_input: ResearchNodeInputV1,
        received_context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1:
        if node_input.research_pass_number == 1:
            return _scripted_pass(case, 1)
        first_source = _scripted_pass(case, 1).sources[0]
        return ResearchNodeOutputV1(
            sources=(
                ResearchSourceV1(
                    source_id=first_source.source_id,
                    title="Changed title under the same stable id",
                    url=first_source.url,
                    snippet=first_source.snippet,
                ),
            ),
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=conflicting_research,
            validate_evidence=nodes.validate_evidence,
            write_report=nodes.write_report,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "source_identity_conflict"


@pytest.mark.asyncio
async def test_model_writable_research_output_cannot_override_pass_counter() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def forged_research(
        node_input: ResearchNodeInputV1,
        received_context: ResearchGraphRuntimeContext,
    ) -> ResearchNodeOutputV1:
        return cast(
            ResearchNodeOutputV1,
            {
                "sources": (),
                "evidence": (),
                "research_pass_count": 99,
                "workspace_id": "forged",
            },
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=forged_research,
            validate_evidence=nodes.validate_evidence,
            write_report=nodes.write_report,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "invalid_node_output"


@pytest.mark.asyncio
async def test_orphan_writer_citation_fails_closed_at_finalize() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def orphan_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        return WriteReportNodeOutputV1(
            summary=(
                ResearchClaimV1(
                    claim_id="C-ORPHAN",
                    text="This claim points at evidence that does not exist.",
                    citations=(
                        ResearchCitationV1(
                            source_id="S1",
                            evidence_id="E-MISSING",
                        ),
                    ),
                ),
            )
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=orphan_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "invalid_final_output"
    assert captured.value.node_name == "finalize"


@pytest.mark.asyncio
async def test_custom_writer_sufficient_insufficient_limitation_fails_closed_at_finalize() -> None:
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def conflicting_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        output = await nodes.write_report(node_input, control)
        return output.model_copy(
            update={
                "limitations": (
                    ResearchLimitationV1(
                        code="insufficient_evidence",
                        detail="A custom writer bypassed the production boundary.",
                    ),
                )
            }
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=conflicting_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(_checkpoint_input(_request(case)), context=runtime_context)

    assert captured.value.category == "invalid_final_output"
    assert captured.value.node_name == "finalize"


@pytest.mark.asyncio
async def test_writer_cannot_add_unrequested_application_draft() -> None:
    case = _cases()["conflicting_sources"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def unexpected_draft_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        evidence = node_input.evidence[0]
        return WriteReportNodeOutputV1(
            application_draft=ApplicationDraftV1(
                paragraphs=(
                    ResearchClaimV1(
                        claim_id="D-UNREQUESTED",
                        text="The model attempted to add an unrequested draft.",
                        citations=(
                            ResearchCitationV1(
                                source_id=evidence.source_id,
                                evidence_id=evidence.evidence_id,
                            ),
                        ),
                    ),
                )
            )
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=unexpected_draft_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "unexpected_application_draft"
    assert captured.value.node_name == "finalize"


@pytest.mark.asyncio
async def test_writer_cannot_add_claims_after_deterministic_refusal() -> None:
    case = _cases()["insufficient"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def unsupported_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        return WriteReportNodeOutputV1(
            summary=(
                ResearchClaimV1(
                    claim_id="C-UNSUPPORTED",
                    text="The model attempted to write without evidence.",
                    citations=(
                        ResearchCitationV1(
                            source_id="S-MISSING",
                            evidence_id="E-MISSING",
                        ),
                    ),
                ),
            )
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=unsupported_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "invalid_final_output"
    assert captured.value.node_name == "finalize"


@pytest.mark.asyncio
async def test_writer_cannot_mismatch_source_and_evidence_citation() -> None:
    case = _cases()["conflicting_sources"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)

    async def mismatched_writer(
        node_input: WriteReportNodeInputV1,
        control: AgentLoopControl,
    ) -> WriteReportNodeOutputV1:
        first, second = node_input.evidence
        return WriteReportNodeOutputV1(
            summary=(
                ResearchClaimV1(
                    claim_id="C-MISMATCHED",
                    text="The model mismatched an otherwise known citation pair.",
                    citations=(
                        ResearchCitationV1(
                            source_id=second.source_id,
                            evidence_id=first.evidence_id,
                        ),
                    ),
                ),
            )
        )

    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=nodes.plan,
            research_agent=nodes.research_agent,
            validate_evidence=nodes.validate_evidence,
            write_report=mismatched_writer,
        )
    )

    with pytest.raises(ResearchGraphProtocolError) as captured:
        await graph.ainvoke(
            _checkpoint_input(_request(case)),
            context=runtime_context,
        )

    assert captured.value.category == "invalid_final_output"
    assert captured.value.node_name == "finalize"


@pytest.mark.parametrize("case_id", ["normal", "insufficient"])
async def test_graph_node_trace_parent_occurrences_and_business_body_privacy(
    case_id, collecting_trace_sink
):
    from app.domain.tracing import current_trace_scope
    from tests.tracing import collecting_segment

    sink = collecting_trace_sink
    case = _cases()[case_id]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle())
    request = _request(case).model_copy(update={"query": "TRACE-BUSINESS-BODY-CANARY"})
    with collecting_segment(sink) as scope:
        result = await graph.ainvoke(_checkpoint_input(request), context=runtime_context)
        assert current_trace_scope() is scope
    assert result["output"]["evidence_sufficient"] is (case_id == "normal")
    starts = [
        (context, span) for context, span in sink.starts.values() if span.span_kind == "graph_node"
    ]
    names = [span.metadata["node_name"] for _, span in starts]
    expected = ["normalize_request", "plan", "research_agent", "validate_evidence"]
    if case_id == "insufficient":
        expected += ["research_agent", "validate_evidence"]
    assert names == [*expected, "write_report", "finalize"]
    counts = {}
    for context, span in starts:
        name = span.metadata["node_name"]
        counts[name] = counts.get(name, 0) + 1
        assert dict(span.metadata) == {
            "node_name": name,
            "graph_version": CURRENT_GRAPH_VERSION,
            "occurrence_ordinal": counts[name],
        }
        assert span.parent == scope.parent
        assert span.trace_identity == scope.trace_identity
        assert span.segment_identity == scope.segment_identity
        assert sink.finishes[context.context_id].status == "succeeded"
    for name in ("research_agent", "validate_evidence"):
        assert [
            span.metadata["occurrence_ordinal"]
            for _, span in starts
            if span.metadata["node_name"] == name
        ] == ([1] if case_id == "normal" else [1, 2])
    assert sink.starts.keys() == sink.finishes.keys()
    assert "TRACE-BUSINESS-BODY-CANARY" not in sink.safe_json()
    assert current_trace_scope() is None


@pytest.mark.parametrize("cancel", [False, True])
async def test_node_trace_failure_and_cancellation_propagate_without_body(cancel):
    import asyncio

    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_segment

    sink = CollectingTraceSink()
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    seen = []
    started = asyncio.Event()
    original = ResearchGraphProtocolError(category="invalid_node_output", node_name="plan")

    async def failing_plan(node_input, control):
        seen.append(current_trace_scope())
        if cancel:
            started.set()
            await asyncio.Event().wait()
        raise original

    graph = build_research_state_graph(replace(nodes.bundle(), plan=failing_plan))
    with collecting_segment(sink):
        task = asyncio.create_task(
            graph.ainvoke(_checkpoint_input(_request(case)), context=runtime_context)
        )
        if cancel:
            await started.wait()
            task.cancel()
        with pytest.raises(
            asyncio.CancelledError if cancel else ResearchGraphProtocolError
        ) as error:
            await task
    if not cancel:
        assert error.value.category == "node_execution_failed"
        assert error.value.cause_category == original.category
    context, span = list(sink.starts.values())[-1]
    assert span.metadata["node_name"] == "plan"
    assert seen[0].parent == context
    finish = sink.finishes[context.context_id]
    assert (finish.status, finish.error_category) == (
        ("cancelled", "node_cancelled") if cancel else ("failed", "node_execution_failed")
    )
    assert "TRACE-BUSINESS-BODY-CANARY" not in sink.safe_json()
    assert current_trace_scope() is None


async def test_intentional_approval_interrupt_finishes_node_successfully():
    from tests.tracing import CollectingTraceSink, collecting_segment

    sink = CollectingTraceSink()
    case = _cases()["normal"]
    runtime_context = _runtime_context()
    nodes = _CaseNodes(case, runtime_context)
    graph = build_research_state_graph(nodes.bundle(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}
    with collecting_segment(sink) as scope:
        result = await graph.ainvoke(
            _checkpoint_input(_request(case), mode="application", resume_document_id=uuid4()),
            config=config,
            context=runtime_context,
            durability="sync",
        )
    assert result["__interrupt__"]
    context, span = list(sink.starts.values())[-1]
    assert span.metadata["node_name"] == "approval_interrupt"
    assert span.parent == scope.parent
    assert sink.finishes[context.context_id].status == "succeeded"
    assert sink.finishes[context.context_id].error_category is None
    assert sink.starts.keys() == sink.finishes.keys()
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("approval_interrupt",)
    assert "TraceParentContext" not in json.dumps(snapshot.values)


@pytest.mark.parametrize(
    ("status", "decision", "expected", "route"),
    [
        (
            ApprovalStatus.APPROVED,
            "approve",
            ["decision_recorded", "resume_consumed"],
            "execute_mock_action",
        ),
        (
            ApprovalStatus.REJECTED,
            "reject",
            ["decision_recorded", "resume_consumed"],
            "cancel_action",
        ),
        (ApprovalStatus.EXPIRED, None, ["expired", "resume_consumed"], "cancel_action"),
        (
            ApprovalStatus.EXPIRED,
            "approve",
            ["decision_recorded", "expired", "resume_consumed"],
            "cancel_action",
        ),
        (ApprovalStatus.PENDING, None, [], None),
    ],
)
@pytest.mark.parametrize("sink_failure", [None, "start", "finish"])
async def test_approval_lifecycle_reconstructed_under_new_segment(
    status, decision, expected, route, sink_failure
):
    from tests.tracing import CollectingTraceSink, collecting_segment

    class Sink(CollectingTraceSink):
        def start(self, span):
            if sink_failure == "start" and span.span_kind == "approval_event":
                raise RuntimeError("PRIVATE-TRACE-ERROR")
            return super().start(span)

        def finish(self, context, outcome):
            super().finish(context, outcome)
            if sink_failure == "finish" and context.span_kind == "approval_event":
                raise RuntimeError("PRIVATE-TRACE-ERROR")

    class Resolver:
        async def resolve_approval_resume(self, **kwargs):
            return SimpleNamespace(
                action_intent_id=kwargs["action_intent_id"],
                decision=decision,
                approval_request=SimpleNamespace(
                    status=status,
                    request_id=kwargs["approval_request_id"],
                    action_intent_id=kwargs["action_intent_id"],
                    approval_binding_version=1,
                ),
            )

    sink = Sink()
    runtime = replace(_runtime_context(), approval_resume_resolver=Resolver())
    case = _cases()["normal"]
    graph = build_research_state_graph(
        _CaseNodes(case, runtime).bundle(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    with collecting_segment(sink):
        first = await graph.ainvoke(
            _checkpoint_input(_request(case), mode="application", resume_document_id=uuid4()),
            config=config,
            context=runtime,
            durability="sync",
        )
    assert sink.starts.keys() == sink.finishes.keys()
    payload = first["__interrupt__"][0].value
    request_id = UUID(payload["approval_request_id"])
    created = [v for v in sink.starts.values() if v[1].span_kind == "approval_event"]
    if sink_failure != "start":
        [(_ctx, span)] = created
        assert span.name == "approval.request_created"
        assert dict(span.metadata) == {
            "approval_request_id": request_id,
            "action_intent_id": UUID(payload["action_intent_id"]),
            "binding_version": 1,
        }
        assert sink.starts[span.parent.context_id][1].metadata["node_name"] == "prepare_action"
    with collecting_segment(sink) as resumed_scope:
        resumed_result = await graph.ainvoke(
            build_approval_resume_input(request_id),
            config=config,
            context=runtime,
            durability="sync",
            interrupt_before=["execute_mock_action", "cancel_action"] if route else None,
        )
    snapshot = await graph.aget_state(config)
    if route:
        assert snapshot.next == (route,)
    else:
        assert resumed_result["__interrupt__"][0].value == payload
    observations = [
        v for v in sink.starts.values() if v[1].span_kind == "approval_event" and v not in created
    ]
    assert [v[1].name for v in observations] == (
        [] if sink_failure == "start" else [f"approval.{n}" for n in expected]
    )
    for _ctx, span in observations:
        assert span.segment_identity == resumed_scope.segment_identity
        node = sink.starts[span.parent.context_id][1]
        assert node.metadata["node_name"] == "approval_interrupt"
        assert node.parent == resumed_scope.parent
        if span.name == "approval.decision_recorded":
            assert span.metadata["decision_type"] == decision
        assert set(span.metadata) <= {
            "approval_request_id",
            "action_intent_id",
            "binding_version",
            "decision_type",
        }
    assert sink.starts.keys() == sink.finishes.keys()
    assert "PRIVATE-TRACE-ERROR" not in sink.safe_json()
    assert "TraceParentContext" not in json.dumps(snapshot.values)
