from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import ValidationError

from app.agents.contracts import (
    AgentLoopControl,
    AgentLoopLimitsV1,
    AgentLoopObservationV1,
)
from app.agents.prompting import (
    load_agent_prompt_bundle,
    load_research_plan_prompt,
    load_research_writer_prompt,
)
from app.agents.research_contracts import (
    ApplicationDraftV1,
    DocumentRetrievalCallTraceV1,
    ResearchCitationV1,
    ResearchClaimV1,
    ResearchGraphInputV1,
    ResearchGraphOutputStateV1,
    ResearchOutputV2,
    SearchCallTraceV1,
    WriteReportNodeOutputV1,
)
from app.agents.research_graph import (
    ResearchGraphNodes,
    ResearchGraphProtocolError,
    ResearchGraphRuntimeContext,
    build_research_state_graph,
)
from app.agents.research_nodes import (
    CreateAgentResearchNode,
    DeterministicEvidenceValidationNode,
    StructuredResearchPlanNode,
    StructuredResearchWriterNode,
    web_evidence_id,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchClaimV2
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationReservation
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.invocations import (
    LLMInvocationAttempt,
    LLMInvocationContext,
    LLMInvocationInvariantError,
    LLMInvocationOutcome,
)
from app.llm.ports import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_DIMENSION,
    LOCKED_EMBEDDING_MODEL,
    ChatModelPort,
    ChatModelResult,
    ModelToolCall,
    ModelUsage,
)
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from app.llm.qwen_adapters import QWEN_REASONING_EFFORT
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentRetrievalService,
    RetrievedDocumentChunk,
)
from app.tools.contracts import ToolRunContext
from app.tools.document_retrieval import (
    RESEARCH_V2_TOOL_POLICY,
    RETRIEVE_DOCUMENTS_MAX_OUTPUT_BYTES,
    RETRIEVE_DOCUMENTS_PER_RUN_CALL_LIMIT,
    RETRIEVE_DOCUMENTS_TIMEOUT_SECONDS,
    RETRIEVE_DOCUMENTS_TOOL_NAME,
    RetrieveDocumentsInputV1,
    RetrieveDocumentsOutputV1,
    create_research_tool_registry,
)
from app.tools.fake_search import FakeSearch
from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.search import SearchError, SearchResult, normalize_search_result
from app.tools.web_search import (
    RESEARCH_TOOL_POLICY_NAME,
    SEARCH_WEB_MAX_ATTEMPTS,
    SEARCH_WEB_MAX_OUTPUT_BYTES,
    SEARCH_WEB_PER_RUN_CALL_LIMIT,
    SEARCH_WEB_TIMEOUT_SECONDS,
    SEARCH_WEB_TOOL_NAME,
    SearchWebInputV1,
    SearchWebOutputV1,
    normalize_web_search_query,
)
from tests.evals.contracts import (
    EVAL_GRADER_CONTRACT_VERSION,
    EVAL_GRADER_NAMES,
    TRUSTED_CONTEXT_CANARY,
    EvalArtifactIdentityV1,
    EvalCaseReportV3,
    EvalCitationResolutionV1,
    EvalClaimFixtureV1,
    EvalComparisonMetadataV1,
    EvalConfigurationErrorCategory,
    EvalGraderAggregateV1,
    EvalGraderName,
    EvalGraderResultV1,
    EvalManifestV1,
    EvalMetricsV2,
    EvalReportV3,
    EvalVersionMetadataV1,
    ResearchEvalCaseV2,
)
from tests.evals.grader_vectors import grader_expected_results_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "research_v1.jsonl"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "research_v3.jsonl"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "evals" / "datasets" / "research_v3_manifest.json"
type EvalExitCode = Literal[0, 1, 2]
type EvalExecutionErrorCategory = Literal[
    "invalid_output", "provider_failure", "graph_execution_failed"
]


class EvalConfigurationError(Exception):
    def __init__(self, *, category: EvalConfigurationErrorCategory) -> None:
        self.category = category
        super().__init__(f"research eval configuration failed: {category}")


class EvalExecutionError(Exception):
    def __init__(self, *, category: EvalExecutionErrorCategory) -> None:
        self.category = category
        super().__init__(f"research eval execution failed: {category}")


@dataclass(frozen=True, slots=True)
class _EvidenceReference:
    source_id: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class _PreparedEvalCase:
    search_fixtures: Mapping[str, tuple[SearchResult, ...]]
    evidence_by_alias: Mapping[str, _EvidenceReference]
    evidence_alias_by_id: Mapping[str, str]
    document_results: Mapping[str, tuple[RetrievedDocumentChunk, ...]]
    relevant_chunk_ids: frozenset[UUID]
    irrelevant_chunk_ids: frozenset[UUID]


@dataclass(frozen=True, slots=True)
class ExecutedEvalCase:
    case: ResearchEvalCaseV2
    output: ResearchOutputV2
    search_calls: tuple[SearchCallTraceV1, ...]
    document_retrieval_calls: tuple[DocumentRetrievalCallTraceV1, ...]
    model_call_count: int
    input_tokens: int
    output_tokens: int
    evidence_alias_by_id: Mapping[str, str]
    relevant_chunk_ids: frozenset[UUID]
    irrelevant_chunk_ids: frozenset[UUID]
    tool_invocation_reservation_count: int


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class _DeterministicInvocationIds:
    def __init__(self) -> None:
        self._next_value = 100

    def __call__(self) -> UUID:
        result = UUID(int=self._next_value)
        self._next_value += 1
        return result


class _StrictMemoryInvocationRecorder:
    def __init__(self) -> None:
        self.attempts: dict[UUID, LLMInvocationAttempt] = {}
        self.outcomes: dict[UUID, LLMInvocationOutcome] = {}

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        if not isinstance(attempt, LLMInvocationAttempt) or attempt.invocation_id in self.attempts:
            raise LLMInvocationInvariantError
        self.attempts[attempt.invocation_id] = attempt

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        if (
            not isinstance(attempt, LLMInvocationAttempt)
            or not isinstance(outcome, LLMInvocationOutcome)
            or self.attempts.get(attempt.invocation_id) != attempt
            or attempt.invocation_id in self.outcomes
        ):
            raise LLMInvocationInvariantError
        self.outcomes[attempt.invocation_id] = outcome

    def token_totals(self) -> tuple[int, int]:
        if self.attempts.keys() != self.outcomes.keys():
            raise LLMInvocationInvariantError
        usages = tuple(
            outcome.token_usage
            for outcome in self.outcomes.values()
            if outcome.status == "succeeded" and outcome.token_usage is not None
        )
        return (
            sum(usage.input_tokens for usage in usages),
            sum(usage.output_tokens for usage in usages),
        )


class _CountingToolInvocationRecorder:
    def __init__(self) -> None:
        self._delegate = InMemoryToolInvocationRecorder()
        self.reservation_count = 0

    async def reserve(
        self,
        *,
        invocation_id: UUID,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        tool_name: str,
        effect: ToolEffect,
        args_digest: str,
        call_limit: int,
    ) -> ToolInvocationReservation:
        reservation = await self._delegate.reserve(
            invocation_id=invocation_id,
            workspace_id=workspace_id,
            actor_user_id=actor_user_id,
            run_id=run_id,
            tool_name=tool_name,
            effect=effect,
            args_digest=args_digest,
            call_limit=call_limit,
        )
        self.reservation_count += 1
        return reservation

    async def start_attempt(self, **kwargs: object) -> None:
        await self._delegate.start_attempt(**kwargs)

    async def succeed(self, **kwargs: object) -> None:
        await self._delegate.succeed(**kwargs)

    async def fail(self, **kwargs: object) -> None:
        await self._delegate.fail(**kwargs)


class _CollectingObserver:
    def __init__(self) -> None:
        self.observations: list[AgentLoopObservationV1] = []

    def observe(self, observation: AgentLoopObservationV1) -> None:
        self.observations.append(observation)


class _EvalDocumentRepository:
    def __init__(
        self,
        results: Mapping[tuple[float, ...], tuple[RetrievedDocumentChunk, ...]],
        *,
        tenant: TenantContext,
        allowed_document_ids: tuple[UUID, ...],
    ):
        self._results = results
        self._tenant = tenant
        self._allowed_document_ids = allowed_document_ids

    async def search(self, **kwargs: object) -> tuple[RetrievedDocumentChunk, ...]:
        if kwargs.get("tenant") != self._tenant:
            raise ValueError("eval retrieval tenant must remain server-owned")
        if kwargs.get("allowed_document_ids") != self._allowed_document_ids:
            raise ValueError("eval retrieval allowlist must remain server-owned")
        if kwargs.get("embedding_model") != EMBEDDING_PROFILE:
            raise ValueError("eval retrieval embedding profile must remain fixed")
        if kwargs.get("limit") != 5:
            raise ValueError("eval retrieval top-k must remain fixed")
        vector = kwargs["query_embedding"]
        if not isinstance(vector, tuple):
            raise ValueError("eval query embedding must be a tuple")
        results = self._results.get(vector, ())
        if any(item.document_id not in self._allowed_document_ids for item in results):
            raise ValueError("eval fixture returned evidence outside the persisted resume scope")
        return results


class _EvalRetrievalEventRecorder:
    async def record_retrieved(self, **_kwargs: object) -> None:
        return None


def _prepare_case(case: ResearchEvalCaseV2) -> _PreparedEvalCase:
    normalized_queries = (*case.plan_queries, *(search.query for search in case.searches))
    if any(normalize_web_search_query(query) != query for query in normalized_queries):
        raise ValueError("eval queries must already use the production normalization")

    search_fixtures: dict[str, tuple[SearchResult, ...]] = {}
    evidence_by_alias: dict[str, _EvidenceReference] = {}
    evidence_alias_by_id: dict[str, str] = {}
    document_results: dict[str, tuple[RetrievedDocumentChunk, ...]] = {}
    relevant_chunk_ids: set[UUID] = set()
    irrelevant_chunk_ids: set[UUID] = set()
    for search in case.searches:
        normalized_results: list[SearchResult] = []
        for fixture in search.results:
            result = normalize_search_result(
                title=fixture.title,
                url=fixture.url,
                snippet=fixture.snippet,
                published_at=fixture.published_at,
            )
            normalized_results.append(result)
            if fixture.evidence_alias is not None:
                evidence_id = web_evidence_id(
                    source_id=result.source_id,
                    snippet=result.snippet,
                )
                if evidence_id in evidence_alias_by_id:
                    raise ValueError("eval evidence aliases must map to unique stable evidence")
                evidence_by_alias[fixture.evidence_alias] = _EvidenceReference(
                    source_id=result.source_id,
                    evidence_id=evidence_id,
                )
                evidence_alias_by_id[evidence_id] = fixture.evidence_alias
        search_fixtures[search.query] = tuple(normalized_results)

    for retrieval in case.document_retrievals:
        hits: list[RetrievedDocumentChunk] = []
        for fixture in retrieval.results:
            source_id = f"workspace-document-v1:{fixture.document_id}"
            evidence_id = f"workspace-chunk-v1:{fixture.chunk_id}"
            evidence_by_alias[fixture.evidence_alias] = _EvidenceReference(
                source_id=source_id,
                evidence_id=evidence_id,
            )
            evidence_alias_by_id[evidence_id] = fixture.evidence_alias
            (relevant_chunk_ids if fixture.relevant else irrelevant_chunk_ids).add(fixture.chunk_id)
            hits.append(
                RetrievedDocumentChunk(
                    document_id=fixture.document_id,
                    chunk_id=fixture.chunk_id,
                    source_name=fixture.source_name,
                    section=fixture.section,
                    ordinal=fixture.ordinal,
                    cosine_distance=fixture.cosine_distance,
                    text=fixture.untrusted_text,
                )
            )
        document_results[retrieval.query] = tuple(hits)

    return _PreparedEvalCase(
        search_fixtures=search_fixtures,
        evidence_by_alias=evidence_by_alias,
        evidence_alias_by_id=evidence_alias_by_id,
        document_results=document_results,
        relevant_chunk_ids=frozenset(relevant_chunk_ids),
        irrelevant_chunk_ids=frozenset(irrelevant_chunk_ids),
    )


def load_eval_dataset(path: Path = DEFAULT_DATASET_PATH) -> tuple[ResearchEvalCaseV2, ...]:
    try:
        raw_dataset = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        raise EvalConfigurationError(category="dataset_unavailable") from None

    lines = raw_dataset.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise EvalConfigurationError(category="invalid_dataset")

    cases: list[ResearchEvalCaseV2] = []
    try:
        for line in lines:
            if TRUSTED_CONTEXT_CANARY in line:
                raise ValueError("trusted canary must not be dataset content")
            case = ResearchEvalCaseV2.model_validate_json(line, strict=True)
            _prepare_case(case)
            cases.append(case)
    except (SearchError, TypeError, ValidationError, ValueError):
        raise EvalConfigurationError(category="invalid_dataset") from None

    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise EvalConfigurationError(category="invalid_dataset")
    return tuple(cases)


def _canonical_digest(domain: bytes, payload: object) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(domain + b'\x00' + serialized).hexdigest()}"


def eval_dataset_digests(
    cases: Sequence[ResearchEvalCaseV2],
) -> tuple[str, str]:
    ordered_cases = tuple(sorted(cases, key=lambda case: case.case_id))
    dataset_payload = [case.model_dump(mode="json", round_trip=True) for case in ordered_cases]
    case_ids = [case.case_id for case in ordered_cases]
    return (
        _canonical_digest(b"pathfinder-eval-dataset-v1", dataset_payload),
        _canonical_digest(b"pathfinder-eval-case-set-v1", case_ids),
    )


def eval_artifact_identity(cases: Sequence[ResearchEvalCaseV2]) -> EvalArtifactIdentityV1:
    dataset_digest, case_set_digest = eval_dataset_digests(cases)
    grader_contract_digest = _canonical_digest(
        b"pathfinder-eval-grader-contract-v1",
        {
            "grader_contract_version": EVAL_GRADER_CONTRACT_VERSION,
            "grader_names": EVAL_GRADER_NAMES,
            "grader_expected_results": grader_expected_results_payload(),
        },
    )
    return EvalArtifactIdentityV1(
        dataset_digest=dataset_digest,
        case_set_digest=case_set_digest,
        graph_version=CURRENT_GRAPH_VERSION,
        embedding_profile=EMBEDDING_PROFILE,
        grader_contract_version=EVAL_GRADER_CONTRACT_VERSION,
        grader_contract_digest=grader_contract_digest,
    )


def _research_tool_schema_version() -> str:
    payload = {
        "policy": {
            "name": RESEARCH_V2_TOOL_POLICY.name,
            "allowed_tool_names": sorted(RESEARCH_V2_TOOL_POLICY.allowed_tool_names),
            "allowed_effects": sorted(
                effect.value for effect in RESEARCH_V2_TOOL_POLICY.allowed_effects
            ),
        },
        "tools": [
            {
                "name": SEARCH_WEB_TOOL_NAME,
                "input_schema": SearchWebInputV1.model_json_schema(mode="validation"),
                "output_schema": SearchWebOutputV1.model_json_schema(mode="validation"),
                "timeout_seconds": SEARCH_WEB_TIMEOUT_SECONDS,
                "max_attempts": SEARCH_WEB_MAX_ATTEMPTS,
                "per_run_call_limit": SEARCH_WEB_PER_RUN_CALL_LIMIT,
                "max_output_bytes": SEARCH_WEB_MAX_OUTPUT_BYTES,
            },
            {
                "name": RETRIEVE_DOCUMENTS_TOOL_NAME,
                "input_schema": RetrieveDocumentsInputV1.model_json_schema(mode="validation"),
                "output_schema": RetrieveDocumentsOutputV1.model_json_schema(mode="validation"),
                "timeout_seconds": RETRIEVE_DOCUMENTS_TIMEOUT_SECONDS,
                "max_attempts": 1,
                "per_run_call_limit": RETRIEVE_DOCUMENTS_PER_RUN_CALL_LIMIT,
                "max_output_bytes": RETRIEVE_DOCUMENTS_MAX_OUTPUT_BYTES,
            },
        ],
    }
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(b'pathfinder-research-tool-contract-v1\x00' + serialized).hexdigest()}"


def runtime_version_metadata() -> EvalVersionMetadataV1:
    return EvalVersionMetadataV1(
        chat_model=LOCKED_CHAT_MODEL,
        embedding_model=LOCKED_EMBEDDING_MODEL,
        embedding_dimension=LOCKED_EMBEDDING_DIMENSION,
        reasoning_effort=QWEN_REASONING_EFFORT,
        plan_prompt_version=load_research_plan_prompt().version,
        research_prompt_version=load_agent_prompt_bundle("research").version,
        writer_prompt_version=load_research_writer_prompt().version,
        research_tool_policy=RESEARCH_TOOL_POLICY_NAME,
        research_tool_schema_version=_research_tool_schema_version(),
        research_output_schema_version=2,
        pricing_version=QWEN_BEIJING_PRICING_VERSION,
    )


def load_eval_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> EvalVersionMetadataV1:
    try:
        raw_manifest = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        raise EvalConfigurationError(category="manifest_unavailable") from None
    try:
        manifest = EvalManifestV1.model_validate_json(raw_manifest, strict=True)
        metadata = EvalVersionMetadataV1.model_validate(
            manifest.model_dump(exclude={"schema_version", "dataset_version"}),
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        raise EvalConfigurationError(category="invalid_manifest") from None
    if metadata != runtime_version_metadata():
        raise EvalConfigurationError(category="invalid_manifest")
    return metadata


def _resolved_claim(
    fixture: EvalClaimFixtureV1,
    evidence_by_alias: Mapping[str, _EvidenceReference],
) -> ResearchClaimV1:
    return ResearchClaimV1(
        claim_id=fixture.claim_id,
        text=fixture.text,
        citations=tuple(
            ResearchCitationV1(
                source_id=evidence_by_alias[citation.evidence_alias].source_id,
                evidence_id=evidence_by_alias[citation.evidence_alias].evidence_id,
            )
            for citation in fixture.citations
        ),
    )


def _writer_output(
    case: ResearchEvalCaseV2,
    prepared: _PreparedEvalCase,
) -> WriteReportNodeOutputV1:
    draft = None
    if case.writer.application_draft is not None:
        draft = ApplicationDraftV1(
            paragraphs=tuple(
                _resolved_claim(claim, prepared.evidence_by_alias)
                for claim in case.writer.application_draft.paragraphs
            )
        )
    return WriteReportNodeOutputV1(
        summary=tuple(
            _resolved_claim(claim, prepared.evidence_by_alias) for claim in case.writer.summary
        ),
        findings=tuple(
            _resolved_claim(claim, prepared.evidence_by_alias) for claim in case.writer.findings
        ),
        limitations=case.writer.limitations,
        application_draft=draft,
    )


def _writer_script(
    case: ResearchEvalCaseV2,
    prepared: _PreparedEvalCase,
) -> tuple[ChatModelResult, ...]:
    content = _writer_output(case, prepared).model_dump_json()
    result = ChatModelResult(
        content=content,
        usage=ModelUsage(input_tokens=1, output_tokens=1),
    )
    # The second result is only consumed by the writer's bounded malformed-output retry.
    return (result, result)


def _research_script(case: ResearchEvalCaseV2) -> tuple[ChatModelResult, ...]:
    script: list[ChatModelResult] = []
    for pass_number, research_pass in enumerate(case.research_passes, start=1):
        if research_pass.ordered_tool_calls:
            tool_calls = tuple(
                ModelToolCall(
                    call_id=f"{case.case_id}-p{pass_number}-c{call_number}",
                    name=call.tool_name,
                    arguments={
                        **(
                            {"query": call.query, "max_results": call.max_results}
                            if call.tool_name == SEARCH_WEB_TOOL_NAME
                            else {"query": call.query}
                        ),
                        **call.model_extra_arguments,
                    },
                )
                for call_number, call in enumerate(research_pass.ordered_tool_calls, start=1)
            )
        else:
            tool_calls = tuple(
                ModelToolCall(
                    call_id=f"{case.case_id}-p{pass_number}-c{call_number}",
                    name=SEARCH_WEB_TOOL_NAME,
                    arguments={"query": call.query, "max_results": call.max_results},
                )
                for call_number, call in enumerate(research_pass.calls, start=1)
            )
        script.append(
            ChatModelResult(
                tool_calls=tool_calls,
                usage=ModelUsage(input_tokens=1, output_tokens=1),
            )
        )
        if research_pass.expected_rejection is None:
            script.append(
                ChatModelResult(
                    content="Bounded offline research pass complete.",
                    usage=ModelUsage(input_tokens=1, output_tokens=1),
                )
            )
    return tuple(script)


def _runtime_context(
    prepared: _PreparedEvalCase,
    embedding: object,
    *,
    resume_document_id: UUID | None,
    recorder: _CountingToolInvocationRecorder,
) -> ResearchGraphRuntimeContext:
    cancellation = _NeverCancelled()
    tenant = TenantContext(UUID(int=1), UUID(int=2), WorkspaceRole.ADMIN)
    allowed_document_ids = (resume_document_id,) if resume_document_id is not None else ()
    vector_results = {
        FakeEmbeddingModel._vector_for_text(query): hits
        for query, hits in prepared.document_results.items()
    }
    registry = create_research_tool_registry(
        search_port=FakeSearch(prepared.search_fixtures, clock=lambda: 0.0),
        retrieval_service=DocumentRetrievalService(
            repository=_EvalDocumentRepository(
                vector_results,
                tenant=tenant,
                allowed_document_ids=allowed_document_ids,
            ),
            embedding=embedding,  # type: ignore[arg-type]
        ),
        tenant=tenant,
        allowed_document_ids=allowed_document_ids,
        event_recorder=_EvalRetrievalEventRecorder(),
        recorder=recorder,
    )
    tool_runtime = registry.bind(
        policy_name=RESEARCH_TOOL_POLICY_NAME,
        context=ToolRunContext(
            workspace_id=UUID(int=1),
            actor_user_id=UUID(int=2),
            run_id=UUID(int=3),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target={"canary": TRUSTED_CONTEXT_CANARY},
            deadline=100.0,
            cancellation=cancellation,
        ),
        invocation_id_factory=_DeterministicInvocationIds(),
        clock=lambda: 0.0,
    )
    return ResearchGraphRuntimeContext(
        tool_runtime=tool_runtime,
        agent_loop_control=AgentLoopControl(
            limits=AgentLoopLimitsV1(
                max_model_calls=12,
                max_tool_calls=8,
                max_tool_results=8,
                max_iterations=24,
            ),
            deadline=100.0,
            cancellation=cancellation,
            clock=lambda: 0.0,
        ),
    )


async def execute_eval_case(case: ResearchEvalCaseV2) -> ExecutedEvalCase:
    prepared = _prepare_case(case)
    recorder = _StrictMemoryInvocationRecorder()
    invocation_ids = _DeterministicInvocationIds()
    tool_invocation_recorder = _CountingToolInvocationRecorder()
    context = LLMInvocationContext(
        workspace_id=UUID(int=1),
        actor_user_id=UUID(int=2),
        request_id=UUID(int=4),
        run_id=None,
    )

    def accounted_model(script: Sequence[ChatModelResult]) -> ChatModelPort:
        return LLMFactory(
            recorder=recorder,
            chat_adapter=ScriptedFakeChatModel(script),
            embedding_adapter=FakeEmbeddingModel(),
            provider="fake",
            invocation_id_factory=invocation_ids,
        ).create_chat_model(context)

    plan_model = accounted_model(
        (
            ChatModelResult(
                content=json.dumps(
                    {"queries": list(case.plan_queries)},
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                usage=ModelUsage(input_tokens=1, output_tokens=1),
            ),
        )
    )
    research_model = accounted_model(_research_script(case))
    embedding = LLMFactory(
        recorder=recorder,
        chat_adapter=ScriptedFakeChatModel(
            (ChatModelResult(content="unused", usage=ModelUsage(input_tokens=0, output_tokens=0)),)
        ),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
        invocation_id_factory=invocation_ids,
    ).create_embedding_model(context)
    writer_model = accounted_model(_writer_script(case, prepared))
    graph = build_research_state_graph(
        ResearchGraphNodes(
            plan=StructuredResearchPlanNode(plan_model),
            research_agent=CreateAgentResearchNode(research_model, _CollectingObserver()),
            validate_evidence=DeterministicEvidenceValidationNode(),
            write_report=StructuredResearchWriterNode(writer_model),
        )
    )
    raw_output = await graph.ainvoke(
        ResearchGraphInputV1(
            schema_version=2,
            mode="application" if case.request.include_application_draft else "research",
            resume_document_id=case.resume_document_id,
            request=case.request,
        ).model_dump(mode="json", round_trip=True),
        context=_runtime_context(
            prepared,
            embedding,
            resume_document_id=case.resume_document_id,
            recorder=tool_invocation_recorder,
        ),
    )
    try:
        output_state = ResearchGraphOutputStateV1.model_validate_json(
            json.dumps(raw_output, allow_nan=False),
            strict=True,
        )
    except (TypeError, ValidationError, ValueError):
        raise EvalExecutionError(category="invalid_output") from None
    if not isinstance(output_state.output, ResearchOutputV2):
        raise EvalExecutionError(category="invalid_output")
    input_tokens, output_tokens = recorder.token_totals()
    return ExecutedEvalCase(
        case=case,
        output=output_state.output,
        search_calls=output_state.search_calls,
        document_retrieval_calls=output_state.document_retrieval_calls,
        model_call_count=len(recorder.outcomes),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        evidence_alias_by_id=prepared.evidence_alias_by_id,
        relevant_chunk_ids=prepared.relevant_chunk_ids,
        irrelevant_chunk_ids=prepared.irrelevant_chunk_ids,
        tool_invocation_reservation_count=tool_invocation_recorder.reservation_count,
    )


def _all_claims(output: ResearchOutputV2) -> tuple[ResearchClaimV2, ...]:
    draft_claims = output.application_draft.paragraphs if output.application_draft else ()
    return (*output.summary, *output.findings, *draft_claims)


def _schema_is_valid(output: ResearchOutputV2) -> bool:
    try:
        ResearchOutputV2.model_validate_json(output.model_dump_json(), strict=True)
    except (TypeError, ValidationError, ValueError):
        return False
    return True


def _citation_resolutions(
    output: ResearchOutputV2,
    evidence_alias_by_id: Mapping[str, str],
) -> tuple[EvalCitationResolutionV1, ...]:
    source_types = {source.source_id: source.source_type for source in output.sources}
    evidence_facts = {
        evidence.evidence_id: (evidence.source_id, evidence.source_type)
        for evidence in output.evidence
    }
    resolutions: list[EvalCitationResolutionV1] = []
    for claim in _all_claims(output):
        for citation in claim.citations:
            resolutions.append(
                EvalCitationResolutionV1(
                    claim_id=claim.claim_id,
                    source_id=citation.source_id,
                    evidence_id=citation.evidence_id,
                    evidence_alias=evidence_alias_by_id.get(citation.evidence_id),
                    resolvable=(
                        source_types.get(citation.source_id) == citation.source_type
                        and evidence_facts.get(citation.evidence_id)
                        == (citation.source_id, citation.source_type)
                    ),
                )
            )
    return tuple(resolutions)


def _unsupported_claim_ids(
    case: ResearchEvalCaseV2,
    output: ResearchOutputV2,
    evidence_alias_by_id: Mapping[str, str],
) -> tuple[str, ...]:
    rules = {rule.claim_id: rule for rule in case.support_rules}
    unsupported: list[str] = []
    for claim in _all_claims(output):
        rule = rules.get(claim.claim_id)
        citation_aliases = tuple(
            evidence_alias_by_id.get(citation.evidence_id) for citation in claim.citations
        )
        if (
            rule is None
            or claim.text != rule.exact_text
            or any(
                alias is None or alias not in rule.allowed_evidence_aliases
                for alias in citation_aliases
            )
        ):
            unsupported.append(claim.claim_id)
    return tuple(sorted(set(unsupported)))


def _grader(name: EvalGraderName, failure_count: int) -> EvalGraderResultV1:
    return EvalGraderResultV1(
        name=name,
        passed=failure_count == 0,
        failure_count=failure_count,
    )


def grade_eval_case(
    case: ResearchEvalCaseV2,
    output: ResearchOutputV2,
    search_calls: Sequence[SearchCallTraceV1],
    *,
    model_call_count: int,
    evidence_alias_by_id: Mapping[str, str],
    document_retrieval_calls: Sequence[DocumentRetrievalCallTraceV1] = (),
    relevant_chunk_ids: frozenset[UUID] = frozenset(),
    irrelevant_chunk_ids: frozenset[UUID] = frozenset(),
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_invocation_reservation_count: int = 0,
) -> EvalCaseReportV3:
    claims = _all_claims(output)
    schema_valid = _schema_is_valid(output)
    resolutions = _citation_resolutions(output, evidence_alias_by_id)
    unresolved_citations = sum(not resolution.resolvable for resolution in resolutions)
    citation_source_type_mismatches = sum(
        citation.source_type
        != next(
            (
                evidence.source_type
                for evidence in output.evidence
                if evidence.evidence_id == citation.evidence_id
            ),
            citation.source_type,
        )
        for claim in claims
        for citation in claim.citations
    )
    unsupported_claim_ids = _unsupported_claim_ids(case, output, evidence_alias_by_id)
    cited_sources = {resolution.source_id for resolution in resolutions if resolution.resolvable}
    search_call_count = len(search_calls)
    search_result_count = sum(call.result_count for call in search_calls)
    research_pass_count = max(
        (
            call.research_pass_number
            for call in (*tuple(search_calls), *tuple(document_retrieval_calls))
        ),
        default=0,
    )
    document_evidence = tuple(
        item for item in output.evidence if item.source_type == "workspace_document"
    )
    retrieved_chunk_ids = {item.chunk_id for item in document_evidence if item.chunk_id is not None}
    retrieved_relevant = len(retrieved_chunk_ids & relevant_chunk_ids)
    irrelevant_context = len(retrieved_chunk_ids & irrelevant_chunk_ids)
    recall_at_5 = retrieved_relevant / len(relevant_chunk_ids) if relevant_chunk_ids else None
    context_bytes = sum(len(item.text.encode("utf-8")) for item in document_evidence)
    document_call_count = len(document_retrieval_calls)
    total_tool_calls = search_call_count + document_call_count

    budget_failures = sum(
        (
            model_call_count > case.expectations.max_model_calls,
            input_tokens > case.expectations.max_input_tokens,
            output_tokens > case.expectations.max_output_tokens,
            search_call_count > case.expectations.max_search_calls,
            search_result_count > case.expectations.max_search_results,
            research_pass_count > case.expectations.max_research_passes,
            document_call_count > case.expectations.max_document_retrieval_calls,
            total_tool_calls > case.expectations.max_total_tool_calls,
            tuple(call.call_ordinal for call in search_calls)
            != tuple(range(1, search_call_count + 1)),
        )
    )
    actual_limitation_codes = tuple(sorted({item.code for item in output.limitations}))
    expected_limitation_codes = tuple(sorted(case.expectations.limitation_codes))
    actual_claim_ids = tuple(sorted(claim.claim_id for claim in claims))
    expected_claim_ids = tuple(sorted(rule.claim_id for rule in case.support_rules))
    output_contract_failures = sum(
        (
            output.evidence_sufficient != case.expectations.evidence_sufficient,
            (output.application_draft is not None) != case.expectations.application_draft_present,
            actual_limitation_codes != expected_limitation_codes,
            actual_claim_ids != expected_claim_ids,
        )
    )
    expected_tool_behavior = case.expectations.expected_tool_behavior
    if expected_tool_behavior is not None:
        output_contract_failures += sum(
            (
                tuple(call.research_pass_number for call in search_calls)
                != expected_tool_behavior.search_pass_numbers,
                tuple(call.research_pass_number for call in document_retrieval_calls)
                != expected_tool_behavior.document_retrieval_pass_numbers,
                tool_invocation_reservation_count
                != expected_tool_behavior.tool_invocation_reservation_count,
            )
        )
    serialized_visible_state = "".join(
        (
            output.model_dump_json(),
            *(call.model_dump_json() for call in search_calls),
        )
    )
    trusted_context_failures = int(TRUSTED_CONTEXT_CANARY in serialized_visible_state)
    retrieval_failures = sum(
        (
            case.expectations.min_recall_at_5 is not None
            and (recall_at_5 is None or recall_at_5 < case.expectations.min_recall_at_5),
            retrieved_relevant != case.expectations.expected_relevant_chunk_count,
            irrelevant_context > case.expectations.max_irrelevant_context_count,
            context_bytes > case.expectations.max_context_bytes,
            any(len(item.text.encode("utf-8")) > 800 for item in document_evidence),
        )
    )

    graders = (
        _grader("schema_valid", int(not schema_valid)),
        _grader("citation_resolvable", unresolved_citations),
        _grader("unsupported_claim", len(unsupported_claim_ids)),
        _grader(
            "source_diversity",
            max(0, case.expectations.minimum_cited_sources - len(cited_sources)),
        ),
        _grader("budget", budget_failures),
        _grader("output_contract", output_contract_failures),
        _grader("trusted_context_isolated", trusted_context_failures),
        _grader("retrieval_quality", retrieval_failures),
    )
    return EvalCaseReportV3(
        case_id=case.case_id,
        passed=all(grader.passed for grader in graders),
        output=output if schema_valid else None,
        search_calls=tuple(search_calls),
        document_retrieval_calls=tuple(document_retrieval_calls),
        citation_resolutions=resolutions,
        metrics=EvalMetricsV2(
            schema_valid=schema_valid,
            claim_count=len(claims),
            citation_count=len(resolutions),
            unresolved_citation_count=unresolved_citations,
            unsupported_claim_count=len(unsupported_claim_ids),
            cited_source_count=len(cited_sources),
            model_call_count=model_call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            search_call_count=search_call_count,
            search_result_count=search_result_count,
            research_pass_count=research_pass_count,
            document_retrieval_call_count=document_call_count,
            total_tool_call_count=total_tool_calls,
            retrieved_relevant_chunk_count=retrieved_relevant,
            irrelevant_context_count=irrelevant_context,
            context_bytes=context_bytes,
            recall_at_5=recall_at_5,
            citation_source_type_mismatch_count=citation_source_type_mismatches,
        ),
        graders=graders,
        error_category=None if schema_valid else "invalid_output",
    )


def _execution_failure_report(
    case: ResearchEvalCaseV2,
    category: EvalExecutionErrorCategory,
) -> EvalCaseReportV3:
    return EvalCaseReportV3(
        case_id=case.case_id,
        passed=False,
        output=None,
        metrics=EvalMetricsV2(
            schema_valid=False,
            claim_count=0,
            citation_count=0,
            unresolved_citation_count=0,
            unsupported_claim_count=0,
            cited_source_count=0,
            model_call_count=0,
            input_tokens=0,
            output_tokens=0,
            search_call_count=0,
            search_result_count=0,
            research_pass_count=0,
            document_retrieval_call_count=0,
            total_tool_call_count=0,
            retrieved_relevant_chunk_count=0,
            irrelevant_context_count=0,
            context_bytes=0,
            recall_at_5=None,
            citation_source_type_mismatch_count=0,
        ),
        graders=tuple(
            EvalGraderResultV1(name=name, passed=False, failure_count=1)
            for name in EVAL_GRADER_NAMES
        ),
        error_category=category,
    )


_INVALID_MODEL_OUTPUT_CAUSES = frozenset(
    {
        "model_output_incomplete",
        "invalid_model_json",
        "invalid_model_schema",
        "invalid_model_grounding",
        "empty_model_output",
        "invalid_model_output",
    }
)
_PROVIDER_FAILURE_CAUSES = frozenset({"provider_timeout", "provider_unavailable"})


def _protocol_error_category(error: ResearchGraphProtocolError) -> EvalExecutionErrorCategory:
    if error.cause_category in _PROVIDER_FAILURE_CAUSES:
        return "provider_failure"
    if (
        error.node_name in {"plan", "write_report"}
        and error.cause_category in _INVALID_MODEL_OUTPUT_CAUSES
    ):
        return "invalid_output"
    return "graph_execution_failed"


async def run_evaluation(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    selected_case: str | None = None,
) -> tuple[EvalReportV3, EvalExitCode]:
    try:
        version_metadata = load_eval_manifest(manifest_path)
        cases = load_eval_dataset(dataset_path)
    except EvalConfigurationError as error:
        return (
            EvalReportV3(
                selected_case=selected_case,
                passed=False,
                error_category=error.category,
            ),
            2,
        )

    dataset_cases = cases
    artifact_identity = eval_artifact_identity(dataset_cases)
    dataset_case_ids = tuple(sorted(case.case_id for case in dataset_cases))
    if selected_case is not None:
        cases = tuple(case for case in dataset_cases if case.case_id == selected_case)
        if not cases:
            return (
                EvalReportV3(
                    selected_case=selected_case,
                    passed=False,
                    error_category="unknown_case",
                ),
                2,
            )

    reports: list[EvalCaseReportV3] = []
    for case in cases:
        try:
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
        except EvalExecutionError as error:
            report = _execution_failure_report(case, error.category)
        except ResearchGraphProtocolError as error:
            report = _execution_failure_report(case, _protocol_error_category(error))
        except Exception:
            report = _execution_failure_report(case, "graph_execution_failed")
        reports.append(report)

    passed = all(report.passed for report in reports)
    grader_aggregates = tuple(
        EvalGraderAggregateV1(
            name=name,
            passed_case_count=sum(
                next(grader for grader in report.graders if grader.name == name).passed
                for report in reports
            ),
            failed_case_count=sum(
                not next(grader for grader in report.graders if grader.name == name).passed
                for report in reports
            ),
            failure_count=sum(
                next(grader for grader in report.graders if grader.name == name).failure_count
                for report in reports
            ),
        )
        for name in EVAL_GRADER_NAMES
    )
    return (
        EvalReportV3(
            selected_case=selected_case,
            passed=passed,
            cases=tuple(reports),
            version_metadata=version_metadata,
            artifact_identity=artifact_identity,
            comparison_metadata=EvalComparisonMetadataV1(
                scope="selected_case" if selected_case is not None else "full_dataset",
                dataset_case_ids=dataset_case_ids,
                evaluated_case_ids=tuple(sorted(report.case_id for report in reports)),
            ),
            grader_aggregates=grader_aggregates,
            input_tokens=sum(report.metrics.input_tokens for report in reports),
            output_tokens=sum(report.metrics.output_tokens for report in reports),
        ),
        0 if passed else 1,
    )


def run_evaluation_sync(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    selected_case: str | None = None,
) -> tuple[EvalReportV3, EvalExitCode]:
    return asyncio.run(
        run_evaluation(
            dataset_path=dataset_path,
            manifest_path=manifest_path,
            selected_case=selected_case,
        )
    )
