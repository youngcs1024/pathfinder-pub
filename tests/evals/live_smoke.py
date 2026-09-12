from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select

from app.agents.contracts import (
    AgentLoopControl,
    AgentLoopLimitsV1,
    AgentLoopObservationV1,
)
from app.agents.prompting import load_agent_prompt_bundle
from app.agents.research_contracts import (
    EvidenceValidationNodeInputV1,
    PlanNodeInputV1,
    ResearchNodeInputV1,
    ResearchRequestV1,
    WriteReportNodeInputV1,
)
from app.agents.research_graph import ResearchGraphRuntimeContext
from app.agents.research_nodes import (
    CreateAgentResearchNode,
    DeterministicEvidenceValidationNode,
    ResearchAgentNodeError,
    ResearchPlanNodeError,
    ResearchWriterNodeError,
    StructuredResearchPlanNode,
    StructuredResearchWriterNode,
)
from app.config import Settings
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import LLMInvocation
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.readiness import DatabaseReadinessProbe
from app.db.session import AsyncSessionFactory, create_database_engine, create_session_factory
from app.domain.provisioning import ProvisioningService
from app.llm.factory import LLMAccountingError, LLMFactory, LLMRetryPolicy
from app.llm.invocations import (
    InvocationRecorderPort,
    LLMInvocationAttempt,
    LLMInvocationContext,
    LLMInvocationOutcome,
)
from app.llm.ports import (
    LOCKED_EMBEDDING_DIMENSION,
    ChatMessage,
    ChatModelPort,
    ChatModelResult,
    ModelToolSchema,
)
from app.llm.pricing import QWEN_BEIJING_PRICE_BOOK
from app.llm.qwen_adapters import QwenAdapterBundle, create_qwen_adapters
from app.obs.langfuse import LangfuseTraceSink, build_trace_sink
from app.tools.adapters.tavily import TavilyAdapterBundle, create_tavily_adapter
from app.tools.contracts import ToolRunContext
from app.tools.search import SearchPort, SearchResult
from app.tools.web_search import RESEARCH_TOOL_POLICY_NAME, create_search_web_tool_registry
from tests.evals.harness import load_eval_manifest
from tests.evals.live_contracts import LiveSmokeErrorCategory, LiveSmokeReportV1

LIVE_SMOKE_AUTH_SUBJECT = "pathfinder-gate3-step35-live-smoke"
LIVE_SMOKE_COST_LIMIT_CNY = Decimal("1.00")
LIVE_SMOKE_MAX_LOGICAL_CHAT_CALLS = 5
LIVE_SMOKE_EMBEDDING_TEXTS = (
    "Pathfinder live smoke embedding contract alpha.",
    "Pathfinder live smoke embedding contract beta.",
)
_REQUIRED_ENVIRONMENT_VARIABLES = (
    "PF_LLM_MODE",
    "PF_QWEN_WORKSPACE_ID",
    "DASHSCOPE_API_KEY",
    "PF_SEARCH_MODE",
    "TAVILY_API_KEY",
    "PF_TRACE_MODE",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_SAMPLE_RATE",
)
type LiveSmokeExitCode = Literal[0, 1, 2]


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class _CollectingObserver:
    def __init__(self) -> None:
        self.observations: list[AgentLoopObservationV1] = []

    def observe(self, observation: AgentLoopObservationV1) -> None:
        self.observations.append(observation)


class _TrackingRecorder:
    def __init__(self, delegate: InvocationRecorderPort) -> None:
        self._delegate = delegate
        self.attempts: dict[UUID, LLMInvocationAttempt] = {}
        self.outcomes: dict[UUID, LLMInvocationOutcome] = {}

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        await self._delegate.prepare(attempt)
        self.attempts[attempt.invocation_id] = attempt

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        await self._delegate.finalize(attempt, outcome)
        self.outcomes[attempt.invocation_id] = outcome

    def known_cost(self) -> Decimal:
        return sum(
            (outcome.estimated_cost or Decimal(0) for outcome in self.outcomes.values()),
            start=Decimal(0),
        )

    def token_totals(self) -> tuple[int, int]:
        usages = tuple(
            outcome.token_usage
            for outcome in self.outcomes.values()
            if outcome.token_usage is not None
        )
        return (
            sum(usage.input_tokens for usage in usages),
            sum(usage.output_tokens for usage in usages),
        )

    def cost_is_unavailable(self) -> bool:
        return any(
            outcome.status == "succeeded"
            and outcome.token_usage is not None
            and outcome.estimated_cost is None
            and (
                (outcome.token_usage.cached_input_tokens or 0) > 0
                or (outcome.token_usage.cache_write_input_tokens or 0) > 0
            )
            for outcome in self.outcomes.values()
        )


class _LiveBudgetError(Exception):
    def __init__(self, category: Literal["budget_exceeded", "cost_unavailable"]) -> None:
        self.category = category
        super().__init__("live smoke cost budget is unavailable or exhausted")


class _BudgetedChatModel:
    def __init__(self, delegate: ChatModelPort, recorder: _TrackingRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self.logical_call_count = 0

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
    ) -> ChatModelResult:
        if self.logical_call_count >= LIVE_SMOKE_MAX_LOGICAL_CHAT_CALLS:
            raise _LiveBudgetError("budget_exceeded")
        self.logical_call_count += 1
        result = await self._delegate.invoke(messages, tools, metadata)
        _check_cost_budget(self._recorder)
        return result


@dataclass(frozen=True, slots=True, repr=False)
class _OneResultSearch:
    delegate: SearchPort

    async def search(
        self,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[SearchResult, ...]:
        return await self.delegate.search(query, min(max_results, 1), deadline)


def missing_live_environment_variables() -> tuple[str, ...]:
    return tuple(name for name in _REQUIRED_ENVIRONMENT_VARIABLES if not os.environ.get(name))


def _check_cost_budget(recorder: _TrackingRecorder) -> None:
    if recorder.cost_is_unavailable():
        raise _LiveBudgetError("cost_unavailable")
    if recorder.known_cost() >= LIVE_SMOKE_COST_LIMIT_CNY:
        raise _LiveBudgetError("budget_exceeded")


def _live_control() -> AgentLoopControl:
    return AgentLoopControl(
        limits=AgentLoopLimitsV1(
            max_model_calls=2,
            max_tool_calls=1,
            max_tool_results=1,
            max_iterations=4,
        ),
        deadline=monotonic() + 120.0,
        cancellation=_NeverCancelled(),
    )


def _error_category(error: Exception) -> LiveSmokeErrorCategory:
    if isinstance(error, _LiveBudgetError):
        return error.category
    if isinstance(error, LLMAccountingError):
        return "accounting_incomplete"
    if isinstance(error, ResearchAgentNodeError):
        return "tool_failure"
    if isinstance(error, ResearchPlanNodeError | ResearchWriterNodeError):
        if error.category == "invalid_model_output":
            return "invalid_output"
        return "provider_failure"
    return "provider_failure"


async def _database_accounting_status(
    *,
    session_factory: AsyncSessionFactory,
    recorder: _TrackingRecorder,
    workspace_id: UUID,
    actor_user_id: UUID,
) -> tuple[bool, bool]:
    if not recorder.attempts:
        return False, False
    async with session_factory() as session:
        rows = (
            await session.scalars(
                select(LLMInvocation).where(
                    LLMInvocation.id.in_(tuple(recorder.attempts)),
                    LLMInvocation.workspace_id == workspace_id,
                    LLMInvocation.actor_user_id == actor_user_id,
                )
            )
        ).all()
    by_id = {row.id: row for row in rows}
    exact = set(by_id) == set(recorder.attempts) == set(recorder.outcomes)
    terminal = exact and all(
        row.status == recorder.outcomes[invocation_id].status
        and row.status in {"succeeded", "failed"}
        and row.provider == recorder.attempts[invocation_id].provider
        and row.model == recorder.attempts[invocation_id].model
        and row.invocation_kind == recorder.attempts[invocation_id].invocation_kind
        and row.graph_node == recorder.attempts[invocation_id].graph_node
        and row.prompt_version == recorder.attempts[invocation_id].prompt_version
        and row.token_usage
        == (
            recorder.outcomes[invocation_id].token_usage.model_dump(mode="json", round_trip=True)
            if recorder.outcomes[invocation_id].token_usage is not None
            else None
        )
        and row.pricing_version == recorder.outcomes[invocation_id].pricing_version
        and row.currency == recorder.outcomes[invocation_id].currency
        and row.estimated_cost == recorder.outcomes[invocation_id].estimated_cost
        and row.trace_ids
        == (
            recorder.outcomes[invocation_id].trace_ids.model_dump(mode="json")
            if recorder.outcomes[invocation_id].trace_ids is not None
            else None
        )
        for invocation_id, row in by_id.items()
    )
    traces = terminal and all(row.trace_ids is not None for row in rows)
    return terminal, traces


async def run_live_smoke() -> tuple[LiveSmokeReportV1, LiveSmokeExitCode, tuple[str, ...]]:
    missing_variables = missing_live_environment_variables()
    if missing_variables:
        return (
            LiveSmokeReportV1(passed=False, error_category="invalid_settings"),
            2,
            missing_variables,
        )
    try:
        settings = Settings()
    except ValidationError:
        return LiveSmokeReportV1(passed=False, error_category="invalid_settings"), 2, ()
    if (
        settings.llm_mode != "qwen"
        or settings.search_mode != "tavily"
        or settings.trace_mode != "langfuse"
        or settings.langfuse_sample_rate != 1.0
    ):
        return LiveSmokeReportV1(passed=False, error_category="invalid_modes"), 2, ()

    try:
        versions = load_eval_manifest()
    except Exception:
        return LiveSmokeReportV1(passed=False, error_category="invalid_settings"), 2, ()
    recorder: _TrackingRecorder | None = None
    qwen_bundle: QwenAdapterBundle | None = None
    tavily_bundle: TavilyAdapterBundle | None = None
    trace_sink: LangfuseTraceSink | None = None
    budgeted_model: _BudgetedChatModel | None = None
    engine = create_database_engine(settings.database_url)
    error_category: LiveSmokeErrorCategory | None = None
    logical_chat_calls = 0
    embedding_batch_count = 0
    embedding_count = 0
    embedding_dimension: Literal[1536] | None = None
    tavily_result_count = 0
    invocations_terminal = False
    trace_ids_present = False
    workspace_id: UUID | None = None
    actor_user_id: UUID | None = None

    try:
        if not await DatabaseReadinessProbe(engine).is_ready():
            error_category = "invalid_database_revision"
        else:
            session_factory = create_session_factory(engine)
            provisioned = await ProvisioningService(
                SqlAlchemyProvisioningStore(session_factory)
            ).provision_personal_workspace(LIVE_SMOKE_AUTH_SUBJECT)
            workspace_id = provisioned.workspace_id
            actor_user_id = provisioned.user_id
            recorder = _TrackingRecorder(SqlAlchemyInvocationRecorder(session_factory))
            trace_candidate = build_trace_sink(settings)
            if not isinstance(trace_candidate, LangfuseTraceSink):
                error_category = "trace_unavailable"
            else:
                trace_sink = trace_candidate
                assert settings.qwen_api_key is not None
                assert settings.qwen_workspace_id is not None
                assert settings.tavily_api_key is not None
                qwen_bundle = create_qwen_adapters(
                    api_key=settings.qwen_api_key,
                    workspace_id=settings.qwen_workspace_id,
                )
                tavily_bundle = create_tavily_adapter(api_key=settings.tavily_api_key)
                context = LLMInvocationContext(
                    workspace_id=workspace_id,
                    actor_user_id=actor_user_id,
                    request_id=uuid4(),
                    run_id=None,
                )
                factory = LLMFactory(
                    recorder=recorder,
                    chat_adapter=qwen_bundle.chat,
                    embedding_adapter=qwen_bundle.embedding,
                    trace_sink=trace_sink,
                    provider="qwen",
                    price_book=QWEN_BEIJING_PRICE_BOOK,
                    retry_policy=LLMRetryPolicy(max_attempts=3),
                )
                model = _BudgetedChatModel(factory.create_chat_model(context), recorder)
                budgeted_model = model
                embedding_model = factory.create_embedding_model(context)
                control = _live_control()

                base_prompt = load_agent_prompt_bundle("base")
                basic = await model.invoke(
                    (
                        ChatMessage(role="system", content=base_prompt.system_prompt),
                        ChatMessage(
                            role="user",
                            content=(
                                "Reply with one short sentence confirming this synthetic smoke "
                                "check."
                            ),
                        ),
                    ),
                    (),
                    {
                        "graph_node": "live_smoke_basic",
                        "prompt_version": base_prompt.version,
                    },
                )
                if basic.content is None or basic.tool_calls:
                    raise ResearchWriterNodeError(category="invalid_model_output")

                request = ResearchRequestV1(
                    query=(
                        "Find one current public source describing Python backend developer "
                        "responsibilities; create exactly one narrow search query."
                    ),
                    include_application_draft=False,
                )
                plan = await StructuredResearchPlanNode(model)(
                    PlanNodeInputV1(
                        normalized_query=request.query,
                        include_application_draft=False,
                    ),
                    control,
                )
                registry = create_search_web_tool_registry(_OneResultSearch(tavily_bundle.search))
                tool_runtime = registry.bind(
                    policy_name=RESEARCH_TOOL_POLICY_NAME,
                    context=ToolRunContext(
                        workspace_id=workspace_id,
                        actor_user_id=actor_user_id,
                        run_id=uuid4(),
                        action_intent_id=None,
                        approval_request_id=None,
                        trusted_target={"kind": "gate_3_live_smoke"},
                        deadline=control.deadline,
                        cancellation=control.cancellation,
                    ),
                )
                research = await CreateAgentResearchNode(model, _CollectingObserver())(
                    ResearchNodeInputV1(
                        request=request,
                        normalized_query=request.query,
                        plan=plan.plan,
                        research_pass_number=1,
                    ),
                    ResearchGraphRuntimeContext(
                        tool_runtime=tool_runtime,
                        agent_loop_control=control,
                    ),
                )
                tavily_result_count = sum(call.result_count for call in research.search_calls)
                if len(research.search_calls) != 1 or tavily_result_count != 1:
                    raise ResearchAgentNodeError(category="invalid_tool_trace")

                validation = await DeterministicEvidenceValidationNode()(
                    EvidenceValidationNodeInputV1(
                        request=request,
                        plan=plan.plan,
                        sources=research.sources,
                        evidence=research.evidence,
                        research_pass_count=1,
                    )
                )
                writer = await StructuredResearchWriterNode(model)(
                    WriteReportNodeInputV1(
                        request=request,
                        evidence=research.evidence,
                        evidence_sufficient=validation.evidence_sufficient,
                        validation_limitations=validation.limitations,
                    ),
                    control,
                )
                if not writer.summary and not writer.findings:
                    raise ResearchWriterNodeError(category="invalid_model_output")

                embedding = await embedding_model.embed(
                    LIVE_SMOKE_EMBEDDING_TEXTS,
                    {"graph_node": "live_smoke_embedding"},
                )
                embedding_batch_count = 1
                embedding_count = len(embedding.vectors)
                if embedding_count != len(LIVE_SMOKE_EMBEDDING_TEXTS) or any(
                    len(vector) != LOCKED_EMBEDDING_DIMENSION for vector in embedding.vectors
                ):
                    raise ValueError("embedding contract failed")
                embedding_dimension = LOCKED_EMBEDDING_DIMENSION
                _check_cost_budget(recorder)
                logical_chat_calls = model.logical_call_count
                if logical_chat_calls != LIVE_SMOKE_MAX_LOGICAL_CHAT_CALLS:
                    raise _LiveBudgetError("budget_exceeded")
    except Exception as error:
        if recorder is not None and recorder.cost_is_unavailable():
            error_category = "cost_unavailable"
        elif recorder is not None and recorder.known_cost() >= LIVE_SMOKE_COST_LIMIT_CNY:
            error_category = "budget_exceeded"
        else:
            error_category = error_category or _error_category(error)

    finally:
        if recorder is not None:
            if budgeted_model is not None:
                logical_chat_calls = budgeted_model.logical_call_count
            if workspace_id is not None and actor_user_id is not None:
                try:
                    invocations_terminal, trace_ids_present = await _database_accounting_status(
                        session_factory=create_session_factory(engine),
                        recorder=recorder,
                        workspace_id=workspace_id,
                        actor_user_id=actor_user_id,
                    )
                except Exception:
                    invocations_terminal = False
                    trace_ids_present = False
                if error_category is None and not invocations_terminal:
                    error_category = "accounting_incomplete"
                elif error_category is None and not trace_ids_present:
                    error_category = "trace_unavailable"

        cleanup_failed = False
        if trace_sink is not None:
            if not trace_sink.flush_and_check() and error_category is None:
                error_category = "trace_unavailable"
            try:
                trace_sink.shutdown()
            except Exception:
                cleanup_failed = True
        if qwen_bundle is not None:
            try:
                await qwen_bundle.aclose()
            except Exception:
                cleanup_failed = True
        if tavily_bundle is not None:
            try:
                await tavily_bundle.aclose()
            except Exception:
                cleanup_failed = True
        try:
            await engine.dispose()
        except Exception:
            cleanup_failed = True
        if cleanup_failed and error_category is None:
            error_category = "cleanup_failed"

    provider_attempts = len(recorder.attempts) if recorder is not None else 0
    input_tokens, output_tokens = recorder.token_totals() if recorder is not None else (0, 0)
    known_cost = recorder.known_cost() if recorder is not None else Decimal(0)
    report = LiveSmokeReportV1(
        passed=error_category is None,
        error_category=error_category,
        versions=versions,
        logical_chat_calls=logical_chat_calls,
        provider_attempts=provider_attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        known_cost_cny=known_cost,
        embedding_batch_count=embedding_batch_count,
        embedding_count=embedding_count,
        embedding_dimension=embedding_dimension,
        tavily_result_count=tavily_result_count,
        invocations_terminal=invocations_terminal,
        trace_ids_present=trace_ids_present,
    )
    exit_code: LiveSmokeExitCode = (
        0
        if report.passed
        else (
            2
            if error_category in {"invalid_settings", "invalid_modes", "invalid_database_revision"}
            else 1
        )
    )
    return report, exit_code, ()


def run_live_smoke_sync() -> tuple[LiveSmokeReportV1, LiveSmokeExitCode, tuple[str, ...]]:
    return asyncio.run(run_live_smoke())
