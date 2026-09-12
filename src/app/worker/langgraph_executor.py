from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from time import monotonic
from uuid import UUID

from pydantic import ValidationError

from app.agents.contracts import AgentLoopControl, AgentLoopLimitsV1
from app.agents.research_contracts import (
    ResearchGraphInputV1,
    ResearchGraphStateV1,
    ResearchNodeOutputV1,
    ResearchOutputV2,
)
from app.agents.research_graph import (
    ResearchGraphNodes,
    ResearchGraphProtocolError,
    ResearchGraphRuntimeContext,
    approval_resume_was_applied,
    build_approval_resume_input,
    build_research_state_graph,
)
from app.agents.research_nodes import (
    CreateAgentResearchNode,
    DeterministicEvidenceValidationNode,
    StructuredResearchPlanNode,
    StructuredResearchWriterNode,
)
from app.domain.action_execution import (
    ActionExecutionFailedError,
    ActionOutcomeUnknownError,
    ActionResultUnconfirmedError,
)
from app.domain.actions import ACTION_KEY, ActionStore
from app.domain.approvals import ApprovalResumeResolver, ApprovalStatus
from app.domain.errors import DomainUnavailableError
from app.domain.research import normalize_research_query
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInput,
    RunExecutionInvalidError,
    RunExecutionReader,
)
from app.domain.runs import CURRENT_GRAPH_VERSION, RunStatus
from app.domain.tenancy import TenantContext
from app.domain.tool_invocations import ToolInvocationRecorderPort
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.invocations import LLMInvocationContext
from app.obs.agent_loop import StructlogAgentLoopObserver
from app.obs.logging import get_logger
from app.retrieval.documents import (
    DocumentRetrievalRepositoryPort,
    DocumentRetrievalService,
)
from app.tools.contracts import ApprovedActionExecutor, CancellationCheck, ToolRunContext
from app.tools.document_retrieval import (
    RetrievalEventRecorderPort,
    create_research_tool_registry,
)
from app.tools.search import SearchPort
from app.tools.web_search import RESEARCH_TOOL_POLICY_NAME
from app.worker.checkpoints import CheckpointHandle
from app.worker.contracts import RunExecutionResult


@dataclass(slots=True)
class _LocalCancellation(CancellationCheck):
    cancelled: bool = False

    def is_cancelled(self) -> bool:
        return self.cancelled


class _ExecutionGuardUnavailableError(Exception):
    """The persisted execution permission could not be checked safely."""


class LangGraphRunExecutor:
    def __init__(
        self,
        *,
        reader: RunExecutionReader,
        checkpointer: CheckpointHandle,
        llm_factory: LLMFactory,
        search_port: SearchPort,
        tool_recorder: ToolInvocationRecorderPort,
        document_repository: DocumentRetrievalRepositoryPort,
        retrieval_event_recorder: RetrievalEventRecorderPort,
        execution_timeout_seconds: float,
        after_research_node: Callable[[ResearchNodeOutputV1], Awaitable[None]] | None = None,
        action_store: ActionStore | None = None,
        approval_resume_resolver: ApprovalResumeResolver | None = None,
        approved_action_executor: ApprovedActionExecutor | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        execution_guard_poll_seconds: float = 0.25,
        checkpoint_error_is_unavailable: Callable[[Exception], bool] = lambda _error: False,
    ) -> None:
        if (
            isinstance(execution_timeout_seconds, bool)
            or not isinstance(execution_timeout_seconds, int | float)
            or not isfinite(float(execution_timeout_seconds))
            or execution_timeout_seconds <= 0
        ):
            raise ValueError("execution timeout must be a finite positive number")
        if execution_guard_poll_seconds <= 0:
            raise ValueError("execution guard poll interval must be positive")
        self._checkpoint_error_is_unavailable = checkpoint_error_is_unavailable
        self._reader = reader
        self._checkpointer = checkpointer
        self._llm_factory = llm_factory
        self._search_port = search_port
        self._tool_recorder = tool_recorder
        self._document_repository = document_repository
        self._retrieval_event_recorder = retrieval_event_recorder
        self._after_research_node = after_research_node
        self._action_store = action_store
        self._approval_resume_resolver = approval_resume_resolver
        self._approved_action_executor = approved_action_executor
        self._clock = clock
        self._execution_timeout_seconds = float(execution_timeout_seconds)
        self._execution_guard_poll_seconds = execution_guard_poll_seconds
        self._logger = get_logger("app.worker.langgraph_executor")

    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        await self._wait_if_checkpoint_broken()
        result = await self._execute(run_id, tenant, graph_version)
        await self._wait_if_checkpoint_broken()
        return result

    async def _wait_if_checkpoint_broken(self) -> None:
        connection = getattr(self._checkpointer, "conn", None)
        if (
            getattr(connection, "closed", False) is True
            or getattr(connection, "broken", False) is True
        ):
            # The existing root supervisor stops the worker and cancels this task.
            # Do not finalize the job or reuse a broken independent connection.
            await asyncio.Future()

    async def _execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        if graph_version != CURRENT_GRAPH_VERSION:
            return self._failed("unknown_graph_version", retryable=False)
        try:
            execution = await self._reader.read_for_execution(
                run_id=run_id,
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                graph_version=graph_version,
            )
            return await self._execute_guarded_graph(execution)
        except asyncio.CancelledError:
            raise
        except RunExecutionCancelledError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except RunExecutionInvalidError as error:
            return self._failed(error.category, retryable=False)
        except _ExecutionGuardUnavailableError:
            return self._failed("execution_guard_unavailable", retryable=True)
        except DomainUnavailableError:
            return self._failed("database_unavailable", retryable=True)
        except LLMAccountingError as error:
            return self._failed("llm_accounting_failed", retryable=error.retryable)
        except LLMProviderError:
            return self._failed("provider_unavailable", retryable=True)
        except ActionResultUnconfirmedError:
            return self._failed("external_action_unconfirmed", retryable=True)
        except ActionOutcomeUnknownError:
            return self._failed("external_outcome_unknown", retryable=False)
        except ActionExecutionFailedError:
            return self._failed("external_action_failed", retryable=False)
        except ResearchGraphProtocolError as error:
            diagnostic = {
                "run_id": str(run_id),
                "workspace_id": str(tenant.workspace_id),
                "graph_version": graph_version,
                "error_category": error.category,
                "node_name": error.node_name,
                "cause_category": error.cause_category,
            }
            if error.schema_error_type is not None:
                diagnostic["schema_error_type"] = error.schema_error_type
            if error.schema_error_path is not None:
                diagnostic["schema_error_path"] = error.schema_error_path
            if error.limit_kind is not None:
                diagnostic["limit_kind"] = error.limit_kind
            if error.limit is not None:
                diagnostic["limit"] = error.limit
            if error.current_count is not None:
                diagnostic["current_count"] = error.current_count
            if error.requested_count is not None:
                diagnostic["requested_count"] = error.requested_count
            self._logger.warning("langgraph_protocol_failed", **diagnostic)
            if error.cause_category == "cancelled":
                return RunExecutionResult(status=RunStatus.CANCELLED)
            if error.cause_category in {"provider_timeout", "provider_unavailable"}:
                return self._failed(error.cause_category, retryable=True)
            if error.cause_category in {
                "deadline_exceeded",
                "model_output_incomplete",
            } and error.node_name in {
                "plan",
                "write_report",
            }:
                return self._failed(error.category, retryable=True)
            return self._failed(error.category, retryable=False)
        except (ValidationError, TypeError, ValueError):
            return self._failed("invalid_checkpoint_state", retryable=False)
        except Exception as error:
            if self._checkpoint_error_is_unavailable(error):
                return self._failed("checkpoint_unavailable", retryable=True)
            return self._failed("executor_unhandled_error", retryable=False)

    async def _execute_graph(
        self,
        execution: RunExecutionInput,
        cancellation: _LocalCancellation,
    ) -> RunExecutionResult:
        deadline = monotonic() + self._execution_timeout_seconds
        agent_limits = AgentLoopLimitsV1.model_validate(
            execution.limits.model_dump(mode="python"),
            strict=True,
        )
        model = self._llm_factory.create_chat_model(
            LLMInvocationContext(
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                run_id=execution.run_id,
            )
        )
        embedding = self._llm_factory.create_embedding_model(
            LLMInvocationContext(
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                run_id=execution.run_id,
            )
        )
        registry = create_research_tool_registry(
            search_port=self._search_port,
            retrieval_service=DocumentRetrievalService(
                repository=self._document_repository,
                embedding=embedding,
            ),
            tenant=TenantContext(
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                role=execution.role,
            ),
            allowed_document_ids=(
                (execution.resume_document_id,) if execution.resume_document_id is not None else ()
            ),
            recorder=self._tool_recorder,
            event_recorder=self._retrieval_event_recorder,
        )
        tool_runtime = registry.bind(
            policy_name=RESEARCH_TOOL_POLICY_NAME,
            context=ToolRunContext(
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                run_id=execution.run_id,
                action_intent_id=None,
                approval_request_id=None,
                trusted_target=None,
                deadline=deadline,
                cancellation=cancellation,
            ),
        )
        graph = build_research_state_graph(
            ResearchGraphNodes(
                plan=StructuredResearchPlanNode(model),
                research_agent=CreateAgentResearchNode(
                    model,
                    StructlogAgentLoopObserver(),
                ),
                validate_evidence=DeterministicEvidenceValidationNode(),
                write_report=StructuredResearchWriterNode(model),
            ),
            checkpointer=self._checkpointer,
        )
        config = {
            "configurable": {"thread_id": str(execution.run_id)},
            "recursion_limit": execution.limits.max_iterations,
        }
        snapshot = await graph.aget_state(config)
        graph_input: object
        checkpoint_tool_calls = 0
        resumed_state: ResearchGraphStateV1 | None = None
        if snapshot.values:
            state = self._checkpoint_state(snapshot.values)
            self._require_matching_identity(state, execution)
            checkpoint_tool_calls = len(state.search_calls) + len(state.document_retrieval_calls)
            if snapshot.interrupts:
                persisted_request_id = self._validate_interrupt_identity(
                    snapshot.interrupts,
                    state,
                )
                resume_request_id = execution.resume_approval_request_id
                if resume_request_id is None:
                    return self._waiting_result(persisted_request_id)
                if resume_request_id != persisted_request_id:
                    raise RunExecutionInvalidError("resume_approval_request_mismatch")
                checkpoint_tuple = await self._checkpoint_tuple(config)
                if approval_resume_was_applied(checkpoint_tuple):
                    if self._approval_resume_resolver is None:
                        raise RunExecutionInvalidError("approval_resume_resolver_missing")
                    resolved = await self._approval_resume_resolver.resolve_approval_resume(
                        tenant=TenantContext(
                            workspace_id=execution.workspace_id,
                            actor_user_id=execution.actor_user_id,
                            role=execution.role,
                        ),
                        run_id=execution.run_id,
                        action_intent_id=state.action_proposal_id,
                        approval_request_id=persisted_request_id,
                    )
                    if resolved.approval_request.status is ApprovalStatus.PENDING:
                        return self._waiting_result(persisted_request_id)
                    if resolved.approval_request.status not in {
                        ApprovalStatus.APPROVED,
                        ApprovalStatus.REJECTED,
                        ApprovalStatus.EXPIRED,
                    }:
                        raise RunExecutionInvalidError("approval_resume_status_invalid")
                    graph_input = None
                else:
                    graph_input = build_approval_resume_input(persisted_request_id)
                resumed_state = state
            elif not snapshot.next:
                if state.output is None:
                    raise ValueError("completed checkpoint has no output")
                return RunExecutionResult(status=RunStatus.COMPLETED, result=state.output)
            else:
                graph_input = None
        else:
            graph_input = ResearchGraphInputV1(
                schema_version=3,
                run_id=execution.run_id,
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                conversation_id=execution.conversation_id,
                graph_version=execution.graph_version,
                mode=execution.mode.value,
                resume_document_id=execution.resume_document_id,
                request=execution.request,
            ).model_dump(mode="json", round_trip=True)

        persisted_tool_calls = await self._tool_recorder.consumed_call_count(
            workspace_id=execution.workspace_id,
            run_id=execution.run_id,
        )
        if persisted_tool_calls < checkpoint_tool_calls:
            raise RunExecutionInvalidError("checkpoint_tool_count_mismatch")
        # The research pass still subtracts checkpoint traces from these limits.
        uncheckpointed_tool_calls = persisted_tool_calls - checkpoint_tool_calls
        effective_max_tool_calls = max(0, agent_limits.max_tool_calls - uncheckpointed_tool_calls)
        agent_limits = agent_limits.model_copy(
            update={
                "max_tool_calls": effective_max_tool_calls,
                "max_tool_results": min(agent_limits.max_tool_results, effective_max_tool_calls),
            }
        )
        runtime_context = ResearchGraphRuntimeContext(
            tool_runtime=tool_runtime,
            agent_loop_control=AgentLoopControl(
                limits=agent_limits,
                deadline=deadline,
                cancellation=cancellation,
            ),
            after_research_node=self._after_research_node,
            tenant=TenantContext(
                workspace_id=execution.workspace_id,
                actor_user_id=execution.actor_user_id,
                role=execution.role,
            ),
            action_store=self._action_store,
            approval_resume_resolver=self._approval_resume_resolver,
            approved_action_executor=self._approved_action_executor,
            clock=self._clock,
        )

        raw_output = await graph.ainvoke(
            graph_input,
            config=config,
            context=runtime_context,
            durability="sync",
        )
        raw_interrupts = raw_output.get("__interrupt__")
        if raw_interrupts:
            request_id = self._validate_interrupt_identity(
                tuple(raw_interrupts),
                resumed_state,
            )
            if (
                execution.resume_approval_request_id is not None
                and execution.resume_approval_request_id != request_id
            ):
                raise RunExecutionInvalidError("resume_approval_request_mismatch")
            return self._waiting_result(request_id)
        output = ResearchOutputV2.model_validate_json(
            json.dumps(raw_output["output"], allow_nan=False, separators=(",", ":")),
            strict=True,
        )
        return RunExecutionResult(status=RunStatus.COMPLETED, result=output)

    async def _checkpoint_tuple(self, config: object) -> object:
        get_tuple = getattr(self._checkpointer, "aget_tuple", None)
        if not callable(get_tuple):
            raise TypeError("checkpointer does not support tuple inspection")
        checkpoint_tuple = await get_tuple(config)
        if checkpoint_tuple is None:
            raise ValueError("persisted interrupt checkpoint is missing")
        return checkpoint_tuple

    async def _execute_guarded_graph(
        self,
        execution: RunExecutionInput,
    ) -> RunExecutionResult:
        cancellation = _LocalCancellation()
        graph_task = asyncio.create_task(self._execute_graph(execution, cancellation))
        guard_task = asyncio.create_task(self._watch_execution_permission(execution, cancellation))
        try:
            done, _pending = await asyncio.wait(
                {graph_task, guard_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if guard_task in done:
                try:
                    guard_task.result()
                except RunExecutionCancelledError:
                    cancellation.cancelled = True
                    await self._cancel_helper_task(graph_task)
                    return RunExecutionResult(status=RunStatus.CANCELLED)
                except RunExecutionInvalidError:
                    await self._cancel_helper_task(graph_task)
                    raise
                except asyncio.CancelledError:
                    raise
                except DomainUnavailableError:
                    await self._cancel_helper_task(graph_task)
                    raise _ExecutionGuardUnavailableError from None
                except Exception:
                    await self._cancel_helper_task(graph_task)
                    raise
                raise RuntimeError("execution guard stopped without a terminal result")
            return graph_task.result()
        finally:
            await self._cancel_helper_task(guard_task)
            await self._cancel_helper_task(graph_task)

    async def _watch_execution_permission(
        self,
        execution: RunExecutionInput,
        cancellation: _LocalCancellation,
    ) -> None:
        while True:
            try:
                await self._reader.assert_execution_allowed(
                    run_id=execution.run_id,
                    workspace_id=execution.workspace_id,
                    actor_user_id=execution.actor_user_id,
                    graph_version=execution.graph_version,
                )
            except RunExecutionCancelledError:
                cancellation.cancelled = True
                raise
            await asyncio.sleep(self._execution_guard_poll_seconds)

    @staticmethod
    async def _cancel_helper_task(task: asyncio.Task[object]) -> None:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _checkpoint_state(values: object) -> ResearchGraphStateV1:
        return ResearchGraphStateV1.model_validate_json(
            json.dumps(values, allow_nan=False, separators=(",", ":"), sort_keys=True),
            strict=True,
        )

    @staticmethod
    def _require_matching_identity(
        state: ResearchGraphStateV1,
        execution: RunExecutionInput,
    ) -> None:
        if (
            state.schema_version != 3
            or state.run_id != execution.run_id
            or state.workspace_id != execution.workspace_id
            or state.actor_user_id != execution.actor_user_id
            or state.conversation_id != execution.conversation_id
            or state.graph_version != execution.graph_version
            or state.mode != execution.mode.value
            or state.resume_document_id != execution.resume_document_id
            or state.request.include_application_draft
            != execution.request.include_application_draft
            or state.request.query != normalize_research_query(execution.request.query)
        ):
            raise RunExecutionInvalidError("checkpoint_identity_mismatch")

    @staticmethod
    def _validate_interrupt_identity(
        interrupts: tuple[object, ...],
        state: ResearchGraphStateV1 | None,
    ) -> UUID:
        if len(interrupts) != 1:
            raise RunExecutionInvalidError("checkpoint_identity_mismatch")
        value = getattr(interrupts[0], "value", None)
        if not isinstance(value, dict) or set(value) != {
            "version",
            "approval_request_id",
            "action_intent_id",
            "action_key",
            "action_revision",
        }:
            raise RunExecutionInvalidError("checkpoint_identity_mismatch")
        try:
            request_id = UUID(value["approval_request_id"])
            action_intent_id = UUID(value["action_intent_id"])
        except (TypeError, ValueError):
            raise RunExecutionInvalidError("checkpoint_identity_mismatch") from None
        if state is not None:
            if state.approval_request_id is None:
                raise RunExecutionInvalidError("checkpoint_identity_mismatch")
            expected = {
                "version": 1,
                "approval_request_id": str(state.approval_request_id),
                "action_intent_id": str(state.action_proposal_id),
                "action_key": state.action_key,
                "action_revision": state.action_revision,
            }
            if value != expected:
                raise RunExecutionInvalidError("checkpoint_identity_mismatch")
        elif (
            type(value["version"]) is not int
            or value["version"] != 1
            or value["action_key"] != ACTION_KEY
            or type(value["action_revision"]) is not int
            or value["action_revision"] != 1
            or action_intent_id.int == 0
        ):
            raise RunExecutionInvalidError("checkpoint_identity_mismatch")
        return request_id

    @staticmethod
    def _waiting_result(request_id: UUID) -> RunExecutionResult:
        return RunExecutionResult(
            status=RunStatus.WAITING_APPROVAL,
            approval_request_id=request_id,
        )

    @staticmethod
    def _failed(category: str, *, retryable: bool) -> RunExecutionResult:
        return RunExecutionResult(
            status=RunStatus.FAILED,
            error_category=category,
            retryable=retryable,
        )
