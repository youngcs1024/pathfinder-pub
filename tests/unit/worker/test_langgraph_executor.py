from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.research_graph import ResearchGraphProtocolError, approval_resume_was_applied
from app.db.checkpoints import create_checkpoint_serializer
from app.domain.approvals import ApprovalStatus
from app.domain.errors import DomainUnavailableError
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInput,
    RunExecutionLimitsV1,
)
from app.domain.runs import CURRENT_GRAPH_VERSION, RunMode, RunStatus
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.obs.logging import configure_logging
from app.retrieval.documents import RetrievedDocumentChunk
from app.tools.fake_search import FakeSearch
from app.tools.invocations import InMemoryToolInvocationRecorder
from app.tools.registry import ToolCancelledError
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor


class _RunReader:
    def __init__(self, execution: RunExecutionInput) -> None:
        self.execution = execution
        self.guard_error: Exception | None = None
        self.guard_checked = asyncio.Event()

    async def read_for_execution(self, **_kwargs: object) -> RunExecutionInput:
        return self.execution

    async def assert_execution_allowed(self, **_kwargs: object) -> None:
        self.guard_checked.set()
        if self.guard_error is not None:
            raise self.guard_error

    async def has_nonterminal_other_graph_versions(self, _graph_version: str) -> bool:
        return False


class _InvocationRecorder:
    def __init__(self) -> None:
        self.prepared_graph_nodes: list[str] = []
        self.outcomes: list[object] = []

    async def prepare(self, attempt: object) -> None:
        self.prepared_graph_nodes.append(attempt.graph_node)

    async def finalize(self, _attempt: object, outcome: object) -> None:
        self.outcomes.append(outcome)


class _DocumentRepository:
    async def search(self, **kwargs: object) -> tuple[RetrievedDocumentChunk, ...]:
        allowed = kwargs.get("allowed_document_ids")
        if not isinstance(allowed, tuple) or not allowed:
            return ()
        return (
            RetrievedDocumentChunk(
                document_id=allowed[0],
                chunk_id=uuid4(),
                source_name="resume.txt",
                section="Experience",
                ordinal=0,
                cosine_distance=0.1,
                text="Synthetic backend engineering experience.",
            ),
        )


class _ActionStore:
    def __init__(self) -> None:
        self.commands = []
        self.request_id = uuid4()
        self.cancel_commands = []

    async def prepare_action(self, command):
        self.commands.append(command)
        return SimpleNamespace(
            intent=SimpleNamespace(
                action_intent_id=command.action_proposal_id,
                workspace_id=command.tenant.workspace_id,
                run_id=command.run_id,
                action_key=command.action_key,
                action_revision=command.action_revision,
            ),
            approval_request=SimpleNamespace(
                request_id=self.request_id,
                workspace_id=command.tenant.workspace_id,
                run_id=command.run_id,
                action_intent_id=command.action_proposal_id,
                expires_at=command.expires_at,
                approval_binding_version=1,
            ),
        )

    async def cancel_action(self, command):
        self.cancel_commands.append(command)
        return SimpleNamespace(changed=len(self.cancel_commands) == 1)


class _ApprovalResumeResolver:
    def __init__(self, status=ApprovalStatus.PENDING) -> None:
        self.status = status

    async def resolve_approval_resume(self, **kwargs):
        return SimpleNamespace(
            action_intent_id=kwargs["action_intent_id"],
            decision={ApprovalStatus.APPROVED: "approve", ApprovalStatus.REJECTED: "reject"}.get(
                self.status
            ),
            approval_request=SimpleNamespace(
                status=self.status,
                request_id=kwargs["approval_request_id"],
                action_intent_id=kwargs["action_intent_id"],
                approval_binding_version=1,
            ),
        )


class _ApprovedActionExecutor:
    def __init__(self) -> None:
        self.identities = []

    async def execute_approved_action(self, identity, **_kwargs):
        self.identities.append(identity)
        return '{"external_ref":"mock-submission:test"}'


class _RetrievalEventRecorder:
    async def record_retrieved(self, **_kwargs: object) -> None:
        return None


class _BlockingChatAdapter(DeterministicResearchFakeChatAdapter):
    def __init__(self, *, block_graph_node: str | None = None) -> None:
        self.block_graph_node = block_graph_node
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def invoke(self, messages, tools, metadata, *, attempt=None):
        if (
            self.block_graph_node is not None
            and metadata.get("graph_node") != self.block_graph_node
        ):
            return await super().invoke(messages, tools, metadata, attempt=attempt)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _MissingApplicationDraftAdapter(DeterministicResearchFakeChatAdapter):
    def __init__(self) -> None:
        self.writer_calls = 0

    async def invoke(self, messages, tools, metadata, *, attempt=None):
        if metadata.get("graph_node") != "write_report":
            return await super().invoke(messages, tools, metadata, attempt=attempt)
        result = await super().invoke(messages[:2], tools, metadata, attempt=attempt)
        self.writer_calls += 1
        payload = json.loads(result.content or "{}")
        payload["application_draft"] = None
        return result.model_copy(
            update={"content": json.dumps(payload, separators=(",", ":"), sort_keys=True)}
        )


def _execution_input() -> RunExecutionInput:
    return RunExecutionInput(
        run_id=uuid4(),
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        conversation_id=uuid4(),
        graph_version=CURRENT_GRAPH_VERSION,
        request=ResearchRequestV1(query="Research a deterministic fake role"),
        limits=RunExecutionLimitsV1(
            max_model_calls=12,
            max_tool_calls=8,
            max_tool_results=8,
            max_iterations=24,
        ),
    )


def test_approval_resume_inspection_rejects_malformed_pending_writes() -> None:
    with pytest.raises(TypeError, match="has no pending writes"):
        approval_resume_was_applied(object())
    with pytest.raises(TypeError, match="pending writes"):
        approval_resume_was_applied(SimpleNamespace(pending_writes=()))
    with pytest.raises(TypeError, match="pending write"):
        approval_resume_was_applied(SimpleNamespace(pending_writes=[("task", "__resume__")]))


def _tool_recorder(checkpointer):
    if not hasattr(checkpointer, "test_tool_recorder"):
        checkpointer.test_tool_recorder = InMemoryToolInvocationRecorder()
    return checkpointer.test_tool_recorder


def _executor(
    execution: RunExecutionInput,
    checkpointer: InMemorySaver,
    recorder: _InvocationRecorder,
    action_store: _ActionStore | None = None,
    approval_status=ApprovalStatus.PENDING,
    execution_timeout_seconds: float = 300.0,
) -> LangGraphRunExecutor:
    resolver = (
        _ApprovalResumeResolver(approval_status)
        if isinstance(approval_status, ApprovalStatus)
        else approval_status
    )
    approved_action_executor = _ApprovedActionExecutor()
    executor = LangGraphRunExecutor(
        reader=_RunReader(execution),
        checkpointer=checkpointer,
        llm_factory=LLMFactory(
            recorder=recorder,
            chat_adapter=DeterministicResearchFakeChatAdapter(),
            embedding_adapter=FakeEmbeddingModel(),
        ),
        search_port=FakeSearch({}),
        tool_recorder=_tool_recorder(checkpointer),
        document_repository=_DocumentRepository(),
        retrieval_event_recorder=_RetrievalEventRecorder(),
        action_store=action_store,
        approval_resume_resolver=resolver,
        approved_action_executor=approved_action_executor,
        execution_timeout_seconds=execution_timeout_seconds,
        clock=lambda: datetime(2026, 8, 24, 12, tzinfo=UTC),
    )
    executor.approved_action_executor = approved_action_executor
    return executor


async def test_tool_cancellation_returns_cancelled_before_watchdog(monkeypatch) -> None:
    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())

    async def cancelled(_execution):
        raise ToolCancelledError("private cancellation detail")

    monkeypatch.setattr(executor, "_execute_guarded_graph", cancelled)
    tenant = TenantContext(execution.workspace_id, execution.actor_user_id, execution.role)
    result = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    assert result.status is RunStatus.CANCELLED
    assert result.error_category is None and result.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_name", "retryable"),
    [
        ("plan", True),
        ("write_report", True),
        ("research_agent", False),
        ("validate_evidence", False),
    ],
)
async def test_protocol_failure_log_contains_only_safe_diagnostic_metadata(
    monkeypatch: pytest.MonkeyPatch,
    node_name: str,
    retryable: bool,
) -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())
    protocol_error = ResearchGraphProtocolError(
        category="node_execution_failed",
        node_name=node_name,
        cause_category="model_output_incomplete",
    )
    protocol_error.__cause__ = RuntimeError(
        "secret-canary resume-secret-canary model-output-secret-canary evidence-token-canary"
    )

    async def fail_graph(_execution: RunExecutionInput) -> RunExecutionInput:
        raise protocol_error

    monkeypatch.setattr(executor, "_execute_guarded_graph", fail_graph)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    result = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    event = json.loads(stream.getvalue())
    assert result.status is RunStatus.FAILED
    assert result.error_category == "node_execution_failed"
    assert result.retryable is retryable
    assert event["event"] == "langgraph_protocol_failed"
    assert event["run_id"] == str(execution.run_id)
    assert event["workspace_id"] == str(execution.workspace_id)
    assert event["graph_version"] == CURRENT_GRAPH_VERSION
    assert event["error_category"] == "node_execution_failed"
    assert event["node_name"] == node_name
    assert event["cause_category"] == "model_output_incomplete"
    for canary in (
        "secret-canary",
        "resume-secret-canary",
        "model-output-secret-canary",
        "evidence-token-canary",
    ):
        assert canary not in stream.getvalue()


@pytest.mark.asyncio
async def test_agent_limit_log_is_nonretryable_and_contains_only_safe_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())
    protocol_error = ResearchGraphProtocolError(
        category="node_execution_failed",
        node_name="research_agent",
        cause_category="agent_limit_exceeded",
        limit_kind="model_calls",
        limit=12,
        current_count=12,
        requested_count=1,
    )
    protocol_error.__cause__ = RuntimeError("limit-error-content-canary")

    async def fail_graph(_execution: RunExecutionInput) -> RunExecutionInput:
        raise protocol_error

    monkeypatch.setattr(executor, "_execute_guarded_graph", fail_graph)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    result = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    event = json.loads(stream.getvalue())
    assert result.status is RunStatus.FAILED
    assert result.error_category == "node_execution_failed"
    assert result.retryable is False
    assert event["event"] == "langgraph_protocol_failed"
    assert event["node_name"] == "research_agent"
    assert event["cause_category"] == "agent_limit_exceeded"
    assert event["limit_kind"] == "model_calls"
    assert event["limit"] == 12
    assert event["current_count"] == 12
    assert event["requested_count"] == 1
    assert "limit-error-content-canary" not in stream.getvalue()


@pytest.mark.asyncio
async def test_invalid_schema_log_exposes_only_bounded_sanitized_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())
    protocol_error = ResearchGraphProtocolError(
        category="node_execution_failed",
        node_name="write_report",
        cause_category="invalid_model_schema",
        schema_error_type="extra_forbidden",
        schema_error_path="application_draft.paragraphs.0.citations.0.source_type",
    )
    protocol_error.__cause__ = RuntimeError(
        "resume-secret-canary model-output-secret-canary application-secret-canary"
    )

    async def fail_graph(_execution: RunExecutionInput) -> RunExecutionInput:
        raise protocol_error

    monkeypatch.setattr(executor, "_execute_guarded_graph", fail_graph)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    result = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    event = json.loads(stream.getvalue())
    assert result.status is RunStatus.FAILED
    assert result.error_category == "node_execution_failed"
    assert event["event"] == "langgraph_protocol_failed"
    assert event["node_name"] == "write_report"
    assert event["cause_category"] == "invalid_model_schema"
    assert event["schema_error_type"] == "extra_forbidden"
    assert event["schema_error_path"] == ("application_draft.paragraphs.0.citations.0.source_type")
    for canary in (
        "resume-secret-canary",
        "model-output-secret-canary",
        "application-secret-canary",
    ):
        assert canary not in stream.getvalue()


class _BlockingCancelActionStore(_ActionStore):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_entered = asyncio.Event()
        self.release_cancel = asyncio.Event()

    async def cancel_action(self, command):
        self.cancel_commands.append(command)
        self.cancel_entered.set()
        await self.release_cancel.wait()
        return SimpleNamespace(changed=True)


class _BlockingApprovalResumeResolver(_ApprovalResumeResolver):
    def __init__(self) -> None:
        super().__init__(ApprovalStatus.REJECTED)
        self.entered = asyncio.Event()

    async def resolve_approval_resume(self, **kwargs):
        self.entered.set()
        await asyncio.Event().wait()


async def test_application_writer_missing_draft_fails_before_action_or_approval() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    adapter = _MissingApplicationDraftAdapter()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    result = await _executor_with_adapter(
        execution,
        InMemorySaver(serde=create_checkpoint_serializer()),
        _InvocationRecorder(),
        action_store,
        adapter,
    ).execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    assert result.status is RunStatus.FAILED
    assert result.error_category == "node_execution_failed"
    assert result.retryable is False
    assert adapter.writer_calls == 2
    assert action_store.commands == []
    assert result.approval_request_id is None


async def test_rejected_durable_resume_marker_continues_without_resending_command() -> None:
    from app.domain.tracing import TraceIdentity
    from tests.tracing import CollectingTraceSink, collecting_segment

    sink = CollectingTraceSink()
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    identity = TraceIdentity(workspace_id=execution.workspace_id, run_id=execution.run_id)
    with collecting_segment(sink, trace_identity=identity):
        first = await _executor(execution, checkpointer, recorder, action_store).execute(
            execution.run_id, tenant, CURRENT_GRAPH_VERSION
        )
    assert first.approval_request_id is not None
    resumed = replace(execution, resume_approval_request_id=first.approval_request_id)
    blocking_resolver = _BlockingApprovalResumeResolver()
    with collecting_segment(sink, trace_identity=identity) as old_scope:
        task = asyncio.create_task(
            _executor(
                resumed,
                checkpointer,
                recorder,
                action_store,
                blocking_resolver,
            ).execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
        )
        await blocking_resolver.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    checkpoint = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": str(execution.run_id)}}
    )
    assert checkpoint is not None and approval_resume_was_applied(checkpoint)
    prior_attempts = len(recorder.prepared_graph_nodes)
    with collecting_segment(sink, trace_identity=identity) as new_scope:
        recovered = await _executor(
            resumed,
            checkpointer,
            recorder,
            action_store,
            ApprovalStatus.REJECTED,
        ).execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    assert new_scope.parent != old_scope.parent
    assert sink.starts.keys() == sink.finishes.keys()
    lifecycle = [v[1] for v in sink.starts.values() if v[1].name == "approval.resume_consumed"]
    assert len(lifecycle) == 1
    assert lifecycle[0].segment_identity == new_scope.segment_identity
    assert "TraceParentContext" not in repr(checkpoint)
    assert recovered.status is RunStatus.COMPLETED
    assert len(action_store.cancel_commands) == 1
    assert len(recorder.prepared_graph_nodes) == prior_attempts


@pytest.mark.parametrize(
    ("approval_status", "expected_status", "cancel_count"),
    [
        (ApprovalStatus.APPROVED, RunStatus.COMPLETED, 0),
        (ApprovalStatus.EXPIRED, RunStatus.COMPLETED, 1),
    ],
)
async def test_approved_repauses_while_expired_cancels_and_finalizes(
    approval_status: ApprovalStatus,
    expected_status: RunStatus,
    cancel_count: int,
) -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    first = await _executor(execution, checkpointer, recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    resumed = replace(execution, resume_approval_request_id=first.approval_request_id)
    prior_attempts = len(recorder.prepared_graph_nodes)
    resumed_executor = _executor(
        resumed,
        checkpointer,
        recorder,
        action_store,
        approval_status,
    )
    result = await resumed_executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    assert result.status is expected_status
    assert len(action_store.cancel_commands) == cancel_count
    assert len(resumed_executor.approved_action_executor.identities) == (
        1 if approval_status is ApprovalStatus.APPROVED else 0
    )
    assert len(recorder.prepared_graph_nodes) == prior_attempts


def _blocking_executor(
    execution: RunExecutionInput,
    reader: _RunReader,
    adapter: _BlockingChatAdapter,
    recorder: _InvocationRecorder,
    *,
    execution_timeout_seconds: float = 300.0,
    checkpointer: InMemorySaver | None = None,
) -> LangGraphRunExecutor:
    checkpointer = checkpointer or InMemorySaver(serde=create_checkpoint_serializer())
    return LangGraphRunExecutor(
        reader=reader,
        checkpointer=checkpointer,
        llm_factory=LLMFactory(
            recorder=recorder,
            chat_adapter=adapter,
            embedding_adapter=FakeEmbeddingModel(),
        ),
        search_port=FakeSearch({}),
        tool_recorder=_tool_recorder(checkpointer),
        document_repository=_DocumentRepository(),
        retrieval_event_recorder=_RetrievalEventRecorder(),
        execution_timeout_seconds=execution_timeout_seconds,
        execution_guard_poll_seconds=0.001,
    )


def _executor_with_adapter(
    execution: RunExecutionInput,
    checkpointer: InMemorySaver,
    recorder: _InvocationRecorder,
    action_store: _ActionStore,
    adapter: DeterministicResearchFakeChatAdapter,
) -> LangGraphRunExecutor:
    return LangGraphRunExecutor(
        reader=_RunReader(execution),
        checkpointer=checkpointer,
        llm_factory=LLMFactory(
            recorder=recorder,
            chat_adapter=adapter,
            embedding_adapter=FakeEmbeddingModel(),
        ),
        search_port=FakeSearch({}),
        tool_recorder=_tool_recorder(checkpointer),
        document_repository=_DocumentRepository(),
        retrieval_event_recorder=_RetrievalEventRecorder(),
        action_store=action_store,
        approval_resume_resolver=_ApprovalResumeResolver(),
        approved_action_executor=_ApprovedActionExecutor(),
        execution_timeout_seconds=300,
    )


def test_execution_timeout_is_a_required_constructor_dependency() -> None:
    parameter = inspect.signature(LangGraphRunExecutor).parameters["execution_timeout_seconds"]

    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "execution_timeout_seconds",
    [0, -1, True, False, float("nan"), float("inf"), float("-inf")],
)
def test_execution_timeout_rejects_non_finite_or_non_positive_values(
    execution_timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="execution timeout must be a finite positive number"):
        _executor(
            _execution_input(),
            InMemorySaver(),
            _InvocationRecorder(),
            execution_timeout_seconds=execution_timeout_seconds,
        )


@pytest.mark.parametrize(
    "query",
    [
        "Senior Python Engineer",
        "Senior  Python Engineer",
        "Senior Python Engineer\n",
        "\uff33\uff45\uff4e\uff49\uff4f\uff52 Python Engineer",
    ],
)
async def test_completed_checkpoint_returns_without_repeating_nodes(query: str) -> None:
    execution = replace(_execution_input(), request=ResearchRequestV1(query=query))
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    executor = _executor(execution, checkpointer, recorder)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    first = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    first_attempt_count = len(recorder.prepared_graph_nodes)
    second = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    assert first.status is RunStatus.COMPLETED
    assert second == first
    assert first_attempt_count == 6
    assert len(recorder.prepared_graph_nodes) == first_attempt_count


@pytest.mark.parametrize("mismatch", ["conversation_id", "query", "include_application_draft"])
async def test_checkpoint_identity_mismatch_fails_closed_before_model_send(mismatch: str) -> None:
    execution = _execution_input()
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    first_recorder = _InvocationRecorder()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    first = await _executor(execution, checkpointer, first_recorder).execute(
        execution.run_id,
        tenant,
        CURRENT_GRAPH_VERSION,
    )
    assert first.status is RunStatus.COMPLETED

    if mismatch == "conversation_id":
        mismatched = replace(execution, conversation_id=uuid4())
    else:
        mismatched = replace(
            execution,
            request=ResearchRequestV1(
                query="Different query" if mismatch == "query" else execution.request.query,
                include_application_draft=mismatch == "include_application_draft",
            ),
        )
    second_recorder = _InvocationRecorder()
    result = await _executor(mismatched, checkpointer, second_recorder).execute(
        execution.run_id,
        tenant,
        CURRENT_GRAPH_VERSION,
    )

    assert result.status is RunStatus.FAILED
    assert result.error_category == "checkpoint_identity_mismatch"
    assert result.retryable is False
    assert second_recorder.prepared_graph_nodes == []


async def test_first_and_recovered_interrupt_return_same_waiting_identity() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    executor = _executor(execution, checkpointer, recorder, action_store)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    first = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    attempt_count = len(recorder.prepared_graph_nodes)
    second = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    assert first.status is RunStatus.WAITING_APPROVAL
    assert first.approval_request_id == action_store.request_id
    assert second == first
    assert len(action_store.commands) == 1
    assert len(recorder.prepared_graph_nodes) == attempt_count


async def test_matching_resume_advances_checkpoint_and_repauses_without_new_work() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    first = await _executor(execution, checkpointer, recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    assert first.approval_request_id is not None
    config = {"configurable": {"thread_id": str(execution.run_id)}}
    before = await checkpointer.aget_tuple(config)
    assert before is not None
    assert approval_resume_was_applied(before) is False
    prior_attempts = len(recorder.prepared_graph_nodes)

    resumed = replace(
        execution,
        resume_approval_request_id=first.approval_request_id,
    )
    second = await _executor(resumed, checkpointer, recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    after = await checkpointer.aget_tuple(config)

    assert second == first
    assert after is not None and after.pending_writes != before.pending_writes
    assert approval_resume_was_applied(after) is True
    assert len(recorder.prepared_graph_nodes) == prior_attempts
    assert len(action_store.commands) == 1


async def test_recovered_repause_does_not_resume_same_interrupt_twice() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    executor = _executor(execution, checkpointer, recorder, action_store)
    first = await executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    assert first.approval_request_id is not None
    resumed = replace(execution, resume_approval_request_id=first.approval_request_id)
    resumed_executor = _executor(resumed, checkpointer, recorder, action_store)
    second = await resumed_executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    assert second == first

    config = {"configurable": {"thread_id": str(execution.run_id)}}
    before_recovery = await checkpointer.aget_tuple(config)
    assert before_recovery is not None
    assert approval_resume_was_applied(before_recovery) is True
    checkpoint_id = before_recovery.checkpoint["id"]
    pending_writes = tuple(before_recovery.pending_writes or ())
    prior_attempts = len(recorder.prepared_graph_nodes)
    prior_commands = len(action_store.commands)

    recovered = await resumed_executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)
    after_recovery = await checkpointer.aget_tuple(config)

    assert recovered == first
    assert after_recovery is not None
    assert after_recovery.checkpoint["id"] == checkpoint_id
    assert tuple(after_recovery.pending_writes or ()) == pending_writes
    assert len(recorder.prepared_graph_nodes) == prior_attempts
    assert len(action_store.commands) == prior_commands == 1


async def test_persisted_resume_marker_does_not_bypass_resume_identity_validation() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    first = await _executor(execution, checkpointer, recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    assert first.approval_request_id is not None
    matching = replace(execution, resume_approval_request_id=first.approval_request_id)
    assert (
        await _executor(matching, checkpointer, recorder, action_store).execute(
            execution.run_id, tenant, CURRENT_GRAPH_VERSION
        )
        == first
    )
    config = {"configurable": {"thread_id": str(execution.run_id)}}
    before = await checkpointer.aget_tuple(config)
    assert before is not None and approval_resume_was_applied(before)
    prior_attempts = len(recorder.prepared_graph_nodes)

    mismatched = replace(execution, resume_approval_request_id=uuid4())
    result = await _executor(mismatched, checkpointer, recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    after = await checkpointer.aget_tuple(config)

    assert result.status is RunStatus.FAILED
    assert result.error_category == "resume_approval_request_mismatch"
    assert result.retryable is False
    assert after is not None and after.checkpoint["id"] == before.checkpoint["id"]
    assert tuple(after.pending_writes or ()) == tuple(before.pending_writes or ())
    assert len(recorder.prepared_graph_nodes) == prior_attempts
    assert len(action_store.commands) == 1


async def test_resume_request_mismatch_fails_before_any_additional_model_attempt() -> None:
    execution = replace(
        _execution_input(),
        mode=RunMode.APPLICATION,
        resume_document_id=uuid4(),
        request=ResearchRequestV1(
            query="Prepare a deterministic fake application",
            include_application_draft=True,
        ),
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    first_recorder = _InvocationRecorder()
    action_store = _ActionStore()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    first = await _executor(execution, checkpointer, first_recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    assert first.status is RunStatus.WAITING_APPROVAL
    prior_attempts = len(first_recorder.prepared_graph_nodes)
    config = {"configurable": {"thread_id": str(execution.run_id)}}
    before = await checkpointer.aget_tuple(config)
    assert before is not None

    mismatched = replace(execution, resume_approval_request_id=uuid4())
    result = await _executor(mismatched, checkpointer, first_recorder, action_store).execute(
        execution.run_id, tenant, CURRENT_GRAPH_VERSION
    )
    assert result.status is RunStatus.FAILED
    assert result.error_category == "resume_approval_request_mismatch"
    assert len(first_recorder.prepared_graph_nodes) == prior_attempts
    after = await checkpointer.aget_tuple(config)
    assert after is not None and after.checkpoint["id"] == before.checkpoint["id"]
    assert len(action_store.commands) == 1


@pytest.mark.parametrize(
    ("node_name", "expected_retryable"),
    [("plan", True), ("write_report", True), ("research_agent", False)],
)
async def test_shared_execution_deadline_retryability_is_model_node_only(
    node_name: str,
    expected_retryable: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from time import monotonic

    import app.worker.langgraph_executor as executor_module
    from app.agents.contracts import AgentLoopControl

    stream = StringIO()
    configure_logging(log_level="INFO", stream=stream)
    execution = _execution_input()
    reader = _RunReader(execution)
    adapter = _BlockingChatAdapter(block_graph_node=node_name)

    def controlled_deadline(*, limits, deadline, cancellation):
        # Expire the shared clock only after the intended in-flight call starts.
        # Graph construction speed is not the retryability contract under test.
        def clock():
            return deadline + 1 if adapter.started.is_set() else monotonic()

        return AgentLoopControl(
            limits=limits, deadline=deadline, cancellation=cancellation, clock=clock
        )

    monkeypatch.setattr(executor_module, "AgentLoopControl", controlled_deadline)
    recorder = _InvocationRecorder()
    executor = _blocking_executor(
        execution,
        reader,
        adapter,
        recorder,
        execution_timeout_seconds=300,
    )
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    result = await asyncio.wait_for(
        executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION),
        timeout=5,
    )
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    diagnostic = next(event for event in events if event["event"] == "langgraph_protocol_failed")

    assert adapter.started.is_set()
    assert adapter.cancelled.is_set()
    assert result.status is RunStatus.FAILED
    assert result.error_category == "node_execution_failed"
    assert result.retryable is expected_retryable
    assert diagnostic["node_name"] == node_name
    assert diagnostic["cause_category"] == "deadline_exceeded"


async def test_writer_deadline_retry_resumes_checkpoint_without_research_repeat() -> None:
    execution = _execution_input()
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    checkpointer = InMemorySaver(serde=create_checkpoint_serializer())
    first_recorder = _InvocationRecorder()
    first = await _blocking_executor(
        execution,
        _RunReader(execution),
        _BlockingChatAdapter(block_graph_node="write_report"),
        first_recorder,
        execution_timeout_seconds=0.25,
        checkpointer=checkpointer,
    ).execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION)

    assert first.retryable is True
    assert first_recorder.prepared_graph_nodes.count("research_agent") > 0

    second_recorder = _InvocationRecorder()
    second = await _executor(execution, checkpointer, second_recorder).execute(
        execution.run_id,
        tenant,
        CURRENT_GRAPH_VERSION,
    )

    assert second.status is RunStatus.COMPLETED
    assert "research_agent" not in second_recorder.prepared_graph_nodes
    assert second_recorder.prepared_graph_nodes == ["write_report"]


async def test_persisted_cancellation_stops_in_flight_model_and_returns_cancelled() -> None:
    execution = _execution_input()
    reader = _RunReader(execution)
    adapter = _BlockingChatAdapter()
    recorder = _InvocationRecorder()
    executor = _blocking_executor(execution, reader, adapter, recorder)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    task = asyncio.create_task(executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION))
    await asyncio.wait_for(adapter.started.wait(), timeout=1)
    reader.guard_error = RunExecutionCancelledError()
    result = await asyncio.wait_for(task, timeout=1)

    assert result.status is RunStatus.CANCELLED
    assert adapter.cancelled.is_set()
    assert len(recorder.outcomes) == 1
    assert recorder.outcomes[0].status == "failed"
    assert recorder.outcomes[0].error_category == "cancelled"


@pytest.mark.parametrize("temporary", [False, True])
async def test_execution_guard_failure_stops_graph_without_inventing_revocation(temporary) -> None:
    execution = _execution_input()
    reader = _RunReader(execution)
    adapter = _BlockingChatAdapter()
    recorder = _InvocationRecorder()
    executor = _blocking_executor(execution, reader, adapter, recorder)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )

    task = asyncio.create_task(executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION))
    await asyncio.wait_for(adapter.started.wait(), timeout=1)
    reader.guard_error = DomainUnavailableError() if temporary else RuntimeError("GUARD-CANARY")
    result = await asyncio.wait_for(task, timeout=1)

    assert result.status is RunStatus.FAILED
    assert result.error_category == (
        "execution_guard_unavailable" if temporary else "executor_unhandled_error"
    )
    assert result.retryable is temporary
    assert adapter.cancelled.is_set()
    assert len(recorder.outcomes) == 1
    assert recorder.outcomes[0].error_category == "cancelled"


async def test_external_executor_cancellation_propagates_and_cleans_helper_tasks() -> None:
    execution = _execution_input()
    reader = _RunReader(execution)
    adapter = _BlockingChatAdapter()
    recorder = _InvocationRecorder()
    executor = _blocking_executor(execution, reader, adapter, recorder)
    tenant = TenantContext(
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        role=WorkspaceRole.ADMIN,
    )
    preexisting_tasks = asyncio.all_tasks()
    task = asyncio.create_task(executor.execute(execution.run_id, tenant, CURRENT_GRAPH_VERSION))
    await asyncio.wait_for(adapter.started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert adapter.cancelled.is_set()
    assert asyncio.all_tasks() <= preexisting_tasks


@pytest.mark.parametrize("no_result", [False, True])
async def test_two_executor_coroutines_isolate_nested_trace_subtrees(no_result):
    from app.domain.tracing import TraceIdentity, current_trace_scope
    from tests.tracing import (
        CollectingLLMTraceSink,
        CollectingTraceSink,
        assert_closed_trace_tree,
        collecting_segment,
    )

    sink = CollectingTraceSink()
    arrivals = {}
    inspected = []

    async def rendezvous(label, ordinal, execution):
        scope = current_trace_scope()
        assert scope is not None
        assert scope.trace_identity == TraceIdentity(
            workspace_id=execution.workspace_id, run_id=execution.run_id
        )
        assert scope.parent is not None
        key = (label, ordinal)
        pair = arrivals.setdefault(key, {})
        pair[execution.run_id] = asyncio.Event()
        if len(pair) == 2:
            for event in pair.values():
                event.set()
        await pair[execution.run_id].wait()
        assert current_trace_scope() is scope
        inspected.append((execution.run_id, label))

    async def run(execution):
        counters = {}

        async def meet(label):
            counters[label] = counters.get(label, 0) + 1
            await rendezvous(label, counters[label], execution)

        class Chat(DeterministicResearchFakeChatAdapter):
            async def invoke(self, *args, **kwargs):
                assert current_trace_scope().parent.span_kind == "graph_node"
                await meet("generation")
                return await super().invoke(*args, **kwargs)

        class Embedding(FakeEmbeddingModel):
            async def embed(self, *args, **kwargs):
                assert current_trace_scope().parent.span_kind == "retrieval"
                await meet("embedding")
                return await super().embed(*args, **kwargs)

        class Documents(_DocumentRepository):
            async def search(self, **kwargs):
                scope = current_trace_scope()
                assert scope.parent.span_kind == "retrieval"
                retrieval = sink.starts[scope.parent.context_id][1]
                assert retrieval.parent.span_kind == "tool"
                assert kwargs["tenant"].workspace_id == execution.workspace_id
                await meet("retrieval")
                return () if no_result else await super().search(**kwargs)

        recorder = _InvocationRecorder()
        tools = InMemoryToolInvocationRecorder()
        saver = InMemorySaver()
        llm_sink = CollectingLLMTraceSink()
        executor = LangGraphRunExecutor(
            reader=_RunReader(execution),
            checkpointer=saver,
            llm_factory=LLMFactory(
                recorder=recorder,
                chat_adapter=Chat(),
                embedding_adapter=Embedding(),
                trace_sink=llm_sink,
            ),
            search_port=FakeSearch({}),
            tool_recorder=tools,
            document_repository=Documents(),
            retrieval_event_recorder=_RetrievalEventRecorder(),
            execution_timeout_seconds=30.0,
        )
        identity = TraceIdentity(workspace_id=execution.workspace_id, run_id=execution.run_id)
        with collecting_segment(sink, trace_identity=identity) as scope:
            result = await executor.execute(
                execution.run_id,
                TenantContext(execution.workspace_id, execution.actor_user_id, WorkspaceRole.ADMIN),
                execution.graph_version,
            )
            assert current_trace_scope() is scope
            assert result.status is RunStatus.COMPLETED
        assert current_trace_scope() is None
        assert not llm_sink.open and recorder.outcomes
        assert all(o.status == "succeeded" for o in recorder.outcomes)
        checkpoint = await saver.aget_tuple({"configurable": {"thread_id": str(execution.run_id)}})
        for context_id in sink.starts:
            assert str(context_id) not in repr(checkpoint)
        return identity

    inputs = [replace(_execution_input(), resume_document_id=uuid4()) for _ in range(2)]
    assert inputs[0].workspace_id != inputs[1].workspace_id
    async with asyncio.timeout(15):
        identities = await asyncio.gather(*(run(execution) for execution in inputs))
    assert current_trace_scope() is None
    assert_closed_trace_tree(sink, segments=2)
    for identity in identities:
        observations = [s for _, s in sink.starts.values() if s.trace_identity == identity]
        assert {s.span_kind for s in observations} >= {
            "execution_segment",
            "graph_node",
            "llm_generation",
            "tool",
            "retrieval",
            "llm_embedding",
        }
        assert {label for run_id, label in inspected if run_id == identity.run_id} == {
            "generation",
            "embedding",
            "retrieval",
        }
    retrievals = [c for c, s in sink.starts.values() if s.span_kind == "retrieval"]
    assert all(
        sink.finishes[c.context_id].status == ("no_result" if no_result else "succeeded")
        for c in retrievals
    )
    assert all(len(pair) == 2 for pair in arrivals.values())


@pytest.mark.parametrize(
    ("persisted", "checkpoint_calls", "remaining"),
    [(3, 0, 5), (3, 3, 5), (5, 3, 3), (8, 0, 0), (2, 3, None)],
)
async def test_retry_tool_budget_counts_only_uncheckpointed_calls(
    monkeypatch, persisted, checkpoint_calls, remaining
):
    from app.agents.research_contracts import (
        ResearchGraphStateV1,
        ResearchNodeInputV1,
        ResearchPlanV1,
        SearchCallTraceV1,
    )
    from app.agents.research_nodes import _research_pass_control
    from app.domain.tool_effects import ToolEffect

    execution = _execution_input()
    saver = InMemorySaver()
    executor = _executor(execution, saver, _InvocationRecorder())
    recorder = _tool_recorder(saver)
    for _ in range(persisted):
        await recorder.reserve(
            invocation_id=uuid4(),
            workspace_id=execution.workspace_id,
            actor_user_id=execution.actor_user_id,
            run_id=execution.run_id,
            tool_name="search_web",
            effect=ToolEffect.READ_ONLY,
            args_digest="test",
            call_limit=8,
        )
    traces = tuple(
        SearchCallTraceV1(
            research_pass_number=1,
            call_ordinal=i + 1,
            query=f"query {i}",
            max_results=1,
            result_count=0,
        )
        for i in range(checkpoint_calls)
    )
    state = ResearchGraphStateV1(
        schema_version=3,
        run_id=execution.run_id,
        workspace_id=execution.workspace_id,
        actor_user_id=execution.actor_user_id,
        conversation_id=execution.conversation_id,
        request=execution.request,
        search_calls=traces,
    )
    observed = []

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(
                values=state.model_dump(mode="json") if checkpoint_calls else {},
                interrupts=(),
                next=("research_agent",),
            )

        async def ainvoke(self, graph_input, *, context, **kwargs):
            control = _research_pass_control(
                ResearchNodeInputV1(
                    request=execution.request,
                    normalized_query=execution.request.query,
                    plan=ResearchPlanV1(queries=("query",)),
                    existing_search_calls=traces,
                    research_pass_number=2,
                ),
                context.agent_loop_control,
            )
            observed.append(control.limits)
            raise ConnectionError("stop after checking actual agent admission")

    monkeypatch.setattr(
        "app.worker.langgraph_executor.build_research_state_graph", lambda *a, **k: Graph()
    )
    result = await executor.execute(
        execution.run_id,
        TenantContext(
            execution.workspace_id,
            execution.actor_user_id,
            WorkspaceRole.ADMIN,
        ),
        CURRENT_GRAPH_VERSION,
    )
    if remaining is None:
        assert observed == []
        assert result.error_category == "checkpoint_tool_count_mismatch"
        assert not result.retryable
    else:
        assert observed[0].max_tool_calls == remaining
        assert observed[0].max_tool_results == remaining


@pytest.mark.parametrize("temporary", [False, True])
async def test_consumed_tool_count_failure_retries_only_known_unavailability(
    monkeypatch, temporary
):
    from app.domain.tool_invocations import ToolInvocationRecorderError

    execution = _execution_input()
    saver = InMemorySaver()
    executor = _executor(execution, saver, _InvocationRecorder())

    async def unavailable(**kwargs):
        raise DomainUnavailableError() if temporary else ToolInvocationRecorderError()

    monkeypatch.setattr(_tool_recorder(saver), "consumed_call_count", unavailable)
    result = await executor.execute(
        execution.run_id,
        TenantContext(
            execution.workspace_id,
            execution.actor_user_id,
            WorkspaceRole.ADMIN,
        ),
        CURRENT_GRAPH_VERSION,
    )
    assert result.error_category == (
        "database_unavailable" if temporary else "executor_unhandled_error"
    )
    assert result.retryable is temporary


@pytest.mark.parametrize("known", [False, True])
async def test_only_classified_checkpoint_errors_are_retryable(monkeypatch, known):
    from psycopg.errors import QueryCanceled

    from app.db.errors import is_checkpoint_unavailable

    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())
    executor._checkpoint_error_is_unavailable = is_checkpoint_unavailable

    async def fail_graph(_execution):
        raise QueryCanceled("BODY-CANARY") if known else RuntimeError("BODY-CANARY")

    monkeypatch.setattr(executor, "_execute_guarded_graph", fail_graph)
    result = await executor.execute(
        execution.run_id,
        TenantContext(execution.workspace_id, execution.actor_user_id, WorkspaceRole.ADMIN),
        CURRENT_GRAPH_VERSION,
    )
    assert result.status is RunStatus.FAILED
    assert result.retryable is known
    assert result.error_category == (
        "checkpoint_unavailable" if known else "executor_unhandled_error"
    )


async def test_broken_checkpoint_waits_for_supervisor_cancellation_without_result(monkeypatch):
    from psycopg import OperationalError

    from app.db.errors import is_checkpoint_unavailable

    execution = _execution_input()
    executor = _executor(execution, InMemorySaver(), _InvocationRecorder())
    connection = SimpleNamespace(broken=False, closed=False)
    executor._checkpointer = SimpleNamespace(conn=connection)
    executor._checkpoint_error_is_unavailable = is_checkpoint_unavailable
    failed = asyncio.Event()

    async def fail_graph(_execution):
        connection.broken = True
        failed.set()
        raise OperationalError("BODY-CANARY")

    monkeypatch.setattr(executor, "_execute_guarded_graph", fail_graph)
    task = asyncio.create_task(
        executor.execute(
            execution.run_id,
            TenantContext(execution.workspace_id, execution.actor_user_id, WorkspaceRole.ADMIN),
            CURRENT_GRAPH_VERSION,
        )
    )
    try:
        await asyncio.wait_for(failed.wait(), timeout=1)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
