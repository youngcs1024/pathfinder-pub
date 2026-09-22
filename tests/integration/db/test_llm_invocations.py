from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update

from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    Conversation,
    LLMInvocation,
    Message,
    Run,
    Workspace,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
    transaction,
)
from app.domain.provisioning import ProvisionedPersonalWorkspace, ProvisioningService
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError, LLMRetryPolicy
from app.llm.fake import (
    FakeChatModel,
    FakeEmbeddingModel,
    ScriptedFakeChatModel,
    ScriptedFakeFailure,
)
from app.llm.invocations import (
    InvocationRecorderPort,
    LLMInvocationAttempt,
    LLMInvocationAuthorizationError,
    LLMInvocationContext,
    LLMInvocationInvariantError,
    LLMInvocationOutcome,
    LLMTraceStart,
    NoOpTraceSink,
    TraceIdentifiers,
    TraceSinkPort,
)
from app.llm.ports import (
    ChatMessage,
    ChatModelResult,
    EmbeddingResult,
    ModelToolSchema,
    ModelUsage,
    ProviderAdapterError,
    ProviderAttemptContext,
)
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION

pytestmark = pytest.mark.integration
PROMPT_VERSION = f"sha256:{'a' * 64}"


@dataclass(frozen=True)
class _Runtime:
    session_factory: AsyncSessionFactory
    provisioning: ProvisioningService
    recorder: SqlAlchemyInvocationRecorder


@dataclass(frozen=True)
class _VisibilityChatAdapter:
    provider = "fake"
    model = "qwen3.6-flash-2026-04-16"

    session_factory: AsyncSessionFactory

    async def invoke(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ModelToolSchema, ...],
        metadata: dict[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        async with self.session_factory() as session:
            started = (
                await session.scalars(
                    select(LLMInvocation).where(LLMInvocation.status == "started")
                )
            ).all()
        assert len(started) == 1
        assert started[0].token_usage is None
        assert started[0].pricing_version is None
        assert started[0].currency is None
        assert started[0].estimated_cost is None
        assert started[0].latency_ms is None
        return ChatModelResult(
            content="visible prepared invocation",
            usage=ModelUsage(input_tokens=2, output_tokens=3),
        )


@dataclass(frozen=True)
class _CostedQwenChatAdapter:
    provider = "qwen"
    model = "qwen3.6-flash-2026-04-16"

    usage: ModelUsage

    async def invoke(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ModelToolSchema, ...],
        metadata: dict[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        return ChatModelResult(
            content="costed Qwen result",
            usage=self.usage,
            provider="qwen",
            provider_response_id=f"resp_{attempt.invocation_id}",
        )


@dataclass(frozen=True)
class _UnusedQwenEmbeddingAdapter:
    provider = "qwen"
    model = "text-embedding-v4"

    async def embed(
        self,
        texts: tuple[str, ...],
        metadata: dict[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> EmbeddingResult:
        raise AssertionError("embedding adapter is not used by these chat tests")


@dataclass
class _RetryingQwenChatAdapter:
    provider = "qwen"
    model = "qwen3.6-flash-2026-04-16"

    attempt_ids: list[UUID]

    async def invoke(
        self,
        messages: tuple[ChatMessage, ...],
        tools: tuple[ModelToolSchema, ...],
        metadata: dict[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        self.attempt_ids.append(attempt.invocation_id)
        if len(self.attempt_ids) == 1:
            raise ProviderAdapterError(category="rate_limited", retryable=True)
        return ChatModelResult(
            content="retry converged",
            usage=ModelUsage(input_tokens=11, output_tokens=13, cached_input_tokens=0),
            provider="qwen",
            provider_response_id="resp_retry_success",
        )


@dataclass
class _TraceIdentifiersSink:
    starts: int = 0

    def start(self, trace: LLMTraceStart) -> TraceIdentifiers:
        self.starts += 1
        return TraceIdentifiers(
            trace_id=f"{self.starts:032x}",
            observation_id=f"{self.starts:016x}",
        )

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None:
        return None


async def _no_retry_delay(_delay: float) -> None:
    return None


@dataclass(frozen=True)
class _FinalizeFailingRecorder:
    delegate: SqlAlchemyInvocationRecorder

    async def prepare(self, attempt: LLMInvocationAttempt) -> None:
        await self.delegate.prepare(attempt)

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None:
        raise RuntimeError("finalize-database-secret-canary")


@dataclass
class _CountingTraceSink:
    starts: int = 0
    finishes: int = 0

    def start(self, trace: LLMTraceStart) -> None:
        self.starts += 1
        return None

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None:
        self.finishes += 1


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        yield _Runtime(
            session_factory=session_factory,
            provisioning=ProvisioningService(SqlAlchemyProvisioningStore(session_factory)),
            recorder=SqlAlchemyInvocationRecorder(session_factory),
        )
    finally:
        await engine.dispose()


def _attempt(
    identity: ProvisionedPersonalWorkspace,
    *,
    invocation_id: UUID | None = None,
    provider: str = "fake",
    run_id: UUID | None = None,
) -> LLMInvocationAttempt:
    return LLMInvocationAttempt(
        invocation_id=invocation_id or uuid4(),
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
        run_id=run_id,
        invocation_kind="chat",
        provider=provider,
        model="qwen3.6-flash-2026-04-16",
        graph_node="plan",
        prompt_version=PROMPT_VERSION,
        request_hash=f"sha256:{'b' * 64}",
    )


async def _invoke_chat(
    runtime: _Runtime,
    *,
    workspace_id: UUID,
    actor_user_id: UUID,
    run_id: UUID | None = None,
    trace_sink: TraceSinkPort | None = None,
) -> None:
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
        trace_sink=trace_sink or NoOpTraceSink(),
    )
    model = factory.create_chat_model(
        LLMInvocationContext(
            workspace_id=workspace_id,
            actor_user_id=actor_user_id,
            run_id=run_id,
        )
    )
    await model.invoke(
        (ChatMessage(role="user", content="sensitive-accounting-canary"),),
        (),
        {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
    )


async def _create_research_run(
    runtime: _Runtime,
    identity: ProvisionedPersonalWorkspace,
) -> UUID:
    async with transaction(runtime.session_factory) as session:
        conversation = Conversation(
            workspace_id=identity.workspace_id,
            created_by_user_id=identity.user_id,
            title="LLM run-link fixture",
        )
        session.add(conversation)
        await session.flush()
        message = Message(
            workspace_id=identity.workspace_id,
            conversation_id=conversation.id,
            actor_user_id=identity.user_id,
            role="user",
            content="Research the linked invocation",
        )
        session.add(message)
        await session.flush()
        run = Run(
            mode="research",
            graph_version="pathfinder-research-v6",
            workspace_id=identity.workspace_id,
            created_by_user_id=identity.user_id,
            conversation_id=conversation.id,
            request_message_id=message.id,
            input_json={},
            limits_json={},
            status="running",
            started_at=datetime.now(UTC),
        )
        session.add(run)
        await session.flush()
        return run.id


async def _invoke_qwen_chat(
    runtime: _Runtime,
    *,
    workspace_id: UUID,
    actor_user_id: UUID,
    usage: ModelUsage,
) -> None:
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=_CostedQwenChatAdapter(usage),
        embedding_adapter=_UnusedQwenEmbeddingAdapter(),
        provider="qwen",
    )
    model = factory.create_chat_model(
        LLMInvocationContext(workspace_id=workspace_id, actor_user_id=actor_user_id)
    )
    await model.invoke(
        (ChatMessage(role="user", content="cost accounting test"),),
        (),
        {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
    )


async def test_accounted_fake_chat_persists_workspace_attempt_and_terminal_usage(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("accounted-user")

    await _invoke_chat(
        runtime,
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
    )

    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.workspace_id == identity.workspace_id
    assert invocation.actor_user_id == identity.user_id
    assert invocation.status == "succeeded"
    assert invocation.token_usage == {"input_tokens": 0, "output_tokens": 0}
    assert invocation.pricing_version is None
    assert invocation.currency is None
    assert invocation.estimated_cost is None
    assert invocation.provider_response_id is None
    assert invocation.latency_ms is not None and invocation.latency_ms >= 0
    assert invocation.error_category is None
    assert invocation.request_hash.startswith("sha256:")
    assert "sensitive-accounting-canary" not in invocation.request_hash


async def test_accounted_attempt_persists_matching_optional_run_id(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("run-linked-user")
    run_id = await _create_research_run(runtime, identity)

    await _invoke_chat(
        runtime,
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
        run_id=run_id,
    )

    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.run_id == run_id


async def test_orphan_and_cross_workspace_run_are_rejected_before_provider(
    runtime: _Runtime,
) -> None:
    first = await runtime.provisioning.provision_personal_workspace("run-link-first")
    second = await runtime.provisioning.provision_personal_workspace("run-link-second")
    second_run_id = await _create_research_run(runtime, second)

    for run_id in (uuid4(), second_run_id):
        with pytest.raises(LLMInvocationAuthorizationError):
            await runtime.recorder.prepare(_attempt(first, run_id=run_id))

    async with runtime.session_factory() as session:
        assert (await session.scalars(select(LLMInvocation))).all() == []


async def test_same_workspace_member_cannot_attach_invocation_to_another_actors_run(
    runtime: _Runtime,
) -> None:
    owner = await runtime.provisioning.provision_personal_workspace("run-owner")
    other = await runtime.provisioning.provision_personal_workspace("run-member")
    run_id = await _create_research_run(runtime, owner)
    async with transaction(runtime.session_factory) as session:
        session.add(
            WorkspaceMembership(
                id=uuid4(),
                workspace_id=owner.workspace_id,
                user_id=other.user_id,
                role="member",
            )
        )

    class _CountingChat(FakeChatModel):
        def __init__(self) -> None:
            self.calls = 0

        async def invoke(self, messages, tools, metadata, *, attempt=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            return await super().invoke(messages, tools, metadata, attempt=attempt)

    adapter = _CountingChat()
    trace_sink = _CountingTraceSink()
    model = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=adapter,
        embedding_adapter=FakeEmbeddingModel(),
        trace_sink=trace_sink,
    ).create_chat_model(
        LLMInvocationContext(
            workspace_id=owner.workspace_id,
            actor_user_id=other.user_id,
            run_id=run_id,
        )
    )

    with pytest.raises(LLMInvocationAuthorizationError):
        await model.invoke(
            (ChatMessage(role="user", content="forbidden provenance splice"),),
            (),
            {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
        )

    assert adapter.calls == 0
    assert trace_sink.starts == 0
    assert trace_sink.finishes == 0
    async with runtime.session_factory() as session:
        assert (await session.scalars(select(LLMInvocation))).all() == []


async def test_prepared_attempt_is_committed_and_visible_before_provider_call(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("visible-before-call")
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=_VisibilityChatAdapter(runtime.session_factory),
        embedding_adapter=FakeEmbeddingModel(),
    )
    model = factory.create_chat_model(
        LLMInvocationContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
        )
    )

    result = await model.invoke(
        (ChatMessage(role="user", content="visibility test"),),
        (),
        {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
    )

    assert result.content == "visible prepared invocation"
    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.status == "succeeded"
    assert invocation.token_usage == {"input_tokens": 2, "output_tokens": 3}
    assert invocation.estimated_cost is None


async def test_qwen_success_persists_usage_and_exact_cost_on_the_same_attempt(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("qwen-cost-user")
    usage = ModelUsage(
        input_tokens=11,
        output_tokens=13,
        total_tokens=24,
        cached_input_tokens=0,
        reasoning_output_tokens=5,
    )

    await _invoke_qwen_chat(
        runtime,
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
        usage=usage,
    )

    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.status == "succeeded"
    assert invocation.token_usage == usage.model_dump(mode="json", round_trip=True)
    assert invocation.pricing_version == QWEN_BEIJING_PRICING_VERSION
    assert invocation.currency == "CNY"
    assert invocation.estimated_cost == Decimal("0.000106800000")


async def test_retrying_qwen_attempts_are_separate_terminal_scoped_facts(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("qwen-retry-user")
    adapter = _RetryingQwenChatAdapter(attempt_ids=[])
    trace_sink = _TraceIdentifiersSink()
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=adapter,
        embedding_adapter=_UnusedQwenEmbeddingAdapter(),
        trace_sink=trace_sink,
        provider="qwen",
        retry_policy=LLMRetryPolicy(max_attempts=3),
        sleeper=_no_retry_delay,
        random_source=lambda: 0.0,
    )
    model = factory.create_chat_model(
        LLMInvocationContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
        )
    )

    result = await model.invoke(
        (ChatMessage(role="user", content="retry integration test"),),
        (),
        {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
    )

    assert result.content == "retry converged"
    assert len(adapter.attempt_ids) == 2
    async with runtime.session_factory() as session:
        rows = (
            await session.scalars(
                select(LLMInvocation).where(LLMInvocation.id.in_(adapter.attempt_ids))
            )
        ).all()
    by_id = {row.id: row for row in rows}
    assert set(by_id) == set(adapter.attempt_ids)
    assert [by_id[item].status for item in adapter.attempt_ids] == ["failed", "succeeded"]
    assert all(row.workspace_id == identity.workspace_id for row in rows)
    assert all(row.actor_user_id == identity.user_id for row in rows)
    assert all(row.trace_ids is not None for row in rows)
    assert by_id[adapter.attempt_ids[0]].estimated_cost is None
    assert by_id[adapter.attempt_ids[1]].estimated_cost == Decimal("0.000106800000")


async def test_embedding_success_persists_zero_usage_without_rewriting_history(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("embedding-usage-user")
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
    )
    model = factory.create_embedding_model(
        LLMInvocationContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
        )
    )

    await model.embed(("alpha", "beta"), {"graph_node": "ingest_documents"})

    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.invocation_kind == "embedding"
    assert invocation.status == "succeeded"
    assert invocation.token_usage == {"input_tokens": 0, "output_tokens": 0}
    assert invocation.provider_response_id is None


async def test_provider_timeout_is_persisted_without_exception_body(runtime: _Runtime) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("timeout-user")
    factory = LLMFactory(
        recorder=runtime.recorder,
        chat_adapter=ScriptedFakeChatModel((ScriptedFakeFailure(kind="timeout"),)),
        embedding_adapter=FakeEmbeddingModel(),
    )
    model = factory.create_chat_model(
        LLMInvocationContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
        )
    )

    with pytest.raises(LLMProviderError) as captured:
        await model.invoke(
            (ChatMessage(role="user", content="timeout-body-canary"),),
            (),
            {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
        )

    assert captured.value.category == "provider_timeout"
    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.status == "failed"
    assert invocation.error_category == "provider_timeout"
    assert invocation.token_usage is None
    assert "canary" not in invocation.request_hash


async def test_finalize_database_failure_leaves_started_fact_and_returns_no_result(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("finalize-failure-user")
    recorder: InvocationRecorderPort = _FinalizeFailingRecorder(runtime.recorder)
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
    )
    model = factory.create_chat_model(
        LLMInvocationContext(
            workspace_id=identity.workspace_id,
            actor_user_id=identity.user_id,
        )
    )

    with pytest.raises(LLMAccountingError) as captured:
        await model.invoke(
            (ChatMessage(role="user", content="finalize failure"),),
            (),
            {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
        )

    assert captured.value.phase == "finalize"
    assert "secret-canary" not in str(captured.value)
    async with runtime.session_factory() as session:
        invocation = (await session.scalars(select(LLMInvocation))).one()
    assert invocation.status == "started"
    assert invocation.latency_ms is None


async def test_recorder_finalize_is_idempotent_but_conflicting_terminal_state_fails(
    runtime: _Runtime,
) -> None:
    identity = await runtime.provisioning.provision_personal_workspace("finalize-user")
    attempt = _attempt(identity, provider="qwen")
    trace_ids = TraceIdentifiers(trace_id="1" * 32, observation_id="2" * 16)
    outcome = LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(
            input_tokens=11,
            output_tokens=13,
            total_tokens=24,
            cached_input_tokens=0,
            reasoning_output_tokens=5,
        ),
        provider_response_id="req_finalize_123",
        latency_ms=12,
        pricing_version=QWEN_BEIJING_PRICING_VERSION,
        currency="CNY",
        estimated_cost=Decimal("0.000106800000"),
        trace_ids=trace_ids,
    )
    await runtime.recorder.prepare(attempt)
    await runtime.recorder.finalize(attempt, outcome)
    await runtime.recorder.finalize(attempt, outcome)

    async with runtime.session_factory() as session:
        invocation = await session.get(LLMInvocation, attempt.invocation_id)
    assert invocation is not None
    assert invocation.trace_ids == trace_ids.model_dump(mode="json")

    conflicting_trace_ids = outcome.model_copy(
        update={"trace_ids": TraceIdentifiers(trace_id="3" * 32, observation_id="4" * 16)}
    )
    with pytest.raises(LLMInvocationInvariantError):
        await runtime.recorder.finalize(attempt, conflicting_trace_ids)

    with pytest.raises(LLMInvocationInvariantError):
        await runtime.recorder.finalize(
            attempt,
            LLMInvocationOutcome(
                status="failed",
                provider_response_id="req_conflict_123",
                latency_ms=12,
                error_category="provider_error",
            ),
        )


async def test_unknown_cross_workspace_and_revoked_actor_are_rejected_before_insert(
    runtime: _Runtime,
) -> None:
    first = await runtime.provisioning.provision_personal_workspace("first-accounting-user")
    second = await runtime.provisioning.provision_personal_workspace("second-accounting-user")

    trace_sink = _CountingTraceSink()
    for workspace_id, actor_user_id in (
        (first.workspace_id, second.user_id),
        (first.workspace_id, uuid4()),
    ):
        with pytest.raises(LLMInvocationAuthorizationError):
            await _invoke_chat(
                runtime,
                workspace_id=workspace_id,
                actor_user_id=actor_user_id,
                trace_sink=trace_sink,
            )

    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(WorkspaceMembership.id == first.membership_id)
            .values(revoked_at=WorkspaceMembership.created_at)
        )
    with pytest.raises(LLMInvocationAuthorizationError):
        await _invoke_chat(
            runtime,
            workspace_id=first.workspace_id,
            actor_user_id=first.user_id,
            trace_sink=trace_sink,
        )

    async with runtime.session_factory() as session:
        assert (await session.scalars(select(LLMInvocation))).all() == []
    assert trace_sink.starts == 0
    assert trace_sink.finishes == 0


async def test_workspace_aggregation_preserves_actor_audit_without_cross_tenant_mix(
    runtime: _Runtime,
) -> None:
    first = await runtime.provisioning.provision_personal_workspace("aggregate-first")
    second = await runtime.provisioning.provision_personal_workspace("aggregate-second")
    team_workspace_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add(
            Workspace(
                id=team_workspace_id,
                kind="team",
                name="Accounting Team Fixture",
                created_by_user_id=first.user_id,
            )
        )
        await session.flush()
        session.add_all(
            (
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=team_workspace_id,
                    user_id=first.user_id,
                    role="admin",
                ),
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=team_workspace_id,
                    user_id=second.user_id,
                    role="member",
                ),
            )
        )

    await _invoke_chat(
        runtime,
        workspace_id=first.workspace_id,
        actor_user_id=first.user_id,
    )
    await _invoke_chat(
        runtime,
        workspace_id=team_workspace_id,
        actor_user_id=first.user_id,
    )
    await _invoke_chat(
        runtime,
        workspace_id=team_workspace_id,
        actor_user_id=second.user_id,
    )

    async with runtime.session_factory() as session:
        invocations = (await session.scalars(select(LLMInvocation))).all()
    assert len(invocations) == 3
    assert sum(item.workspace_id == first.workspace_id for item in invocations) == 1
    assert {
        item.actor_user_id for item in invocations if item.workspace_id == team_workspace_id
    } == {first.user_id, second.user_id}


async def test_cost_aggregation_remains_partitioned_by_workspace(runtime: _Runtime) -> None:
    first = await runtime.provisioning.provision_personal_workspace("cost-first-workspace")
    second = await runtime.provisioning.provision_personal_workspace("cost-second-workspace")
    team_workspace_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add(
            Workspace(
                id=team_workspace_id,
                kind="team",
                name="Cost Aggregation Team Fixture",
                created_by_user_id=first.user_id,
            )
        )
        await session.flush()
        session.add_all(
            (
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=team_workspace_id,
                    user_id=first.user_id,
                    role="admin",
                ),
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=team_workspace_id,
                    user_id=second.user_id,
                    role="member",
                ),
            )
        )

    await _invoke_qwen_chat(
        runtime,
        workspace_id=first.workspace_id,
        actor_user_id=first.user_id,
        usage=ModelUsage(
            input_tokens=11,
            output_tokens=13,
            total_tokens=24,
            cached_input_tokens=0,
        ),
    )
    await _invoke_qwen_chat(
        runtime,
        workspace_id=team_workspace_id,
        actor_user_id=first.user_id,
        usage=ModelUsage(
            input_tokens=20,
            output_tokens=5,
            total_tokens=25,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
        ),
    )
    await _invoke_qwen_chat(
        runtime,
        workspace_id=team_workspace_id,
        actor_user_id=second.user_id,
        usage=ModelUsage(
            input_tokens=10,
            output_tokens=1,
            total_tokens=11,
            cached_input_tokens=0,
            cache_write_input_tokens=0,
        ),
    )

    async with runtime.session_factory() as session:
        rows = (
            await session.execute(
                select(
                    LLMInvocation.workspace_id,
                    func.sum(LLMInvocation.estimated_cost),
                )
                .where(LLMInvocation.estimated_cost.is_not(None))
                .group_by(LLMInvocation.workspace_id)
            )
        ).all()
    costs_by_workspace = dict(rows)
    assert costs_by_workspace == {
        first.workspace_id: Decimal("0.000106800000"),
        team_workspace_id: Decimal("0.000079200000"),
    }

    async with runtime.session_factory() as session:
        team_actors = set(
            await session.scalars(
                select(LLMInvocation.actor_user_id).where(
                    LLMInvocation.workspace_id == team_workspace_id
                )
            )
        )
    assert team_actors == {first.user_id, second.user_id}
