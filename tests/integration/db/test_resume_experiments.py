"""New allocation survives recorder reconstruction without legacy timeout exceptions."""

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantService
from app.llm.invocations import LLMInvocationAttempt, LLMInvocationOutcome
from app.llm.ports import LOCKED_CHAT_MODEL, ModelUsage
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from tests.evals.product_acceptance_contracts import AcceptanceError
from tests.evals.resume_experiment_budget import ExperimentRecorder
from tests.evals.resume_experiment_contracts import ExperimentBudget

pytestmark = pytest.mark.integration


@pytest.fixture
async def ledger(migrated_database_url, tmp_path):
    tmp_path.chmod(0o700)
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    owner = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(f"new-experiment-{uuid4()}")
    tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
        workspace_id=owner.workspace_id, actor_user_id=owner.user_id
    )
    inputs = SimpleNamespace(
        allocation_id=uuid4(), digest="synthetic-new-input", budget=ExperimentBudget(attempt_cap=2)
    )
    kwargs = dict(root=tmp_path, inputs=inputs, database_identity="owned-test-database")
    recorder = ExperimentRecorder(sessions, tenant, **kwargs)
    await recorder.initialize()
    try:
        yield sessions, tenant, kwargs, recorder
    finally:
        await engine.dispose()


def attempt(tenant):
    return LLMInvocationAttempt(
        invocation_id=uuid4(),
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
        invocation_kind="chat",
        provider="qwen",
        model=LOCKED_CHAT_MODEL,
        graph_node="annotate",
        prompt_version="sha256:" + "a" * 64,
        request_hash="sha256:" + "b" * 64,
    )


def known():
    return LLMInvocationOutcome(
        status="succeeded",
        latency_ms=1,
        token_usage=ModelUsage(input_tokens=10, output_tokens=2),
        pricing_version=QWEN_BEIJING_PRICING_VERSION,
        currency="CNY",
        estimated_cost=Decimal("0.01"),
    )


async def test_reconstruction_and_cumulative_cap(ledger):
    sessions, tenant, kwargs, recorder = ledger
    first = attempt(tenant)
    await recorder.prepare(first)
    await recorder.finalize(first, known())
    resumed = ExperimentRecorder(sessions, tenant, **kwargs)
    await resumed.initialize()
    assert (await resumed.measurement())["attempts"] == 1
    second = attempt(tenant)
    await resumed.prepare(second)
    await resumed.finalize(second, known())
    latest = ExperimentRecorder(sessions, tenant, **kwargs)
    await latest.initialize()
    measured = await latest.measurement()
    assert measured["attempts"] == 2 and Decimal(measured["known_cost_cny"]) == Decimal("0.02")
    with pytest.raises(AcceptanceError, match="budget_exhausted"):
        await latest.prepare(attempt(tenant))
    assert (await latest.measurement())["attempts"] == 2


@pytest.mark.parametrize("terminal", [False, True])
async def test_unknown_or_unfinished_stops_after_restart(ledger, terminal):
    sessions, tenant, kwargs, recorder = ledger
    first = attempt(tenant)
    await recorder.prepare(first)
    if terminal:
        with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
            await recorder.finalize(
                first,
                LLMInvocationOutcome(
                    status="failed", latency_ms=100, error_category="provider_timeout"
                ),
            )
    resumed = ExperimentRecorder(sessions, tenant, **kwargs)
    await resumed.initialize()
    with pytest.raises(AcceptanceError, match=r"unknown_usage_or_cost|unfinished_accounting"):
        await resumed.prepare(attempt(tenant))
    assert (await resumed.measurement())["attempts"] == 1


async def test_identity_and_foreign_scope_rejected(ledger):
    sessions, tenant, kwargs, recorder = ledger
    with pytest.raises(AcceptanceError, match="scope"):
        await recorder.prepare(attempt(tenant).model_copy(update={"workspace_id": uuid4()}))
    assert (await recorder.measurement())["attempts"] == 0
    changed = ExperimentRecorder(sessions, tenant, **{**kwargs, "database_identity": "replacement"})
    with pytest.raises(AcceptanceError, match="binding"):
        await changed.initialize()


async def test_authoritative_ledger_missing_attempt_is_not_reset(ledger):
    sessions, tenant, kwargs, _recorder = ledger
    from app.db.llm_invocations import SqlAlchemyInvocationRecorder

    # A write outside the experiment journal cannot silently become an accepted allocation.
    await SqlAlchemyInvocationRecorder(sessions).prepare(attempt(tenant))
    resumed = ExperimentRecorder(sessions, tenant, **kwargs)
    with pytest.raises(AcceptanceError, match="ledger_attempt_mismatch"):
        await resumed.initialize()


async def test_factory_timeout_is_counted_and_stops_retry(ledger):
    from app.llm.factory import LLMAccountingError, LLMFactory
    from app.llm.invocations import LLMInvocationContext
    from app.llm.ports import LOCKED_EMBEDDING_MODEL, ChatMessage, ProviderAdapterError

    sessions, tenant, kwargs, recorder = ledger

    class TimeoutChat:
        provider = "qwen"
        model = LOCKED_CHAT_MODEL
        calls = 0

        async def invoke(self, messages, tools, metadata, *, attempt):
            self.calls += 1
            raise ProviderAdapterError(category="provider_timeout", retryable=True)

    class UnusedEmbedding:
        provider = "qwen"
        model = LOCKED_EMBEDDING_MODEL

        async def embed(self, texts, metadata, *, attempt):
            pytest.fail("no embedding requested")

    adapter = TimeoutChat()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=adapter,
        embedding_adapter=UnusedEmbedding(),
        provider="qwen",
    )
    model = factory.create_chat_model(
        LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
    )
    with pytest.raises(LLMAccountingError):
        await model.invoke(
            (ChatMessage(role="user", content="Synthetic request"),),
            (),
            {"graph_node": "annotate", "prompt_version": "sha256:" + "a" * 64},
        )
    assert adapter.calls == 1
    resumed = ExperimentRecorder(sessions, tenant, **kwargs)
    await resumed.initialize()
    measured = await resumed.measurement()
    assert measured["attempts"] == measured["unknown_cost"] == measured["unknown_usage"] == 1


@pytest.mark.parametrize("new_unknown", [False, True])
async def test_reserved_timeout_survives_restart_and_new_unknown_stops(ledger, new_unknown):
    from tests.evals.product_acceptance_contracts import publish
    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_experiment_budget import CHAT_HEADROOM_CNY

    sessions, tenant, kwargs, recorder = ledger
    root = kwargs["root"]
    first = attempt(tenant)
    started = {"binding": kwargs["inputs"].digest, "before": {"invocation_ids": []}}
    publish(root / "annotation-example-started.json", started)
    await recorder.prepare(first)
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await recorder.finalize(
            first,
            LLMInvocationOutcome(status="failed", latency_ms=1, error_category="provider_timeout"),
        )
    from tests.evals.product_acceptance_contracts import read_private_json

    original = read_private_json(root / f"accounted-{first.invocation_id}.json")
    publish(
        root / "timeout-exception.json",
        {
            "binding": recorder.binding,
            "authorization": "Synthetic explicit single-call authorization",
            "policy": "single_terminal_timeout_v1",
            "invocation_id": str(first.invocation_id),
            "accounted_digest": quality_identity_digest(original),
            "stage": "annotation-example",
            "started_digest": quality_identity_digest(started),
            "reserved_cost_cny": str(CHAT_HEADROOM_CNY),
        },
    )
    resumed = ExperimentRecorder(sessions, tenant, **kwargs)
    await resumed.initialize()
    value = await resumed.check_admission()
    assert value["attempts"] == value["unknown_cost"] == value["unknown_usage"] == 1
    assert Decimal(value["budget_occupied_cny"]) == CHAT_HEADROOM_CNY
    second = attempt(tenant)
    await resumed.prepare(second)
    if not new_unknown:
        await resumed.finalize(second, known())
        latest = ExperimentRecorder(sessions, tenant, **kwargs)
        await latest.initialize()
        value = await latest.check_admission(after=True)
        assert value["attempts"] == 2 and value["unknown_cost"] == 1
        assert Decimal(value["budget_occupied_cny"]) == CHAT_HEADROOM_CNY + Decimal("0.01")
        assert read_private_json(root / f"accounted-{first.invocation_id}.json") == original
        return
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await resumed.finalize(
            second,
            LLMInvocationOutcome(status="failed", latency_ms=1, error_category="provider_timeout"),
        )
    latest = ExperimentRecorder(sessions, tenant, **kwargs)
    await latest.initialize()
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await latest.check_admission()
    value = await latest.measurement()
    assert value["attempts"] == value["unknown_cost"] == 2
    assert value["reserved_unknown_attempts"] == 1
    assert read_private_json(root / f"accounted-{first.invocation_id}.json") == original


@pytest.mark.parametrize("last_node", ["score_review", "write_draft"])
async def test_a_timeout_two_reservations_bound_to_score_calls(ledger, last_node):
    from tests.evals.resume_experiments import preserve
    from tests.evals.resume_initial_reconciliation import ARecorder

    sessions, tenant, kwargs, _ = ledger
    kwargs["inputs"].budget = ExperimentBudget()
    recorder = ARecorder(sessions, tenant, **kwargs)
    await recorder.initialize()
    recorder.enable_policy()
    for index in range(3):
        row = attempt(tenant).model_copy(
            update={"graph_node": "score_initial" if index < 2 else last_node}
        )
        path = kwargs["root"] / "a" / "synthetic" / f"stage-{index}" / "call-00-started.json"
        path.parent.mkdir(mode=0o700, parents=True)
        preserve(
            path, {"identity": f"synthetic-{index}", "before": await recorder.check_admission()}
        )
        recorder.active_score_call = (
            (path, f"synthetic-{index}") if row.graph_node.startswith("score_") else None
        )
        await recorder.prepare(row)
        outcome = LLMInvocationOutcome(
            status="failed", latency_ms=1, error_category="provider_timeout"
        )
        if index == 2:
            with pytest.raises(AcceptanceError, match=r"a_timeout_limit|unknown_usage_or_cost"):
                await recorder.finalize(row, outcome)
        else:
            await recorder.finalize(row, outcome)
            usage = await recorder.check_admission(after=True)
            assert usage["unknown_cost"] == usage["a_reserved_unknown_attempts"] == index + 1
            assert usage["unfinished"] == 0
            rebuilt = ARecorder(sessions, tenant, **kwargs)
            await rebuilt.initialize()
            assert await rebuilt.check_admission(after=True) == usage
    assert len(await recorder.rows()) == 3
    assert len(list((kwargs["root"] / "a-score-timeouts").glob("*.json"))) == 2


async def test_a_score_timeout_retry_and_completed_replay(ledger):
    from app.llm.factory import LLMFactory, LLMRetryPolicy
    from app.llm.invocations import LLMInvocationContext
    from app.llm.ports import LOCKED_EMBEDDING_MODEL, ChatModelResult, ProviderAdapterError
    from tests.evals.resume_initial_reconciliation import ARecorder, scoring_call

    sessions, tenant, kwargs, _ = ledger
    kwargs["inputs"].budget = ExperimentBudget()
    recorder = ARecorder(sessions, tenant, **kwargs)
    await recorder.initialize()
    recorder.enable_policy()

    class Chat:
        provider = "qwen"
        model = LOCKED_CHAT_MODEL
        calls = 0

        async def invoke(self, messages, tools, metadata, *, attempt):
            self.calls += 1
            assert 290 < attempt.timeout_seconds <= 300
            if self.calls == 1:
                raise ProviderAdapterError(category="provider_timeout", retryable=True)
            return ChatModelResult(
                content='{"ok":true}',
                provider="qwen",
                model=LOCKED_CHAT_MODEL,
                usage=ModelUsage(input_tokens=10, output_tokens=2),
            )

    class Embedding:
        provider = "qwen"
        model = LOCKED_EMBEDDING_MODEL

        async def embed(self, texts, metadata, *, attempt):
            pytest.fail("unexpected embedding")

    adapter = Chat()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=adapter,
        embedding_adapter=Embedding(),
        provider="qwen",
        retry_policy=LLMRetryPolicy(max_attempts=1, deadline_seconds=300),
    )
    context = LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
    path = kwargs["root"] / "a" / "synthetic" / "score"
    result = await scoring_call(
        factory, recorder, context, path, {"synthetic": True}, "initial", "synthetic prompt"
    )
    before = await recorder.measurement()
    assert result.content == '{"ok":true}' and adapter.calls == 2
    assert before["unknown_cost"] == before["unknown_usage"] == 1
    assert (
        await scoring_call(
            factory, recorder, context, path, {"synthetic": True}, "initial", "synthetic prompt"
        )
        == result
    )
    assert await recorder.measurement() == before and adapter.calls == 2
