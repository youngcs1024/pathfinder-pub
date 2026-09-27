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
