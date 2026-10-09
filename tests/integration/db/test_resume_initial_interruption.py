"""Explicit one-call interruption settlement keeps unknown accounting and never replays."""

import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import update

from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import RunJob
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.tenancy import TenantService
from tests.evals.product_acceptance_contracts import AcceptanceError, read_private_json
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiment_contracts import ExperimentBudget
from tests.evals.resume_experiments import preserve
from tests.evals.resume_initial import usage_for
from tests.evals.resume_initial_fixture import seed
from tests.evals.resume_initial_interruption import AUTHORIZATION, reconcile
from tests.evals.resume_initial_reconciliation import ARecorder
from tests.evals.resume_live_budget import ledger_identity
from tests.integration.db.test_resume_experiments import attempt
from tests.integration.db.test_resume_fault_matrix import claim
from tests.integration.db.test_resume_initial import snapshot_material

pytestmark = pytest.mark.integration


@pytest.fixture
async def rig(migrated_database_url, tmp_path):
    directory, _, _, generation, _, _, pid, _, _, _ = snapshot_material(tmp_path)
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        owner = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace(f"a-interruption-{uuid4()}")
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        fixture = await seed(sessions, tenant, tmp_path, directory)
        store = SqlAlchemyResumeGenerationStore(sessions)
        created = await store.create(
            tenant,
            SessionCreateV1(
                profile_version_id=UUID(fixture["profile_version_id"]),
                preference_version=1,
                project_ids=(pid,),
                job=JobInputV1(source="paste", text=generation.job_text),
                budget=GenerationBudgetV1(
                    max_model_calls=6, max_tool_calls=0, max_cost_cny=Decimal("1")
                ),
            ),
            uuid4(),
        )
        yield SimpleNamespace(
            sessions=sessions,
            tenant=tenant,
            store=store,
            session_id=created.receipt.resource_id,
            jobs=SqlAlchemyWorkerJobStore(sessions, lambda attempt: timedelta(0)),
        )
    finally:
        await engine.dispose()


async def test_exact_interruption_settlement_and_replay_guard(rig, tmp_path):
    tmp_path.chmod(0o700)
    root = tmp_path / "experiment"
    root.mkdir(mode=0o700)
    inputs = SimpleNamespace(allocation_id=uuid4(), digest="synthetic", budget=ExperimentBudget())
    recorder = ARecorder(
        rig.sessions, rig.tenant, root=root, inputs=inputs, database_identity="synthetic"
    )
    assert await recorder.rows() == []
    await recorder.initialize()
    recorder.enable_policy()
    job = await claim(rig)
    row_attempt = attempt(rig.tenant).model_copy(
        update={"run_id": job.run_id, "graph_node": "analyze_job"}
    )
    origin = root / "a" / ("a" * 40)
    directory = root / "a" / ("b" * 40)
    path = origin / "formal/test-r1-pathfinder"
    for folder in (origin.parent, origin, path.parent, path, directory):
        folder.mkdir(mode=0o700)
    before = await recorder.check_admission()
    preserve(path / "call-00-started.json", {"identity": "synthetic", "before": before})
    sample = {
        "sample_id": "test-r1-pathfinder",
        "arm": "pathfinder",
        "case_id": "test",
        "repeat": 1,
        "c_start": True,
    }
    preserve(path / "generation-started.json", {"usage": before, "sample": sample})
    preserve(path / "input.json", {"digest": "synthetic-input"})
    preserve(
        origin / "formal/test-r1-session.json",
        {"run_id": str(job.run_id), "session_id": str(rig.session_id)},
    )
    await recorder.prepare(row_attempt)
    row = (await recorder.rows())[0]
    authorization = {
        "policy": "single_interrupted_generation_no_replay_v1",
        "binding": recorder.binding,
        "authorization": "Synthetic explicit approval",
        "reserved_cost_cny": "10.4623104",
        "invocation_id": str(row.id),
        "ledger": ledger_identity(row),
        "run_id": str(job.run_id),
        "source_sha": origin.name,
        "sample_id": sample["sample_id"],
        "started_path": str((path / "call-00-started.json").relative_to(root)),
        "started_digest": quality_identity_digest(read_private_json(path / "call-00-started.json")),
    }
    preserve(root / AUTHORIZATION, authorization)
    inventory = {
        str(p.relative_to(origin)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in origin.rglob("*.json")
    }
    with pytest.raises(AcceptanceError, match="interruption_worker_may_be_active"):
        await reconcile(recorder, rig.sessions, rig.tenant, origin, directory, inventory, usage_for)
    async with rig.sessions() as db, db.begin():
        await db.execute(
            update(RunJob)
            .where(RunJob.id == job.job_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    await reconcile(recorder, rig.sessions, rig.tenant, origin, directory, inventory, usage_for)
    measured = await recorder.check_admission()
    assert measured["attempts"] == measured["unfinished"] == measured["unknown_cost"] == 1
    assert measured["reserved_unfinished_attempts"] == 1
    assert measured["reserved_cost_cny"] == "10.4623104"
    assert (await rig.store.get_session(rig.tenant, rig.session_id))["run_status"] == "cancelled"
    assert (await recorder.rows())[0].status == "started"
    assert not list(root.glob("accounted-*.json"))
    failed = read_private_json(directory / "formal/test-r1-pathfinder/generation.json")
    assert failed["output"] is None and failed["generation_status"] == "FAILED"
    assert failed["elapsed_seconds"] is None
    await reconcile(recorder, rig.sessions, rig.tenant, origin, directory, inventory, usage_for)
    assert await recorder.check_admission() == measured
    # An unrelated new unknown is never covered by this one-call approval.
    await recorder.prepare(attempt(rig.tenant))
    with pytest.raises(AcceptanceError, match="unfinished_accounting"):
        await recorder.check_admission()


async def test_unbound_nonempty_ledger_remains_rejected(rig, tmp_path):
    from app.db.llm_invocations import SqlAlchemyInvocationRecorder

    root = tmp_path / "unbound"
    root.mkdir(mode=0o700)
    await SqlAlchemyInvocationRecorder(rig.sessions).prepare(attempt(rig.tenant))
    recorder = ARecorder(
        rig.sessions,
        rig.tenant,
        root=root,
        inputs=SimpleNamespace(
            allocation_id=uuid4(), digest="synthetic", budget=ExperimentBudget()
        ),
        database_identity="synthetic",
    )
    with pytest.raises(AcceptanceError, match="unbound_ledger_not_empty"):
        await recorder.initialize()
    assert len(await recorder.rows()) == 1
    assert not (root / "ledger-binding.json").exists()


async def test_authorized_terminal_estimates_resume_without_rewriting_ledger(rig, tmp_path):
    from app.llm.invocations import LLMInvocationOutcome
    from tests.evals.resume_initial_estimates import POLICY_FILE

    root = tmp_path / "estimated-experiment"
    root.mkdir(mode=0o700)
    inputs = SimpleNamespace(
        allocation_id=uuid4(), digest="estimate-test", budget=ExperimentBudget(attempt_cap=2)
    )
    kwargs = dict(root=root, inputs=inputs, database_identity="synthetic")
    recorder = ARecorder(rig.sessions, rig.tenant, **kwargs)
    await recorder.initialize()
    recorder.enable_policy()
    first = attempt(rig.tenant).model_copy(update={"graph_node": "selection"})
    await recorder.prepare(first)
    outcome = LLMInvocationOutcome(
        status="failed", latency_ms=10, error_category="provider_unavailable"
    )
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await recorder.finalize(first, outcome)
    preserve(
        root / POLICY_FILE,
        {
            "version": "a_terminal_unknown_estimate_v1",
            "binding": recorder.binding,
            "authorization": "Synthetic explicit authorization",
            "estimated_cost_cny": "1",
            "historical_invocation_ids": [str(first.invocation_id)],
            "eligible_existing_ids": [str(first.invocation_id)],
        },
    )
    resumed = ARecorder(rig.sessions, rig.tenant, **kwargs)
    await resumed.initialize()
    measured = await resumed.check_admission()
    assert measured["attempts"] == 1 and measured["estimated_unknown_cost_cny"] == "1"
    second = attempt(rig.tenant).model_copy(update={"graph_node": "selection"})
    await resumed.prepare(second)
    await resumed.finalize(second, outcome)
    measured = await resumed.check_admission(after=True)
    assert measured["unknown_cost"] == measured["unknown_usage"] == 2
    assert measured["known_cost_cny"] == "0"
    assert measured["estimated_unknown_cost_cny"] == "2"
    assert await resumed.check_admission(after=True) == measured
    with pytest.raises(AcceptanceError, match="budget_exhausted"):
        await resumed.prepare(attempt(rig.tenant))
    assert len(await resumed.rows()) == 2
    assert all(
        row.estimated_cost is None and row.token_usage is None for row in await resumed.rows()
    )
