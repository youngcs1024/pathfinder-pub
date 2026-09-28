"""Small D CI matrix runs the same worker, subprocess and connection fault harness."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from app.db.session import create_database_engine, create_session_factory
from tests.evals.product_acceptance_contracts import publish
from tests.evals.resume_recovery_contracts import cases
from tests.evals.resume_recovery_faults import ledger, run_case
from tests.evals.resume_recovery_fixture import prepare, snapshot

pytestmark = pytest.mark.integration
SMALL = [
    c
    for c in cases()
    if (c["kind"] == "controlled" and c["ordinal"] == 0)
    or (c["kind"] == "real" and c["ordinal"] == 90)
    or (c["kind"] == "blocking" and c["ordinal"] in (0, 1))
]


@pytest.mark.parametrize("case", SMALL, ids=[c["case_id"] for c in SMALL])
async def test_d_worker_fault_matrix(migrated_database_url, tmp_path: Path, case):
    root = tmp_path / case["case_id"]
    root.mkdir(mode=0o700)
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        state = await prepare(sessions, root, case)
        result = await run_case(migrated_database_url, sessions, root, state)
        assert result["status"] == "PASS", result
        assert result["old_owner_rejected"] and result["duplicate_side_effects"] == 0
        assert result["model_attempts"] == len(await ledger(sessions, state))
        publish(root / "result.json", result)
        assert await prepare(sessions, root, case) == state
        before = await snapshot(sessions, state)
        # Controller restart with a published result must not need another invocation.
        replayed = await run_case(migrated_database_url, sessions, root, state)
        assert replayed["status"] == "PASS"
        assert await snapshot(sessions, state) == before
        assert replayed["model_attempts"] == result["model_attempts"]
    finally:
        await engine.dispose()


@pytest.mark.parametrize("new_paid_state", ["unchanged", "unfinished", "unknown", "known"])
async def test_d_audit_preserves_original_allocation(
    migrated_database_url, tmp_path, new_paid_state
):
    from types import SimpleNamespace
    from uuid import uuid4

    from app.db.llm_invocations import SqlAlchemyInvocationRecorder
    from app.db.provisioning import SqlAlchemyProvisioningStore
    from app.db.tenancy import SqlAlchemyTenantResolver
    from app.domain.provisioning import ProvisioningService
    from app.domain.tenancy import TenantService
    from app.llm.invocations import LLMInvocationOutcome
    from tests.evals.product_acceptance_contracts import AcceptanceError
    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_experiment_budget import ExperimentRecorder
    from tests.evals.resume_experiment_contracts import ExperimentBudget
    from tests.evals.resume_recovery import paid_audit
    from tests.integration.db.test_resume_experiments import attempt, known

    tmp_path.chmod(0o700)
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
        tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
        owner = await provisioning.provision_personal_workspace(f"original-{uuid4()}")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        inputs = SimpleNamespace(
            allocation_id=uuid4(), digest="d-audit-input", budget=ExperimentBudget()
        )
        database_identity = {"synthetic_database": str(uuid4())}
        publish(tmp_path / "database-owner.json", database_identity)
        recorder = ExperimentRecorder(
            sessions,
            tenant,
            root=tmp_path,
            inputs=inputs,
            database_identity=quality_identity_digest(database_identity),
        )
        await recorder.initialize()
        first = attempt(tenant)
        await recorder.prepare(first)
        await recorder.finalize(first, known())
        frozen = {"usage": await recorder.measurement()}
        assert await paid_audit(sessions, tmp_path, inputs, frozen) == frozen["usage"]
        fake_owner = await provisioning.provision_personal_workspace(f"d-fake-{uuid4()}")
        fake_tenant = await tenancy.resolve_tenant(
            workspace_id=fake_owner.workspace_id, actor_user_id=fake_owner.user_id
        )
        fake = attempt(fake_tenant).model_copy(update={"provider": "fake"})
        delegate = SqlAlchemyInvocationRecorder(sessions)
        await delegate.prepare(fake)
        await delegate.finalize(
            fake,
            known().model_copy(
                update={"pricing_version": None, "currency": None, "estimated_cost": None}
            ),
        )
        assert await paid_audit(sessions, tmp_path, inputs, frozen) == frozen["usage"]
        if new_paid_state != "unchanged":
            extra = attempt(tenant)
            await recorder.prepare(extra)
            if new_paid_state == "unknown":
                with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
                    await recorder.finalize(
                        extra,
                        LLMInvocationOutcome(
                            status="failed", latency_ms=1, error_category="provider_timeout"
                        ),
                    )
            elif new_paid_state == "known":
                await recorder.finalize(extra, known())
            with pytest.raises(
                AcceptanceError,
                match=r"unfinished_accounting|unknown_usage_or_cost|paid_ledger_changed",
            ):
                await paid_audit(sessions, tmp_path, inputs, frozen)
    finally:
        await engine.dispose()
