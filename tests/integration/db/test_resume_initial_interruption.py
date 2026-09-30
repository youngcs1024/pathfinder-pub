"""Explicit one-call interruption settlement keeps unknown accounting and never replays."""

import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import update

from app.db.models import RunJob
from tests.evals.product_acceptance_contracts import AcceptanceError, read_private_json
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiment_contracts import ExperimentBudget
from tests.evals.resume_experiments import preserve
from tests.evals.resume_initial import usage_for
from tests.evals.resume_initial_interruption import AUTHORIZATION, reconcile
from tests.evals.resume_initial_reconciliation import ARecorder
from tests.evals.resume_live_budget import ledger_identity
from tests.integration.db.test_resume_experiments import attempt
from tests.integration.db.test_resume_fault_matrix import claim, rig  # noqa: F401

pytestmark = pytest.mark.integration


async def test_exact_interruption_settlement_and_replay_guard(rig, tmp_path):  # noqa: F811
    tmp_path.chmod(0o700)
    root = tmp_path / "experiment"
    root.mkdir(mode=0o700)
    inputs = SimpleNamespace(allocation_id=uuid4(), digest="synthetic", budget=ExperimentBudget())
    recorder = ARecorder(
        rig.sessions, rig.tenant, root=root, inputs=inputs, database_identity="synthetic"
    )
    await recorder.initialize()
    recorder.enable_policy()
    job = await claim(rig)
    row_attempt = attempt(rig.tenant).model_copy(
        update={"run_id": job.run_id, "graph_node": "analyze_job"}
    )
    origin = root / "a" / ("a" * 40)
    directory = root / "a" / ("b" * 40)
    path = origin / "formal/test-r1-pathfinder"
    path.mkdir(mode=0o700, parents=True)
    directory.mkdir(mode=0o700)
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
