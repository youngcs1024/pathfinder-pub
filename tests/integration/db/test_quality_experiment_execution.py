"""Real PG accounting and isolated fake child execution; no live or adoption claims."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.db.models import ActionIntent, ApprovalRequest, LLMInvocation
from tests.evals.quality_experiment_binding import ExperimentError, source_identity
from tests.evals.quality_experiment_database import OwnedExperimentDatabase, require_owned_database
from tests.evals.quality_experiment_execution import ExperimentHooks
from tests.evals.quality_generation_fixtures import generation_inputs
from tests.evals.quality_run import (
    finish_quality_generation,
    prepare_quality_generation,
    run_quality_generation_slot,
)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
async def owned():
    manager = OwnedExperimentDatabase()
    async with manager as handle:
        yield handle
    assert not manager.cleanup_failed


def arguments(tmp_path, name, arm="candidate"):
    private = tmp_path / "private"
    private.mkdir(mode=0o700, exist_ok=True)
    accounting = tmp_path / name
    accounting.mkdir(mode=0o700)
    hooks = ExperimentHooks(accounting)
    args = generation_inputs(
        tmp_path / f"output_{name}", private, arm=arm, selected=["mixed_alpha"], experiment_id=name
    )
    return hooks, args


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
async def test_owned_pg_hooks_include_ingestion_and_final_attempts(owned, tmp_path, arm):
    hooks, args = arguments(tmp_path, "normal", arm)
    state = await prepare_quality_generation(owned_database=owned, experiment_hooks=hooks, **args)
    before = await hooks.usage()
    assert before.attempts > 0  # Real fake embedding accounting, not a free preparation phase.
    result = await run_quality_generation_slot(state, 0)
    report = await finish_quality_generation(state)
    after = await hooks.usage()
    assert result.observation.status == "succeeded"
    assert after.attempts == report.total_usage.provider_attempts > before.attempts
    assert after.unknown_cost_attempts == report.total_usage.cost.unknown_cost_attempts
    assert after.started == 0
    diagnostic = json.loads((tmp_path / "normal/diagnostic-0000.json").read_text())
    assert diagnostic["output_digest"] == result.observation.output_digest
    assert diagnostic["statistics"] is not None
    async with require_owned_database(owned)._sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ActionIntent)) == 0
        assert await session.scalar(select(func.count()).select_from(ApprovalRequest)) == 0
        assert (
            await session.scalar(select(func.count()).select_from(LLMInvocation)) == after.attempts
        )


async def test_persistent_budget_survives_new_hook_and_blocks_before_http(owned, tmp_path):
    hooks, args = arguments(tmp_path, "budget")
    state = await prepare_quality_generation(owned_database=owned, experiment_hooks=hooks, **args)
    usage = await hooks.usage()
    second = tmp_path / "second"
    second.mkdir(mode=0o700)
    replacement = ExperimentHooks(second)
    replacement.bind(owned, args["manifest"], args["policy"])
    replacement.admission = replacement.admission.model_copy(
        update={"provider_attempt_cap": usage.attempts}
    )
    attempt = SimpleNamespace(
        workspace_id=state.tenant.workspace_id, actor_user_id=state.tenant.actor_user_id
    )
    with pytest.raises(ExperimentError, match="budget_exhausted"):
        await replacement.before_attempt(attempt)
    assert (await replacement.usage()).attempts == usage.attempts
    assert not list(second.iterdir())
    state.stop = "budget"
    await finish_quality_generation(state)


async def test_foreign_actor_does_not_borrow_existing_arm_budget(owned, tmp_path):
    from uuid import uuid4

    hooks, args = arguments(tmp_path, "tenant")
    state = await prepare_quality_generation(owned_database=owned, experiment_hooks=hooks, **args)
    before = await hooks.usage()
    with pytest.raises(ExperimentError, match="budget_tenant_mismatch"):
        await hooks.before_attempt(
            SimpleNamespace(workspace_id=state.tenant.workspace_id, actor_user_id=uuid4())
        )
    assert (await hooks.usage()).attempts == before.attempts
    state.stop = "integrity"
    await finish_quality_generation(state)


CHILD = """import asyncio, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.pycache_prefix = sys.argv[3] + "/unwritten-bytecode"
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[2]]
from tests.evals.quality_experiment_binding import verify_imports, source_identity
from tests.evals.quality_experiment_database import OwnedExperimentDatabase
from tests.evals.quality_experiment_execution import ExperimentHooks
from tests.evals.quality_generation_fixtures import generation_inputs
from tests.evals.quality_run import (
    prepare_quality_generation, run_quality_generation_slot, finish_quality_generation,
)
async def run():
    source, harness, root = map(Path, sys.argv[1:4])
    arm, sha = sys.argv[4:6]
    source_identity(source, sha)
    (root / "private").mkdir(mode=0o700)
    (root / "accounting").mkdir(mode=0o700)
    manager = OwnedExperimentDatabase()
    async with manager as handle:
        args = generation_inputs(root / "public", root / "private", arm=arm,
            selected=["mixed_alpha"], experiment_id="child_contract")
        hooks = ExperimentHooks(root / "accounting")
        state = await prepare_quality_generation(
            owned_database=handle, experiment_hooks=hooks, **args)
        result = await run_quality_generation_slot(state, 0)
        report = await finish_quality_generation(state)
        count = verify_imports(source, harness)
        assert result.observation.status == "succeeded" and report.measurement_complete
        assert count > 0
    assert not manager.cleanup_failed
    print(json.dumps({"status":"contract_success", "arm":arm, "imports":count}))
asyncio.run(run())
"""


async def test_two_real_source_roots_use_isolated_processes_and_owned_pg(tmp_path):
    # Local Git fixtures work in shallow CI without fetching history during a test.
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG"}}
    env.update(
        PF_LLM_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_AUTH_MODE="fake",
        PF_TRACE_MODE="off",
        TESTCONTAINERS_RYUK_DISABLED="true",
    )
    for arm in ("baseline", "candidate"):
        source = tmp_path / f"source_{arm}"
        source.mkdir()
        shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))

        def git(*args, source=source):
            return (
                subprocess.check_output(
                    ["git", "-C", str(source), *args], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )

        git("init")
        git("add", "src")
        git(
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            arm,
        )
        sha = git("rev-parse", "HEAD")
        source_identity(source, sha)
        output = tmp_path / f"run_{arm}"
        output.mkdir(mode=0o700)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            CHILD,
            str(source),
            str(ROOT),
            str(output),
            arm,
            sha,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, _stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=120)
        except TimeoutError:
            process.send_signal(signal.SIGINT)  # asyncio.run cancels and closes owned PG.
            try:
                await asyncio.wait_for(asyncio.shield(communication), timeout=30)
            except TimeoutError:
                process.kill()
                await communication
                pytest.fail("child_timeout_cleanup_unconfirmed")
            pytest.fail("child_contract_timeout")
        assert process.returncode == 0, (
            f"child contract failed: arm={arm}, exit={process.returncode}"
        )
        result = json.loads(stdout)
        assert result["status"] == "contract_success" and result["imports"] > 0
