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


@pytest.mark.parametrize(
    "entrypoint,invalid_response",
    [("diagnostic", False), ("diagnostic", True), ("child", True), ("child-abort", False)],
)
async def test_bounded_embedding_diagnostic_uses_real_pg_and_preserves_failure(
    tmp_path, entrypoint, invalid_response
):
    # Use the actual isolated-runner boundary: the complete pytest suite is not an arm.
    script = """import asyncio, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1] + '/src', sys.argv[1]]
import pytest
from tests.integration.db.test_quality_experiment_execution import _diagnostic_contract
async def run():
    with pytest.MonkeyPatch.context() as patch:
        await _diagnostic_contract(Path(sys.argv[2]), patch, sys.argv[4] == 'true', sys.argv[3])
    print('diagnostic_contract_passed')
asyncio.run(run())
"""
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG"}}
    env.update(
        PF_LLM_MODE="fake",
        PF_SEARCH_MODE="fake",
        PF_AUTH_MODE="fake",
        PF_TRACE_MODE="off",
        TESTCONTAINERS_RYUK_DISABLED="true",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        "-c",
        script,
        str(ROOT),
        str(tmp_path),
        entrypoint,
        str(invalid_response).lower(),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), 120)
    except TimeoutError:
        process.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(asyncio.shield(communication), 60)
        except TimeoutError:
            process.kill()
            await communication
        pytest.fail("diagnostic_subprocess_timeout_cleanup_unconfirmed")
    assert process.returncode == 0, stderr.decode()[-2500:]
    assert b"diagnostic_contract_passed" in stdout


async def _diagnostic_contract(tmp_path, monkeypatch, invalid_response, entrypoint):
    import docker
    import httpx
    from pydantic import SecretStr

    from app.llm import qwen_adapters
    from tests.evals import quality_experiment_execution as module
    from tests.evals import quality_pilot
    from tests.evals.quality_experiment_database import POSTGRES_IMAGE
    from tests.evals.test_quality_experiment_execution import (
        binding_fixture,
        diagnostic_authorization,
    )

    tmp_path.chmod(0o700)
    client = docker.from_env()
    try:
        image = client.images.get(POSTGRES_IMAGE).id
    finally:
        client.close()
    binding = binding_fixture(tmp_path)
    binding = binding.model_copy(
        update={"environment": binding.environment.model_copy(update={"image_id": image})}
    )
    module.write_new(tmp_path / "binding.json", binding)
    module.write_new(tmp_path / "auth.json", diagnostic_authorization(binding))
    monkeypatch.setattr(module, "verify_binding", lambda *a, **kw: None)
    monkeypatch.setattr(module, "verify_imports", lambda *a, **kw: None)
    monkeypatch.setattr(module, "source_identity", lambda *a, **kw: None)
    monkeypatch.setattr(module, "probe_environment", lambda: binding.environment)
    monkeypatch.setattr(
        quality_pilot,
        "credentials",
        lambda _: {
            "DASHSCOPE_API_KEY": SecretStr("fixture-key"),
            "PF_QWEN_WORKSPACE_ID": SecretStr("fixture-space"),
        },
    )
    calls = []

    def transport(request):
        payload = json.loads(request.content)
        calls.append(payload["input"])
        usage = {"total_tokens": 10}
        if not invalid_response:
            usage["prompt_tokens"] = 10
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-v4",
                "object": "list",
                "usage": usage,
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 1536}
                    for i in range(len(payload["input"]))
                ],
            },
        )

    original = qwen_adapters.create_qwen_adapters

    def adapters(**kwargs):
        kwargs["http_async_client"]._transport = httpx.MockTransport(transport)
        kwargs["http_async_client"]._mounts = {}
        return original(**kwargs)

    monkeypatch.setattr(qwen_adapters, "create_qwen_adapters", adapters)
    original_owner = module.OwnedExperimentDatabase
    closed = []

    class CheckedOwner(original_owner):
        async def aclose(self):
            if self._sessions is not None:
                async with self._sessions() as session:
                    assert await session.scalar(select(func.count()).select_from(ActionIntent)) == 0
                    assert (
                        await session.scalar(select(func.count()).select_from(ApprovalRequest)) == 0
                    )
            await super().aclose()
            closed.append(not self.cleanup_failed)

    monkeypatch.setattr(module, "OwnedExperimentDatabase", CheckedOwner)
    root = tmp_path / "diagnostic"
    if entrypoint.startswith("child"):
        root.mkdir(mode=0o700)
        for name in ("baseline", "outputs", "private"):
            (root / name).mkdir(mode=0o700)
        for name in ("resources", "accounting"):
            (root / "baseline" / name).mkdir(mode=0o700)

        async def prepare(*args):
            from time import monotonic

            if entrypoint == "child-abort":
                return {"command": "abort"}
            return {"command": "prepare", "deadline": monotonic() + 300}

        replies = []

        def reply(category, **fields):
            if category == "failed":
                assert closed == [True]
                assert (root / "baseline/cleanup.json").is_file()
                if entrypoint == "child-abort":
                    assert (
                        module.read_json(root / "baseline/failure.json")["category"] == "cancelled"
                    )
                    return replies.append(category)
                assert (root / "baseline/resources/report.json").is_file()
                assert (root / "baseline/failure-usage.json").is_file(), module.read_json(
                    root / "baseline/cleanup.json"
                )
            replies.append(category)

        monkeypatch.setattr(module, "_input", prepare)
        monkeypatch.setattr(module, "_reply", reply)
        try:
            result = await module.child_main(
                str(tmp_path / "binding.json"),
                "baseline",
                str(root),
                str(tmp_path / "unused"),
                str(os.getpid()),
            )
        finally:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
        assert result == 1 and replies == ["ready", "failed"]
        assert len(calls) == (0 if entrypoint == "child-abort" else 1)
        assert module.read_json(root / "baseline/cleanup.json")["cleanup_complete"]
        return
    report = await module.diagnose_embedding(
        tmp_path / "binding.json", root, tmp_path / "unused", tmp_path / "auth.json"
    )
    assert closed == [True] and report["cleanup"]["cleanup_complete"]
    assert report["complete"] is (not invalid_response), report
    assert len(calls) == (1 if invalid_response else 3), report
    facts = module.read_json(root / "baseline/accounting/invocations.json")["invocations"]
    assert len(facts) == len(calls)
    if invalid_response:
        assert report["stop_category"] == "unknown_provider_usage"
        assert [s["status"] for s in report["slots"]] == ["failed", "not_run", "not_run"]
        assert facts[0]["error_category"] == "invalid_provider_response"
        assert facts[0]["token_usage"] is None and facts[0]["estimated_cost"] is None
    else:
        assert all(row["status"] == "succeeded" for row in facts)
    with pytest.raises(FileExistsError):
        await module.diagnose_embedding(
            tmp_path / "binding.json", root, tmp_path / "unused", tmp_path / "auth.json"
        )
    with pytest.raises(ExperimentError, match="artifact_publication_failed"):
        await module.diagnose_embedding(
            tmp_path / "binding.json",
            tmp_path / "reroll",
            tmp_path / "unused",
            tmp_path / "auth.json",
        )
    assert len(calls) == (1 if invalid_response else 3), report
