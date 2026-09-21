"""Paired protocol/failure retention and persistent-admission unit contracts."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment import load_experiment_plan
from tests.evals.quality_experiment_binding import (
    PLAN,
    ROOT,
    BindingV1,
    EnvironmentV1,
    ExperimentError,
    ci_proof,
    read_json,
    write_new,
)
from tests.evals.quality_experiment_execution import (
    AuthorizationV1,
    AuthorizationV2,
    ExperimentHooks,
    SlotV1,
    arm_manifest,
    check_slot,
    configured_policy,
    execute,
    live_identity_factory,
    planned_slots,
    read_authorization,
    summarize_rows,
)
from tests.evals.test_quality_experiment_binding import ci_summary


def binding_fixture(tmp_path):
    plan = load_experiment_plan(ROOT / PLAN)
    environment = EnvironmentV1(
        image_id="sha256:" + "a" * 64,
        docker_version="test",
        host_digest="sha256:" + "b" * 64,
        software_digest="sha256:" + "c" * 64,
        python_version="3.12.13",
        uv_version="0.11.32",
        cpu_count=2,
    )
    return BindingV1(
        experiment_id="e7a7_contract",
        ci=ci_proof(ci_summary()),
        plan_digest=quality_identity_digest(plan.model_dump(mode="json")),
        baseline_source_sha=plan.identity.baseline_source_sha,
        candidate_source_sha="a" * 40,
        baseline_src_tree=plan.identity.baseline_src_tree,
        candidate_src_tree=plan.identity.baseline_src_tree,
        baseline_root=str(tmp_path / "baseline"),
        candidate_root=str(tmp_path / "candidate"),
        harness_root=str(ROOT),
        harness_digest="sha256:" + "d" * 64,
        reviewed_allowed_diff_digest="sha256:" + "e" * 64,
        candidate_prompt_digest="sha256:" + "f" * 64,
        candidate_configuration_digest="sha256:" + "1" * 64,
        candidate_graph_version="pathfinder-research-e7a-exp-v1",
        environment=environment,
        environment_digest=quality_identity_digest(environment.model_dump(mode="json")),
    )


def test_fixed_schedule_has_no_duplicates_and_exact_adjacent_pairs():
    slots = planned_slots(load_experiment_plan(ROOT / PLAN))
    assert len(slots) == len({(s.arm, s.case_id, s.repeat_index) for s in slots}) == 144
    assert [s.ordinal for s in slots] == list(range(144))
    for i in range(0, 144, 2):
        a, b = slots[i : i + 2]
        assert (a.case_id, a.repeat_index) == (b.case_id, b.repeat_index)
        assert a.arm != b.arm


@pytest.mark.parametrize(
    "field,value", [("ordinal", 1), ("arm", "candidate"), ("case_id", "wrong"), ("repeat_index", 1)]
)
def test_slot_identity_cannot_be_reordered(field, value):
    expected = SlotV1(ordinal=0, arm="baseline", case_id="case", repeat_index=0, status="planned")
    with pytest.raises(ExperimentError, match="slot_order_mismatch"):
        check_slot(expected, expected.model_copy(update={field: value}))


def test_budget_totals_include_failed_started_and_unknown_attempts():
    rows = [
        NS(
            id=uuid4(),
            status="succeeded",
            estimated_cost=Decimal("1.25"),
            token_usage={"input_tokens": 10, "output_tokens": 3},
        ),
        NS(id=uuid4(), status="failed", estimated_cost=None, token_usage=None),
        NS(id=uuid4(), status="started", estimated_cost=None, token_usage=None),
    ]
    total = summarize_rows(rows)
    assert (
        total.attempts,
        total.started,
        total.unknown_cost_attempts,
        total.unknown_usage_attempts,
    ) == (3, 1, 2, 2)
    assert total.known_cost_cny == Decimal("1.25")
    assert (total.input_tokens, total.output_tokens) == (10, 3)


async def test_identity_factory_cannot_make_a_provider_call():
    factory = live_identity_factory(_StrictMemoryInvocationRecorder())
    assert factory.provider == "qwen"
    with pytest.raises(ExperimentError, match="not_executable"):
        await factory.chat_adapter.invoke((), (), {})


def test_manifest_retains_frozen_baseline_and_budget(tmp_path):
    binding = binding_fixture(tmp_path)
    factory = live_identity_factory(_StrictMemoryInvocationRecorder())
    manifest, policy = arm_manifest(binding, "baseline", factory)
    assert manifest.execution_source_sha == binding.baseline_source_sha
    assert len(manifest.execution_order) == 72
    assert manifest.cost_admission_budget_cny == 30 and manifest.provider_attempt_cap == 1200
    assert (manifest.input_token_cap, manifest.output_token_cap) == (4500000, 600000)
    assert policy == configured_policy()


async def test_budget_source_failure_happens_before_pg_or_provider(tmp_path):
    def changed():
        raise ExperimentError("identity_drift")

    hooks = ExperimentHooks(tmp_path, source_check=changed)
    with pytest.raises(ExperimentError, match="identity_drift"):
        await hooks.before_attempt(None)
    assert hooks.failure == "identity_drift"
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("fault", [None, "stopped", "wrong_index", "cancelled"])
async def test_coordinator_fixed_order_and_failure_preservation(
    tmp_path, monkeypatch, fault, version
):
    import tests.evals.quality_experiment_execution as module

    tmp_path.chmod(0o700)
    binding = binding_fixture(tmp_path)
    binding_path = tmp_path / "binding.json"
    write_new(binding_path, binding)
    auth = (AuthorizationV1 if version == 1 else AuthorizationV2)(
        binding_digest=quality_identity_digest(binding.model_dump(mode="json")),
        source="user_explicit_e7a7_live_and_agent_review"
        if version == 1
        else "user_explicit_e7a7_live_only",
        synthetic_materials=True,
        frozen_144_slots=True,
        per_arm_cny=30,
        total_cny=60,
        own_wsl_resources=True,
        agent_initial_and_recheck=version == 1,
    )
    auth_path = tmp_path / "auth.json"
    write_new(auth_path, auth)
    monkeypatch.setattr(module, "verify_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(module, "probe_environment", lambda: binding.environment)
    calls = []

    class Process:
        returncode = None

        def send_signal(self, _):
            self.returncode = 1

        def kill(self):
            self.returncode = 1

        async def wait(self):
            self.returncode = self.returncode or 0
            return self.returncode

    class FakeChild:
        def __init__(self, arm, root):
            self.arm, self.root, self.process, self.message = arm, root, Process(), None

        async def send(self, message):
            self.message = message

        async def receive(self, *args):
            if self.message is None:
                return {"category": "ready"}
            command = self.message["command"]
            if command == "prepare":
                return {"category": "prepared"}
            if command == "slot":
                index = self.message["index"]
                calls.append((self.arm, index))
                if len(calls) == 6 and fault == "cancelled":
                    raise asyncio.CancelledError
                return {
                    "category": "slot",
                    "index": index + (len(calls) == 6 and fault == "wrong_index"),
                    "status": "succeeded",
                    "case_digest": "sha256:" + "a" * 64,
                    "stop": "budget" if len(calls) == 6 and fault == "stopped" else None,
                }
            assert command == "finish"
            (self.root / "outputs" / self.arm).mkdir(mode=0o700)
            write_new(self.root / "outputs" / self.arm / "report.json", {"contract_fixture": True})
            write_new(self.root / self.arm / "resources/report.json", {"contract_fixture": True})
            write_new(
                self.root / self.arm / "cleanup.json",
                {"cleanup_complete": True, "stop_category": None},
            )
            return {"category": "finished"}

    async def launch(path, value, arm, root, credentials):
        return FakeChild(arm, root)

    monkeypatch.setattr(module, "launch_child", launch)
    root = tmp_path / "run"
    report = await execute(binding_path, root, tmp_path / "unused-credentials", auth_path)
    assert len(report.slots) == 144
    if fault is None:
        assert report.execution_complete and len(calls) == 144
        counts = {"baseline": 0, "candidate": 0}
        for slot, called in zip(report.slots, calls, strict=True):
            assert called == (slot.arm, counts[slot.arm])
            counts[slot.arm] += 1
    else:
        assert not report.execution_complete and len(calls) == 6
        assert (root / "slots/start-0005.json").exists()
        assert sum(s.status == "not_run" for s in report.slots) == 138
        assert report.slots[5].status == ("succeeded" if fault == "stopped" else "missing")
    assert read_authorization(auth_path) == auth
    assert not (root / "review").exists()
    assert read_json(root / "execution.json")["execution_complete"] == report.execution_complete
    with pytest.raises(FileExistsError):
        await execute(binding_path, root, tmp_path / "unused-credentials", auth_path)
    with pytest.raises(ExperimentError, match="artifact_publication_failed"):
        await execute(binding_path, tmp_path / "reroll", tmp_path / "unused-credentials", auth_path)
    assert not (tmp_path / "reroll").exists()


async def test_unknown_tokens_stop_before_next_attempt_even_with_cost_reserve(tmp_path):
    hooks = ExperimentHooks(tmp_path)

    async def usage(attempt):
        return NS(started=0, unknown_usage_attempts=1)

    hooks.usage = usage
    with pytest.raises(ExperimentError, match="unknown_provider_usage"):
        await hooks.before_attempt(NS())
    assert hooks.failure == "unknown_provider_usage"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_kind": "future_authorization"},
        {"artifact_kind": "e7a7_run_authorization_v1"},
        {"source": "user_explicit_e7a7_live_and_agent_review"},
        {"agent_initial_and_recheck": True},
        {"agent_initial_and_recheck": "false"},
        {"binding_digest": "sha256:" + "9" * 64},
        {"per_arm_cny": 31},
        {"total_cny": 61},
        {"synthetic_materials": False},
        {"frozen_144_slots": False},
        {"own_wsl_resources": False},
        {"production_adoption": True},
        {"deployment": True},
        {"unexpected_permission": True},
    ],
)
async def test_live_only_authorization_rejected_before_any_execution(tmp_path, monkeypatch, change):
    import tests.evals.quality_experiment_execution as module

    tmp_path.chmod(0o700)
    binding = binding_fixture(tmp_path)
    binding_path = tmp_path / "binding.json"
    write_new(binding_path, binding)
    value = AuthorizationV2(
        binding_digest=quality_identity_digest(binding.model_dump(mode="json")),
        source="user_explicit_e7a7_live_only",
        synthetic_materials=True,
        frozen_144_slots=True,
        per_arm_cny=30,
        total_cny=60,
        own_wsl_resources=True,
        agent_initial_and_recheck=False,
    ).model_dump(mode="json")
    value.update(change)
    auth_path = tmp_path / "authorization.json"
    write_new(auth_path, value)
    monkeypatch.setattr(module, "verify_binding", lambda *a, **kw: binding)
    monkeypatch.setattr(module, "probe_environment", lambda: binding.environment)

    async def unexpected_child(*args):
        pytest.fail("authorization_rejection_started_child")

    monkeypatch.setattr(module, "launch_child", unexpected_child)
    with pytest.raises(ExperimentError):
        await execute(binding_path, tmp_path / "run", tmp_path / "unread-credentials", auth_path)
    assert not (tmp_path / "run").exists()
    assert not list(tmp_path.glob("*.execution-claim.json"))


@pytest.mark.parametrize(
    "value", [None, [], "authorization", {}, {"artifact_kind": "unknown"}, {"artifact_kind": []}]
)
def test_authorization_requires_known_object_version(tmp_path, value):
    path = tmp_path / "authorization.json"
    tmp_path.chmod(0o700)
    write_new(path, value)
    with pytest.raises(ExperimentError):
        read_authorization(path)
