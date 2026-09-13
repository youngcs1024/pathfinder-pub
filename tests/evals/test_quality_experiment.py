"""Preregistration integrity and offline/privacy boundaries; no candidate execution."""

import io
import json
import socket
from collections import Counter
from pathlib import Path

import pytest

from tests.evals.quality_experiment import (
    ExperimentPlanError,
    load_experiment_plan,
    main,
    validate_experiment_plan,
)

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "evals/experiments/e63-evidence-sufficiency-v1.json"
CANARY = "private-experiment-error-canary"


@pytest.fixture
def raw():
    return json.loads(PLAN.read_bytes())


def write_plan(tmp_path, raw):
    target = tmp_path / "plan.json"
    target.write_text(json.dumps(raw))
    return target


def test_registered_plan_has_exact_paired_slots_and_fixed_denominators():
    plan = load_experiment_plan(PLAN)
    sample = plan.samples
    assert len(sample.dev_case_ids) == 9
    assert len(sample.validation_case_ids) == 15
    assert len(sample.scope_case_ids) == 3
    assert len(sample.normal_control_case_ids) == 7
    counts = Counter((s.arm, s.case_id) for s in sample.execution_order)
    assert len(counts) == 48 and set(counts.values()) == {3}
    assert (sample.planned_slots, sample.primary_slots_per_arm, sample.semantic_slots_per_arm) == (
        144,
        39,
        63,
    )
    for index in range(0, 144, 2):
        left, right = sample.execution_order[index : index + 2]
        assert (left.case_id, left.repeat_index) == (right.case_id, right.repeat_index)
        case_index = sample.selected_case_ids.index(left.case_id)
        expected = "baseline" if (case_index + left.repeat_index) % 2 == 0 else "candidate"
        assert left.arm == expected and right.arm != expected
    assert plan.budget.per_arm_cny * 2 == plan.budget.total_cny == 60
    assert plan.benefit.minimum_net_reduction == 3
    assert plan.benefit.minimum_improved_case_ids == 2
    assert plan.benefit.minimum_improved_repeats == 2
    assert plan.benefit.maximum_regressed_repeats == 0


@pytest.mark.parametrize(
    "field",
    [
        "accepted_digest",
        "lock_digest",
        "dataset_digest",
        "split_digest",
        "case_set_digest",
        "rubric_digest",
        "freeze_digest",
        "mapping_digest",
        "retrieval_policy_digest",
        "baseline_prompt_digest",
        "baseline_configuration_digest",
    ],
)
def test_identity_digest_mismatch_is_rejected(raw, tmp_path, field):
    raw["identity"][field] = "sha256:" + "0" * 64
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


@pytest.mark.parametrize("field", ["baseline_source_sha", "baseline_src_tree", "candidate_policy"])
def test_source_or_candidate_identity_cannot_be_replaced(raw, tmp_path, field):
    raw["identity"][field] = "a" * 40
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


@pytest.mark.parametrize(
    "change",
    [
        "missing_slot",
        "duplicate_slot",
        "swapped_arms",
        "swapped_cases",
        "new_case",
        "scope_as_semantic",
        "scope_as_validation",
        "removed_normal_control",
        "different_split",
        "relabeled_primary_denominator",
        "wrong_repeat",
        "different_capacity_evidence",
    ],
)
def test_selection_order_and_negative_scope_cannot_be_gamed(raw, tmp_path, change):
    sample = raw["samples"]
    slots = sample["execution_order"]
    if change == "missing_slot":
        slots.pop()
    elif change == "duplicate_slot":
        slots[-1] = slots[0]
    elif change == "swapped_arms":
        slots[0], slots[1] = slots[1], slots[0]
    elif change == "swapped_cases":
        slots[0:4] = slots[2:4] + slots[0:2]
    elif change == "new_case":
        sample["selected_case_ids"][0] = "gpu_gap"
    elif change == "scope_as_semantic":
        sample["scope_case_ids"].remove("scope_profile")
    elif change == "scope_as_validation":
        sample["validation_case_ids"].append("scope_foreign")
    elif change == "removed_normal_control":
        sample["normal_control_case_ids"].pop()
    elif change == "different_split":
        sample["dev_case_ids"].append(sample["validation_case_ids"].pop())
    elif change == "relabeled_primary_denominator":
        sample["primary_slots_per_arm"] = 45
    elif change == "wrong_repeat":
        slots[0]["repeat_index"] = 1
    else:
        raw["capacity_evidence"].pop("faults_e59_v1")
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("benefit", "minimum_net_reduction", 2),
        ("benefit", "minimum_improved_case_ids", 1),
        ("benefit", "minimum_improved_repeats", 1),
        ("benefit", "maximum_regressed_repeats", 1),
        ("benefit", "incomplete_evidence", "ignore_missing"),
        ("budget", "per_arm_cny", 31),
        ("budget", "total_cny", 61),
        ("budget", "per_arm_provider_attempts", 1201),
        ("budget", "per_arm_input_tokens", 4500001),
        ("budget", "per_arm_output_tokens", 600001),
        ("budget", "additional_dependencies", 1),
        ("budget", "model_calls_per_run", 13),
        ("budget", "tool_calls_per_run", 9),
        ("budget", "research_passes_per_run", 3),
        ("budget", "assessment_calls_per_pass", 2),
        ("budget", "transfer_between_arms", True),
        ("resources", "cost_increase_percent_max", 26),
        ("resources", "input_token_increase_percent_max", 26),
        ("resources", "output_token_increase_percent_max", 26),
        ("resources", "generation_p95_increase_percent_max", 26),
        ("resources", "generation_p95_increase_seconds_max", 21),
        ("resources", "memory_increase_bytes_max", 268435457),
        ("resources", "database_memory_bytes", 4294967296),
        ("resources", "runner_tree_rss_bytes", 4294967296),
        ("resources", "execution_slots", 2),
        ("resources", "unknown_cost", "treat_as_zero"),
        ("resources", "case_timeout_seconds", 601),
        ("resources", "execution_window_seconds", 14401),
        ("non_regression", "safety_violation_max", 1),
        ("non_regression", "unnecessary_refusal", "ignore_refusals"),
        ("review", "gold_in_model_context", True),
        ("review", "e7_j", True),
        ("exit_rules", "reroll", True),
        ("exit_rules", "case_replacement", True),
        ("exit_rules", "lower_threshold_after_observation", True),
        ("identity", "execution_blocked_until", "ready"),
    ],
)
def test_approved_budgets_thresholds_and_boundaries_cannot_be_relaxed(
    raw, tmp_path, section, field, replacement
):
    raw[section][field] = replacement
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


@pytest.mark.parametrize("section", ["identity", "non_regression", "exit_rules"])
def test_required_guard_cannot_be_dropped_or_unknown_change_added(raw, tmp_path, section):
    raw[section].pop(next(iter(raw[section])))
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


def test_broader_candidate_diff_is_rejected(raw, tmp_path):
    raw["identity"]["allowed_changes"].append("hybrid_retrieval")
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


def test_constructed_models_are_revalidated():
    plan = load_experiment_plan(PLAN)
    altered = plan.model_copy(update={"budget": plan.budget.model_copy(update={"per_arm_cny": 99})})
    with pytest.raises(ExperimentPlanError):
        validate_experiment_plan(altered)


@pytest.mark.parametrize("resource", ["uv.lock", "accepted", "dataset", "freeze", "mapping"])
def test_changed_source_bytes_are_rejected(monkeypatch, resource):
    plan = load_experiment_plan(PLAN)
    target = {
        "uv.lock": ROOT / "uv.lock",
        "accepted": ROOT / "evals/baselines/quality/e410-agent-v1.json",
        "dataset": ROOT / "evals/datasets/quality_expanded_v1/rubric.json",
        "freeze": ROOT / "evals/datasets/quality_expanded_v1/freeze.json",
        "mapping": ROOT / "evals/datasets/quality_expanded_v1/mapping.json",
    }[resource]
    original_open = Path.open
    injected = []

    def changed(path, *args, **kwargs):
        if path == target:
            injected.append(path)
            return io.BytesIO(b"{}")
        return original_open(path, *args, **kwargs)

    # Both bounded dataset reads and read_bytes use open; assert the fault actually ran.
    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(ExperimentPlanError):
        validate_experiment_plan(plan)
    assert injected == [target]


@pytest.mark.parametrize(
    "mutation", ["unknown_root", "unknown_nested", "boolean_integer", "float_integer"]
)
def test_strict_unknown_fields_and_json_types(raw, tmp_path, mutation):
    if mutation == "unknown_root":
        raw[CANARY] = CANARY
    elif mutation == "unknown_nested":
        raw["identity"][CANARY] = CANARY
    elif mutation == "boolean_integer":
        raw["experiment_only"] = 1
    else:
        raw["budget"]["per_arm_cny"] = 30.0
    with pytest.raises(ExperimentPlanError):
        load_experiment_plan(write_plan(tmp_path, raw))


def test_duplicate_json_keys_are_rejected(tmp_path):
    target = tmp_path / "plan.json"
    target.write_text('{"plan_id":"first","plan_id":"second"}')
    with pytest.raises(ExperimentPlanError, match=r"^invalid_experiment_plan$"):
        load_experiment_plan(target)


def test_cli_is_read_only_offline_and_does_not_imply_execution_readiness(monkeypatch, capsys):
    def prohibited(*args, **kwargs):
        pytest.fail("validation attempted an external action")

    monkeypatch.setattr(socket, "create_connection", prohibited)
    monkeypatch.setattr(socket.socket, "connect", prohibited)
    monkeypatch.setattr(Path, "write_text", prohibited)
    monkeypatch.setattr(Path, "write_bytes", prohibited)
    monkeypatch.setenv("DASHSCOPE_API_KEY", CANARY)
    monkeypatch.setenv("PF_LLM_MODE", "qwen")
    assert main(["validate", "--plan", str(PLAN)]) == 0
    output = capsys.readouterr()
    value = json.loads(output.out)
    assert value["category"] == "preregistration_valid"
    assert value["execution_readiness"] == "not_checked_requires_binding_and_authorization"
    assert (value["planned_slots"], value["primary_slots_per_arm"]) == (144, 39)
    assert value["digest"] == validate_experiment_plan(load_experiment_plan(PLAN))
    assert not output.err and CANARY not in output.out


@pytest.mark.parametrize("kind", ["arguments", "path", "json", "schema", "source"])
def test_cli_failures_never_echo_private_content(raw, tmp_path, capsys, monkeypatch, kind):
    target = tmp_path / CANARY
    args = ["validate", "--plan", str(target)]
    if kind == "arguments":
        args = ["run", CANARY]
    elif kind == "json":
        target.write_text(CANARY)
    elif kind == "schema":
        raw["budget"]["per_arm_cny"] = CANARY
        target.write_text(json.dumps(raw))
    elif kind == "source":
        args = ["validate", "--plan", str(PLAN)]
        monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError(CANARY)))
    assert main(args) == 1
    output = capsys.readouterr()
    assert set(json.loads(output.out)) == {"category"}
    assert CANARY not in output.out + output.err
    assert not output.err
