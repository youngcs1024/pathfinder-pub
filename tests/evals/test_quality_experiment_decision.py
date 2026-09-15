"""Hand-computed frozen adoption gates; fixtures never constitute live evidence."""

from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from tests.evals.quality_experiment import load_experiment_plan
from tests.evals.quality_experiment_binding import PLAN, ROOT
from tests.evals.quality_experiment_decision import frozen_rules, macro_support, score_projection
from tests.evals.quality_generation_support import ExperimentGenerationReportV1
from tests.evals.test_quality_score import inputs


def comparison_fixture():
    plan = load_experiment_plan(ROOT / PLAN)
    primary = [c for c in plan.samples.validation_case_ids if c not in plan.samples.scope_case_ids]
    scores = {}
    for arm in ("baseline", "candidate"):
        cases = []
        for case_id in plan.samples.selected_case_ids:
            for repeat in range(3):
                overclaim = arm == "baseline" and case_id in primary[:2] and repeat < 2
                cases.append(
                    NS(
                        observation=NS(
                            case_id=case_id,
                            repeat_index=repeat,
                            status="succeeded",
                            stage_times=[NS(stage="total", seconds=10.0)],
                        ),
                        assessment_complete=True,
                        insufficiency="overclaim" if overclaim else "appropriate",
                        facts=NS(
                            supported=1,
                            unsupported=0,
                            contradicted=0,
                            unassessed=0,
                            not_assessable=0,
                        ),
                        citations=NS(
                            supported=1,
                            unsupported=0,
                            contradicted=0,
                            unassessed=0,
                            not_assessable=0,
                        ),
                        business_success=not overclaim,
                    )
                )
        scores[arm] = NS(cases=cases, total_cost=NS(total_cost_cny=Decimal("10")))
    resources = {a: NS(complete=True, peak_combined_bytes=1000) for a in scores}
    return plan, scores, resources


def by_name(plan, scores, resources):
    return {r.name: r for r in frozen_rules(plan, scores, resources)}


def test_four_fewer_overclaims_across_two_cases_and_repeats_pass():
    plan, scores, resources = comparison_fixture()
    rules = by_name(plan, scores, resources)
    assert all(r.passed is True for r in rules.values())
    assert (rules["minimum_net_reduction"].baseline, rules["minimum_net_reduction"].candidate) == (
        4,
        0,
    )
    assert rules["independent_case_improvement"].candidate == 2


def test_three_net_improvements_is_inclusive_boundary():
    plan, scores, resources = comparison_fixture()
    row = next(c for c in scores["baseline"].cases if c.insufficiency == "overclaim")
    row.insufficiency = "appropriate"
    assert by_name(plan, scores, resources)["minimum_net_reduction"].passed is True
    row = next(c for c in scores["baseline"].cases if c.insufficiency == "overclaim")
    row.insufficiency = "appropriate"
    assert by_name(plan, scores, resources)["minimum_net_reduction"].passed is False


def test_one_repeat_worse_blocks_even_with_net_benefit():
    plan, scores, resources = comparison_fixture()
    case = next(
        c
        for c in scores["candidate"].cases
        if c.observation.case_id in plan.samples.validation_case_ids
        and c.observation.case_id not in plan.samples.scope_case_ids
        and c.observation.repeat_index == 2
    )
    case.insufficiency = "overclaim"
    rules = by_name(plan, scores, resources)
    assert rules["minimum_net_reduction"].passed
    assert not rules["repeat_non_regression"].passed


def test_increased_refusal_does_not_count_as_acceptable_improvement():
    plan, scores, resources = comparison_fixture()
    case = next(
        c
        for c in scores["candidate"].cases
        if c.observation.case_id in plan.samples.validation_case_ids
    )
    case.insufficiency = "unnecessary_refusal"
    rules = by_name(plan, scores, resources)
    assert rules["minimum_net_reduction"].passed
    assert not rules["validation_unnecessary_refusal"].passed
    assert not rules["semantic_unnecessary_refusal"].passed


@pytest.mark.parametrize("field", ["facts", "citations"])
def test_support_is_per_output_macro_not_pooled(field):
    _plan, scores, _ = comparison_fixture()
    cases = scores["candidate"].cases[:2]
    getattr(cases[0], field).supported = 9
    getattr(cases[0], field).unsupported = 1
    getattr(cases[1], field).supported = 1
    getattr(cases[1], field).unsupported = 1
    assert macro_support(cases, field) == Decimal("0.7")
    getattr(cases[1], field).supported = 0
    getattr(cases[1], field).unsupported = 0
    assert macro_support(cases, field) is None


@pytest.mark.parametrize("kind", ["unassessed", "failed", "missing", "unknown_inventory"])
def test_incomplete_semantic_evidence_never_yields_improvement(kind):
    plan, scores, resources = comparison_fixture()
    row = next(
        c
        for c in scores["candidate"].cases
        if c.observation.case_id in plan.samples.validation_case_ids
    )
    if kind == "unassessed":
        row.assessment_complete = False
    elif kind == "failed":
        row.observation.status = "failed"
    elif kind == "missing":
        scores["candidate"].cases.remove(row)
    else:
        row.facts.not_assessable = 1
    rules = by_name(plan, scores, resources)
    assert any(r.passed is None for r in rules.values())
    if kind != "unknown_inventory":
        assert rules["minimum_net_reduction"].passed is None


@pytest.mark.parametrize("cost,expected", [("12.5", True), ("12.500001", False), (None, None)])
def test_cost_cap_and_unknown_cost(cost, expected):
    plan, scores, resources = comparison_fixture()
    scores["candidate"].total_cost.total_cost_cny = Decimal(cost) if cost else None
    assert by_name(plan, scores, resources)["cost_increase"].passed is expected


def test_zero_baseline_does_not_divide_or_allow_positive_cost():
    plan, scores, resources = comparison_fixture()
    for score in scores.values():
        score.total_cost.total_cost_cny = Decimal(0)
    assert by_name(plan, scores, resources)["cost_increase"].passed
    scores["candidate"].total_cost.total_cost_cny = Decimal("0.001")
    assert not by_name(plan, scores, resources)["cost_increase"].passed


@pytest.mark.parametrize("delta,expected", [(268435456, True), (268435457, False)])
def test_memory_increment_boundary(delta, expected):
    plan, scores, resources = comparison_fixture()
    resources["candidate"].peak_combined_bytes += delta
    assert by_name(plan, scores, resources)["memory_increase"].passed is expected
    resources["candidate"].complete = False
    assert by_name(plan, scores, resources)["memory_increase"].passed is None


@pytest.mark.parametrize(
    "baseline,candidate,relative,absolute",
    [
        (10, 12.5, True, True),
        (10, 13, False, True),
        (100, 124, True, False),
        (100, 120, True, True),
    ],
)
def test_latency_requires_both_limits(baseline, candidate, relative, absolute):
    plan, scores, resources = comparison_fixture()
    for arm, seconds in (("baseline", baseline), ("candidate", candidate)):
        for case in scores[arm].cases:
            case.observation.stage_times[0].seconds = float(seconds)
    rules = by_name(plan, scores, resources)
    assert rules["latency_relative"].passed is relative
    assert rules["latency_absolute"].passed is absolute


def test_normal_control_cannot_regress_while_other_cases_improve():
    plan, scores, resources = comparison_fixture()
    case_id = plan.samples.normal_control_case_ids[-1]
    next(
        c for c in scores["candidate"].cases if c.observation.case_id == case_id
    ).business_success = False
    assert not by_name(plan, scores, resources)[f"control_{case_id}"].passed


@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_scoring_projection_preserves_actual_source_graph_prompt_and_outputs(arm):
    _, old, _ = inputs()
    raw = old.model_dump(mode="json")
    raw["artifact_contract"] = "e7a-generation-report-v1"
    raw["start"].update(
        artifact_contract="e7a-generation-start-v1",
        arm=arm,
        fixture_version="e7a-generation-database-v1",
        schema_digest="sha256:" + "1" * 64,
        production_schema=False,
        output_contract="e7a_research_output_v1" if arm == "candidate" else "research_output_v2",
        assessment_policy_version="e7a-evidence-assessment-v1" if arm == "candidate" else None,
        assessment_prompt_digest="sha256:" + "2" * 64 if arm == "candidate" else None,
    )
    raw["start"]["manifest"]["graph_version"] = (
        "pathfinder-research-e7a-exp-v1" if arm == "candidate" else "pathfinder-research-v6"
    )
    import json

    report = ExperimentGenerationReportV1.model_validate_json(json.dumps(raw))
    projection = score_projection(report)
    assert projection.start.manifest == report.start.manifest
    assert projection.cases == report.cases
    assert projection.total_usage == report.total_usage
    assert projection.start.manifest.graph_version.endswith(
        "exp-v1" if arm == "candidate" else "v6"
    )


def test_review_export_and_import_reject_rebound_output_and_changed_input(tmp_path, monkeypatch):
    from tests.evals import quality_experiment_decision as decision
    from tests.evals.quality_experiment_binding import ExperimentError, encoded, read_json
    from tests.evals.quality_experiment_execution import ExecutionV1, planned_slots
    from tests.evals.test_quality_experiment_execution import binding_fixture

    tmp_path.chmod(0o700)
    binding = binding_fixture(tmp_path)
    plan = load_experiment_plan(ROOT / PLAN)
    execution = ExecutionV1(
        binding_digest=decision.digest(binding),
        slots=tuple(s.model_copy(update={"status": "not_run"}) for s in planned_slots(plan)),
        stop_category="contract_fixture",
        cleanup_complete=True,
        arm_report_digests={},
        resource_report_digests={},
        elapsed_seconds=1.0,
        execution_complete=False,
    )
    monkeypatch.setattr(decision, "load_execution", lambda *a: (plan, execution, {}, {}, {}, {}))
    review = tmp_path / "review"
    decision.export_review(binding, tmp_path, review)
    receipt, scores = decision.collect_review(binding, tmp_path, review)
    assert not receipt.complete and scores == {}
    view_path = review / "inputs/review_0000.json"
    view = read_json(view_path)
    assert "arm" not in view and view["source_units"]
    # Even updating the export's view digest cannot replace frozen review material.
    view["required_unit_ids"] = []
    view["repeat_index"] = 99
    view_path.write_bytes(encoded(view))
    export_path = review / "export.json"
    export = read_json(export_path)
    export["assignments"][0]["input_digest"] = decision.digest(view)
    export_path.write_bytes(encoded(export))
    with pytest.raises(ExperimentError, match="review_input_mismatch"):
        decision.collect_review(binding, tmp_path, review)
    export["assignments"][0]["output_digest"] = "sha256:" + "9" * 64
    export_path.write_bytes(encoded(export))
    with pytest.raises(ExperimentError, match="review_output_mismatch"):
        decision.collect_review(binding, tmp_path, review)
