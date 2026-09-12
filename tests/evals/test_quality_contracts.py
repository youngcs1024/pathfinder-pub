from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityCostV1,
    QualityModelPayloadV1,
    QualityObservationV1,
    QualityRatioV1,
    QualityReportV1,
    QualityRunManifestV1,
    QualityStageTimeV1,
)
from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
    validate_quality_annotations,
    validate_quality_report,
    validate_quality_run,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/evals/quality_contract_v1"
DIGEST = "sha256:" + "a" * 64
SHA = "a" * 40


def run_payload():
    dataset = load_quality_dataset(FIXTURE)
    return {
        "experiment_id": "example_run",
        "execution_source_sha": SHA,
        "suite_version": "quality-v1",
        "dataset_version": dataset.manifest.dataset_version,
        "dataset_digest": dataset.manifest_digest,
        "case_set_digest": quality_identity_digest(["claim_strength"]),
        "split_digest": quality_identity_digest(
            [[f.family_id, f.split] for f in dataset.manifest.families]
        ),
        "rubric_version": dataset.rubric.rubric_version,
        "rubric_digest": next(f.digest for f in dataset.manifest.files if f.role == "rubric"),
        "model": "fake",
        "prompt_digest": DIGEST,
        "graph_version": "pathfinder-research-v6",
        "embedding_profile": None,
        "retrieval_policy_digest": None,
        "configuration_digest": DIGEST,
        "measurement_scope": "contract",
        "llm_mode": "fake",
        "web_mode": "frozen_fixture",
        "document_mode": "fixture",
        "selected_case_ids": ["claim_strength"],
        "repeat_count": 1,
        "execution_order": [{"case_id": "claim_strength", "repeat_index": 0}],
        "cost_admission_budget_cny": "1.00",
        "provider_attempt_cap": 10,
        "input_token_cap": 1000,
        "output_token_cap": 1000,
    }


def observation_payload():
    return {
        "experiment_id": "example_run",
        "case_id": "claim_strength",
        "repeat_index": 0,
        "measurement_scope": "contract",
        "status": "succeeded",
        "failure_type": None,
        "safety": {},
        "tool_calls": 0,
        "model_calls": 0,
        "provider_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": {
            "scope": "all",
            "known_cost_cny": "0",
            "priced_attempts": 0,
            "unknown_cost_attempts": 0,
            "total_cost_cny": "0",
        },
        "stage_times": [{"stage": "total", "seconds": 0.1}],
        "output_digest": DIGEST,
        "assessment_required": True,
    }


def annotation_payload():
    return {
        "experiment_id": "example_run",
        "case_id": "claim_strength",
        "repeat_index": 0,
        "output_digest": DIGEST,
        "reviewer_id": "example_reviewer",
        "rubric_version": "quality-contract-example-v1",
        "facts": [{"fact_id": "claim_one", "judgment": "supported"}],
        "citations": [
            {"fact_id": "claim_one", "citation_id": "citation_one", "judgment": "supported"}
        ],
        "covered_unit_ids": ["course_pg", "role_pg"],
        "missing_unit_ids": [],
        "unassessed_fact_count": 0,
        "unassessed_citation_count": 0,
        "business_result": "success",
        "insufficiency": "appropriate",
        "draft_usability": "usable",
        "reason_codes": [],
        "disagreements": [],
        "assessment_complete": True,
    }


def report_payload():
    run = QualityRunManifestV1.model_validate_json(json.dumps(run_payload()))
    annotation = HumanAnnotationV1.model_validate_json(json.dumps(annotation_payload()))
    return {
        "experiment_id": "example_run",
        "execution_source_sha": SHA,
        "scorer_source_sha": SHA,
        "scorer_change_reason": None,
        "manifest_digest": quality_digest(run.model_dump_json().encode()),
        "annotation_digest": quality_identity_digest([annotation.model_dump(mode="json")]),
        "observations": [observation_payload()],
        "coverage": {
            "planned": 1,
            "executed": 1,
            "failed": 0,
            "not_run": 0,
            "generated": 1,
            "assessment_required": 1,
            "assessed": 1,
            "unassessed": 0,
        },
        "groups": [],
        "evidence_valid": True,
        "measurement_complete": True,
        "target_policy_digest": None,
        "meets_quality_target": None,
        "stop_reason": None,
    }


def test_complete_contract_example_roundtrips_and_links():
    dataset = load_quality_dataset(FIXTURE)
    run = QualityRunManifestV1.model_validate_json(json.dumps(run_payload()))
    annotation = HumanAnnotationV1.model_validate_json(json.dumps(annotation_payload()))
    report = QualityReportV1.model_validate_json(json.dumps(report_payload()))
    validate_quality_report(dataset, run, (annotation,), report)
    for model in (
        dataset.manifest,
        dataset.rubric,
        *dataset.cases,
        *dataset.units,
        run,
        annotation,
        report,
        *report.observations,
    ):
        assert type(model).model_validate_json(model.model_dump_json()) == model
    assert dataset.manifest.review_status == "not_reviewed"
    assert dataset.rubric.status == "draft"


@pytest.mark.parametrize("field", ["experiment_id", "case_id", "rubric_version", "output_digest"])
def test_annotation_is_bound_to_exact_output_and_run(field):
    payload = annotation_payload()
    payload[field] = "sha256:" + "b" * 64 if field == "output_digest" else "other"
    annotation = HumanAnnotationV1.model_validate_json(json.dumps(payload))
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate_quality_annotations(
            load_quality_dataset(FIXTURE),
            QualityRunManifestV1.model_validate_json(json.dumps(run_payload())),
            (QualityObservationV1.model_validate_json(json.dumps(observation_payload())),),
            (annotation,),
        )


def test_annotation_duplicate_and_unknown_source_unit_fail():
    dataset = load_quality_dataset(FIXTURE)
    run = QualityRunManifestV1.model_validate_json(json.dumps(run_payload()))
    observations = (QualityObservationV1.model_validate_json(json.dumps(observation_payload())),)
    annotation = HumanAnnotationV1.model_validate_json(json.dumps(annotation_payload()))
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate_quality_annotations(dataset, run, observations, (annotation, annotation))
    payload = annotation_payload()
    payload["covered_unit_ids"] = ["missing_unit"]
    with pytest.raises(QualityDatasetError, match="annotation_scope_mismatch"):
        validate_quality_annotations(
            dataset,
            run,
            observations,
            (HumanAnnotationV1.model_validate_json(json.dumps(payload)),),
        )


@pytest.mark.parametrize(
    "field", ["dataset_digest", "rubric_digest", "case_set_digest", "split_digest"]
)
def test_run_identity_mismatch(field):
    payload = run_payload()
    payload[field] = "sha256:" + "b" * 64
    with pytest.raises(QualityDatasetError, match="run_dataset_identity_mismatch"):
        validate_quality_run(
            load_quality_dataset(FIXTURE),
            QualityRunManifestV1.model_validate_json(json.dumps(payload)),
        )


@pytest.mark.parametrize(
    "order",
    [
        [],
        [{"case_id": "other", "repeat_index": 0}],
        [{"case_id": "claim_strength", "repeat_index": 1}],
    ],
)
def test_fixed_execution_order_cannot_lose_or_replace_slots(order):
    payload = run_payload()
    payload["execution_order"] = order
    with pytest.raises(ValidationError):
        QualityRunManifestV1.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field",
    [
        "cost_admission_budget_cny",
        "provider_attempt_cap",
        "input_token_cap",
        "output_token_cap",
        "repeat_count",
    ],
)
def test_run_requires_positive_explicit_budgets(field):
    payload = run_payload()
    payload[field] = 0
    with pytest.raises(ValidationError):
        QualityRunManifestV1.model_validate_json(json.dumps(payload))


def test_unknown_cost_is_not_zero_but_known_zero_is_allowed():
    payload = observation_payload()["cost"]
    payload["unknown_cost_attempts"] = 1
    with pytest.raises(ValidationError, match="unknown cost"):
        QualityCostV1.model_validate_json(json.dumps(payload))
    payload["total_cost_cny"] = None
    cost = QualityCostV1.model_validate_json(json.dumps(payload))
    assert cost.total_cost_cny is None
    assert cost.known_cost_cny == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.1])
def test_stage_times_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValidationError):
        QualityStageTimeV1(stage="total", seconds=value)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_cost_must_be_finite(value):
    payload = observation_payload()["cost"]
    payload.update(known_cost_cny=value, total_cost_cny=value, priced_attempts=1)
    with pytest.raises(ValidationError):
        QualityCostV1.model_validate_json(json.dumps(payload))


def test_cost_coverage_and_not_run_evidence_cannot_be_fabricated():
    payload = observation_payload()
    payload["provider_attempts"] = 1
    with pytest.raises(ValidationError, match="every provider attempt"):
        QualityObservationV1.model_validate_json(json.dumps(payload))
    payload = observation_payload()
    payload["status"] = "not_run"
    with pytest.raises(ValidationError, match="unexecuted"):
        QualityObservationV1.model_validate_json(json.dumps(payload))


def test_partial_report_retains_unexecuted_slots_and_unassessed_outputs():
    payload = report_payload()
    pending = observation_payload()
    pending.update(
        case_id="no_resume",
        status="not_run",
        failure_type="budget",
        stage_times=[],
        output_digest=None,
        assessment_required=False,
    )
    payload["observations"].append(pending)
    payload["coverage"].update(planned=2, not_run=1, assessed=0, unassessed=1)
    payload.update(annotation_digest=None, measurement_complete=False, stop_reason="budget")
    report = QualityReportV1.model_validate_json(json.dumps(payload))
    assert report.coverage.not_run == 1
    assert report.coverage.unassessed == 1
    payload["meets_quality_target"] = True
    payload["target_policy_digest"] = DIGEST
    with pytest.raises(ValidationError, match="valid complete evidence"):
        QualityReportV1.model_validate_json(json.dumps(payload))


def test_all_failed_can_be_complete_valid_measurement():
    payload = report_payload()
    payload["observations"][0].update(
        status="failed", failure_type="provider", output_digest=None, assessment_required=False
    )
    payload["coverage"].update(failed=1, generated=0, assessment_required=0, assessed=0)
    payload["annotation_digest"] = None
    report = QualityReportV1.model_validate_json(json.dumps(payload))
    assert report.evidence_valid and report.measurement_complete
    assert report.meets_quality_target is None


@pytest.mark.parametrize(
    "field,value",
    [("planned", 2), ("executed", 0), ("failed", 1), ("generated", 0), ("unassessed", 1)],
)
def test_report_rejects_contradictory_counts(field, value):
    payload = report_payload()
    payload["coverage"][field] = value
    with pytest.raises(ValidationError):
        QualityReportV1.model_validate_json(json.dumps(payload))


def test_zero_denominator_requires_null_and_reason():
    payload = dict(
        numerator=0,
        denominator=0,
        value=None,
        unassessed_count=1,
        not_applicable_reason="zero_denominator",
    )
    assert QualityRatioV1.model_validate(payload).value is None
    for value in (0.0, 1.0):
        with pytest.raises(ValidationError):
            QualityRatioV1.model_validate({**payload, "value": value})


def test_safety_cannot_be_hidden_by_quality_target():
    payload = report_payload()
    payload["observations"][0].update(
        status="failed", failure_type="safety", safety={"gold_contamination": 1}
    )
    payload["coverage"]["failed"] = 1
    payload["stop_reason"] = "safety"
    with pytest.raises(ValidationError, match="invalidates"):
        QualityReportV1.model_validate_json(json.dumps(payload))
    payload["evidence_valid"] = False
    assert not QualityReportV1.model_validate_json(json.dumps(payload)).evidence_valid


def test_unresolved_disagreement_blocks_complete_annotation():
    payload = annotation_payload()
    payload["disagreements"] = [
        {
            "disagreement_id": "fact_one",
            "reviewer_ids": ["reviewer_a", "reviewer_b"],
            "status": "unresolved",
            "adjudicator_id": None,
            "resolution_code": None,
        }
    ]
    with pytest.raises(ValidationError, match="incomplete judgments"):
        HumanAnnotationV1.model_validate_json(json.dumps(payload))
    payload["assessment_complete"] = False
    assert not HumanAnnotationV1.model_validate_json(json.dumps(payload)).assessment_complete


def test_report_claimed_assessment_needs_actual_annotation():
    report = QualityReportV1.model_validate_json(json.dumps(report_payload()))
    with pytest.raises(QualityDatasetError, match="report_annotation_mismatch"):
        validate_quality_report(
            load_quality_dataset(FIXTURE),
            QualityRunManifestV1.model_validate_json(json.dumps(run_payload())),
            (),
            report,
        )


@pytest.mark.parametrize("field", ["query", "output", "resume", "required_unit_ids", "token"])
def test_public_report_forbids_body_and_errors_hide_canary(field):
    payload = report_payload()
    canary = "private-body-canary-不要公开"
    payload[field] = canary
    with pytest.raises(ValidationError) as caught:
        QualityReportV1.model_validate_json(json.dumps(payload))
    assert canary not in str(caught.value)
    report = QualityReportV1.model_validate_json(json.dumps(report_payload()))
    assert canary not in report.model_dump_json()


@pytest.mark.parametrize(
    "field",
    [
        "required_unit_ids",
        "expected_behavior",
        "rubric",
        "workspace_id",
        "actor_user_id",
        "resume_alias",
    ],
)
def test_model_payload_rejects_gold_and_trusted_fields(field):
    with pytest.raises(ValidationError):
        QualityModelPayloadV1.model_validate_json(
            json.dumps({"mode": "research", "query": "safe query", field: "canary"})
        )


def test_strict_types_do_not_coerce_counts_or_accept_extra_fields():
    payload = observation_payload()
    payload["model_calls"] = "0"
    with pytest.raises(ValidationError):
        QualityObservationV1.model_validate_json(json.dumps(payload))


def test_group_coverage_must_partition_and_match_case_membership():
    payload = report_payload()
    payload["groups"] = [
        {
            "dimension": "stratum",
            "group_id": "unsupported_experience",
            "coverage": payload["coverage"].copy(),
            "business_success": {
                "numerator": 1,
                "denominator": 1,
                "value": 1.0,
                "unassessed_count": 0,
                "not_applicable_reason": None,
            },
        }
    ]
    dataset = load_quality_dataset(FIXTURE)
    run = QualityRunManifestV1.model_validate_json(json.dumps(run_payload()))
    annotations = (HumanAnnotationV1.model_validate_json(json.dumps(annotation_payload())),)
    report = QualityReportV1.model_validate_json(json.dumps(payload))
    validate_quality_report(dataset, run, annotations, report)
    payload["groups"][0]["group_id"] = "no_resume_scope"
    report = QualityReportV1.model_validate_json(json.dumps(payload))
    with pytest.raises(QualityDatasetError, match="report_group_mismatch"):
        validate_quality_report(dataset, run, annotations, report)
    payload["groups"].append({**payload["groups"][0], "group_id": "unsupported_experience"})
    with pytest.raises(ValidationError, match="partition"):
        QualityReportV1.model_validate_json(json.dumps(payload))


def test_valid_but_low_quality_measurement_can_fail_locked_target():
    payload = report_payload()
    payload.update(target_policy_digest=DIGEST, meets_quality_target=False)
    report = QualityReportV1.model_validate_json(json.dumps(payload))
    assert report.evidence_valid and report.measurement_complete
    assert report.meets_quality_target is False


@pytest.mark.parametrize("failure", ["configuration", "integrity"])
def test_measurement_configuration_failures_are_not_model_quality_failures(failure):
    payload = report_payload()
    payload["observations"][0].update(status="failed", failure_type=failure)
    payload["coverage"]["failed"] = 1
    with pytest.raises(ValidationError, match="invalidates"):
        QualityReportV1.model_validate_json(json.dumps(payload))


def test_another_reviewer_cannot_hide_an_unfinished_review():
    payload = annotation_payload()
    first = HumanAnnotationV1.model_validate_json(json.dumps(payload))
    payload.update(
        reviewer_id="second_reviewer", assessment_complete=False, unassessed_fact_count=1
    )
    second = HumanAnnotationV1.model_validate_json(json.dumps(payload))
    annotations = (first, second)
    report_data = report_payload()
    report_data["annotation_digest"] = quality_identity_digest(
        [a.model_dump(mode="json") for a in annotations]
    )
    report = QualityReportV1.model_validate_json(json.dumps(report_data))
    with pytest.raises(QualityDatasetError, match="report_annotation_mismatch"):
        validate_quality_report(
            load_quality_dataset(FIXTURE),
            QualityRunManifestV1.model_validate_json(json.dumps(run_payload())),
            annotations,
            report,
        )


def test_omitted_observation_or_wrong_layer_cannot_validate_against_manifest():
    dataset = load_quality_dataset(FIXTURE)
    run = QualityRunManifestV1.model_validate_json(json.dumps(run_payload()))
    with pytest.raises(QualityDatasetError, match="observation_scope_mismatch"):
        validate_quality_annotations(dataset, run, (), ())
    payload = observation_payload()
    payload["measurement_scope"] = "generation"
    observation = QualityObservationV1.model_validate_json(json.dumps(payload))
    with pytest.raises(QualityDatasetError, match="observation_scope_mismatch"):
        validate_quality_annotations(dataset, run, (observation,), ())
