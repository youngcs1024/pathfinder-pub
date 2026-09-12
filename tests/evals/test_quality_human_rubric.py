"""E4.7 teaching fixtures and existing annotation boundary; no human quality claim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityObservationV1,
    QualityRubricV1,
    QualityRunManifestV1,
)
from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    quality_digest,
    validate_quality_annotations,
)
from tests.evals.test_quality_contracts import (
    FIXTURE,
    annotation_payload,
    observation_payload,
    run_payload,
)

MATERIALS = Path(__file__).resolve().parents[2] / "tests/fixtures/quality_reviews/e47-rubric-v1"


def parse(model, payload):
    return model.model_validate_json(json.dumps(payload))


def test_six_teaching_examples_bind_exact_bytes_and_preserve_label_distinctions():
    rubric = QualityRubricV1.model_validate_json((MATERIALS / "rubric.json").read_bytes())
    assert rubric.status == "draft"
    assert rubric.review_policy == "single_reviewer"
    package = json.loads((MATERIALS / "examples.json").read_bytes())
    assert package["provenance"] == "agent_authored_teaching_only"
    assert package["human_pilot_count"] == 0
    expected = {
        "wrong_citation": (["supported"], ["unsupported"], "appropriate", "major_edits"),
        "partial_support": (
            ["supported", "unsupported"],
            ["supported", "unsupported"],
            "overclaim",
            "major_edits",
        ),
        "exaggerated_experience": (["contradicted"], ["contradicted"], "overclaim", "unusable"),
        "appropriate_refusal": (["supported"], [], "appropriate", "not_applicable"),
        "unnecessary_refusal": (["contradicted"], [], "unnecessary_refusal", "not_applicable"),
        "normal_citation": (["supported"], ["supported"], "appropriate", "usable"),
    }
    assert len(package["examples"]) == len(expected)
    assert {e["case_id"] for e in package["examples"]} == set(expected)
    for example in package["examples"]:
        annotation = parse(HumanAnnotationV1, example["annotation"])
        assert annotation.case_id == example["case_id"]
        assert annotation.reviewer_id == "agent_teaching_author"
        assert annotation.rubric_version == rubric.rubric_version
        assert Path(example["output_file"]).name == example["output_file"]
        content = (MATERIALS / example["output_file"]).read_bytes()
        assert annotation.output_digest == quality_digest(content)
        assert annotation.output_digest != quality_digest(content + b"\n")
        assert set(example["fact_texts"]) == {f.fact_id for f in annotation.facts}
        assert {c.citation_id for c in annotation.citations} <= set(example["source_evidence"])
        assert (
            [f.judgment for f in annotation.facts],
            [c.judgment for c in annotation.citations],
            annotation.insufficiency,
            annotation.draft_usability,
        ) == expected[annotation.case_id]


def validate(payloads, observation=None):
    validate_quality_annotations(
        load_quality_dataset(FIXTURE),
        parse(QualityRunManifestV1, run_payload()),
        (parse(QualityObservationV1, observation or observation_payload()),),
        tuple(parse(HumanAnnotationV1, payload) for payload in payloads),
    )


def test_matching_annotation_is_accepted():
    validate([annotation_payload()])


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_digest", "sha256:" + "b" * 64),
        ("experiment_id", "other_experiment"),
        ("case_id", "other_case"),
        ("repeat_index", 1),
        ("rubric_version", "quality-human-candidate-v1"),
    ],
)
def test_annotation_identity_mismatch_is_rejected(field, value):
    payload = annotation_payload()
    payload[field] = value
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate([payload])


def test_duplicate_reviewer_slot_is_rejected():
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate([annotation_payload(), annotation_payload()])


def test_out_of_scope_evidence_is_rejected():
    payload = annotation_payload()
    payload["covered_unit_ids"] = ["foreign_workspace_unit"]
    with pytest.raises(QualityDatasetError, match="annotation_scope_mismatch"):
        validate([payload])


def test_no_output_failure_has_no_annotation_but_retains_observation():
    observation = observation_payload()
    observation.update(
        status="failed", failure_type="provider", output_digest=None, assessment_required=False
    )
    validate([], observation)
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate([annotation_payload()], observation)


def test_unresolved_disagreement_cannot_claim_complete():
    payload = annotation_payload()
    payload["disagreements"] = [
        {
            "disagreement_id": "support_disagreement",
            "reviewer_ids": ["reviewer_one", "reviewer_two"],
            "status": "unresolved",
            "adjudicator_id": None,
            "resolution_code": None,
        }
    ]
    with pytest.raises(ValidationError, match="incomplete judgments"):
        parse(HumanAnnotationV1, payload)
    payload["assessment_complete"] = False
    validate([payload])


def test_assessed_not_assessable_is_distinct_from_unassessed():
    payload = annotation_payload()
    payload["facts"][0]["judgment"] = "not_assessable"
    payload["citations"][0]["judgment"] = "not_assessable"
    payload["business_result"] = "not_assessable"
    payload["reason_codes"] = ["evidence_unavailable"]
    validate([payload])
    payload["unassessed_fact_count"] = 1
    with pytest.raises(ValidationError, match="incomplete judgments"):
        parse(HumanAnnotationV1, payload)
