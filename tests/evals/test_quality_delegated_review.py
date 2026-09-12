"""Delegated provenance and freeze failures; synthetic fixtures are not pilot evidence."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evals.quality_delegated_review import (
    DelegatedCalibrationV2,
    DelegationV2,
    export_delegated,
    freeze_delegated,
    import_delegated,
    main,
    public_coverage,
)
from tests.evals.quality_pilot_contracts import CalibrationV1
from tests.evals.quality_review import (
    ReviewError,
    encoded,
    freeze_review,
    import_calibration,
    quality_digest,
    write_new,
)
from tests.evals.test_quality_review import CANDIDATE, FIXTURE, bundle, fill


def prepared(tmp_path):
    root, run, _ = bundle(tmp_path)
    auth = DelegationV2(
        source_base_sha="a" * 40,
        implementation_digest="sha256:" + "b" * 64,
        source_report_digest=quality_digest((run / "report.json").read_bytes()),
    )
    rubric = json.loads(CANDIDATE.read_text())
    packages = []
    for number in (1, 2):
        rubric["rubric_version"] = f"quality-agent-delegated-candidate-v{number}"
        path = root / f"candidate-{number}.json"
        path.write_text(json.dumps(rubric))
        dest = root / f"review-{number}"
        package = export_delegated(run, FIXTURE, path, dest, auth)
        fill(dest, package)
        packages.append(dest)
    first, second = packages
    initial = import_delegated(first, FIXTURE, round="initial")
    initial_path = root / "initial.json"
    write_new(initial_path, encoded(initial))
    final = import_delegated(
        second, FIXTURE, round="recheck", predecessor_digest=quality_digest(encoded(initial))
    )
    final_path = root / "final.json"
    write_new(final_path, encoded(final))
    return root, first, second, initial_path, final_path, final


def test_delegated_freeze_is_not_human_or_baseline(tmp_path):
    root, _, dest, initial, final, calibration = prepared(tmp_path)
    record = freeze_delegated(
        dest, FIXTURE, initial, final, root / "freeze", confirm_agent_stable=True
    )
    assert record.reviewed_outputs == 10
    assert not record.delegation.human_reviewed and not record.baseline_accepted
    assert json.loads((root / "freeze/rubric.json").read_text())["rubric_version"] == (
        "quality-agent-delegated-frozen-v1"
    )
    with pytest.raises(ValidationError):
        CalibrationV1.model_validate_json(encoded(calibration))
    with pytest.raises(ReviewError, match="delegated_review_requires_v2"):
        import_calibration(dest, FIXTURE, "human_one")
    with pytest.raises(ReviewError):
        freeze_review(dest, FIXTURE, final, root / "human", confirm_human_stable=True)
    with pytest.raises(ReviewError):
        freeze_delegated(dest, FIXTURE, initial, final, root / "freeze", confirm_agent_stable=True)


@pytest.mark.parametrize(
    "field,value",
    [("reviewer_kind", "human"), ("human_reviewed", True), ("authorization_source", "implicit")],
)
def test_authorization_is_explicit_and_typed(tmp_path, field, value):
    _, _, _, _, _, calibration = prepared(tmp_path)
    payload = calibration.delegation.model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        DelegationV2.model_validate_json(json.dumps(payload))


def test_authorization_digest_and_predecessor_are_bound(tmp_path):
    root, _, dest, initial, final, calibration = prepared(tmp_path)
    payload = calibration.model_dump(mode="json")
    payload["authorization_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        DelegatedCalibrationV2.model_validate_json(json.dumps(payload))
    with pytest.raises(ReviewError, match="agent_freeze_confirmation_required"):
        freeze_delegated(dest, FIXTURE, initial, final, root / "freeze", confirm_agent_stable=False)
    forged = root / "forged.json"
    payload = calibration.model_dump(mode="json")
    payload["predecessor_digest"] = "sha256:" + "0" * 64
    write_new(forged, json.dumps(payload).encode())
    with pytest.raises(ReviewError, match="incomplete_or_changed"):
        freeze_delegated(dest, FIXTURE, initial, forged, root / "freeze", confirm_agent_stable=True)


@pytest.mark.parametrize("replacement", ["未评", "存在规则疑问"])
def test_pending_or_changed_forms_cannot_freeze(tmp_path, replacement):
    root, _, dest, initial, final, _ = prepared(tmp_path)
    form = dest / "review-0000.md"
    form.write_text(form.read_text().replace("| 规则疑问 | 无 |", f"| 规则疑问 | {replacement} |"))
    with pytest.raises(ReviewError, match="incomplete_or_changed"):
        freeze_delegated(dest, FIXTURE, initial, final, root / "freeze", confirm_agent_stable=True)
    current = import_delegated(dest, FIXTURE, round="initial")
    assert not current.assessment.complete


def test_safety_discovery_blocks_completion(tmp_path):
    _, _, dest, _, _, _ = prepared(tmp_path)
    form = dest / "review-0000.md"
    form.write_text(
        form.read_text().replace("| 发现新增安全问题 | 否 |", "| 发现新增安全问题 | 是 |")
    )
    assert not import_delegated(dest, FIXTURE, round="initial").assessment.complete


def test_changed_rubric_requires_new_package_and_recheck(tmp_path):
    root, _, dest, initial, final, _ = prepared(tmp_path)
    rubric = dest / "rubric.json"
    rubric.write_bytes(rubric.read_bytes() + b"\n")
    with pytest.raises(ReviewError, match="calibration_identity_mismatch"):
        freeze_delegated(dest, FIXTURE, initial, final, root / "freeze", confirm_agent_stable=True)


def test_public_coverage_does_not_copy_private_text(tmp_path):
    _, _, dest, _, _, calibration = prepared(tmp_path)
    payload = calibration.model_dump(mode="json")
    payload["assessment"]["cases"][0]["fact_texts"][0]["text"] = "PRIVATE_BODY_CANARY"
    payload["assessment"]["cases"][0]["rule_questions"] = "PRIVATE_QUESTION_CANARY"
    private = DelegatedCalibrationV2.model_validate_json(json.dumps(payload))
    public = json.dumps(public_coverage(dest, FIXTURE, private))
    assert "PRIVATE_" not in public and "测试事实" not in public
    assert '"human_reviewed": false' in public


def test_cli_errors_do_not_leak_paths_or_exception_bodies(capsys):
    assert main(["import", "--dataset", "/PRIVATE_BODY_CANARY", "--review-dir", "/bad"]) == 1
    text = capsys.readouterr().out
    assert "PRIVATE_BODY_CANARY" not in text and "Traceback" not in text
    assert "delegated_operation_rejected" in text
    assert main(["import", "--unrecognized", "PRIVATE_ARGUMENT_CANARY"]) == 1
    captured = capsys.readouterr()
    assert "PRIVATE_ARGUMENT_CANARY" not in captured.out + captured.err


def test_six_fixed_examples_have_independent_expected_labels():
    root = (
        Path(__file__).resolve().parents[2]
        / "tests/fixtures/quality_reviews/e47-agent-delegated-v1"
    )
    examples = json.loads((root / "examples.json").read_text())
    expected = {
        "wrong_citation": ("supported", "unsupported", "appropriate"),
        "partial_support": ("unsupported", "unsupported", "overclaim"),
        "inflated_experience": ("contradicted", "contradicted", "overclaim"),
        "correct_refusal": ("supported", "supported", "appropriate"),
        "unnecessary_refusal": ("contradicted", "contradicted", "unnecessary_refusal"),
        "normal_citation": ("supported", "supported", "appropriate"),
    }
    actual = {x["id"]: (x["fact"], x["citation"], x["insufficiency"]) for x in examples}
    assert actual == expected and len(examples) == 6
    # A constant-positive replacement or omitted negative cannot satisfy the oracle.
    assert {k: ("supported", "supported", "appropriate") for k in actual} != expected
    assert {k: v for k, v in actual.items() if k != "wrong_citation"} != expected


def test_delegated_labels_still_rejected_by_old_run_rubric(tmp_path):
    from tests.evals.quality_contracts import QualityGenerationReportV1
    from tests.evals.quality_dataset import (
        QualityDatasetError,
        load_quality_dataset,
        validate_quality_annotations,
    )

    _, _, dest, _, _, calibration = prepared(tmp_path)
    report = QualityGenerationReportV1.model_validate_json(
        (dest / "source-report.json").read_bytes()
    )
    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate_quality_annotations(
            load_quality_dataset(FIXTURE),
            report.start.manifest,
            tuple(c.observation for c in report.cases),
            tuple(c.annotation for c in calibration.assessment.cases),
        )


def test_wrong_source_authorization_fails_before_export(tmp_path):
    root, run, _ = bundle(tmp_path)
    auth = DelegationV2(
        source_base_sha="a" * 40,
        implementation_digest="sha256:" + "b" * 64,
        source_report_digest="sha256:" + "0" * 64,
    )
    with pytest.raises(ReviewError, match="delegation_source_mismatch"):
        export_delegated(run, FIXTURE, CANDIDATE, root / "review", auth)
    assert not (root / "review").exists()


def test_original_candidate_cannot_enter_delegated_flow(tmp_path):
    root, run, _ = bundle(tmp_path)
    auth = DelegationV2(
        source_base_sha="a" * 40,
        implementation_digest="sha256:" + "b" * 64,
        source_report_digest=quality_digest((run / "report.json").read_bytes()),
    )
    with pytest.raises(ReviewError, match="delegated_candidate_required"):
        export_delegated(run, FIXTURE, CANDIDATE, root / "review", auth)
