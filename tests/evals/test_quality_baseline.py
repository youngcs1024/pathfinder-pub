"""Synthetic E4.10 evidence oracles. No live calls, human labels, or accepted real baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from app.retrieval.chunking import normalize_document_content
from tests.evals import quality_baseline as baseline
from tests.evals.quality_baseline_cli import main
from tests.evals.quality_baseline_contracts import (
    QualityCandidateV1,
    QualityEvidencePathsV1,
    QualityPreparationV1,
    QualityTargetV1,
)
from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityGenerationReportV1,
    QualityRetrievalReportV1,
    QualityRunManifestV1,
)
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
)
from tests.evals.quality_freeze import check_frozen, identity
from tests.evals.quality_generation_support import QualityGenerationError
from tests.evals.quality_score import score_quality, write_quality_score
from tests.evals.test_quality_contracts import SHA, annotation_payload, run_payload
from tests.evals.test_quality_score import inputs, parse

DATA = Path(__file__).resolve().parents[2] / "evals/datasets/quality_expanded_v1"
CASES = ("wheel_abi", "oci_manifest")
OTHER_SHA = "b" * 40


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))
    path.chmod(0o600)
    return path


@pytest.fixture(scope="module")
def frozen():
    return check_frozen(DATA)[0]


@pytest.fixture
def evidence(tmp_path, monkeypatch, frozen):
    # Freeze itself has dedicated real-recomputation tests. Other checks remain real here.
    monkeypatch.setattr(baseline, "check_frozen", lambda root: (frozen, ()))
    dataset = load_quality_dataset(DATA)
    case_index = {c.case_id: c for c in dataset.cases}
    policy = inputs()[1].start.policy.retrieval
    manifests = []
    for layer in ("retrieval", "generation"):
        value = run_payload()
        value.update(
            experiment_id=f"baseline_{layer}",
            dataset_version=dataset.manifest.dataset_version,
            dataset_digest=dataset.manifest_digest,
            rubric_version=dataset.rubric.rubric_version,
            rubric_digest=frozen.rubric_digest,
            split_digest=frozen.split_digest,
            selected_case_ids=list(CASES),
            case_set_digest=quality_identity_digest(list(CASES)),
            execution_order=[{"case_id": c, "repeat_index": 0} for c in CASES],
            model=LOCKED_EMBEDDING_MODEL if layer == "retrieval" else LOCKED_CHAT_MODEL,
            suite_version=f"quality-{layer}-v1",
            measurement_scope=layer,
            llm_mode="qwen",
            web_mode="none" if layer == "retrieval" else "frozen_fixture",
            document_mode="real_embedding_db",
            embedding_profile=policy.embedding_profile,
            retrieval_policy_digest=identity(policy),
        )
        manifests.append(parse(QualityRunManifestV1, value))
    preparation = baseline.prepare(
        DATA,
        preparation_id="initial_quality",
        preparation_source_sha=OTHER_SHA,
        selected_case_ids=CASES,
        repeat_count=1,
        manifests=tuple(manifests),
    )
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    payload = inputs(("succeeded", "succeeded"))[1].model_dump(mode="json")
    payload["start"].update(
        manifest=manifests[1].model_dump(mode="json"), mapping_digest=frozen.mapping_digest
    )
    inventory = []

    def private_file(name, value):
        path = write(private / name, value)
        digest = quality_digest(path.read_bytes())
        inventory.append({"name": name, "digest": digest, "byte_count": path.stat().st_size})
        return digest

    private_file("manifest.json", payload["start"])
    sources = []
    for s in dataset.manifest.sources:
        text = normalize_document_content((DATA / s.path).read_text())
        sources.append(
            {
                "source_alias": s.alias,
                "kind": s.kind,
                "text": text,
                "digest": quality_digest(text.encode()),
            }
        )
    private_file("sources.json", {"sources": sources})
    reviews = []
    for i, (item, case_id) in enumerate(zip(payload["cases"], CASES, strict=True)):
        case = case_index[case_id]
        digest = private_file(
            f"output-{i:04d}.json",
            {
                "case_id": case_id,
                "repeat_index": 0,
                "output": {"summary": "synthetic fixture output, not a model measurement"},
            },
        )
        private_digest = private_file(
            f"case-{i:04d}.json",
            {
                "case_id": case_id,
                "repeat_index": 0,
                "input": {"mode": case.mode, "query": case.query},
                "resume_alias": case.resume_alias,
                "web_scenario_alias": case.web_scenario_alias,
                "tools": [],
                "output_digest": digest,
                "failure_type": None,
            },
        )
        o = item["observation"]
        o.update(
            experiment_id=manifests[1].experiment_id,
            case_id=case_id,
            repeat_index=0,
            measurement_scope="generation",
            output_digest=digest,
            provider_attempts=1,
            model_calls=1,
        )
        o["cost"]["priced_attempts"] = 1
        item["private_case_digest"] = private_digest
        review = annotation_payload()
        review.update(
            experiment_id=manifests[1].experiment_id,
            case_id=case_id,
            rubric_version=dataset.rubric.rubric_version,
            output_digest=digest,
            covered_unit_ids=list(case.required_unit_ids),
            missing_unit_ids=[],
            business_result="failure",
        )
        reviews.append(parse(HumanAnnotationV1, review))
    payload["private_files"] = inventory
    payload["total_usage"]["provider_attempts"] = 2
    payload["total_usage"]["cost"]["priced_attempts"] = 2
    generation = parse(QualityGenerationReportV1, payload)
    retrieval = {
        k: payload[k]
        for k in (
            "representations",
            "representation_complete",
            "ingestion_usage",
            "total_usage",
            "ingestion_seconds",
            "executed",
            "not_run",
            "evidence_valid",
            "measurement_complete",
            "stop_reason",
        )
    }
    rows = []
    for item in payload["cases"]:
        o = dict(item["observation"])
        o.update(
            experiment_id=manifests[0].experiment_id,
            measurement_scope="retrieval",
            status="failed",
            failure_type="provider",
            output_digest=None,
            assessment_required=False,
            model_calls=0,
        )
        rows.append({"observation": o, "refs": [], "metrics": None})
    retrieval.update(
        manifest=manifests[0].model_dump(mode="json"),
        mapping_digest=frozen.mapping_digest,
        policy=policy.model_dump(mode="json"),
        semantic_quality_claim=False,
        cases=rows,
        failed=2,
        scope_leakage_count=0,
        full_coverage_cases={
            "numerator": 0,
            "denominator": 0,
            "value": None,
            "unassessed_count": 2,
            "not_applicable_reason": "zero_denominator",
        },
    )
    retrieval = parse(QualityRetrievalReportV1, retrieval)
    paths = QualityEvidencePathsV1(
        retrieval_report=str(tmp_path / "retrieval.json"),
        generation_report=str(tmp_path / "generation.json"),
        generation_private_dir=str(private),
        annotations=str(private / "labels.jsonl"),
        retrieval_score=str(tmp_path / "retrieval-score.json"),
        generation_score=str(tmp_path / "generation-score.json"),
    )
    write(Path(paths.retrieval_report), retrieval.model_dump(mode="json"))
    write(Path(paths.generation_report), generation.model_dump(mode="json"))
    Path(paths.annotations).write_text("\n".join(a.model_dump_json() for a in reviews))
    Path(paths.annotations).chmod(0o600)
    scores = (
        baseline.score_inputs(DATA, preparation, retrieval, scorer_source_sha=SHA),
        baseline.score_inputs(
            DATA,
            preparation,
            generation,
            tuple(reviews),
            private_dir=private,
            scorer_source_sha=SHA,
        ),
    )
    for path, score in zip((paths.retrieval_score, paths.generation_score), scores, strict=True):
        write_quality_score(Path(path), score)
    return preparation, paths, scores


def accept_fixture(value, paths, **kwargs):
    return baseline.accept(
        DATA,
        value,
        paths,
        baseline.baseline_diff(value),
        baseline_id="initial_quality",
        accepting_source_sha=SHA,
        confirm=True,
        human_reviewed=True,
        safety_reviewed=True,
        prepared_before_run=True,
        **kwargs,
    )


def test_real_frozen_preparation_and_acceptance_recompute_original_inputs(evidence, monkeypatch):
    preparation, paths, _ = evidence
    monkeypatch.setattr(baseline, "check_frozen", check_frozen)
    baseline.verify_preparation(DATA, preparation)
    result = baseline.candidate(DATA, preparation, paths)
    assert result.acceptance_ready and result.measurement_complete
    assert result.meets_quality_target is None
    assert result.scores[0].summary.coverage.failed == 2
    assert result.scores[1].summary.business_success.value == 0  # Low quality is valid evidence.
    assert preparation.preparation_source_sha != result.scores[0].execution_source_sha
    assert not preparation.live_authorized and not preparation.measurement_complete
    accepted = accept_fixture(result, paths)
    assert accepted.stage == "BASELINE_COMPLETE"
    assert accepted.confirmation_kind == "operator_attestation"
    baseline.validate_accepted(accepted)


@pytest.mark.parametrize("change", ["selection", "budget", "freeze", "validation", "order"])
def test_preparation_rejects_mismatched_or_invalid_scope(evidence, change):
    prep, _, _ = evidence
    payload = prep.model_dump(mode="json")
    if change == "selection":
        payload["manifests"][1]["selected_case_ids"] = [CASES[0]]
    elif change == "budget":
        payload["manifests"][0]["provider_attempt_cap"] = 0
    elif change == "freeze":
        payload["freeze_digest"] = "sha256:" + "0" * 64
    elif change == "validation":
        payload["validation_case_ids"] = [CASES[0]]
    else:
        payload["manifests"][1]["execution_order"].reverse()
    with pytest.raises((ValueError, ValidationError)):
        baseline.verify_preparation(DATA, parse(QualityPreparationV1, payload))


@pytest.mark.parametrize(
    "flag", ["confirm", "human_reviewed", "safety_reviewed", "prepared_before_run"]
)
def test_accept_requires_each_explicit_operator_confirmation(evidence, flag):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    flags = dict(confirm=True, human_reviewed=True, safety_reviewed=True, prepared_before_run=True)
    flags[flag] = False
    with pytest.raises(baseline.QualityBaselineError, match="explicit_confirmation_required"):
        baseline.accept(
            DATA,
            value,
            paths,
            baseline.baseline_diff(value),
            baseline_id="test",
            accepting_source_sha=SHA,
            **flags,
        )


@pytest.mark.parametrize("mutation", ["body", "label", "score", "report", "missing_output"])
def test_accept_reopens_all_evidence_instead_of_trusting_candidate(evidence, mutation):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    if mutation == "body":
        Path(paths.generation_private_dir, "output-0000.json").write_text("changed private body")
    elif mutation == "label":
        path = Path(paths.annotations)
        path.write_text(path.read_text().replace('"failure"', '"success"'))
    elif mutation == "score":
        path = Path(paths.generation_score)
        raw = json.loads(path.read_text())
        raw["summary"]["fact_support"]["value"] = 0.2
        write(path, raw)
    elif mutation == "missing_output":
        Path(paths.generation_private_dir, "output-0000.json").rename(
            Path(paths.generation_private_dir, "retained-output.json")
        )
    else:
        path = Path(paths.retrieval_report)
        raw = json.loads(path.read_text())
        raw["manifest"]["execution_source_sha"] = OTHER_SHA
        write(path, raw)
    with pytest.raises((ValueError, OSError)):
        accept_fixture(value, paths)


def test_cross_run_annotation_and_late_scorer_identity(evidence):
    prep, paths, _ = evidence
    report = baseline.load(Path(paths.generation_report), QualityGenerationReportV1)
    labels = baseline.annotations(Path(paths.annotations))
    with pytest.raises(ValueError):
        baseline.score_inputs(
            DATA,
            prep,
            report,
            (labels[0].model_copy(update={"experiment_id": "foreign"}),),
            private_dir=Path(paths.generation_private_dir),
            scorer_source_sha=SHA,
        )
    with pytest.raises(ValueError):
        baseline.score_inputs(
            DATA,
            prep,
            report,
            labels,
            private_dir=Path(paths.generation_private_dir),
            scorer_source_sha=OTHER_SHA,
        )
    result = baseline.score_inputs(
        DATA,
        prep,
        report,
        labels,
        private_dir=Path(paths.generation_private_dir),
        scorer_source_sha=OTHER_SHA,
        scorer_change_reason="scorer_fix",
    )
    assert result.execution_source_sha == SHA and result.scorer_source_sha == OTHER_SHA
    assert result.annotation_digest is not None


def test_missing_annotations_blocks_candidate_without_fabricating_scores(evidence):
    prep, paths, scores = evidence
    generation = baseline.load(Path(paths.generation_report), QualityGenerationReportV1)
    score = score_quality(load_quality_dataset(DATA), generation, (), scorer_source_sha=SHA)
    value = baseline.derive_candidate(prep, (scores[0], score), "sha256:" + "1" * 64)
    assert "missing_assessment" in value.blockers
    assert value.scores[1].summary.coverage.unassessed == 2
    assert not value.acceptance_ready


def test_exposure_after_candidate_blocks_acceptance(evidence, monkeypatch, frozen):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    monkeypatch.setattr(baseline, "check_frozen", lambda root: (frozen, (object(),)))
    with pytest.raises(baseline.QualityBaselineError, match="candidate_inputs_changed"):
        accept_fixture(value, paths)
    blocked = baseline.candidate(DATA, prep, paths)
    assert "validation_exposed" in blocked.blockers and not blocked.evidence_valid
    with pytest.raises(baseline.QualityBaselineError, match="validation_exposed"):
        baseline.prepare(DATA, selected_case_ids=CASES)


def test_fake_claim_and_zero_attempts_cannot_become_live_baseline(evidence):
    prep, _, scores = evidence
    fake_manifests = tuple(m.model_copy(update={"llm_mode": "fake"}) for m in prep.manifests)
    fake_prep = prep.model_copy(update={"manifests": fake_manifests})
    fake_scores = tuple(
        s.model_copy(
            update={
                "manifest": m,
                "manifest_digest": quality_digest(m.model_dump_json().encode()),
                "semantic_quality_claim": False,
            }
        )
        for s, m in zip(scores, fake_manifests, strict=True)
    )
    value = baseline.derive_candidate(fake_prep, fake_scores, "sha256:" + "1" * 64)
    assert "not_live" in value.blockers and not value.acceptance_ready


def test_locked_target_failure_does_not_hide_valid_low_quality(evidence):
    prep, paths, _ = evidence
    prep = prep.model_copy(
        update={
            "targets": (
                QualityTargetV1(layer="generation", metric="business_success", minimum=0.9),
            )
        }
    )
    result = baseline.candidate(DATA, prep, paths)
    assert result.acceptance_ready and result.meets_quality_target is False
    assert accept_fixture(result, paths).baseline_accepted


def test_first_and_later_diff_bind_scope_versions_and_results(evidence):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    first = baseline.baseline_diff(value)
    assert first.first_acceptance and first.previous is None
    assert first.previous_baseline_digest is None
    previous = accept_fixture(value, paths)
    later = baseline.baseline_diff(value, previous)
    assert not later.first_acceptance
    assert not later.scope_changed and not later.versions_changed and not later.results_changed
    assert later.previous_baseline_digest == identity(previous)
    with pytest.raises(baseline.QualityBaselineError, match="reviewed_diff_mismatch"):
        baseline.accept(
            DATA,
            value,
            paths,
            first,
            previous=previous,
            baseline_id="next",
            accepting_source_sha=SHA,
            confirm=True,
            human_reviewed=True,
            safety_reviewed=True,
            prepared_before_run=True,
        )


def test_publication_create_only_and_leak_detection(evidence, tmp_path, monkeypatch):
    prep, paths, _ = evidence
    result = baseline.candidate(DATA, prep, paths)
    path = tmp_path / "candidate.json"
    assert baseline.publish(path, result) == quality_digest(path.read_bytes())
    assert path.stat().st_mode & 0o777 == 0o600
    assert "synthetic fixture output" not in path.read_text()
    assert "generation_private_dir" not in path.read_text()
    with pytest.raises(FileExistsError):
        baseline.publish(path, result)
    monkeypatch.setenv("PF_TEST_SECRET", "baseline_generation")
    with pytest.raises(baseline.QualityBaselineError, match="unsafe_public_artifact"):
        baseline.publish(tmp_path / "unsafe.json", result)
    assert not (tmp_path / "unsafe.json").exists()
    with pytest.raises(baseline.QualityBaselineError, match="invalid_public_artifact"):
        baseline.publish(tmp_path / "private.json", paths)


def test_cli_is_offline_and_reports_only_safe_categories(evidence, tmp_path, capsys, monkeypatch):
    prep, paths, _ = evidence
    import socket

    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("network"))
    monkeypatch.setattr("tests.evals.quality_baseline_cli.probe_clean_git_head", lambda: SHA)
    prep_path = write(tmp_path / "preparation.json", prep.model_dump(mode="json"))
    private = Path(paths.generation_private_dir)
    inventory = write(private / "inputs.json", paths.model_dump())
    output = tmp_path / "candidate.json"
    assert (
        main(
            [
                "candidate",
                "--dataset",
                str(DATA),
                "--preparation",
                str(prep_path),
                "--inputs",
                str(inventory),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert baseline.load(output, QualityCandidateV1).acceptance_ready
    diff = tmp_path / "diff.json"
    assert main(["diff", "--candidate", str(output), "--first", "--output", str(diff)]) == 0
    assert main(["candidate", "--private-body-secret"]) == 1
    bad = write(tmp_path / "bad.json", {"private_body_secret": "do not print"})
    assert main(["diff", "--candidate", str(bad), "--first", "--output", str(diff)]) == 1
    printed = capsys.readouterr()
    assert "private_body_secret" not in printed.out and "private-body-secret" not in printed.out
    assert str(tmp_path) not in printed.out and not printed.err


def test_not_run_slot_remains_in_denominator_and_blocks_completion(evidence):
    prep, paths, scores = evidence
    payload = baseline.load(Path(paths.generation_report), QualityGenerationReportV1).model_dump(
        mode="json"
    )
    last = payload["cases"][1]
    last["private_case_digest"] = None
    o = last["observation"]
    o.update(
        status="not_run",
        failure_type=None,
        output_digest=None,
        assessment_required=False,
        provider_attempts=0,
        model_calls=0,
        stage_times=[],
    )
    o["cost"]["priced_attempts"] = 0
    payload["private_files"] = [f for f in payload["private_files"] if "0001" not in f["name"]]
    payload.update(executed=1, not_run=1, measurement_complete=False)
    payload["total_usage"]["provider_attempts"] = 1
    payload["total_usage"]["cost"]["priced_attempts"] = 1
    report = parse(QualityGenerationReportV1, payload)
    labels = baseline.annotations(Path(paths.annotations))[:1]
    score = score_quality(load_quality_dataset(DATA), report, labels, scorer_source_sha=SHA)
    value = baseline.derive_candidate(prep, (scores[0], score), "sha256:" + "1" * 64)
    assert "incomplete_measurement" in value.blockers
    assert score.summary.coverage.planned == 2 and score.summary.coverage.not_run == 1
    assert score.summary.business_success.denominator == 1
    assert not value.acceptance_ready


def test_safety_violation_cannot_be_averaged_away(evidence):
    prep, paths, scores = evidence
    payload = baseline.load(Path(paths.retrieval_report), QualityRetrievalReportV1).model_dump(
        mode="json"
    )
    payload["cases"][0]["observation"].update(
        failure_type="safety", safety={"gold_contamination": 1}
    )
    payload.update(evidence_valid=False, stop_reason="safety", measurement_complete=False)
    score = score_quality(
        load_quality_dataset(DATA), parse(QualityRetrievalReportV1, payload), scorer_source_sha=SHA
    )
    value = baseline.derive_candidate(prep, (score, scores[1]), "sha256:" + "1" * 64)
    assert "safety_violation" in value.blockers and not value.acceptance_ready


def test_unknown_cost_stays_unknown_without_invalidating_complete_measurement(evidence):
    prep, paths, scores = evidence
    payload = baseline.load(Path(paths.retrieval_report), QualityRetrievalReportV1).model_dump(
        mode="json"
    )
    payload["cases"][0]["observation"]["cost"].update(
        priced_attempts=0, unknown_cost_attempts=1, total_cost_cny=None
    )
    payload["total_usage"]["cost"].update(
        priced_attempts=1, unknown_cost_attempts=1, total_cost_cny=None
    )
    score = score_quality(
        load_quality_dataset(DATA), parse(QualityRetrievalReportV1, payload), scorer_source_sha=SHA
    )
    value = baseline.derive_candidate(prep, (score, scores[1]), "sha256:" + "1" * 64)
    assert value.acceptance_ready and score.total_cost.total_cost_cny is None
    assert score.total_cost.unknown_cost_attempts == 1


def test_qwen_label_without_attempt_evidence_is_blocked(evidence):
    prep, paths, scores = evidence
    payload = baseline.load(Path(paths.retrieval_report), QualityRetrievalReportV1).model_dump(
        mode="json"
    )
    for row in payload["cases"]:
        row["observation"]["provider_attempts"] = 0
        row["observation"]["cost"]["priced_attempts"] = 0
    payload["total_usage"]["provider_attempts"] = 0
    payload["total_usage"]["cost"]["priced_attempts"] = 0
    score = score_quality(
        load_quality_dataset(DATA), parse(QualityRetrievalReportV1, payload), scorer_source_sha=SHA
    )
    value = baseline.derive_candidate(prep, (score, scores[1]), "sha256:" + "1" * 64)
    assert "missing_live_attempts" in value.blockers


def test_candidate_flag_tampering_is_rejected(evidence):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    with pytest.raises(baseline.QualityBaselineError, match="candidate_recomputation_mismatch"):
        baseline.validate_candidate(value.model_copy(update={"measurement_complete": False}))


def test_private_permissions_and_symlinks_fail_closed(evidence, tmp_path):
    prep, paths, _ = evidence
    Path(paths.annotations).chmod(0o644)
    with pytest.raises(ValueError):
        baseline.candidate(DATA, prep, paths)
    Path(paths.annotations).chmod(0o600)
    link = tmp_path / "linked"
    link.symlink_to(Path(paths.generation_private_dir), target_is_directory=True)
    with pytest.raises(QualityGenerationError):
        baseline.candidate(
            DATA, prep, paths.model_copy(update={"generation_private_dir": str(link)})
        )


def test_diff_publication_refuses_forged_change_flags(evidence, tmp_path):
    prep, paths, _ = evidence
    value = baseline.candidate(DATA, prep, paths)
    diff = baseline.baseline_diff(value)
    with pytest.raises(baseline.QualityBaselineError, match="diff_recomputation_mismatch"):
        baseline.publish(
            tmp_path / "forged-diff.json", diff.model_copy(update={"scope_changed": False})
        )


def test_cli_prepare_score_and_accept_are_explicit_offline_operations(
    evidence, tmp_path, monkeypatch
):
    prep, paths, _ = evidence
    monkeypatch.setattr("tests.evals.quality_baseline_cli.probe_clean_git_head", lambda: SHA)
    rmanifest = write(tmp_path / "rmanifest.json", prep.manifests[0].model_dump(mode="json"))
    gmanifest = write(tmp_path / "gmanifest.json", prep.manifests[1].model_dump(mode="json"))
    ppath = tmp_path / "prepared.json"
    assert (
        main(
            [
                "prepare",
                "--dataset",
                str(DATA),
                "--preparation-id",
                "cli_initial",
                "--retrieval-manifest",
                str(rmanifest),
                "--generation-manifest",
                str(gmanifest),
                "--case",
                CASES[0],
                "--case",
                CASES[1],
                "--repeat",
                "1",
                "--output",
                str(ppath),
            ]
        )
        == 0
    )
    actual = baseline.load(ppath, QualityPreparationV1)
    assert not actual.live_authorized
    for layer, report in (
        ("retrieval", paths.retrieval_report),
        ("generation", paths.generation_report),
    ):
        args = [
            "score",
            "--dataset",
            str(DATA),
            "--preparation",
            str(ppath),
            "--layer",
            layer,
            "--report",
            report,
            "--output",
            str(tmp_path / f"cli-{layer}-score.json"),
        ]
        if layer == "generation":
            args += [
                "--annotations",
                paths.annotations,
                "--private-dir",
                paths.generation_private_dir,
            ]
        assert main(args) == 0
    value = baseline.candidate(DATA, actual, paths)
    cpath = tmp_path / "cli-candidate.json"
    dpath = tmp_path / "cli-diff.json"
    baseline.publish(cpath, value)
    baseline.publish(dpath, baseline.baseline_diff(value))
    inventory = write(Path(paths.generation_private_dir) / "cli-inputs.json", paths.model_dump())
    accepted = tmp_path / "synthetic-accepted.json"
    args = [
        "accept",
        "--dataset",
        str(DATA),
        "--candidate",
        str(cpath),
        "--inputs",
        str(inventory),
        "--reviewed-diff",
        str(dpath),
        "--baseline-id",
        "synthetic_cli",
        "--first",
        "--output",
        str(accepted),
    ]
    assert main(args) == 1 and not accepted.exists()
    assert (
        main(
            [
                *args,
                "--confirm-accept",
                "--confirm-human-reviewed",
                "--confirm-safety-reviewed",
                "--confirm-prepared-before-run",
            ]
        )
        == 0
    )
    assert json.loads(accepted.read_text())["baseline_accepted"] is True
