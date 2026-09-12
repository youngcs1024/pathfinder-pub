"""Offline E4.10 preparation and evidence verification. No execution entrypoint."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from app.retrieval.chunking import normalize_document_content
from tests.evals.quality_baseline_contracts import (
    QualityAcceptedBaselineV1,
    QualityBaselineDiffV1,
    QualityCandidateV1,
    QualityEvidencePathsV1,
    QualityPreparationV1,
)
from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityGenerationReportV1,
    QualityGenerationStartV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityRetrievalReportV1,
)
from tests.evals.quality_dataset import (
    _read_file,
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
    validate_quality_run,
)
from tests.evals.quality_freeze import check_frozen, identity
from tests.evals.quality_generation_support import checked_directory, secret_markers
from tests.evals.quality_review import read_private
from tests.evals.quality_score import score_quality, validate_scored_report
from tests.evals.quality_score_contracts import QualityScoredReportV1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_TYPES = (
    QualityPreparationV1,
    QualityCandidateV1,
    QualityBaselineDiffV1,
    QualityAcceptedBaselineV1,
)


class QualityBaselineError(ValueError):
    """Only fixed categories leave the CLI boundary; no paths or rejected input."""


def load(path: Path, model):
    checked_directory(path.parent, private=False)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise QualityBaselineError("invalid_input_file")
        data = stream.read(20_000_001)
    if len(data) > 20_000_000:
        raise QualityBaselineError("input_too_large")
    return model.model_validate_json(data)


def private_path(path: Path) -> Path:
    parent = checked_directory(path.parent)
    if parent.is_relative_to(PROJECT_ROOT) or PROJECT_ROOT.is_relative_to(parent):
        raise QualityBaselineError("private_path_required")
    return parent / path.name


def annotations(path: Path) -> tuple[HumanAnnotationV1, ...]:
    data = read_private(private_path(path), limit=1_000_000)
    return tuple(
        HumanAnnotationV1.model_validate_json(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    )


def verify_preparation(dataset_root: Path, preparation: QualityPreparationV1):
    preparation = QualityPreparationV1.model_validate_json(preparation.model_dump_json())
    frozen, exposures = check_frozen(dataset_root)
    dataset = load_quality_dataset(dataset_root)
    if preparation.freeze_digest != identity(frozen) or preparation.mapping_digest != (
        frozen.mapping_digest
    ):
        raise QualityBaselineError("preparation_freeze_mismatch")
    validation = tuple(c for c in preparation.selected_case_ids if c in frozen.validation_case_ids)
    if preparation.validation_case_ids != validation:
        raise QualityBaselineError("validation_selection_mismatch")
    for manifest in preparation.manifests:
        validate_quality_run(dataset, manifest)
    return dataset, exposures


def prepare(dataset_root: Path, **kwargs) -> QualityPreparationV1:
    frozen, exposures = check_frozen(dataset_root)
    if exposures:
        raise QualityBaselineError("validation_exposed")
    selected = kwargs["selected_case_ids"]
    result = QualityPreparationV1(
        freeze_digest=identity(frozen),
        mapping_digest=frozen.mapping_digest,
        validation_case_ids=tuple(c for c in selected if c in frozen.validation_case_ids),
        **kwargs,
    )
    verify_preparation(dataset_root, result)
    return result


def verify_private(report: QualityGenerationReportV1, root: Path, *, dataset_root: Path) -> str:
    """Read every declared original artifact; never export bodies or reviewer prose."""
    private_path(root / "manifest.json")
    files = {}
    for item in report.private_files:
        raw = read_private(root / item.name)
        if len(raw) != item.byte_count or quality_digest(raw) != item.digest:
            raise QualityBaselineError("private_digest_mismatch")
        if any(marker and marker in raw.decode("utf-8") for marker in secret_markers()):
            raise QualityBaselineError("unsafe_private_artifact")
        files[item.name] = raw
    if "manifest.json" not in files or "sources.json" not in files:
        raise QualityBaselineError("private_inventory_incomplete")
    if QualityGenerationStartV1.model_validate_json(files["manifest.json"]) != report.start:
        raise QualityBaselineError("private_manifest_mismatch")
    sources = QualityPrivateSourcesV1.model_validate_json(files["sources.json"])
    if len({s.source_alias for s in sources.sources}) != len(sources.sources) or any(
        quality_digest(s.text.encode()) != s.digest for s in sources.sources
    ):
        raise QualityBaselineError("private_sources_mismatch")
    dataset = load_quality_dataset(dataset_root)
    expected_sources = tuple(
        (s.alias, s.kind, normalize_document_content(_read_file(dataset_root, s.path).decode()))
        for s in dataset.manifest.sources
    )
    if tuple((s.source_alias, s.kind, s.text) for s in sources.sources) != expected_sources:
        raise QualityBaselineError("private_dataset_mismatch")
    cases = {c.case_id: c for c in dataset.cases}
    expected = {"manifest.json", "sources.json"}
    for i, item in enumerate(report.cases):
        o = item.observation
        if o.status != "not_run":
            name = f"case-{i:04d}.json"
            expected.add(name)
            if name not in files:
                raise QualityBaselineError("private_inventory_incomplete")
            case = QualityPrivateCaseV1.model_validate_json(files[name])
            original = cases[o.case_id]
            if (case.input.mode, case.input.query, case.resume_alias, case.web_scenario_alias) != (
                original.mode,
                original.query,
                original.resume_alias,
                original.web_scenario_alias,
            ):
                raise QualityBaselineError("private_case_input_mismatch")
            if (case.case_id, case.repeat_index, case.output_digest, case.failure_type) != (
                o.case_id,
                o.repeat_index,
                o.output_digest,
                o.failure_type,
            ):
                raise QualityBaselineError("private_case_mismatch")
        if o.output_digest is not None:
            name = f"output-{i:04d}.json"
            expected.add(name)
            output = QualityPrivateOutputV1.model_validate_json(files[name])
            if (output.case_id, output.repeat_index) != (o.case_id, o.repeat_index):
                raise QualityBaselineError("private_output_mismatch")
    if set(files) != expected:
        raise QualityBaselineError("private_inventory_mismatch")
    return quality_identity_digest([f.model_dump(mode="json") for f in report.private_files])


def score_inputs(dataset_root, preparation, report, labels=(), *, private_dir=None, **kwargs):
    dataset, _ = verify_preparation(dataset_root, preparation)
    if type(report) is QualityGenerationReportV1:
        manifest, mapping, index = report.start.manifest, report.start.mapping_digest, 1
        if private_dir is None:
            raise QualityBaselineError("private_path_required")
        verify_private(report, private_dir, dataset_root=dataset_root)
    elif type(report) is QualityRetrievalReportV1:
        manifest, mapping, index = report.manifest, report.mapping_digest, 0
        if labels:
            raise QualityBaselineError("retrieval_annotations_not_supported")
    else:
        raise QualityBaselineError("invalid_report_type")
    if manifest != preparation.manifests[index] or mapping != preparation.mapping_digest:
        raise QualityBaselineError("execution_plan_mismatch")
    policy = report.start.policy.retrieval if index else report.policy
    if manifest.retrieval_policy_digest != identity(policy):
        raise QualityBaselineError("retrieval_policy_mismatch")
    return score_quality(dataset, report, labels, **kwargs)


def derive_candidate(preparation, scores, private_digest, *, exposed=False):
    """Deterministic readiness; real input verification is mandatory before publication/accept."""
    blockers = set()
    if exposed:
        blockers.add("validation_exposed")
    for i, score in enumerate(scores):
        validate_scored_report(score)
        layer = ("retrieval", "generation")[i]
        if score.manifest != preparation.manifests[i] or score.execution_kind != layer:
            raise QualityBaselineError("score_plan_mismatch")
        m = score.manifest
        if (
            m.llm_mode != "qwen"
            or m.document_mode != "real_embedding_db"
            or m.measurement_scope != layer
            or m.web_mode != ("none" if i == 0 else "frozen_fixture")
            or m.model != (LOCKED_EMBEDDING_MODEL if i == 0 else LOCKED_CHAT_MODEL)
            or m.suite_version != f"quality-{layer}-v1"
        ):
            blockers.add("not_live")
        if not score.evidence_valid:
            blockers.add("invalid_evidence")
        if not score.measurement_complete:
            blockers.add("incomplete_measurement")
        if score.summary.coverage.unassessed:
            blockers.add("missing_assessment")
        if any(any(c.observation.safety.model_dump().values()) for c in score.cases):
            blockers.add("safety_violation")
        if not score.summary.provider_attempts:
            blockers.add("missing_live_attempts")
    valid = not blockers.intersection(
        {
            "validation_exposed",
            "invalid_evidence",
            "safety_violation",
            "not_live",
            "missing_live_attempts",
        }
    )
    complete = all(s.measurement_complete for s in scores)
    targets = []
    for target in preparation.targets:
        s = scores[0 if target.layer == "retrieval" else 1].summary
        metric = (
            s.retrieval_micro["recall_at_5"]
            if target.metric == "recall_at_5"
            else getattr(s, target.metric)
        )
        targets.append(
            None
            if metric.value is None or metric.unassessed_count
            else metric.value >= target.minimum
        )
    target_met = (
        all(targets)
        if targets and valid and complete and all(t is not None for t in targets)
        else None
    )
    return QualityCandidateV1(
        preparation=preparation,
        preparation_digest=identity(preparation),
        private_artifacts_digest=private_digest,
        scores=scores,
        blockers=tuple(sorted(blockers)),
        evidence_valid=valid,
        measurement_complete=complete,
        meets_quality_target=target_met,
        acceptance_ready=not blockers,
    )


def candidate(dataset_root: Path, preparation, paths: QualityEvidencePathsV1):
    _, exposures = verify_preparation(dataset_root, preparation)
    reports = (
        load(Path(paths.retrieval_report), QualityRetrievalReportV1),
        load(Path(paths.generation_report), QualityGenerationReportV1),
    )
    scores = (
        load(Path(paths.retrieval_score), QualityScoredReportV1),
        load(Path(paths.generation_score), QualityScoredReportV1),
    )
    labels = annotations(Path(paths.annotations))
    for i, (report, score) in enumerate(zip(reports, scores, strict=True)):
        recalculated = score_inputs(
            dataset_root,
            preparation,
            report,
            labels if i else (),
            private_dir=Path(paths.generation_private_dir) if i else None,
            scorer_source_sha=score.scorer_source_sha,
            scorer_change_reason=score.scorer_change_reason,
        )
        if recalculated != score:
            raise QualityBaselineError("score_recomputation_mismatch")
    private_digest = verify_private(
        reports[1], Path(paths.generation_private_dir), dataset_root=dataset_root
    )
    return derive_candidate(preparation, scores, private_digest, exposed=bool(exposures))


def validate_candidate(value):
    checked = QualityCandidateV1.model_validate_json(value.model_dump_json())
    expected = derive_candidate(
        checked.preparation,
        checked.scores,
        checked.private_artifacts_digest,
        exposed="validation_exposed" in checked.blockers,
    )
    if checked != expected:
        raise QualityBaselineError("candidate_recomputation_mismatch")


def validate_accepted(value):
    value = QualityAcceptedBaselineV1.model_validate_json(value.model_dump_json())
    validate_candidate(value.candidate)
    if not value.candidate.acceptance_ready or value.candidate_digest != identity(value.candidate):
        raise QualityBaselineError("invalid_accepted_baseline")


def baseline_diff(current, previous=None):
    validate_candidate(current)
    if previous is not None:
        validate_accepted(previous)
    old = previous.candidate if previous else None
    return _diff(current, old, identity(previous) if previous else None)


def _diff(current, old, previous_digest):
    scope_fields = ("selected_case_ids", "validation_case_ids", "repeat_count")
    old_scope = tuple(getattr(old.preparation, f) for f in scope_fields) if old else None
    new_scope = tuple(getattr(current.preparation, f) for f in scope_fields)
    return QualityBaselineDiffV1(
        candidate_digest=identity(current),
        previous_baseline_digest=previous_digest,
        first_acceptance=old is None,
        scope_changed=old_scope != new_scope,
        versions_changed=old is None
        or (
            old.preparation.freeze_digest != current.preparation.freeze_digest
            or tuple(s.manifest for s in old.scores) != tuple(s.manifest for s in current.scores)
            or tuple(s.scorer_source_sha for s in old.scores)
            != tuple(s.scorer_source_sha for s in current.scores)
            or old.preparation.targets != current.preparation.targets
        ),
        results_changed=old is None
        or tuple((s.summary, s.cases, s.total_cost) for s in old.scores)
        != tuple((s.summary, s.cases, s.total_cost) for s in current.scores),
        previous=old,
        current=current,
    )


def validate_diff(value):
    validate_candidate(value.current)
    if value.previous is not None:
        validate_candidate(value.previous)
    if (value.previous is None) != (value.previous_baseline_digest is None) or value != _diff(
        value.current, value.previous, value.previous_baseline_digest
    ):
        raise QualityBaselineError("diff_recomputation_mismatch")


def accept(
    dataset_root,
    expected,
    paths,
    reviewed_diff,
    *,
    previous=None,
    baseline_id,
    accepting_source_sha,
    confirm=False,
    human_reviewed=False,
    safety_reviewed=False,
    prepared_before_run=False,
):
    if not all(
        flag is True for flag in (confirm, human_reviewed, safety_reviewed, prepared_before_run)
    ):
        raise QualityBaselineError("explicit_confirmation_required")
    actual = candidate(dataset_root, expected.preparation, paths)
    if actual != expected:
        raise QualityBaselineError("candidate_inputs_changed")
    if not actual.acceptance_ready:
        raise QualityBaselineError("candidate_not_ready")
    diff = baseline_diff(actual, previous)
    if diff != reviewed_diff:
        raise QualityBaselineError("reviewed_diff_mismatch")
    return QualityAcceptedBaselineV1(
        baseline_id=baseline_id,
        accepting_source_sha=accepting_source_sha,
        candidate=actual,
        candidate_digest=identity(actual),
        diff_digest=identity(diff),
        previous_baseline_digest=diff.previous_baseline_digest,
        human_output_review_confirmed=True,
        safety_review_confirmed=True,
        preparation_preceded_execution_confirmed=True,
    )


def publish(path: Path, value):
    """Typed, scanned metadata only; exclusive creation retains every partial failure."""
    if type(value) not in PUBLIC_TYPES:
        raise QualityBaselineError("invalid_public_artifact")
    checked = type(value).model_validate_json(value.model_dump_json())
    if isinstance(checked, QualityCandidateV1):
        validate_candidate(checked)
    elif isinstance(checked, QualityBaselineDiffV1):
        validate_diff(checked)
    elif isinstance(checked, QualityAcceptedBaselineV1):
        validate_accepted(checked)
    data = (checked.model_dump_json(indent=2) + "\n").encode()
    if any(m and m in data.decode() for m in secret_markers()):
        raise QualityBaselineError("unsafe_public_artifact")
    parent = checked_directory(path.parent, private=False)
    descriptor = os.open(parent, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return quality_digest(data)
