"""E4.10 explicit Agent review provenance; no provider calls or human attestation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from tests.evals import quality_baseline as base
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.quality_baseline_contracts import QualityCandidateV1
from tests.evals.quality_contracts import (
    Count,
    HumanAnnotationV1,
    QualityGenerationReportV1,
    SourceSHA,
)
from tests.evals.quality_dataset import quality_digest, quality_identity_digest
from tests.evals.quality_freeze import identity
from tests.evals.quality_review import encoded, read_private, write_new
from tests.evals.quality_score import validate_scored_report
from tests.evals.quality_score_contracts import QualityScoredReportV1

Safety = Literal["clear", "blocked", "unknown"]


class Delegation(EvalContractModel):
    authorization_id: Literal["e410_agent_review_accept_v1"]
    authorization_source: Literal["user_explicit_e410_review_accept_commit_push_plan"]
    reviewer_kind: Literal["agent"] = "agent"
    reviewer_id: Literal["codex_agent"] = "codex_agent"
    agent_identity: Literal["codex_gpt6"] = "codex_gpt6"
    human_reviewed: Literal[False] = False
    independent_review: Literal[False] = False
    preparation_digest: EvalDigest
    report_digest: EvalDigest
    rubric_digest: EvalDigest


class ReviewNote(EvalContractModel):
    """Private prose and label vocabulary, never a public artifact or human claim."""

    case_id: EvalIdentifier
    repeat_index: Count
    output_digest: EvalDigest | None
    private_case_digest: EvalDigest | None
    predecessor_digest: EvalDigest | None
    annotation: HumanAnnotationV1 | None
    safety: Safety
    explanation: str = Field(min_length=1, max_length=100_000)


class ReviewSlot(EvalContractModel):
    case_id: EvalIdentifier
    repeat_index: Count
    output_digest: EvalDigest | None
    initial_digest: EvalDigest
    recheck_digest: EvalDigest
    safety: Safety


class SafetyNote(EvalContractModel):
    private_artifacts_digest: EvalDigest
    safety: Safety
    explanation: str = Field(min_length=1, max_length=100_000)


class DelegatedReviewV1(EvalContractModel):
    artifact_kind: Literal["e410_delegated_review_v1"] = "e410_delegated_review_v1"
    delegation: Delegation
    slots: tuple[ReviewSlot, ...]
    annotation_digest: EvalDigest
    safety_note_digest: EvalDigest
    safety: Safety


class DelegatedScoreV1(EvalContractModel):
    artifact_kind: Literal["e410_delegated_score_v1"] = "e410_delegated_score_v1"
    review: DelegatedReviewV1
    score: QualityScoredReportV1


class DelegatedCandidateV1(EvalContractModel):
    artifact_kind: Literal["e410_delegated_candidate_v1"] = "e410_delegated_candidate_v1"
    review: DelegatedReviewV1
    candidate: QualityCandidateV1
    review_blockers: tuple[Literal["safety_review_blocked", "safety_review_unknown"], ...]
    acceptance_ready: bool


class DelegatedDiffV1(EvalContractModel):
    artifact_kind: Literal["e410_delegated_diff_v1"] = "e410_delegated_diff_v1"
    current: DelegatedCandidateV1
    previous: DelegatedCandidateV1 | None
    previous_baseline_digest: EvalDigest | None
    first_acceptance: bool
    scope_changed: bool
    versions_changed: bool
    results_changed: bool
    review_changed: bool


class DelegatedAcceptedV1(EvalContractModel):
    artifact_kind: Literal["e410_delegated_accepted_v1"] = "e410_delegated_accepted_v1"
    baseline_id: EvalIdentifier
    accepting_source_sha: SourceSHA
    candidate: DelegatedCandidateV1
    candidate_digest: EvalDigest
    reviewed_diff: DelegatedDiffV1
    diff_digest: EvalDigest
    confirmation_kind: Literal["user_authorized_agent_attestation"] = (
        "user_authorized_agent_attestation"
    )
    preparation_preceded_execution_confirmed: Literal[True]
    baseline_accepted: Literal[True] = True
    stage: Literal["BASELINE_COMPLETE"] = "BASELINE_COMPLETE"


def verify_review(dataset_root, preparation, paths, review, review_dir):
    """Reopen original evidence and both immutable review rounds before using labels."""
    review = DelegatedReviewV1.model_validate_json(review.model_dump_json())
    base.verify_preparation(dataset_root, preparation)
    report = base.load(Path(paths.generation_report), QualityGenerationReportV1)
    private_digest = base.verify_private(
        report, Path(paths.generation_private_dir), dataset_root=dataset_root
    )
    d = review.delegation
    if (
        d.preparation_digest != identity(preparation)
        or d.report_digest != identity(report)
        or report.start.manifest != preparation.manifests[1]
        or d.rubric_digest != preparation.manifests[1].rubric_digest
    ):
        raise base.QualityBaselineError("delegation_identity_mismatch")
    labels = base.annotations(Path(paths.annotations))
    if review.annotation_digest != quality_identity_digest(
        [a.model_dump(mode="json") for a in labels]
    ):
        raise base.QualityBaselineError("delegated_annotation_mismatch")
    observations = tuple(
        (c.observation.case_id, c.observation.repeat_index, c.observation.output_digest)
        for c in report.cases
    )
    if tuple((s.case_id, s.repeat_index, s.output_digest) for s in review.slots) != observations:
        raise base.QualityBaselineError("review_slot_mismatch")
    final_labels = []
    for i, (slot, case) in enumerate(zip(review.slots, report.cases, strict=True)):
        initial = None
        for phase, digest in (("initial", slot.initial_digest), ("recheck", slot.recheck_digest)):
            data = read_private(base.private_path(review_dir / f"{phase}-{i:04d}.json"))
            if quality_digest(data) != digest:
                raise base.QualityBaselineError("review_note_digest_mismatch")
            note = ReviewNote.model_validate_json(data)
            if (
                (note.case_id, note.repeat_index, note.output_digest) != observations[i]
                or note.private_case_digest != case.private_case_digest
                or note.predecessor_digest != (None if phase == "initial" else slot.initial_digest)
            ):
                raise base.QualityBaselineError("review_note_identity_mismatch")
            if phase == "initial":
                initial = note
            else:
                if note.safety != slot.safety or (
                    initial.safety == "blocked" and note.safety != "blocked"
                ):
                    raise base.QualityBaselineError("review_safety_mismatch")
                if note.annotation is not None:
                    a = note.annotation
                    if (a.case_id, a.repeat_index, a.output_digest) != observations[
                        i
                    ] or a.reviewer_id != d.reviewer_id:
                        raise base.QualityBaselineError("review_label_identity_mismatch")
                    final_labels.append(a)
                if (note.annotation is not None) != (slot.output_digest is not None):
                    raise base.QualityBaselineError("review_output_coverage_mismatch")
    if tuple(final_labels) != labels:
        raise base.QualityBaselineError("review_labels_changed")
    data = read_private(base.private_path(review_dir / "safety.json"))
    if quality_digest(data) != review.safety_note_digest:
        raise base.QualityBaselineError("safety_note_digest_mismatch")
    safety = SafetyNote.model_validate_json(data)
    if safety.private_artifacts_digest != private_digest or safety.safety != review.safety:
        raise base.QualityBaselineError("safety_evidence_mismatch")
    return labels


def validate_score(value):
    validate_scored_report(value.score)
    r, s = value.review, value.score
    if (
        s.execution_kind != "generation"
        or s.input_report_digest != r.delegation.report_digest
        or s.annotation_digest != r.annotation_digest
        or s.manifest.rubric_digest != r.delegation.rubric_digest
        or tuple(
            (c.observation.case_id, c.observation.repeat_index, c.observation.output_digest)
            for c in s.cases
        )
        != tuple((c.case_id, c.repeat_index, c.output_digest) for c in r.slots)
    ):
        raise base.QualityBaselineError("delegated_score_mismatch")


def wrap_candidate(review, value):
    validate_score(DelegatedScoreV1(review=review, score=value.scores[1]))
    if review.delegation.preparation_digest != identity(value.preparation):
        raise base.QualityBaselineError("delegation_identity_mismatch")
    statuses = {review.safety, *(s.safety for s in review.slots)}
    blockers = tuple(f"safety_review_{s}" for s in ("blocked", "unknown") if s in statuses)
    return DelegatedCandidateV1(
        review=review,
        candidate=value,
        review_blockers=blockers,
        acceptance_ready=value.acceptance_ready and not blockers,
    )


def candidate(dataset_root, preparation, paths, review, review_dir, scored):
    verify_review(dataset_root, preparation, paths, review, review_dir)
    validate_score(scored)
    value = base.candidate(dataset_root, preparation, paths)
    if scored.review != review or scored.score != value.scores[1]:
        raise base.QualityBaselineError("delegated_score_mismatch")
    return wrap_candidate(review, value)


def validate_candidate(value):
    base.validate_candidate(value.candidate)
    if value != wrap_candidate(value.review, value.candidate):
        raise base.QualityBaselineError("delegated_candidate_mismatch")


def diff(current, previous=None):
    validate_candidate(current)
    if previous is not None:
        validate_accepted(previous)
    return make_diff(
        current, previous.candidate if previous else None, identity(previous) if previous else None
    )


def make_diff(current, old, previous_digest):
    details = base._diff(current.candidate, old.candidate if old else None, previous_digest)
    return DelegatedDiffV1(
        current=current,
        previous=old,
        previous_baseline_digest=previous_digest,
        first_acceptance=old is None,
        scope_changed=details.scope_changed,
        versions_changed=details.versions_changed,
        results_changed=details.results_changed,
        review_changed=old is None or old.review != current.review,
    )


def validate_diff(value):
    validate_candidate(value.current)
    if value.previous is not None:
        validate_candidate(value.previous)
    if (value.previous is None) != (value.previous_baseline_digest is None) or value != make_diff(
        value.current, value.previous, value.previous_baseline_digest
    ):
        raise base.QualityBaselineError("delegated_diff_mismatch")


def accept(
    dataset_root,
    expected,
    paths,
    review_dir,
    scored,
    reviewed_diff,
    *,
    previous=None,
    baseline_id,
    accepting_source_sha,
    confirm=False,
    prepared_before_run=False,
):
    if confirm is not True or prepared_before_run is not True:
        raise base.QualityBaselineError("explicit_confirmation_required")
    actual = candidate(
        dataset_root, expected.candidate.preparation, paths, expected.review, review_dir, scored
    )
    if actual != expected or not actual.acceptance_ready:
        raise base.QualityBaselineError("delegated_candidate_not_ready")
    if reviewed_diff != diff(actual, previous):
        raise base.QualityBaselineError("reviewed_diff_mismatch")
    return DelegatedAcceptedV1(
        baseline_id=baseline_id,
        accepting_source_sha=accepting_source_sha,
        candidate=actual,
        candidate_digest=identity(actual),
        reviewed_diff=reviewed_diff,
        diff_digest=identity(reviewed_diff),
        preparation_preceded_execution_confirmed=True,
    )


def validate_accepted(value):
    validate_candidate(value.candidate)
    validate_diff(value.reviewed_diff)
    if (
        not value.candidate.acceptance_ready
        or value.candidate_digest != identity(value.candidate)
        or value.diff_digest != identity(value.reviewed_diff)
        or value.reviewed_diff.current != value.candidate
    ):
        raise base.QualityBaselineError("delegated_acceptance_mismatch")


def publish(path, value):
    validators = {
        DelegatedReviewV1: lambda v: None,
        DelegatedScoreV1: validate_score,
        DelegatedCandidateV1: validate_candidate,
        DelegatedDiffV1: validate_diff,
        DelegatedAcceptedV1: validate_accepted,
    }
    if type(value) not in validators:
        raise base.QualityBaselineError("invalid_public_artifact")
    value = type(value).model_validate_json(value.model_dump_json())
    validators[type(value)](value)
    write_new(path, encoded(value))
    return quality_digest(encoded(value))
