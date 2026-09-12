"""Offline delegated provenance tests; synthetic labels are not real Agent reviews."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evals import quality_baseline as base
from tests.evals import quality_baseline_delegated as agent
from tests.evals.quality_baseline_delegated_cli import main
from tests.evals.quality_contracts import QualityGenerationReportV1
from tests.evals.quality_dataset import quality_digest, quality_identity_digest
from tests.evals.quality_freeze import identity
from tests.evals.quality_review import ReviewError, encoded, write_new
from tests.evals.test_quality_baseline import DATA, OTHER_SHA, evidence, frozen  # noqa: F401


@pytest.fixture
def reviewed(evidence, tmp_path):  # noqa: F811
    prep, paths, scores = evidence
    report = base.load(Path(paths.generation_report), QualityGenerationReportV1)
    labels = base.annotations(Path(paths.annotations))
    labels = tuple(a.model_copy(update={"reviewer_id": "codex_agent"}) for a in labels)
    Path(paths.annotations).write_text("\n".join(a.model_dump_json() for a in labels))
    scores = (
        scores[0],
        base.score_inputs(
            DATA,
            prep,
            report,
            labels,
            private_dir=Path(paths.generation_private_dir),
            scorer_source_sha=OTHER_SHA,
            scorer_change_reason="e410_delegated_review",
        ),
    )
    # Fixture repair only. Production publishers remain exclusive.
    Path(paths.generation_score).write_bytes(encoded(scores[1]))
    root = tmp_path / "agent"
    root.mkdir(mode=0o700)
    slots = []
    for i, (case, label) in enumerate(zip(report.cases, labels, strict=True)):
        o = case.observation
        initial = agent.ReviewNote(
            case_id=o.case_id,
            repeat_index=o.repeat_index,
            output_digest=o.output_digest,
            private_case_digest=case.private_case_digest,
            predecessor_digest=None,
            annotation=label,
            safety="clear",
            explanation="Synthetic initial review",
        )
        write_new(root / f"initial-{i:04d}.json", encoded(initial))
        digest = quality_digest(encoded(initial))
        recheck = initial.model_copy(
            update={"predecessor_digest": digest, "explanation": "Synthetic recheck"}
        )
        write_new(root / f"recheck-{i:04d}.json", encoded(recheck))
        slots.append(
            agent.ReviewSlot(
                case_id=o.case_id,
                repeat_index=o.repeat_index,
                output_digest=o.output_digest,
                initial_digest=digest,
                recheck_digest=quality_digest(encoded(recheck)),
                safety="clear",
            )
        )
    safety = agent.SafetyNote(
        private_artifacts_digest=base.verify_private(
            report, Path(paths.generation_private_dir), dataset_root=DATA
        ),
        safety="clear",
        explanation="Synthetic safety review",
    )
    write_new(root / "safety.json", encoded(safety))
    review = agent.DelegatedReviewV1(
        delegation=agent.Delegation(
            authorization_id="e410_agent_review_accept_v1",
            authorization_source="user_explicit_e410_review_accept_commit_push_plan",
            preparation_digest=identity(prep),
            report_digest=identity(report),
            rubric_digest=prep.manifests[1].rubric_digest,
        ),
        slots=tuple(slots),
        annotation_digest=quality_identity_digest([a.model_dump(mode="json") for a in labels]),
        safety_note_digest=quality_digest(encoded(safety)),
        safety="clear",
    )
    scored = agent.DelegatedScoreV1(review=review, score=scores[1])
    return prep, paths, root, review, scored


def build(items):
    prep, paths, root, review, scored = items
    return agent.candidate(DATA, prep, paths, review, root, scored)


def accept(items, value, previous=None):
    _, paths, root, _, scored = items
    return agent.accept(
        DATA,
        value,
        paths,
        root,
        scored,
        agent.diff(value, previous),
        previous=previous,
        baseline_id="e410_test",
        accepting_source_sha=OTHER_SHA,
        confirm=True,
        prepared_before_run=True,
    )


def test_delegated_roundtrip_preserves_failure_and_scorer_provenance(reviewed, tmp_path):
    current = build(reviewed)
    assert current.acceptance_ready
    assert current.candidate.scores[0].summary.coverage.failed == 2
    assert current.candidate.scores[1].scorer_source_sha == OTHER_SHA
    assert current.candidate.meets_quality_target is None
    accepted = accept(reviewed, current)
    assert not accepted.candidate.review.delegation.human_reviewed
    assert not accepted.candidate.review.delegation.independent_review
    assert accepted.reviewed_diff.first_acceptance
    later = agent.diff(current, accepted)
    assert not later.first_acceptance
    assert not later.results_changed
    assert later.previous_baseline_digest == identity(accepted)
    agent.validate_accepted(accept(reviewed, current, accepted))
    directory = tmp_path / "public"
    directory.mkdir(mode=0o700)
    path = directory / "accepted.json"
    agent.publish(path, accepted)
    with pytest.raises(ReviewError):
        agent.publish(path, accepted)


@pytest.mark.parametrize(
    "field,value",
    [
        ("authorization_id", "e47_agent_delegation_v1"),
        ("human_reviewed", True),
        ("independent_review", True),
        ("reviewer_kind", "human"),
    ],
)
def test_no_old_authorization_or_human_impersonation(reviewed, field, value):
    raw = reviewed[3].model_dump(mode="json")
    raw["delegation"][field] = value
    with pytest.raises(ValidationError):
        agent.DelegatedReviewV1.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("field", ["preparation_digest", "report_digest", "rubric_digest"])
def test_authorization_binding(reviewed, field):
    prep, paths, root, review, _ = reviewed
    changed = review.model_copy(
        update={"delegation": review.delegation.model_copy(update={field: "sha256:" + "0" * 64})}
    )
    with pytest.raises(base.QualityBaselineError):
        agent.verify_review(DATA, prep, paths, changed, root)


@pytest.mark.parametrize("name", ["initial-0000.json", "recheck-0000.json", "safety.json"])
def test_review_tampering(reviewed, name):
    (reviewed[2] / name).write_text("{}")
    with pytest.raises(base.QualityBaselineError):
        build(reviewed)


def test_missing_slot_and_cross_run_labels(reviewed):
    prep, paths, root, review, _ = reviewed
    with pytest.raises(base.QualityBaselineError):
        agent.verify_review(
            DATA, prep, paths, review.model_copy(update={"slots": review.slots[:-1]}), root
        )
    labels = base.annotations(Path(paths.annotations))
    Path(paths.annotations).write_text(
        "\n".join(
            a.model_copy(update={"experiment_id": "other_run"}).model_dump_json() for a in labels
        )
    )
    with pytest.raises(base.QualityBaselineError):
        build(reviewed)


@pytest.mark.parametrize("status", ["blocked", "unknown"])
def test_additional_safety_review_blocks_without_rewriting_observation(reviewed, status):
    current = build(reviewed)
    changed = agent.wrap_candidate(
        current.review.model_copy(update={"safety": status}), current.candidate
    )
    assert not changed.acceptance_ready
    assert current.candidate.acceptance_ready
    assert changed.review_blockers == (f"safety_review_{status}",)


def test_accept_reopens_inputs_and_requires_confirmation(reviewed):
    current = build(reviewed)
    _, paths, root, _, scored = reviewed
    with pytest.raises(base.QualityBaselineError, match="explicit_confirmation"):
        agent.accept(
            DATA,
            current,
            paths,
            root,
            scored,
            agent.diff(current),
            baseline_id="test",
            accepting_source_sha=OTHER_SHA,
            prepared_before_run=True,
        )
    (root / "safety.json").write_text("{}")
    with pytest.raises(base.QualityBaselineError):
        accept(reviewed, current)


def test_diff_tamper_and_public_type_whitelist(reviewed, tmp_path):
    current = build(reviewed)
    with pytest.raises(base.QualityBaselineError):
        agent.validate_diff(agent.diff(current).model_copy(update={"results_changed": False}))
    note = agent.ReviewNote.model_validate_json((reviewed[2] / "initial-0000.json").read_bytes())
    with pytest.raises(base.QualityBaselineError, match="invalid_public_artifact"):
        agent.publish(tmp_path / "leak.json", note)
    raw = current.model_dump(mode="json")
    raw["private_body"] = "sensitive"
    with pytest.raises(ValidationError):
        agent.DelegatedCandidateV1.model_validate_json(json.dumps(raw))


def test_cli_failure_does_not_echo_private_arguments(capsys):
    assert main(["accept", "--output", "/private-canary/out"]) == 1
    assert capsys.readouterr().out == "delegated_baseline_failed\n"
