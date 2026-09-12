"""Explicit offline E4.10 Agent scoring and acceptance; never calls a provider."""

from pathlib import Path

from tests.evals import quality_baseline as base
from tests.evals import quality_baseline_delegated as delegated
from tests.evals.live_baseline import probe_clean_git_head
from tests.evals.quality_baseline_cli import Parser, evidence_paths
from tests.evals.quality_baseline_contracts import QualityPreparationV1
from tests.evals.quality_contracts import QualityGenerationReportV1
from tests.evals.quality_score import write_quality_score


def main(argv=None):
    try:
        parser = Parser(description=__doc__)
        parser.add_argument("command", choices=("score", "candidate", "diff", "accept"))
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--preparation", type=Path)
        parser.add_argument("--inputs", type=Path)
        parser.add_argument("--review", type=Path)
        parser.add_argument("--review-dir", type=Path)
        parser.add_argument("--scored", type=Path)
        parser.add_argument("--candidate", type=Path)
        parser.add_argument("--reviewed-diff", type=Path)
        previous = parser.add_mutually_exclusive_group()
        previous.add_argument("--previous", type=Path)
        previous.add_argument("--first", action="store_true")
        parser.add_argument("--baseline-id")
        parser.add_argument("--scorer-change-reason")
        parser.add_argument("--confirm-accept", action="store_true")
        parser.add_argument("--confirm-prepared-before-run", action="store_true")
        args = parser.parse_args(argv)
        if args.command in {"score", "candidate", "accept"}:
            paths = evidence_paths(args.inputs)
        if args.command in {"score", "candidate"}:
            prep = base.load(args.preparation, QualityPreparationV1)
            review = base.load(args.review, delegated.DelegatedReviewV1)
        if args.command == "score":
            labels = delegated.verify_review(args.dataset, prep, paths, review, args.review_dir)
            score = base.score_inputs(
                args.dataset,
                prep,
                base.load(Path(paths.generation_report), QualityGenerationReportV1),
                labels,
                private_dir=Path(paths.generation_private_dir),
                scorer_source_sha=probe_clean_git_head(),
                scorer_change_reason=args.scorer_change_reason,
            )
            value = delegated.DelegatedScoreV1(review=review, score=score)
            delegated.validate_score(value)
            write_quality_score(Path(paths.generation_score), score)
        elif args.command == "candidate":
            value = delegated.candidate(
                args.dataset,
                prep,
                paths,
                review,
                args.review_dir,
                base.load(args.scored, delegated.DelegatedScoreV1),
            )
        else:
            if not (args.first or args.previous):
                raise base.QualityBaselineError("previous_baseline_choice_required")
            current = base.load(args.candidate, delegated.DelegatedCandidateV1)
            old = base.load(args.previous, delegated.DelegatedAcceptedV1) if args.previous else None
            if args.command == "diff":
                value = delegated.diff(current, old)
            else:
                value = delegated.accept(
                    args.dataset,
                    current,
                    paths,
                    args.review_dir,
                    base.load(args.scored, delegated.DelegatedScoreV1),
                    base.load(args.reviewed_diff, delegated.DelegatedDiffV1),
                    previous=old,
                    baseline_id=args.baseline_id,
                    accepting_source_sha=probe_clean_git_head(),
                    confirm=args.confirm_accept,
                    prepared_before_run=args.confirm_prepared_before_run,
                )
        delegated.publish(args.output, value)
        blocked = isinstance(value, delegated.DelegatedCandidateV1) and not value.acceptance_ready
        print("delegated_candidate_blocked" if blocked else "delegated_artifact_written")
        return 2 if blocked else 0
    except Exception:
        print("delegated_baseline_failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
