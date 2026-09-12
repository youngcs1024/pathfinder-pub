"""Manual offline CLI: prepare, score, candidate, diff, accept. Never calls a provider."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.evals.live_baseline import probe_clean_git_head
from tests.evals.quality_baseline import (
    QualityBaselineError,
    accept,
    annotations,
    baseline_diff,
    candidate,
    load,
    prepare,
    private_path,
    publish,
    score_inputs,
)
from tests.evals.quality_baseline_contracts import (
    QualityAcceptedBaselineV1,
    QualityBaselineDiffV1,
    QualityCandidateV1,
    QualityEvidencePathsV1,
    QualityPreparationV1,
    QualityTargetV1,
)
from tests.evals.quality_contracts import (
    QualityGenerationReportV1,
    QualityRetrievalReportV1,
    QualityRunManifestV1,
)
from tests.evals.quality_review import read_private
from tests.evals.quality_score import write_quality_score


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise QualityBaselineError("invalid_arguments")


def parser():
    result = Parser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("prepare", "score", "candidate", "diff", "accept"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name != "diff":
            command.add_argument("--dataset", type=Path, required=True)
        if name in {"score", "candidate"}:
            command.add_argument("--preparation", type=Path, required=True)
        if name in {"candidate", "accept"}:
            command.add_argument(
                "--inputs",
                type=Path,
                required=True,
                help="Private JSON inventory of original report/label/score paths.",
            )
        if name in {"diff", "accept"}:
            command.add_argument("--candidate", type=Path, required=True)
            previous = command.add_mutually_exclusive_group(required=True)
            previous.add_argument("--previous", type=Path)
            previous.add_argument("--first", action="store_true")
        if name == "prepare":
            command.add_argument("--preparation-id", required=True)
            command.add_argument("--retrieval-manifest", type=Path, required=True)
            command.add_argument("--generation-manifest", type=Path, required=True)
            command.add_argument("--case", action="append", required=True)
            command.add_argument("--repeat", type=int, required=True)
            command.add_argument(
                "--targets",
                type=Path,
                help="Optional private JSON array of pre-run minimum targets.",
            )
        if name == "score":
            command.add_argument("--layer", choices=("retrieval", "generation"), required=True)
            command.add_argument("--report", type=Path, required=True)
            command.add_argument("--annotations", type=Path)
            command.add_argument("--private-dir", type=Path)
            command.add_argument("--scorer-change-reason")
        if name == "accept":
            command.add_argument("--reviewed-diff", type=Path, required=True)
            command.add_argument("--baseline-id", required=True)
            command.add_argument("--confirm-accept", action="store_true")
            command.add_argument("--confirm-human-reviewed", action="store_true")
            command.add_argument("--confirm-safety-reviewed", action="store_true")
            command.add_argument("--confirm-prepared-before-run", action="store_true")
    return result


def evidence_paths(path):
    value = QualityEvidencePathsV1.model_validate_json(read_private(private_path(path)))
    # Relative input paths are relative to the operator inventory, never the repository.
    return QualityEvidencePathsV1(
        **{name: str(path.parent / item) for name, item in value.model_dump().items()}
    )


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command == "prepare":
            targets = ()
            if args.targets is not None:
                raw = json.loads(read_private(private_path(args.targets)))
                targets = tuple(QualityTargetV1.model_validate_json(json.dumps(t)) for t in raw)
            value = prepare(
                args.dataset,
                preparation_id=args.preparation_id,
                preparation_source_sha=probe_clean_git_head(),
                selected_case_ids=tuple(args.case),
                repeat_count=args.repeat,
                manifests=(
                    load(args.retrieval_manifest, QualityRunManifestV1),
                    load(args.generation_manifest, QualityRunManifestV1),
                ),
                targets=targets,
            )
        elif args.command == "score":
            is_generation = args.layer == "generation"
            if not is_generation and (args.annotations or args.private_dir):
                raise QualityBaselineError("invalid_arguments")
            value = score_inputs(
                args.dataset,
                load(args.preparation, QualityPreparationV1),
                load(
                    args.report,
                    QualityGenerationReportV1 if is_generation else QualityRetrievalReportV1,
                ),
                annotations(args.annotations) if args.annotations else (),
                private_dir=args.private_dir,
                scorer_source_sha=probe_clean_git_head(),
                scorer_change_reason=args.scorer_change_reason,
            )
            digest = write_quality_score(args.output, value)
            print(json.dumps({"category": "score_written", "digest": digest}))
            return 0
        elif args.command == "candidate":
            value = candidate(
                args.dataset,
                load(args.preparation, QualityPreparationV1),
                evidence_paths(args.inputs),
            )
        else:
            current = load(args.candidate, QualityCandidateV1)
            previous = load(args.previous, QualityAcceptedBaselineV1) if args.previous else None
            if args.command == "diff":
                value = baseline_diff(current, previous)
            else:
                value = accept(
                    args.dataset,
                    current,
                    evidence_paths(args.inputs),
                    load(args.reviewed_diff, QualityBaselineDiffV1),
                    previous=previous,
                    baseline_id=args.baseline_id,
                    accepting_source_sha=probe_clean_git_head(),
                    confirm=args.confirm_accept,
                    human_reviewed=args.confirm_human_reviewed,
                    safety_reviewed=args.confirm_safety_reviewed,
                    prepared_before_run=args.confirm_prepared_before_run,
                )
        digest = publish(args.output, value)
        blocked = isinstance(value, QualityCandidateV1) and not value.acceptance_ready
        print(
            json.dumps(
                {
                    "category": "candidate_blocked" if blocked else "artifact_written",
                    "digest": digest,
                }
            )
        )
        return 2 if blocked else 0
    except QualityBaselineError as error:
        print(json.dumps({"category": str(error)}))
        return 1
    except Exception:
        # Pydantic, JSON, filesystem and subprocess failures can contain sensitive input.
        print(json.dumps({"category": "quality_baseline_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
