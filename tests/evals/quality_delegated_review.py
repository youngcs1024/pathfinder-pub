"""Opt-in, user-authorized Agent calibration; never human or baseline evidence."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel, EvalDigest
from tests.evals.quality_contracts import QualityGenerationReportV1, QualityRubricV1, SourceSHA
from tests.evals.quality_dataset import load_quality_dataset, quality_digest
from tests.evals.quality_pilot_contracts import CalibrationV1, ReviewPackageV1
from tests.evals.quality_review import (
    ReviewError,
    _import_calibration,
    encoded,
    export_reviews,
    read_private,
    write_new,
)


class DelegationV2(EvalContractModel):
    schema_version: Literal[2] = 2
    authorization_id: Literal["e47_agent_delegation_v1"] = "e47_agent_delegation_v1"
    authorization_source: Literal["user_explicit_e47_plan_approval"] = (
        "user_explicit_e47_plan_approval"
    )
    purpose: Literal["e49_preparation_only"] = "e49_preparation_only"
    reviewer_kind: Literal["agent"] = "agent"
    reviewer_id: Literal["codex_agent"] = "codex_agent"
    agent_identity: Literal["codex_gpt6"] = "codex_gpt6"
    confirmation: Literal["user_authorized_agent"] = "user_authorized_agent"
    source_base_sha: SourceSHA
    implementation_digest: EvalDigest
    source_report_digest: EvalDigest
    human_reviewed: Literal[False] = False
    independent_review: Literal[False] = False


class DelegatedCalibrationV2(EvalContractModel):
    schema_version: Literal[2] = 2
    purpose: Literal["delegated_rubric_calibration_not_baseline"] = (
        "delegated_rubric_calibration_not_baseline"
    )
    delegation: DelegationV2
    authorization_digest: EvalDigest
    round: Literal["initial", "recheck"]
    predecessor_digest: EvalDigest | None = None
    # Reuse the validated label vocabulary, not the legacy human provenance envelope.
    assessment: CalibrationV1

    @model_validator(mode="after")
    def identity(self):
        if (
            self.authorization_digest != quality_digest(encoded(self.delegation))
            or self.assessment.reviewer_id != self.delegation.reviewer_id
            or self.assessment.source_report_digest != self.delegation.source_report_digest
            or (self.round == "recheck") != (self.predecessor_digest is not None)
        ):
            raise ValueError("delegated_identity_mismatch")
        return self


class DelegatedFreezeV2(EvalContractModel):
    schema_version: Literal[2] = 2
    purpose: Literal["delegated_rubric_freeze_not_baseline"] = (
        "delegated_rubric_freeze_not_baseline"
    )
    delegation: DelegationV2
    authorization_digest: EvalDigest
    initial_calibration_digest: EvalDigest
    final_calibration_digest: EvalDigest
    source_report_digest: EvalDigest
    candidate_rubric_digest: EvalDigest
    frozen_rubric_digest: EvalDigest
    reviewed_outputs: int = Field(ge=10, le=15)
    confirmed_stable: Literal[True] = True
    baseline_accepted: Literal[False] = False
    semantic_quality_claim: Literal[False] = False


def export_delegated(run, dataset, candidate, destination, delegation: DelegationV2):
    """Explicit provenance is installed before any forms can be filled."""
    try:
        if quality_digest(read_private(run / "report.json")) != delegation.source_report_digest:
            raise ReviewError("delegation_source_mismatch")
        rubric = QualityRubricV1.model_validate_json(candidate.read_bytes())
        if not rubric.rubric_version.startswith("quality-agent-delegated-candidate-"):
            raise ReviewError("delegated_candidate_required")
        package = export_reviews(run, dataset, candidate, destination)
        write_new(destination / "delegation.json", encoded(delegation))
        return package
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("delegated_export_failed") from None


def import_delegated(review_dir, dataset, *, round, predecessor_digest=None):
    try:
        delegation = DelegationV2.model_validate_json(read_private(review_dir / "delegation.json"))
        package = ReviewPackageV1.model_validate_json(read_private(review_dir / "package.json"))
        if not package.candidate_rubric.rubric_version.startswith(
            "quality-agent-delegated-candidate-"
        ):
            raise ReviewError("delegated_candidate_required")
        return DelegatedCalibrationV2(
            delegation=delegation,
            authorization_digest=quality_digest(encoded(delegation)),
            round=round,
            predecessor_digest=predecessor_digest,
            assessment=_import_calibration(review_dir, dataset, delegation.reviewer_id),
        )
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("delegated_import_failed") from None


def freeze_delegated(
    review_dir: Path,
    dataset: Path,
    initial_path: Path,
    final_path: Path,
    destination: Path,
    *,
    confirm_agent_stable: bool,
):
    try:
        if not confirm_agent_stable:
            raise ReviewError("agent_freeze_confirmation_required")
        initial_raw, final_raw = read_private(initial_path), read_private(final_path)
        initial = DelegatedCalibrationV2.model_validate_json(initial_raw)
        final = DelegatedCalibrationV2.model_validate_json(final_raw)
        current = import_delegated(
            review_dir, dataset, round="recheck", predecessor_digest=quality_digest(initial_raw)
        )

        def slots(c):
            return tuple(
                (x.annotation.case_id, x.annotation.repeat_index, x.annotation.output_digest)
                for x in c.assessment.cases
            )

        if (
            initial.round != "initial"
            or final != current
            or not final.assessment.complete
            or initial.delegation != final.delegation
            or slots(initial) != slots(final)
            or initial.assessment.candidate_rubric_digest
            == final.assessment.candidate_rubric_digest
        ):
            raise ReviewError("delegated_freeze_incomplete_or_changed")
        package = ReviewPackageV1.model_validate_json(read_private(review_dir / "package.json"))
        source = QualityGenerationReportV1.model_validate_json(
            read_private(review_dir / "source-report.json")
        )
        if any(any(c.observation.safety.model_dump().values()) for c in source.cases):
            raise ReviewError("delegated_safety_blocked")
        frozen = QualityRubricV1.model_validate_json(
            package.candidate_rubric.model_copy(
                update={"rubric_version": "quality-agent-delegated-frozen-v1", "status": "frozen"}
            ).model_dump_json()
        )
        record = DelegatedFreezeV2(
            delegation=final.delegation,
            authorization_digest=final.authorization_digest,
            initial_calibration_digest=quality_digest(initial_raw),
            final_calibration_digest=quality_digest(final_raw),
            source_report_digest=final.assessment.source_report_digest,
            candidate_rubric_digest=final.assessment.candidate_rubric_digest,
            frozen_rubric_digest=quality_digest(encoded(frozen)),
            reviewed_outputs=len(final.assessment.cases),
        )
        destination.mkdir(mode=0o700, exist_ok=False)
        write_new(destination / "rubric.json", encoded(frozen))
        write_new(destination / "freeze.json", encoded(record))
        return record
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("delegated_freeze_failed") from None


def public_coverage(review_dir: Path, dataset: Path, calibration: DelegatedCalibrationV2):
    """Allowlisted identifiers/counts only. No copying fact text, questions or locations."""
    package = ReviewPackageV1.model_validate_json(read_private(review_dir / "package.json"))
    if calibration.assessment.package_digest != quality_digest(encoded(package)):
        raise ReviewError("delegated_coverage_mismatch")
    indexed = {c.case_id: c for c in load_quality_dataset(dataset).cases}
    return {
        "purpose": "delegated_pilot_coverage_not_validation",
        "human_reviewed": False,
        "independent_review": False,
        "planned": package.planned,
        "failed": package.failed,
        "not_run": package.not_run,
        "reviewed_outputs": len(calibration.assessment.cases),
        "family_count": len({indexed[s.case_id].family_id for s in package.slots}),
        "cases": [
            {
                "case_id": c.annotation.case_id,
                "repeat_index": c.annotation.repeat_index,
                "output_digest": c.annotation.output_digest,
                "business_result": c.annotation.business_result,
                "draft_usability": c.annotation.draft_usability,
                "insufficiency": c.annotation.insufficiency,
                "fact_counts": {
                    label: sum(f.judgment == label for f in c.annotation.facts)
                    for label in ("supported", "contradicted", "unsupported", "not_assessable")
                },
                "citation_counts": {
                    label: sum(f.judgment == label for f in c.annotation.citations)
                    for label in ("supported", "contradicted", "unsupported", "not_assessable")
                },
                "covered_unit_ids": c.annotation.covered_unit_ids,
                "missing_unit_ids": c.annotation.missing_unit_ids,
            }
            for c in calibration.assessment.cases
        ],
        "validation_frozen": False,
        "baseline_accepted": False,
    }


def main(argv=None):
    """Offline only; explicit files and confirmation, no provider or Docker access."""
    import argparse

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise ReviewError("delegated_arguments_invalid")

    parser = Parser(description=__doc__)
    parser.add_argument("command", choices=("export", "import", "freeze"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--final", type=Path)
    parser.add_argument("--confirm-agent-stable", action="store_true")
    try:
        args = parser.parse_args(argv)
        if args.command == "export":
            export_delegated(
                args.run,
                args.dataset,
                args.candidate,
                args.review_dir,
                DelegationV2.model_validate_json(read_private(args.authorization)),
            )
        elif args.command == "import":
            result = import_delegated(
                args.review_dir,
                args.dataset,
                round="recheck" if args.initial else "initial",
                predecessor_digest=quality_digest(read_private(args.initial))
                if args.initial
                else None,
            )
            write_new(args.output, encoded(result))
        else:
            freeze_delegated(
                args.review_dir,
                args.dataset,
                args.initial,
                args.final,
                args.output,
                confirm_agent_stable=args.confirm_agent_stable,
            )
    except Exception:
        print('{"status":"failed","category":"delegated_operation_rejected"}')
        return 1
    print('{"status":"written","purpose":"agent_calibration_not_baseline"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
