"""Offline E4.9 coverage, contamination review and create-only publication.

Text similarity is a review aid, never proof of semantic independence. This module
has no provider, database, model scoring, or baseline acceptance entry point.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path

from tests.evals.contracts import EvalContractModel
from tests.evals.quality_contracts import QualityMappingReportV1, QualityMappingV1
from tests.evals.quality_dataset import (
    QualityDataset,
    QualityDatasetError,
    _read_file,
    load_quality_dataset,
    prepare_quality_mapping_sources,
    quality_digest,
    quality_identity_digest,
    validate_quality_mapping,
)
from tests.evals.quality_freeze_contracts import (
    CoverageV1,
    ExposureV1,
    FrozenSetV1,
    LeakPairV1,
    LeakReportV1,
    LeakReviewV1,
    ReviewsV1,
)

STRATA = (
    "mixed_technical_versions",
    "claim_strength",
    "multi_paragraph_support",
    "unsupported_experience",
    "no_or_partial_answer",
    "web_conflict_or_time",
    "similar_experience_distractor",
    "prompt_injection",
    "scope_negative",
)
VALIDATION_COUNTS = {s: (1 if s in (STRATA[0], STRATA[5], STRATA[6]) else 2) for s in STRATA}
ALGORITHM = "nfkc-casefold-space-alias-number-char5-jaccard-v1"


class FreezeError(ValueError):
    """Safe diagnostics contain a fixed category, never source text or paths."""


def identity(value: EvalContractModel) -> str:
    return quality_identity_digest(value.model_dump(mode="json"))


def read_artifact(root: Path, name: str, model):
    try:
        return model.model_validate_json(_read_file(root, name))
    except (ValueError, OSError):
        raise FreezeError("invalid_freeze_artifact") from None


def write_artifact(path: Path, value: EvalContractModel) -> None:
    """Exclusive creation; partial failures remain on disk and are never rerolled."""
    try:
        checked = type(value).model_validate_json(value.model_dump_json())
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise FreezeError("invalid_publication_path")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(checked.model_dump_json(indent=2) + "\n")
    except (ValueError, OSError):
        raise FreezeError("freeze_publication_failed") from None


def validate_coverage(dataset: QualityDataset, coverage: CoverageV1) -> None:
    coverage = CoverageV1.model_validate_json(coverage.model_dump_json())
    rubric = next(f.digest for f in dataset.manifest.files if f.role == "rubric")
    cases = {c.case_id: c for c in dataset.cases}
    if (
        coverage.dataset_digest != dataset.manifest_digest
        or coverage.rubric_digest != rubric
        or {c.case_id for c in coverage.cases} != cases.keys()
        or set(dataset.manifest.strata) != set(STRATA)
        or dataset.rubric.status != "frozen"
        or dataset.manifest.review_status != "reviewed"
        or dataset.manifest.reviewer_ids != ("codex_agent",)
    ):
        raise FreezeError("coverage_identity_mismatch")
    for split, expected in (("dev", dict.fromkeys(STRATA, 5)), ("validation", VALIDATION_COUNTS)):
        if Counter(c.stratum for c in dataset.cases if c.split == split) != expected:
            raise FreezeError("coverage_count_mismatch")
    groups: dict[tuple[str, str], set[str]] = {}
    used_sources = set()
    for row in coverage.cases:
        case = cases[row.case_id]
        aliases = {a for a in (case.resume_alias, case.web_scenario_alias) if a}
        if (row.family_id, row.split) != (case.family_id, case.split) or set(
            row.source_aliases
        ) != aliases:
            raise FreezeError("coverage_case_mismatch")
        used_sources.update(aliases)
        for kind, keys in (
            ("family", [row.family_id]),
            ("template", [row.template_id]),
            ("source", row.source_aliases),
        ):
            for key in keys:
                groups.setdefault((kind, key), set()).add(row.split)
    if any(len(splits) != 1 for splits in groups.values()):
        raise FreezeError("cross_split_group")
    if used_sources != {s.alias for s in dataset.manifest.sources} or {
        c.family_id for c in dataset.cases
    } != {f.family_id for f in dataset.manifest.families}:
        raise FreezeError("unused_coverage_identity")
    if any(Path(s.path).suffix not in (".md", ".txt") for s in dataset.manifest.sources):
        raise FreezeError("unsupported_material_format")


def _canonical(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _fingerprint(text: str, aliases: tuple[str, ...]) -> str:
    text = _canonical(text)
    for alias in sorted(aliases, key=len, reverse=True):
        text = text.replace(_canonical(alias), "<alias>")
    return re.sub(r"\d+", "#", text)


def text_similarity(left: str, right: str, aliases: tuple[str, ...]) -> tuple[bool, float]:
    exact = _canonical(left) == _canonical(right)
    a, b = (_fingerprint(text, aliases) for text in (left, right))
    if a == b:
        return exact, 1.0
    grams = [set(text[i : i + 5] for i in range(max(0, len(text) - 4))) for text in (a, b)]
    union = grams[0] | grams[1]
    return exact, len(grams[0] & grams[1]) / len(union) if union else 0.0


def scan_leakage(root: Path, dataset: QualityDataset, coverage: CoverageV1) -> LeakReportV1:
    validate_coverage(dataset, coverage)
    aliases = tuple(s.alias for s in dataset.manifest.sources)
    source_split = {alias: c.split for c in coverage.cases for alias in c.source_aliases}
    items = {
        "query": [(c.case_id, c.query, c.split) for c in dataset.cases],
        "source": [
            (s.alias, _read_file(root, s.path).decode("utf-8"), source_split[s.alias])
            for s in dataset.manifest.sources
        ],
        "unit": [(u.unit_id, u.quote, source_split[u.source_alias]) for u in dataset.units],
    }
    pairs = []
    for kind, rows in items.items():
        for left, right in combinations(sorted(rows), 2):
            exact, similarity = text_similarity(left[1], right[1], aliases)
            if exact or similarity >= 0.8:
                pairs.append(
                    LeakPairV1(
                        pair_id=quality_identity_digest([kind, left[0], right[0]]),
                        kind=kind,
                        left=left[0],
                        right=right[0],
                        cross_split=left[2] != right[2],
                        exact=exact,
                        similarity=similarity,
                    )
                )
    return LeakReportV1(
        algorithm=ALGORITHM,
        dataset_digest=dataset.manifest_digest,
        coverage_digest=identity(coverage),
        pairs=tuple(pairs),
    )


def _validate_reviews(dataset: QualityDataset, coverage: CoverageV1, reviews: ReviewsV1) -> None:
    cases = {c.case_id: c for c in dataset.cases}
    if (
        reviews.dataset_digest != dataset.manifest_digest
        or reviews.coverage_digest != identity(coverage)
        or {r.case_id for r in reviews.cases} != cases.keys()
        or any(
            r.case_digest != identity(cases[r.case_id]) or r.decision != "reviewed"
            for r in reviews.cases
        )
    ):
        raise FreezeError("case_review_incomplete")


def _validate_leak_review(dataset, coverage, report, review):
    decisions = {d.pair_id: d for d in review.decisions}
    if review.report_digest != identity(report) or decisions.keys() != {
        p.pair_id for p in report.pairs
    }:
        raise FreezeError("leak_review_incomplete")
    families = {("query", c.case_id): {c.family_id} for c in coverage.cases}
    for source in dataset.manifest.sources:
        families["source", source.alias] = {
            c.family_id for c in coverage.cases if source.alias in c.source_aliases
        }
    for unit in dataset.units:
        families["unit", unit.unit_id] = families["source", unit.source_alias]
    for pair in report.pairs:
        decision = decisions[pair.pair_id]
        if decision.disposition == "needs_review" or (pair.cross_split and pair.exact):
            raise FreezeError("unresolved_leakage")
        if decision.disposition == "same_family_variant" and (
            pair.cross_split or families[pair.kind, pair.left] != families[pair.kind, pair.right]
        ):
            raise FreezeError("invalid_variant_group")
        # A distinct_context decision is an explicit semantic self-review, not an
        # automatic waiver. Identical masked text cannot be waived across splits.
        if pair.cross_split and pair.similarity == 1.0:
            raise FreezeError("cross_split_duplicate")


def build_freeze(root: Path, *, source_sha: str) -> FrozenSetV1:
    """Recompute all evidence before issuing a candidate; does not publish by itself."""
    try:
        dataset = load_quality_dataset(root)
        coverage = read_artifact(root, "coverage.json", CoverageV1)
        validate_coverage(dataset, coverage)
        reviews = read_artifact(root, "reviews.json", ReviewsV1)
        _validate_reviews(dataset, coverage, reviews)
        leakage = scan_leakage(root, dataset, coverage)
        if leakage != read_artifact(root, "leakage.json", LeakReportV1):
            raise FreezeError("leak_report_mismatch")
        leak_review = read_artifact(root, "leakage-review.json", LeakReviewV1)
        _validate_leak_review(dataset, coverage, leakage, leak_review)
        mapping = read_artifact(root, "mapping.json", QualityMappingV1)
        if any(
            chunk.reviewer_id != "codex_agent"
            for source in mapping.sources
            for chunk in source.chunks
        ) or any(j.reviewer_id != "codex_agent" for j in mapping.judgments):
            raise FreezeError("mapping_reviewer_mismatch")
        rules_digest = quality_digest(_read_file(root, "mapping-rules.md"))
        report = validate_quality_mapping(
            dataset, mapping, prepare_quality_mapping_sources(root), rules_digest=rules_digest
        )
        if not report.review_complete or report != read_artifact(
            root, "mapping-report.json", QualityMappingReportV1
        ):
            raise FreezeError("mapping_review_incomplete")
        return FrozenSetV1(
            freeze_version="quality-expanded-freeze-v1",
            execution_source_sha=source_sha,
            dataset_digest=dataset.manifest_digest,
            rubric_digest=coverage.rubric_digest,
            split_digest=quality_identity_digest(
                [[f.family_id, f.split] for f in dataset.manifest.families]
            ),
            coverage_digest=identity(coverage),
            reviews_digest=identity(reviews),
            leakage_digest=identity(leakage),
            leakage_review_digest=identity(leak_review),
            mapping_digest=identity(mapping),
            mapping_report_digest=identity(report),
            mapping_rules_digest=rules_digest,
            dev_count=sum(c.split == "dev" for c in dataset.cases),
            validation_count=sum(c.split == "validation" for c in dataset.cases),
            family_count=len({c.family_id for c in dataset.cases}),
            validation_case_ids=tuple(c.case_id for c in dataset.cases if c.split == "validation"),
        )
    except FreezeError:
        raise
    except (ValueError, OSError, KeyError, UnicodeError):
        raise FreezeError("invalid_freeze_inputs") from None


def check_frozen(root: Path) -> tuple[FrozenSetV1, tuple[ExposureV1, ...]]:
    frozen = read_artifact(root, "freeze.json", FrozenSetV1)
    if build_freeze(root, source_sha=frozen.execution_source_sha) != frozen:
        raise FreezeError("freeze_identity_mismatch")
    exposures = []
    directory = root / "exposures"
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise FreezeError("invalid_exposure_directory")
        for path in sorted(directory.iterdir()):
            record = read_artifact(root, f"exposures/{path.name}", ExposureV1)
            if (
                path.name != f"{record.exposure_id}.json"
                or record.freeze_digest != identity(frozen)
                or record.case_id not in frozen.validation_case_ids
            ):
                raise FreezeError("exposure_identity_mismatch")
            exposures.append(record)
    return frozen, tuple(exposures)


def record_exposure(root: Path, record: ExposureV1) -> None:
    record = ExposureV1.model_validate_json(record.model_dump_json())
    frozen, _ = check_frozen(root)
    if record.freeze_digest != identity(frozen) or record.case_id not in frozen.validation_case_ids:
        raise FreezeError("exposure_identity_mismatch")
    directory = root / "exposures"
    directory.mkdir(exist_ok=True)
    write_artifact(directory / f"{record.exposure_id}.json", record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "freeze", "expose"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--confirm-agent-reviewed", action="store_true")
    parser.add_argument("--exposure-id")
    parser.add_argument("--case-id")
    parser.add_argument("--reason", choices=("prompt_tuning", "code_debugging"))
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            if not args.confirm_agent_reviewed or not args.source_sha:
                raise FreezeError("freeze_confirmation_required")
            if (args.dataset / "exposures").exists():
                raise FreezeError("existing_exposure_history")
            write_artifact(
                args.dataset / "freeze.json", build_freeze(args.dataset, source_sha=args.source_sha)
            )
        elif args.command == "expose":
            frozen, _ = check_frozen(args.dataset)
            record_exposure(
                args.dataset,
                ExposureV1(
                    exposure_id=args.exposure_id,
                    freeze_digest=identity(frozen),
                    case_id=args.case_id,
                    execution_source_sha=args.source_sha,
                    reason=args.reason,
                ),
            )
        frozen, exposures = check_frozen(args.dataset)
        print(
            json.dumps(
                {
                    "freeze_digest": identity(frozen),
                    "dev_count": frozen.dev_count,
                    "validation_count": frozen.validation_count,
                    "exposure_count": len(exposures),
                    "replacement_required": bool(exposures),
                    "measurement_complete": False,
                    "baseline_accepted": False,
                }
            )
        )
        return 2 if exposures else 0
    except (FreezeError, QualityDatasetError, ValueError, OSError):
        print('{"status":"failed","category":"quality_freeze_failed"}')
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
