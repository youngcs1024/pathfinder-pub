"""Offline dataset linkage and create-only mappings; no experiment runner or scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.retrieval.chunking import (
    PreparedIngestionSource,
    normalize_and_chunk_batch,
    normalize_document_content,
)
from app.retrieval.ingestion import (
    IngestionInputError,
    ValidatedIngestionBatch,
    ValidatedIngestionSource,
)
from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityCaseV1,
    QualityChunkJudgmentV1,
    QualityChunkMappingV1,
    QualityChunkSpanV1,
    QualityDatasetManifestV1,
    QualityMappingReportV1,
    QualityMappingV1,
    QualityModelPayloadV1,
    QualityObservationV1,
    QualityReportV1,
    QualityRubricV1,
    QualityRunManifestV1,
    QualitySourceMappingV1,
    QualityUnitMappingV1,
    SourceEvidenceUnitV1,
    require_unique,
)

MAX_FILE_BYTES = 1_000_000


class QualityDatasetError(ValueError):
    """Safe boundary error: never includes paths, source content or validation inputs."""


def quality_digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def quality_identity_digest(payload: object) -> str:
    return quality_digest(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


@dataclass(frozen=True)
class QualityDataset:
    manifest: QualityDatasetManifestV1 = field(repr=False)
    manifest_digest: str
    rubric: QualityRubricV1 = field(repr=False)
    cases: tuple[QualityCaseV1, ...] = field(repr=False)
    units: tuple[SourceEvidenceUnitV1, ...] = field(repr=False)


def _read_file(root: Path, relative: str) -> bytes:
    path = Path(relative)
    if path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise QualityDatasetError("invalid_file_path")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise QualityDatasetError("symlink_not_allowed")
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise QualityDatasetError("invalid_file_path")
    with current.open("rb") as stream:
        content = stream.read(MAX_FILE_BYTES + 1)
    if len(content) > MAX_FILE_BYTES:
        raise QualityDatasetError("file_too_large")
    return content


def _scope_units(dataset: QualityDataset, case: QualityCaseV1) -> set[str]:
    aliases = {a for a in (case.resume_alias, case.web_scenario_alias) if a is not None}
    return {u.unit_id for u in dataset.units if u.source_alias in aliases}


def _validate_dataset(dataset: QualityDataset, contents: dict[str, bytes]) -> None:
    manifest = dataset.manifest
    require_unique(tuple(c.case_id for c in dataset.cases))
    require_unique(tuple(u.unit_id for u in dataset.units))
    if not dataset.cases:
        raise ValueError("dataset requires cases")
    sources = {s.alias: s for s in manifest.sources}
    units = {u.unit_id: u for u in dataset.units}
    families = {f.family_id: f.split for f in manifest.families}
    texts = {
        alias: normalize_document_content(contents[source.path].decode("utf-8"))
        for alias, source in sources.items()
    }
    for unit in dataset.units:
        source = sources.get(unit.source_alias)
        if source is None or source.kind != unit.source_kind:
            raise ValueError("source unit references an invalid source")
        text = texts[unit.source_alias]
        if quality_digest(text.encode()) != unit.normalized_text_digest:
            raise ValueError("normalized evidence digest mismatch")
        if text[unit.start : unit.end] != unit.quote:
            raise ValueError("evidence range does not match normalized source")
        for alternative_id in unit.alternative_unit_ids:
            alternative = units.get(alternative_id)
            if alternative is None or alternative.source_kind != unit.source_kind:
                raise ValueError("invalid alternative evidence reference")
    for case in dataset.cases:
        if families.get(case.family_id) != case.split or case.stratum not in manifest.strata:
            raise ValueError("case split or stratum does not match manifest")
        for alias, kind in ((case.resume_alias, "resume"), (case.web_scenario_alias, "web")):
            if alias is not None and (alias not in sources or sources[alias].kind != kind):
                raise ValueError("case references an invalid source alias")
        allowed = _scope_units(dataset, case)
        if not set(case.required_unit_ids).issubset(allowed):
            raise ValueError("gold references evidence outside the case scope")
        for unit_id in case.required_unit_ids:
            if not set(units[unit_id].alternative_unit_ids).issubset(allowed):
                raise ValueError("alternative evidence crosses the case scope")


def load_quality_dataset(root: Path) -> QualityDataset:
    try:
        manifest_bytes = _read_file(root, "manifest.json")
        manifest = QualityDatasetManifestV1.model_validate_json(manifest_bytes)
        contents = {}
        for entry in manifest.files:
            content = _read_file(root, entry.path)
            if quality_digest(content) != entry.digest:
                raise QualityDatasetError("file_digest_mismatch")
            contents[entry.path] = content
        roles = {entry.role: entry.path for entry in manifest.files}
        rubric = QualityRubricV1.model_validate_json(contents[roles["rubric"]])
        cases = tuple(
            QualityCaseV1.model_validate_json(line)
            for line in contents[roles["cases"]].decode("utf-8").splitlines()
            if line.strip()
        )
        units = tuple(
            SourceEvidenceUnitV1.model_validate_json(line)
            for line in contents[roles["source_units"]].decode("utf-8").splitlines()
            if line.strip()
        )
        dataset = QualityDataset(manifest, quality_digest(manifest_bytes), rubric, cases, units)
        _validate_dataset(dataset, contents)
        return dataset
    except QualityDatasetError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise QualityDatasetError("invalid_quality_dataset") from None


def project_model_payload(case: QualityCaseV1) -> QualityModelPayloadV1:
    # Revalidate even model_copy/model_construct objects before projection. A rejected
    # scope is a harness negative case, never permission to call a model on that scope.
    try:
        checked = QualityCaseV1.model_validate_json(case.model_dump_json())
        if checked.scope_expectation == "reject":
            raise QualityDatasetError("scope_rejected")
        return QualityModelPayloadV1(mode=checked.mode, query=checked.query)
    except ValidationError:
        raise QualityDatasetError("invalid_quality_case") from None


def validate_quality_run(dataset: QualityDataset, run: QualityRunManifestV1) -> None:
    """Check frozen identities, selection and family split without executing anything."""
    cases = {c.case_id: c for c in dataset.cases}
    rubric_file = next(f for f in dataset.manifest.files if f.role == "rubric")
    splits = [[f.family_id, f.split] for f in dataset.manifest.families]
    if (
        run.dataset_version != dataset.manifest.dataset_version
        or run.dataset_digest != dataset.manifest_digest
        or run.rubric_version != dataset.rubric.rubric_version
        or run.rubric_digest != rubric_file.digest
        or run.case_set_digest != quality_identity_digest(list(run.selected_case_ids))
        or run.split_digest != quality_identity_digest(splits)
        or not set(run.selected_case_ids).issubset(cases)
    ):
        raise QualityDatasetError("run_dataset_identity_mismatch")


def validate_quality_annotations(
    dataset: QualityDataset,
    run: QualityRunManifestV1,
    observations: tuple[QualityObservationV1, ...],
    annotations: tuple[HumanAnnotationV1, ...],
) -> None:
    validate_quality_run(dataset, run)
    expected = tuple((s.case_id, s.repeat_index) for s in run.execution_order)
    actual = tuple((o.case_id, o.repeat_index) for o in observations)
    if actual != expected or any(
        o.experiment_id != run.experiment_id or o.measurement_scope != run.measurement_scope
        for o in observations
    ):
        raise QualityDatasetError("observation_scope_mismatch")
    by_slot = {(o.case_id, o.repeat_index): o for o in observations}
    cases = {c.case_id: c for c in dataset.cases}
    seen = set()
    for annotation in annotations:
        key = (annotation.case_id, annotation.repeat_index, annotation.reviewer_id)
        observation = by_slot.get((annotation.case_id, annotation.repeat_index))
        if (
            key in seen
            or observation is None
            or not observation.assessment_required
            or annotation.experiment_id != run.experiment_id
            or annotation.output_digest != observation.output_digest
            or annotation.rubric_version != run.rubric_version
        ):
            raise QualityDatasetError("annotation_identity_mismatch")
        seen.add(key)
        allowed = _scope_units(dataset, cases[annotation.case_id])
        if not set(annotation.covered_unit_ids + annotation.missing_unit_ids).issubset(allowed):
            raise QualityDatasetError("annotation_scope_mismatch")


def validate_quality_report(
    dataset: QualityDataset,
    run: QualityRunManifestV1,
    annotations: tuple[HumanAnnotationV1, ...],
    report: QualityReportV1,
) -> None:
    validate_quality_annotations(dataset, run, report.observations, annotations)
    if (
        report.experiment_id != run.experiment_id
        or report.execution_source_sha != run.execution_source_sha
        or report.manifest_digest != quality_digest(run.model_dump_json().encode())
    ):
        raise QualityDatasetError("report_manifest_mismatch")
    expected_digest = (
        quality_identity_digest([a.model_dump(mode="json") for a in annotations])
        if annotations
        else None
    )
    assessed = set()
    required_reviewers = (
        2 if dataset.rubric.review_policy == "independent_review_with_adjudication" else 1
    )
    for slot in run.execution_order:
        reviews = [
            a
            for a in annotations
            if (a.case_id, a.repeat_index) == (slot.case_id, slot.repeat_index)
        ]
        if (
            len(reviews) >= required_reviewers
            and all(a.assessment_complete for a in reviews)
            and not any(d.status == "unresolved" for a in reviews for d in a.disagreements)
        ):
            assessed.add((slot.case_id, slot.repeat_index))
    if report.annotation_digest != expected_digest or report.coverage.assessed != len(assessed):
        raise QualityDatasetError("report_annotation_mismatch")
    cases = {c.case_id: c for c in dataset.cases}
    for group in report.groups:
        attribute = {"stratum": "stratum", "family": "family_id", "split": "split"}[group.dimension]
        members = [
            o for o in report.observations if getattr(cases[o.case_id], attribute) == group.group_id
        ]
        required = sum(o.assessment_required for o in members)
        assessed_count = sum((o.case_id, o.repeat_index) in assessed for o in members)
        expected_coverage = {
            "planned": len(members),
            "executed": sum(o.status != "not_run" for o in members),
            "failed": sum(o.status == "failed" for o in members),
            "not_run": sum(o.status == "not_run" for o in members),
            "generated": sum(o.output_digest is not None for o in members),
            "assessment_required": required,
            "assessed": assessed_count,
            "unassessed": required - assessed_count,
        }
        if not members or group.coverage.model_dump() != expected_coverage:
            raise QualityDatasetError("report_group_mismatch")


def prepare_quality_mapping_sources(root: Path) -> dict[str, PreparedIngestionSource]:
    """Offline production chunk output, with no DB, embeddings or Web ingestion."""
    try:
        dataset = load_quality_dataset(root)
        prepared = {}
        for source in dataset.manifest.sources:
            if source.kind != "resume":
                continue
            raw = _read_file(root, source.path)
            entry = next(f for f in dataset.manifest.files if f.path == source.path)
            if quality_digest(raw) != entry.digest:
                raise QualityDatasetError("file_digest_mismatch")
            text = raw.decode("utf-8")
            suffix = Path(source.path).suffix
            if suffix not in (".md", ".txt"):
                raise QualityDatasetError("invalid_mapping_source")
            batch = ValidatedIngestionBatch(
                sources=(
                    ValidatedIngestionSource(
                        source_name=source.alias,
                        source_type="markdown" if suffix == ".md" else "text",
                        title=source.alias,
                        raw_text=text,
                        character_count=len(text),
                    ),
                )
            )
            prepared[source.alias] = normalize_and_chunk_batch(batch).sources[0]
        return prepared
    except (OSError, UnicodeError, ValueError, IngestionInputError):
        raise QualityDatasetError("invalid_mapping_source") from None


def _occurrences(text: str, fragment: str) -> tuple[int, ...]:
    matches = []
    position = text.find(fragment)
    while position >= 0:
        matches.append(position)
        position = text.find(fragment, position + 1)
    return tuple(matches)


def suggest_quality_source_mapping(
    source_alias: str, source: PreparedIngestionSource
) -> QualitySourceMappingV1:
    """Suggest whole-chunk exact matches only; ambiguity never selects a position."""
    chunks = []
    for chunk in source.chunks:
        candidates = tuple(
            QualityChunkSpanV1(
                source_start=start,
                source_end=start + len(chunk.text),
                chunk_start=0,
                chunk_end=len(chunk.text),
            )
            for start in _occurrences(source.content, chunk.text)
        )
        chunks.append(
            QualityChunkMappingV1(
                ordinal=chunk.ordinal,
                content_digest=quality_digest(chunk.text.encode()),
                codepoint_count=len(chunk.text),
                candidate_spans=candidates,
                spans=candidates if len(candidates) == 1 else (),
                resolution="unique_exact" if len(candidates) == 1 else "needs_review",
            )
        )
    return QualitySourceMappingV1(
        source_alias=source_alias,
        normalized_text_digest=quality_digest(source.content.encode()),
        chunks=tuple(chunks),
    )


def map_quality_units(
    dataset: QualityDataset, sources: tuple[QualitySourceMappingV1, ...]
) -> tuple[QualityUnitMappingV1, ...]:
    """Count the union of exact source ranges, including whitespace, once per unit."""
    by_alias = {source.source_alias: source for source in sources}
    mappings = []
    for unit in dataset.units:
        covered: set[int] = set()
        ordinals = []
        source = by_alias.get(unit.source_alias)
        if source is not None and unit.source_kind == "resume":
            for chunk in source.chunks:
                intersected = set()
                for span in chunk.spans:
                    intersected.update(
                        range(max(unit.start, span.source_start), min(unit.end, span.source_end))
                    )
                if intersected:
                    ordinals.append(chunk.ordinal)
                    covered.update(intersected)
        if unit.source_kind == "web":
            status = "not_applicable"
        elif len(covered) == unit.end - unit.start:
            status = "complete"
        else:
            status = "partial" if covered else "unmapped"
        mappings.append(
            QualityUnitMappingV1(
                unit_id=unit.unit_id,
                source_alias=unit.source_alias,
                status=status,
                covered_codepoints=len(covered),
                chunk_ordinals=tuple(ordinals),
            )
        )
    return tuple(mappings)


def _judgment_slots(
    dataset: QualityDataset, sources: tuple[QualitySourceMappingV1, ...]
) -> set[tuple[str, str, int]]:
    by_alias = {source.source_alias: source for source in sources}
    return {
        (case.case_id, case.resume_alias, chunk.ordinal)
        for case in dataset.cases
        if case.scope_expectation == "allowed" and case.resume_alias in by_alias
        for chunk in by_alias[case.resume_alias].chunks
    }


def quality_context_judgments(
    dataset: QualityDataset, mapping: QualityMappingV1
) -> tuple[QualityChunkJudgmentV1, ...]:
    """Missing judgments stay unjudged, including chunks without gold unit links."""
    supplied = {(j.case_id, j.source_alias, j.chunk_ordinal): j for j in mapping.judgments}
    slots = _judgment_slots(dataset, mapping.sources)
    if len(supplied) != len(mapping.judgments) or not supplied.keys() <= slots:
        raise QualityDatasetError("mapping_judgment_scope_mismatch")
    return tuple(
        supplied.get(slot)
        or QualityChunkJudgmentV1(
            case_id=slot[0],
            source_alias=slot[1],
            chunk_ordinal=slot[2],
            relevance="unjudged",
            reason="not_judged",
        )
        for slot in sorted(slots)
    )


def _validate_source_mapping(
    mapping: QualitySourceMappingV1, source: PreparedIngestionSource
) -> None:
    expected = suggest_quality_source_mapping(mapping.source_alias, source)
    if mapping.normalized_text_digest != expected.normalized_text_digest or tuple(
        c.ordinal for c in mapping.chunks
    ) != tuple(c.ordinal for c in source.chunks):
        raise QualityDatasetError("mapping_source_identity_mismatch")
    for mapped, proposed, chunk in zip(mapping.chunks, expected.chunks, source.chunks, strict=True):
        if (
            mapped.content_digest != proposed.content_digest
            or mapped.codepoint_count != len(chunk.text)
            or mapped.candidate_spans != proposed.candidate_spans
        ):
            raise QualityDatasetError("mapping_chunk_identity_mismatch")
        if mapped.resolution != "reviewed" and mapped != proposed:
            raise QualityDatasetError("mapping_requires_review")
        covered: set[int] = set()
        for span in mapped.spans:
            if (
                span.source_end > len(source.content)
                or span.chunk_end > len(chunk.text)
                or source.content[span.source_start : span.source_end]
                != chunk.text[span.chunk_start : span.chunk_end]
            ):
                raise QualityDatasetError("invalid_mapping_fragment")
            positions = set(range(span.chunk_start, span.chunk_end))
            if covered.intersection(positions):
                raise QualityDatasetError("invalid_mapping_fragment")
            covered.update(positions)
        # Whitespace rewritten by the chunker may be left unmapped. It never grants
        # source coverage; any resulting hole in a unit remains partial/unmapped.
        if mapped.resolution == "reviewed" and any(
            not char.isspace() and index not in covered for index, char in enumerate(chunk.text)
        ):
            raise QualityDatasetError("incomplete_reviewed_chunk")


def validate_quality_mapping(
    dataset: QualityDataset,
    mapping: QualityMappingV1,
    prepared: dict[str, PreparedIngestionSource],
    *,
    rules_digest: str,
) -> QualityMappingReportV1:
    """Validate against actual chunk outputs; return a body-free completeness report."""
    try:
        mapping = QualityMappingV1.model_validate_json(mapping.model_dump_json())
        aliases = {s.alias for s in dataset.manifest.sources if s.kind == "resume"}
        if (
            mapping.dataset_digest != dataset.manifest_digest
            or mapping.dataset_version != dataset.manifest.dataset_version
            or mapping.rules_digest != rules_digest
            or set(prepared) != aliases
            or {s.source_alias for s in mapping.sources} != aliases
        ):
            raise QualityDatasetError("mapping_dataset_identity_mismatch")
        manifest_sources = {s.alias: s for s in dataset.manifest.sources}
        for source_mapping in mapping.sources:
            source = prepared[source_mapping.source_alias]
            manifest_source = manifest_sources[source_mapping.source_alias]
            if (
                source.normalization_version != mapping.normalization_version
                or source.normalization_version != manifest_source.normalization_version
                or source.chunking_version != mapping.chunking_version
                or quality_digest(source.content.encode()) != "sha256:" + source.content_hash
                or any(
                    unit.normalized_text_digest != source_mapping.normalized_text_digest
                    or source.content[unit.start : unit.end] != unit.quote
                    for unit in dataset.units
                    if unit.source_alias == source_mapping.source_alias
                )
            ):
                raise QualityDatasetError("mapping_profile_mismatch")
            _validate_source_mapping(source_mapping, source)
        if mapping.units != map_quality_units(dataset, mapping.sources):
            raise QualityDatasetError("mapping_unit_coverage_mismatch")
        judgments = quality_context_judgments(dataset, mapping)
        chunks = [chunk for source in mapping.sources for chunk in source.chunks]
        statuses = [unit.status for unit in mapping.units]
        pending = sum(chunk.resolution != "reviewed" for chunk in chunks)
        unjudged = sum(j.relevance == "unjudged" for j in judgments)
        unresolved = sum(chunk.resolution == "needs_review" for chunk in chunks)
        complete = not unresolved and not {"partial", "unmapped"}.intersection(statuses)
        return QualityMappingReportV1(
            mapping_digest=quality_identity_digest(mapping.model_dump(mode="json")),
            total_units=len(statuses),
            complete_units=statuses.count("complete"),
            partial_units=statuses.count("partial"),
            unmapped_units=statuses.count("unmapped"),
            not_applicable_units=statuses.count("not_applicable"),
            total_chunks=len(chunks),
            ambiguous_chunks=sum(len(c.candidate_spans) > 1 for c in chunks),
            unresolved_chunks=unresolved,
            pending_review_chunks=pending,
            relevant_contexts=sum(j.relevance == "relevant" for j in judgments),
            irrelevant_contexts=sum(j.relevance == "irrelevant" for j in judgments),
            unjudged_contexts=unjudged,
            mapping_complete=complete,
            review_complete=complete and pending == 0 and unjudged == 0,
        )
    except QualityDatasetError:
        raise
    except (ValueError, KeyError, TypeError):
        raise QualityDatasetError("invalid_quality_mapping") from None


def write_quality_mapping(path: Path, mapping: QualityMappingV1) -> None:
    """Create only; validation and review precede publication by the caller."""
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(mapping.model_dump_json(indent=2) + "\n")
    except (OSError, ValueError):
        raise QualityDatasetError("mapping_publication_failed") from None
