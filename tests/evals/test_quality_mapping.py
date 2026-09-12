"""Offline mapping contracts, not retrieval quality or independent human review."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.evals.quality_contracts import (
    QualityChunkSpanV1,
    QualityMappingReportV1,
    QualityMappingV1,
    SourceEvidenceUnitV1,
)
from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    map_quality_units,
    prepare_quality_mapping_sources,
    quality_context_judgments,
    quality_digest,
    suggest_quality_source_mapping,
    validate_quality_mapping,
    write_quality_mapping,
)

PILOT = Path(__file__).resolve().parents[2] / "evals/datasets/quality_v1"
ARTIFACT = PILOT.parent / "quality_mappings/pilot_v1_heading_v1"
RULES = quality_digest(b"test mapping rules")


def bundle(text, ranges, source_type="markdown"):
    source = normalize_and_chunk_batch(
        ValidatedIngestionBatch(
            sources=(
                ValidatedIngestionSource("resume_alpha", source_type, "test", text, len(text)),
            )
        )
    ).sources[0]
    original = load_quality_dataset(PILOT)
    units = tuple(
        SourceEvidenceUnitV1(
            unit_id=f"unit_{index}",
            source_alias="resume_alpha",
            source_kind="resume",
            normalized_text_digest=quality_digest(source.content.encode()),
            start=start,
            end=end,
            quote=source.content[start:end],
            fact="synthetic mapping vector",
        )
        for index, (start, end) in enumerate(ranges)
    )
    dataset = replace(
        original,
        manifest=original.manifest.model_copy(update={"sources": original.manifest.sources[:1]}),
        cases=(
            original.cases[0].model_copy(
                update={"required_unit_ids": tuple(u.unit_id for u in units)}
            ),
        ),
        units=units,
    )
    sources = (suggest_quality_source_mapping("resume_alpha", source),)
    mapping = QualityMappingV1(
        mapping_version="test-v1",
        execution_source_sha="a" * 40,
        dataset_version=dataset.manifest.dataset_version,
        dataset_digest=dataset.manifest_digest,
        normalization_version=source.normalization_version,
        chunking_version=source.chunking_version,
        rules_digest=RULES,
        sources=sources,
        units=map_quality_units(dataset, sources),
        judgments=(),
    )
    return dataset, mapping, {"resume_alpha": source}


def check(dataset, mapping, prepared):
    return validate_quality_mapping(dataset, mapping, prepared, rules_digest=RULES)


def with_chunks(dataset, mapping, chunks):
    sources = (mapping.sources[0].model_copy(update={"chunks": tuple(chunks)}),)
    return mapping.model_copy(
        update={"sources": sources, "units": map_quality_units(dataset, sources)}
    )


def test_unicode_offsets_and_byte_budget_are_distinct():
    dataset, mapping, prepared = bundle("甲🙂乙", [(1, 2)])
    assert mapping.units[0].covered_codepoints == 1
    assert prepared["resume_alpha"].chunks[0].token_count == 10
    report = check(dataset, mapping, prepared)
    assert report.mapping_complete
    assert not report.review_complete
    assert report.pending_review_chunks == 1
    assert report.unjudged_contexts == 1


def test_repeated_heading_and_quote_require_review_without_first_match():
    text = "## 同题\n\n同句。\n\n## 同题\n\n同句。\n\n"
    start = text.rindex("同句。")
    dataset, mapping, prepared = bundle(text, [(start, start + 3)])
    assert all(c.resolution == "needs_review" for c in mapping.sources[0].chunks)
    assert all(len(c.candidate_spans) == 2 and not c.spans for c in mapping.sources[0].chunks)
    assert check(dataset, mapping, prepared).ambiguous_chunks == 2
    assert mapping.units[0].status == "unmapped"
    chunks = tuple(
        chunk.model_copy(
            update={
                "spans": (chunk.candidate_spans[index],),
                "resolution": "reviewed",
                "reviewer_id": "test_reviewer",
            }
        )
        for index, chunk in enumerate(mapping.sources[0].chunks)
    )
    reviewed = with_chunks(dataset, mapping, chunks)
    assert check(dataset, reviewed, prepared).mapping_complete
    assert reviewed.units[0].chunk_ordinals == (1,)


def test_discontinuous_fragments_after_whitespace_rewrite_keep_holes_visible():
    dataset, mapping, prepared = bundle("甲\n \n乙", [(0, 1), (4, 5), (0, 5)], "text")
    chunk = mapping.sources[0].chunks[0]
    assert chunk.resolution == "needs_review"
    assert chunk.candidate_spans == ()
    spans = (
        QualityChunkSpanV1(source_start=0, source_end=1, chunk_start=0, chunk_end=1),
        QualityChunkSpanV1(source_start=4, source_end=5, chunk_start=3, chunk_end=4),
    )
    reviewed = with_chunks(
        dataset,
        mapping,
        [chunk.model_copy(update={"spans": spans, "resolution": "reviewed", "reviewer_id": "r"})],
    )
    report = check(dataset, reviewed, prepared)
    assert report.complete_units == 2
    assert report.partial_units == 1
    assert not report.mapping_complete
    assert reviewed.units[-1].covered_codepoints == 2


def test_one_fact_spans_chunks_and_multiple_required_units_remain_separate():
    text = "甲" * 300
    dataset, mapping, prepared = bundle(text, [(0, 300)], "text")
    # Repetitive content has multiple possible chunk origins: require explicit review.
    position = 0
    chunks = []
    for chunk in mapping.sources[0].chunks:
        size = chunk.codepoint_count
        span = QualityChunkSpanV1(
            source_start=position, source_end=position + size, chunk_start=0, chunk_end=size
        )
        chunks.append(
            chunk.model_copy(
                update={"spans": (span,), "resolution": "reviewed", "reviewer_id": "r"}
            )
        )
        position += size
    reviewed = with_chunks(dataset, mapping, chunks)
    assert check(dataset, reviewed, prepared).complete_units == 1
    assert reviewed.units[0].chunk_ordinals == (0, 1)
    dataset, mapping, prepared = bundle("# 一\n\n甲\n\n# 二\n\n乙", [(5, 6), (13, 14)])
    assert len(mapping.units) == 2
    assert [u.chunk_ordinals for u in mapping.units] == [(0,), (1,)]
    assert check(dataset, mapping, prepared).complete_units == 2


@pytest.mark.parametrize("field", ["dataset_digest", "rules_digest", "chunking_version"])
def test_stale_identity_is_rejected(field):
    dataset, mapping, prepared = bundle("甲乙", [(0, 2)])
    value = "other-v2" if field == "chunking_version" else "sha256:" + "0" * 64
    with pytest.raises(QualityDatasetError, match=r"^mapping_.*mismatch$"):
        check(dataset, mapping.model_copy(update={field: value}), prepared)


def test_strategy_change_needs_new_mapping_without_changing_facts():
    text = "# 一\n\n甲\n\n# 二\n\n乙"
    dataset, mapping, prepared = bundle(text, [(5, 6), (13, 14)])
    old_units = tuple(u.model_dump_json() for u in dataset.units)
    # A test-only alternative input strategy produces one merged chunk, not two.
    _, _, alternative = bundle(text, [(5, 6), (13, 14)], "text")
    changed = {"resume_alpha": replace(alternative["resume_alpha"], chunking_version="test-v2")}
    assert len(prepared["resume_alpha"].chunks) == 2
    assert len(changed["resume_alpha"].chunks) == 1
    with pytest.raises(QualityDatasetError, match=r"^mapping_profile_mismatch$"):
        check(dataset, mapping, changed)
    sources = (suggest_quality_source_mapping("resume_alpha", changed["resume_alpha"]),)
    new = mapping.model_copy(
        update={
            "mapping_version": "test-v2",
            "chunking_version": "test-v2",
            "sources": sources,
            "units": map_quality_units(dataset, sources),
        }
    )
    assert check(dataset, new, changed).mapping_complete
    assert [u.chunk_ordinals for u in new.units] == [(0,), (0,)]
    assert tuple(u.model_dump_json() for u in dataset.units) == old_units


@pytest.mark.parametrize("failure", ["digest", "range", "overlap", "missing", "coverage"])
def test_corrupt_mapping_is_rejected_safely(failure):
    dataset, mapping, prepared = bundle("private-mapping-canary", [(0, 7)])
    chunk = mapping.sources[0].chunks[0]
    if failure == "digest":
        chunks = [chunk.model_copy(update={"content_digest": "sha256:" + "0" * 64})]
    elif failure == "missing":
        chunks = []
    else:
        span = chunk.spans[0]
        if failure == "range":
            spans = (
                span.model_copy(update={"source_start": 1, "source_end": span.source_end + 1}),
            )
        elif failure == "overlap":
            spans = (span, span)
        else:
            spans = (QualityChunkSpanV1(source_start=0, source_end=1, chunk_start=0, chunk_end=1),)
        chunks = [
            chunk.model_copy(update={"spans": spans, "resolution": "reviewed", "reviewer_id": "r"})
        ]
    with pytest.raises(QualityDatasetError) as error:
        check(dataset, with_chunks(dataset, mapping, chunks), prepared)
    assert "canary" not in str(error.value)


def test_relevance_is_explicit_and_rejected_scope_never_has_contexts():
    dataset, mapping, prepared = bundle("甲乙", [(0, 2)])
    judgment = quality_context_judgments(dataset, mapping)[0]
    assert judgment.relevance == "unjudged"
    for relevance, reason in (("relevant", "task_context"), ("irrelevant", "off_topic")):
        assessed = judgment.model_copy(
            update={"relevance": relevance, "reason": reason, "reviewer_id": "r"}
        )
        report = check(dataset, mapping.model_copy(update={"judgments": (assessed,)}), prepared)
        assert report.unjudged_contexts == 0
        assert report.relevant_contexts == (relevance == "relevant")
        assert report.irrelevant_contexts == (relevance == "irrelevant")
    denied = replace(
        dataset, cases=(dataset.cases[0].model_copy(update={"scope_expectation": "reject"}),)
    )
    assert quality_context_judgments(denied, mapping) == ()
    for update in ({"source_alias": "resume_beta"}, {"case_id": "other"}, {"chunk_ordinal": 99}):
        with pytest.raises(QualityDatasetError, match=r"^mapping_judgment_scope_mismatch$"):
            changed = mapping.model_copy(
                update={"judgments": (judgment.model_copy(update=update),)}
            )
            check(dataset, changed, prepared)
    with pytest.raises(QualityDatasetError, match=r"^mapping_judgment_scope_mismatch$"):
        check(denied, mapping.model_copy(update={"judgments": (judgment,)}), prepared)


def test_duplicate_judgments_and_false_coverage_fail():
    dataset, mapping, prepared = bundle("甲乙", [(0, 2)])
    judgment = quality_context_judgments(dataset, mapping)[0]
    with pytest.raises(QualityDatasetError, match=r"^invalid_quality_mapping$"):
        check(dataset, mapping.model_copy(update={"judgments": (judgment, judgment)}), prepared)
    unit = mapping.units[0].model_copy(update={"covered_codepoints": 1})
    with pytest.raises(QualityDatasetError, match=r"^mapping_unit_coverage_mismatch$"):
        check(dataset, mapping.model_copy(update={"units": (unit,)}), prepared)


def test_publication_is_create_only_and_body_free(tmp_path):
    _, mapping, _ = bundle("private-mapping-canary", [(0, 7)])
    path = tmp_path / "mapping.json"
    write_quality_mapping(path, mapping)
    original = path.read_bytes()
    assert b"canary" not in original
    assert b"quote" not in original
    with pytest.raises(QualityDatasetError, match=r"^mapping_publication_failed$"):
        write_quality_mapping(path, mapping)
    assert path.read_bytes() == original


def test_committed_pilot_mapping_matches_actual_production_chunks():
    dataset = load_quality_dataset(PILOT)
    mapping = QualityMappingV1.model_validate_json((ARTIFACT / "mapping.json").read_bytes())
    report = validate_quality_mapping(
        dataset,
        mapping,
        prepare_quality_mapping_sources(PILOT),
        rules_digest=quality_digest((ARTIFACT / "rules.md").read_bytes()),
    )
    saved = QualityMappingReportV1.model_validate_json((ARTIFACT / "report.json").read_bytes())
    assert report == saved
    assert report.mapping_complete and report.review_complete
    assert report.total_units == 30
    assert report.complete_units == 20
    assert report.not_applicable_units == 10
    assert report.total_chunks == 23
    assert all(u.chunk_ordinals == () for u in mapping.units if u.status == "not_applicable")
    assert {j.case_id for j in mapping.judgments}.isdisjoint({"scope_missing", "scope_rejected"})
    # Gold/source bodies remain in the dataset, never copied into public mapping reports.
    public = json.dumps(mapping.model_dump(mode="json"), ensure_ascii=False)
    assert all(u.quote not in public and u.fact not in public for u in dataset.units)
    for case in dataset.cases:
        assert case.query not in public
