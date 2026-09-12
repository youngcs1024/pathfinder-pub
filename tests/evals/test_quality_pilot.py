"""E4.3 pilot integrity checks, not human review or a semantic quality evaluation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from app.retrieval.chunking import normalize_document_content
from app.retrieval.ingestion import validate_and_load_documents
from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    project_model_payload,
    quality_digest,
)

PILOT = Path(__file__).resolve().parents[2] / "evals/datasets/quality_v1"
EXPECTED_STRATA = {
    "mixed_technical_versions": 3,
    "claim_strength": 3,
    "multi_paragraph_support": 3,
    "unsupported_experience": 3,
    "no_or_partial_answer": 3,
    "web_conflict_or_time": 3,
    "similar_experience_distractor": 2,
    "prompt_injection": 2,
    "scope_negative": 2,
}


@pytest.fixture
def pilot_copy(tmp_path):
    for source in PILOT.rglob("*"):
        if source.is_file():
            target = tmp_path / source.relative_to(PILOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return tmp_path


def rewrite_cases(root, change):
    path = root / "cases.dev.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if entry["role"] == "cases":
            entry["digest"] = quality_digest(path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))


def test_pilot_coverage_families_and_review_are_explicit():
    dataset = load_quality_dataset(PILOT)
    assert dataset.manifest.dataset_version == "quality-pilot-v1"
    assert dataset.manifest.purpose == "pilot"
    assert dataset.manifest.license_category == "synthetic"
    assert dataset.manifest.review_status == "not_reviewed"
    assert dataset.manifest.reviewer_ids == ()
    assert dataset.rubric.status == "draft"
    assert len(dataset.cases) == 24
    assert Counter(case.stratum for case in dataset.cases) == EXPECTED_STRATA
    assert set(dataset.manifest.strata) == set(EXPECTED_STRATA)
    assert {case.split for case in dataset.cases} == {"dev"}
    assert {family.split for family in dataset.manifest.families} == {"dev"}
    families = defaultdict(set)
    for case in dataset.cases:
        families[case.resume_alias].add(case.family_id)
    assert len(families) == 4
    assert all(len(group) == 1 for group in families.values())
    assert len(set().union(*families.values())) == 4
    review = (PILOT / "review-checklist.md").read_text()
    for case in dataset.cases:
        assert review.count(f"\n## {case.case_id}\n") == 1
        assert f"| {case.case_id} | | | |" in review


def test_markdown_sources_pass_existing_file_input_boundary():
    dataset = load_quality_dataset(PILOT)
    resumes = [source for source in dataset.manifest.sources if source.kind == "resume"]
    assert len(resumes) == 3
    # Check each file separately. Web is never included in a resume ingestion batch.
    # No DB, embedding provider, actual ingestion or chunk mapping runs here.
    for source in dataset.manifest.sources:
        assert Path(source.path).suffix == ".md"
        batch = validate_and_load_documents([PILOT / source.path])
        assert len(batch.sources) == 1
        assert batch.sources[0].source_type == "markdown"
        normalized = normalize_document_content(batch.sources[0].raw_text)
        assert normalized == batch.sources[0].raw_text
        for unit in dataset.units:
            if unit.source_alias == source.alias:
                assert normalized[unit.start : unit.end] == unit.quote
                assert quality_digest(normalized.encode()) == unit.normalized_text_digest


def test_required_units_stay_in_scope_and_multisection_cases_need_both_parts():
    dataset = load_quality_dataset(PILOT)
    units = {unit.unit_id: unit for unit in dataset.units}
    for case in dataset.cases:
        allowed = {case.resume_alias, case.web_scenario_alias} - {None}
        required = [units[unit_id] for unit_id in case.required_unit_ids]
        assert all(unit.source_alias in allowed for unit in required)
        if case.stratum == "multi_paragraph_support":
            assert len(required) == 2
            assert {unit.source_alias for unit in required} == {case.resume_alias}
            assert required[0].end < required[1].start
            assert "\n## " in next(
                (PILOT / source.path).read_text()[required[0].end : required[1].start]
                for source in dataset.manifest.sources
                if source.alias == case.resume_alias
            )
    no_answer = next(case for case in dataset.cases if "no_answer" in case.tags)
    assert no_answer.required_unit_ids == ()


def test_scope_negative_distinguishes_optional_resume_from_denied_resource():
    dataset = load_quality_dataset(PILOT)
    cases = {case.case_id: case for case in dataset.cases}
    missing = cases["scope_missing"]
    assert missing.resume_alias is None
    assert missing.mode == "research"
    assert missing.scope_expectation == "allowed"
    assert missing.web_scenario_alias is not None
    assert project_model_payload(missing).mode == "research"
    denied = cases["scope_rejected"]
    assert denied.resume_alias in {source.alias for source in dataset.manifest.sources}
    assert denied.required_unit_ids == ()
    with pytest.raises(QualityDatasetError, match=r"^scope_rejected$"):
        project_model_payload(denied)


def test_every_allowed_payload_excludes_structured_gold_and_source_identity():
    dataset = load_quality_dataset(PILOT)
    for case in dataset.cases:
        if case.scope_expectation == "reject":
            continue
        payload = project_model_payload(case).model_dump()
        assert payload == {"mode": case.mode, "query": case.query}
        serialized = json.dumps(payload, ensure_ascii=False)
        excluded = (
            case.expected_behavior,
            *case.required_unit_ids,
            case.resume_alias,
            case.web_scenario_alias,
            dataset.rubric.rubric_version,
        )
        assert all(value not in serialized for value in excluded if value)


def test_injection_vectors_are_source_data_and_do_not_change_scope():
    dataset = load_quality_dataset(PILOT)
    cases = [case for case in dataset.cases if case.stratum == "prompt_injection"]
    assert {tag for case in cases for tag in case.tags} == {
        "document_injection",
        "search_injection",
    }
    units = {unit.unit_id: unit for unit in dataset.units}
    for case, attack_id in zip(cases, ("alpha_injection", "web_attack"), strict=True):
        assert attack_id in case.required_unit_ids
        attack = units[attack_id]
        assert attack.quote not in project_model_payload(case).query
        assert case.scope_expectation == "allowed"
        assert attack.source_alias in {case.resume_alias, case.web_scenario_alias}


def test_tampered_source_is_rejected_without_disclosing_content(pilot_copy):
    path = pilot_copy / "documents/resume_alpha.md"
    with path.open("a") as stream:
        stream.write("private-quality-pilot-tamper-canary")
    with pytest.raises(QualityDatasetError, match=r"^file_digest_mismatch$") as error:
        load_quality_dataset(pilot_copy)
    assert "canary" not in str(error.value)
    assert str(pilot_copy) not in str(error.value)


@pytest.mark.parametrize("unit_id", ["missing_unit", "beta_versions"])
def test_invalid_gold_is_rejected_even_with_fresh_file_digest(pilot_copy, unit_id):
    rewrite_cases(pilot_copy, lambda rows: rows[0].update(required_unit_ids=[unit_id]))
    with pytest.raises(QualityDatasetError, match=r"^invalid_quality_dataset$"):
        load_quality_dataset(pilot_copy)


def test_rejection_is_not_permission_to_include_cross_scope_gold(pilot_copy):
    rewrite_cases(pilot_copy, lambda rows: rows[-1].update(required_unit_ids=["alpha_task"]))
    with pytest.raises(QualityDatasetError, match=r"^invalid_quality_dataset$"):
        load_quality_dataset(pilot_copy)
