from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    project_model_payload,
    quality_digest,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/evals/quality_contract_v1"


@pytest.fixture
def sample(tmp_path):
    for source in FIXTURE.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    return tmp_path


def change_jsonl(root: Path, name: str, change):
    path = root / name
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    refresh_digest(root, name)


def refresh_digest(root: Path, name: str):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    for entry in manifest["files"]:
        if entry["path"] == name:
            entry["digest"] = quality_digest((root / name).read_bytes())
    path.write_text(json.dumps(manifest))


def test_projection_has_only_model_input_not_gold_or_source_aliases():
    dataset = load_quality_dataset(FIXTURE)
    case = dataset.cases[0]
    payload = project_model_payload(case).model_dump()
    assert payload == {"mode": case.mode, "query": case.query}
    for gold in (case.expected_behavior, *case.required_unit_ids, case.resume_alias):
        assert gold not in json.dumps(payload)
    with pytest.raises(QualityDatasetError, match="scope_rejected"):
        project_model_payload(dataset.cases[2])


@pytest.mark.parametrize("name", ["cases.jsonl", "source_units.jsonl"])
def test_duplicate_ids_fail_even_with_updated_digest(sample, name):
    change_jsonl(sample, name, lambda rows: rows.append(rows[0].copy()))
    with pytest.raises(QualityDatasetError, match="invalid_quality_dataset"):
        load_quality_dataset(sample)


@pytest.mark.parametrize(
    "field,value",
    [
        ("resume_alias", "missing_alias"),
        ("web_scenario_alias", "resume_alpha"),
        ("required_unit_ids", ["missing_unit"]),
        ("split", "test"),
        ("split", "validation"),
        ("stratum", "missing_stratum"),
        ("family_id", "missing_family"),
    ],
)
def test_missing_references_and_invalid_splits_fail(sample, field, value):
    change_jsonl(sample, "cases.jsonl", lambda rows: rows[0].update({field: value}))
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


def test_same_family_cannot_cross_splits(sample):
    change_jsonl(sample, "cases.jsonl", lambda rows: rows[2].update(split="validation"))
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


@pytest.mark.parametrize("expectation", ["allowed", "reject"])
def test_cross_scope_gold_fails_even_for_expected_rejection(sample, expectation):
    change_jsonl(
        sample,
        "cases.jsonl",
        lambda rows: rows[1].update(required_unit_ids=["course_pg"], scope_expectation=expectation),
    )
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", 0),
        ("end", 999),
        ("quote", "错误引文"),
        ("source_alias", "missing_alias"),
        ("source_kind", "web"),
        ("normalized_text_digest", "sha256:" + "b" * 64),
        ("alternative_unit_ids", ["missing_unit"]),
        ("alternative_unit_ids", ["role_pg"]),
    ],
)
def test_bad_evidence_ranges_digests_and_alternatives_fail(sample, field, value):
    change_jsonl(sample, "source_units.jsonl", lambda rows: rows[0].update({field: value}))
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


def test_changed_file_bytes_are_rejected_before_parsing(sample):
    with (sample / "cases.jsonl").open("a") as stream:
        stream.write("private-content-canary")
    with pytest.raises(QualityDatasetError, match="file_digest_mismatch") as caught:
        load_quality_dataset(sample)
    assert "private-content-canary" not in str(caught.value)


def test_parse_errors_do_not_expose_body_or_private_path(sample):
    path = sample / "cases.jsonl"
    path.write_text('{"query": "private-content-canary"}\n')
    refresh_digest(sample, "cases.jsonl")
    with pytest.raises(QualityDatasetError) as caught:
        load_quality_dataset(sample)
    assert str(caught.value) == "invalid_quality_dataset"
    assert "private-content-canary" not in str(caught.value)
    assert str(sample) not in str(caught.value)


def test_manifest_cannot_escape_root(sample):
    path = sample / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = "../private.jsonl"
    path.write_text(json.dumps(manifest))
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


def test_symlink_file_is_rejected_without_reading_target(sample):
    # Use a new path: no file deletion is needed for this negative case.
    link = sample / "linked.jsonl"
    link.symlink_to(sample / "cases.jsonl")
    path = sample / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][0]["path"] = link.name
    path.write_text(json.dumps(manifest))
    with pytest.raises(QualityDatasetError, match="symlink_not_allowed"):
        load_quality_dataset(sample)


def test_evidence_offsets_are_unicode_not_utf8_bytes():
    dataset = load_quality_dataset(FIXTURE)
    text = (FIXTURE / "resume.md").read_text()
    unit = dataset.units[0]
    assert text[unit.start : unit.end] == unit.quote
    assert unit.start != len(text[: unit.start].encode("utf-8"))
    assert unit.end - unit.start == len(unit.quote)


def test_rubric_cannot_declare_labels_the_annotation_schema_cannot_read(sample):
    path = sample / "rubric.json"
    rubric = json.loads(path.read_text())
    rubric["dimensions"][0]["labels"] = ["always_correct"]
    path.write_text(json.dumps(rubric))
    refresh_digest(sample, "rubric.json")
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)


def test_valid_same_source_alternative_and_cross_scope_alternative(sample):
    def add_alternative(rows):
        alternative = {**rows[0], "unit_id": "course_pg_alternative"}
        rows.append(alternative)
        rows[0]["alternative_unit_ids"] = ["course_pg_alternative"]

    change_jsonl(sample, "source_units.jsonl", add_alternative)
    assert load_quality_dataset(sample).units[0].alternative_unit_ids == ("course_pg_alternative",)
    # A second declared resume remains outside the case's single-resume scope.
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"].append({**manifest["sources"][0], "alias": "resume_other"})
    manifest_path.write_text(json.dumps(manifest))
    change_jsonl(
        sample, "source_units.jsonl", lambda rows: rows[2].update(source_alias="resume_other")
    )
    with pytest.raises(QualityDatasetError):
        load_quality_dataset(sample)
