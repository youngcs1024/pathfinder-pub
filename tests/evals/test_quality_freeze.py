"""E4.9 deterministic data gates, not semantic quality or live measurement."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.evals.quality_dataset import (
    QualityDatasetError,
    load_quality_dataset,
    project_model_payload,
    quality_digest,
)
from tests.evals.quality_freeze import (
    FreezeError,
    _validate_leak_review,
    build_freeze,
    check_frozen,
    identity,
    main,
    read_artifact,
    record_exposure,
    scan_leakage,
    text_similarity,
    validate_coverage,
    write_artifact,
)
from tests.evals.quality_freeze_contracts import (
    CoverageV1,
    ExposureV1,
    FrozenSetV1,
    LeakReviewV1,
    ReviewsV1,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evals/datasets/quality_expanded_v1"
SHA = "c7b724bd30e85dd3cb4b21cca0c2112b92979cd3"


@pytest.fixture
def bundle(tmp_path):
    for source in DATA.rglob("*"):
        if source.is_file():
            target = tmp_path / source.relative_to(DATA)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    return tmp_path


def rewrite(root, name, change):
    path = root / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, ensure_ascii=False))


def test_expanded_dataset_freeze_is_reproducible_and_not_a_measured_baseline():
    frozen, exposures = check_frozen(DATA)
    assert (frozen.dev_count, frozen.validation_count, frozen.family_count) == (45, 15, 60)
    assert len(frozen.validation_case_ids) == 15
    assert not exposures
    assert not frozen.blind and not frozen.human_reviewed
    assert not frozen.measurement_complete and not frozen.baseline_accepted
    assert frozen == build_freeze(DATA, source_sha=SHA)
    assert (DATA / "rubric.json").read_bytes() == (
        ROOT / "tests/fixtures/quality_reviews/e47-agent-delegated-v1/rubric.json"
    ).read_bytes()
    dataset = load_quality_dataset(DATA)
    assert dataset.manifest.dataset_version == "quality-expanded-v1"
    assert len(dataset.manifest.sources) == 66
    assert len(dataset.units) == 80
    assert load_quality_dataset(ROOT / "evals/datasets/quality_v1").rubric.status == "draft"


def test_materials_unicode_ranges_and_projection_never_expose_structured_gold():
    dataset = load_quality_dataset(DATA)
    texts = {s.alias: (DATA / s.path).read_text() for s in dataset.manifest.sources}
    assert any("é" in text for text in texts.values())
    for unit in dataset.units:
        assert texts[unit.source_alias][unit.start : unit.end] == unit.quote
    for case in dataset.cases:
        if case.scope_expectation == "reject":
            with pytest.raises(QualityDatasetError, match=r"^scope_rejected$"):
                project_model_payload(case)
        else:
            payload = project_model_payload(case).model_dump()
            assert payload == {"mode": case.mode, "query": case.query}
            assert all(uid not in payload["query"] for uid in case.required_unit_ids)
            assert case.expected_behavior not in payload["query"]


@pytest.mark.parametrize("name", ["coverage.json", "reviews.json", "mapping.json", "freeze.json"])
def test_identity_tampering_cannot_pass_frozen_check(bundle, name):
    rewrite(bundle, name, lambda value: value.update(dataset_digest="sha256:" + "0" * 64))
    with pytest.raises(FreezeError):
        check_frozen(bundle)


def test_source_tampering_fails_without_disclosing_body_or_path(bundle):
    path = bundle / "documents/resume_wheel_abi.md"
    path.write_text(path.read_text() + "private-e49-canary")
    with pytest.raises(FreezeError, match=r"^invalid_freeze_inputs$") as error:
        check_frozen(bundle)
    assert "canary" not in str(error.value) and str(bundle) not in str(error.value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unreviewed", "case_digest"])
def test_incomplete_or_stale_case_review_blocks_freeze(bundle, mutation):
    def change(value):
        if mutation == "missing":
            value["cases"].pop()
        elif mutation == "duplicate":
            value["cases"].append(value["cases"][0])
        elif mutation == "unreviewed":
            value["cases"][0]["decision"] = "needs_review"
        else:
            value["cases"][0]["case_digest"] = "sha256:" + "0" * 64

    rewrite(bundle, "reviews.json", change)
    with pytest.raises(FreezeError):
        build_freeze(bundle, source_sha=SHA)


def test_review_cannot_claim_human_or_independent_authority():
    review = read_artifact(DATA, "reviews.json", ReviewsV1).model_dump(mode="json")
    review["human_reviewed"] = True
    with pytest.raises(ValidationError):
        ReviewsV1.model_validate_json(json.dumps(review))


@pytest.mark.parametrize("kind", ["family", "template", "source"])
def test_cross_split_groups_fail_even_when_ids_differ(kind):
    dataset = load_quality_dataset(DATA)
    coverage = read_artifact(DATA, "coverage.json", CoverageV1)
    rows = list(coverage.cases)
    index = next(i for i, row in enumerate(rows) if row.split == "validation")
    cases = list(dataset.cases)
    if kind == "template":
        rows[index] = rows[index].model_copy(update={"template_id": rows[0].template_id})
    elif kind == "family":
        rows[index] = rows[index].model_copy(update={"family_id": rows[0].family_id})
        cases[index] = cases[index].model_copy(update={"family_id": cases[0].family_id})
    else:
        rows[index] = rows[index].model_copy(update={"source_aliases": rows[0].source_aliases})
        cases[index] = cases[index].model_copy(update={"resume_alias": cases[0].resume_alias})
    with pytest.raises(FreezeError, match=r"^cross_split_group$"):
        validate_coverage(
            replace(dataset, cases=tuple(cases)), coverage.model_copy(update={"cases": tuple(rows)})
        )


def test_counts_are_computed_not_taken_from_declared_report():
    dataset = load_quality_dataset(DATA)
    coverage = read_artifact(DATA, "coverage.json", CoverageV1)
    cases = list(dataset.cases)
    cases[0] = cases[0].model_copy(update={"stratum": "claim_strength"})
    with pytest.raises(FreezeError, match=r"^coverage_count_mismatch$"):
        validate_coverage(replace(dataset, cases=tuple(cases)), coverage)


def test_pdf_cannot_enter_through_refreshed_manifest():
    dataset = load_quality_dataset(DATA)
    coverage = read_artifact(DATA, "coverage.json", CoverageV1)
    sources = list(dataset.manifest.sources)
    sources[0] = sources[0].model_copy(update={"path": "documents/example.pdf"})
    dataset = replace(
        dataset, manifest=dataset.manifest.model_copy(update={"sources": tuple(sources)})
    )
    with pytest.raises(FreezeError, match=r"^unsupported_material_format$"):
        validate_coverage(dataset, coverage)


def test_similarity_catches_alias_number_whitespace_and_unicode_rewrites():
    exact, similarity = text_similarity(
        "候选resume_a在\uff12\uff10\uff12\uff14年使用缓存",
        "候选resume_b在2025年使用缓存",
        ("resume_a", "resume_b"),
    )
    assert not exact and similarity == 1.0
    assert text_similarity("\uff23\uff21\uff26É\n作品", "cafe\u0301 作品", ()) == (True, 1.0)
    assert text_similarity("abcdeX", "abcdeY", ())[1] == pytest.approx(1 / 3)
    assert text_similarity("甲", "乙", ()) == (False, 0.0)


def test_changed_case_id_does_not_hide_a_cross_split_query_clone():
    dataset = load_quality_dataset(DATA)
    coverage = read_artifact(DATA, "coverage.json", CoverageV1)
    cases = list(dataset.cases)
    index = next(i for i, c in enumerate(cases) if c.split == "validation")
    cases[index] = cases[index].model_copy(update={"query": cases[0].query})
    changed = replace(dataset, cases=tuple(cases))
    report = scan_leakage(DATA, changed, coverage)
    assert any(p.kind == "query" and p.cross_split and p.exact for p in report.pairs)
    with pytest.raises(FreezeError, match=r"^leak_review_incomplete$"):
        _validate_leak_review(
            changed, coverage, report, LeakReviewV1(report_digest=identity(report), decisions=())
        )
    review = LeakReviewV1.model_validate_json(
        json.dumps(
            {
                "report_digest": identity(report),
                "decisions": [
                    {
                        "pair_id": p.pair_id,
                        "disposition": "distinct_context",
                        "reviewer_id": "codex_agent",
                        "rationale": "This attempted waiver must not permit an identical query.",
                    }
                    for p in report.pairs
                ],
            }
        )
    )
    with pytest.raises(FreezeError, match=r"^unresolved_leakage$"):
        _validate_leak_review(changed, coverage, report, review)


def test_partial_mapping_blocks_freeze(bundle):
    rewrite(bundle, "mapping-report.json", lambda value: value.update(review_complete=False))
    with pytest.raises(FreezeError, match=r"^mapping_review_incomplete$"):
        build_freeze(bundle, source_sha=SHA)


def test_mapping_review_does_not_invent_another_reviewer(bundle):
    rewrite(
        bundle,
        "mapping.json",
        lambda value: value["sources"][0]["chunks"][0].update(reviewer_id="fictional_human"),
    )
    with pytest.raises(FreezeError, match=r"^mapping_reviewer_mismatch$"):
        build_freeze(bundle, source_sha=SHA)


def test_unreviewed_similarity_alert_cannot_be_omitted(bundle):
    rewrite(
        bundle,
        "leakage-review.json",
        lambda value: value.update(report_digest="sha256:" + "0" * 64),
    )
    with pytest.raises(FreezeError, match=r"^leak_review_incomplete$"):
        build_freeze(bundle, source_sha=SHA)


def test_duplicate_source_text_is_detected_separately_from_query(bundle):
    dataset = load_quality_dataset(bundle)
    coverage = read_artifact(bundle, "coverage.json", CoverageV1)
    a = dataset.manifest.sources[0]
    b = next(s for s in dataset.manifest.sources if s.alias == "resume_oci_manifest")
    # scan_leakage's low-level comparison sees source content independently; the
    # complete freeze path additionally rejects stale manifest hashes first.
    (bundle / b.path).write_bytes((bundle / a.path).read_bytes())
    report = scan_leakage(bundle, dataset, coverage)
    assert any(p.kind == "source" and p.cross_split and p.exact for p in report.pairs)
    with pytest.raises(FreezeError):
        check_frozen(bundle)


def test_create_only_never_overwrites_existing_artifact(bundle):
    path = bundle / "freeze.json"
    before = path.read_bytes()
    value = read_artifact(bundle, "freeze.json", FrozenSetV1)
    with pytest.raises(FreezeError, match=r"^freeze_publication_failed$"):
        write_artifact(path, value)
    assert path.read_bytes() == before


def test_exposure_is_separate_create_only_and_requires_replacement(bundle):
    frozen, _ = check_frozen(bundle)
    original = (bundle / "freeze.json").read_bytes()
    record = ExposureV1(
        exposure_id="debug_case",
        freeze_digest=identity(frozen),
        case_id=frozen.validation_case_ids[0],
        execution_source_sha=SHA,
        reason="code_debugging",
    )
    record_exposure(bundle, record)
    assert check_frozen(bundle)[1] == (record,)
    assert (bundle / "freeze.json").read_bytes() == original
    with pytest.raises(FreezeError, match=r"^freeze_publication_failed$"):
        record_exposure(bundle, record)
    for update in ({"case_id": "wheel_abi"}, {"freeze_digest": "sha256:" + "0" * 64}):
        with pytest.raises(FreezeError, match=r"^exposure_identity_mismatch$"):
            record_exposure(bundle, record.model_copy(update=update))


def test_cli_check_is_offline_and_body_free(monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("offline check attempted network")

    monkeypatch.setattr("socket.socket.connect", no_network)
    monkeypatch.setattr("sys.argv", ["quality_freeze", "check", "--dataset", str(DATA)])
    assert main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dev_count"] == 45 and output["validation_count"] == 15
    assert not output["replacement_required"]
    assert set(output) == {
        "freeze_digest",
        "dev_count",
        "validation_count",
        "exposure_count",
        "replacement_required",
        "measurement_complete",
        "baseline_accepted",
    }


def test_cli_freeze_requires_explicit_review_confirmation(bundle, monkeypatch, capsys):
    before = (bundle / "freeze.json").read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["quality_freeze", "freeze", "--dataset", str(bundle), "--source-sha", SHA]
    )
    assert main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "category": "quality_freeze_failed",
    }
    assert (bundle / "freeze.json").read_bytes() == before


def test_cli_exposure_returns_nonzero_for_replacement(bundle, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "quality_freeze",
            "expose",
            "--dataset",
            str(bundle),
            "--source-sha",
            SHA,
            "--exposure-id",
            "prompt_fix",
            "--case-id",
            "oci_manifest",
            "--reason",
            "prompt_tuning",
        ],
    )
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["replacement_required"] is True
    monkeypatch.setattr("sys.argv", ["quality_freeze", "check", "--dataset", str(bundle)])
    assert main() == 2
    assert json.loads(capsys.readouterr().out)["exposure_count"] == 1


def test_symlink_artifact_is_rejected_without_following_it(bundle, tmp_path):
    # Use a new filename; no file deletion is needed to exercise publication safety.
    target = tmp_path / "outside.json"
    target.write_text("unchanged")
    link = bundle / "linked.json"
    link.symlink_to(target)
    frozen = read_artifact(bundle, "freeze.json", FrozenSetV1)
    with pytest.raises(FreezeError):
        write_artifact(link, frozen)
    assert target.read_text() == "unchanged"


def test_manifest_file_hash_matches_actual_original_material():
    dataset = load_quality_dataset(DATA)
    assert all(
        quality_digest((DATA / f.path).read_bytes()) == f.digest for f in dataset.manifest.files
    )
