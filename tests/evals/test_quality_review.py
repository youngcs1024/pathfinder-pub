"""Synthetic private bundle tests, never actual human or paid measurement evidence."""

import json
from pathlib import Path

import pytest

from tests.evals.quality_contracts import (
    QualityGenerationReportV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityPrivateSourceV1,
    QualityRubricV1,
)
from tests.evals.quality_dataset import (
    QualityDatasetError,
    quality_digest,
    validate_quality_annotations,
)
from tests.evals.quality_review import (
    ReviewError,
    encoded,
    export_reviews,
    freeze_review,
    import_calibration,
    read_private,
    write_new,
)
from tests.evals.test_quality_score import FIXTURE, inputs, parse

CANDIDATE = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/quality_reviews/e47-rubric-v1/rubric.json"
)


def private_file(path, data):
    path.write_bytes(data)
    path.chmod(0o600)


def bundle(tmp_path, count=10):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    run = root / "run"
    run.mkdir(mode=0o700)
    dataset, report, _ = inputs(("succeeded",) * count)
    payload = report.model_dump(mode="json")
    payload["start"]["manifest"].update(llm_mode="qwen", measurement_scope="generation")
    sources = []
    for source in dataset.manifest.sources:
        raw = (FIXTURE / source.path).read_text()
        sources.append(
            QualityPrivateSourceV1(
                source_alias=source.alias,
                kind=source.kind,
                text=raw,
                digest=quality_digest(raw.encode()),
            )
        )
    files = {}
    files["sources.json"] = encoded(QualityPrivateSourcesV1(sources=tuple(sources)))
    for i, case in enumerate(payload["cases"]):
        observation = case["observation"]
        observation["measurement_scope"] = "generation"
        output = QualityPrivateOutputV1(
            case_id="claim_strength",
            repeat_index=i,
            output={
                "text": "fixture only `code` <script>bad()</script> [link](https://invalid.test)"
            },
        )
        raw = encoded(output)
        files[f"output-{i:04d}.json"] = raw
        observation["output_digest"] = quality_digest(raw)
        private_case = QualityPrivateCaseV1(
            case_id="claim_strength",
            repeat_index=i,
            input={"mode": "application", "query": "fixture only"},
            resume_alias="resume_alpha",
            web_scenario_alias="role_backend",
            tools=(),
            output_digest=quality_digest(raw),
            failure_type=None,
        )
        files[f"case-{i:04d}.json"] = encoded(private_case)
        case["private_case_digest"] = quality_digest(encoded(private_case))
    payload["private_files"] = [
        {"name": n, "digest": quality_digest(b), "byte_count": len(b)} for n, b in files.items()
    ]
    report = parse(QualityGenerationReportV1, payload)
    files["manifest.json"] = encoded(report.start)
    files["report.json"] = encoded(report)
    for name, raw in files.items():
        private_file(run / name, raw)
    return root, run, report


def exported(tmp_path, count=10):
    root, run, report = bundle(tmp_path, count)
    dest = root / "reviews"
    package = export_reviews(run, FIXTURE, CANDIDATE, dest)
    return root, run, dest, package, report


def fill(dest, package):
    # A deterministic test oracle only; production exporter leaves all cells unassessed.
    for slot in package.slots:
        path = dest / slot.form_name
        text = path.read_text()
        for key in ("事实已逐项列全", "引用已逐项列全", "安全证据已核对"):
            text = text.replace(f"| {key} | 未评 |", f"| {key} | 是 |")
        for key, value in (
            ("发现新增安全问题", "否"),
            ("业务结果", "成功"),
            ("信息不足处理", "恰当"),
            ("草稿可用性", "可直接用"),
            ("规则疑问", "无"),
        ):
            text = text.replace(f"| {key} | 未评 |", f"| {key} | {value} |")
        text = text.replace("| course_pg | 未评 |", "| course_pg | 已覆盖 |").replace(
            "| role_pg | 未评 |", "| role_pg | 已覆盖 |"
        )
        text = text.replace(
            "| fact_id | 原子事实 | 判定 |\n| --- | --- | --- |\n",
            "| fact_id | 原子事实 | 判定 |\n| --- | --- | --- |\n| fact_one | 测试事实 | 支持 |\n",
        )
        text = text.replace(
            "| fact_id | citation_id | 引用定位 | 判定 |\n| --- | --- | --- | --- |\n",
            "| fact_id | citation_id | 引用定位 | 判定 |\n| --- | --- | --- | --- |\n"
            "| fact_one | source_one | 来源第一段 | 支持 |\n",
        )
        path.write_text(text)


def test_export_blank_and_untrusted_text_fenced(tmp_path):
    _, run, dest, package, _ = exported(tmp_path)
    assert len(package.slots) == 10
    assert dest.stat().st_mode & 0o777 == 0o700
    text = (dest / package.slots[0].form_name).read_text()
    assert "| 业务结果 | 未评 |" in text
    assert "```text" in text and "<script>" in text
    result = import_calibration(dest, FIXTURE, "human_one")
    assert not result.complete
    assert all(not c.annotation.assessment_complete for c in result.cases)
    assert package.source_rubric_version != package.candidate_rubric.rubric_version
    with pytest.raises(ReviewError):
        export_reviews(run, FIXTURE, CANDIDATE, dest)


def test_complete_calibration_remains_incompatible_with_old_rubric(tmp_path):
    _, _, dest, package, report = exported(tmp_path)
    fill(dest, package)
    result = import_calibration(dest, FIXTURE, "human_one")
    assert result.complete and len(result.cases) == 10
    assert all(c.annotation.reviewer_id == "human_one" for c in result.cases)
    from tests.evals.quality_dataset import load_quality_dataset

    with pytest.raises(QualityDatasetError, match="annotation_identity_mismatch"):
        validate_quality_annotations(
            load_quality_dataset(FIXTURE),
            report.start.manifest,
            tuple(c.observation for c in report.cases),
            tuple(c.annotation for c in result.cases),
        )


def test_freeze_requires_actual_confirmation_and_immutable_snapshot(tmp_path):
    root, _, dest, package, _ = exported(tmp_path)
    fill(dest, package)
    calibration = import_calibration(dest, FIXTURE, "human_one")
    snapshot = root / "calibration.json"
    write_new(snapshot, encoded(calibration))
    with pytest.raises(ReviewError, match="human_freeze_confirmation_required"):
        freeze_review(dest, FIXTURE, snapshot, root / "frozen", confirm_human_stable=False)
    record = freeze_review(dest, FIXTURE, snapshot, root / "frozen", confirm_human_stable=True)
    assert record.reviewed_outputs == 10 and not record.baseline_accepted
    rubric = QualityRubricV1.model_validate_json(read_private(root / "frozen/rubric.json"))
    assert (
        rubric.status == "frozen"
        and rubric.rubric_version != package.candidate_rubric.rubric_version
    )
    assert "测试事实" not in (root / "frozen/freeze.json").read_text()
    assert json.loads((root / "frozen/coverage.json").read_text())["validation_frozen"] is False
    with pytest.raises(ReviewError):
        freeze_review(dest, FIXTURE, snapshot, root / "frozen", confirm_human_stable=True)


@pytest.mark.parametrize("count", [1, 9, 16])
def test_outside_ten_to_fifteen_cannot_freeze(tmp_path, count):
    root, _, dest, package, _ = exported(tmp_path, count)
    fill(dest, package)
    result = import_calibration(dest, FIXTURE, "human_one")
    assert not result.complete
    snapshot = root / "calibration.json"
    write_new(snapshot, encoded(result))
    with pytest.raises(ReviewError, match="calibration_incomplete_or_changed"):
        freeze_review(dest, FIXTURE, snapshot, root / "frozen", confirm_human_stable=True)


@pytest.mark.parametrize(
    "before,after",
    [
        ("| 规则疑问 | 无 |", "| 规则疑问 | 仍有歧义 |"),
        ("| 发现新增安全问题 | 否 |", "| 发现新增安全问题 | 是 |"),
        ("| 事实已逐项列全 | 是 |", "| 事实已逐项列全 | 未评 |"),
        ("| role_pg | 已覆盖 |", "| role_pg | 未评 |"),
    ],
)
def test_partial_or_safety_or_questions_block_freeze(tmp_path, before, after):
    _, _, dest, package, _ = exported(tmp_path)
    fill(dest, package)
    path = dest / package.slots[0].form_name
    path.write_text(path.read_text().replace(before, after))
    assert not import_calibration(dest, FIXTURE, "human_one").complete


@pytest.mark.parametrize(
    "change", ["prefix", "duplicate", "foreign_unit", "unknown_label", "malformed"]
)
def test_bad_form_errors_have_only_safe_location(tmp_path, change):
    _, _, dest, package, _ = exported(tmp_path)
    fill(dest, package)
    path = dest / package.slots[0].form_name
    text = path.read_text()
    edits = {
        "prefix": ("fixture only", "private-secret-marker"),
        "duplicate": (
            "| fact_one | 测试事实 | 支持 |",
            "| fact_one | 测试事实 | 支持 |\n| fact_one | duplicate | 支持 |",
        ),
        "foreign_unit": ("| role_pg |", "| foreign |"),
        "unknown_label": ("| 测试事实 | 支持 |", "| 测试事实 | private-secret-marker |"),
        "malformed": ("## 总评", "## unknown"),
    }
    text = text.replace(*edits[change])
    path.write_text(text)
    with pytest.raises(ReviewError) as error:
        import_calibration(dest, FIXTURE, "human_one")
    assert "private-secret-marker" not in str(error.value)
    assert str(error.value).endswith(":review-0000.md")


def test_changed_source_bytes_and_changed_calibration_rejected(tmp_path):
    root, run, dest, package, _ = exported(tmp_path)
    fill(dest, package)
    snapshot = root / "calibration.json"
    write_new(snapshot, encoded(import_calibration(dest, FIXTURE, "human_one")))
    path = dest / package.slots[0].form_name
    path.write_text(path.read_text().replace("测试事实", "已改事实"))
    with pytest.raises(ReviewError, match="calibration_incomplete_or_changed"):
        freeze_review(dest, FIXTURE, snapshot, root / "frozen", confirm_human_stable=True)
    path = run / "output-0000.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReviewError, match="private_digest_mismatch"):
        export_reviews(run, FIXTURE, CANDIDATE, root / "another")


def test_private_permissions_symlink_and_secret_publication(tmp_path, monkeypatch):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "file"
    path.write_text("value")
    path.chmod(0o644)
    with pytest.raises(ReviewError, match="private_file_invalid"):
        read_private(path)
    link = root / "link"
    link.symlink_to(path)
    with pytest.raises(ReviewError):
        read_private(link)
    monkeypatch.setenv("E47_TEST_SECRET", "private-secret-marker")
    with pytest.raises(ReviewError, match="review_publication_failed"):
        write_new(root / "leak", b"private-secret-marker")
    assert not (root / "leak").exists()
