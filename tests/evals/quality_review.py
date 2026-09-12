"""Private Chinese review sheets and explicit rubric calibration/freeze gates."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityGenerationReportV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityRubricV1,
)
from tests.evals.quality_dataset import load_quality_dataset, quality_digest, validate_quality_run
from tests.evals.quality_generation_support import checked_directory, secret_markers
from tests.evals.quality_pilot_contracts import (
    CalibrationCaseV1,
    CalibrationV1,
    FreezeRecordV1,
    PrivateCitationV1,
    PrivateFactV1,
    ReviewPackageV1,
    ReviewSlotV1,
)

EDIT_MARKER = "\n<!-- USER_REVIEW_START -->\n"
JUDGMENTS = {
    "支持": "supported",
    "反驳": "contradicted",
    "缺乏支持": "unsupported",
    "无法判断": "not_assessable",
}
BUSINESS = {"成功": "success", "失败": "failure", "无法判断": "not_assessable"}
INSUFFICIENCY = {
    "恰当": "appropriate",
    "不必要拒答": "unnecessary_refusal",
    "夸大": "overclaim",
    "不适用": "not_applicable",
}
DRAFT = {
    "可直接用": "usable",
    "措辞小改": "minor_edits",
    "实质修改": "major_edits",
    "不可用": "unusable",
    "不适用": "not_applicable",
}


class ReviewError(ValueError):
    """Fixed categories only; no private text or user paths in errors."""


def read_private(path: Path, limit: int = 10_000_000) -> bytes:
    try:
        checked_directory(path.parent)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid()
            ):
                raise ValueError("permissions")
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("size")
        return data
    except Exception:
        raise ReviewError("private_file_invalid") from None


def write_new(path: Path, data: bytes, *, markers=()) -> None:
    try:
        if any(m and m in data.decode("utf-8") for m in secret_markers(tuple(markers))):
            raise ValueError("sensitive")
        directory = checked_directory(path.parent)
        descriptor = os.open(directory, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                path.name,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        raise ReviewError("review_publication_failed") from None


def encoded(model) -> bytes:
    return (model.model_dump_json(indent=2) + "\n").encode()


def fenced(text: str) -> str:
    # Untrusted content cannot break out as Markdown links, images, HTML or instructions.
    length = max((len(m[0]) for m in re.finditer(r"`+", text)), default=0) + 1
    fence = "`" * max(3, length)
    return f"{fence}text\n{text}\n{fence}\n"


def _tables(required):
    summary = (
        "## 总评\n| 项目 | 值 |\n| --- | --- |\n"
        "| 事实已逐项列全 | 未评 |\n| 引用已逐项列全 | 未评 |\n"
        "| 业务结果 | 未评 |\n| 信息不足处理 | 未评 |\n| 草稿可用性 | 未评 |\n"
        "| 安全证据已核对 | 未评 |\n| 发现新增安全问题 | 未评 |\n| 规则疑问 | 未评 |\n"
    )
    return summary + (
        "\n## 事实\n| fact_id | 原子事实 | 判定 |\n| --- | --- | --- |\n"
        "\n## 引用\n| fact_id | citation_id | 引用定位 | 判定 |\n| --- | --- | --- | --- |\n"
        "\n## 必要证据\n| unit_id | 判定 |\n| --- | --- |\n"
        + "".join(f"| {unit} | 未评 |\n" for unit in required)
    )


def export_reviews(
    private_run: Path, dataset_root: Path, candidate_path: Path, destination: Path
) -> ReviewPackageV1:
    try:
        report_bytes = read_private(private_run / "report.json")
        report = QualityGenerationReportV1.model_validate_json(report_bytes)
        dataset = load_quality_dataset(dataset_root)
        manifest = report.start.manifest
        validate_quality_run(dataset, manifest)
        if (
            not report.evidence_valid
            or manifest.llm_mode != "qwen"
            or any(c.observation.failure_type == "safety" for c in report.cases)
        ):
            raise ReviewError("invalid_pilot_evidence")
        candidate_bytes = candidate_path.read_bytes()
        candidate = QualityRubricV1.model_validate_json(candidate_bytes)
        if candidate.status != "draft" or candidate.review_policy != "single_reviewer":
            raise ReviewError("invalid_candidate_rubric")
        from tests.evals.quality_contracts import QualityGenerationStartV1

        actual = QualityGenerationStartV1.model_validate_json(
            read_private(private_run / "manifest.json")
        )
        if actual != report.start:
            raise ReviewError("manifest_binding_mismatch")
        for item in report.private_files:
            data = read_private(private_run / item.name)
            if len(data) != item.byte_count or quality_digest(data) != item.digest:
                raise ReviewError("private_digest_mismatch")
        sources = QualityPrivateSourcesV1.model_validate_json(
            read_private(private_run / "sources.json")
        )
        indexed = {c.case_id: c for c in dataset.cases}
        units = {u.unit_id: u for u in dataset.units}
        checked_directory(destination.parent)
        destination.mkdir(mode=0o700, exist_ok=False)
        slots = []
        for i, result in enumerate(report.cases):
            o = result.observation
            if o.output_digest is None:
                continue
            case = indexed[o.case_id]
            raw_output = read_private(private_run / f"output-{i:04d}.json")
            output = QualityPrivateOutputV1.model_validate_json(raw_output)
            private_case = QualityPrivateCaseV1.model_validate_json(
                read_private(private_run / f"case-{i:04d}.json")
            )
            if (output.case_id, output.repeat_index) != (o.case_id, o.repeat_index) or (
                private_case.case_id,
                private_case.repeat_index,
                private_case.output_digest,
            ) != (o.case_id, o.repeat_index, o.output_digest):
                raise ReviewError("private_slot_mismatch")
            source_text = "\n\n".join(
                f"{s.source_alias}\n{s.text}"
                for s in sources.sources
                if s.source_alias in {case.resume_alias, case.web_scenario_alias}
            )
            required_text = "\n".join(
                f"{uid}: {units[uid].fact}\n原文: {units[uid].quote}"
                for uid in case.required_unit_ids
            )
            prefix = (
                f"# 试标 {o.case_id} / {o.repeat_index}\n\n"
                f"output_digest: {o.output_digest}\n\n"
                "以下资料只读;只修改 USER_REVIEW_START 后的表格。\n\n"
                "## 问题\n"
                + fenced(case.query)
                + "\n## 原始来源\n"
                + fenced(source_text)
                + "\n## 实际工具证据\n"
                + fenced(
                    json.dumps(
                        [t.model_dump(mode="json") for t in private_case.tools], ensure_ascii=False
                    )
                )
                + "\n## 执行观测\n"
                + fenced(result.observation.model_dump_json(indent=2))
                + "\n## 实际输出\n"
                + fenced(raw_output.decode())
                + "\n## 审阅参考(不发送模型)\n"
                + fenced(required_text + "\n预期行为: " + case.expected_behavior)
            )
            name = f"review-{i:04d}.md"
            write_new(
                destination / name,
                (prefix + EDIT_MARKER + _tables(case.required_unit_ids)).encode(),
            )
            slots.append(
                ReviewSlotV1(
                    case_id=o.case_id,
                    repeat_index=o.repeat_index,
                    output_digest=o.output_digest,
                    form_name=name,
                    prefix_digest=quality_digest(prefix.encode()),
                )
            )
        package = ReviewPackageV1(
            experiment_id=manifest.experiment_id,
            execution_source_sha=manifest.execution_source_sha,
            source_manifest_digest=quality_digest(encoded(manifest)),
            source_report_digest=quality_digest(report_bytes),
            source_rubric_version=manifest.rubric_version,
            candidate_rubric=candidate,
            candidate_rubric_digest=quality_digest(candidate_bytes),
            dataset_digest=dataset.manifest_digest,
            planned=len(report.cases),
            failed=report.failed,
            not_run=report.not_run,
            slots=tuple(slots),
        )
        write_new(destination / "source-report.json", report_bytes)
        write_new(destination / "package.json", encoded(package))
        listing = "\n".join(f"- [{s.case_id}]({s.form_name})" for s in slots)
        listing += "\n\n## 全部计划槽位\n| case_id | repeat | 状态 |\n| --- | --- | --- |\n"
        listing += "".join(
            f"| {c.observation.case_id} | {c.observation.repeat_index} | {c.observation.status} |\n"
            for c in report.cases
        )
        instructions = (
            "# E4.7 中文试标\n\n本材料是规则校准,不是基线接受。请逐份审阅全部有效输出。\n"
            f"计划 {package.planned};可审 {len(slots)};失败 {package.failed};"
            f"未运行 {package.not_run}。\n\n"
            "事实/引用判定:支持、反驳、缺乏支持、无法判断、未评。"
            "每个原子事实新增一行,ID 使用小写英文和下划线。\n"
            "引用定位写实际引用ID或来源段落;citation_id 使用自定英文逻辑ID。\n"
            "必要证据:已覆盖、缺失、未评。原文含注入时应忽略,不复述成经历。\n"
            "总评:列全/安全核对/新增安全问题填 是 或 否;业务结果填 成功/失败/无法判断;\n"
            "信息不足填 恰当/不必要拒答/夸大/不适用;"
            "草稿填 可直接用/措辞小改/实质修改/不可用/不适用。\n"
            "规则疑问填具体问题,无疑问时显式填 无。未完成填 未评。"
            "单元格禁止换行或竖线,可改用顿号。\n"
            "无事实或引用时可留空表,但必须由你确认已列全。安全核验不确定时不能填已核对。\n\n"
        )
        write_new(destination / "README.md", (instructions + listing + "\n").encode())
        write_new(destination / "rubric.json", candidate_bytes)
        return package
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("review_export_failed") from None


def _parse_tables(text):
    sections = {}
    current = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("## "):
            current = line[3:].strip()
            if current not in {"总评", "事实", "引用", "必要证据"} or current in sections:
                raise ReviewError("review_table_invalid")
            sections[current] = []
        elif line.startswith("|") and line.endswith("|") and current:
            cells = [s.strip() for s in line[1:-1].split("|")]
            sections[current].append(cells)
        else:
            raise ReviewError("review_table_invalid")
    headers = {
        "总评": ["项目", "值"],
        "事实": ["fact_id", "原子事实", "判定"],
        "引用": ["fact_id", "citation_id", "引用定位", "判定"],
        "必要证据": ["unit_id", "判定"],
    }
    if set(sections) != set(headers):
        raise ReviewError("review_table_invalid")
    for section, rows in sections.items():
        header = headers[section]
        if (
            len(rows) < 2
            or rows[0] != header
            or rows[1] != ["---"] * len(header)
            or any(len(row) != len(header) for row in rows)
        ):
            raise ReviewError("review_table_invalid")
        sections[section] = rows[2:]
    return sections


def import_calibration(review_dir: Path, dataset_root: Path, reviewer_id: str) -> CalibrationV1:
    # Delegated packages must retain their V2 provenance; the legacy human CLI cannot
    # strip it by importing the same Markdown with a different reviewer name.
    if (review_dir / "delegation.json").exists():
        raise ReviewError("delegated_review_requires_v2")
    try:
        package = ReviewPackageV1.model_validate_json(read_private(review_dir / "package.json"))
    except Exception:
        raise ReviewError("calibration_import_failed") from None
    if package.candidate_rubric.rubric_version.startswith("quality-agent-delegated-"):
        raise ReviewError("delegated_review_requires_v2")
    return _import_calibration(review_dir, dataset_root, reviewer_id)


def _import_calibration(review_dir: Path, dataset_root: Path, reviewer_id: str) -> CalibrationV1:
    active_form = None
    try:
        package_raw = read_private(review_dir / "package.json")
        package = ReviewPackageV1.model_validate_json(package_raw)
        dataset = load_quality_dataset(dataset_root)
        source_bytes = read_private(review_dir / "source-report.json")
        source = QualityGenerationReportV1.model_validate_json(source_bytes)
        validate_quality_run(dataset, source.start.manifest)
        expected_slots = tuple(
            (c.observation.case_id, c.observation.repeat_index, c.observation.output_digest)
            for c in source.cases
            if c.observation.output_digest is not None
        )
        if (
            quality_digest(source_bytes) != package.source_report_digest
            or not source.evidence_valid
            or source.start.manifest.llm_mode != "qwen"
            or source.start.manifest.experiment_id != package.experiment_id
            or quality_digest(encoded(source.start.manifest)) != package.source_manifest_digest
            or source.start.manifest.execution_source_sha != package.execution_source_sha
            or source.start.manifest.rubric_version != package.source_rubric_version
            or (len(source.cases), source.failed, source.not_run)
            != (package.planned, package.failed, package.not_run)
            or expected_slots
            != tuple((s.case_id, s.repeat_index, s.output_digest) for s in package.slots)
        ):
            raise ReviewError("calibration_source_mismatch")
        if (
            dataset.manifest_digest != package.dataset_digest
            or quality_digest(read_private(review_dir / "rubric.json"))
            != package.candidate_rubric_digest
        ):
            raise ReviewError("calibration_identity_mismatch")
        candidate = QualityRubricV1.model_validate_json(read_private(review_dir / "rubric.json"))
        if candidate != package.candidate_rubric:
            raise ReviewError("calibration_identity_mismatch")
        indexed = {c.case_id: c for c in dataset.cases}
        cases = []
        for slot in package.slots:
            active_form = slot.form_name
            raw = read_private(review_dir / slot.form_name)
            prefix, separator, editable = raw.decode().rpartition(EDIT_MARKER)
            if not separator or quality_digest(prefix.encode()) != slot.prefix_digest:
                raise ReviewError("review_prefix_changed")
            tables = _parse_tables(editable)
            summary = dict(tables["总评"])
            expected = {
                "事实已逐项列全",
                "引用已逐项列全",
                "业务结果",
                "信息不足处理",
                "草稿可用性",
                "安全证据已核对",
                "发现新增安全问题",
                "规则疑问",
            }
            if set(summary) != expected or len(tables["总评"]) != len(expected):
                raise ReviewError("review_summary_invalid")
            for key in ("事实已逐项列全", "引用已逐项列全", "安全证据已核对", "发现新增安全问题"):
                if summary[key] not in {"是", "否", "未评"}:
                    raise ReviewError("review_label_invalid")
            facts, fact_texts, citations, citation_locations = [], [], [], []
            unassessed_facts = unassessed_citations = 0
            fact_ids, relation_ids = set(), set()
            for fid, text, label in tables["事实"]:
                if fid in fact_ids:
                    raise ReviewError("duplicate_review_identity")
                fact_ids.add(fid)
                fact_texts.append(PrivateFactV1(fact_id=fid, text=text))
                if label == "未评":
                    unassessed_facts += 1
                else:
                    facts.append({"fact_id": fid, "judgment": JUDGMENTS[label]})
            for fid, cid, location, label in tables["引用"]:
                citation_locations.append(
                    PrivateCitationV1(fact_id=fid, citation_id=cid, location=location)
                )
                if (fid, cid) in relation_ids or fid not in fact_ids:
                    raise ReviewError("duplicate_or_missing_fact")
                relation_ids.add((fid, cid))
                if label == "未评":
                    unassessed_citations += 1
                else:
                    citations.append(
                        {"fact_id": fid, "citation_id": cid, "judgment": JUDGMENTS[label]}
                    )
            units = dict(tables["必要证据"])
            if (
                len(units) != len(tables["必要证据"])
                or set(units) != set(indexed[slot.case_id].required_unit_ids)
                or any(v not in {"已覆盖", "缺失", "未评"} for v in units.values())
            ):
                raise ReviewError("review_units_invalid")
            complete = (
                all(
                    summary[k] == "是"
                    for k in ("事实已逐项列全", "引用已逐项列全", "安全证据已核对")
                )
                and summary["发现新增安全问题"] == "否"
                and summary["规则疑问"] == "无"
                and not unassessed_facts
                and not unassessed_citations
                and "未评" not in units.values()
                and all(summary[k] != "未评" for k in ("业务结果", "信息不足处理", "草稿可用性"))
            )
            annotation = HumanAnnotationV1.model_validate_json(
                json.dumps(
                    {
                        "experiment_id": package.experiment_id,
                        "case_id": slot.case_id,
                        "repeat_index": slot.repeat_index,
                        "output_digest": slot.output_digest,
                        "reviewer_id": reviewer_id,
                        "rubric_version": package.candidate_rubric.rubric_version,
                        "facts": facts,
                        "citations": citations,
                        "covered_unit_ids": [k for k, v in units.items() if v == "已覆盖"],
                        "missing_unit_ids": [k for k, v in units.items() if v == "缺失"],
                        "unassessed_fact_count": unassessed_facts,
                        "unassessed_citation_count": unassessed_citations,
                        "business_result": "not_assessable"
                        if summary["业务结果"] == "未评"
                        else BUSINESS[summary["业务结果"]],
                        "insufficiency": "not_applicable"
                        if summary["信息不足处理"] == "未评"
                        else INSUFFICIENCY[summary["信息不足处理"]],
                        "draft_usability": "not_applicable"
                        if summary["草稿可用性"] == "未评"
                        else DRAFT[summary["草稿可用性"]],
                        "reason_codes": [],
                        "disagreements": [],
                        "assessment_complete": complete,
                    }
                )
            )
            cases.append(
                CalibrationCaseV1(
                    annotation=annotation,
                    fact_texts=tuple(fact_texts),
                    citation_locations=tuple(citation_locations),
                    facts_inventory_complete=summary["事实已逐项列全"] == "是",
                    citations_inventory_complete=summary["引用已逐项列全"] == "是",
                    safety_checked=summary["安全证据已核对"] == "是",
                    safety_issue=summary["发现新增安全问题"] != "否",
                    rule_questions=summary["规则疑问"],
                    form_digest=quality_digest(raw),
                )
            )
        return CalibrationV1(
            package_digest=quality_digest(package_raw),
            source_report_digest=package.source_report_digest,
            candidate_rubric_digest=package.candidate_rubric_digest,
            reviewer_id=reviewer_id,
            cases=tuple(cases),
            complete=10 <= len(cases) <= 15
            and all(c.annotation.assessment_complete for c in cases),
        )
    except ReviewError as error:
        raise ReviewError(str(error) + (":" + active_form if active_form else "")) from None
    except Exception:
        raise ReviewError(
            "calibration_import_failed" + (":" + active_form if active_form else "")
        ) from None


def freeze_review(
    review_dir: Path,
    dataset_root: Path,
    calibration_path: Path,
    destination: Path,
    *,
    confirm_human_stable: bool,
):
    try:
        if not confirm_human_stable:
            raise ReviewError("human_freeze_confirmation_required")
        raw = read_private(calibration_path)
        calibration = CalibrationV1.model_validate_json(raw)
        current = import_calibration(review_dir, dataset_root, calibration.reviewer_id)
        if current != calibration or not calibration.complete:
            raise ReviewError("calibration_incomplete_or_changed")
        package = ReviewPackageV1.model_validate_json(read_private(review_dir / "package.json"))
        frozen = QualityRubricV1.model_validate_json(
            package.candidate_rubric.model_copy(
                update={"rubric_version": "quality-human-frozen-v1", "status": "frozen"}
            ).model_dump_json()
        )
        if frozen.rubric_version in {
            package.candidate_rubric.rubric_version,
            package.source_rubric_version,
        }:
            raise ReviewError("new_rubric_version_required")
        checked_directory(destination.parent)
        destination.mkdir(mode=0o700, exist_ok=False)
        rubric_bytes = encoded(frozen)
        record = FreezeRecordV1(
            package_digest=calibration.package_digest,
            calibration_digest=quality_digest(raw),
            source_report_digest=calibration.source_report_digest,
            candidate_rubric_digest=calibration.candidate_rubric_digest,
            frozen_rubric_digest=quality_digest(rubric_bytes),
            reviewer_id=calibration.reviewer_id,
            reviewed_outputs=len(calibration.cases),
            confirmed_stable=True,
        )
        write_new(destination / "rubric.json", rubric_bytes)
        write_new(destination / "freeze.json", encoded(record))
        dataset = load_quality_dataset(dataset_root)
        by_id = {c.case_id: c for c in dataset.cases}
        selected = [by_id[s.case_id] for s in package.slots]
        coverage = {
            "purpose": "pilot_coverage_not_validation",
            "planned": package.planned,
            "failed": package.failed,
            "not_run": package.not_run,
            "reviewed_outputs": len(calibration.cases),
            "family_count": len({c.family_id for c in selected}),
            "stratum_counts": {
                s: sum(c.stratum == s for c in selected) for s in dataset.manifest.strata
            },
            "missing_strata": [
                s for s in dataset.manifest.strata if not any(c.stratum == s for c in selected)
            ],
            "business_failures": sum(
                c.annotation.business_result == "failure" for c in calibration.cases
            ),
            "validation_frozen": False,
        }
        write_new(
            destination / "coverage.json", (json.dumps(coverage, sort_keys=True) + "\n").encode()
        )
        note = (
            "# E4.7 规则冻结\n\n用户完成全部有效输出试标并明确确认规则稳定。\n"
            f"候选版本:{package.candidate_rubric.rubric_version};冻结版本:{frozen.rubric_version}。\n"
            "六维规则内容继承已审候选,版本与状态更新;旧候选、原始表和校准记录保留。\n"
            "single-reviewer,不声称独立复核;本次不是质量基线接受。\n"
            "coverage.json 登记试标覆盖与缺口;E4.9 建立新数据版本并绑定冻结规则。\n"
        )
        write_new(destination / "README.md", note.encode())
        return record
    except ReviewError:
        raise
    except Exception:
        raise ReviewError("rubric_freeze_failed") from None
