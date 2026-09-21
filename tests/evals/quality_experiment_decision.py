"""Frozen E7-A.7 paired rules and digest-bound Agent review; never baseline acceptance."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import Field

from app.domain.research import ResearchOutputV2
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.quality_compare import compare_quality
from tests.evals.quality_contracts import (
    HumanAnnotationV1,
    QualityGenerationReportV1,
    QualityGenerationStartV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
)
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
)
from tests.evals.quality_e7a_contracts import E7AResearchOutputV1
from tests.evals.quality_experiment import DATASET, load_experiment_plan
from tests.evals.quality_experiment_binding import (
    PLAN,
    ROOT,
    CIProofV1,
    ci_proof,
    command,
    encoded,
    fail,
    git,
    harness_inventory,
    no_links,
    read_binding,
    read_json,
    source_identity,
    verify_binding,
    verify_imports,
    write_new,
)
from tests.evals.quality_experiment_execution import (
    ARMS,
    ExecutionV1,
    UsageV1,
    arm_manifest,
    check_slot,
    live_identity_factory,
    planned_slots,
)
from tests.evals.quality_experiment_resources import ResourceReportV1, ResourceSampleV1
from tests.evals.quality_generation_support import (
    ExperimentGenerationReportV1,
    ExperimentGenerationStartV1,
    checked_directory,
    secret_markers,
)
from tests.evals.quality_score import samples, score_quality, validate_scored_report
from tests.evals.quality_score_contracts import QualityScoredReportV1


def digest(value):
    return quality_identity_digest(
        value.model_dump(mode="json") if isinstance(value, EvalContractModel) else value
    )


def score_projection(report):
    """A scoring view, not a replacement for the experiment's authoritative report.

    Keep the actual graph, source SHA, prompt, output digest and all observations.
    The wrapper records BOTH hashes so legacy scoring cannot erase provenance.
    """
    report = ExperimentGenerationReportV1.model_validate_json(report.model_dump_json())
    raw = report.model_dump(mode="json")
    raw = {k: raw[k] for k in QualityGenerationReportV1.model_fields}
    raw["start"] = {k: raw["start"][k] for k in QualityGenerationStartV1.model_fields}
    return QualityGenerationReportV1.model_validate_json(encoded(raw))


def verify_private(report, directory, dataset_root):
    checked_directory(directory)
    dataset = load_quality_dataset(dataset_root)
    source_by_alias = {s.alias: s for s in dataset.manifest.sources}
    cases = {c.case_id: c for c in dataset.cases}
    files = {}
    for item in report.private_files:
        path = no_links(directory / item.name)
        if not path.is_relative_to(directory):
            fail("private_path_required")
        info = path.stat()
        if info.st_mode & 0o077 or info.st_size > 20_000_000:
            fail("private_file_permissions_or_size")
        raw = path.read_bytes()
        if len(raw) != item.byte_count or quality_digest(raw) != item.digest:
            fail("private_digest_mismatch")
        if any(m and m in raw.decode() for m in secret_markers()):
            fail("unsafe_private_artifact")
        files[item.name] = read_json(path)
    if (
        ExperimentGenerationStartV1.model_validate_json(encoded(files.get("manifest.json")))
        != report.start
    ):
        fail("private_manifest_mismatch")
    sources = QualityPrivateSourcesV1.model_validate_json(encoded(files.get("sources.json")))
    from app.retrieval.chunking import normalize_document_content

    if {s.source_alias for s in sources.sources} != set(source_by_alias):
        fail("private_source_mismatch")
    for source in sources.sources:
        original = source_by_alias[source.source_alias]
        content = normalize_document_content((dataset_root / original.path).read_text())
        if (
            source.text != content
            or source.kind != original.kind
            or source.digest != quality_digest(content.encode())
        ):
            fail("private_source_mismatch")
    expected = {"manifest.json", "sources.json"}
    for index, row in enumerate(report.cases):
        o = row.observation
        if row.private_case_digest is not None:
            name = f"case-{index:04d}.json"
            expected.add(name)
            case = QualityPrivateCaseV1.model_validate_json(encoded(files[name]))
            original = cases[o.case_id]
            if (
                case.case_id,
                case.repeat_index,
                case.input.query,
                case.input.mode,
                case.resume_alias,
                case.web_scenario_alias,
                case.output_digest,
                case.failure_type,
            ) != (
                o.case_id,
                o.repeat_index,
                original.query,
                original.mode,
                original.resume_alias,
                original.web_scenario_alias,
                o.output_digest,
                o.failure_type,
            ):
                fail("private_case_mismatch")
            item = next(x for x in report.private_files if x.name == name)
            if item.digest != row.private_case_digest:
                fail("private_case_digest_mismatch")
        if o.output_digest is not None:
            name = f"output-{index:04d}.json"
            expected.add(name)
            output = QualityPrivateOutputV1.model_validate_json(encoded(files[name]))
            if (output.case_id, output.repeat_index) != (o.case_id, o.repeat_index):
                fail("private_output_mismatch")
            item = next(x for x in report.private_files if x.name == name)
            if item.digest != o.output_digest:
                fail("private_output_digest_mismatch")
            model = E7AResearchOutputV1 if report.start.arm == "candidate" else ResearchOutputV2
            model.model_validate_json(encoded(output.output))
    if set(files) != expected:
        fail("private_inventory_mismatch")
    return files


def load_execution(binding, root):
    plan = load_experiment_plan(Path(binding.harness_root) / PLAN, root=Path(binding.harness_root))
    execution = read_json(root / "execution.json", ExecutionV1)
    if execution.binding_digest != digest(binding):
        fail("execution_binding_mismatch")
    saved_plan = read_json(root / "plan.json")
    if saved_plan["binding_digest"] != digest(binding) or saved_plan["slots"] != [
        s.model_dump(mode="json") for s in planned_slots(plan)
    ]:
        fail("execution_plan_mismatch")
    claim = read_json(Path(saved_plan["execution_claim"]))
    if (
        digest(claim) != saved_plan["execution_claim_digest"]
        or claim.get("binding_digest") != digest(binding)
        or claim.get("run_root") != str(no_links(root))
        or claim.get("slots") != saved_plan["slots"]
    ):
        fail("execution_claim_mismatch")
    reports, private, resources, diagnostics = {}, {}, {}, {}
    for arm in ARMS:
        path = root / "outputs" / arm / "report.json"
        if not path.is_file():
            continue
        report = read_json(path, ExperimentGenerationReportV1)
        manifest = report.start.manifest
        expected_graph = (
            plan.identity.baseline_graph_version
            if arm == "baseline"
            else binding.candidate_graph_version
        )
        expected_config = (
            plan.identity.baseline_configuration_digest
            if arm == "baseline"
            else binding.candidate_configuration_digest
        )
        expected_prompt = (
            plan.identity.baseline_prompt_digest
            if arm == "baseline"
            else binding.candidate_prompt_digest
        )
        if (
            manifest
            != arm_manifest(
                binding,
                arm,
                live_identity_factory(_StrictMemoryInvocationRecorder()),
                root=Path(binding.harness_root),
            )[0]
            or digest(report) != execution.arm_report_digests.get(arm)
            or report.start.arm != arm
            or manifest.execution_source_sha != getattr(binding, f"{arm}_source_sha")
            or manifest.graph_version != expected_graph
            or manifest.configuration_digest != expected_config
            or manifest.prompt_digest != expected_prompt
            or manifest.llm_mode != "qwen"
            or manifest.measurement_scope != "generation"
            or tuple((s.case_id, s.repeat_index) for s in manifest.execution_order)
            != tuple(
                (s.case_id, s.repeat_index) for s in plan.samples.execution_order if s.arm == arm
            )
        ):
            fail("execution_report_mismatch")
        reports[arm] = report
        private[arm] = verify_private(
            report, root / "private" / manifest.experiment_id, Path(binding.harness_root) / DATASET
        )
        resource_path = root / arm / "resources" / "report.json"
        if resource_path.is_file():
            resource = read_json(resource_path, ResourceReportV1)
            if resource.arm != arm or digest(resource) != execution.resource_report_digests.get(
                arm
            ):
                fail("resource_report_mismatch")
            for i, sample in enumerate(resource.samples):
                if (
                    read_json(resource_path.parent / f"sample-{i:05d}.json", ResourceSampleV1)
                    != sample
                ):
                    fail("resource_sample_mismatch")
            resources[arm] = resource
        usage_path = root / arm / "usage.json"
        if usage_path.is_file():
            usage = read_json(usage_path, UsageV1)
            total = report.total_usage
            if (
                usage.started
                or usage.attempts != total.provider_attempts
                or usage.known_cost_cny != total.cost.known_cost_cny
                or usage.unknown_cost_attempts != total.cost.unknown_cost_attempts
                or usage.unknown_usage_attempts != total.unknown_usage_attempts
                or usage.input_tokens != total.input_tokens
                or usage.output_tokens != total.output_tokens
            ):
                fail("persistent_usage_mismatch")
        elif execution.execution_complete:
            fail("persistent_usage_missing")
        diagnostics[arm] = []
        for i, case in enumerate(report.cases):
            diagnostic = root / arm / "accounting" / f"diagnostic-{i:04d}.json"
            row = read_json(diagnostic) if diagnostic.is_file() else None
            if row is not None and (
                row.get("slot_index") != i
                or row.get("output_digest") != case.observation.output_digest
                or row.get("status") != case.observation.status
            ):
                fail("diagnostic_mismatch")
            diagnostics[arm].append(row)
    counters = dict.fromkeys(ARMS, 0)
    for planned, actual in zip(planned_slots(plan), execution.slots, strict=True):
        check_slot(planned, actual)
        start_path = root / "slots" / f"start-{actual.ordinal:04d}.json"
        if actual.status != "not_run":
            if read_json(start_path) != planned.model_dump(mode="json"):
                fail("slot_start_mismatch")
        elif start_path.exists():
            fail("unexecuted_slot_started")
        if actual.status in {"succeeded", "failed"}:
            if read_json(root / "slots" / f"result-{actual.ordinal:04d}.json") != actual.model_dump(
                mode="json"
            ):
                fail("slot_receipt_mismatch")
            if actual.arm not in reports:
                fail("slot_report_missing")
            case = reports[actual.arm].cases[counters[actual.arm]]
            if digest(case) != actual.case_digest or case.observation.status != actual.status:
                fail("slot_result_mismatch")
        counters[actual.arm] += 1
    cleanup = all(
        (root / arm / "cleanup.json").is_file()
        and read_json(root / arm / "cleanup.json").get("cleanup_complete") is True
        for arm in ARMS
    )
    if cleanup != execution.cleanup_complete:
        fail("cleanup_receipt_mismatch")
    complete = (
        execution.stop_category is None
        and execution.cleanup_complete
        and set(reports) == set(ARMS)
        and set(resources) == set(ARMS)
        and all(s.status in {"succeeded", "failed"} for s in execution.slots)
    )
    if complete != execution.execution_complete:
        fail("execution_completeness_mismatch")
    return plan, execution, reports, private, resources, diagnostics


class ReviewNoteV1(EvalContractModel):
    artifact_kind: Literal["e7a7_review_note_v1"] = "e7a7_review_note_v1"
    review_id: EvalIdentifier
    output_digest: EvalDigest | None
    predecessor_digest: EvalDigest | None
    annotation: HumanAnnotationV1 | None
    safety: Literal["clear", "blocked", "unknown"]
    resolution: Literal["initial", "confirmed", "corrected", "unresolved"]
    explanation: str = Field(min_length=1, max_length=50000)
    reviewer_kind: Literal["agent"] = "agent"
    reviewer_id: Literal["codex_agent"] = "codex_agent"
    human_reviewed: Literal[False] = False


class ReviewReceiptV1(EvalContractModel):
    artifact_kind: Literal["e7a7_review_receipt_v1"] = "e7a7_review_receipt_v1"
    binding_digest: EvalDigest
    export_digest: EvalDigest
    note_digests: tuple[EvalDigest, ...]
    score_digests: dict[Literal["baseline", "candidate"], EvalDigest]
    original_report_digests: dict[Literal["baseline", "candidate"], EvalDigest]
    projection_digests: dict[Literal["baseline", "candidate"], EvalDigest]
    safety: Literal["clear", "blocked", "unknown"]
    complete: bool
    unresolved: bool
    reviewer_kind: Literal["agent"] = "agent"
    independent_human: Literal[False] = False
    strictly_blind: Literal[False] = False
    validation_previously_exposed: Literal[True] = True


def review_view(plan, slot, index, report, files, dataset):
    """Only frozen reviewer material; never passed to a model or production graph."""
    review_id = f"review_{slot.ordinal:04d}"
    case = report.cases[index] if report else None
    output_digest = case.observation.output_digest if case else None
    output = files.get(f"output-{index:04d}.json")
    view = {
        "review_id": review_id,
        "case_id": slot.case_id,
        "repeat_index": slot.repeat_index,
        "output_digest": output_digest,
        "rubric_version": plan.identity.rubric_version,
        "input": files.get(f"case-{index:04d}.json"),
        "sources": files.get("sources.json"),
        "output": None,
    }
    if output:
        # Same review-facing fields. Schema/policy/arm metadata is kept in originals only.
        view["output"] = {
            k: output["output"].get(k)
            for k in (
                "evidence_sufficient",
                "summary",
                "findings",
                "limitations",
                "application_draft",
                "sources",
                "evidence",
            )
        }
    original = next(c for c in dataset.cases if c.case_id == slot.case_id)
    view["required_unit_ids"] = list(original.required_unit_ids)
    view["source_units"] = [u.model_dump(mode="json") for u in dataset.units]
    return view


def export_review(binding, root, destination):
    plan, execution, reports, private, _, _ = load_execution(binding, root)
    destination = no_links(destination)
    checked_directory(destination.parent)
    if destination.is_relative_to(Path(binding.harness_root)):
        fail("private_root_required")
    destination.mkdir(mode=0o700, exist_ok=False)
    for name in ("inputs", "initial", "recheck"):
        (destination / name).mkdir(mode=0o700)
    dataset = load_quality_dataset(Path(binding.harness_root) / DATASET)
    assignments, counters = [], dict.fromkeys(ARMS, 0)
    for slot in execution.slots:
        index = counters[slot.arm]
        counters[slot.arm] += 1
        review_id = f"review_{slot.ordinal:04d}"
        report = reports.get(slot.arm)
        case = report.cases[index] if report else None
        output_digest = case.observation.output_digest if case else None
        files = private.get(slot.arm, {})
        view = review_view(plan, slot, index, report, files, dataset)
        write_new(destination / "inputs" / f"{review_id}.json", view)
        assignments.append(
            {
                "review_id": review_id,
                "arm": slot.arm,
                "index": index,
                "case_id": slot.case_id,
                "repeat_index": slot.repeat_index,
                "output_digest": output_digest,
                "input_digest": digest(view),
            }
        )
    export = {
        "artifact_kind": "e7a7_review_export_v1",
        "binding_digest": digest(binding),
        "execution_digest": digest(execution),
        "rubric_digest": plan.identity.rubric_digest,
        "assignments": assignments,
    }
    write_new(destination / "export.json", export)
    rubric = read_json(Path(binding.harness_root) / DATASET / "rubric.json")
    write_new(destination / "inputs" / "rubric.json", rubric)
    return export


def collect_review(binding, root, directory, *, evaluator_sha=None):
    plan, execution, reports, private, _, _ = load_execution(binding, root)
    dataset = load_quality_dataset(Path(binding.harness_root) / DATASET)
    export = read_json(directory / "export.json")
    if (
        export.get("binding_digest") != digest(binding)
        or export.get("execution_digest") != digest(execution)
        or export.get("rubric_digest") != plan.identity.rubric_digest
        or len(export.get("assignments", [])) != 144
    ):
        fail("review_export_mismatch")
    if read_json(directory / "inputs" / "rubric.json") != read_json(
        Path(binding.harness_root) / DATASET / "rubric.json"
    ):
        fail("review_rubric_mismatch")
    annotations, notes = {a: [] for a in ARMS}, []
    safety, unresolved, complete = "clear", False, True
    counters = dict.fromkeys(ARMS, 0)
    for slot, assignment in zip(execution.slots, export["assignments"], strict=True):
        arm, index = slot.arm, counters[slot.arm]
        counters[arm] += 1
        if (
            assignment["arm"],
            assignment["index"],
            assignment["case_id"],
            assignment["repeat_index"],
        ) != (arm, index, slot.case_id, slot.repeat_index):
            fail("review_assignment_mismatch")
        report = reports.get(arm)
        expected_digest = report.cases[index].observation.output_digest if report else None
        if assignment["output_digest"] != expected_digest:
            fail("review_output_mismatch")
        rid = assignment["review_id"]
        if rid != f"review_{slot.ordinal:04d}":
            fail("review_identity_mismatch")
        view = read_json(directory / "inputs" / f"{rid}.json")
        if (
            view != review_view(plan, slot, index, report, private.get(arm, {}), dataset)
            or digest(view) != assignment["input_digest"]
        ):
            fail("review_input_mismatch")
        previous = None
        previous_note = None
        for phase in ("initial", "recheck"):
            path = directory / phase / f"{rid}.json"
            if not path.is_file():
                complete = False
                previous_note = None
                continue
            note = read_json(path, ReviewNoteV1)
            if (
                note.review_id != rid
                or note.output_digest != assignment["output_digest"]
                or note.predecessor_digest != previous
                or (phase == "initial") != (note.resolution == "initial")
            ):
                fail("review_note_mismatch")
            if note.safety == "blocked":
                safety = "blocked"
            elif note.safety == "unknown" and safety != "blocked":
                safety = "unknown"
            if phase == "recheck":
                if previous_note is None:
                    fail("review_predecessor_missing")
                changed = (
                    previous_note.annotation != note.annotation
                    or previous_note.safety != note.safety
                )
                if changed and note.resolution == "confirmed":
                    fail("review_disagreement_hidden")
                unresolved |= note.resolution == "unresolved"
                annotation = note.annotation
                if note.output_digest is not None:
                    if annotation is None:
                        complete = False
                    else:
                        if (
                            annotation.experiment_id != "e7a7_blinded_review"
                            or annotation.case_id != slot.case_id
                            or annotation.repeat_index != slot.repeat_index
                            or annotation.output_digest != note.output_digest
                            or annotation.rubric_version != plan.identity.rubric_version
                            or annotation.reviewer_id != "codex_agent"
                        ):
                            fail("annotation_binding_mismatch")
                        annotations[arm].append(
                            annotation.model_copy(
                                update={"experiment_id": reports[arm].start.manifest.experiment_id}
                            )
                        )
                        complete &= annotation.assessment_complete
                elif annotation is not None:
                    fail("annotation_without_output")
            previous = digest(note)
            previous_note = note
            notes.append(previous)
    dataset = load_quality_dataset(Path(binding.harness_root) / DATASET)
    scores, projections = {}, {}
    for arm, report in reports.items():
        projected = score_projection(report)
        projections[arm] = digest(projected)
        scores[arm] = score_quality(
            dataset,
            projected,
            tuple(annotations[arm]),
            scorer_source_sha=evaluator_sha or binding.candidate_source_sha,
            scorer_change_reason="e7a7_scoring_projection",
        )
    receipt = ReviewReceiptV1(
        binding_digest=digest(binding),
        export_digest=digest(export),
        note_digests=tuple(notes),
        score_digests={a: digest(s) for a, s in scores.items()},
        original_report_digests={a: digest(r) for a, r in reports.items()},
        projection_digests=projections,
        safety=safety,
        complete=complete,
        unresolved=unresolved,
    )
    return receipt, scores


class RuleV1(EvalContractModel):
    name: EvalIdentifier
    passed: bool | None
    baseline: float | int | str | None = None
    candidate: float | int | str | None = None


class DecisionV1(EvalContractModel):
    artifact_kind: Literal["e7a7_decision_v1"] = "e7a7_decision_v1"
    binding_digest: EvalDigest
    execution_digest: EvalDigest
    review_digest: EvalDigest
    comparison_digest: EvalDigest | None
    recommendation: Literal["recommend_adopt", "recommend_reject", "insufficient_evidence"]
    rules: tuple[RuleV1, ...]
    evidence_complete: bool
    production_adopted: Literal[False] = False
    baseline_accepted: Literal[False] = False
    deployed: Literal[False] = False
    measurement_scope: Literal["generation_not_full_business_recovery"] = (
        "generation_not_full_business_recovery"
    )


def macro_support(cases, field):
    values = []
    for case in cases:
        counts = getattr(case, field)
        denom = counts.supported + counts.unsupported + counts.contradicted
        if not case.assessment_complete or not denom or counts.unassessed or counts.not_assessable:
            return None
        values.append(Decimal(counts.supported) / Decimal(denom))
    return sum(values, Decimal(0)) / len(values) if values else None


def generation_elapsed(observation):
    times = [t.seconds for t in observation.stage_times if t.stage == "generation"]
    if len(times) != 1:
        return None
    value = times[0]
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
        and value >= 0
        else None
    )


def frozen_rules(plan, scores, resources):
    """Pure paired gates; all denominators/thresholds come from the frozen plan."""
    rules = []

    def add(name, passed, a=None, b=None):
        rules.append(
            RuleV1(
                name=name,
                passed=passed,
                baseline=str(a) if isinstance(a, Decimal) else a,
                candidate=str(b) if isinstance(b, Decimal) else b,
            )
        )

    def nonincrease(name, a, b):
        add(name, None if a is None or b is None else b <= a, a, b)

    primary_ids = set(plan.samples.validation_case_ids) - set(plan.samples.scope_case_ids)
    semantic_ids = set(plan.samples.selected_case_ids) - set(plan.samples.scope_case_ids)
    grouped = {}
    for scope, ids in (("validation", primary_ids), ("semantic", semantic_ids)):
        grouped[scope] = arms = {
            a: [c for c in scores[a].cases if c.observation.case_id in ids] for a in ARMS
        }
        full = all(
            len(cs) == len(ids) * 3
            and all(c.assessment_complete and c.observation.status == "succeeded" for c in cs)
            for cs in arms.values()
        )
        add(f"{scope}_complete", full)
        for label in ("overclaim", "unnecessary_refusal"):
            a, b = (sum(c.insufficiency == label for c in arms[x]) if full else None for x in ARMS)
            nonincrease(f"{scope}_{label}", a, b)
        for field in ("facts", "citations"):
            a, b = (macro_support(arms[x], field) if full else None for x in ARMS)
            add(f"{scope}_{field}_macro", None if a is None or b is None else b >= a, a, b)
        a, b = (
            sum(c.citations.unsupported + c.citations.contradicted for c in arms[x])
            if full
            else None
            for x in ARMS
        )
        nonincrease(f"{scope}_incorrect_citations", a, b)
    arms = grouped["validation"]
    usable = all(
        c.assessment_complete and c.observation.status == "succeeded"
        for cs in arms.values()
        for c in cs
    ) and all(len(cs) == plan.samples.primary_slots_per_arm for cs in arms.values())
    a, b = (sum(c.insufficiency == "overclaim" for c in arms[x]) for x in ARMS)
    add(
        "minimum_net_reduction",
        a - b >= plan.benefit.minimum_net_reduction if usable else None,
        a,
        b,
    )
    changes = {}
    for case_id in primary_ids:
        changes[case_id] = sum(
            c.insufficiency == "overclaim"
            for c in arms["baseline"]
            if c.observation.case_id == case_id
        ) - sum(
            c.insufficiency == "overclaim"
            for c in arms["candidate"]
            if c.observation.case_id == case_id
        )
    improved = sum(v > 0 for v in changes.values())
    add(
        "independent_case_improvement",
        improved >= plan.benefit.minimum_improved_case_ids if usable else None,
        0,
        improved,
    )
    repeat_changes = [
        sum(
            c.insufficiency == "overclaim"
            for c in arms["baseline"]
            if c.observation.repeat_index == i
        )
        - sum(
            c.insufficiency == "overclaim"
            for c in arms["candidate"]
            if c.observation.repeat_index == i
        )
        for i in range(3)
    ]
    add(
        "repeat_improvement",
        sum(x > 0 for x in repeat_changes) >= plan.benefit.minimum_improved_repeats
        if usable
        else None,
    )
    add(
        "repeat_non_regression",
        sum(x < 0 for x in repeat_changes) <= plan.benefit.maximum_regressed_repeats
        if usable
        else None,
    )
    for case_id in plan.samples.normal_control_case_ids:
        a, b = (
            sum(
                c.business_success is True
                for c in scores[x].cases
                if c.observation.case_id == case_id
            )
            for x in ARMS
        )
        add(f"control_{case_id}", b >= a if usable else None, a, b)
    a, b = (scores[arm].total_cost.total_cost_cny for arm in ARMS)
    add(
        "cost_increase",
        None
        if a is None or b is None
        else b <= a * (100 + plan.resources.cost_increase_percent_max) / 100,
        a,
        b,
    )
    a, b = (
        resources[x].peak_combined_bytes if x in resources and resources[x].complete else None
        for x in ARMS
    )
    add(
        "memory_increase",
        None if a is None or b is None else b - a <= plan.resources.memory_increase_bytes_max,
        a,
        b,
    )
    latencies = []
    for arm in ARMS:
        times = [generation_elapsed(c.observation) for c in grouped["semantic"][arm]]
        latencies.append(
            samples(times).p95
            if len(times) == plan.samples.semantic_slots_per_arm and None not in times
            else None
        )
    a, b = latencies
    add(
        "latency_relative",
        None
        if a is None or b is None
        else b <= a * (100 + plan.resources.generation_p95_increase_percent_max) / 100,
        a,
        b,
    )
    add(
        "latency_absolute",
        None
        if a is None or b is None
        else b <= a + plan.resources.generation_p95_increase_seconds_max,
        a,
        b,
    )
    return tuple(rules)


def decide(
    binding, plan, execution, reports, resources, receipt, scores, *, diagnostics_complete=False
):
    if set(scores) != set(ARMS) or set(reports) != set(ARMS):
        return DecisionV1(
            binding_digest=digest(binding),
            execution_digest=digest(execution),
            review_digest=digest(receipt),
            comparison_digest=None,
            recommendation="insufficient_evidence",
            rules=(RuleV1(name="both_arms_available", passed=False),),
            evidence_complete=False,
        ), None
    for score in scores.values():
        validate_scored_report(score)
    comparison = compare_quality(scores["baseline"], scores["candidate"])
    rules = list(frozen_rules(plan, scores, resources))
    for field in ("input_tokens", "output_tokens"):
        a, b = (getattr(reports[arm].total_usage, field) for arm in ARMS)
        known = all(not reports[arm].total_usage.unknown_usage_attempts for arm in ARMS)
        rules.append(
            RuleV1(
                name=field + "_increase",
                passed=(
                    b
                    <= a
                    * (
                        100
                        + getattr(
                            plan.resources,
                            "input_token_increase_percent_max"
                            if field == "input_tokens"
                            else "output_token_increase_percent_max",
                        )
                    )
                    / 100
                )
                if known
                else None,
                baseline=a,
                candidate=b,
            )
        )
    for arm in ARMS:
        usage = reports[arm].total_usage
        known = not usage.unknown_usage_attempts and not usage.cost.unknown_cost_attempts
        rules.append(
            RuleV1(
                name=f"{arm}_absolute_budget",
                passed=(
                    usage.provider_attempts <= plan.budget.per_arm_provider_attempts
                    and usage.input_tokens <= plan.budget.per_arm_input_tokens
                    and usage.output_tokens <= plan.budget.per_arm_output_tokens
                    and usage.cost.known_cost_cny <= plan.budget.per_arm_cny
                )
                if known
                else None,
            )
        )
    rules.append(RuleV1(name="diagnostics_complete", passed=diagnostics_complete))
    complete = (
        execution.execution_complete
        and diagnostics_complete
        and receipt.complete
        and not receipt.unresolved
        and all(r.measurement_complete and r.evidence_valid for r in reports.values())
        and all(r.complete for r in resources.values())
        and set(resources) == set(ARMS)
        and receipt.safety != "unknown"
        and all(s.measurement_complete for s in scores.values())
    )
    safe = receipt.safety == "clear" and not any(
        any(c.observation.safety.model_dump().values()) for r in reports.values() for c in r.cases
    )
    rules.extend(
        (
            RuleV1(name="execution_and_review_complete", passed=complete),
            RuleV1(name="safety_invariants", passed=safe),
            RuleV1(
                name="full_ci_and_resume_only_regression",
                passed=binding.ci.source_sha == binding.candidate_source_sha,
            ),
        )
    )
    if any(r.passed is None for r in rules) or not complete:
        recommendation = "insufficient_evidence"
    elif all(r.passed for r in rules):
        recommendation = "recommend_adopt"
    else:
        recommendation = "recommend_reject"
    return DecisionV1(
        binding_digest=digest(binding),
        execution_digest=digest(execution),
        review_digest=digest(receipt),
        comparison_digest=digest(comparison),
        recommendation=recommendation,
        rules=tuple(rules),
        evidence_complete=complete and all(r.passed is not None for r in rules),
    ), comparison


class EvaluationProvenanceV1(EvalContractModel):
    artifact_kind: Literal["e7a7_evaluation_provenance_v1"] = "e7a7_evaluation_provenance_v1"
    binding_digest: EvalDigest
    execution_digest: EvalDigest
    evaluator_source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    evaluator_root: str
    evaluator_source_digest: EvalDigest
    evaluator_diff_digest: EvalDigest
    ci: CIProofV1
    algorithm_version: Literal["e7a7-generation-latency-v2"] = "e7a7-generation-latency-v2"


def evaluator_provenance(binding, run_root, ci_path):
    supplied = ci_proof(read_json(ci_path))
    source_identity(ROOT, supplied.source_sha)
    verify_imports(ROOT, ROOT)
    fresh = ci_proof(
        json.loads(
            command(
                [
                    sys.executable,
                    str(ROOT / "scripts/ci_inspect_run.py"),
                    "--repo",
                    "youngcs1024/pathfinder-pub",
                    "--run-id",
                    str(supplied.run_id),
                    "--expected-sha",
                    supplied.source_sha,
                    "--attempt",
                    str(supplied.attempt),
                ]
            )
        )
    )
    if fresh != supplied:
        fail("evaluator_ci_drift")
    allowed = {
        "tests/evals/quality_experiment_decision.py",
        "tests/evals/test_quality_experiment_decision.py",
        "tests/evals/quality_experiment_execution.py",
    }
    delta = git(
        ROOT,
        "diff",
        "--name-status",
        "--no-renames",
        binding.candidate_source_sha,
        supplied.source_sha,
    ).decode()
    for line in delta.splitlines():
        status, name = line.split("\t")
        if status != "M" or name not in allowed:
            fail("unapproved_evaluator_diff")
    execution = read_json(run_root / "execution.json", ExecutionV1)
    if execution.binding_digest != digest(binding):
        fail("execution_binding_mismatch")
    return EvaluationProvenanceV1(
        binding_digest=digest(binding),
        execution_digest=digest(execution),
        evaluator_source_sha=supplied.source_sha,
        evaluator_root=str(ROOT),
        evaluator_source_digest=digest(harness_inventory(ROOT)),
        evaluator_diff_digest=quality_digest(
            git(
                ROOT,
                "diff",
                "--binary",
                "--no-ext-diff",
                binding.candidate_source_sha,
                supplied.source_sha,
            )
        ),
        ci=supplied,
    )


def decision_command(args):
    binding = read_binding(args.binding)
    verify_binding(binding)
    ci_path = getattr(args, "evaluator_ci_evidence", None)
    provenance = evaluator_provenance(binding, args.run_root, ci_path) if ci_path else None
    if (
        provenance is None
        and git(ROOT, "rev-parse", "HEAD").decode().strip() != binding.candidate_source_sha
    ):
        fail("evaluator_evidence_required")
    if args.command == "experiment-review-export":
        export_review(binding, args.run_root, args.review_root)
        if provenance is not None:
            write_new(args.review_root / "evaluator.json", provenance)
        return 0
    if provenance is not None:
        if read_json(args.review_root / "evaluator.json", EvaluationProvenanceV1) != provenance:
            fail("evaluator_provenance_drift")
    elif (args.review_root / "evaluator.json").exists():
        fail("evaluator_evidence_required")
    receipt, scores = collect_review(
        binding,
        args.run_root,
        args.review_root,
        evaluator_sha=provenance.evaluator_source_sha if provenance else None,
    )
    if args.command == "experiment-review-import":
        for arm, score in scores.items():
            write_new(args.review_root / f"score-{arm}.json", score)
        write_new(args.review_root / "review.json", receipt)
        return 0 if receipt.complete and not receipt.unresolved else 1
    if read_json(args.review_root / "review.json", ReviewReceiptV1) != receipt:
        fail("review_receipt_mismatch")
    for arm, score in scores.items():
        if read_json(args.review_root / f"score-{arm}.json", QualityScoredReportV1) != score:
            fail("score_artifact_mismatch")
    plan, execution, reports, _, resources, diagnostics = load_execution(binding, args.run_root)
    diagnostics_complete = set(diagnostics) == set(ARMS) and all(
        len(rows) == 72 and all(row is not None for row in rows) for rows in diagnostics.values()
    )
    decision, comparison = decide(
        binding,
        plan,
        execution,
        reports,
        resources,
        receipt,
        scores,
        diagnostics_complete=diagnostics_complete,
    )
    if comparison is not None:
        write_new(args.review_root / "comparison.json", comparison)
    write_new(args.review_root / "diagnostics.json", diagnostics)
    write_new(args.review_root / "decision.json", decision)
    return 0 if decision.evidence_complete else 1
