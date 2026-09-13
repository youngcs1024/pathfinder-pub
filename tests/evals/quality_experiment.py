"""E6.3 offline preregistration validation, never execution or adoption approval.

The v1 conditions are the approved experiment, not configurable defaults. Changed
conditions need a new reviewed version before any candidate observations. Candidate
source binding, interleaved execution and resource measurement belong to E7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.quality_contracts import QualityRunManifestV1, SourceSHA
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
    validate_quality_run,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = "evals/datasets/quality_expanded_v1"
ACCEPTED = "evals/baselines/quality/e410-agent-v1.json"
SOURCE_SHA = "6ee09703fd19dd3f68813bb40f2b314fba512995"
ACCEPTED_DIGEST = "sha256:d3c29c48b9f168ce47d9a9f413efebbb1996654811ba403c62dc77cc5f72bb19"
LOCK_DIGEST = "sha256:46792d0e0554a696cbb39461d85b2cbd9cccbc4525aae1ab0cf07e63c79fca4e"
CAPACITY_EVIDENCE = {
    "capacity_e56_v1": "sha256:8fbfe72fd6a1a2f116b12a8fe17c6187ca90de10b819d52cf4a8811d681f384a",
    "queue_e57_v1": "sha256:7a24d0cfa066e25bc18c4a72c863c0aacabd231ed286c7ecbd80ef6d7d2280ae",
    "retrieval_e58_v1": "sha256:4837fe738495b5e4724a6ff0b39803e5c17d07ed4c4952fd48ff35175d772183",
    "faults_e59_v1": "sha256:425d2255e7b64c15e385cbcbc8e0008b5565b32b1505c40d492fb485b61bb827",
}


class ExperimentPlanError(ValueError):
    """Only fixed categories may leave the validator, including filesystem errors."""


class ComparisonIdentityV1(EvalContractModel):
    baseline_source_sha: Literal["6ee09703fd19dd3f68813bb40f2b314fba512995"]
    baseline_src_tree: Literal["99d2c2195add7c2644fe4f8efd7f26d1337faa15"]
    baseline_role: Literal["contemporaneous_rerun"]
    historical_baseline_role: Literal["selection_evidence_only"]
    accepted_digest: EvalDigest
    lock_digest: EvalDigest
    baseline_graph_version: Literal["pathfinder-research-v6"]
    baseline_prompt_digest: EvalDigest
    baseline_configuration_digest: EvalDigest
    dataset_version: Literal["quality-expanded-v1"]
    dataset_digest: EvalDigest
    split_digest: EvalDigest
    case_set_digest: EvalDigest
    rubric_version: Literal["quality-agent-delegated-frozen-v1"]
    rubric_digest: EvalDigest
    freeze_digest: EvalDigest
    mapping_digest: EvalDigest
    model: Literal["qwen3.6-flash-2026-04-16"]
    reasoning_effort: Literal["medium"]
    max_output_tokens: Literal[4096]
    embedding_profile: Literal["qwen-beijing-text-embedding-v4-1536-v1"]
    retrieval_policy_digest: EvalDigest
    web_mode: Literal["frozen_fixture"]
    document_mode: Literal["real_embedding_db"]
    trace_mode: Literal["off"]
    candidate_policy: Literal["e7-a-evidence-sufficiency-v1"]
    allowed_changes: tuple[
        Literal["evidence_assessment"],
        Literal["bounded_gap_followup"],
        Literal["writer_grounding"],
        Literal["graph_output_compatibility"],
    ]
    candidate_binding: Literal["create_only_before_first_run"]
    binding_fields: tuple[
        Literal["plan_digest"],
        Literal["baseline_source_sha"],
        Literal["candidate_source_sha"],
        Literal["candidate_prompt_digest"],
        Literal["candidate_graph_version"],
        Literal["candidate_configuration_digest"],
        Literal["environment_digest"],
        Literal["reviewed_allowed_diff_digest"],
    ]
    execution_blocked_until: Literal["binding_and_separate_run_authorization"]


class ExperimentSlotV1(EvalContractModel):
    arm: Literal["baseline", "candidate"]
    case_id: EvalIdentifier
    repeat_index: Literal[0, 1, 2]


class ExperimentSampleV1(EvalContractModel):
    selected_case_ids: tuple[EvalIdentifier, ...]
    dev_case_ids: tuple[EvalIdentifier, ...]
    validation_case_ids: tuple[EvalIdentifier, ...]
    scope_case_ids: tuple[EvalIdentifier, ...]
    normal_control_case_ids: tuple[EvalIdentifier, ...]
    repeat_count: Literal[3]
    order_policy: Literal["repeat_case_alternating_arms_v1"]
    execution_order: tuple[ExperimentSlotV1, ...]
    planned_slots: Literal[144]
    primary_slots_per_arm: Literal[39]
    semantic_slots_per_arm: Literal[63]
    scope_evidence: Literal["preclassified_not_authorization_measurement"]


class BenefitRulesV1(EvalContractModel):
    primary_metric: Literal["validation_overclaim_count"]
    label_source: Literal["frozen_rubric_insufficiency_overclaim"]
    denominator: Literal["fixed_validation_non_scope_slots"]
    minimum_net_reduction: Literal[3]
    minimum_improved_case_ids: Literal[2]
    minimum_improved_repeats: Literal[2]
    maximum_regressed_repeats: Literal[0]
    all_semantic_overclaim_increase_max: Literal[0]
    incomplete_evidence: Literal["blocks_adoption_never_improvement"]
    family_aggregation: Literal["repeats_are_not_independent_cases"]
    statistical_claim: Literal["paired_descriptive_not_p_value_only"]


class NonRegressionV1(EvalContractModel):
    scope: Literal["validation_and_all_semantic_slots"]
    normal_controls: Literal["per_case_success_count_not_lower"]
    unnecessary_refusal: Literal["raw_insufficiency_label_count_not_higher"]
    fact_support: Literal["macro_per_output_supported_fraction_not_lower"]
    citation_support: Literal["macro_per_output_supported_fraction_not_lower"]
    incorrect_citations: Literal["unsupported_plus_contradicted_count_not_higher"]
    missing_denominators: Literal["blocks_adoption_never_zero_or_perfect"]
    legitimate_resume_only_draft: Literal["required_separate_behavior_regression"]
    safety: tuple[
        Literal["workspace_and_document_scope"],
        Literal["exact_approval_binding"],
        Literal["no_duplicate_mock_effect"],
        Literal["no_secret_or_business_body_leak"],
        Literal["no_new_injection_instruction_as_experience"],
    ]
    safety_violation_max: Literal[0]
    excluded_gates: tuple[
        Literal["required_evidence_coverage_i01"],
        Literal["legacy_no_answer_handling"],
        Literal["legacy_unnecessary_refusal_ratio"],
    ]
    diagnostics: tuple[
        Literal["business_success"],
        Literal["draft_usability"],
        Literal["model_tool_provider_attempt_counts"],
        Literal["followup_new_evidence_fraction"],
        Literal["generation_sample_latency"],
        Literal["known_and_unknown_cost"],
    ]


class ReviewRulesV1(EvalContractModel):
    method: Literal["agent_delegated_frozen_rubric"]
    ordering: Literal["arm_identity_hidden_interleaved_initial_and_recheck"]
    provenance: Literal["output_digest_bound_create_only_annotations"]
    unresolved_disagreement: Literal["blocks_adoption"]
    limitations: tuple[
        Literal["not_independent_human_review"],
        Literal["validation_previously_exposed"],
        Literal["not_strictly_blind"],
    ]
    gold_in_model_context: Literal[False]
    e7_j: Literal[False]


class ExperimentBudgetV1(EvalContractModel):
    additional_dependencies: Literal[0]
    model_calls_per_run: Literal[12]
    tool_calls_per_run: Literal[8]
    research_passes_per_run: Literal[2]
    assessment_calls_per_pass: Literal[1]
    assessment_allocation: Literal["within_existing_shared_run_budget"]
    per_arm_cny: Literal[30]
    total_cny: Literal[60]
    per_arm_provider_attempts: Literal[1200]
    per_arm_input_tokens: Literal[4500000]
    per_arm_output_tokens: Literal[600000]
    includes: tuple[Literal["ingestion"], Literal["retries"], Literal["failed_attempts"]]
    transfer_between_arms: Literal[False]
    accounting: Literal["existing_factory_including_unknown_cost_reserve"]


class ResourceRulesV1(EvalContractModel):
    cost_increase_percent_max: Literal[25]
    input_token_increase_percent_max: Literal[25]
    output_token_increase_percent_max: Literal[25]
    generation_p95_increase_percent_max: Literal[25]
    generation_p95_increase_seconds_max: Literal[20]
    latency_scope: Literal["63_semantic_slots_each_arm_including_failed_elapsed"]
    latency_algorithm: Literal["nearest_rank_both_limits_must_hold"]
    memory_increase_bytes_max: Literal[268435456]
    memory_measure: Literal["peak_combined_runner_tree_rss_and_database_memory"]
    memory_sampling_seconds: Literal[1]
    memory_window: Literal["ingestion_through_execution_before_cleanup"]
    measurement_missing: Literal["blocks_adoption"]
    unknown_cost: Literal["blocks_adoption_never_zero"]
    zero_baseline: Literal["candidate_must_also_be_zero_for_relative_limit"]
    execution_slots: Literal[1]
    database_cpu: Literal[2]
    database_memory_bytes: Literal[2147483648]
    runner_tree_rss_bytes: Literal[2147483648]
    database_access: Literal["owned_exclusive_loopback_no_arbitrary_dsn"]
    case_timeout_seconds: Literal[600]
    execution_window_seconds: Literal[14400]
    environment: Literal["same_host_resources_versions_load_and_observer_both_arms"]


class ExitRulesV1(EvalContractModel):
    stop_immediately: tuple[
        Literal["safety_violation"],
        Literal["resource_limit"],
        Literal["identity_drift"],
        Literal["out_of_scope"],
    ]
    non_adoption: tuple[Literal["no_stable_benefit"], Literal["regression"], Literal["uncertainty"]]
    preservation: Literal["create_only_all_planned_failed_partial_missing_and_unassessed"]
    reroll: Literal[False]
    case_replacement: Literal[False]
    lower_threshold_after_observation: Literal[False]
    fallback: Literal["retain_baseline_default_no_candidate_deployment"]
    candidate_binding_interleaving_and_measurement: Literal["e64_e65_handoff_then_e7"]


class QualityExperimentPlanV1(EvalContractModel):
    artifact_kind: Literal["quality_experiment_preregistration_v1"]
    plan_id: Literal["e63_evidence_sufficiency_v1"]
    source_commit: SourceSHA
    selected_branch: Literal["A"]
    experiment_only: Literal[True]
    identity: ComparisonIdentityV1
    capacity_evidence: dict[str, EvalDigest]
    samples: ExperimentSampleV1
    benefit: BenefitRulesV1
    non_regression: NonRegressionV1
    review: ReviewRulesV1
    budget: ExperimentBudgetV1
    resources: ResourceRulesV1
    exit_rules: ExitRulesV1


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentPlanError("invalid_experiment_plan")
        result[key] = value
    return result


def validate_experiment_plan(plan: QualityExperimentPlanV1, *, root: Path = ROOT) -> str:
    """Verify the frozen registration, not runtime source availability or live readiness.

    Source SHA/tree identify the preregistered baseline; this works in a shallow CI
    checkout. Resolving both real source trees and verifying their allowed diff is
    mandatory in the separate pre-run binding, never inferred from this return value.
    Private E5 files are not needed or read by public CI; only their registered
    historical digests are checked here.
    """
    try:
        plan = QualityExperimentPlanV1.model_validate_json(plan.model_dump_json())
        raw = (root / ACCEPTED).read_bytes()
        if (
            plan.source_commit != SOURCE_SHA
            or plan.identity.accepted_digest != ACCEPTED_DIGEST
            or quality_digest(raw) != ACCEPTED_DIGEST
            or plan.identity.lock_digest != LOCK_DIGEST
            or quality_digest((root / "uv.lock").read_bytes()) != LOCK_DIGEST
            or plan.capacity_evidence != CAPACITY_EVIDENCE
        ):
            raise ExperimentPlanError("experiment_source_mismatch")
        accepted = json.loads(raw)["candidate"]["candidate"]
        preparation = accepted["preparation"]
        reference = next(
            s for s in accepted["scores"] if s["manifest"]["measurement_scope"] == "generation"
        )
        manifest = QualityRunManifestV1.model_validate_json(json.dumps(reference["manifest"]))
        dataset = load_quality_dataset(root / DATASET)
        validate_quality_run(dataset, manifest)
        identity = plan.identity.model_dump()
        for name in (
            "dataset_version",
            "dataset_digest",
            "split_digest",
            "case_set_digest",
            "rubric_version",
            "rubric_digest",
            "model",
            "embedding_profile",
            "retrieval_policy_digest",
            "web_mode",
            "document_mode",
        ):
            if identity[name] != getattr(manifest, name):
                raise ExperimentPlanError("experiment_dataset_mismatch")
        for name in ("graph_version", "prompt_digest", "configuration_digest"):
            if identity["baseline_" + name] != getattr(manifest, name):
                raise ExperimentPlanError("experiment_policy_mismatch")
        for name, file in (("freeze_digest", "freeze.json"), ("mapping_digest", "mapping.json")):
            content = json.loads((root / DATASET / file).read_bytes())
            if identity[name] != preparation[name] or identity[name] != quality_identity_digest(
                content
            ):
                raise ExperimentPlanError("experiment_dataset_mismatch")
        sample = plan.samples
        selected = tuple(preparation["selected_case_ids"])
        cases = {c.case_id: c for c in dataset.cases}
        expected_groups = {
            "selected_case_ids": selected,
            "dev_case_ids": tuple(c for c in selected if cases[c].split == "dev"),
            "validation_case_ids": tuple(c for c in selected if cases[c].split == "validation"),
            "scope_case_ids": tuple(c for c in selected if cases[c].scope_expectation == "reject"),
            "normal_control_case_ids": tuple(
                c["observation"]["case_id"] for c in reference["cases"] if c["business_success"]
            ),
        }
        if any(getattr(sample, field) != value for field, value in expected_groups.items()):
            raise ExperimentPlanError("experiment_selection_mismatch")
        expected_order = tuple(
            (arm, case, repeat)
            for repeat in range(3)
            for index, case in enumerate(selected)
            for arm in (
                ("baseline", "candidate")
                if (repeat + index) % 2 == 0
                else ("candidate", "baseline")
            )
        )
        actual_order = tuple((s.arm, s.case_id, s.repeat_index) for s in sample.execution_order)
        if actual_order != expected_order:
            raise ExperimentPlanError("experiment_order_mismatch")
        primary = set(sample.validation_case_ids) - set(sample.scope_case_ids)
        semantic = set(selected) - set(sample.scope_case_ids)
        if (len(actual_order), len(primary) * 3, len(semantic) * 3) != (144, 39, 63):
            raise ExperimentPlanError("experiment_denominator_mismatch")
        return quality_identity_digest(plan.model_dump(mode="json"))
    except ExperimentPlanError:
        raise
    except Exception:
        # JSON, validation and filesystem errors can contain private arguments/content.
        raise ExperimentPlanError("invalid_experiment_plan") from None


def load_experiment_plan(path: Path, *, root: Path = ROOT) -> QualityExperimentPlanV1:
    try:
        raw = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
        plan = QualityExperimentPlanV1.model_validate_json(json.dumps(raw))
        if quality_identity_digest(raw) != quality_identity_digest(plan.model_dump(mode="json")):
            raise ExperimentPlanError("invalid_experiment_plan")
        validate_experiment_plan(plan, root=root)
        return plan
    except ExperimentPlanError:
        raise
    except Exception:
        raise ExperimentPlanError("invalid_experiment_plan") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ExperimentPlanError("invalid_experiment_arguments")


def main(argv=None) -> int:
    try:
        parser = _Parser(description=__doc__, allow_abbrev=False)
        commands = parser.add_subparsers(dest="command", required=True)
        validate = commands.add_parser("validate", allow_abbrev=False)
        validate.add_argument("--plan", type=Path, required=True)
        args = parser.parse_args(argv)
        plan = load_experiment_plan(args.plan)
        print(
            json.dumps(
                {
                    "category": "preregistration_valid",
                    "digest": quality_identity_digest(plan.model_dump(mode="json")),
                    "planned_slots": plan.samples.planned_slots,
                    "primary_slots_per_arm": plan.samples.primary_slots_per_arm,
                    "semantic_slots_per_arm": plan.samples.semantic_slots_per_arm,
                    "execution_readiness": "not_checked_requires_binding_and_authorization",
                }
            )
        )
        return 0
    except ExperimentPlanError as error:
        print(json.dumps({"category": str(error)}))
        return 1
    except Exception:
        print(json.dumps({"category": "invalid_experiment_plan"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
