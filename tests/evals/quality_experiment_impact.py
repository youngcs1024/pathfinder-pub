"""E6.4 read-only impact registration; no candidate, database or deployment execution.

Source identities describe the reviewed baseline, not the current checkout forever.
Actual candidate source binding and execution readiness remain separate E7 work.
The registered fixture exception is a future test-only scope, never DDL permission
for this validator or a claim that a new graph can already execute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from tests.evals.contracts import EvalContractModel, EvalIdentifier
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment import load_experiment_plan, validate_experiment_plan

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = "evals/experiments/e63-evidence-sufficiency-v1.json"
PLAN_DIGEST = "sha256:7e7ae136f40a1eb2f7d292fa0f6ddb191fbd2e0b9e95726e8f0a7755210284ff"
MAX_REVIEW_BYTES = 131_072

# Reviewed source locations and required decisions, not permission to edit these files.
AREAS = {
    "document_representation": (
        "unchanged",
        ("src/app/retrieval/chunking.py", "src/app/retrieval/documents.py"),
        ("keep_normalization_chunking_embedding", "preserve_document_chunk_references"),
    ),
    "retrieval_strategy": (
        "unchanged",
        ("src/app/db/documents.py", "src/app/tools/document_retrieval.py"),
        ("keep_ranking_top_k_profile", "preserve_workspace_allowlist"),
    ),
    "output_contract": (
        "dev_only",
        ("src/app/domain/research.py", "src/app/agents/research_nodes.py"),
        (
            "independent_experiment_output_contract",
            "grounded_sufficient_draft_only",
            "partial_insufficient_conflicting_without_action",
            "retain_legitimate_resume_only_draft",
            "technical_errors_are_not_insufficiency",
        ),
    ),
    "graph_nodes": (
        "dev_only",
        ("src/app/agents/research_graph.py", "src/app/agents/research_contracts.py"),
        (
            "distinct_experiment_graph_identity",
            "reuse_langgraph_factory_registry_ports",
            "bounded_gap_followup_no_scope_expansion",
            "no_second_runtime_or_production_root",
        ),
    ),
    "model_budget": (
        "dev_only",
        ("src/app/domain/runs.py", "src/app/llm/factory.py"),
        ("assessor_uses_existing_shared_budget", "keep_all_e63_resource_and_exit_limits"),
    ),
    "db_schema": (
        "dev_only",
        (
            "src/app/db/models.py",
            "src/app/db/readiness.py",
            "tests/evals/quality_generation.py",
        ),
        (
            "owned_disposable_handle_only",
            "extend_runs_graph_check_only",
            "same_fixture_schema_for_both_arms",
            "persist_truthful_candidate_version_and_output",
            "test_only_candidate_reader",
            "no_production_migration_or_new_tables",
        ),
    ),
    "http_behavior": (
        "unchanged",
        ("src/app/api/schemas/runs.py", "src/app/db/runs.py"),
        ("keep_production_api_ui_and_result_readers", "generation_is_not_full_business_evidence"),
    ),
    "sse_latency": (
        "unchanged",
        ("src/app/api/routes/events.py",),
        ("keep_sse_polling_replay_and_authorization",),
    ),
    "request_admission": (
        "unchanged",
        ("src/app/domain/runs.py", "src/app/db/jobs.py"),
        ("keep_admission_idempotency_and_single_worker",),
    ),
    "rollback": (
        "adoption_only",
        ("src/app/worker/main.py", "src/app/worker/langgraph_executor.py", "scripts/release.sh"),
        (
            "retain_baseline_and_failed_partial_artifacts",
            "drain_or_cancel_nonterminal_with_old_version",
            "reconcile_executing_action_before_switch",
            "never_migrate_pending_approval_or_delete_checkpoint",
            "rollback_requires_all_result_versions_readable",
            "reject_audit_or_idempotency_losing_downgrade",
        ),
    ),
}


class ImpactReviewError(ValueError):
    """Only fixed categories cross the CLI boundary."""


class ImpactAreaV1(EvalContractModel):
    area: EvalIdentifier
    disposition: Literal["unchanged", "dev_only", "adoption_only"]
    reviewed_files: tuple[str, ...] = Field(min_length=1, max_length=8)
    requirements: tuple[EvalIdentifier, ...] = Field(min_length=1, max_length=16)


class ExperimentDatabaseScopeV1(EvalContractModel):
    admission: Literal["owned_disposable_handle_only_no_arbitrary_dsn"]
    schema_change: Literal["extend_runs_graph_check_only"]
    both_arms: Literal["same_fixture_schema"]
    business_enums: Literal["unchanged"]
    result_authority: Literal["test_postgresql"]
    result_reader: Literal["test_only_candidate_contract"]
    production_migration: Literal["none"]
    new_tables: Literal[False]


class ExperimentOutputPolicyV1(EvalContractModel):
    sufficient: Literal["grounded_valid_draft_eligible_not_authorized"]
    partial: Literal["supported_facts_and_gaps_no_action"]
    insufficient: Literal["explicit_unknowns_no_action"]
    conflicting: Literal["conflicts_and_sources_no_action"]
    resume_only: Literal["allowed_without_invented_job_match"]
    technical_failure: Literal["existing_failure_or_cancellation_not_semantic_outcome"]


class ExperimentBudgetAllocationV1(EvalContractModel):
    model_calls_per_run: Literal[12]
    tool_calls_per_run: Literal[8]
    research_passes_per_run: Literal[2]
    assessor_calls_per_pass: Literal[1]
    allocation: Literal["existing_shared_budget"]


class QualityExperimentImpactV1(EvalContractModel):
    artifact_kind: Literal["quality_experiment_impact_v1"]
    review_id: Literal["e64_evidence_sufficiency_v1"]
    reviewed_source_sha: Literal["cbe8faf383e5a9fe72730d33128cc2c10c733b68"]
    baseline_source_sha: Literal["6ee09703fd19dd3f68813bb40f2b314fba512995"]
    baseline_src_tree: Literal["99d2c2195add7c2644fe4f8efd7f26d1337faa15"]
    plan_path: Literal["evals/experiments/e63-evidence-sufficiency-v1.json"]
    plan_digest: Literal[PLAN_DIGEST]
    selected_branch: Literal["A"]
    experiment_only: Literal[True]
    candidate_location: Literal["tests/evals"]
    candidate_graph_version: Literal["pathfinder-research-e7a-exp-v1"]
    candidate_output_contract: Literal["e7a_research_output_v1"]
    reviewed_production_graph_version: Literal["pathfinder-research-v6"]
    reviewed_output_versions: tuple[Literal[1], Literal[2]]
    reviewed_database_revision: Literal["0015_e3_run_request_identity"]
    image_boundary: Literal["tests_and_evals_excluded_no_production_assembly"]
    database: ExperimentDatabaseScopeV1
    output_policy: ExperimentOutputPolicyV1
    budget: ExperimentBudgetAllocationV1
    impacts: tuple[ImpactAreaV1, ...] = Field(min_length=10, max_length=10)
    adoption_requirements: tuple[
        Literal["separate_implementation_and_deployment_authorization"],
        Literal["new_production_graph_and_versioned_output_union"],
        Literal["writer_executor_api_ui_fake_prompt_eval_alignment"],
        Literal["append_from_actual_alembic_head_if_needed"],
        Literal["orm_check_readiness_release_backup_restore_alignment"],
        Literal["old_terminal_reading_and_nonterminal_drain"],
        Literal["approval_binding_reconciliation_interrupt_resume_budget_tests"],
        Literal["compatible_rollback_preserves_audit_and_result_reading"],
    ]
    e65_handoff: tuple[
        Literal["real_source_binding_and_reviewed_allowed_diff"],
        Literal["cross_source_paired_slot_execution"],
        Literal["owned_database_fixture_and_candidate_reader"],
        Literal["resource_collection_and_budget_enforcement"],
        Literal["adoption_rule_execution_and_failure_retention"],
        Literal["semantic_scope_budget_and_legitimate_resume_regressions"],
        Literal["separate_full_business_and_recovery_acceptance"],
    ]
    verification_limit: Literal["impact_registration_only"]
    execution_readiness: Literal["NOT_VERIFIED"]
    candidate_execution: Literal["NOT_RUN"]
    adoption: Literal["NOT_RUN"]
    deployment: Literal["NOT_RUN"]


def _public_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or relative.is_absolute()
        or relative.as_posix() != name
        or any(part in {".", ".."} for part in relative.parts)
        or "\\" in name
        or ":" in name
    ):
        raise ImpactReviewError("invalid_impact_path")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ImpactReviewError("invalid_impact_path")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ImpactReviewError("invalid_impact_path")
    return path


def validate_impact_review(review: QualityExperimentImpactV1, *, root: Path = ROOT) -> str:
    """Check registration integrity only; never resolve Git or open a database."""
    try:
        original = review.model_dump(mode="json")
        review = QualityExperimentImpactV1.model_validate_json(review.model_dump_json())
        if quality_identity_digest(original) != quality_identity_digest(
            review.model_dump(mode="json")
        ):
            raise ImpactReviewError("invalid_impact_review")
        plan = load_experiment_plan(_public_file(root, review.plan_path), root=root)
        if (
            validate_experiment_plan(plan, root=root) != review.plan_digest
            or plan.identity.baseline_source_sha != review.baseline_source_sha
            or plan.identity.baseline_src_tree != review.baseline_src_tree
            or plan.identity.baseline_graph_version != review.reviewed_production_graph_version
            or plan.selected_branch != review.selected_branch
            or plan.experiment_only != review.experiment_only
            or plan.budget.model_calls_per_run != review.budget.model_calls_per_run
            or plan.budget.tool_calls_per_run != review.budget.tool_calls_per_run
            or plan.budget.research_passes_per_run != review.budget.research_passes_per_run
            or plan.budget.assessment_calls_per_pass != review.budget.assessor_calls_per_pass
        ):
            raise ImpactReviewError("impact_plan_mismatch")
        if len({item.area for item in review.impacts}) != len(AREAS):
            raise ImpactReviewError("impact_coverage_mismatch")
        for item in review.impacts:
            for name in item.reviewed_files:
                _public_file(root, name)
            if AREAS.get(item.area) != (
                item.disposition,
                item.reviewed_files,
                item.requirements,
            ):
                raise ImpactReviewError("impact_coverage_mismatch")
        return quality_identity_digest(review.model_dump(mode="json"))
    except ImpactReviewError:
        raise
    except Exception:
        raise ImpactReviewError("invalid_impact_review") from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImpactReviewError("invalid_impact_review")
        result[key] = value
    return result


def load_impact_review(path: Path, *, root: Path = ROOT) -> QualityExperimentImpactV1:
    try:
        if path.is_symlink():
            raise ImpactReviewError("invalid_impact_path")
        with path.open("rb") as stream:
            content = stream.read(MAX_REVIEW_BYTES + 1)
        if len(content) > MAX_REVIEW_BYTES:
            raise ImpactReviewError("invalid_impact_review")
        raw = json.loads(content, object_pairs_hook=_unique_object)
        review = QualityExperimentImpactV1.model_validate_json(json.dumps(raw))
        if quality_identity_digest(raw) != quality_identity_digest(review.model_dump(mode="json")):
            raise ImpactReviewError("invalid_impact_review")
        validate_impact_review(review, root=root)
        return review
    except ImpactReviewError:
        raise
    except Exception:
        raise ImpactReviewError("invalid_impact_review") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ImpactReviewError("invalid_impact_arguments")


def main(argv=None) -> int:
    try:
        parser = _Parser(description=__doc__, allow_abbrev=False)
        commands = parser.add_subparsers(dest="command", required=True)
        validate = commands.add_parser("validate", allow_abbrev=False)
        validate.add_argument("--review", type=Path, required=True)
        args = parser.parse_args(argv)
        review = load_impact_review(args.review)
        print(
            json.dumps(
                {
                    "category": "impact_review_valid",
                    "digest": quality_identity_digest(review.model_dump(mode="json")),
                    "impact_areas": len(review.impacts),
                    "verification_limit": review.verification_limit,
                    "execution_readiness": review.execution_readiness,
                    "candidate_execution": review.candidate_execution,
                    "adoption": review.adoption,
                    "deployment": review.deployment,
                }
            )
        )
        return 0
    except ImpactReviewError as error:
        category = str(error)
        if category not in {
            "invalid_impact_arguments",
            "invalid_impact_path",
            "invalid_impact_review",
            "impact_plan_mismatch",
            "impact_coverage_mismatch",
        }:
            category = "invalid_impact_review"
        print(json.dumps({"category": category}))
        return 1
    except Exception:
        print(json.dumps({"category": "invalid_impact_review"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
