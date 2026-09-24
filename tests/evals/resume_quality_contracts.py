"""R7.1 versioned comparisons; deterministic evidence never asserts semantic quality."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from app.llm.invocations import LOCKED_CHAT_MODEL
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.product_acceptance_contracts import require
from tests.evals.quality_dataset import quality_identity_digest


class ComparisonBudget(EvalContractModel):
    cost_admission_budget_cny: Decimal = Field(default=Decimal("20"), gt=0)
    unknown_attempt_reserve_cny: Decimal = Field(default=Decimal("0.10"), gt=0)
    provider_attempt_cap: int = Field(default=100, gt=0)
    input_token_cap: int = Field(default=500000, gt=0)
    output_token_cap: int = Field(default=200000, gt=0)

    @model_validator(mode="after")
    def reserve_within_budget(self):
        if self.unknown_attempt_reserve_cny > self.cost_admission_budget_cny:
            raise ValueError("reserve exceeds budget")
        return self


class SupportedRequirement(EvalContractModel):
    requirement_id: EvalIdentifier
    quote: str = Field(min_length=1)
    fact_indexes: tuple[int, ...] = Field(min_length=1)
    kind: Literal["explicit", "preferred"] = "explicit"


class ComparisonCase(EvalContractModel):
    case_id: EvalIdentifier
    relationship: Literal["close", "different"]
    jd: str = Field(min_length=1, max_length=32000)
    supported_requirements: tuple[SupportedRequirement, ...]
    revisions: tuple[str, str]

    @model_validator(mode="after")
    def references(self):
        ids = [r.requirement_id for r in self.supported_requirements]
        if len(ids) != len(set(ids)) or any(
            r.quote not in self.jd for r in self.supported_requirements
        ):
            raise ValueError("invalid frozen requirements")
        if any(not value.strip() for value in self.revisions):
            raise ValueError("revision instructions are required")
        return self


class ComparisonDataset(EvalContractModel):
    artifact_kind: Literal["resume_quality_dataset_v1"] = "resume_quality_dataset_v1"
    material_kind: Literal["synthetic", "real"] = "synthetic"
    facts: tuple[str, ...] = Field(min_length=1)
    cases: tuple[ComparisonCase, ComparisonCase, ComparisonCase]

    @model_validator(mode="after")
    def case_set(self):
        if (
            len({c.case_id for c in self.cases}) != 3
            or [c.relationship for c in self.cases].count("close") != 2
        ):
            raise ValueError("requires two close jobs and one different job")
        for case in self.cases:
            for requirement in case.supported_requirements:
                if any(i < 0 or i >= len(self.facts) for i in requirement.fact_indexes):
                    raise ValueError("unknown supporting fact")
        return self


class LivePermission(EvalContractModel):
    """A future live controller must bind permission to the exact private inputs."""

    material_digest: EvalDigest
    environment_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    materials_may_leave_machine: Literal[True]
    budget: ComparisonBudget
    model: Literal[LOCKED_CHAT_MODEL] = LOCKED_CHAT_MODEL


def validate_live_permission(permission, *, material_digest, environment_id, budget):
    require(permission is not None, "live_permission_required")
    require(
        permission.material_digest == material_digest
        and permission.environment_id == environment_id
        and permission.budget == budget
        and permission.model == LOCKED_CHAT_MODEL,
        "live_permission_mismatch",
    )


def coverage(requirements, fully=(), partially=()):
    """Human-reviewed support only; inferred and unverified requirements stay outside."""
    denominator = {r.requirement_id for r in requirements}
    full, partial = set(fully), set(partially)
    require(
        full <= denominator and partial <= denominator and not full & partial, "coverage_invalid"
    )
    return {
        "denominator_ids": sorted(denominator),
        "fully_covered_ids": sorted(full),
        "partially_covered_ids": sorted(partial),
        "ratio": len(full) / len(denominator) if denominator else None,
        "status": "ASSESSED" if denominator else "NOT_APPLICABLE",
    }


def review_template(input_digest):
    return {
        "input_digest": input_digest,
        "initial_quality": {
            "status": "NOT_REVIEWED",
            "facts": None,
            "conditions": None,
            "relevance": None,
            "selection": None,
            "coverage": None,
            "gap_classification": None,
        },
        "finalization_cost": {
            "human_minutes": None,
            "human_rounds": None,
            "human_change_amount": None,
            "status": "NOT_REVIEWED",
        },
        "final_quality": {
            "fact_review": "NOT_REVIEWED",
            "compiled": "NOT_RUN",
            "tex_sha256": None,
            "compiler_environment": None,
            "pages": None,
            "layout": None,
            "acceptable_for_use": None,
        },
    }


def input_change(before, after):
    first, last = quality_identity_digest(before), quality_identity_digest(after)
    return {
        "before": first,
        "after": last,
        "changed": first != last,
        "same_input_advantage_claim_allowed": first == last,
    }
