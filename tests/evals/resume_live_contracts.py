"""Private R7.1 live inputs and delegated reviews, separate from synthetic v1."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.product_acceptance_contracts import require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_quality_contracts import ComparisonBudget


class LiveProject(EvalContractModel):
    alias: EvalIdentifier
    title: str = Field(min_length=1)
    profile_project_title: str = Field(min_length=1)
    files: tuple[str, ...] = Field(min_length=1, max_length=20)
    provenance: tuple[dict[str, str | int], ...] = Field(min_length=1)


class LiveCase(EvalContractModel):
    case_id: EvalIdentifier
    relationship: Literal["close", "different"]
    jd: str = Field(min_length=1, max_length=32000)
    source_url: str = Field(pattern=r"^https://")
    retrieved_at: str = Field(min_length=1)


class LiveInputs(EvalContractModel):
    artifact_kind: Literal["resume_live_inputs_v1"] = "resume_live_inputs_v1"
    allocation_id: UUID
    execution_root: str
    environment_id: Literal["wsl-local-owned-postgres"] = "wsl-local-owned-postgres"
    reviewer: Literal["codex_agent_delegated"] = "codex_agent_delegated"
    materials_may_leave_machine: Literal[True]
    approved_by: Literal["user_explicit_r71_delegation"] = "user_explicit_r71_delegation"
    budget: ComparisonBudget
    projects: tuple[LiveProject, ...] = Field(min_length=1, max_length=20)
    cases: tuple[LiveCase, LiveCase, LiveCase]
    resume_file: str = "resume.tex"
    files: dict[str, EvalDigest]
    chat_model: Literal[LOCKED_CHAT_MODEL] = LOCKED_CHAT_MODEL
    embedding_model: Literal[LOCKED_EMBEDDING_MODEL] = LOCKED_EMBEDDING_MODEL
    pricing_version: Literal[QWEN_BEIJING_PRICING_VERSION] = QWEN_BEIJING_PRICING_VERSION

    @model_validator(mode="after")
    def valid_scope(self):
        if not Path(self.execution_root).is_absolute():
            raise ValueError("execution root must be absolute")
        required = {self.resume_file, *(f for project in self.projects for f in project.files)}
        if required != self.files.keys():
            raise ValueError("input file scope differs")
        for name in self.files:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
                raise ValueError("unsafe input path")
        if len({p.alias for p in self.projects}) != len(self.projects):
            raise ValueError("duplicate project")
        if (
            len({c.case_id for c in self.cases}) != 3
            or sum(c.relationship == "close" for c in self.cases) != 2
        ):
            raise ValueError("requires two close cases and one different case")
        if self.budget.cost_admission_budget_cny > 20 or self.budget.provider_attempt_cap > 100:
            raise ValueError("exceeds approved allocation")
        return self

    @property
    def digest(self):
        return quality_identity_digest(self.model_dump(mode="json"))


def require_review(review, *, binding, kind):
    require(isinstance(review, dict), "review_required")
    require(
        review.get("reviewer") == "codex_agent_delegated"
        and review.get("authorization_digest") == binding
        and review.get("kind") == kind,
        "review_binding_mismatch",
    )
    require(bool(review.get("rationale")), "review_rationale_required")
