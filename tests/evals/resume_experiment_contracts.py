"""Independent A-D experiment identities; no R7 allocation or rubric inheritance."""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL
from app.llm.pricing import QWEN_BEIJING_PRICING_VERSION
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.product_acceptance_contracts import require
from tests.evals.quality_dataset import quality_identity_digest

VERSION = "resume-experiments-v1"
QUOTAS = {"backend": 8, "agent": 8, "retrieval_data": 4}
PLANNED = {"a": 180, "b_live": 400, "b_patch": 1000, "c": 40, "d": 1000, "d_block": 60}
MINIMUM = "safe_supported_content_with_one_fully_covered_supported_requirement_v1"


class ExperimentBudget(EvalContractModel):
    cost_cap_cny: Decimal = Field(default=Decimal("100"), gt=0, le=100)
    attempt_cap: int = Field(default=3000, gt=0, le=3000)


class JobCase(EvalContractModel):
    case_id: EvalIdentifier
    category: Literal["backend", "agent", "retrieval_data"]
    official_job_id: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https://")
    retrieved_at: str = Field(min_length=1)
    title: str = Field(min_length=1)
    jd: str = Field(min_length=1, max_length=32000)
    raw_file: str
    body_file: str


class Inputs(EvalContractModel):
    artifact_kind: Literal["resume-experiments-v1"] = VERSION
    allocation_id: UUID
    execution_root: str
    source_material_identity: EvalDigest
    files: dict[str, EvalDigest]
    cases: tuple[JobCase, ...] = Field(min_length=20, max_length=20)
    pilot_job_ids: tuple[str, str, str]
    facts_file: str = "facts.json"
    profile_file: str = "profile.json"
    template_file: str = "resume.tex"
    budget: ExperimentBudget = ExperimentBudget()
    chat_model: Literal[LOCKED_CHAT_MODEL] = LOCKED_CHAT_MODEL
    embedding_model: Literal[LOCKED_EMBEDDING_MODEL] = LOCKED_EMBEDDING_MODEL
    pricing_version: Literal[QWEN_BEIJING_PRICING_VERSION] = QWEN_BEIJING_PRICING_VERSION
    approved_by: Literal["user_explicit_experiment_preparation"]
    materials_may_leave_machine: Literal[True]
    review_kind: Literal["AGENT_ASSESSED"] = "AGENT_ASSESSED"
    blind_seed: int = 20260927

    @model_validator(mode="after")
    def scope(self):
        if not Path(self.execution_root).is_absolute():
            raise ValueError("absolute execution root required")
        if Counter(c.category for c in self.cases) != QUOTAS:
            raise ValueError("invalid category quotas")
        if len({c.case_id for c in self.cases}) != 20:
            raise ValueError("duplicate case")
        identities = {c.official_job_id.casefold() for c in self.cases}
        bodies = {re.sub(r"\s+", "", c.jd).casefold() for c in self.cases}
        if len(identities) != 20 or len(bodies) != 20:
            raise ValueError("duplicate job")
        if identities & {v.casefold() for v in self.pilot_job_ids}:
            raise ValueError("pilot included in formal cases")
        required = {self.facts_file, self.profile_file, self.template_file}
        required.update(c.raw_file for c in self.cases)
        required.update(c.body_file for c in self.cases)
        if not required <= self.files.keys():
            raise ValueError("material missing")
        for name in self.files:
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or p.as_posix() != name:
                raise ValueError("unsafe material path")
        return self

    @property
    def digest(self):
        return quality_identity_digest(self.model_dump(mode="json"))


class Requirement(EvalContractModel):
    requirement_id: EvalIdentifier
    quote: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: Literal["explicit", "preferred"]
    applicable: bool
    support: Literal["full", "partial", "unsupported", "insufficient"]
    fact_version_ids: tuple[str, ...]
    profile_item_ids: tuple[str, ...] = ()
    necessary_conditions: tuple[str, ...]
    rationale: str = Field(min_length=1)


class Annotation(EvalContractModel):
    requirements: tuple[Requirement, ...] = Field(min_length=1)


def normalize_json(raw):
    """No repair, coercion, extraction from prose, duplicate keys or nonfinite values."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "duplicate_json_key")
            value[key] = item
        return value

    def invalid(_):
        raise ValueError("nonfinite_json")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "json_object_required")
    return value


def checked_annotation(value, case, facts, profile=None):
    result = Annotation.model_validate_json(json.dumps(value))
    ids = [r.requirement_id for r in result.requirements]
    require(len(ids) == len(set(ids)), "duplicate_requirement")
    known = {f["version_id"] for f in facts}
    profile_ids = {
        item["id"] for kind in ("education", "skills") for item in (profile or {}).get(kind, [])
    }
    for row in result.requirements:
        # Unique exact quotes can have offsets deterministically located; never fuzzy match.
        require(case.jd[row.start : row.end] == row.quote, "invalid_requirement_quote")
        require(set(row.fact_version_ids) <= known, "unknown_fact")
        require(set(row.profile_item_ids) <= profile_ids, "unknown_profile_item")
        require(len(set(row.fact_version_ids)) == len(row.fact_version_ids), "duplicate_fact")
        require(
            row.support not in {"full", "partial"} or row.fact_version_ids or row.profile_item_ids,
            "support_missing",
        )
        require(
            row.support != "unsupported" or not (row.fact_version_ids or row.profile_item_ids),
            "unsupported_has_facts",
        )
    return result


def locate_annotation(value, case):
    """Correct only unique literal quotations, preserving all semantic fields."""
    value = json.loads(json.dumps(value))
    for row in value.get("requirements", []):
        quote = row.get("quote")
        if isinstance(quote, str) and quote and case.jd.count(quote) == 1:
            row["start"] = case.jd.index(quote)
            row["end"] = row["start"] + len(quote)
    return value
