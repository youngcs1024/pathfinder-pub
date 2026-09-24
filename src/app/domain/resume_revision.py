"""Typed, scoped edits for one immutable resume content version."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.errors import DomainValidationError
from app.domain.resume_profile import (
    EmphasisSpanV1,
    JobPreferenceOverrideV1,
    ResumeContentV1,
    ResumePreferencesV1,
    hard_constraint_issues,
    require_locked_items_unchanged,
)


class RevisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class PatchV1(RevisionModel):
    operation: Literal["replace_text", "replace_items", "remove", "move", "emphasize"]
    item_id: UUID
    field: (
        Literal[
            "bullet",
            "title",
            "summary",
            "period",
            "institution",
            "qualification",
            "label",
            "technologies",
            "items",
        ]
        | None
    ) = None
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    items: tuple[str, ...] | None = Field(default=None, max_length=40)
    position: int | None = Field(default=None, ge=0)
    emphasis: tuple[EmphasisSpanV1, ...] | None = None
    fact_version_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def valid_shape(self) -> PatchV1:
        if len(set(self.fact_version_ids)) != len(self.fact_version_ids):
            raise ValueError("duplicate fact reference")
        if self.operation == "replace_text":
            valid = self.field not in (None, "technologies", "items") and self.text is not None
            valid &= self.items is None and self.position is None and self.emphasis is None
        elif self.operation == "replace_items":
            valid = self.field in ("technologies", "items") and self.items is not None
            valid &= self.text is None and self.position is None and self.emphasis is None
            valid &= all(bool(value.strip()) and len(value) <= 120 for value in self.items)
        elif self.operation == "move":
            valid = self.field is None and self.position is not None
            valid &= self.text is None and self.items is None and self.emphasis is None
            valid &= not self.fact_version_ids
        elif self.operation == "emphasize":
            valid = self.field not in (None, "technologies", "items")
            valid &= self.emphasis is not None and self.text is None and self.items is None
            valid &= self.position is None and not self.fact_version_ids
        else:
            valid = self.field is None and self.text is None and self.items is None
            valid &= self.position is None and self.emphasis is None and not self.fact_version_ids
        if not valid:
            raise ValueError("patch operation fields do not match")
        return self


class ContentFeedbackV1(RevisionModel):
    kind: Literal["content"] = "content"
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID
    target_item_ids: tuple[UUID, ...] = Field(min_length=1, max_length=40)
    patches: tuple[PatchV1, ...] = Field(default=(), max_length=40)
    instruction: str | None = Field(default=None, min_length=1, max_length=2000)
    max_model_calls: int = Field(default=6, ge=1, le=12)
    max_tool_calls: int = Field(default=2, ge=0, le=8)
    max_cost_cny: Decimal = Field(default=Decimal("2"), gt=0, max_digits=12, decimal_places=6)

    @model_validator(mode="after")
    def valid_request(self) -> ContentFeedbackV1:
        if len(set(self.target_item_ids)) != len(self.target_item_ids):
            raise ValueError("duplicate target")
        if bool(self.patches) == bool(self.instruction):
            raise ValueError("provide patches or an instruction")
        if any(patch.item_id not in self.target_item_ids for patch in self.patches):
            raise ValueError("patch target is outside the declared scope")
        return self


class AnswerFeedbackV1(RevisionModel):
    kind: Literal["answer"] = "answer"
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID | None
    question_id: UUID
    answer: str = Field(min_length=1, max_length=2000)
    max_model_calls: int = Field(default=6, ge=1, le=12)
    max_tool_calls: int = Field(default=2, ge=0, le=8)
    max_cost_cny: Decimal = Field(default=Decimal("2"), gt=0, max_digits=12, decimal_places=6)


class PreferenceFeedbackV1(RevisionModel):
    kind: Literal["preference"] = "preference"
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID | None
    scope: Literal["round", "session", "global"]
    preferences: ResumePreferencesV1 | JobPreferenceOverrideV1
    expected_global_preference_version: int | None = Field(default=None, ge=1)
    max_model_calls: int = Field(default=6, ge=1, le=12)
    max_tool_calls: int = Field(default=2, ge=0, le=8)
    max_cost_cny: Decimal = Field(default=Decimal("2"), gt=0, max_digits=12, decimal_places=6)

    @model_validator(mode="after")
    def global_precondition(self) -> PreferenceFeedbackV1:
        if (self.scope == "global") != (self.expected_global_preference_version is not None):
            raise ValueError("global preference version precondition is invalid")
        if (self.scope == "global") != isinstance(self.preferences, ResumePreferencesV1):
            raise ValueError("preference type does not match scope")
        return self


class UserFactInputV1(RevisionModel):
    project_id: UUID
    scope: Literal["session", "project"]
    claim: str = Field(min_length=1, max_length=2000)
    kind: Literal["implementation", "plan", "experiment", "personal_statement"]
    environment: str | None = Field(default=None, max_length=500)
    fact_scope: str | None = Field(default=None, max_length=500)
    metric_basis: str | None = Field(default=None, max_length=500)
    supersedes_material_version_id: UUID | None = None
    supersedes_user_version_id: UUID | None = None

    @model_validator(mode="after")
    def one_prior_fact(self) -> UserFactInputV1:
        if self.supersedes_material_version_id and self.supersedes_user_version_id:
            raise ValueError("a correction can supersede only one fact")
        return self


class FactFeedbackV1(RevisionModel):
    kind: Literal["fact"] = "fact"
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID | None
    fact: UserFactInputV1 | None = None
    adopt_material_version_id: UUID | None = None

    @model_validator(mode="after")
    def one_action(self) -> FactFeedbackV1:
        if (self.fact is None) == (self.adopt_material_version_id is None):
            raise ValueError("choose a fact supplement or an existing fact to adopt")
        return self


class FactReviewV1(RevisionModel):
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID | None
    fact_version_id: UUID
    decision: Literal["confirm", "reject"]
    attested: bool = False


class LockChangeV1(RevisionModel):
    expected_session_revision: int = Field(ge=0)
    base_version_id: UUID
    item_id: UUID
    locked: bool


@dataclass(frozen=True, slots=True)
class RevisionDiff:
    changes: tuple[dict[str, object], ...]
    impact: tuple[str, ...]


def _locate(data: dict[str, object], item_id: UUID):
    key = str(item_id)
    for section in ("education", "projects", "skills"):
        group = data[section]
        assert isinstance(group, list)
        for index, item in enumerate(group):
            assert isinstance(item, dict)
            if item["id"] == key:
                return group, index, item, section
            if section == "projects":
                bullet_ids = item["bullet_ids"]
                assert isinstance(bullet_ids, list)
                if key in bullet_ids:
                    return item["bullets"], bullet_ids.index(key), item, "bullet"
    raise DomainValidationError("patch target does not exist")


def apply_scoped_patches(
    original: ResumeContentV1,
    patches: tuple[PatchV1, ...],
    target_item_ids: tuple[UUID, ...],
    preferences: ResumePreferencesV1,
    permitted_fact_ids: frozenset[UUID],
    fact_claims: dict[UUID, tuple[str, str]] | None = None,
) -> tuple[ResumeContentV1, RevisionDiff]:
    """Construct changes from operations so non-target fields cannot be supplied by a model."""
    if not patches or not set(p.item_id for p in patches) <= set(target_item_ids):
        raise DomainValidationError("patch scope is invalid")
    data = original.model_dump(mode="json")
    changes: list[dict[str, object]] = []
    for patch in patches:
        if not set(patch.fact_version_ids) <= permitted_fact_ids:
            raise DomainValidationError("patch cites an unavailable fact")
        group, index, item, section = _locate(data, patch.item_id)
        locked = set(preferences.locked_item_ids)
        if patch.item_id in locked or (section == "bullet" and UUID(item["id"]) in locked):
            raise DomainValidationError("patch target is locked")
        if (
            section == "projects"
            and patch.operation == "remove"
            and any(UUID(value) in locked for value in item["bullet_ids"])
        ):
            raise DomainValidationError("project contains a locked bullet")
        if patch.operation == "remove":
            if section == "bullet":
                ids = item["bullet_ids"]
                assert isinstance(ids, list)
                ids.pop(index)
            group.pop(index)
            changes.append({"item_id": str(patch.item_id), "operation": "remove"})
            continue
        if patch.operation == "move":
            assert patch.position is not None
            if patch.position >= len(group):
                raise DomainValidationError("patch position is outside its section")
            moved = group.pop(index)
            group.insert(patch.position, moved)
            if section == "bullet":
                ids = item["bullet_ids"]
                assert isinstance(ids, list)
                moved_id = ids.pop(index)
                ids.insert(patch.position, moved_id)
            changes.append(
                {"item_id": str(patch.item_id), "operation": "move", "position": patch.position}
            )
            continue
        allowed_fields = {
            "bullet": {"bullet"},
            "projects": {"title", "summary", "period", "technologies"},
            "education": {"institution", "qualification", "period"},
            "skills": {"label", "items"},
        }[section]
        if patch.field not in allowed_fields:
            raise DomainValidationError("patch field is invalid for the target")
        if section == "bullet":
            value = group[index]
        else:
            value = item[patch.field]
        if patch.operation in {"replace_text", "replace_items"}:
            claims = fact_claims or {}
            cited = [claims[value] for value in patch.fact_version_ids if value in claims]
            if len(cited) != len(patch.fact_version_ids):
                raise DomainValidationError("patch fact details are unavailable")
            new_text = patch.text or " ".join(patch.items or ())
            old_text = value.get("text", "") if isinstance(value, dict) else " ".join(value)
            allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", old_text))
            for claim, kind in cited:
                allowed_numbers.update(re.findall(r"\d+(?:\.\d+)?%?", claim))
                if kind == "plan" and not re.search(
                    r"\b(plan|planned|proposed)\b|计划|规划", new_text, re.I
                ):
                    raise DomainValidationError("a planned fact needs explicit planned wording")
            if not set(re.findall(r"\d+(?:\.\d+)?%?", new_text)) <= allowed_numbers:
                raise DomainValidationError("new numbers are not present in cited facts")
        if patch.operation == "replace_items":
            assert patch.items is not None
            if not patch.fact_version_ids:
                raise DomainValidationError("item changes require confirmed fact references")
            replacement: object = list(patch.items)
        else:
            assert isinstance(value, dict)
            replacement = dict(value)
            if patch.operation == "replace_text":
                if not patch.fact_version_ids:
                    raise DomainValidationError("text changes require confirmed fact references")
                replacement["text"] = patch.text
                replacement["emphasis"] = []
            else:
                assert patch.emphasis is not None
                replacement["emphasis"] = [span.model_dump() for span in patch.emphasis]
        if section == "bullet":
            group[index] = replacement
        else:
            item[patch.field] = replacement
        changes.append(
            {
                "item_id": str(patch.item_id),
                "operation": patch.operation,
                "field": patch.field,
                "fact_version_ids": [str(v) for v in patch.fact_version_ids],
            }
        )
    try:
        updated = ResumeContentV1.model_validate(data)
        require_locked_items_unchanged(original, updated, preferences)
    except ValueError:
        raise DomainValidationError("revision violates content or lock constraints") from None
    if hard_constraint_issues(updated, preferences):
        raise DomainValidationError("revision violates hard constraints")
    impact = tuple(
        (
            f"Removed item {change['item_id']}; check JD coverage and page layout."
            if change["operation"] == "remove"
            else f"Moved item {change['item_id']}; check section order and page layout."
            if change["operation"] == "move"
            else (
                f"Updated {change['field']} of item {change['item_id']}; "
                "check facts, JD coverage and page layout."
            )
        )
        for change in changes
    )
    return updated, RevisionDiff(tuple(changes), impact)


class RevisionCandidateV1(RevisionModel):
    content: ResumeContentV1 | None
    patches: tuple[PatchV1, ...]
    diff: tuple[dict[str, object], ...]
    impact: tuple[str, ...]
    questions: tuple[str, ...]
    correction_count: int = Field(ge=0, le=1)
    prompt_version: str
    model_id: str
    retrieval_config_version: str


@dataclass(frozen=True, slots=True)
class RevisionInputs:
    session_id: UUID
    run_id: UUID
    feedback_id: UUID
    base_version_id: UUID | None
    base_content: ResumeContentV1 | None
    profile_content: ResumeContentV1
    preferences: ResumePreferencesV1
    target_item_ids: tuple[UUID, ...]
    request: ContentFeedbackV1 | AnswerFeedbackV1 | PreferenceFeedbackV1
    permitted_fact_ids: frozenset[UUID]
    fact_claims: tuple[tuple[UUID, str, str, dict[str, object]], ...]
    instruction: str | None


def user_fact_issues(fact: UserFactInputV1) -> tuple[str, ...]:
    issues: list[str] = []
    if fact.kind == "experiment" and not fact.environment:
        issues.append("experiment_environment_missing")
    if re.search(r"\d", fact.claim):
        if not fact.environment:
            issues.append("metric_environment_missing")
        if not fact.fact_scope:
            issues.append("metric_scope_missing")
        if not fact.metric_basis:
            issues.append("metric_basis_missing")
    return tuple(issues)


class ResumeRevisionPort:
    async def command(self, tenant, session_id: UUID, payload, key: UUID): ...
    async def list_feedback(self, tenant, session_id: UUID): ...
    async def list_user_facts(self, tenant, session_id: UUID): ...
    async def questions(self, tenant, session_id: UUID): ...


class ResumeRevisionService:
    def __init__(self, port: ResumeRevisionPort) -> None:
        self.port = port

    async def command(self, tenant, session_id: UUID, payload, key: UUID):
        return await self.port.command(tenant, session_id, payload, key)

    async def list_feedback(self, tenant, session_id: UUID):
        return await self.port.list_feedback(tenant, session_id)

    async def list_user_facts(self, tenant, session_id: UUID):
        return await self.port.list_user_facts(tenant, session_id)

    async def questions(self, tenant, session_id: UUID):
        return await self.port.questions(tenant, session_id)
