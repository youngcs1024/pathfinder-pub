"""Versioned resume content, privacy projection, and preference rules."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.errors import DomainValidationError


class ResumeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SourceLocationV1(ResumeModel):
    start_line: int = Field(ge=1)
    start_column: int = Field(ge=1)
    end_line: int = Field(ge=1)
    end_column: int = Field(ge=1)


class EmphasisSpanV1(ResumeModel):
    start: int = Field(ge=0)
    end: int = Field(ge=1)


class StyledTextV1(ResumeModel):
    text: str = Field(min_length=1, max_length=4000)
    emphasis: tuple[EmphasisSpanV1, ...] = ()

    @model_validator(mode="after")
    def check_spans(self) -> StyledTextV1:
        if any(span.end > len(self.text) or span.start >= span.end for span in self.emphasis):
            raise ValueError("invalid emphasis range")
        return self


class ContactFieldV1(ResumeModel):
    id: UUID
    kind: Literal["phone", "email", "other"]
    label: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=300)
    locked: bool = True
    review_status: Literal["pending", "reviewed"] = "pending"
    source: SourceLocationV1


class EducationEntryV1(ResumeModel):
    id: UUID
    institution: StyledTextV1
    qualification: StyledTextV1
    period: StyledTextV1
    source: SourceLocationV1
    review_status: Literal["pending", "reviewed"] = "pending"


class ProjectEntryV1(ResumeModel):
    id: UUID
    title: StyledTextV1
    period: StyledTextV1
    technologies: tuple[str, ...]
    summary: StyledTextV1
    bullets: tuple[StyledTextV1, ...]
    bullet_ids: tuple[UUID, ...]
    source: SourceLocationV1
    review_status: Literal["pending", "reviewed"] = "pending"

    @model_validator(mode="after")
    def check_bullets(self) -> ProjectEntryV1:
        if len(self.bullets) != len(self.bullet_ids):
            raise ValueError("project bullet identities are inconsistent")
        return self


class SkillCategoryV1(ResumeModel):
    id: UUID
    label: str = Field(min_length=1, max_length=120)
    items: tuple[str, ...]
    source: SourceLocationV1
    review_status: Literal["pending", "reviewed"] = "pending"


class ResumeContentV1(ResumeModel):
    schema_version: Literal[1] = 1
    display_name_id: UUID
    display_name: str = Field(min_length=1, max_length=200)
    display_name_review_status: Literal["pending", "reviewed"] = "pending"
    contact: tuple[ContactFieldV1, ...]
    education: tuple[EducationEntryV1, ...]
    projects: tuple[ProjectEntryV1, ...]
    skills: tuple[SkillCategoryV1, ...]

    def item_ids(self) -> frozenset[UUID]:
        return frozenset(
            [
                self.display_name_id,
                *(item.id for item in self.contact),
                *(item.id for item in self.education),
                *(item.id for item in self.projects),
                *(bid for p in self.projects for bid in p.bullet_ids),
                *(item.id for item in self.skills),
            ]
        )

    def model_projection(self) -> dict[str, object]:
        """Only adaptation text; source, name and contact stay out of model input."""
        result: dict[str, object] = {
            "education": [
                {
                    "id": str(e.id),
                    "institution": e.institution.text,
                    "qualification": e.qualification.text,
                    "period": e.period.text,
                }
                for e in self.education
            ],
            "projects": [
                {
                    "id": str(p.id),
                    "title": p.title.text,
                    "period": p.period.text,
                    "technologies": list(p.technologies),
                    "summary": p.summary.text,
                    "bullets": [b.text for b in p.bullets],
                }
                for p in self.projects
            ],
            "skills": [
                {"id": str(s.id), "label": s.label, "items": list(s.items)} for s in self.skills
            ],
        }
        check_model_input_privacy(self, result)
        return result


def check_model_input_privacy(content: ResumeContentV1, value: object) -> None:
    """Fail closed before imported fragments enter chat or embedding input."""
    email_pattern = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
    phone_pattern = re.compile(r"(?<!\d)(?:\+?\d[\d(). -]{5,}\d)(?!\d)")
    protected = [content.display_name, *(field.value for field in content.contact)]
    phone_digits = [
        "".join(c for c in field.value if c.isdigit())
        for field in content.contact
        if field.kind == "phone"
    ]

    def inspect(part: object) -> None:
        if isinstance(part, str):
            digits = "".join(c for c in part if c.isdigit())
            has_phone = any(
                10 <= len("".join(c for c in match.group() if c.isdigit())) <= 15
                for match in phone_pattern.finditer(part)
            )
            if (
                email_pattern.search(part)
                or has_phone
                or any(secret and secret in part for secret in protected)
                or any(len(phone) >= 7 and phone in digits for phone in phone_digits)
            ):
                raise DomainValidationError("resume model input contains protected data")
        elif isinstance(part, dict):
            for child in part.values():
                inspect(child)
        elif isinstance(part, (list, tuple)):
            for child in part:
                inspect(child)

    inspect(value)


class ResumeClaimV1(ResumeModel):
    id: UUID
    project_item_id: UUID
    field: Literal["title", "period", "technologies", "summary", "bullet"]
    item_id: UUID
    text: str = Field(min_length=1, max_length=4000)
    source: SourceLocationV1


class ResumePreferencesV1(ResumeModel):
    schema_version: Literal[1] = 1
    excluded_project_ids: tuple[UUID, ...] = ()
    excluded_field_ids: tuple[UUID, ...] = ()
    banned_terms: tuple[str, ...] = ()
    max_bullets_per_project: int | None = Field(default=None, ge=0, le=20)
    section_order: tuple[Literal["education", "projects", "skills"], ...] = (
        "education",
        "projects",
        "skills",
    )
    locked_item_ids: tuple[UUID, ...] = ()
    page_target: Literal[1, 2] = 1
    writing_advice: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def check_unique(self) -> ResumePreferencesV1:
        lists = (
            self.excluded_project_ids,
            self.excluded_field_ids,
            self.banned_terms,
            self.section_order,
            self.locked_item_ids,
        )
        if any(len(values) != len(set(values)) for values in lists):
            raise ValueError("preference entries must be unique")
        if set(self.section_order) != {"education", "projects", "skills"}:
            raise ValueError("section order must include every section")
        if any(not term.strip() or len(term) > 100 for term in self.banned_terms):
            raise ValueError("banned terms must be short, nonempty text")
        return self


class JobPreferenceOverrideV1(ResumeModel):
    excluded_project_ids: tuple[UUID, ...] | None = None
    excluded_field_ids: tuple[UUID, ...] | None = None
    banned_terms: tuple[str, ...] | None = None
    max_bullets_per_project: int | None = Field(default=None, ge=0, le=20)
    section_order: tuple[Literal["education", "projects", "skills"], ...] | None = None
    page_target: Literal[1, 2] | None = None
    writing_advice: str | None = Field(default=None, max_length=2000)


def effective_preferences(
    global_preferences: ResumePreferencesV1, override: JobPreferenceOverrideV1
) -> ResumePreferencesV1:
    values = global_preferences.model_dump()
    for key in override.model_fields_set:
        values[key] = getattr(override, key)
    # A job can add exclusions or locks, but cannot undo global protection.
    for key in ("excluded_project_ids", "excluded_field_ids", "banned_terms"):
        replacement = getattr(override, key)
        if replacement is not None:
            values[key] = tuple(dict.fromkeys((*getattr(global_preferences, key), *replacement)))
    return ResumePreferencesV1.model_validate(values)


def validate_preference_targets(content: ResumeContentV1, preferences: ResumePreferencesV1) -> None:
    known = content.item_ids()
    if any(
        item not in known
        for item in (*preferences.excluded_field_ids, *preferences.locked_item_ids)
    ):
        raise DomainValidationError("preference target is not in the current profile")
    if any(
        item not in {p.id for p in content.projects} for item in preferences.excluded_project_ids
    ):
        raise DomainValidationError("excluded project is not in the current profile")


def hard_constraint_issues(
    content: ResumeContentV1, preferences: ResumePreferencesV1
) -> tuple[str, ...]:
    """Deterministic rules; writing advice and page count are not machine checks."""
    issues: list[str] = []
    if {p.id for p in content.projects}.intersection(preferences.excluded_project_ids):
        issues.append("excluded_project_present")
    if content.item_ids().intersection(preferences.excluded_field_ids):
        issues.append("excluded_field_present")
    if preferences.max_bullets_per_project is not None and any(
        len(project.bullets) > preferences.max_bullets_per_project for project in content.projects
    ):
        issues.append("bullet_limit_exceeded")
    searchable = [
        *(entry.institution.text + " " + entry.qualification.text for entry in content.education),
        *(
            project.title.text
            + " "
            + project.summary.text
            + " "
            + " ".join(bullet.text for bullet in project.bullets)
            + " "
            + " ".join(project.technologies)
            for project in content.projects
        ),
        *(skill.label + " " + " ".join(skill.items) for skill in content.skills),
    ]
    if any(
        term.casefold() in text.casefold()
        for term in preferences.banned_terms
        for text in searchable
    ):
        issues.append("explicit_banned_term_present")
    return tuple(issues)


def require_locked_items_unchanged(
    original: ResumeContentV1,
    candidate: ResumeContentV1,
    preferences: ResumePreferencesV1,
) -> None:
    """Model edits cannot alter personal fields or user locked content."""
    if original.display_name != candidate.display_name or original.contact != candidate.contact:
        raise DomainValidationError("protected personal fields changed")

    def values(content: ResumeContentV1) -> dict[UUID, object]:
        result: dict[UUID, object] = {}
        for item in (*content.education, *content.projects, *content.skills):
            result[item.id] = item.model_dump(mode="json")
        for project in content.projects:
            result.update(zip(project.bullet_ids, project.bullets, strict=True))
        return result

    before, after = values(original), values(candidate)
    if any(before.get(item_id) != after.get(item_id) for item_id in preferences.locked_item_ids):
        raise DomainValidationError("locked resume content changed")
