from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.domain.errors import DomainValidationError
from app.domain.resume_profile import (
    JobPreferenceOverrideV1,
    ResumePreferencesV1,
    check_model_input_privacy,
    effective_preferences,
    hard_constraint_issues,
    require_locked_items_unchanged,
    validate_preference_targets,
)
from app.resume.template_import import parse_resume_source

SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


def _parse(source: str):
    preamble = source.split("\\begin{document}", 1)[0]
    return parse_resume_source(
        source,
        expected_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        expected_preamble_sha256=hashlib.sha256(preamble.encode()).hexdigest(),
    )


def test_synthetic_source_preserves_structure_ids_and_pending_claims() -> None:
    source = SOURCE.read_text()
    first = _parse(source)
    second = _parse(source)
    assert first.complete and not first.issues
    assert first.content is not None and second.content is not None
    assert first.content.model_dump() == second.content.model_dump()
    assert len(first.content.contact) == 3
    assert len(first.content.education) == 1
    assert len(first.content.projects) == 1
    assert len(first.content.projects[0].bullets) == 2
    assert first.content.projects[0].bullets[0].emphasis
    assert "25%" in first.content.projects[0].bullets[1].text
    assert len(first.content.skills) == 1
    assert all(claim.source.start_line > 1 for claim in first.claims)
    assert {claim.field for claim in first.claims} == {
        "title",
        "period",
        "technologies",
        "summary",
        "bullet",
    }
    assert all(field.locked for field in first.content.contact)


def test_source_identity_and_unsupported_structure_block_import_with_positions() -> None:
    source = SOURCE.read_text()
    mismatch = parse_resume_source(source)
    assert not mismatch.complete
    assert mismatch.issues[0].code == "source_identity_mismatch"
    changed = source.replace(
        "\\resumeItem{Implemented", "\\unknown{value}\n\\resumeItem{Implemented"
    )
    preview = _parse(changed)
    assert not preview.complete and preview.content is None
    assert preview.issues[0].code == "unsupported_command"
    assert preview.issues[0].line > 10
    assert preview.issues[0].column == 1
    extra = source.replace(
        "\\resumeProjectListEnd", "Unparsed synthetic prose.\n\\resumeProjectListEnd"
    )
    extra_preview = _parse(extra)
    assert not extra_preview.complete
    assert extra_preview.issues[0].code == "unsupported_structure"


def test_contact_canary_is_excluded_and_injected_copy_is_blocked() -> None:
    source = SOURCE.read_text()
    content = _parse(source).content
    assert content is not None
    projection = content.model_projection()
    assert "canary@example.test" not in str(projection)
    assert "555-0100" not in str(projection)
    injected = source.replace("bounded parser.", "bounded parser canary@example.test.")
    changed = _parse(injected).content
    assert changed is not None
    with pytest.raises(DomainValidationError):
        changed.model_projection()
    with pytest.raises(DomainValidationError):
        check_model_input_privacy(content, ("Unrelated evidence", "Phone: 555 0100"))
    with pytest.raises(DomainValidationError):
        check_model_input_privacy(content, "Send this to a-different@example.test")
    check_model_input_privacy(content, {"fact_version_id": "1c5b44b0-1e28-4060-9497-50fdc2bb257a"})
    check_model_input_privacy(content, '{"fact_version_id":"1c5b44b0-1e28-4060-9497-50fdc2bb257a"}')


def test_preferences_keep_global_exclusions_and_soft_advice_separate() -> None:
    content = _parse(SOURCE.read_text()).content
    assert content is not None
    project_id = content.projects[0].id
    global_value = ResumePreferencesV1(
        excluded_project_ids=(project_id,),
        banned_terms=("explicit-only",),
        writing_advice="Be concise",
        locked_item_ids=(content.contact[0].id,),
    )
    effective = effective_preferences(
        global_value,
        JobPreferenceOverrideV1(
            page_target=2,
            banned_terms=("another-explicit-term",),
            writing_advice="Emphasize systems",
        ),
    )
    assert effective.page_target == 2
    assert effective.excluded_project_ids == (project_id,)
    assert effective.banned_terms == ("explicit-only", "another-explicit-term")
    assert effective.locked_item_ids == global_value.locked_item_ids
    assert effective.writing_advice == "Emphasize systems"
    validate_preference_targets(content, effective)
    assert "excluded_project_present" in hard_constraint_issues(content, effective)
    assert "bullet_limit_exceeded" in hard_constraint_issues(
        content, ResumePreferencesV1(max_bullets_per_project=1)
    )
    assert "explicit_banned_term_present" in hard_constraint_issues(
        content, ResumePreferencesV1(banned_terms=("synthetic",))
    )
    require_locked_items_unchanged(content, content, global_value)
    altered = content.model_copy(
        update={
            "contact": (
                content.contact[0].model_copy(update={"value": "555-0199"}),
                *content.contact[1:],
            )
        }
    )
    with pytest.raises(DomainValidationError):
        require_locked_items_unchanged(content, altered, ResumePreferencesV1())
    with pytest.raises(DomainValidationError):
        validate_preference_targets(
            content, ResumePreferencesV1(excluded_project_ids=(content.education[0].id,))
        )


@pytest.mark.parametrize("period", ["2025.9 -- 2028.6", "2021.09 - 2025.06", "2024/1\u20132025/12"])
def test_year_month_ranges_are_not_generic_phone_numbers(period):
    content = _parse(SOURCE.read_text()).content
    assert content is not None
    check_model_input_privacy(content, period)
    education = content.education[0].model_copy(
        update={"period": content.education[0].period.model_copy(update={"text": period})}
    )
    content.model_copy(update={"education": (education,)}).model_projection()
    with pytest.raises(DomainValidationError):
        check_model_input_privacy(content, period + " phone: +86 13900001111")


def test_date_exception_does_not_hide_known_contact_digits():
    content = _parse(SOURCE.read_text()).content
    assert content is not None
    contact = tuple(
        field.model_copy(update={"value": "2025 9 2028 6"}) if field.kind == "phone" else field
        for field in content.contact
    )
    protected = content.model_copy(update={"contact": contact})
    with pytest.raises(DomainValidationError):
        check_model_input_privacy(protected, "2025.9 -- 2028.6")
