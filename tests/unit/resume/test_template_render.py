"""Synthetic-only deterministic rendering and TeX injection boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.domain.errors import DomainValidationError
from app.domain.resume_profile import ResumePreferencesV1, StyledTextV1
from app.resume.template_import import parse_resume_source
from app.resume.template_render import TemplateIdentity, render_resume_tex

SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


def _inputs():
    source = SOURCE.read_bytes()
    preamble = source.split(b"\\begin{document}", 1)[0]
    identity = TemplateIdentity(
        source_sha256=hashlib.sha256(source).hexdigest(),
        preamble_sha256=hashlib.sha256(preamble).hexdigest(),
    )
    preview = parse_resume_source(
        source.decode(),
        expected_source_sha256=identity.source_sha256,
        expected_preamble_sha256=identity.preamble_sha256,
    )
    assert preview.complete and preview.content is not None
    return source, identity, preview.content


def test_render_is_deterministic_and_keeps_approved_preamble_and_structure() -> None:
    source, identity, content = _inputs()
    preferences = ResumePreferencesV1(section_order=("skills", "education", "projects"))
    first = render_resume_tex(
        source_bytes=source,
        identity=identity,
        profile_content=content,
        content=content,
        preferences=preferences,
    )
    second = render_resume_tex(
        source_bytes=source,
        identity=identity,
        profile_content=content,
        content=content,
        preferences=preferences,
    )
    assert first == second
    assert hashlib.sha256(first.tex_bytes).hexdigest() == first.tex_sha256
    assert first.tex_bytes.startswith(source.split(b"\\begin{document}", 1)[0])
    text = first.tex_bytes.decode()
    assert text.index(r"\section{Skills}") < text.index(r"\section{Education}")
    assert text.index(r"\section{Education}") < text.index(r"\section{Projects}")
    assert r"\resumeProjectHeading" in text
    assert r"\textbf{synthetic}" in text
    assert "canary@example.test" in text


def test_special_characters_long_terms_and_local_change_stay_in_target_content() -> None:
    source, identity, content = _inputs()
    original = render_resume_tex(
        source_bytes=source,
        identity=identity,
        profile_content=content,
        content=content,
        preferences=ResumePreferencesV1(),
    ).tex_bytes.decode()
    project = content.projects[0]
    candidate = content.model_copy(
        update={
            "projects": (
                project.model_copy(
                    update={
                        "technologies": ("C++/超长技术名称" * 25, "A_B", "100%"),
                        "bullets": (
                            StyledTextV1(text=r"中文 & 20% $x_1$ {a} \input{secret} ~ ^"),
                            *project.bullets[1:],
                        ),
                    }
                ),
            )
        }
    )
    changed = render_resume_tex(
        source_bytes=source,
        identity=identity,
        profile_content=content,
        content=candidate,
        preferences=ResumePreferencesV1(),
    ).tex_bytes.decode()
    assert r"\input{secret}" not in changed
    assert r"\textbackslash{}input\{secret\}" in changed
    assert r"\&" in changed and r"\%" in changed and r"\$" in changed
    assert r"\_" in changed and r"\textasciitilde{}" in changed
    assert original.split(r"\section{Projects}", 1)[0] == changed.split(r"\section{Projects}", 1)[0]
    assert original.split(r"\section{Skills}", 1)[1] == changed.split(r"\section{Skills}", 1)[1]


def test_invalid_template_protected_fields_constraints_and_controls_fail_closed() -> None:
    source, identity, content = _inputs()
    basic = dict(
        source_bytes=source,
        identity=identity,
        profile_content=content,
        content=content,
        preferences=ResumePreferencesV1(),
    )
    with pytest.raises(DomainValidationError):
        render_resume_tex(**{**basic, "source_bytes": source + b" "})
    changed_contact = content.model_copy(
        update={
            "contact": (
                content.contact[0].model_copy(update={"value": "555-0199"}),
                *content.contact[1:],
            )
        }
    )
    with pytest.raises(DomainValidationError):
        render_resume_tex(**{**basic, "content": changed_contact})
    with pytest.raises(DomainValidationError):
        render_resume_tex(
            **{
                **basic,
                "preferences": ResumePreferencesV1(excluded_project_ids=(content.projects[0].id,)),
            }
        )
    project = content.projects[0]
    with_control = content.model_copy(
        update={
            "projects": (
                project.model_copy(
                    update={
                        "title": StyledTextV1(text="invalid\x00control"),
                    }
                ),
            )
        }
    )
    with pytest.raises(DomainValidationError):
        render_resume_tex(**{**basic, "content": with_control})
