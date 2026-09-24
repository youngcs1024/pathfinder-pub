"""Shared validation applies before every renderer; compile evidence stays synthetic."""

from dataclasses import replace

import pytest

from app.domain.errors import DomainValidationError
from app.domain.resume_profile import ResumePreferencesV1
from app.domain.resume_templates import CompilePreparation, CompileResult
from app.resume.template_render import render_with_template
from tests.resume_extensions import ALTERNATE_MANIFEST, ALTERNATE_SOURCE, AlternateRenderer
from tests.unit.resume.test_template_render import _inputs


def test_alternate_template_reuses_content_and_compilation_binds_source():
    _, _, content = _inputs()
    renderer = AlternateRenderer()
    kwargs = dict(
        source_bytes=ALTERNATE_SOURCE,
        manifest=ALTERNATE_MANIFEST,
        profile_content=content,
        content=content,
        preferences=ResumePreferencesV1(),
    )
    first = render_with_template(renderer, **kwargs)
    assert first == render_with_template(renderer, **kwargs)
    assert b"\\point{" in first.tex_bytes and b"\\resumeItem{" not in first.tex_bytes
    preparation = CompilePreparation(first.tex_sha256, first.manifest)
    result = CompileResult(first.tex_sha256, "success", simulated=True)
    result.require_source(preparation)
    assert preparation.status == "NOT_RUN" and result.simulated
    with pytest.raises(DomainValidationError):
        replace(result, tex_sha256="0" * 64).require_source(preparation)


@pytest.mark.parametrize("problem", ["personal", "lock", "identity", "style", "template"])
def test_alternate_renderer_cannot_skip_input_rules(problem):
    _, _, content = _inputs()
    renderer = AlternateRenderer()
    candidate = content
    preferences = ResumePreferencesV1()
    manifest = ALTERNATE_MANIFEST
    if problem == "personal":
        candidate = content.model_copy(update={"display_name": "Changed"})
    elif problem == "lock":
        project = content.projects[0]
        preferences = ResumePreferencesV1(locked_item_ids=(project.id,))
        candidate = content.model_copy(
            update={"projects": (project.model_copy(update={"technologies": ("Changed",)}),)}
        )
    elif problem == "identity":
        candidate = content.model_copy(update={"display_name_id": content.projects[0].id})
    elif problem == "style":
        project = content.projects[0]
        bullet = project.bullets[0].model_copy(update={"emphasis": ({"start": -1, "end": 99999},)})
        candidate = content.model_copy(
            update={
                "projects": (
                    project.model_copy(update={"bullets": (bullet, *project.bullets[1:])}),
                )
            }
        )
    else:
        manifest = replace(manifest, source_sha256="0" * 64)
    with pytest.raises(DomainValidationError):
        render_with_template(
            renderer,
            source_bytes=ALTERNATE_SOURCE,
            manifest=manifest,
            profile_content=content,
            content=candidate,
            preferences=preferences,
        )
    assert renderer.calls == 0


def test_renderer_bad_result_is_rejected():
    _, _, content = _inputs()

    class Broken(AlternateRenderer):
        def render(self, **kwargs):
            return replace(super().render(**kwargs), tex_sha256="0" * 64)

    with pytest.raises(DomainValidationError, match="result identity"):
        render_with_template(
            Broken(),
            source_bytes=ALTERNATE_SOURCE,
            manifest=ALTERNATE_MANIFEST,
            profile_content=content,
            content=content,
            preferences=ResumePreferencesV1(),
        )
