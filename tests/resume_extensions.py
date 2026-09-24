"""Synthetic R6.1 adapters: deliberately absent from production composition."""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.domain.errors import DomainValidationError
from app.domain.resume_artifacts import CompileInstructionsV1
from app.domain.resume_templates import TemplateManifest, TemplateRenderResult
from app.resume.template_render import _digest, _styled, _text

ALTERNATE_SOURCE = (Path(__file__).parent / "fixtures/resume/alternate.tex").read_bytes()
ALTERNATE_MANIFEST = TemplateManifest(
    commit="synthetic-alternate-v1",
    source_sha256=hashlib.sha256(ALTERNATE_SOURCE).hexdigest(),
    preamble_sha256=hashlib.sha256(ALTERNATE_SOURCE.split(b"\\begin{document}", 1)[0]).hexdigest(),
    renderer_version="synthetic-alternate-renderer-v1",
    compile=CompileInstructionsV1(packages=("fontspec",)),
)


class AlternateRenderer:
    def __init__(self):
        self.calls = 0

    def render(self, *, source_bytes, manifest, profile_content, content, preferences):
        self.calls += 1
        if manifest != ALTERNATE_MANIFEST:
            raise DomainValidationError("unsupported test template")
        lines = [source_bytes.split(b"\\begin{document}", 1)[0].decode(), r"\begin{document}"]
        contact = " | ".join(_text(item.value) for item in profile_content.contact)
        lines.append(r"\person{" + _text(profile_content.display_name) + "}{" + contact + "}")
        for section in preferences.section_order:
            lines.append(r"\section*{" + section + "}")
            if section == "education":
                for entry in content.education:
                    lines.append(
                        r"\entry{"
                        + _styled(entry.institution)
                        + "}{"
                        + _styled(entry.qualification)
                        + "}{"
                        + _styled(entry.period)
                        + "}{}"
                    )
            elif section == "projects":
                for entry in content.projects:
                    lines.append(
                        r"\entry{"
                        + _styled(entry.title)
                        + "}{"
                        + _styled(entry.period)
                        + "}{"
                        + _text(", ".join(entry.technologies))
                        + "}{"
                        + _styled(entry.summary)
                        + "}"
                    )
                    lines.extend(r"\point{" + _styled(bullet) + "}" for bullet in entry.bullets)
            elif section == "skills":
                lines.extend(
                    r"\point{" + _text(entry.label) + ": " + _text(", ".join(entry.items)) + "}"
                    for entry in content.skills
                )
        lines.append(r"\end{document}")
        payload = ("\n".join(lines) + "\n").encode()
        return TemplateRenderResult(
            payload,
            hashlib.sha256(payload).hexdigest(),
            _digest(content.model_dump(mode="json")),
            _digest({"section_order": preferences.section_order}),
            manifest,
        )


def alternate_kwargs():
    return {
        "renderer": AlternateRenderer(),
        "template_manifest": ALTERNATE_MANIFEST,
        "template_source": ALTERNATE_SOURCE,
    }
