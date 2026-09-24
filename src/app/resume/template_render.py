"""Deterministic TeX rendering for the approved, privately imported template."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.domain.errors import DomainValidationError
from app.domain.resume_profile import (
    ResumeContentV1,
    ResumePreferencesV1,
    StyledTextV1,
    hard_constraint_issues,
    require_locked_items_unchanged,
)
from app.domain.resume_templates import TemplateManifest, TemplateRenderer, TemplateRenderResult
from app.resume.template_import import (
    FIXED_PREAMBLE_SHA256,
    FIXED_SOURCE_SHA256,
    TEMPLATE_COMMIT,
    parse_resume_source,
)

RENDERER_VERSION = "resume-tex-v1"
MAX_TEX_BYTES = 512 * 1024
_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_COLON = "\uff1a"


@dataclass(frozen=True)
class TemplateIdentity:
    commit: str = TEMPLATE_COMMIT
    source_sha256: str = FIXED_SOURCE_SHA256
    preamble_sha256: str = FIXED_PREAMBLE_SHA256


@dataclass(frozen=True)
class RenderedTex:
    tex_bytes: bytes
    tex_sha256: str
    content_sha256: str
    config_sha256: str
    identity: TemplateIdentity


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: str) -> str:
    result = []
    for char in value:
        if char in "\r\n\t\u2028\u2029":
            result.append(" ")
        elif ord(char) < 32 or ord(char) == 127 or 0xD800 <= ord(char) <= 0xDFFF:
            raise DomainValidationError("resume text contains an unsupported character")
        else:
            result.append(_ESCAPES.get(char, char))
    return "".join(result)


def _styled(value: StyledTextV1) -> str:
    spans = []
    for span in sorted(value.emphasis, key=lambda item: (item.start, item.end)):
        if spans and span.start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], span.end))
        else:
            spans.append((span.start, span.end))
    result = []
    offset = 0
    for start, end in spans:
        result.append(_text(value.text[offset:start]))
        result.append(r"\textbf{" + _text(value.text[start:end]) + "}")
        offset = end
    result.append(_text(value.text[offset:]))
    return "".join(result)


def _section_education(content: ResumeContentV1) -> list[str]:
    lines = [r"\resumeEducationListStart"]
    for entry in content.education:
        lines.append(
            r"\resumeEducationHeading"
            + "{"
            + _styled(entry.institution)
            + "}"
            + "{"
            + _styled(entry.qualification)
            + "}"
            + "{"
            + _styled(entry.period)
            + "}"
        )
    return [*lines, r"\resumeEducationListEnd"]


def _section_projects(content: ResumeContentV1) -> list[str]:
    lines = [r"\vspace{-5pt}", r"\resumeProjectListStart"]
    for entry in content.projects:
        if not entry.bullets:
            raise DomainValidationError("resume project needs at least one bullet")
        technologies = r"\skillgap".join(
            r"\skilltag{" + _text(value) + "}" for value in entry.technologies
        )
        lines.append(
            r"\resumeProjectHeading"
            + "{"
            + _styled(entry.title)
            + "}"
            + "{"
            + _styled(entry.period)
            + "}"
            + "{"
            + technologies
            + "}"
            + "{"
            + _styled(entry.summary)
            + "}"
        )
        lines.append(r"\resumeItemListStart")
        lines.extend(r"\resumeItem{" + _styled(bullet) + "}" for bullet in entry.bullets)
        lines.append(r"\resumeItemListEnd")
    return [*lines, r"\resumeProjectListEnd", r"\vspace{-15pt}"]


def _section_skills(content: ResumeContentV1) -> list[str]:
    lines = [r"\begin{itemize}", r"\normalsize{\item{"]
    for entry in content.skills:
        items = "、".join(_text(item) for item in entry.items)
        lines.append(r"\textbf{" + _text(entry.label) + "}{" + _COLON + items + "} \\\\")
    return [*lines, "}}", r"\end{itemize}", r"\vspace{-16pt}"]


def render_resume_tex(
    *,
    source_bytes: bytes,
    identity: TemplateIdentity,
    profile_content: ResumeContentV1,
    content: ResumeContentV1,
    preferences: ResumePreferencesV1,
) -> RenderedTex:
    """Render text only; callers own authorization and artifact transactions."""
    validate_render_content(profile_content, content, preferences)
    if identity.commit != TEMPLATE_COMMIT:
        raise DomainValidationError("resume template version is unsupported")
    if hashlib.sha256(source_bytes).hexdigest() != identity.source_sha256:
        raise DomainValidationError("resume template source digest is invalid")
    try:
        source = source_bytes.decode("utf-8")
        preamble, _ = source.split(r"\begin{document}", 1)
    except (UnicodeDecodeError, ValueError) as error:
        raise DomainValidationError("resume template source is invalid") from error
    if hashlib.sha256(preamble.encode("utf-8")).hexdigest() != identity.preamble_sha256:
        raise DomainValidationError("resume template preamble is invalid")
    preview = parse_resume_source(
        source,
        expected_source_sha256=identity.source_sha256,
        expected_preamble_sha256=identity.preamble_sha256,
    )
    if not preview.complete:
        raise DomainValidationError("resume template structure is unsupported")
    sections = re.findall(r"\\section\{([^{}\\\n]+)\}", source.split(r"\begin{document}", 1)[1])
    color = re.search(r"\\LARGE\\bfseries\\color\{([A-Za-z]+)\}", source)
    if len(sections) != 3 or color is None:
        raise DomainValidationError("resume template layout is unsupported")
    titles = dict(zip(("education", "projects", "skills"), sections, strict=True))
    name = _text(profile_content.display_name)
    contact = " | ".join(
        _text(field.label) + _COLON + _text(field.value) for field in profile_content.contact
    )
    lines = [
        preamble + r"\begin{document}",
        "",
        r"\begin{center}",
        r"{\LARGE\bfseries\color{" + color.group(1) + "} " + name + r"} \\ \vspace{1pt}",
        r"\large " + contact,
        r"\vspace{-8pt}",
        r"\end{center}",
        "",
    ]
    builders = {
        "education": _section_education,
        "projects": _section_projects,
        "skills": _section_skills,
    }
    for section in preferences.section_order:
        lines.extend((r"\section{" + titles[section] + "}", *builders[section](content), ""))
    lines.append(r"\end{document}")
    rendered = ("\n".join(lines) + "\n").encode("utf-8")
    if len(rendered) > MAX_TEX_BYTES:
        raise DomainValidationError("resume tex exceeds the file limit")
    return RenderedTex(
        tex_bytes=rendered,
        tex_sha256=hashlib.sha256(rendered).hexdigest(),
        content_sha256=_digest(content.model_dump(mode="json")),
        config_sha256=_digest({"section_order": preferences.section_order}),
        identity=identity,
    )


def validate_render_content(
    profile_content: ResumeContentV1,
    content: ResumeContentV1,
    preferences: ResumePreferencesV1,
) -> None:
    try:
        profile_content = ResumeContentV1.model_validate(profile_content.model_dump(mode="json"))
        content = ResumeContentV1.model_validate(content.model_dump(mode="json"))
        preferences = ResumePreferencesV1.model_validate(preferences.model_dump(mode="json"))
    except ValidationError:
        raise DomainValidationError("resume render input is invalid") from None
    require_locked_items_unchanged(profile_content, content, preferences)
    if hard_constraint_issues(content, preferences):
        raise DomainValidationError("resume content violates a hard constraint")
    if any(not project.bullets for project in content.projects):
        raise DomainValidationError("resume project needs at least one bullet")
    all_ids = [
        content.display_name_id,
        *(item.id for item in content.contact),
        *(item.id for item in content.education),
        *(item.id for item in content.projects),
        *(item.id for item in content.skills),
        *(item_id for project in content.projects for item_id in project.bullet_ids),
    ]
    if len(all_ids) != len(set(all_ids)):
        raise DomainValidationError("resume item identities are duplicated")


class FixedTemplateRenderer:
    def render(
        self,
        *,
        source_bytes: bytes,
        manifest: TemplateManifest,
        profile_content: ResumeContentV1,
        content: ResumeContentV1,
        preferences: ResumePreferencesV1,
    ) -> TemplateRenderResult:
        if manifest.renderer_version != RENDERER_VERSION:
            raise DomainValidationError("resume renderer version is unsupported")
        result = render_resume_tex(
            source_bytes=source_bytes,
            identity=TemplateIdentity(
                manifest.commit, manifest.source_sha256, manifest.preamble_sha256
            ),
            profile_content=profile_content,
            content=content,
            preferences=preferences,
        )
        return TemplateRenderResult(
            result.tex_bytes,
            result.tex_sha256,
            result.content_sha256,
            result.config_sha256,
            manifest,
        )


def render_with_template(
    renderer: TemplateRenderer,
    *,
    source_bytes: bytes,
    manifest: TemplateManifest,
    profile_content: ResumeContentV1,
    content: ResumeContentV1,
    preferences: ResumePreferencesV1,
) -> TemplateRenderResult:
    validate_render_content(profile_content, content, preferences)
    if hashlib.sha256(source_bytes).hexdigest() != manifest.source_sha256:
        raise DomainValidationError("resume template source digest is invalid")
    preamble, separator, _ = source_bytes.partition(b"\\begin{document}")
    if not separator or hashlib.sha256(preamble).hexdigest() != manifest.preamble_sha256:
        raise DomainValidationError("resume template preamble is invalid")
    result = renderer.render(
        source_bytes=source_bytes,
        manifest=manifest,
        profile_content=profile_content,
        content=content,
        preferences=preferences,
    )
    if (
        result.manifest != manifest
        or not result.tex_bytes
        or len(result.tex_bytes) > MAX_TEX_BYTES
        or hashlib.sha256(result.tex_bytes).hexdigest() != result.tex_sha256
        or result.content_sha256 != _digest(content.model_dump(mode="json"))
        or result.config_sha256 != _digest({"section_order": preferences.section_order})
    ):
        raise DomainValidationError("resume renderer result identity is invalid")
    return result
