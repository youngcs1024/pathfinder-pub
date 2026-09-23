"""Parse only the approved resume source structure; never execute TeX."""

from __future__ import annotations

import hashlib
import re
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field

from app.domain.resume_profile import (
    ContactFieldV1,
    EducationEntryV1,
    EmphasisSpanV1,
    ProjectEntryV1,
    ResumeClaimV1,
    ResumeContentV1,
    ResumeModel,
    SkillCategoryV1,
    SourceLocationV1,
    StyledTextV1,
)

TEMPLATE_COMMIT = "c056056fcc6f362f8aa9a98f553c5915e4e2eabc"
FIXED_SOURCE_SHA256 = "6c7fd0263b934623bc2e6218b0a1e8b02c70f4c65cb157af68cf9cf0ec276134"
FIXED_PREAMBLE_SHA256 = "a8f0f53ab43e7fb32bf6dba783cc617fb37bb0baedd723746162a08fe61dd2eb"
MAX_SOURCE_BYTES = 131072
_BODY_COMMANDS = frozenset(
    {
        "LARGE",
        "begin",
        "bfseries",
        "color",
        "end",
        "item",
        "large",
        "normalsize",
        "resumeEducationHeading",
        "resumeEducationListEnd",
        "resumeEducationListStart",
        "resumeItem",
        "resumeItemListEnd",
        "resumeItemListStart",
        "resumeProjectHeading",
        "resumeProjectListEnd",
        "resumeProjectListStart",
        "section",
        "skillgap",
        "skilltag",
        "textbf",
        "vspace",
    }
)


class ParseIssueV1(ResumeModel):
    code: str = Field(min_length=1, max_length=80)
    line: int = Field(ge=1)
    column: int = Field(ge=1)


class ResumeImportPreviewV1(ResumeModel):
    template_commit: str = TEMPLATE_COMMIT
    source_sha256: str
    complete: bool
    content: ResumeContentV1 | None
    claims: tuple[ResumeClaimV1, ...] = ()
    issues: tuple[ParseIssueV1, ...] = ()


class _ParseFailure(Exception):
    def __init__(self, code: str, offset: int) -> None:
        self.code = code
        self.offset = offset


def _position(source: str, offset: int) -> tuple[int, int]:
    return source.count("\n", 0, offset) + 1, offset - source.rfind("\n", 0, offset)


def _location(source: str, start: int, end: int) -> SourceLocationV1:
    start_line, start_column = _position(source, start)
    end_line, end_column = _position(source, end)
    return SourceLocationV1(
        start_line=start_line,
        start_column=start_column,
        end_line=end_line,
        end_column=end_column,
    )


def _group(source: str, offset: int) -> tuple[str, int, int]:
    while offset < len(source) and source[offset].isspace():
        offset += 1
    if offset >= len(source) or source[offset] != "{":
        raise _ParseFailure("expected_group", offset)
    start, depth, index = offset, 1, offset + 1
    while index < len(source):
        char = source[index]
        if char == "\\" and index + 1 < len(source) and source[index + 1] in "{}%&#_$":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index], start, index + 1
        index += 1
    raise _ParseFailure("unclosed_group", start)


def _macro_groups(source: str, match: re.Match[str], count: int) -> list[tuple[str, int, int]]:
    offset = match.end()
    groups = []
    for _ in range(count):
        value, start, offset = _group(source, offset)
        groups.append((value, start, offset))
    return groups


def _styled(value: str, offset: int) -> StyledTextV1:
    result: list[str] = []
    spans: list[EmphasisSpanV1] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            result.append(value[index])
            index += 1
            continue
        if index + 1 < len(value) and value[index + 1] in "%&#_{}$":
            result.append(value[index + 1])
            index += 2
            continue
        match = re.match(r"\\([A-Za-z]+)", value[index:])
        if match is None:
            raise _ParseFailure("unsupported_inline_command", offset + index)
        command = match.group(1)
        if command in {"textbf", "skilltag"}:
            inner, _, end = _group(value, index + len(match.group()))
            rendered = _styled(inner, offset + index + len(match.group()) + 1)
            begin = len("".join(result))
            result.append(rendered.text)
            if command == "textbf":
                spans.append(EmphasisSpanV1(start=begin, end=begin + len(rendered.text)))
            spans.extend(
                EmphasisSpanV1(start=begin + s.start, end=begin + s.end) for s in rendered.emphasis
            )
            index = end
        elif command == "skillgap":
            result.append(", ")
            index += len(match.group())
        else:
            raise _ParseFailure("unsupported_inline_command", offset + index)
    cleaned = "".join(result).strip()
    if not cleaned:
        raise _ParseFailure("empty_content", offset)
    # All supported source text uses trim-only normalization. Emphasis spans
    # are adjusted for the leading whitespace removed by strip().
    leading = len("".join(result)) - len("".join(result).lstrip())
    return StyledTextV1(
        text=cleaned,
        emphasis=tuple(
            EmphasisSpanV1(start=s.start - leading, end=s.end - leading)
            for s in spans
            if s.start >= leading and s.end - leading <= len(cleaned)
        ),
    )


def _id(digest: str, path: str):
    return uuid5(NAMESPACE_URL, f"pathfinder-resume:{digest}:{path}")


def _issue(source: str, code: str, offset: int) -> ParseIssueV1:
    line, column = _position(source, max(0, min(offset, len(source))))
    return ParseIssueV1(code=code, line=line, column=column)


def _check_body_coverage(body: str, covered: list[bool], body_start: int) -> None:
    """Reject extra prose even when it uses no TeX command."""
    layout = re.compile(
        r"\\(?:begin|end)\{(?:document|center|itemize)\}(?:\[[^\]]*\])?"
        r"|\\section\{[^{}]+\}"
        r"|\\resume(?:Education|Project|Item)List(?:Start|End)"
        r"|\\vspace\{[^{}]+\}|\\normalsize|\\item|\\\\"
    )
    offset = 0
    for line in body.splitlines(keepends=True):
        remainder = "".join(" " if covered[offset + i] else char for i, char in enumerate(line))
        if remainder.lstrip().startswith("%"):
            offset += len(line)
            continue
        remainder = layout.sub("", remainder).strip(" \t\r\n{}")
        if remainder:
            original = next(
                (
                    i
                    for i, char in enumerate(line)
                    if not covered[offset + i] and not char.isspace()
                ),
                0,
            )
            raise _ParseFailure("unsupported_structure", body_start + offset + original)
        offset += len(line)


def parse_resume_source(
    source: str,
    *,
    expected_source_sha256: str = FIXED_SOURCE_SHA256,
    expected_preamble_sha256: str = FIXED_PREAMBLE_SHA256,
) -> ResumeImportPreviewV1:
    """Preview a frozen source. Tests may inject a synthetic source identity."""
    raw = source.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_SOURCE_BYTES:
        return ResumeImportPreviewV1(
            source_sha256=digest,
            complete=False,
            content=None,
            issues=(_issue(source, "source_too_large", 0),),
        )
    marker = "\\begin{document}"
    if marker not in source:
        return ResumeImportPreviewV1(
            source_sha256=digest,
            complete=False,
            content=None,
            issues=(_issue(source, "document_missing", 0),),
        )
    preamble, body = source.split(marker, 1)
    if (
        digest != expected_source_sha256
        or hashlib.sha256(preamble.encode("utf-8")).hexdigest() != expected_preamble_sha256
    ):
        return ResumeImportPreviewV1(
            source_sha256=digest,
            complete=False,
            content=None,
            issues=(_issue(source, "source_identity_mismatch", 0),),
        )
    body_start = len(preamble) + len(marker)
    issues = []
    for match in re.finditer(r"\\([A-Za-z]+|.)", body):
        if match.group(1) not in _BODY_COMMANDS | {"%", "\\"}:
            issues.append(_issue(source, "unsupported_command", body_start + match.start()))
    for match in re.finditer(r"\\(?:begin|end)\{([^}]+)\}", body):
        if match.group(1) not in {"document", "center", "itemize"}:
            issues.append(_issue(source, "unsupported_environment", body_start + match.start()))
    if issues:
        return ResumeImportPreviewV1(
            source_sha256=digest, complete=False, content=None, issues=tuple(issues)
        )
    try:
        covered = [False] * len(body)

        def mark(start: int, end: int) -> None:
            covered[start:end] = [True] * (end - start)

        def mark_line(start: int) -> None:
            line_start = body.rfind("\n", 0, start) + 1
            line_end = body.find("\n", start)
            mark(line_start, len(body) if line_end < 0 else line_end)

        sections = list(re.finditer(r"\\section\{[^{}]+\}", body))
        if len(sections) != 3:
            raise _ParseFailure("section_structure", body_start)
        name_match = re.search(r"\\LARGE\\bfseries\\color\{[^{}]+\}\s*([^{}\\]+)", body)
        contact_match = re.search(r"\\large\s+([^\n]+)", body)
        if name_match is None or contact_match is None:
            raise _ParseFailure("basic_info_missing", body_start)
        if (
            name_match.start() >= sections[0].start()
            or contact_match.start() >= sections[0].start()
        ):
            raise _ParseFailure("basic_info_position", body_start)
        mark_line(name_match.start())
        mark_line(contact_match.start())
        name = name_match.group(1).strip()
        contact = []
        contact_offset = body_start + contact_match.start(1)
        for ordinal, segment in enumerate(contact_match.group(1).split("|")):
            label, separator, value = segment.strip().partition("\uff1a")
            value = value.strip().strip("{}")
            if not separator or not value:
                raise _ParseFailure("contact_structure", contact_offset)
            kind = "email" if "@" in value else "phone" if re.search(r"\d", value) else "other"
            contact.append(
                ContactFieldV1(
                    id=_id(digest, f"contact/{ordinal}"),
                    kind=kind,
                    label=label,
                    value=value,
                    source=_location(source, contact_offset, contact_offset + len(segment)),
                )
            )
            contact_offset += len(segment) + 1
        education = []
        for ordinal, match in enumerate(re.finditer(r"\\resumeEducationHeading(?![A-Za-z])", body)):
            groups = _macro_groups(body, match, 3)
            if not sections[0].end() < match.start() < groups[-1][2] < sections[1].start():
                raise _ParseFailure("education_position", body_start + match.start())
            mark(match.start(), groups[-1][2])
            education.append(
                EducationEntryV1(
                    id=_id(digest, f"education/{ordinal}"),
                    institution=_styled(groups[0][0], body_start + groups[0][1] + 1),
                    qualification=_styled(groups[1][0], body_start + groups[1][1] + 1),
                    period=_styled(groups[2][0], body_start + groups[2][1] + 1),
                    source=_location(
                        source, body_start + match.start(), body_start + groups[-1][2]
                    ),
                )
            )
        projects = []
        claims = []
        project_matches = list(re.finditer(r"\\resumeProjectHeading(?![A-Za-z])", body))
        project_end = body.find("\\resumeProjectListEnd")
        if project_end < 0:
            raise _ParseFailure("project_list_missing", body_start + sections[1].start())
        for ordinal, match in enumerate(project_matches):
            groups = _macro_groups(body, match, 4)
            if not sections[1].end() < match.start() < groups[-1][2] < sections[2].start():
                raise _ParseFailure("project_position", body_start + match.start())
            mark(match.start(), groups[-1][2])
            project_id = _id(digest, f"project/{ordinal}")
            boundary = (
                project_matches[ordinal + 1].start()
                if ordinal + 1 < len(project_matches)
                else project_end
            )
            if boundary < groups[-1][2]:
                raise _ParseFailure("project_structure", body_start + match.start())
            bullets = []
            bullet_ids = []
            for bullet_ordinal, bullet_match in enumerate(
                re.finditer(r"\\resumeItem(?![A-Za-z])", body[groups[-1][2] : boundary])
            ):
                absolute_match = re.compile(r"\\resumeItem(?![A-Za-z])").search(
                    body, groups[-1][2] + bullet_match.start()
                )
                assert absolute_match is not None
                bullet_group = _macro_groups(body, absolute_match, 1)[0]
                mark(absolute_match.start(), bullet_group[2])
                bullet = _styled(bullet_group[0], body_start + bullet_group[1] + 1)
                bullet_id = _id(digest, f"project/{ordinal}/bullet/{bullet_ordinal}")
                bullets.append(bullet)
                bullet_ids.append(bullet_id)
                claims.append(
                    ResumeClaimV1(
                        id=_id(digest, f"claim/{ordinal}/bullet/{bullet_ordinal}"),
                        project_item_id=project_id,
                        field="bullet",
                        item_id=bullet_id,
                        text=bullet.text,
                        source=_location(
                            source,
                            body_start + absolute_match.start(),
                            body_start + bullet_group[2],
                        ),
                    )
                )
            if not bullets:
                raise _ParseFailure("project_bullets_missing", body_start + match.start())
            values = [_styled(group[0], body_start + group[1] + 1) for group in groups]
            technologies = tuple(part.strip() for part in values[2].text.split(",") if part.strip())
            projects.append(
                ProjectEntryV1(
                    id=project_id,
                    title=values[0],
                    period=values[1],
                    technologies=technologies,
                    summary=values[3],
                    bullets=tuple(bullets),
                    bullet_ids=tuple(bullet_ids),
                    source=_location(source, body_start + match.start(), body_start + boundary),
                )
            )
            for field, value in (
                ("title", values[0].text),
                ("period", values[1].text),
                ("technologies", values[2].text),
                ("summary", values[3].text),
            ):
                claims.append(
                    ResumeClaimV1(
                        id=_id(digest, f"claim/{ordinal}/{field}"),
                        project_item_id=project_id,
                        field=field,
                        item_id=project_id,
                        text=value,
                        source=_location(
                            source, body_start + match.start(), body_start + groups[-1][2]
                        ),
                    )
                )
        skills_region = body[sections[2].end() :]
        skills = []
        for ordinal, match in enumerate(
            re.finditer(r"\\textbf\{[^{}]+\}\{[^{}]+\}", skills_region)
        ):
            command = re.compile(r"\\textbf").match(skills_region, match.start())
            assert command is not None
            groups = _macro_groups(skills_region, command, 2)
            mark(sections[2].end() + match.start(), sections[2].end() + match.end())
            label, items = groups[0][0].strip(), groups[1][0].lstrip("\uff1a: ").strip()
            skills.append(
                SkillCategoryV1(
                    id=_id(digest, f"skill/{ordinal}"),
                    label=label,
                    items=tuple(
                        part.strip() for part in re.split("[\u3001\uff0c]", items) if part.strip()
                    ),
                    source=_location(
                        source,
                        body_start + sections[2].end() + match.start(),
                        body_start + sections[2].end() + match.end(),
                    ),
                )
            )
        if not education or not projects or not skills:
            raise _ParseFailure("content_section_missing", body_start)
        _check_body_coverage(body, covered, body_start)
        content = ResumeContentV1(
            display_name_id=_id(digest, "display_name"),
            display_name=name,
            contact=tuple(contact),
            education=tuple(education),
            projects=tuple(projects),
            skills=tuple(skills),
        )
    except (_ParseFailure, ValueError) as error:
        code = error.code if isinstance(error, _ParseFailure) else "invalid_content"
        offset = error.offset if isinstance(error, _ParseFailure) else body_start
        return ResumeImportPreviewV1(
            source_sha256=digest,
            complete=False,
            content=None,
            issues=(_issue(source, code, offset),),
        )
    return ResumeImportPreviewV1(
        source_sha256=digest, complete=True, content=content, claims=tuple(claims)
    )
