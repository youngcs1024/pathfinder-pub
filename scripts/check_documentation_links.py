#!/usr/bin/env python3
"""Check local links in root/docs Markdown without dependencies or network access.

Supports the repository's inline/image/reference links, ATX/Setext headings and
explicit HTML anchors. This is a small existence check, not a CommonMark renderer;
raw HTML links, external URLs and non-Markdown fragments are not validated.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

_LABEL = r"\[((?:\\.|[^\[\]\\])+)\]"
_DESTINATION = r"<[^>\n]*>|(?:\\.|[^\s()\\]|\([^()\n]*\))+"
_INLINE = re.compile(
    rf"!?{_LABEL}\(\s*({_DESTINATION})?"
    r"""(?:\s+(?:"[^"\n]*"|'[^'\n]*'|\([^\n]*\)))?\s*\)"""
)
_DEFINITION = re.compile(rf"^ {{0,3}}{_LABEL}:\s*({_DESTINATION})", re.MULTILINE)
_REFERENCE = re.compile(rf"!?{_LABEL}(?:\[([^\]\n]*)\])?")
_HTML_ANCHOR = re.compile(r"""<(?:a|h[1-6])\b[^>]*\b(?:id|name)=["']([^"']+)["']""")


def _blank(value: str) -> str:
    return re.sub(r"[^\n]", " ", value)


def _without_blocks(text: str) -> str:
    """Mask examples/comments while preserving offsets for diagnostics."""
    text = re.sub(r"<!--[\s\S]*?-->", lambda match: _blank(match[0]), text)
    result: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is not None:
            result.append(_blank(line))
            if re.fullmatch(rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
                fence = None
        elif marker:
            fence = marker[1]
            result.append(_blank(line))
        elif line.startswith(("    ", "\t")):
            result.append(_blank(line))
        else:
            result.append(line)
    return "".join(result)


def _without_inline_code(text: str) -> str:
    return re.sub(
        r"(?<!`)(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)",
        lambda match: _blank(match[0]),
        text,
    )


def _slug(heading: str) -> str:
    heading = _INLINE.sub(lambda match: match[1], heading)
    heading = re.sub(r"<[^>]*>", "", heading)
    heading = html.unescape(heading).lower().replace("`", "")
    # GitHub-style heading IDs retain letters/numbers/marks, hyphens and underscores.
    heading = "".join(
        char for char in heading if char in " -_" or unicodedata.category(char)[0] in "LNM"
    )
    return heading.replace(" ", "-")


def anchors(text: str) -> set[str]:
    text = _without_blocks(text)
    found = {html.unescape(match[1]) for match in _HTML_ANCHOR.finditer(_without_inline_code(text))}
    generated: set[str] = set()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        heading = match[1] if match else None
        if heading is None and index + 1 < len(lines) and line.strip():
            if re.fullmatch(r" {0,3}(?:=+|-+)\s*", lines[index + 1]):
                heading = line.strip()
        if heading is None:
            continue
        base = _slug(heading)
        slug = base
        suffix = 0
        while slug in generated:
            suffix += 1
            slug = f"{base}-{suffix}"
        generated.add(slug)
    return found | generated


def _reference_key(label: str) -> str:
    return " ".join(label.split()).casefold()


def links(text: str) -> list[tuple[int, str]]:
    """Return (line, destination); undefined references are Markdown plain text."""
    text = _without_inline_code(_without_blocks(text))
    found: list[tuple[int, str]] = []
    definitions: dict[str, str] = {}
    masked = list(text)
    for match in _DEFINITION.finditer(text):
        definitions.setdefault(_reference_key(match[1]), match[2])
        found.append((text.count("\n", 0, match.start()) + 1, match[2]))
        masked[match.start() : match.end()] = _blank(match[0])
    text = "".join(masked)
    for match in _INLINE.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "\\":
            continue
        found.append((text.count("\n", 0, match.start()) + 1, match[2] or ""))
        masked[match.start() : match.end()] = _blank(match[0])
    text = "".join(masked)
    for match in _REFERENCE.finditer(text):
        key = _reference_key(match[2] or match[1])
        if key in definitions:
            found.append((text.count("\n", 0, match.start()) + 1, definitions[key]))
    return found


@dataclass(frozen=True)
class LinkProblem:
    source: Path
    line: int
    target: str
    reason: str

    def __str__(self) -> str:
        return f"{self.source}:{self.line}: {self.reason}: {self.target}"


def check_documentation(repository: Path) -> tuple[int, int, list[LinkProblem]]:
    repository = repository.resolve()
    documents = sorted([*repository.glob("*.md"), *(repository / "docs").rglob("*.md")])
    problems: list[LinkProblem] = []
    anchor_cache: dict[Path, set[str]] = {}
    checked = 0
    for source in documents:
        for line, raw in links(source.read_text(encoding="utf-8")):
            destination = html.unescape(
                re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])", r"\1", raw)
            )
            destination = destination.removeprefix("<").removesuffix(">")
            parsed = urlsplit(destination)
            if parsed.scheme or parsed.netloc:
                continue
            checked += 1
            path = unquote(parsed.path)
            target = source if not path else source.parent / path
            if path.startswith("/"):
                target = repository / path.lstrip("/")
            target = target.resolve()
            reason = None
            if not target.is_relative_to(repository):
                reason = "local target escapes repository"
            elif not target.exists():
                reason = "missing local target"
            elif parsed.fragment and target.suffix.lower() == ".md" and target.is_file():
                if target not in anchor_cache:
                    anchor_cache[target] = anchors(target.read_text(encoding="utf-8"))
                if unquote(parsed.fragment) not in anchor_cache[target]:
                    reason = "missing Markdown anchor"
            if reason:
                problems.append(LinkProblem(source.relative_to(repository), line, raw, reason))
    return len(documents), checked, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        documents, checked, problems = check_documentation(args.repository)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"documentation links: cannot complete check: {exc}", file=sys.stderr)
        return 1
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"documentation links: {documents} files, {checked} local links, {len(problems)} errors")
    if not documents:
        print("documentation links: no documents found", file=sys.stderr)
    return 1 if problems or not documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
