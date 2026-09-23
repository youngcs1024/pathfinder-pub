from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from app.retrieval.ingestion import (
    MAX_CHUNK_TOKENS,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENTS_PER_COMMAND,
    IngestionErrorCode,
    IngestionInputError,
    ValidatedIngestionBatch,
    ValidatedIngestionSource,
)

NORMALIZATION_VERSION = "nfc-lf-v1"
CHUNKING_VERSION = "heading-paragraph-utf8-budget-800-v1"

_ATX_HEADING = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(?P<title>.*?))?[ \t]*$")
_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n+")


@dataclass(frozen=True, slots=True)
class PreparedIngestionChunk:
    ordinal: int
    section: str | None
    text: str = field(repr=False)
    content_hash: str
    token_count: int
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedIngestionSource:
    source_name: str
    source_type: str
    title: str
    content: str = field(repr=False)
    content_hash: str
    normalization_version: str
    chunking_version: str
    chunks: tuple[PreparedIngestionChunk, ...]


@dataclass(frozen=True, slots=True)
class PreparedIngestionBatch:
    sources: tuple[PreparedIngestionSource, ...]


def normalize_document_content(raw_text: str) -> str:
    """Canonicalize newlines and Unicode without otherwise rewriting document text."""

    return unicodedata.normalize("NFC", raw_text.replace("\r\n", "\n").replace("\r", "\n"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _budget_count(text: str) -> int:
    """Return the v1 local UTF-8 budget count, not provider billing token usage."""

    return len(text.encode("utf-8"))


def _safe_source_name(source_name: str) -> str | None:
    safe = "".join(character if character.isprintable() else "?" for character in source_name)
    return safe[:255] or None


def _input_error(
    code: IngestionErrorCode,
    *,
    source: ValidatedIngestionSource,
    file_index: int,
) -> IngestionInputError:
    return IngestionInputError(
        code,
        file_index=file_index,
        source_name=_safe_source_name(source.source_name),
    )


def _validate_and_normalize_source(
    source: ValidatedIngestionSource,
    *,
    file_index: int,
) -> str:
    raw_text = source.raw_text
    if len(raw_text) > MAX_DOCUMENT_CHARACTERS:
        raise _input_error(
            IngestionErrorCode.DOCUMENT_TOO_LARGE,
            source=source,
            file_index=file_index,
        )
    try:
        raw_bytes = raw_text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise _input_error(
            IngestionErrorCode.INVALID_UTF8,
            source=source,
            file_index=file_index,
        ) from None
    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise _input_error(
            IngestionErrorCode.DOCUMENT_TOO_LARGE,
            source=source,
            file_index=file_index,
        )

    content = normalize_document_content(raw_text)
    if not content.strip():
        raise _input_error(
            IngestionErrorCode.EMPTY_DOCUMENT,
            source=source,
            file_index=file_index,
        )
    return content


def _heading_title(line: str) -> str | None:
    match = _ATX_HEADING.fullmatch(line)
    if match is None:
        return None
    title = (match.group("title") or "").strip()
    title = re.sub(r"[ \t]+#+[ \t]*$", "", title).strip()
    return title


def _markdown_sections(content: str) -> tuple[tuple[str | None, str], ...]:
    sections: list[tuple[str | None, str]] = []
    section: str | None = None
    accumulated: list[str] = []

    for line in content.splitlines(keepends=True):
        heading = _heading_title(line.removesuffix("\n"))
        if heading is not None:
            if accumulated and "".join(accumulated).strip():
                sections.append((section, "".join(accumulated)))
            accumulated = [line]
            section = heading
        else:
            accumulated.append(line)

    if accumulated and "".join(accumulated).strip():
        sections.append((section, "".join(accumulated)))
    return tuple(sections)


def _max_prefix_index(text: str) -> int:
    byte_count = 0
    for index, character in enumerate(text):
        next_count = byte_count + len(character.encode("utf-8"))
        if next_count > MAX_CHUNK_TOKENS:
            return index
        byte_count = next_count
    return len(text)


def _split_oversized(text: str) -> tuple[str, ...]:
    pieces: list[str] = []
    remaining = text
    while _budget_count(remaining) > MAX_CHUNK_TOKENS:
        limit = _max_prefix_index(remaining)
        prefix = remaining[:limit]

        line_boundary = prefix.rfind("\n")
        if line_boundary >= 0:
            split_at = line_boundary + 1
        else:
            whitespace_boundaries = [
                index + 1 for index, character in enumerate(prefix) if character.isspace()
            ]
            split_at = whitespace_boundaries[-1] if whitespace_boundaries else limit

        pieces.append(remaining[:split_at])
        remaining = remaining[split_at:]

    if remaining:
        pieces.append(remaining)
    return tuple(pieces)


def _chunk_section(section: str | None, text: str) -> list[tuple[str | None, str]]:
    paragraphs = tuple(paragraph for paragraph in _PARAGRAPH_BOUNDARY.split(text) if paragraph)
    chunks: list[tuple[str | None, str]] = []
    current = ""

    for paragraph_index, paragraph in enumerate(paragraphs):
        pieces = _split_oversized(paragraph)
        for piece_index, piece in enumerate(pieces):
            separator = "\n\n" if paragraph_index > 0 and piece_index == 0 else ""
            candidate = f"{current}{separator}{piece}" if current else piece
            if _budget_count(candidate) <= MAX_CHUNK_TOKENS:
                current = candidate
                continue
            if current:
                chunks.append((section, current))
            current = piece

    if current:
        chunks.append((section, current))
    return chunks


def _chunk_content(content: str, *, source_type: str) -> tuple[PreparedIngestionChunk, ...]:
    structural_sections = (
        _markdown_sections(content) if source_type == "markdown" else ((None, content),)
    )
    final_chunks: list[tuple[str | None, str]] = []
    for section, section_text in structural_sections:
        final_chunks.extend(_chunk_section(section, section_text))

    return tuple(
        PreparedIngestionChunk(
            ordinal=ordinal,
            section=section,
            text=text,
            content_hash=_sha256(text),
            token_count=_budget_count(text),
        )
        for ordinal, (section, text) in enumerate(final_chunks)
    )


def normalize_and_chunk_batch(batch: ValidatedIngestionBatch) -> PreparedIngestionBatch:
    """Prepare a deterministic immutable batch using only the captured Step 5.1 text.

    ``token_count`` is a conservative local UTF-8 byte budget for this chunking profile.
    It exists only to enforce the deterministic upper bound and is not Qwen/provider
    billing token usage.
    """

    if not batch.sources:
        raise IngestionInputError(IngestionErrorCode.NO_DOCUMENTS)
    if len(batch.sources) > MAX_DOCUMENTS_PER_COMMAND:
        raise IngestionInputError(IngestionErrorCode.TOO_MANY_DOCUMENTS)

    prepared_sources: list[PreparedIngestionSource] = []
    for file_index, source in enumerate(batch.sources, start=1):
        content = _validate_and_normalize_source(source, file_index=file_index)
        chunks = _chunk_content(content, source_type=source.source_type)
        prepared_sources.append(
            PreparedIngestionSource(
                source_name=source.source_name,
                source_type=source.source_type,
                title=source.title,
                content=content,
                content_hash=_sha256(content),
                normalization_version=NORMALIZATION_VERSION,
                chunking_version=CHUNKING_VERSION,
                chunks=chunks,
            )
        )
    return PreparedIngestionBatch(sources=tuple(prepared_sources))
