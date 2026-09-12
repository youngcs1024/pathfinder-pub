from __future__ import annotations

import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

ALLOWED_SUFFIXES = frozenset({".md", ".txt"})
MAX_DOCUMENTS_PER_COMMAND = 10
MAX_DOCUMENT_CHARACTERS = 100_000
MAX_DOCUMENT_BYTES = MAX_DOCUMENT_CHARACTERS * 4
MAX_CHUNK_TOKENS = 800


class IngestionErrorCode(StrEnum):
    NO_DOCUMENTS = "no_documents"
    TOO_MANY_DOCUMENTS = "too_many_documents"
    FILE_NOT_FOUND = "file_not_found"
    NOT_REGULAR_FILE = "not_regular_file"
    SYMLINK_NOT_ALLOWED = "symlink_not_allowed"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    INVALID_UTF8 = "invalid_utf8"
    EMPTY_DOCUMENT = "empty_document"
    DOCUMENT_TOO_LARGE = "document_too_large"
    WORKSPACE_ACCESS_DENIED = "workspace_access_denied"
    FILE_READ_FAILED = "file_read_failed"


class IngestionInputError(Exception):
    """A safe, stable ingestion-input failure that never includes document text."""

    def __init__(
        self,
        code: IngestionErrorCode,
        *,
        file_index: int | None = None,
        source_name: str | None = None,
    ) -> None:
        self.code = code
        self.file_index = file_index
        self.source_name = source_name
        location = ""
        if file_index is not None:
            location = f" at file argument {file_index}"
        if source_name is not None:
            location += f" ({source_name})"
        super().__init__(f"{code.value}{location}")


@dataclass(frozen=True, slots=True)
class ValidatedIngestionSource:
    source_name: str
    source_type: str
    title: str
    raw_text: str = field(repr=False)
    character_count: int


@dataclass(frozen=True, slots=True)
class ValidatedIngestionBatch:
    sources: tuple[ValidatedIngestionSource, ...]


BinaryOpener = Callable[[Path], BinaryIO]


def _open_binary(path: Path) -> BinaryIO:
    return path.open("rb")


def _error(
    code: IngestionErrorCode,
    *,
    file_index: int,
    path: Path,
) -> IngestionInputError:
    source_name = "".join(character if character.isprintable() else "?" for character in path.name)[
        :255
    ]
    return IngestionInputError(
        code,
        file_index=file_index,
        source_name=source_name or None,
    )


def _lstat_without_symlinks(path: Path, *, file_index: int) -> os.stat_result:
    absolute_path = path if path.is_absolute() else Path.cwd() / path
    parts = absolute_path.parts
    current = Path(parts[0])
    result: os.stat_result | None = None

    for part in parts[1:]:
        current /= part
        try:
            result = os.lstat(current)
        except FileNotFoundError:
            raise _error(
                IngestionErrorCode.FILE_NOT_FOUND,
                file_index=file_index,
                path=path,
            ) from None
        except OSError:
            raise _error(
                IngestionErrorCode.FILE_READ_FAILED,
                file_index=file_index,
                path=path,
            ) from None
        if stat.S_ISLNK(result.st_mode):
            raise _error(
                IngestionErrorCode.SYMLINK_NOT_ALLOWED,
                file_index=file_index,
                path=path,
            )

    if result is None:
        raise _error(
            IngestionErrorCode.NOT_REGULAR_FILE,
            file_index=file_index,
            path=path,
        )
    return result


def _load_source(
    path: Path,
    *,
    file_index: int,
    opener: BinaryOpener,
) -> ValidatedIngestionSource:
    path_stat = _lstat_without_symlinks(path, file_index=file_index)
    if not stat.S_ISREG(path_stat.st_mode):
        raise _error(
            IngestionErrorCode.NOT_REGULAR_FILE,
            file_index=file_index,
            path=path,
        )
    if path.suffix not in ALLOWED_SUFFIXES:
        raise _error(
            IngestionErrorCode.UNSUPPORTED_EXTENSION,
            file_index=file_index,
            path=path,
        )

    try:
        with opener(path) as stream:
            raw_bytes = stream.read(MAX_DOCUMENT_BYTES + 1)
    except OSError:
        raise _error(
            IngestionErrorCode.FILE_READ_FAILED,
            file_index=file_index,
            path=path,
        ) from None

    if len(raw_bytes) > MAX_DOCUMENT_BYTES:
        raise _error(
            IngestionErrorCode.DOCUMENT_TOO_LARGE,
            file_index=file_index,
            path=path,
        )
    try:
        raw_text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _error(
            IngestionErrorCode.INVALID_UTF8,
            file_index=file_index,
            path=path,
        ) from None
    if len(raw_text) > MAX_DOCUMENT_CHARACTERS:
        raise _error(
            IngestionErrorCode.DOCUMENT_TOO_LARGE,
            file_index=file_index,
            path=path,
        )
    if not raw_text.strip():
        raise _error(
            IngestionErrorCode.EMPTY_DOCUMENT,
            file_index=file_index,
            path=path,
        )

    return ValidatedIngestionSource(
        source_name=path.name,
        source_type="markdown" if path.suffix == ".md" else "text",
        title=path.stem,
        raw_text=raw_text,
        character_count=len(raw_text),
    )


def validate_and_load_documents(
    paths: Sequence[str | os.PathLike[str]],
    *,
    opener: BinaryOpener = _open_binary,
) -> ValidatedIngestionBatch:
    """Validate and bounded-load one fail-closed batch of explicitly supplied files."""

    document_paths = tuple(Path(path) for path in paths)
    if not document_paths:
        raise IngestionInputError(IngestionErrorCode.NO_DOCUMENTS)
    if len(document_paths) > MAX_DOCUMENTS_PER_COMMAND:
        raise IngestionInputError(IngestionErrorCode.TOO_MANY_DOCUMENTS)

    sources = tuple(
        _load_source(path, file_index=index, opener=opener)
        for index, path in enumerate(document_paths, start=1)
    )
    return ValidatedIngestionBatch(sources=sources)
