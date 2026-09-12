from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.retrieval.ingestion import (
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARACTERS,
    MAX_DOCUMENTS_PER_COMMAND,
    IngestionErrorCode,
    IngestionInputError,
    validate_and_load_documents,
)


def _assert_code(paths: list[Path], code: IngestionErrorCode, **kwargs: object) -> str:
    with pytest.raises(IngestionInputError) as raised:
        validate_and_load_documents(paths, **kwargs)
    assert raised.value.code is code
    return str(raised.value)


@pytest.mark.parametrize(
    ("filename", "source_type"),
    [("resume.md", "markdown"), ("notes.txt", "text")],
)
def test_allowed_file_contract_and_deterministic_metadata(
    tmp_path: Path,
    filename: str,
    source_type: str,
) -> None:
    path = tmp_path / filename
    path.write_text("Candidate profile\n", encoding="utf-8")

    first = validate_and_load_documents([path])
    second = validate_and_load_documents([path])

    assert first == second
    assert first.sources[0].source_name == filename
    assert first.sources[0].source_type == source_type
    assert first.sources[0].title == Path(filename).stem
    assert first.sources[0].raw_text == "Candidate profile\n"
    assert first.sources[0].character_count == 18


def test_exact_character_limit_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "exact.txt"
    path.write_text("x" * MAX_DOCUMENT_CHARACTERS, encoding="utf-8")

    source = validate_and_load_documents([path]).sources[0]

    assert source.character_count == MAX_DOCUMENT_CHARACTERS


def test_character_limit_plus_one_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * (MAX_DOCUMENT_CHARACTERS + 1), encoding="utf-8")

    _assert_code([path], IngestionErrorCode.DOCUMENT_TOO_LARGE)


def test_binary_loader_never_requests_more_than_the_byte_bound(tmp_path: Path) -> None:
    path = tmp_path / "bounded.txt"
    path.write_text("placeholder", encoding="utf-8")
    requested_sizes: list[int] = []

    class TrackingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return super().read(size)

    validate_and_load_documents([path], opener=lambda _path: TrackingStream(b"valid"))

    assert requested_sizes == [MAX_DOCUMENT_BYTES + 1]


def test_exact_document_count_is_accepted(tmp_path: Path) -> None:
    paths = []
    for index in range(MAX_DOCUMENTS_PER_COMMAND):
        path = tmp_path / f"{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        paths.append(path)

    assert len(validate_and_load_documents(paths).sources) == MAX_DOCUMENTS_PER_COMMAND


def test_document_count_contract_rejects_empty_and_over_limit(tmp_path: Path) -> None:
    _assert_code([], IngestionErrorCode.NO_DOCUMENTS)
    paths = [tmp_path / f"{index}.txt" for index in range(MAX_DOCUMENTS_PER_COMMAND + 1)]
    _assert_code(paths, IngestionErrorCode.TOO_MANY_DOCUMENTS)


@pytest.mark.parametrize("suffix", [".pdf", ".MD", ".TXT"])
def test_unsupported_extensions_are_rejected(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"resume{suffix}"
    path.write_text("private body", encoding="utf-8")

    message = _assert_code([path], IngestionErrorCode.UNSUPPORTED_EXTENSION)

    assert "private body" not in message


def test_directory_and_missing_file_are_rejected(tmp_path: Path) -> None:
    _assert_code([tmp_path], IngestionErrorCode.NOT_REGULAR_FILE)
    _assert_code([tmp_path / "missing.txt"], IngestionErrorCode.FILE_NOT_FOUND)


def test_leaf_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("private body", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    _assert_code([link], IngestionErrorCode.SYMLINK_NOT_ALLOWED)


def test_symlinked_parent_directory_is_rejected(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "resume.txt").write_text("private body", encoding="utf-8")
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(actual, target_is_directory=True)

    _assert_code(
        [linked_parent / "resume.txt"],
        IngestionErrorCode.SYMLINK_NOT_ALLOWED,
    )


def test_invalid_utf8_empty_and_whitespace_only_are_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    whitespace = tmp_path / "whitespace.md"
    whitespace.write_text(" \r\n\t", encoding="utf-8")

    _assert_code([invalid], IngestionErrorCode.INVALID_UTF8)
    _assert_code([empty], IngestionErrorCode.EMPTY_DOCUMENT)
    _assert_code([whitespace], IngestionErrorCode.EMPTY_DOCUMENT)


def test_file_read_failure_has_stable_safe_error(tmp_path: Path) -> None:
    path = tmp_path / "resume.txt"
    body = "DO NOT DISCLOSE THIS DOCUMENT BODY"
    path.write_text(body, encoding="utf-8")

    def fail_open(_path: Path) -> io.BytesIO:
        raise PermissionError("injected")

    with pytest.raises(IngestionInputError) as raised:
        validate_and_load_documents([path], opener=fail_open)

    assert raised.value.code is IngestionErrorCode.FILE_READ_FAILED
    assert body not in str(raised.value)
    assert body not in repr(raised.value)
    assert raised.value.source_name == "resume.txt"
    assert raised.value.file_index == 1


def test_raw_text_is_preserved_without_step_52_normalization(tmp_path: Path) -> None:
    path = tmp_path / "resume.md"
    raw_text = "# Heading\r\n\r\nBody  \r\n"
    path.write_bytes(raw_text.encode("utf-8"))

    source = validate_and_load_documents([path]).sources[0]

    assert source.raw_text == raw_text
    assert source.character_count == len(raw_text)


def test_batch_fails_closed_when_any_explicit_file_is_invalid(tmp_path: Path) -> None:
    paths = []
    for index in range(9):
        path = tmp_path / f"valid-{index}.md"
        path.write_text("valid", encoding="utf-8")
        paths.append(path)
    invalid = tmp_path / "invalid.pdf"
    invalid.write_text("invalid", encoding="utf-8")

    _assert_code([*paths, invalid], IngestionErrorCode.UNSUPPORTED_EXTENSION)
