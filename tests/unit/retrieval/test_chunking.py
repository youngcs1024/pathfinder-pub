from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.retrieval.chunking import (
    CHUNKING_VERSION,
    NORMALIZATION_VERSION,
    normalize_and_chunk_batch,
    normalize_document_content,
)
from app.retrieval.ingestion import (
    MAX_CHUNK_TOKENS,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARACTERS,
    IngestionErrorCode,
    IngestionInputError,
    ValidatedIngestionBatch,
    ValidatedIngestionSource,
    validate_and_load_documents,
)


def _batch(text: str, *, source_type: str = "text") -> ValidatedIngestionBatch:
    return ValidatedIngestionBatch(
        sources=(
            ValidatedIngestionSource(
                source_name="resume.md" if source_type == "markdown" else "resume.txt",
                source_type=source_type,
                title="resume",
                raw_text=text,
                character_count=len(text),
            ),
        )
    )


def _source(text: str, *, source_type: str = "text"):
    return normalize_and_chunk_batch(_batch(text, source_type=source_type)).sources[0]


def _assert_budget(source: object) -> None:
    chunks = source.chunks  # type: ignore[attr-defined]
    assert chunks
    assert all(0 < chunk.token_count <= MAX_CHUNK_TOKENS for chunk in chunks)
    assert all(chunk.token_count == len(chunk.text.encode("utf-8")) for chunk in chunks)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("first\r\nsecond\r\n", "first\nsecond\n"),
        ("first\rsecond\r", "first\nsecond\n"),
        ("first\nsecond\n", "first\nsecond\n"),
    ],
)
def test_newline_normalization(raw: str, expected: str) -> None:
    assert normalize_document_content(raw) == expected


def test_nfc_equivalence_and_normalization_idempotence() -> None:
    decomposed = "Cafe\u0301\r\n"
    composed = "Caf\u00e9\n"

    normalized = normalize_document_content(decomposed)

    assert normalized == normalize_document_content(composed)
    assert normalize_document_content(normalized) == normalized


def test_document_hash_is_over_normalized_content() -> None:
    first = _source("Cafe\u0301\r\n")
    equivalent = _source("Caf\u00e9\n")
    different = _source("Caf\u00e9!\n")

    assert first.content == equivalent.content
    assert first.content_hash == equivalent.content_hash
    assert first.content_hash == hashlib.sha256(first.content.encode("utf-8")).hexdigest()
    assert different.content_hash != first.content_hash
    assert first.normalization_version == NORMALIZATION_VERSION
    assert first.chunking_version == CHUNKING_VERSION


def test_plain_text_single_and_greedily_packed_paragraphs_have_no_section() -> None:
    source = _source("one\n\ntwo\n\nthree")

    assert [chunk.text for chunk in source.chunks] == ["one\n\ntwo\n\nthree"]
    assert [chunk.section for chunk in source.chunks] == [None]


def test_plain_text_paragraphs_split_stably_at_budget() -> None:
    source = _source(f"{'a' * 500}\n\n{'b' * 400}\n\nend")

    assert [chunk.text for chunk in source.chunks] == ["a" * 500, f"{'b' * 400}\n\nend"]
    _assert_budget(source)


def test_markdown_heading_is_section_boundary_and_remains_in_chunk_text() -> None:
    source = _source("preface\n\n# First\n\nbody\n\n## Second\n\nmore", source_type="markdown")

    assert [chunk.section for chunk in source.chunks] == [None, "First", "Second"]
    assert [chunk.text for chunk in source.chunks] == [
        "preface",
        "# First\n\nbody",
        "## Second\n\nmore",
    ]


def test_markdown_never_merges_across_headings_even_under_budget() -> None:
    source = _source("# One\nsmall\n# Two\nsmall", source_type="markdown")

    assert len(source.chunks) == 2
    assert [chunk.section for chunk in source.chunks] == ["One", "Two"]


def test_markdown_consecutive_and_heading_only_sections_are_preserved() -> None:
    source = _source("# One\n## Two\n### Three", source_type="markdown")

    assert [chunk.section for chunk in source.chunks] == ["One", "Two", "Three"]
    assert [chunk.text.rstrip("\n") for chunk in source.chunks] == ["# One", "## Two", "### Three"]


def test_text_source_does_not_parse_markdown_headings() -> None:
    source = _source("# Not a section\n\nbody", source_type="text")

    assert len(source.chunks) == 1
    assert source.chunks[0].section is None


@pytest.mark.parametrize(
    ("text", "expected_counts"),
    [
        ("x" * MAX_CHUNK_TOKENS, [MAX_CHUNK_TOKENS]),
        ("x" * (MAX_CHUNK_TOKENS + 1), [MAX_CHUNK_TOKENS, 1]),
        ("word " * 300, None),
        ("https://example.test/" + "a" * 1600, None),
        ("汉" * 600, None),
        ("🙂" * 300, None),
        ("z" * 1601, None),
        (f"{'a' * 700}\n{'b' * 700}", None),
    ],
)
def test_hard_budget_handles_oversized_content(
    text: str,
    expected_counts: list[int] | None,
) -> None:
    source = _source(text)

    _assert_budget(source)
    if expected_counts is not None:
        assert [chunk.token_count for chunk in source.chunks] == expected_counts


def test_chunk_hash_uses_only_exact_final_text() -> None:
    source = _source("first\n\nsecond")
    chunk = source.chunks[0]

    assert chunk.content_hash == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_ordinals_are_zero_based_gapless_and_retry_is_deterministic() -> None:
    batch = _batch("# A\n" + "x" * 1000 + "\n# B\nbody", source_type="markdown")

    first = normalize_and_chunk_batch(batch)
    second = normalize_and_chunk_batch(batch)

    assert first == second
    chunks = first.sources[0].chunks
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_prepared_repr_does_not_disclose_document_or_chunk_body() -> None:
    canary = "PRIVATE-DOCUMENT-BODY-CANARY"
    prepared = normalize_and_chunk_batch(_batch(canary))

    assert canary not in repr(prepared)
    assert canary not in repr(prepared.sources[0])
    assert canary not in repr(prepared.sources[0].chunks[0])


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (" \r\n\t", IngestionErrorCode.EMPTY_DOCUMENT),
        ("\ud800", IngestionErrorCode.INVALID_UTF8),
        ("x" * (MAX_DOCUMENT_CHARACTERS + 1), IngestionErrorCode.DOCUMENT_TOO_LARGE),
        ("🙂" * ((MAX_DOCUMENT_BYTES // 4) + 1), IngestionErrorCode.DOCUMENT_TOO_LARGE),
    ],
)
def test_hand_constructed_invalid_source_fails_with_safe_typed_error(
    text: str,
    code: IngestionErrorCode,
) -> None:
    with pytest.raises(IngestionInputError) as raised:
        normalize_and_chunk_batch(_batch(text))

    assert raised.value.code is code
    assert text not in str(raised.value)
    assert text not in repr(raised.value)


def test_batch_fails_closed_when_second_source_is_invalid() -> None:
    batch = ValidatedIngestionBatch(sources=(_batch("valid").sources[0], _batch(" \n").sources[0]))

    with pytest.raises(IngestionInputError) as raised:
        normalize_and_chunk_batch(batch)

    assert raised.value.code is IngestionErrorCode.EMPTY_DOCUMENT
    assert raised.value.file_index == 2


def test_step_51_loaded_text_is_the_snapshot_boundary(tmp_path: Path) -> None:
    path = tmp_path / "resume.md"
    captured = "# Profile\r\n\r\nCafe\u0301 experience\r\n"
    path.write_bytes(captured.encode("utf-8"))
    batch = validate_and_load_documents([path])
    assert batch.sources[0].raw_text == captured

    path.write_text("replacement that Step 5.2 must never read", encoding="utf-8")
    prepared = normalize_and_chunk_batch(batch).sources[0]

    assert prepared.content == "# Profile\n\nCaf\u00e9 experience\n"
    assert "replacement" not in prepared.content
    assert prepared.content_hash == hashlib.sha256(prepared.content.encode("utf-8")).hexdigest()
    assert [chunk.text for chunk in prepared.chunks] == ["# Profile\n\nCaf\u00e9 experience\n"]
