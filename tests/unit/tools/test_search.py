from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.tools.search import (
    SearchDeadlineExceededError,
    SearchInputError,
    SearchResponseError,
    SearchResult,
    canonicalize_search_url,
    normalize_search_result,
    search_source_id,
    validate_search_call,
)


def test_search_result_is_strict_frozen_and_json_serializable() -> None:
    result = normalize_search_result(
        title="Synthetic role",
        url="https://example.test/jobs/backend",
        snippet="Recorded evidence.",
        published_at="2026-08-12T12:00:00Z",
    )

    assert json.loads(result.model_dump_json()) == {
        "source_id": search_source_id("https://example.test/jobs/backend"),
        "title": "Synthetic role",
        "url": "https://example.test/jobs/backend",
        "snippet": "Recorded evidence.",
        "published_at": "2026-08-12T12:00:00Z",
        "truncated": False,
    }
    with pytest.raises(ValidationError, match="frozen"):
        result.title = "changed"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SearchResult.model_validate({**result.model_dump(), "workspace_id": "forged"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "HTTPS://EXAMPLE.COM:443/jobs/backend#details",
            "https://example.com/jobs/backend",
        ),
        ("http://Example.COM:80", "http://example.com/"),
        (
            "https://münich.example/岗位?q=后端&order=1",
            "https://xn--mnich-kva.example/%E5%B2%97%E4%BD%8D?q=%E5%90%8E%E7%AB%AF&order=1",
        ),
        (
            "https://example.com/redirect?url=https%3A%2F%2Ftarget.test%2Fx#ignored",
            "https://example.com/redirect?url=https%3A%2F%2Ftarget.test%2Fx",
        ),
    ],
)
def test_canonicalize_search_url_has_locked_policy(raw: str, expected: str) -> None:
    assert canonicalize_search_url(raw) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.test/file",
        "javascript:alert(1)",
        "https://user:password@example.test/private",
        "https:///missing-host",
        "https://example.test:bad/path",
    ],
)
def test_canonicalize_search_url_rejects_untrusted_or_invalid_urls(url: str) -> None:
    with pytest.raises(SearchResponseError) as captured:
        canonicalize_search_url(url)

    assert str(captured.value) == "search failed: invalid_provider_response"


def test_source_id_is_stable_for_equivalent_urls_and_query_order_is_semantic() -> None:
    first = search_source_id("HTTPS://EXAMPLE.TEST:443/path#fragment")
    equivalent = search_source_id("https://example.test/path")
    reordered_query = search_source_id("https://example.test/path?b=2&a=1")
    original_query = search_source_id("https://example.test/path?a=1&b=2")

    assert first == equivalent
    assert first.startswith("web-v1:")
    assert len(first) == len("web-v1:") + 64
    assert reordered_query != original_query


def test_normalize_search_result_trims_and_truncates_bounded_text() -> None:
    result = normalize_search_result(
        title=f"  {'T' * 501}  ",
        url="https://example.test/long",
        snippet=f"  {'S' * 4_001}  ",
        published_at="  2026-08-12  ",
    )

    assert len(result.title) == 500
    assert len(result.snippet) == 4_000
    assert result.published_at == "2026-08-12"
    assert result.truncated is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": " "},
        {"title": 1},
        {"snippet": None},
        {"published_at": " "},
        {"published_at": "x" * 65},
    ],
)
def test_normalize_search_result_rejects_malformed_provider_fields(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "title": "Title",
        "url": "https://example.test/",
        "snippet": "Snippet",
        "published_at": None,
    }
    values.update(overrides)

    with pytest.raises(SearchResponseError):
        normalize_search_result(**values)  # type: ignore[arg-type]


def test_search_result_rejects_source_id_that_does_not_match_url() -> None:
    valid = normalize_search_result(
        title="Title",
        url="https://example.test/",
        snippet="Snippet",
    )

    with pytest.raises(ValidationError, match="source identity"):
        SearchResult.model_validate(
            {**valid.model_dump(), "source_id": "web-v1:" + "0" * 64},
            strict=True,
        )


def test_validate_search_call_returns_remaining_deadline() -> None:
    query, max_results, remaining = validate_search_call(
        "backend roles",
        5,
        13.5,
        clock=lambda: 10.0,
    )

    assert (query, max_results, remaining) == ("backend roles", 5, 3.5)


@pytest.mark.parametrize(
    ("query", "max_results", "deadline"),
    [
        ("", 1, 11.0),
        ("x" * 2_001, 1, 11.0),
        ("query", 0, 11.0),
        ("query", 21, 11.0),
        ("query", True, 11.0),
        ("query", 1, float("nan")),
    ],
)
def test_validate_search_call_rejects_invalid_inputs(
    query: object,
    max_results: object,
    deadline: object,
) -> None:
    with pytest.raises(SearchInputError):
        validate_search_call(query, max_results, deadline, clock=lambda: 10.0)


def test_validate_search_call_rejects_expired_deadline() -> None:
    with pytest.raises(SearchDeadlineExceededError):
        validate_search_call("query", 1, 10.0, clock=lambda: 10.0)
