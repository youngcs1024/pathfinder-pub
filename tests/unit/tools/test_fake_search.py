from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from app.tools.fake_search import FakeSearch, FakeSearchFailure
from app.tools.search import (
    SearchDeadlineExceededError,
    SearchInputError,
    SearchPermanentError,
    SearchPort,
    SearchResult,
    SearchTransientError,
    normalize_search_result,
)

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "tools" / "search_results_v1.json"


def _recorded_cases() -> dict[str, dict[str, object]]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    return {case["case_id"]: case for case in payload["cases"]}


def _recorded_results(case_id: str) -> tuple[SearchResult, ...]:
    case = _recorded_cases()[case_id]
    raw_results = cast(list[dict[str, object]], case["results"])
    return tuple(
        normalize_search_result(
            title=result["title"],
            url=result["url"],
            snippet=result["snippet"],
            published_at=result.get("published_at"),
        )
        for result in raw_results
    )


def test_recorded_fixture_has_locked_gate_2_2_cases() -> None:
    assert set(_recorded_cases()) == {
        "normal",
        "empty",
        "duplicate_url",
        "unicode",
        "truncated",
    }


@pytest.mark.asyncio
async def test_fake_search_implements_port_copies_fixtures_and_limits_results() -> None:
    fixture_values = list(_recorded_results("normal"))
    search = FakeSearch({"backend roles": fixture_values}, clock=lambda: 10.0)
    fixture_values.clear()

    first = await search.search("backend roles", max_results=1, deadline=11.0)
    all_results = await search.search("backend roles", max_results=5, deadline=11.0)

    assert isinstance(search, SearchPort)
    assert len(first) == 1
    assert len(all_results) == 2
    assert first[0] is not all_results[0]


@pytest.mark.asyncio
async def test_fake_search_deduplicates_canonical_urls_in_provider_rank_order() -> None:
    results = _recorded_results("duplicate_url")
    search = FakeSearch({"duplicate": results}, clock=lambda: 10.0)

    output = await search.search("duplicate", max_results=20, deadline=11.0)

    assert len(output) == 1
    assert output[0].title == "First ranked duplicate"
    assert str(output[0].url) == "https://example.test/jobs/backend?ref=one"


@pytest.mark.asyncio
async def test_fake_search_preserves_unicode_and_recorded_truncation() -> None:
    unicode_search = FakeSearch(
        {"unicode": _recorded_results("unicode")},
        clock=lambda: 10.0,
    )
    truncated_search = FakeSearch(
        {"truncated": _recorded_results("truncated")},
        clock=lambda: 10.0,
    )

    unicode_result = (await unicode_search.search("unicode", 1, 11.0))[0]
    truncated_result = (await truncated_search.search("truncated", 1, 11.0))[0]

    assert "xn--mnich-kva.example" in str(unicode_result.url)
    assert unicode_result.title == "后端工程师"
    assert truncated_result.truncated is True
    assert len(truncated_result.snippet) == 4_000


@pytest.mark.asyncio
async def test_fake_search_unknown_query_is_an_empty_tuple() -> None:
    search = FakeSearch({"known": _recorded_results("normal")}, clock=lambda: 10.0)

    assert await search.search("unknown", max_results=3, deadline=11.0) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "x" * 2_001])
async def test_fake_search_rejects_invalid_query(query: str) -> None:
    search = FakeSearch({}, clock=lambda: 10.0)

    with pytest.raises(SearchInputError) as captured:
        await search.search(query, max_results=1, deadline=11.0)

    assert captured.value.category == "invalid_input"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_results", [0, -1, 21, True, 1.5])
async def test_fake_search_rejects_invalid_max_results(max_results: object) -> None:
    search = FakeSearch({}, clock=lambda: 10.0)

    with pytest.raises(SearchInputError):
        await search.search("query", max_results=max_results, deadline=11.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, "later"])
async def test_fake_search_rejects_invalid_deadline(deadline: object) -> None:
    search = FakeSearch({}, clock=lambda: 10.0)

    with pytest.raises(SearchInputError):
        await search.search("query", max_results=1, deadline=deadline)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_fake_search_rejects_expired_deadline() -> None:
    search = FakeSearch({}, clock=lambda: 10.0)

    with pytest.raises(SearchDeadlineExceededError) as captured:
        await search.search("query", max_results=1, deadline=10.0)

    assert captured.value.retryable is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "error_type", "category", "retryable"),
    [
        ("timeout", SearchTransientError, "provider_timeout", True),
        ("transient", SearchTransientError, "provider_unavailable", True),
        ("permanent", SearchPermanentError, "provider_rejected", False),
    ],
)
async def test_fake_search_supports_safe_failure_injection(
    kind: str,
    error_type: type[SearchTransientError] | type[SearchPermanentError],
    category: str,
    retryable: bool,
) -> None:
    search = FakeSearch(
        {},
        failures={"failure": FakeSearchFailure(kind=kind)},  # type: ignore[arg-type]
        clock=lambda: 10.0,
    )

    with pytest.raises(error_type) as captured:
        await search.search("failure", max_results=1, deadline=11.0)

    assert captured.value.category == category
    assert captured.value.retryable is retryable


def test_fake_search_rejects_invalid_fixtures_and_failures() -> None:
    with pytest.raises(ValueError, match="fixture queries"):
        FakeSearch({" ": ()})
    with pytest.raises(ValueError, match="valid SearchResult"):
        FakeSearch({"query": (object(),)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="failure queries"):
        FakeSearch({}, failures={" ": FakeSearchFailure(kind="timeout")})
    with pytest.raises(ValueError, match="FakeSearchFailure"):
        FakeSearch({}, failures={"query": object()})  # type: ignore[dict-item]
