from __future__ import annotations

from inspect import signature
from typing import cast

import httpx
import pytest
from pydantic import SecretStr
from tavily import AsyncTavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import (
    TimeoutError as TavilyTimeoutError,
)

from app.tools.adapters.tavily import (
    AsyncTavilySearchClient,
    TavilySearch,
    create_tavily_adapter,
)
from app.tools.fake_search import FakeSearch
from app.tools.search import (
    SearchPermanentError,
    SearchResponseError,
    SearchTransientError,
    normalize_search_result,
)

_DEFAULT_RESPONSE = object()


class _ClientDouble:
    def __init__(
        self,
        response: object = _DEFAULT_RESPONSE,
        failure: Exception | None = None,
    ) -> None:
        self.response = {"results": []} if response is _DEFAULT_RESPONSE else response
        self.failure = failure
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def search(self, query: str, **kwargs: object) -> object:
        self.calls.append((query, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.response


def _response() -> dict[str, object]:
    return {
        "request_id": "ignored-provider-request-id",
        "results": [
            {
                "title": "Synthetic backend role",
                "url": "HTTPS://EXAMPLE.TEST:443/jobs/backend#details",
                "content": "The recorded role uses Python.",
                "published_date": "2026-08-01T12:00:00Z",
                "score": 0.99,
                "raw_content": "ignored",
            },
            {
                "title": "Lower ranked duplicate",
                "url": "https://example.test/jobs/backend",
                "content": "This duplicate must not replace the first result.",
                "published_date": None,
            },
            {
                "title": "Synthetic hiring guide",
                "url": "https://example.test/hiring/guide?role=backend",
                "content": "The guide describes the interview process.",
            },
        ],
    }


@pytest.mark.asyncio
async def test_adapter_calls_tavily_once_with_locked_parameters_and_deadline() -> None:
    client = _ClientDouble(_response())
    search = TavilySearch(client, clock=lambda: 10.0)

    results = await search.search("backend roles", max_results=5, deadline=14.5)

    assert client.calls == [
        (
            "backend roles",
            {
                "search_depth": "basic",
                "topic": "general",
                "max_results": 5,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
                "auto_parameters": False,
                "include_usage": False,
                "timeout": 4.5,
            },
        )
    ]
    assert [result.title for result in results] == [
        "Synthetic backend role",
        "Synthetic hiring guide",
    ]
    assert str(results[0].url) == "https://example.test/jobs/backend"
    assert results[0].published_at == "2026-08-01T12:00:00Z"


@pytest.mark.asyncio
async def test_recorded_fake_and_tavily_adapter_share_normalized_contract() -> None:
    raw_result = cast(dict[str, object], cast(list[object], _response()["results"])[0])
    expected = normalize_search_result(
        title=raw_result["title"],
        url=raw_result["url"],
        snippet=raw_result["content"],
        published_at=raw_result["published_date"],
    )
    fake = FakeSearch({"backend roles": (expected,)}, clock=lambda: 10.0)
    adapter = TavilySearch(
        _ClientDouble({"results": [raw_result]}),
        clock=lambda: 10.0,
    )

    fake_results = await fake.search("backend roles", 5, 12.0)
    adapter_results = await adapter.search("backend roles", 5, 12.0)

    assert adapter_results == fake_results
    assert adapter_results[0].model_dump_json() == fake_results[0].model_dump_json()


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.tavily.com/search")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("provider body secret canary", request=request, response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (UsageLimitExceededError("provider body secret canary"), "rate_limited"),
        (TavilyTimeoutError(5.0), "provider_timeout"),
        (
            httpx.ConnectError(
                "provider body secret canary",
                request=httpx.Request("POST", "https://api.tavily.com/search"),
            ),
            "provider_unavailable",
        ),
        (_http_status_error(500), "provider_unavailable"),
    ],
)
async def test_transient_provider_failures_are_typed_without_retry_or_error_text(
    failure: Exception,
    category: str,
) -> None:
    client = _ClientDouble(failure=failure)
    search = TavilySearch(client, clock=lambda: 10.0)

    with pytest.raises(SearchTransientError) as captured:
        await search.search("backend roles", 5, 12.0)

    assert len(client.calls) == 1
    assert captured.value.category == category
    assert captured.value.retryable is True
    assert "provider body secret canary" not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "category"),
    [
        (InvalidAPIKeyError("provider body secret canary"), "provider_authentication"),
        (ForbiddenError("provider body secret canary"), "provider_authentication"),
        (BadRequestError("provider body secret canary"), "provider_rejected"),
        (_http_status_error(422), "provider_rejected"),
        (RuntimeError("provider body secret canary"), "provider_failure"),
    ],
)
async def test_permanent_provider_failures_are_typed_without_retry_or_error_text(
    failure: Exception,
    category: str,
) -> None:
    client = _ClientDouble(failure=failure)
    search = TavilySearch(client, clock=lambda: 10.0)

    with pytest.raises(SearchPermanentError) as captured:
        await search.search("backend roles", 5, 12.0)

    assert len(client.calls) == 1
    assert captured.value.category == category
    assert captured.value.retryable is False
    assert "provider body secret canary" not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"results": {}},
        {"results": ["not-an-object"]},
        {"results": [{"title": "Missing fields"}]},
        {
            "results": [
                {
                    "title": "Invalid URL",
                    "url": "file:///etc/passwd",
                    "content": "Unsafe",
                }
            ]
        },
        {
            "results": [
                {
                    "title": "Invalid date",
                    "url": "https://example.test/",
                    "content": "Content",
                    "published_date": "x" * 65,
                }
            ]
        },
    ],
)
async def test_malformed_provider_response_fails_closed(response: object) -> None:
    client = _ClientDouble(response=response)
    search = TavilySearch(client, clock=lambda: 10.0)

    with pytest.raises(SearchResponseError) as captured:
        await search.search("backend roles", 5, 12.0)

    assert len(client.calls) == 1
    assert captured.value.category == "invalid_provider_response"
    assert captured.value.retryable is False


@pytest.mark.asyncio
async def test_locked_tavily_sdk_exposes_expected_async_contract_without_network() -> None:
    parameters = signature(AsyncTavilyClient.search).parameters
    assert {
        "query",
        "search_depth",
        "topic",
        "max_results",
        "include_answer",
        "include_raw_content",
        "include_images",
        "timeout",
        "auto_parameters",
        "include_usage",
    }.issubset(parameters)

    client = AsyncTavilyClient(api_key="tavily-sdk-compatibility-canary")
    try:
        assert isinstance(client, AsyncTavilySearchClient)
        assert "tavily-sdk-compatibility-canary" not in repr(client)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_factory_owns_a_closable_official_client_without_base_url_override() -> None:
    assert set(signature(create_tavily_adapter).parameters) == {"api_key"}
    bundle = create_tavily_adapter(api_key=SecretStr("tavily-sdk-compatibility-canary"))
    try:
        assert isinstance(bundle.client, AsyncTavilyClient)
        assert isinstance(bundle.search, TavilySearch)
        assert "tavily-sdk-compatibility-canary" not in repr(bundle)
    finally:
        await bundle.aclose()


@pytest.mark.parametrize("api_key", ["", " padded"])
def test_factory_rejects_missing_or_untrimmed_key(api_key: str) -> None:
    with pytest.raises(ValueError):
        create_tavily_adapter(api_key=SecretStr(api_key))
