from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Protocol, runtime_checkable

import httpx
from pydantic import SecretStr
from tavily import AsyncTavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import (
    TimeoutError as TavilyTimeoutError,
)

from app.tools.search import (
    MonotonicClock,
    SearchPermanentError,
    SearchResponseError,
    SearchResult,
    SearchTransientError,
    normalize_search_result,
    unique_ranked_search_results,
    validate_search_call,
)


@runtime_checkable
class AsyncTavilySearchClient(Protocol):
    async def search(self, query: str, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True, repr=False)
class TavilyAdapterBundle:
    search: TavilySearch
    client: AsyncTavilyClient

    async def aclose(self) -> None:
        await self.client.close()


def create_tavily_adapter(*, api_key: SecretStr) -> TavilyAdapterBundle:
    if (
        not isinstance(api_key, SecretStr)
        or not api_key.get_secret_value()
        or api_key.get_secret_value() != api_key.get_secret_value().strip()
    ):
        raise ValueError("Tavily API key is required")
    client = AsyncTavilyClient(api_key=api_key.get_secret_value())
    return TavilyAdapterBundle(search=TavilySearch(client), client=client)


class TavilySearch:
    def __init__(
        self,
        client: AsyncTavilySearchClient,
        *,
        clock: MonotonicClock = monotonic,
    ) -> None:
        if not isinstance(client, AsyncTavilySearchClient):
            raise ValueError("Tavily search client must implement the async search contract")
        if not callable(clock):
            raise ValueError("Tavily search clock must be callable")
        self._client = client
        self._clock = clock

    async def search(
        self,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[SearchResult, ...]:
        validated_query, validated_max_results, remaining = validate_search_call(
            query,
            max_results,
            deadline,
            clock=self._clock,
        )
        try:
            response = await self._client.search(
                validated_query,
                search_depth="basic",
                topic="general",
                max_results=validated_max_results,
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                auto_parameters=False,
                include_usage=False,
                timeout=remaining,
            )
        except UsageLimitExceededError:
            raise SearchTransientError(category="rate_limited") from None
        except (TavilyTimeoutError, httpx.TimeoutException):
            raise SearchTransientError(category="provider_timeout") from None
        except httpx.TransportError:
            raise SearchTransientError(category="provider_unavailable") from None
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if 500 <= status_code <= 599:
                raise SearchTransientError(category="provider_unavailable") from None
            if status_code in {401, 403}:
                raise SearchPermanentError(category="provider_authentication") from None
            if 400 <= status_code <= 499:
                raise SearchPermanentError(category="provider_rejected") from None
            raise SearchPermanentError(category="provider_failure") from None
        except (InvalidAPIKeyError, MissingAPIKeyError, ForbiddenError):
            raise SearchPermanentError(category="provider_authentication") from None
        except BadRequestError:
            raise SearchPermanentError(category="provider_rejected") from None
        except Exception:
            raise SearchPermanentError(category="provider_failure") from None

        try:
            return _normalize_tavily_response(response, max_results=validated_max_results)
        except SearchResponseError:
            raise
        except Exception:
            raise SearchResponseError from None


def _normalize_tavily_response(
    response: object,
    *,
    max_results: int,
) -> tuple[SearchResult, ...]:
    if not isinstance(response, Mapping):
        raise SearchResponseError
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise SearchResponseError

    normalized: list[SearchResult] = []
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise SearchResponseError
        if not {"title", "url", "content"}.issubset(raw_result):
            raise SearchResponseError
        normalized.append(
            normalize_search_result(
                title=raw_result["title"],
                url=raw_result["url"],
                snippet=raw_result["content"],
                published_at=raw_result.get("published_date"),
            )
        )

    return unique_ranked_search_results(tuple(normalized), max_results=max_results)
