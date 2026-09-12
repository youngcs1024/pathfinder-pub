from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from pydantic import ValidationError

from app.tools.search import (
    SearchPermanentError,
    SearchResponseError,
    SearchResult,
    SearchTransientError,
    unique_ranked_search_results,
    validate_search_call,
)

FakeSearchFailureKind = Literal["timeout", "transient", "permanent"]


@dataclass(frozen=True, slots=True)
class FakeSearchFailure:
    kind: FakeSearchFailureKind

    def __post_init__(self) -> None:
        if self.kind not in {"timeout", "transient", "permanent"}:
            raise ValueError("fake search failure kind is invalid")


class FakeSearch:
    def __init__(
        self,
        fixtures: Mapping[str, Sequence[SearchResult]],
        *,
        failures: Mapping[str, FakeSearchFailure] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        copied_fixtures: dict[str, tuple[SearchResult, ...]] = {}
        for query, results in fixtures.items():
            if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
                raise ValueError("fixture queries must be valid search query strings")
            try:
                copied_results = tuple(
                    SearchResult.model_validate(result.model_dump(), strict=True)
                    for result in results
                    if isinstance(result, SearchResult)
                )
            except (AttributeError, ValidationError):
                raise ValueError("fixture values must contain valid SearchResult values") from None
            if len(copied_results) != len(results):
                raise ValueError("fixture values must contain valid SearchResult values")
            copied_fixtures[query] = copied_results

        copied_failures: dict[str, FakeSearchFailure] = {}
        for query, failure in (failures or {}).items():
            if not isinstance(query, str) or not query.strip() or len(query) > 2_000:
                raise ValueError("failure queries must be valid search query strings")
            if not isinstance(failure, FakeSearchFailure):
                raise ValueError("failure values must use FakeSearchFailure")
            copied_failures[query] = failure

        if not callable(clock):
            raise ValueError("fake search clock must be callable")
        self._fixtures = copied_fixtures
        self._failures = copied_failures
        self._clock = clock

    async def search(
        self,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[SearchResult, ...]:
        validated_query, validated_max_results, _remaining = validate_search_call(
            query,
            max_results,
            deadline,
            clock=self._clock,
        )
        failure = self._failures.get(validated_query)
        if failure is not None:
            if failure.kind == "timeout":
                raise SearchTransientError(category="provider_timeout")
            if failure.kind == "transient":
                raise SearchTransientError(category="provider_unavailable")
            if failure.kind == "permanent":
                raise SearchPermanentError(category="provider_rejected")
            raise SearchResponseError

        return unique_ranked_search_results(
            self._fixtures.get(validated_query, ()),
            max_results=validated_max_results,
        )
