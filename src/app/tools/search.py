from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from math import isfinite
from time import monotonic
from typing import Annotated, Literal, Protocol, runtime_checkable
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    model_validator,
)

MAX_SEARCH_QUERY_LENGTH = 2_000
MAX_SEARCH_RESULTS = 20
MAX_SEARCH_TITLE_LENGTH = 500
MAX_SEARCH_SNIPPET_LENGTH = 4_000
MAX_SEARCH_PUBLISHED_AT_LENGTH = 64
SOURCE_ID_VERSION = "web-v1"

SearchErrorCategory = Literal[
    "invalid_input",
    "deadline_exceeded",
    "rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "provider_authentication",
    "provider_rejected",
    "provider_failure",
    "invalid_provider_response",
]
SearchTransientCategory = Literal[
    "rate_limited",
    "provider_timeout",
    "provider_unavailable",
]
SearchPermanentCategory = Literal[
    "provider_authentication",
    "provider_rejected",
    "provider_failure",
]


class SearchError(Exception):
    def __init__(self, *, category: SearchErrorCategory, retryable: bool) -> None:
        self.category = category
        self.retryable = retryable
        super().__init__(f"search failed: {category}")


class SearchInputError(SearchError):
    def __init__(self) -> None:
        super().__init__(category="invalid_input", retryable=False)


class SearchDeadlineExceededError(SearchError):
    def __init__(self) -> None:
        super().__init__(category="deadline_exceeded", retryable=False)


class SearchTransientError(SearchError):
    def __init__(self, *, category: SearchTransientCategory) -> None:
        super().__init__(category=category, retryable=True)


class SearchPermanentError(SearchError):
    def __init__(self, *, category: SearchPermanentCategory) -> None:
        super().__init__(category=category, retryable=False)


class SearchResponseError(SearchError):
    def __init__(self) -> None:
        super().__init__(category="invalid_provider_response", retryable=False)


SearchSourceId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^web-v1:[0-9a-f]{64}$",
    ),
]
SearchTitle = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_SEARCH_TITLE_LENGTH),
]
SearchSnippet = Annotated[
    str,
    StringConstraints(strict=True, max_length=MAX_SEARCH_SNIPPET_LENGTH),
]
SearchPublishedAt = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=MAX_SEARCH_PUBLISHED_AT_LENGTH),
]


class SearchResult(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    source_id: SearchSourceId
    title: SearchTitle
    url: AnyHttpUrl
    snippet: SearchSnippet
    published_at: SearchPublishedAt | None = None
    truncated: bool = False

    @model_validator(mode="after")
    def source_id_must_match_canonical_url(self) -> SearchResult:
        canonical_url = canonicalize_search_url(str(self.url))
        if str(self.url) != canonical_url or self.source_id != search_source_id(canonical_url):
            raise ValueError("search source identity must match its canonical URL")
        if not self.title.strip() or self.title != self.title.strip():
            raise ValueError("search title must be normalized and non-blank")
        if self.snippet != self.snippet.strip():
            raise ValueError("search snippet must be normalized")
        if self.published_at is not None and (
            not self.published_at.strip() or self.published_at != self.published_at.strip()
        ):
            raise ValueError("search published_at must be normalized and non-blank")
        return self


@runtime_checkable
class SearchPort(Protocol):
    async def search(
        self,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[SearchResult, ...]: ...


type MonotonicClock = Callable[[], float]


def validate_search_call(
    query: object,
    max_results: object,
    deadline: object,
    *,
    clock: MonotonicClock = monotonic,
) -> tuple[str, int, float]:
    if type(query) is not str or not query.strip() or len(query) > MAX_SEARCH_QUERY_LENGTH:
        raise SearchInputError
    if type(max_results) is not int or not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise SearchInputError
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, int | float)
        or not isfinite(float(deadline))
    ):
        raise SearchInputError

    remaining = float(deadline) - clock()
    if remaining <= 0:
        raise SearchDeadlineExceededError
    return query, max_results, remaining


def _canonical_host(parsed: SplitResult) -> str:
    if parsed.username is not None or parsed.password is not None:
        raise SearchResponseError

    hostname = parsed.hostname
    if hostname is None or not hostname:
        raise SearchResponseError
    try:
        port = parsed.port
        if ":" in hostname:
            host = f"[{hostname.lower()}]"
        else:
            host = hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        raise SearchResponseError from None

    is_default_port = (parsed.scheme.lower(), port) in {("http", 80), ("https", 443)}
    return host if port is None or is_default_port else f"{host}:{port}"


def canonicalize_search_url(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise SearchResponseError
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        raise SearchResponseError from None

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise SearchResponseError

    canonical = urlunsplit(
        (
            scheme,
            _canonical_host(parsed),
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    try:
        validated = AnyHttpUrl(canonical)
    except (TypeError, ValidationError, ValueError):
        raise SearchResponseError from None
    return str(validated)


def search_source_id(canonical_url: object) -> str:
    normalized = canonicalize_search_url(canonical_url)
    digest = sha256(normalized.encode("utf-8")).hexdigest()
    return f"{SOURCE_ID_VERSION}:{digest}"


def normalize_search_result(
    *,
    title: object,
    url: object,
    snippet: object,
    published_at: object = None,
) -> SearchResult:
    if type(title) is not str or type(snippet) is not str:
        raise SearchResponseError
    normalized_title = title.strip()
    normalized_snippet = snippet.strip()
    if not normalized_title:
        raise SearchResponseError

    normalized_published_at: str | None
    if published_at is None:
        normalized_published_at = None
    elif type(published_at) is str and published_at.strip():
        normalized_published_at = published_at.strip()
        if len(normalized_published_at) > MAX_SEARCH_PUBLISHED_AT_LENGTH:
            raise SearchResponseError
    else:
        raise SearchResponseError

    canonical_url = canonicalize_search_url(url)
    truncated = (
        len(normalized_title) > MAX_SEARCH_TITLE_LENGTH
        or len(normalized_snippet) > MAX_SEARCH_SNIPPET_LENGTH
    )
    try:
        return SearchResult(
            source_id=search_source_id(canonical_url),
            title=normalized_title[:MAX_SEARCH_TITLE_LENGTH],
            url=canonical_url,
            snippet=normalized_snippet[:MAX_SEARCH_SNIPPET_LENGTH],
            published_at=normalized_published_at,
            truncated=truncated,
        )
    except ValidationError:
        raise SearchResponseError from None


def unique_ranked_search_results(
    results: tuple[SearchResult, ...],
    *,
    max_results: int,
) -> tuple[SearchResult, ...]:
    selected: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        if not isinstance(result, SearchResult):
            raise SearchResponseError
        canonical_url = str(result.url)
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        selected.append(result.model_copy(deep=True))
        if len(selected) == max_results:
            break
    return tuple(selected)
