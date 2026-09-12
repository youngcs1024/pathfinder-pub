from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

import pytest

from app.domain.tool_effects import ToolEffect
from app.llm.ports import ModelToolCall
from app.tools.contracts import ToolRunContext
from app.tools.registry import (
    ToolCallLimitExceededError,
    ToolExecutionError,
    ToolInputValidationError,
    ToolNotAllowedError,
    ToolOutputTooLargeError,
)
from app.tools.search import (
    SearchPermanentError,
    SearchResult,
    SearchTransientError,
    normalize_search_result,
)
from app.tools.web_search import (
    RESEARCH_TOOL_POLICY,
    RESEARCH_TOOL_POLICY_NAME,
    SEARCH_WEB_MAX_ATTEMPTS,
    SEARCH_WEB_MAX_OUTPUT_BYTES,
    SEARCH_WEB_PER_RUN_CALL_LIMIT,
    SEARCH_WEB_TIMEOUT_SECONDS,
    SEARCH_WEB_TOOL_NAME,
    SearchWebOutputV1,
    create_search_web_tool_registry,
)


class _NeverCancelled:
    def is_cancelled(self) -> bool:
        return False


class _ScriptedSearch:
    def __init__(self, actions: Sequence[object]) -> None:
        self.actions = list(actions)
        self.calls: list[tuple[str, int, float]] = []

    async def search(
        self,
        query: str,
        max_results: int,
        deadline: float,
    ) -> tuple[SearchResult, ...]:
        self.calls.append((query, max_results, deadline))
        action = self.actions.pop(0) if self.actions else ()
        if isinstance(action, BaseException):
            raise action
        assert isinstance(action, tuple)
        return action


def _result(
    *,
    url: str = "https://example.test/jobs/backend",
    title: str = "Synthetic backend role",
    snippet: str = "The role uses Python and PostgreSQL.",
) -> SearchResult:
    return normalize_search_result(
        title=title,
        url=url,
        snippet=snippet,
        published_at="2026-08-01T12:00:00Z",
    )


def _runtime(search: _ScriptedSearch):
    registry = create_search_web_tool_registry(search)
    return registry.bind(
        policy_name=RESEARCH_TOOL_POLICY_NAME,
        context=ToolRunContext(
            workspace_id=UUID(int=1),
            actor_user_id=UUID(int=2),
            run_id=UUID(int=3),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=100.0,
            cancellation=_NeverCancelled(),
        ),
        invocation_id_factory=lambda: UUID(int=100),
        clock=lambda: 0.0,
    )


def _call(
    *,
    call_id: str = "search-1",
    name: str = SEARCH_WEB_TOOL_NAME,
    arguments: dict[str, object] | None = None,
) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name=name,
        arguments=(
            {"query": "backend engineer", "max_results": 3} if arguments is None else arguments
        ),  # type: ignore[arg-type]
    )


def test_search_web_policy_and_model_schema_are_static_and_narrow() -> None:
    runtime = _runtime(_ScriptedSearch([()]))
    schemas = runtime.model_tools()

    assert SEARCH_WEB_TIMEOUT_SECONDS == 10.0
    assert SEARCH_WEB_MAX_ATTEMPTS == 2
    assert SEARCH_WEB_PER_RUN_CALL_LIMIT == 8
    assert SEARCH_WEB_MAX_OUTPUT_BYTES == 64 * 1024
    assert RESEARCH_TOOL_POLICY.allowed_tool_names == frozenset({SEARCH_WEB_TOOL_NAME})
    assert RESEARCH_TOOL_POLICY.allowed_effects == frozenset({ToolEffect.READ_ONLY})
    assert len(schemas) == 1
    assert schemas[0].name == SEARCH_WEB_TOOL_NAME
    assert set(schemas[0].input_schema["properties"]) == {"query", "max_results"}
    assert schemas[0].input_schema["properties"]["max_results"]["maximum"] == 8


@pytest.mark.asyncio
async def test_search_web_normalizes_query_and_forwards_only_bounded_port_arguments() -> None:
    search = _ScriptedSearch([(_result(),)])
    runtime = _runtime(search)

    serialized = await runtime.execute(
        _call(
            arguments={
                "query": "  \uff22\uff41\uff43\uff4b\uff45\uff4e\uff44   Engineer  ",
                "max_results": 3,
            }
        )
    )
    output = SearchWebOutputV1.model_validate_json(serialized, strict=True)

    assert search.calls == [("Backend Engineer", 3, 100.0)]
    assert output.query == "Backend Engineer"
    assert output.result_count == 1
    assert output.results[0].rank == 1
    assert str(output.results[0].canonical_url) == "https://example.test/jobs/backend"


@pytest.mark.asyncio
async def test_search_web_retries_one_typed_transient_failure_with_same_trusted_deadline() -> None:
    search = _ScriptedSearch(
        [
            SearchTransientError(category="provider_unavailable"),
            (_result(),),
        ]
    )
    runtime = _runtime(search)

    output = SearchWebOutputV1.model_validate_json(
        await runtime.execute(_call()),
        strict=True,
    )

    assert output.result_count == 1
    assert search.calls == [
        ("backend engineer", 3, 100.0),
        ("backend engineer", 3, 100.0),
    ]


@pytest.mark.asyncio
async def test_search_web_does_not_retry_permanent_provider_failure_or_leak_detail() -> None:
    canary = "provider-secret-canary"
    failure = SearchPermanentError(category="provider_rejected")
    failure.add_note(canary)
    search = _ScriptedSearch([failure])
    runtime = _runtime(search)

    with pytest.raises(ToolExecutionError) as captured:
        await runtime.execute(_call())

    assert len(search.calls) == 1
    assert canary not in str(captured.value)


@pytest.mark.asyncio
async def test_search_web_rejects_reserved_context_fields_and_unknown_tool() -> None:
    search = _ScriptedSearch([()])
    runtime = _runtime(search)

    with pytest.raises(ToolInputValidationError):
        await runtime.execute(
            _call(
                arguments={
                    "query": "backend",
                    "max_results": 3,
                    "workspace_id": "forged",
                }
            )
        )
    with pytest.raises(ToolNotAllowedError):
        await runtime.execute(_call(name="retrieve_documents"))

    assert search.calls == []


@pytest.mark.asyncio
async def test_search_web_logical_call_limit_is_shared_by_bound_runtime() -> None:
    eight_results = tuple(
        _result(
            url=f"https://example.test/jobs/{index}",
            title=f"Synthetic role {index}",
            snippet=f"Synthetic fact {index}.",
        )
        for index in range(8)
    )
    search = _ScriptedSearch([eight_results] * SEARCH_WEB_PER_RUN_CALL_LIMIT)
    runtime = _runtime(search)

    for ordinal in range(1, SEARCH_WEB_PER_RUN_CALL_LIMIT + 1):
        output = SearchWebOutputV1.model_validate_json(
            await runtime.execute(
                _call(
                    call_id=f"search-{ordinal}",
                    arguments={"query": f"query {ordinal}", "max_results": 8},
                )
            ),
            strict=True,
        )
        assert output.result_count == 8
    with pytest.raises(ToolCallLimitExceededError):
        await runtime.execute(_call(call_id="search-9"))

    assert len(search.calls) == SEARCH_WEB_PER_RUN_CALL_LIMIT


@pytest.mark.asyncio
async def test_search_web_preserves_empty_results_in_valid_output() -> None:
    runtime = _runtime(_ScriptedSearch([()]))

    output = SearchWebOutputV1.model_validate_json(
        await runtime.execute(_call()),
        strict=True,
    )

    assert output.result_count == 0
    assert output.results == ()


@pytest.mark.asyncio
async def test_search_web_registry_rejects_utf8_output_over_64_kib() -> None:
    results = tuple(
        _result(
            url=f"https://example.test/jobs/{index}",
            title=f"Synthetic role {index}",
            snippet="😀" * 4_000,
        )
        for index in range(8)
    )
    runtime = _runtime(_ScriptedSearch([results]))

    with pytest.raises(ToolOutputTooLargeError):
        await runtime.execute(_call(arguments={"query": "backend", "max_results": 8}))


def test_search_web_output_json_contains_no_execution_context() -> None:
    output = SearchWebOutputV1(query="backend", result_count=0)
    payload = json.loads(output.model_dump_json())

    assert set(payload) == {"query", "result_count", "results"}
    assert not {
        "workspace_id",
        "actor_user_id",
        "deadline",
        "budget",
        "credential",
        "target",
    }.intersection(payload)
