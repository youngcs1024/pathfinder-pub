from __future__ import annotations

import json
from collections.abc import Callable
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from app.llm.ports import (
    ChatMessage,
    ModelToolSchema,
    ModelUsage,
    ProviderAdapterError,
    ProviderAttemptContext,
)
from app.llm.qwen_adapters import (
    QWEN_API_BASE_URL_TEMPLATE,
    QWEN_EMBEDDING_DIMENSION,
    QWEN_MAX_EMBEDDING_BATCH_SIZE,
    QWEN_MAX_OUTPUT_TOKENS,
    create_qwen_adapters,
)

PROMPT_VERSION = f"sha256:{'a' * 64}"
ATTEMPT = ProviderAttemptContext(invocation_id=UUID(int=7), timeout_seconds=15.0)
WORKSPACE_ID = "workspace-test"
QWEN_API_BASE_URL = QWEN_API_BASE_URL_TEMPLATE.format(workspace_id=WORKSPACE_ID)


def _response_payload(
    *,
    model: str = "qwen3.6-flash-2026-04-16",
    status: str = "completed",
    output: list[object] | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": "resp_test_123",
        "object": "response",
        "created_at": 0,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output
        or [
            {
                "id": "msg_test_123",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Visible answer.",
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": usage
        or {
            "input_tokens": 11,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 13,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 24,
        },
        "user": None,
        "metadata": {},
    }


def _embedding_payload(
    *,
    model: str = "text-embedding-v4",
    dimension: int = QWEN_EMBEDDING_DIMENSION,
    count: int = 2,
) -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [float(index + 1) / dimension for index in range(dimension)],
                "index": item_index,
            }
            for item_index in range(count)
        ],
        "model": model,
        "usage": {"prompt_tokens": 17, "total_tokens": 17},
    }


async def _bundle_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
):
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bundle = create_qwen_adapters(
        api_key=SecretStr("sk-test-secret-canary"),
        workspace_id=SecretStr(WORKSPACE_ID),
        http_async_client=http_client,
    )
    return bundle, http_client


async def test_chat_uses_locked_responses_profile_and_normalizes_visible_output() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"x-request-id": "req_chat_123"},
            json=_response_payload(),
        )

    bundle, http_client = await _bundle_with_handler(handler)
    tool = ModelToolSchema(
        name="search_web",
        description="Search the local registry-backed web tool",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    try:
        result = await bundle.chat.invoke(
            (ChatMessage(role="user", content="Find roles"),),
            (tool,),
            {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == f"{QWEN_API_BASE_URL}/responses"
    assert "x-client-request-id" not in request.headers
    assert request.headers["authorization"] == "Bearer sk-test-secret-canary"
    assert body["model"] == "qwen3.6-flash-2026-04-16"
    assert body["reasoning"] == {"effort": "medium"}
    assert QWEN_MAX_OUTPUT_TOKENS == 4096
    assert body["max_output_tokens"] == QWEN_MAX_OUTPUT_TOKENS
    assert body["store"] is False
    assert "previous_response_id" not in body
    assert "service_tier" not in body
    assert "background" not in body
    assert body["parallel_tool_calls"] is False
    assert "web_search" not in json.dumps(body)
    assert body["tools"] == [
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the local registry-backed web tool",
            "parameters": tool.input_schema,
        }
    ]
    assert result.content == "Visible answer."
    assert result.tool_calls == ()
    assert result.usage == ModelUsage(
        input_tokens=11,
        output_tokens=13,
        total_tokens=24,
        cached_input_tokens=0,
        reasoning_output_tokens=5,
    )
    assert result.provider == "qwen"
    assert result.model == "qwen3.6-flash-2026-04-16"
    assert result.provider_response_id == "resp_test_123"
    assert result.finish_status == "completed"
    assert "reasoning" not in result.model_dump(mode="json")


async def test_chat_normalizes_reasoning_only_completed_response_without_visible_text() -> None:
    payload = _response_payload(
        output=[
            {
                "id": "msg_reasoning_only",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [],
            }
        ]
    )
    bundle, http_client = await _bundle_with_handler(
        lambda request: httpx.Response(200, json=payload)
    )
    try:
        result = await bundle.chat.invoke(
            (ChatMessage(role="user", content="Reason carefully"),),
            (),
            {"graph_node": "research_agent", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert result.content is None
    assert result.tool_calls == ()
    assert result.finish_status == "completed"


async def test_embedding_uses_locked_endpoint_profile_usage_and_dimension() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"x-request-id": "req_embedding_123"},
            json=_embedding_payload(),
        )

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        result = await bundle.embedding.embed(
            ("alpha", "beta"),
            {"graph_node": "ingest_documents"},
            attempt=ATTEMPT,
        )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert str(request.url) == f"{QWEN_API_BASE_URL}/embeddings"
    assert "x-client-request-id" not in request.headers
    assert body == {
        "input": ["alpha", "beta"],
        "model": "text-embedding-v4",
        "dimensions": 1536,
        "encoding_format": "float",
    }
    assert len(result.vectors) == 2
    assert {len(vector) for vector in result.vectors} == {QWEN_EMBEDDING_DIMENSION}
    assert result.usage == ModelUsage(input_tokens=17, output_tokens=0, total_tokens=17)
    assert result.provider == "qwen"
    assert result.provider_response_id == "req_embedding_123"


@pytest.mark.parametrize(
    ("status_code", "expected_category", "expected_retryable"),
    [
        (429, "rate_limited", True),
        (500, "provider_unavailable", True),
        (401, "provider_authentication", False),
        (400, "provider_rejected", False),
    ],
)
async def test_sdk_sends_one_http_request_and_adapter_classifies_status(
    status_code: int,
    expected_category: str,
    expected_retryable: bool,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            status_code,
            headers={"x-request-id": f"req_error_{status_code}", "retry-after": "1.25"},
            json={"error": {"message": "provider-error-secret-canary", "type": "test"}},
        )

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.chat.invoke(
                (ChatMessage(role="user", content="Failure request canary"),),
                (),
                {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 1
    assert captured.value.category == expected_category
    assert captured.value.retryable is expected_retryable
    assert captured.value.provider_response_id == f"req_error_{status_code}"
    assert captured.value.retry_after_seconds == (1.25 if expected_retryable else None)
    assert "secret-canary" not in str(captured.value)


@pytest.mark.parametrize(
    ("transport_error", "expected_category"),
    [
        (httpx.ConnectError("connect-secret-canary"), "provider_unavailable"),
        (httpx.ReadTimeout("timeout-secret-canary"), "provider_timeout"),
    ],
)
async def test_sdk_transport_failures_are_single_attempt_and_safely_classified(
    transport_error: httpx.TransportError,
    expected_category: str,
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        transport_error.request = request
        raise transport_error

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.chat.invoke(
                (ChatMessage(role="user", content="Transport failure"),),
                (),
                {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 1
    assert captured.value.category == expected_category
    assert captured.value.retryable is True
    assert "secret-canary" not in str(captured.value)


async def test_missing_request_header_is_allowed_and_body_response_id_is_preserved() -> None:
    payloads = [
        _response_payload(),
        _response_payload(model="request-overridden-model"),
        _response_payload(
            output=[
                {
                    "id": "unsupported_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "forbidden"},
                }
            ]
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        first = await bundle.chat.invoke(
            (ChatMessage(role="user", content="No id"),),
            (),
            {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
        assert first.provider_response_id == "resp_test_123"

        for _ in range(2):
            with pytest.raises(ProviderAdapterError) as captured:
                await bundle.chat.invoke(
                    (ChatMessage(role="user", content="Malformed"),),
                    (),
                    {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                    attempt=ATTEMPT,
                )
            assert captured.value.category == "invalid_provider_response"
            assert captured.value.retryable is False
    finally:
        await bundle.aclose()
        await http_client.aclose()


@pytest.mark.parametrize(
    ("error_code", "expected_category"),
    [
        ("Arrearage", "provider_rejected"),
        ("CommodityNotPurchased", "provider_rejected"),
        ("PrepaidBillOverdue", "provider_rejected"),
        ("PostpaidBillOverdue", "provider_rejected"),
        ("AllocationQuota.FreeTierOnly", "provider_rejected"),
        ("InvalidParameter", "provider_rejected"),
        ("ModelNotFound", "provider_rejected"),
        ("DataInspectionFailed", "provider_rejected"),
        ("AccessDenied", "provider_authentication"),
        ("InvalidApiKey", "provider_authentication"),
    ],
)
async def test_official_non_retryable_codes_override_a_429_status(
    error_code: str,
    expected_category: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "1.25"},
            json={
                "error": {"code": error_code, "message": "provider-secret-canary"},
                "request_id": "req_rejected_123",
            },
        )

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.chat.invoke(
                (ChatMessage(role="user", content="Billing rejection"),),
                (),
                {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert captured.value.category == expected_category
    assert captured.value.retryable is False
    assert captured.value.retry_after_seconds is None
    assert captured.value.provider_response_id == "req_rejected_123"
    assert "secret-canary" not in str(captured.value)


async def test_incomplete_response_is_preserved_without_provider_state_dependency() -> None:
    bundle, http_client = await _bundle_with_handler(
        lambda request: httpx.Response(200, json=_response_payload(status="incomplete"))
    )
    try:
        result = await bundle.chat.invoke(
            (ChatMessage(role="user", content="Bounded response"),),
            (),
            {"graph_node": "writer", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert result.finish_status == "incomplete"
    assert result.provider_response_id == "resp_test_123"


async def test_custom_function_round_trip_resends_full_local_history() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        _response_payload(
            output=[
                {
                    "id": "fc_test_123",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_test_123",
                    "name": "search_web",
                    "arguments": '{"query":"backend roles"}',
                }
            ]
        ),
        _response_payload(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    tool = ModelToolSchema(
        name="search_web",
        description="Search through the local registry",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    bundle, http_client = await _bundle_with_handler(handler)
    try:
        first = await bundle.chat.invoke(
            (ChatMessage(role="user", content="Find roles"),),
            (tool,),
            {"graph_node": "research", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
        second = await bundle.chat.invoke(
            (
                ChatMessage(role="user", content="Find roles"),
                ChatMessage(role="assistant", tool_calls=first.tool_calls),
                ChatMessage(
                    role="tool",
                    content='{"results":[]}',
                    tool_call_id="call_test_123",
                ),
            ),
            (tool,),
            {"graph_node": "research", "prompt_version": PROMPT_VERSION},
            attempt=ATTEMPT,
        )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert first.tool_calls[0].name == "search_web"
    assert first.tool_calls[0].arguments == {"query": "backend roles"}
    assert second.content == "Visible answer."
    assert len(requests) == 2
    assert "previous_response_id" not in requests[1]
    serialized_history = json.dumps(requests[1]["input"], sort_keys=True)
    assert "function_call" in serialized_history
    assert "function_call_output" in serialized_history
    assert "call_test_123" in serialized_history


async def test_embedding_rejects_batches_larger_than_official_beijing_limit() -> None:
    bundle, http_client = await _bundle_with_handler(
        lambda request: httpx.Response(500, json={"error": {"message": "must not run"}})
    )
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.embedding.embed(
                tuple(f"text-{index}" for index in range(QWEN_MAX_EMBEDDING_BATCH_SIZE + 1)),
                {"graph_node": "ingest_documents"},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert captured.value.category == "provider_rejected"
    assert captured.value.retryable is False


@pytest.mark.parametrize("texts", [(), ("",), ("   ",)])
async def test_embedding_rejects_empty_input_without_an_http_attempt(
    texts: tuple[str, ...],
) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(500)

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.embedding.embed(
                texts,
                {"graph_node": "ingest_documents"},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 0
    assert captured.value.category == "provider_rejected"
    assert captured.value.retryable is False


async def test_embedding_wrong_result_count_is_rejected_without_retry() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=_embedding_payload(count=1))

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.embedding.embed(
                ("alpha", "beta"),
                {"graph_node": "ingest_documents"},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 1
    assert captured.value.category == "invalid_provider_response"
    assert captured.value.retryable is False


async def test_inconsistent_usage_details_fail_closed_without_retry() -> None:
    payloads = [
        _response_payload(
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 13,
                "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 25,
            }
        ),
        _response_payload(
            usage={
                "input_tokens": 11,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 13,
                "output_tokens_details": {"reasoning_tokens": 14},
                "total_tokens": 24,
            }
        ),
    ]
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=payloads.pop(0))

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        for _ in range(2):
            with pytest.raises(ProviderAdapterError) as captured:
                await bundle.chat.invoke(
                    (ChatMessage(role="user", content="Malformed usage"),),
                    (),
                    {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                    attempt=ATTEMPT,
                )
            assert captured.value.category == "invalid_provider_response"
            assert captured.value.retryable is False
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 2


async def test_embedding_wrong_dimension_is_rejected_without_retry() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=_embedding_payload(dimension=2, count=1))

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.embedding.embed(
                ("alpha",),
                {"graph_node": "ingest_documents"},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 1
    assert captured.value.category == "invalid_provider_response"
    assert captured.value.retryable is False


async def test_malformed_function_arguments_fail_closed() -> None:
    request_count = 0
    payload = _response_payload(
        output=[
            {
                "id": "fc_test_123",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_test_123",
                "name": "search_web",
                "arguments": "{not-json",
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=payload)

    bundle, http_client = await _bundle_with_handler(handler)
    try:
        with pytest.raises(ProviderAdapterError) as captured:
            await bundle.chat.invoke(
                (ChatMessage(role="user", content="Malformed tool call"),),
                (),
                {"graph_node": "plan", "prompt_version": PROMPT_VERSION},
                attempt=ATTEMPT,
            )
    finally:
        await bundle.aclose()
        await http_client.aclose()

    assert request_count == 1
    assert captured.value.category == "invalid_provider_response"
    assert captured.value.retryable is False


async def test_bundle_uses_only_injected_mock_transport_and_closes_owned_sdk_clients() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response_payload())

    bundle, http_client = await _bundle_with_handler(handler)
    assert bundle.client.max_retries == 0
    assert bundle.sync_client.max_retries == 0
    assert str(bundle.client.base_url) == f"{QWEN_API_BASE_URL}/"
    assert str(bundle.sync_client.base_url) == f"{QWEN_API_BASE_URL}/"

    await bundle.aclose()
    await http_client.aclose()

    assert bundle.client.is_closed()
    assert bundle.sync_client.is_closed()
