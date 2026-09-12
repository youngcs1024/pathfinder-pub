from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Never
from urllib.parse import urlsplit

import httpx
import openai
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, ValidationError

from app.llm.ports import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_DIMENSION,
    LOCKED_EMBEDDING_MODEL,
    MAX_PROVIDER_RESPONSE_ID_LENGTH,
    ChatMessage,
    ChatModelResult,
    EmbeddingResult,
    ModelToolCall,
    ModelToolSchema,
    ModelUsage,
    ProviderAdapterError,
    ProviderAttemptContext,
)

QWEN_REASONING_EFFORT = "medium"
QWEN_MAX_RETRIES = 0
QWEN_API_BASE_URL_TEMPLATE = (
    "https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
QWEN_CHAT_OUTPUT_VERSION = "responses/v1"
QWEN_MAX_OUTPUT_TOKENS = 4096
QWEN_EMBEDDING_DIMENSION = LOCKED_EMBEDDING_DIMENSION
QWEN_MAX_EMBEDDING_BATCH_SIZE = 10
_QWEN_WORKSPACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_NON_RETRYABLE_BILLING_CODE_FRAGMENTS = (
    "arrearage",
    "billoverdue",
    "commoditynotpurchased",
    "freetieronly",
)
_AUTHENTICATION_CODE_FRAGMENTS = (
    "accessdenied",
    "invalidapikey",
    "workspace.notfound",
)
_NON_RETRYABLE_REQUEST_CODE_FRAGMENTS = (
    "datainspectionfailed",
    "invalidparameter",
    "modelnotfound",
)
_QWEN_BEIJING_HOST_SUFFIX = ".cn-beijing.maas.aliyuncs.com"
_QWEN_API_PATH = "/compatible-mode/v1"


@dataclass(frozen=True, slots=True, repr=False)
class QwenAdapterBundle:
    chat: QwenChatAdapter
    embedding: QwenEmbeddingAdapter
    client: openai.AsyncOpenAI
    sync_client: openai.OpenAI

    async def aclose(self) -> None:
        try:
            await self.client.close()
        finally:
            self.sync_client.close()


class QwenChatAdapter:
    provider = "qwen"
    model = LOCKED_CHAT_MODEL

    def __init__(self, model: ChatOpenAI) -> None:
        if (
            not isinstance(model, ChatOpenAI)
            or model.model_name != LOCKED_CHAT_MODEL
            or model.reasoning_effort != QWEN_REASONING_EFFORT
            or model.use_responses_api is not True
            or model.use_previous_response_id is not False
            or model.store is not False
            or model.max_retries != QWEN_MAX_RETRIES
            or model.include_response_headers is not True
            or model.output_version != QWEN_CHAT_OUTPUT_VERSION
            or model.max_tokens != QWEN_MAX_OUTPUT_TOKENS
            or not _is_trusted_qwen_base_url(getattr(model.root_async_client, "base_url", None))
            or not _is_trusted_qwen_base_url(getattr(model.root_client, "base_url", None))
        ):
            raise ValueError("Qwen chat adapter configuration is invalid")
        self._model = model

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult:
        _validate_attempt(attempt)
        _validate_metadata(metadata)
        try:
            provider_messages = _provider_messages(messages)
            provider_tools = _provider_tools(tools)
        except Exception:
            raise ProviderAdapterError(
                category="provider_rejected",
                retryable=False,
            ) from None

        runnable = (
            self._model.bind_tools(provider_tools, parallel_tool_calls=False)
            if provider_tools
            else self._model.bind(parallel_tool_calls=False)
        )
        try:
            response = await runnable.ainvoke(
                provider_messages,
                timeout=attempt.timeout_seconds,
            )
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
            raise ProviderAdapterError(
                category="invalid_provider_response",
                retryable=False,
            ) from None
        except openai.OpenAIError as error:
            _raise_provider_error(error)
        except Exception:
            raise ProviderAdapterError(category="provider_error", retryable=False) from None

        try:
            return _normalize_chat_response(response)
        except ProviderAdapterError:
            raise
        except Exception:
            raise ProviderAdapterError(
                category="invalid_provider_response",
                retryable=False,
            ) from None


class QwenEmbeddingAdapter:
    provider = "qwen"
    model = LOCKED_EMBEDDING_MODEL

    def __init__(self, client: openai.AsyncOpenAI) -> None:
        if (
            not isinstance(client, openai.AsyncOpenAI)
            or client.max_retries != QWEN_MAX_RETRIES
            or not _is_trusted_qwen_base_url(client.base_url)
        ):
            raise ValueError("Qwen embedding adapter configuration is invalid")
        self._client = client

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> EmbeddingResult:
        _validate_attempt(attempt)
        _validate_metadata(metadata)
        if (
            not texts
            or len(texts) > QWEN_MAX_EMBEDDING_BATCH_SIZE
            or any(not isinstance(text, str) or not text.strip() for text in texts)
        ):
            raise ProviderAdapterError(category="provider_rejected", retryable=False)

        try:
            response = await self._client.embeddings.create(
                input=list(texts),
                model=LOCKED_EMBEDDING_MODEL,
                dimensions=QWEN_EMBEDDING_DIMENSION,
                encoding_format="float",
                timeout=attempt.timeout_seconds,
            )
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
            raise ProviderAdapterError(
                category="invalid_provider_response",
                retryable=False,
            ) from None
        except openai.OpenAIError as error:
            _raise_provider_error(error)
        except Exception:
            raise ProviderAdapterError(category="provider_error", retryable=False) from None

        try:
            if response.model != LOCKED_EMBEDDING_MODEL:
                raise ValueError("provider returned an unexpected embedding model")
            usage = _model_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,
                total_tokens=response.usage.total_tokens,
            )
            if len(response.data) != len(texts):
                raise ValueError("provider returned the wrong number of embeddings")
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(texts))):
                raise ValueError("provider returned invalid embedding indexes")
            vectors = tuple(_embedding_vector(item.embedding) for item in ordered)
            if any(len(vector) != QWEN_EMBEDDING_DIMENSION for vector in vectors):
                raise ValueError("provider returned an unexpected embedding dimension")
            return EmbeddingResult(
                vectors=vectors,
                usage=usage,
                provider="qwen",
                model=LOCKED_EMBEDDING_MODEL,
                provider_response_id=_provider_response_id(getattr(response, "_request_id", None)),
            )
        except Exception:
            raise ProviderAdapterError(
                category="invalid_provider_response",
                retryable=False,
            ) from None


def create_qwen_adapters(
    *,
    api_key: SecretStr,
    workspace_id: SecretStr,
    http_async_client: httpx.AsyncClient | None = None,
) -> QwenAdapterBundle:
    if (
        not isinstance(api_key, SecretStr)
        or not api_key.get_secret_value()
        or api_key.get_secret_value() != api_key.get_secret_value().strip()
    ):
        raise ValueError("Qwen API key is required")
    if not isinstance(workspace_id, SecretStr):
        raise ValueError("Qwen workspace ID is required")
    untrimmed_workspace_id = workspace_id.get_secret_value()
    raw_workspace_id = untrimmed_workspace_id.strip()
    if (
        untrimmed_workspace_id != raw_workspace_id
        or _QWEN_WORKSPACE_ID_PATTERN.fullmatch(raw_workspace_id) is None
    ):
        raise ValueError("Qwen workspace ID is invalid")
    if http_async_client is not None and not isinstance(http_async_client, httpx.AsyncClient):
        raise ValueError("Qwen HTTP client uses the wrong contract")

    raw_key = api_key.get_secret_value()
    base_url = QWEN_API_BASE_URL_TEMPLATE.format(workspace_id=raw_workspace_id)
    client = openai.AsyncOpenAI(
        api_key=raw_key,
        base_url=base_url,
        max_retries=QWEN_MAX_RETRIES,
        http_client=http_async_client,
    )
    sync_client = openai.OpenAI(
        api_key=raw_key,
        base_url=base_url,
        max_retries=QWEN_MAX_RETRIES,
    )

    async def async_api_key() -> str:
        return raw_key

    chat_model = ChatOpenAI(
        model=LOCKED_CHAT_MODEL,
        reasoning_effort=QWEN_REASONING_EFFORT,
        use_responses_api=True,
        use_previous_response_id=False,
        store=False,
        max_retries=QWEN_MAX_RETRIES,
        include_response_headers=True,
        output_version=QWEN_CHAT_OUTPUT_VERSION,
        max_tokens=QWEN_MAX_OUTPUT_TOKENS,
        root_async_client=client,
        async_client=client.chat.completions,
        root_client=sync_client,
        client=sync_client.chat.completions,
        api_key=async_api_key,
    )
    return QwenAdapterBundle(
        chat=QwenChatAdapter(chat_model),
        embedding=QwenEmbeddingAdapter(client),
        client=client,
        sync_client=sync_client,
    )


def _validate_attempt(attempt: ProviderAttemptContext) -> None:
    if not isinstance(attempt, ProviderAttemptContext):
        raise ProviderAdapterError(category="provider_rejected", retryable=False)


def _is_trusted_qwen_base_url(value: object) -> bool:
    try:
        parsed = urlsplit(str(value))
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme != "https"
        or hostname is None
        or not hostname.endswith(_QWEN_BEIJING_HOST_SUFFIX)
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != _QWEN_API_PATH
    ):
        return False
    workspace_id = hostname[: -len(_QWEN_BEIJING_HOST_SUFFIX)]
    return _QWEN_WORKSPACE_ID_PATTERN.fullmatch(workspace_id) is not None


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ProviderAdapterError(category="provider_rejected", retryable=False)


def _provider_messages(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    if not messages or any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("chat messages use the wrong contract")
    converted: list[BaseMessage] = []
    for message in messages:
        if message.role == "system":
            converted.append(SystemMessage(content=message.content or ""))
        elif message.role == "user":
            converted.append(HumanMessage(content=message.content or ""))
        elif message.role == "assistant":
            converted.append(
                AIMessage(
                    content=message.content or "",
                    tool_calls=[
                        {
                            "id": call.call_id,
                            "name": call.name,
                            "args": call.arguments,
                            "type": "tool_call",
                        }
                        for call in message.tool_calls
                    ],
                )
            )
        else:
            converted.append(
                ToolMessage(
                    content=message.content or "",
                    tool_call_id=message.tool_call_id or "",
                )
            )
    return converted


def _provider_tools(tools: Sequence[ModelToolSchema]) -> list[dict[str, object]]:
    if any(not isinstance(tool, ModelToolSchema) for tool in tools):
        raise ValueError("chat tools use the wrong contract")
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
        for tool in tools
    ]


def _normalize_chat_response(response: object) -> ChatModelResult:
    if not isinstance(response, AIMessage) or response.invalid_tool_calls:
        raise ProviderAdapterError(
            category="invalid_provider_response",
            retryable=False,
        )
    metadata = response.response_metadata
    if not isinstance(metadata, Mapping) or metadata.get("model_name") != LOCKED_CHAT_MODEL:
        raise ProviderAdapterError(
            category="invalid_provider_response",
            retryable=False,
        )
    finish_status = metadata.get("status")
    if finish_status not in {"completed", "incomplete"}:
        raise ProviderAdapterError(
            category="invalid_provider_response",
            retryable=False,
        )
    usage_metadata = response.usage_metadata
    if not isinstance(usage_metadata, Mapping):
        raise ProviderAdapterError(
            category="invalid_provider_response",
            retryable=False,
        )
    input_token_details = _usage_details(usage_metadata.get("input_token_details"))
    output_token_details = _usage_details(usage_metadata.get("output_token_details"))
    usage = _model_usage(
        input_tokens=usage_metadata.get("input_tokens"),
        output_tokens=usage_metadata.get("output_tokens"),
        total_tokens=usage_metadata.get("total_tokens"),
        cached_input_tokens=input_token_details.get("cache_read"),
        cache_write_input_tokens=input_token_details.get("cache_creation"),
        reasoning_output_tokens=output_token_details.get("reasoning"),
    )
    tool_calls = tuple(_tool_call(item) for item in response.tool_calls)
    content = _visible_text(response.content)
    response_id = _provider_response_id(metadata.get("id"))
    return ChatModelResult(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        provider="qwen",
        model=LOCKED_CHAT_MODEL,
        provider_response_id=response_id,
        finish_status=finish_status,
    )


def _visible_text(content: object) -> str | None:
    if isinstance(content, str):
        return content if content.strip() else None
    if not isinstance(content, list):
        raise ValueError("provider chat content uses an unsupported shape")
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, Mapping):
            block_type = block.get("type")
            if block_type in {"text", "output_text"}:
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("provider text block is invalid")
                chunks.append(text)
            elif block_type == "refusal":
                refusal = block.get("refusal")
                if not isinstance(refusal, str):
                    raise ValueError("provider refusal block is invalid")
                chunks.append(refusal)
            elif block_type not in {"reasoning", "function_call"}:
                raise ValueError("provider chat content block is unsupported")
        else:
            raise ValueError("provider chat content block is invalid")
    joined = "".join(chunks)
    return joined if joined.strip() else None


def _tool_call(value: object) -> ModelToolCall:
    if not isinstance(value, Mapping):
        raise ValueError("provider tool call is invalid")
    arguments = value.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("provider tool arguments are invalid")
    return ModelToolCall(
        call_id=value.get("id"),
        name=value.get("name"),
        arguments=arguments,
    )


def _usage_details(value: object) -> Mapping[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("provider usage details are invalid")
    return value


def _model_usage(
    *,
    input_tokens: object,
    output_tokens: object,
    total_tokens: object = None,
    cached_input_tokens: object = None,
    cache_write_input_tokens: object = None,
    reasoning_output_tokens: object = None,
) -> ModelUsage:
    for value in (
        input_tokens,
        output_tokens,
        total_tokens,
        cached_input_tokens,
        cache_write_input_tokens,
        reasoning_output_tokens,
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("provider usage is invalid")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_input_tokens=cache_write_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )


def _embedding_vector(value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("provider embedding vector is invalid")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not isfinite(float(item)):
            raise ValueError("provider embedding vector is invalid")
        normalized.append(float(item))
    return tuple(normalized)


def _provider_response_id(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > MAX_PROVIDER_RESPONSE_ID_LENGTH
    ):
        return None
    return value


def _raise_provider_error(error: openai.OpenAIError) -> Never:
    response_id = _error_response_id(error)
    error_code = _qwen_error_code(error)
    if isinstance(error, openai.APIResponseValidationError):
        raise ProviderAdapterError(
            category="invalid_provider_response",
            retryable=False,
            provider_response_id=response_id,
        ) from None
    if _contains_code_fragment(error_code, _NON_RETRYABLE_BILLING_CODE_FRAGMENTS):
        raise ProviderAdapterError(
            category="provider_rejected",
            retryable=False,
            provider_response_id=response_id,
        ) from None
    if _contains_code_fragment(error_code, _AUTHENTICATION_CODE_FRAGMENTS):
        raise ProviderAdapterError(
            category="provider_authentication",
            retryable=False,
            provider_response_id=response_id,
        ) from None
    if _contains_code_fragment(error_code, _NON_RETRYABLE_REQUEST_CODE_FRAGMENTS):
        raise ProviderAdapterError(
            category="provider_rejected",
            retryable=False,
            provider_response_id=response_id,
        ) from None
    if isinstance(error, openai.RateLimitError):
        raise ProviderAdapterError(
            category="rate_limited",
            retryable=True,
            retry_after_seconds=_retry_after_seconds(error.response.headers),
            provider_response_id=response_id,
        ) from None
    if isinstance(error, openai.APITimeoutError):
        raise ProviderAdapterError(
            category="provider_timeout",
            retryable=True,
            provider_response_id=response_id,
        ) from None
    if isinstance(error, openai.APIConnectionError):
        raise ProviderAdapterError(
            category="provider_unavailable",
            retryable=True,
            provider_response_id=response_id,
        ) from None
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        raise ProviderAdapterError(
            category="provider_authentication",
            retryable=False,
            provider_response_id=response_id,
        ) from None
    if isinstance(error, openai.APIStatusError):
        if 500 <= error.status_code <= 599:
            raise ProviderAdapterError(
                category="provider_unavailable",
                retryable=True,
                retry_after_seconds=_retry_after_seconds(error.response.headers),
                provider_response_id=response_id,
            ) from None
        if 400 <= error.status_code <= 499:
            raise ProviderAdapterError(
                category="provider_rejected",
                retryable=False,
                provider_response_id=response_id,
            ) from None
    raise ProviderAdapterError(
        category="provider_error",
        retryable=False,
        provider_response_id=response_id,
    ) from None


def _error_response_id(error: openai.OpenAIError) -> str | None:
    if isinstance(error, openai.APIStatusError):
        body = getattr(error, "body", None)
        if isinstance(body, Mapping):
            body_id = _provider_response_id(body.get("request_id") or body.get("requestId"))
            if body_id is not None:
                return body_id
        try:
            response_body = error.response.json()
        except (TypeError, ValueError):
            response_body = None
        if isinstance(response_body, Mapping):
            response_id = _provider_response_id(
                response_body.get("request_id") or response_body.get("requestId")
            )
            if response_id is not None:
                return response_id
        return _provider_response_id(error.response.headers.get("x-request-id"))
    return _provider_response_id(getattr(error, "request_id", None))


def _qwen_error_code(error: openai.OpenAIError) -> str | None:
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return None
    nested_error = body.get("error")
    candidate = nested_error.get("code") if isinstance(nested_error, Mapping) else body.get("code")
    return candidate if isinstance(candidate, str) and candidate.strip() else None


def _contains_code_fragment(code: str | None, fragments: Sequence[str]) -> bool:
    if code is None:
        return False
    normalized = code.casefold()
    return any(fragment in normalized for fragment in fragments)


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        seconds = (parsed - datetime.now(UTC)).total_seconds()
    if not isfinite(seconds) or seconds < 0:
        return None
    return seconds
