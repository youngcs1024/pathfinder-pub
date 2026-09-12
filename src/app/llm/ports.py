from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

ChatRole = Literal["system", "user", "assistant", "tool"]
ModelProvider = Literal["fake", "qwen"]
ChatFinishStatus = Literal["completed", "incomplete"]
ProviderFailureCategory = Literal[
    "rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "provider_authentication",
    "provider_rejected",
    "provider_error",
    "invalid_provider_response",
]
LOCKED_CHAT_MODEL = "qwen3.6-flash-2026-04-16"
LOCKED_EMBEDDING_MODEL = "text-embedding-v4"
LOCKED_EMBEDDING_DIMENSION = 1536
MAX_PROVIDER_RESPONSE_ID_LENGTH = 512


class _ContractModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)


class ModelToolSchema(_ContractModel):
    name: str
    description: str
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("name", "description")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model tool name and description must not be blank")
        return value


class ModelToolCall(_ContractModel):
    call_id: str
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("call_id", "name")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model tool call id and name must not be blank")
        return value


class ChatMessage(_ContractModel):
    role: ChatRole
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    tool_call_id: str | None = None

    @field_validator("content", "tool_call_id")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("chat message text fields must not be blank")
        return value

    @model_validator(mode="after")
    def fields_must_match_role(self) -> "ChatMessage":
        if self.role in {"system", "user"}:
            if self.content is None or self.tool_calls or self.tool_call_id is not None:
                raise ValueError(
                    "system and user messages require content and forbid tool-call fields"
                )
            return self

        if self.role == "assistant":
            if self.tool_call_id is not None:
                raise ValueError("assistant messages forbid tool_call_id")
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant messages require content or tool calls")
            call_ids = [call.call_id for call in self.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                raise ValueError("assistant tool call ids must be unique")
            return self

        if self.content is None or self.tool_call_id is None or self.tool_calls:
            raise ValueError(
                "tool messages require content and tool_call_id and forbid nested tool calls"
            )
        return self


class ModelUsage(_ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    cached_input_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    cache_write_input_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
    reasoning_output_tokens: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def token_details_must_match_parent_totals(self) -> "ModelUsage":
        if self.total_tokens is not None and self.total_tokens != (
            self.input_tokens + self.output_tokens
        ):
            raise ValueError("total tokens must equal input plus output tokens")

        cached_input_tokens = self.cached_input_tokens or 0
        cache_write_input_tokens = self.cache_write_input_tokens or 0
        if cached_input_tokens + cache_write_input_tokens > self.input_tokens:
            raise ValueError("cache token details must not exceed input tokens")

        if (
            self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise ValueError("reasoning token details must not exceed output tokens")
        return self


class ProviderAttemptContext(_ContractModel):
    invocation_id: UUID
    timeout_seconds: float = Field(gt=0)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def timeout_must_be_a_finite_number(cls, value: object) -> object:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
        ):
            raise ValueError("provider attempt timeout must be a positive finite number")
        return value


class ChatModelResult(_ContractModel):
    content: str | None = None
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: ModelUsage = Field(default_factory=ModelUsage)
    provider: ModelProvider = "fake"
    model: Literal["qwen3.6-flash-2026-04-16"] = LOCKED_CHAT_MODEL
    provider_response_id: str | None = Field(
        default=None,
        max_length=MAX_PROVIDER_RESPONSE_ID_LENGTH,
    )
    finish_status: ChatFinishStatus = "completed"

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("chat model content must not be blank")
        return value

    @field_validator("provider_response_id")
    @classmethod
    def provider_response_id_must_be_trimmed(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("provider response id must be non-blank and trimmed")
        return value

    @model_validator(mode="after")
    def tool_call_ids_must_be_unique(self) -> "ChatModelResult":
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("chat model tool call ids must be unique")
        return self


class EmbeddingResult(_ContractModel):
    vectors: tuple[tuple[float, ...], ...]
    usage: ModelUsage = Field(default_factory=ModelUsage)
    provider: ModelProvider = "fake"
    model: Literal["text-embedding-v4"] = LOCKED_EMBEDDING_MODEL
    provider_response_id: str | None = Field(
        default=None,
        max_length=MAX_PROVIDER_RESPONSE_ID_LENGTH,
    )

    @field_validator("provider_response_id")
    @classmethod
    def provider_response_id_must_be_trimmed(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("provider response id must be non-blank and trimmed")
        return value

    @model_validator(mode="after")
    def vectors_must_have_one_consistent_dimension(self) -> "EmbeddingResult":
        if not self.vectors:
            raise ValueError("embedding result must contain at least one vector")

        dimensions = {len(vector) for vector in self.vectors}
        if dimensions != {LOCKED_EMBEDDING_DIMENSION}:
            raise ValueError(
                f"embedding vectors must be exactly {LOCKED_EMBEDDING_DIMENSION} dimensions"
            )
        return self


@runtime_checkable
class ChatModelPort(Protocol):
    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
    ) -> ChatModelResult: ...


@runtime_checkable
class EmbeddingPort(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
    ) -> EmbeddingResult: ...


@runtime_checkable
class ChatAdapterPort(Protocol):
    provider: ModelProvider
    model: Literal["qwen3.6-flash-2026-04-16"]

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> ChatModelResult: ...


@runtime_checkable
class EmbeddingAdapterPort(Protocol):
    provider: ModelProvider
    model: Literal["text-embedding-v4"]

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext,
    ) -> EmbeddingResult: ...


class ProviderAdapterError(Exception):
    def __init__(
        self,
        *,
        category: ProviderFailureCategory,
        retryable: bool,
        retry_after_seconds: float | None = None,
        provider_response_id: str | None = None,
    ) -> None:
        transient_categories = {
            "rate_limited",
            "provider_timeout",
            "provider_unavailable",
        }
        if retryable != (category in transient_categories):
            raise ValueError("provider retry classification is inconsistent")
        if retry_after_seconds is not None and (
            isinstance(retry_after_seconds, bool)
            or not isinstance(retry_after_seconds, int | float)
            or not isfinite(float(retry_after_seconds))
            or float(retry_after_seconds) < 0
            or not retryable
        ):
            raise ValueError("provider retry-after value is invalid")
        if provider_response_id is not None and (
            not isinstance(provider_response_id, str)
            or not provider_response_id.strip()
            or provider_response_id != provider_response_id.strip()
            or len(provider_response_id) > MAX_PROVIDER_RESPONSE_ID_LENGTH
        ):
            raise ValueError("provider response id is invalid")
        self.category = category
        self.retryable = retryable
        self.retry_after_seconds = (
            float(retry_after_seconds) if retry_after_seconds is not None else None
        )
        self.provider_response_id = provider_response_id
        super().__init__(f"model provider attempt failed: {category}")
