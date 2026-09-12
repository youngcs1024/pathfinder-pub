from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.tracing import TraceParentContext
from app.llm.ports import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_MODEL,
    MAX_PROVIDER_RESPONSE_ID_LENGTH,
    ChatMessage,
    ModelToolSchema,
    ModelUsage,
)

REQUEST_HASH_VERSION = 1
_REQUEST_HASH_PREFIX = b"pathfinder-llm-request-v1\x00"
_GRAPH_NODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")

type LLMInvocationKind = Literal["chat", "embedding"]
type LLMInvocationProvider = Literal["fake", "qwen"]
type LLMInvocationStatus = Literal["started", "succeeded", "failed"]
type LLMInvocationErrorCategory = Literal[
    "rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "provider_authentication",
    "provider_rejected",
    "provider_error",
    "cancelled",
    "invalid_provider_response",
]


class _InvocationContractModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )


@dataclass(frozen=True, slots=True, repr=False)
class LLMInvocationContext:
    workspace_id: UUID
    actor_user_id: UUID
    request_id: UUID | None = None
    run_id: UUID | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workspace_id, UUID)
            or not isinstance(self.actor_user_id, UUID)
            or (self.request_id is not None and not isinstance(self.request_id, UUID))
            or (self.run_id is not None and not isinstance(self.run_id, UUID))
        ):
            raise ValueError("LLM invocation context identity fields must be UUID values")


class TraceIdentifiers(_InvocationContractModel):
    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    observation_id: str = Field(pattern=r"^[0-9a-f]{16}$")

    @model_validator(mode="after")
    def identifiers_must_not_be_zero(self) -> TraceIdentifiers:
        if int(self.trace_id, 16) == 0 or int(self.observation_id, 16) == 0:
            raise ValueError("trace identifiers must not be all zero")
        return self


class LLMInvocationAttempt(_InvocationContractModel):
    invocation_id: UUID
    workspace_id: UUID
    actor_user_id: UUID
    run_id: UUID | None = None
    invocation_kind: LLMInvocationKind
    provider: LLMInvocationProvider
    model: str = Field(min_length=1, max_length=100)
    graph_node: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    prompt_version: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def profile_must_match_invocation_kind(self) -> LLMInvocationAttempt:
        if self.invocation_kind == "chat":
            if self.model != LOCKED_CHAT_MODEL or self.prompt_version is None:
                raise ValueError("chat invocation must use the locked model and a prompt version")
        elif self.model != LOCKED_EMBEDDING_MODEL or self.prompt_version is not None:
            raise ValueError(
                "embedding invocation must use the locked model without a prompt version"
            )
        return self


class LLMInvocationOutcome(_InvocationContractModel):
    status: Literal["succeeded", "failed"]
    token_usage: ModelUsage | None = None
    provider_response_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_PROVIDER_RESPONSE_ID_LENGTH,
        pattern=r"^\S(?:.*\S)?$",
    )
    latency_ms: int = Field(ge=0)
    error_category: LLMInvocationErrorCategory | None = None
    pricing_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    )
    currency: Literal["CNY"] | None = None
    estimated_cost: Decimal | None = Field(
        default=None,
        ge=0,
        max_digits=20,
        decimal_places=12,
    )
    trace_ids: TraceIdentifiers | None = None

    @model_validator(mode="after")
    def terminal_fields_must_match_status(self) -> LLMInvocationOutcome:
        cost_fields = (
            self.pricing_version,
            self.currency,
            self.estimated_cost,
        )
        if any(value is None for value in cost_fields) and any(
            value is not None for value in cost_fields
        ):
            raise ValueError("invocation cost fields must be all present or all absent")
        if self.status == "succeeded" and self.error_category is not None:
            raise ValueError("successful invocation outcome cannot contain an error category")
        if self.estimated_cost is not None and self.token_usage is None:
            raise ValueError("estimated cost requires token usage")
        if self.status == "failed":
            if (
                self.error_category is None
                or self.token_usage is not None
                or self.estimated_cost is not None
            ):
                raise ValueError(
                    "failed invocation outcome requires an error category and forbids usage or cost"
                )
        return self


class LLMTraceStart(_InvocationContractModel):
    parent_context: TraceParentContext | None = None
    request_id: UUID | None = None
    workspace_id: UUID
    run_id: UUID | None = None
    invocation_id: UUID
    graph_node: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    prompt_version: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    provider: LLMInvocationProvider
    model: str = Field(min_length=1, max_length=100)
    invocation_kind: LLMInvocationKind
    attempt_number: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def profile_must_match_invocation_kind(self) -> LLMTraceStart:
        if self.parent_context is not None and (
            self.parent_context.trace_identity.workspace_id != self.workspace_id
            or self.parent_context.trace_identity.run_id != self.run_id
        ):
            raise ValueError("LLM trace parent identity mismatch")
        if self.invocation_kind == "chat":
            if self.model != LOCKED_CHAT_MODEL or self.prompt_version is None:
                raise ValueError("chat trace must use the locked model and a prompt version")
        elif self.model != LOCKED_EMBEDDING_MODEL or self.prompt_version is not None:
            raise ValueError("embedding trace must use the locked model without a prompt version")
        return self


class LLMInvocationAuthorizationError(Exception):
    """The trusted actor has no active membership in the invocation workspace."""

    def __init__(self) -> None:
        super().__init__("LLM invocation authorization is unavailable")


class LLMInvocationInvariantError(Exception):
    """Persisted invocation state conflicts with the requested transition."""

    def __init__(self) -> None:
        super().__init__("LLM invocation state violates an invariant")


@runtime_checkable
class InvocationRecorderPort(Protocol):
    async def prepare(self, attempt: LLMInvocationAttempt) -> None: ...

    async def finalize(
        self,
        attempt: LLMInvocationAttempt,
        outcome: LLMInvocationOutcome,
    ) -> None: ...


@runtime_checkable
class TraceSinkPort(Protocol):
    def start(self, trace: LLMTraceStart) -> TraceIdentifiers | None: ...

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NoOpTraceSink:
    def start(self, trace: LLMTraceStart) -> None:
        return None

    def finish(
        self,
        identifiers: TraceIdentifiers,
        outcome: LLMInvocationOutcome | None,
    ) -> None:
        return None


def _canonical_request_hash(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(_REQUEST_HASH_PREFIX + serialized).hexdigest()}"


def chat_request_hash(
    *,
    model: str,
    messages: Sequence[ChatMessage],
    tools: Sequence[ModelToolSchema],
) -> str:
    if model != LOCKED_CHAT_MODEL:
        raise ValueError("chat request must use the locked model")
    if not messages or any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("chat request messages use the wrong contract")
    if any(not isinstance(tool, ModelToolSchema) for tool in tools):
        raise ValueError("chat request tools use the wrong contract")
    return _canonical_request_hash(
        {
            "canonicalization_version": REQUEST_HASH_VERSION,
            "invocation_kind": "chat",
            "model": model,
            "messages": [message.model_dump(mode="json", round_trip=True) for message in messages],
            "tools": [tool.model_dump(mode="json", round_trip=True) for tool in tools],
        }
    )


def embedding_request_hash(*, model: str, texts: Sequence[str]) -> str:
    if model != LOCKED_EMBEDDING_MODEL:
        raise ValueError("embedding request must use the locked model")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embedding request texts must contain non-blank strings")
    return _canonical_request_hash(
        {
            "canonicalization_version": REQUEST_HASH_VERSION,
            "invocation_kind": "embedding",
            "model": model,
            "texts": list(texts),
        }
    )


def invocation_metadata(
    metadata: Mapping[str, str],
    *,
    require_prompt_version: bool,
) -> tuple[str, str | None]:
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("LLM invocation metadata must contain string keys and values")
    graph_node = metadata.get("graph_node")
    if graph_node is None or _GRAPH_NODE_PATTERN.fullmatch(graph_node) is None:
        raise ValueError("LLM invocation metadata requires a valid graph_node")
    prompt_version = metadata.get("prompt_version")
    if require_prompt_version:
        if (
            prompt_version is None
            or len(prompt_version) != 71
            or not prompt_version.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in prompt_version[7:])
        ):
            raise ValueError("chat invocation metadata requires a valid prompt_version")
    elif prompt_version is not None:
        raise ValueError("embedding invocation metadata forbids prompt_version")
    return graph_node, prompt_version
