from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.llm.ports import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_DIMENSION,
    LOCKED_EMBEDDING_MODEL,
    ChatMessage,
    ChatModelResult,
    EmbeddingResult,
    ModelToolSchema,
    ModelUsage,
    ProviderAttemptContext,
)

FAKE_EMBEDDING_DIMENSION = LOCKED_EMBEDDING_DIMENSION
_DIRECT_CHAT_RESULT = ChatModelResult(
    content="Offline fake response.",
    usage=ModelUsage(input_tokens=0, output_tokens=0),
)


class ScriptedFakeFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: Literal["timeout", "provider_error"]


type ScriptedFakeStep = ChatModelResult | ScriptedFakeFailure


class ScriptedFakeProviderError(RuntimeError):
    """A synthetic provider failure with no caller-controlled message."""


class ScriptedFakeExhaustedError(RuntimeError):
    """A valid invocation had no remaining scripted step."""


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    if not isinstance(metadata, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError("metadata keys and values must be strings")


def _validate_chat_invocation(
    messages: Sequence[ChatMessage],
    tools: Sequence[ModelToolSchema],
    metadata: Mapping[str, str],
) -> None:
    if not messages:
        raise ValueError("messages must not be empty")
    if any(not isinstance(message, ChatMessage) for message in messages):
        raise ValueError("messages must contain ChatMessage values")
    if any(not isinstance(tool, ModelToolSchema) for tool in tools):
        raise ValueError("tools must contain ModelToolSchema values")
    _validate_metadata(metadata)


class FakeChatModel:
    provider = "fake"
    model = LOCKED_CHAT_MODEL

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext | None = None,
    ) -> ChatModelResult:
        _validate_chat_invocation(messages, tools, metadata)
        if attempt is not None and not isinstance(attempt, ProviderAttemptContext):
            raise ValueError("provider attempt context uses the wrong contract")

        return _DIRECT_CHAT_RESULT.model_copy(deep=True)


class ScriptedFakeChatModel:
    provider = "fake"
    model = LOCKED_CHAT_MODEL

    def __init__(self, script: Sequence[ScriptedFakeStep]) -> None:
        if not script:
            raise ValueError("script must not be empty")
        if any(not isinstance(step, ChatModelResult | ScriptedFakeFailure) for step in script):
            raise ValueError("script must contain ChatModelResult or ScriptedFakeFailure values")

        self._script = tuple(step.model_copy(deep=True) for step in script)
        self._next_step_index = 0
        self._invoke_count = 0

    @property
    def invoke_count(self) -> int:
        return self._invoke_count

    @property
    def consumed_step_count(self) -> int:
        return self._next_step_index

    @property
    def remaining_step_count(self) -> int:
        return len(self._script) - self._next_step_index

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext | None = None,
    ) -> ChatModelResult:
        _validate_chat_invocation(messages, tools, metadata)
        if attempt is not None and not isinstance(attempt, ProviderAttemptContext):
            raise ValueError("provider attempt context uses the wrong contract")
        self._invoke_count += 1

        if self._next_step_index >= len(self._script):
            raise ScriptedFakeExhaustedError("scripted fake chat script exhausted")

        step = self._script[self._next_step_index]
        self._next_step_index += 1

        if isinstance(step, ChatModelResult):
            return step.model_copy(deep=True)
        if step.kind == "timeout":
            raise TimeoutError("scripted fake chat timeout")
        raise ScriptedFakeProviderError("scripted fake chat provider error")


class FakeEmbeddingModel:
    provider = "fake"
    model = LOCKED_EMBEDDING_MODEL

    async def embed(
        self,
        texts: Sequence[str],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext | None = None,
    ) -> EmbeddingResult:
        if not texts:
            raise ValueError("texts must not be empty")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-blank strings")
        _validate_metadata(metadata)
        if attempt is not None and not isinstance(attempt, ProviderAttemptContext):
            raise ValueError("provider attempt context uses the wrong contract")

        return EmbeddingResult(
            vectors=tuple(self._vector_for_text(text) for text in texts),
        )

    @staticmethod
    def _vector_for_text(text: str) -> tuple[float, ...]:
        digest = sha256(text.encode("utf-8")).digest()
        return tuple(
            digest[index % len(digest)] / 255.0 for index in range(FAKE_EMBEDDING_DIMENSION)
        )
