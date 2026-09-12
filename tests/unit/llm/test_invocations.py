from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.llm.invocations import (
    LOCKED_CHAT_MODEL,
    LOCKED_EMBEDDING_MODEL,
    LLMInvocationAttempt,
    LLMInvocationContext,
    LLMInvocationOutcome,
    LLMTraceStart,
    TraceIdentifiers,
    chat_request_hash,
    embedding_request_hash,
    invocation_metadata,
)
from app.llm.ports import ChatMessage, ModelToolSchema, ModelUsage

PROMPT_VERSION = f"sha256:{'1' * 64}"


def _chat_attempt(**overrides: object) -> LLMInvocationAttempt:
    values: dict[str, object] = {
        "invocation_id": UUID(int=1),
        "workspace_id": UUID(int=2),
        "actor_user_id": UUID(int=3),
        "invocation_kind": "chat",
        "provider": "fake",
        "model": LOCKED_CHAT_MODEL,
        "graph_node": "research_agent",
        "prompt_version": PROMPT_VERSION,
        "request_hash": f"sha256:{'2' * 64}",
    }
    values.update(overrides)
    return LLMInvocationAttempt.model_validate(values, strict=True)


def test_invocation_context_and_attempts_are_strict_and_profile_bound() -> None:
    context = LLMInvocationContext(
        workspace_id=UUID(int=1),
        actor_user_id=UUID(int=2),
        request_id=UUID(int=3),
        run_id=UUID(int=4),
    )
    assert context.workspace_id == UUID(int=1)
    assert context.request_id == UUID(int=3)
    assert context.run_id == UUID(int=4)
    assert _chat_attempt().model == LOCKED_CHAT_MODEL

    with pytest.raises(ValueError, match="identity fields must be UUID"):
        LLMInvocationContext(workspace_id="forged", actor_user_id=UUID(int=2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="identity fields must be UUID"):
        LLMInvocationContext(  # type: ignore[arg-type]
            workspace_id=UUID(int=1), actor_user_id=UUID(int=2), request_id="forged"
        )
    with pytest.raises(ValidationError):
        _chat_attempt(model="request-overridden-model")
    with pytest.raises(ValidationError):
        _chat_attempt(invocation_kind="embedding", model=LOCKED_EMBEDDING_MODEL)
    with pytest.raises(ValidationError):
        _chat_attempt(workspace_id=str(UUID(int=2)))


def test_trace_contracts_are_strict_profile_bound_and_nonzero() -> None:
    identifiers = TraceIdentifiers(trace_id="1" * 32, observation_id="2" * 16)
    trace = LLMTraceStart(
        request_id=UUID(int=1),
        workspace_id=UUID(int=2),
        run_id=UUID(int=3),
        invocation_id=UUID(int=4),
        graph_node="research_agent",
        prompt_version=PROMPT_VERSION,
        provider="qwen",
        model=LOCKED_CHAT_MODEL,
        invocation_kind="chat",
        attempt_number=2,
    )

    assert trace.attempt_number == 2
    assert identifiers.trace_id == "1" * 32
    assert (
        LLMInvocationOutcome(
            status="succeeded",
            token_usage=ModelUsage(input_tokens=1, output_tokens=2),
            latency_ms=3,
            trace_ids=identifiers,
        ).trace_ids
        == identifiers
    )

    for values in (
        {"trace_id": "0" * 32, "observation_id": "2" * 16},
        {"trace_id": "A" * 32, "observation_id": "2" * 16},
        {"trace_id": "1" * 31, "observation_id": "2" * 16},
        {"trace_id": "1" * 32, "observation_id": "0" * 16},
        {"trace_id": "1" * 32, "observation_id": "2" * 16, "extra": "forged"},
    ):
        with pytest.raises(ValidationError):
            TraceIdentifiers.model_validate(values, strict=True)

    with pytest.raises(ValidationError):
        LLMTraceStart(
            workspace_id=UUID(int=2),
            invocation_id=UUID(int=4),
            graph_node="research_agent",
            prompt_version=None,
            provider="qwen",
            model=LOCKED_CHAT_MODEL,
            invocation_kind="chat",
            attempt_number=1,
        )
    with pytest.raises(ValidationError):
        LLMTraceStart(
            workspace_id=UUID(int=2),
            invocation_id=UUID(int=4),
            graph_node="ingest_documents",
            prompt_version=None,
            provider="qwen",
            model=LOCKED_EMBEDDING_MODEL,
            invocation_kind="embedding",
            attempt_number=4,
        )


def test_outcome_fields_must_match_terminal_status() -> None:
    usage = ModelUsage(input_tokens=3, output_tokens=5)
    assert (
        LLMInvocationOutcome(status="succeeded", token_usage=usage, latency_ms=7).error_category
        is None
    )
    assert (
        LLMInvocationOutcome(
            status="failed",
            provider_response_id="req_timeout_123",
            latency_ms=7,
            error_category="provider_timeout",
        ).token_usage
        is None
    )

    with pytest.raises(ValidationError):
        LLMInvocationOutcome(
            status="succeeded",
            token_usage=usage,
            latency_ms=7,
            error_category="provider_error",
        )
    with pytest.raises(ValidationError):
        LLMInvocationOutcome(status="failed", latency_ms=7)
    with pytest.raises(ValidationError):
        LLMInvocationOutcome(
            status="failed",
            token_usage=usage,
            latency_ms=7,
            error_category="provider_error",
        )


def test_outcome_cost_fields_are_atomic_and_require_success_usage() -> None:
    usage = ModelUsage(
        input_tokens=3,
        output_tokens=5,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
    )
    outcome = LLMInvocationOutcome(
        status="succeeded",
        token_usage=usage,
        latency_ms=7,
        pricing_version="qwen-cn-beijing-cny-2026-08-13-v1",
        currency="CNY",
        estimated_cost=Decimal("0.000066"),
    )
    assert outcome.estimated_cost == Decimal("0.000066")

    invalid_cost_fields = (
        {"pricing_version": "qwen-cn-beijing-cny-2026-08-13-v1"},
        {"currency": "CNY"},
        {"estimated_cost": Decimal("0.01")},
    )
    for fields in invalid_cost_fields:
        with pytest.raises(ValidationError):
            LLMInvocationOutcome(
                status="succeeded",
                token_usage=usage,
                latency_ms=7,
                **fields,
            )

    with pytest.raises(ValidationError):
        LLMInvocationOutcome(
            status="succeeded",
            latency_ms=7,
            pricing_version="qwen-cn-beijing-cny-2026-08-13-v1",
            currency="CNY",
            estimated_cost=Decimal("0.01"),
        )

    with pytest.raises(ValidationError):
        LLMInvocationOutcome(
            status="failed",
            latency_ms=7,
            error_category="provider_error",
            pricing_version="qwen-cn-beijing-cny-2026-08-13-v1",
            currency="CNY",
            estimated_cost=Decimal("0.01"),
        )


def test_chat_request_hash_is_canonical_sensitive_and_contains_no_plaintext() -> None:
    messages = (
        ChatMessage(role="system", content="secret-system-canary"),
        ChatMessage(role="user", content="secret-user-canary"),
    )
    tools = (
        ModelToolSchema(
            name="lookup",
            description="Lookup synthetic data",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
    )

    first = chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=tools)
    second = chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=tools)
    changed = chat_request_hash(
        model=LOCKED_CHAT_MODEL,
        messages=(*messages[:-1], ChatMessage(role="user", content="changed")),
        tools=tools,
    )

    assert first == second
    assert first != changed
    assert first.startswith("sha256:") and len(first) == 71
    assert "secret" not in first


def test_embedding_request_hash_is_order_sensitive_and_profile_bound() -> None:
    first = embedding_request_hash(model=LOCKED_EMBEDDING_MODEL, texts=("alpha", "beta"))
    reordered = embedding_request_hash(model=LOCKED_EMBEDDING_MODEL, texts=("beta", "alpha"))

    assert first != reordered
    with pytest.raises(ValueError, match="locked model"):
        embedding_request_hash(model="overridden", texts=("alpha",))


def test_metadata_extracts_only_code_owned_accounting_fields() -> None:
    graph_node, prompt_version = invocation_metadata(
        {
            "graph_node": "research_agent",
            "prompt_version": PROMPT_VERSION,
            "request_label": "unpersisted-canary",
            "agent_iteration": "2",
        },
        require_prompt_version=True,
    )

    assert graph_node == "research_agent"
    assert prompt_version == PROMPT_VERSION
    assert invocation_metadata(
        {"graph_node": "ingest_documents"}, require_prompt_version=False
    ) == ("ingest_documents", None)

    with pytest.raises(ValueError, match="graph_node"):
        invocation_metadata({"prompt_version": PROMPT_VERSION}, require_prompt_version=True)
    with pytest.raises(ValueError, match="prompt_version"):
        invocation_metadata(
            {"graph_node": "plan", "prompt_version": "forged"},
            require_prompt_version=True,
        )
    with pytest.raises(ValueError, match="forbids prompt_version"):
        invocation_metadata(
            {"graph_node": "ingest_documents", "prompt_version": PROMPT_VERSION},
            require_prompt_version=False,
        )
