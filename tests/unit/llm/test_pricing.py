from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.llm.ports import ModelUsage
from app.llm.pricing import (
    QWEN_BEIJING_CURRENCY,
    QWEN_BEIJING_PRICE_BOOK,
    QWEN_BEIJING_PRICING_VERSION,
    QWEN_LONG_CONTEXT_THRESHOLD,
    QWEN_MAX_CONTEXT_TOKENS,
    InvocationCost,
)


def test_standard_short_context_chat_cost_uses_beijing_list_rates() -> None:
    usage = ModelUsage(
        input_tokens=11,
        output_tokens=13,
        total_tokens=24,
        cached_input_tokens=0,
        reasoning_output_tokens=5,
    )

    cost = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        usage=usage,
    )

    assert cost == InvocationCost(
        pricing_version=QWEN_BEIJING_PRICING_VERSION,
        currency=QWEN_BEIJING_CURRENCY,
        estimated_cost=Decimal("0.0001068"),
    )
    assert isinstance(cost.estimated_cost, Decimal)


def test_long_context_applies_long_rates_to_the_full_request() -> None:
    usage = ModelUsage(
        input_tokens=QWEN_LONG_CONTEXT_THRESHOLD + 1,
        output_tokens=1_000,
        total_tokens=QWEN_LONG_CONTEXT_THRESHOLD + 1_001,
        cached_input_tokens=0,
        reasoning_output_tokens=250,
    )

    cost = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        usage=usage,
    )

    assert cost is not None
    assert cost.estimated_cost == Decimal("1.2576048")


def test_exact_long_context_threshold_still_uses_short_rates() -> None:
    usage = ModelUsage(
        input_tokens=QWEN_LONG_CONTEXT_THRESHOLD,
        output_tokens=1,
        total_tokens=QWEN_LONG_CONTEXT_THRESHOLD + 1,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_output_tokens=1,
    )

    cost = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        usage=usage,
    )

    assert cost is not None
    assert cost.estimated_cost == Decimal("0.3072072")


def test_reasoning_detail_is_observed_but_not_double_counted() -> None:
    without_reasoning = ModelUsage(
        input_tokens=10,
        output_tokens=20,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        reasoning_output_tokens=0,
    )
    with_reasoning = without_reasoning.model_copy(update={"reasoning_output_tokens": 15})

    first = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        usage=without_reasoning,
    )
    second = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        usage=with_reasoning,
    )

    assert first == second


def test_embedding_cost_uses_input_tokens_only() -> None:
    cost = QWEN_BEIJING_PRICE_BOOK.estimate(
        provider="qwen",
        model="text-embedding-v4",
        usage=ModelUsage(input_tokens=17, output_tokens=0, total_tokens=17),
    )

    assert cost is not None
    assert cost.estimated_cost == Decimal("0.0000085")


@pytest.mark.parametrize(
    ("provider", "model", "usage"),
    [
        (
            "fake",
            "qwen3.6-flash-2026-04-16",
            ModelUsage(
                input_tokens=1,
                output_tokens=1,
                cached_input_tokens=0,
                cache_write_input_tokens=0,
            ),
        ),
        ("qwen", "unknown-model", ModelUsage(input_tokens=1, output_tokens=1)),
        (
            "qwen",
            "qwen3.6-flash-2026-04-16",
            ModelUsage(input_tokens=1, output_tokens=1, cached_input_tokens=1),
        ),
        (
            "qwen",
            "qwen3.6-flash-2026-04-16",
            ModelUsage(input_tokens=1, output_tokens=1, cache_write_input_tokens=1),
        ),
        (
            "qwen",
            "qwen3.6-flash-2026-04-16",
            ModelUsage(input_tokens=QWEN_MAX_CONTEXT_TOKENS + 1, output_tokens=1),
        ),
    ],
)
def test_unknown_profile_or_incomplete_chat_usage_has_no_cost(
    provider: str,
    model: str,
    usage: ModelUsage,
) -> None:
    assert (
        QWEN_BEIJING_PRICE_BOOK.estimate(
            provider=provider,
            model=model,
            usage=usage,
        )
        is None
    )


def test_cost_contract_rejects_float_amounts() -> None:
    with pytest.raises(ValidationError):
        InvocationCost(
            pricing_version=QWEN_BEIJING_PRICING_VERSION,
            currency="CNY",
            estimated_cost=0.01,  # type: ignore[arg-type]
        )
