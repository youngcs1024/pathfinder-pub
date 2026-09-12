from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.llm.ports import LOCKED_CHAT_MODEL, LOCKED_EMBEDDING_MODEL, ModelUsage

QWEN_BEIJING_PRICING_VERSION = "qwen-cn-beijing-cny-2026-08-13-v1"
QWEN_BEIJING_CURRENCY = "CNY"
QWEN_LONG_CONTEXT_THRESHOLD = 256_000
QWEN_MAX_CONTEXT_TOKENS = 1_000_000
_TOKENS_PER_MILLION = Decimal(1_000_000)


class InvocationCost(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        strict=True,
    )

    pricing_version: str = Field(min_length=1, max_length=100)
    currency: Literal["CNY"]
    estimated_cost: Decimal = Field(ge=0, max_digits=20, decimal_places=12)


@runtime_checkable
class PriceBookPort(Protocol):
    version: str
    currency: str

    def estimate(
        self,
        *,
        provider: str,
        model: str,
        usage: ModelUsage,
    ) -> InvocationCost | None: ...


@dataclass(frozen=True, slots=True)
class QwenBeijingPriceBook:
    version: str = QWEN_BEIJING_PRICING_VERSION
    currency: str = QWEN_BEIJING_CURRENCY

    def estimate(
        self,
        *,
        provider: str,
        model: str,
        usage: ModelUsage,
    ) -> InvocationCost | None:
        if provider != "qwen" or not isinstance(usage, ModelUsage):
            return None
        if model == LOCKED_CHAT_MODEL:
            amount = self._chat_cost(usage)
        elif model == LOCKED_EMBEDDING_MODEL:
            amount = Decimal(usage.input_tokens) * Decimal("0.5") / _TOKENS_PER_MILLION
        else:
            return None
        if amount is None:
            return None
        return InvocationCost(
            pricing_version=self.version,
            currency="CNY",
            estimated_cost=amount,
        )

    @staticmethod
    def _chat_cost(usage: ModelUsage) -> Decimal | None:
        if (
            (usage.cached_input_tokens or 0) != 0
            or (usage.cache_write_input_tokens or 0) != 0
            or usage.input_tokens > QWEN_MAX_CONTEXT_TOKENS
        ):
            return None

        if usage.input_tokens > QWEN_LONG_CONTEXT_THRESHOLD:
            input_rate = Decimal("4.8")
            output_rate = Decimal("28.8")
        else:
            input_rate = Decimal("1.2")
            output_rate = Decimal("7.2")

        return (
            Decimal(usage.input_tokens) * input_rate + Decimal(usage.output_tokens) * output_rate
        ) / _TOKENS_PER_MILLION


QWEN_BEIJING_PRICE_BOOK = QwenBeijingPriceBook()
