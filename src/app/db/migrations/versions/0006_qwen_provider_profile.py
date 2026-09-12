"""Add the Qwen Beijing provider profile while preserving legacy OpenAI evidence."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_qwen_provider_profile"
down_revision: str | None = "0005_langfuse_tracing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDER_CONSTRAINT = "ck_llm_invocations_provider"
_PROFILE_CONSTRAINT = "ck_llm_invocations_profile"
_CURRENCY_CONSTRAINT = "ck_llm_invocations_currency"
_COST_FIELDS_CONSTRAINT = "ck_llm_invocations_cost_fields"

_LEGACY_PROFILE = (
    "(invocation_kind = 'chat' AND model = 'gpt-5.6-terra' "
    "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
    "(invocation_kind = 'embedding' AND model = 'text-embedding-3-small' "
    "AND prompt_version IS NULL)"
)
_QWEN_PROFILE = (
    "(invocation_kind = 'chat' AND model = 'qwen3.6-flash-2026-04-16' "
    "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
    "(invocation_kind = 'embedding' AND model = 'text-embedding-v4' "
    "AND prompt_version IS NULL)"
)


def _drop_changed_constraints() -> None:
    for name in (
        _COST_FIELDS_CONSTRAINT,
        _CURRENCY_CONSTRAINT,
        _PROFILE_CONSTRAINT,
        _PROVIDER_CONSTRAINT,
    ):
        op.drop_constraint(op.f(name), "llm_invocations", type_="check")


def upgrade() -> None:
    _drop_changed_constraints()
    op.create_check_constraint(
        op.f(_PROVIDER_CONSTRAINT),
        "llm_invocations",
        "provider IN ('fake', 'openai', 'qwen')",
    )
    op.create_check_constraint(
        op.f(_PROFILE_CONSTRAINT),
        "llm_invocations",
        "(provider = 'openai' AND (" + _LEGACY_PROFILE + ")) OR "
        "(provider = 'qwen' AND (" + _QWEN_PROFILE + ")) OR "
        "(provider = 'fake' AND ((" + _LEGACY_PROFILE + ") OR (" + _QWEN_PROFILE + ")))",
    )
    op.create_check_constraint(
        op.f(_CURRENCY_CONSTRAINT),
        "llm_invocations",
        "currency IS NULL OR currency IN ('USD', 'CNY')",
    )
    op.create_check_constraint(
        op.f(_COST_FIELDS_CONSTRAINT),
        "llm_invocations",
        "(pricing_version IS NULL AND currency IS NULL AND estimated_cost IS NULL) OR "
        "(pricing_version IS NOT NULL AND currency IS NOT NULL "
        "AND estimated_cost IS NOT NULL AND status = 'succeeded' "
        "AND token_usage IS NOT NULL AND "
        "((provider = 'openai' AND currency = 'USD') OR "
        "(provider = 'qwen' AND currency = 'CNY')))",
    )


def downgrade() -> None:
    connection = op.get_bind()
    incompatible_count = connection.scalar(
        sa.text(
            "SELECT count(*) FROM llm_invocations "
            "WHERE provider = 'qwen' "
            "OR model IN ('qwen3.6-flash-2026-04-16', 'text-embedding-v4')"
        )
    )
    if incompatible_count:
        raise RuntimeError("cannot downgrade 0006 while Qwen provider-profile evidence exists")

    _drop_changed_constraints()
    op.create_check_constraint(
        op.f(_PROVIDER_CONSTRAINT),
        "llm_invocations",
        "provider IN ('fake', 'openai')",
    )
    op.create_check_constraint(
        op.f(_PROFILE_CONSTRAINT),
        "llm_invocations",
        _LEGACY_PROFILE,
    )
    op.create_check_constraint(
        op.f(_CURRENCY_CONSTRAINT),
        "llm_invocations",
        "currency IS NULL OR currency = 'USD'",
    )
    op.create_check_constraint(
        op.f(_COST_FIELDS_CONSTRAINT),
        "llm_invocations",
        "(pricing_version IS NULL AND currency IS NULL AND estimated_cost IS NULL) OR "
        "(pricing_version IS NOT NULL AND currency IS NOT NULL "
        "AND estimated_cost IS NOT NULL AND provider = 'openai' "
        "AND status = 'succeeded' AND token_usage IS NOT NULL)",
    )
