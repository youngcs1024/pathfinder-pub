"""Add versioned invocation-level LLM cost estimates."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_llm_costs"
down_revision: str | None = "0003_openai_adapters"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_invocations",
        sa.Column("pricing_version", sa.Text(), nullable=True),
    )
    op.add_column(
        "llm_invocations",
        sa.Column("currency", sa.Text(), nullable=True),
    )
    op.add_column(
        "llm_invocations",
        sa.Column("estimated_cost", sa.Numeric(precision=20, scale=12), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_pricing_version"),
        "llm_invocations",
        "pricing_version IS NULL OR "
        "(char_length(pricing_version) BETWEEN 1 AND 100 "
        "AND pricing_version ~ '^[a-z0-9][a-z0-9._-]*$')",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_currency"),
        "llm_invocations",
        "currency IS NULL OR currency = 'USD'",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_estimated_cost"),
        "llm_invocations",
        "estimated_cost IS NULL OR estimated_cost >= 0",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_cost_fields"),
        "llm_invocations",
        "(pricing_version IS NULL AND currency IS NULL AND estimated_cost IS NULL) OR "
        "(pricing_version IS NOT NULL AND currency IS NOT NULL "
        "AND estimated_cost IS NOT NULL AND provider = 'openai' "
        "AND status = 'succeeded' AND token_usage IS NOT NULL)",
    )


def downgrade() -> None:
    for constraint_name in (
        "ck_llm_invocations_cost_fields",
        "ck_llm_invocations_estimated_cost",
        "ck_llm_invocations_currency",
        "ck_llm_invocations_pricing_version",
    ):
        op.drop_constraint(op.f(constraint_name), "llm_invocations", type_="check")
    op.drop_column("llm_invocations", "estimated_cost")
    op.drop_column("llm_invocations", "currency")
    op.drop_column("llm_invocations", "pricing_version")
