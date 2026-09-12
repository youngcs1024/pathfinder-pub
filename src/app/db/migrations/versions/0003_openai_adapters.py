"""Add OpenAI adapter invocation metadata and error categories."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_openai_adapters"
down_revision: str | None = "0002_llm_invocations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_invocations",
        sa.Column("provider_response_id", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_provider_response_id"),
        "llm_invocations",
        "provider_response_id IS NULL OR "
        "(char_length(provider_response_id) BETWEEN 1 AND 512 "
        "AND btrim(provider_response_id) = provider_response_id)",
    )
    _drop_changed_constraints()
    op.create_check_constraint(
        op.f("ck_llm_invocations_error_category"),
        "llm_invocations",
        "error_category IS NULL OR error_category IN "
        "('rate_limited', 'provider_timeout', 'provider_unavailable', "
        "'provider_authentication', 'provider_rejected', 'provider_error', "
        "'cancelled', 'invalid_provider_response')",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_terminal_fields"),
        "llm_invocations",
        "(status = 'started' AND token_usage IS NULL AND latency_ms IS NULL "
        "AND provider_response_id IS NULL AND error_category IS NULL) OR "
        "(status = 'succeeded' AND latency_ms IS NOT NULL AND error_category IS NULL) OR "
        "(status = 'failed' AND token_usage IS NULL AND latency_ms IS NOT NULL "
        "AND error_category IS NOT NULL)",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_success_usage"),
        "llm_invocations",
        "status <> 'succeeded' OR "
        "(invocation_kind = 'chat' AND token_usage IS NOT NULL) OR "
        "invocation_kind = 'embedding'",
    )


def downgrade() -> None:
    _drop_changed_constraints()
    op.drop_constraint(
        op.f("ck_llm_invocations_provider_response_id"),
        "llm_invocations",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_error_category"),
        "llm_invocations",
        "error_category IS NULL OR error_category IN "
        "('provider_timeout', 'provider_error', 'cancelled', 'invalid_provider_response')",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_terminal_fields"),
        "llm_invocations",
        "(status = 'started' AND token_usage IS NULL AND latency_ms IS NULL "
        "AND error_category IS NULL) OR "
        "(status = 'succeeded' AND latency_ms IS NOT NULL AND error_category IS NULL) OR "
        "(status = 'failed' AND token_usage IS NULL AND latency_ms IS NOT NULL "
        "AND error_category IS NOT NULL)",
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_success_usage"),
        "llm_invocations",
        "status <> 'succeeded' OR "
        "(invocation_kind = 'chat' AND token_usage IS NOT NULL) OR "
        "(invocation_kind = 'embedding' AND token_usage IS NULL)",
    )
    op.drop_column("llm_invocations", "provider_response_id")


def _drop_changed_constraints() -> None:
    for constraint_name in (
        "ck_llm_invocations_error_category",
        "ck_llm_invocations_terminal_fields",
        "ck_llm_invocations_success_usage",
    ):
        op.drop_constraint(op.f(constraint_name), "llm_invocations", type_="check")
