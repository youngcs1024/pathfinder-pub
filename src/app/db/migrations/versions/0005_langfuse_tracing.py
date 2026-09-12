"""Add optional Langfuse trace correlation identifiers."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_langfuse_tracing"
down_revision: str | None = "0004_llm_costs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "llm_invocations",
        sa.Column(
            "trace_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        op.f("ck_llm_invocations_trace_ids"),
        "llm_invocations",
        "trace_ids IS NULL OR (status IN ('succeeded', 'failed') "
        "AND jsonb_typeof(trace_ids) = 'object' "
        "AND trace_ids ? 'trace_id' AND trace_ids ? 'observation_id' "
        "AND trace_ids - 'trace_id' - 'observation_id' = '{}'::jsonb "
        "AND jsonb_typeof(trace_ids -> 'trace_id') = 'string' "
        "AND jsonb_typeof(trace_ids -> 'observation_id') = 'string' "
        "AND trace_ids ->> 'trace_id' ~ '^[0-9a-f]{32}$' "
        "AND trace_ids ->> 'trace_id' !~ '^0{32}$' "
        "AND trace_ids ->> 'observation_id' ~ '^[0-9a-f]{16}$' "
        "AND trace_ids ->> 'observation_id' !~ '^0{16}$')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_llm_invocations_trace_ids"),
        "llm_invocations",
        type_="check",
    )
    op.drop_column("llm_invocations", "trace_ids")
