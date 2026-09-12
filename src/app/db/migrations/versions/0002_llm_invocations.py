"""Create workspace-scoped LLM invocation accounting."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_llm_invocations"
down_revision: str | None = "0001_identity_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_invocations",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("invocation_kind", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("graph_node", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=True),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("token_usage", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("latency_ms", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_category", sa.Text(), nullable=True),
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "invocation_kind IN ('chat', 'embedding')",
            name=op.f("ck_llm_invocations_invocation_kind"),
        ),
        sa.CheckConstraint(
            "provider IN ('fake', 'openai')",
            name=op.f("ck_llm_invocations_provider"),
        ),
        sa.CheckConstraint(
            "status IN ('started', 'succeeded', 'failed')",
            name=op.f("ck_llm_invocations_status"),
        ),
        sa.CheckConstraint(
            "error_category IS NULL OR error_category IN "
            "('provider_timeout', 'provider_error', 'cancelled', 'invalid_provider_response')",
            name=op.f("ck_llm_invocations_error_category"),
        ),
        sa.CheckConstraint(
            "graph_node ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_llm_invocations_graph_node"),
        ),
        sa.CheckConstraint(
            "request_hash ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_llm_invocations_request_hash"),
        ),
        sa.CheckConstraint(
            "(invocation_kind = 'chat' AND model = 'gpt-5.6-terra' "
            "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
            "(invocation_kind = 'embedding' AND model = 'text-embedding-3-small' "
            "AND prompt_version IS NULL)",
            name=op.f("ck_llm_invocations_profile"),
        ),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name=op.f("ck_llm_invocations_latency_ms"),
        ),
        sa.CheckConstraint(
            "(status = 'started' AND token_usage IS NULL AND latency_ms IS NULL "
            "AND error_category IS NULL) OR "
            "(status = 'succeeded' AND latency_ms IS NOT NULL AND error_category IS NULL) OR "
            "(status = 'failed' AND token_usage IS NULL AND latency_ms IS NOT NULL "
            "AND error_category IS NOT NULL)",
            name=op.f("ck_llm_invocations_terminal_fields"),
        ),
        sa.CheckConstraint(
            "status <> 'succeeded' OR "
            "(invocation_kind = 'chat' AND token_usage IS NOT NULL) OR "
            "(invocation_kind = 'embedding' AND token_usage IS NULL)",
            name=op.f("ck_llm_invocations_success_usage"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_llm_invocations_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_llm_invocations_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_llm_invocations")),
    )
    op.create_index(
        "ix_llm_invocations_workspace_id",
        "llm_invocations",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_llm_invocations_workspace_id_actor_user_id",
        "llm_invocations",
        ["workspace_id", "actor_user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_llm_invocations_workspace_id_actor_user_id",
        table_name="llm_invocations",
    )
    op.drop_index("ix_llm_invocations_workspace_id", table_name="llm_invocations")
    op.drop_table("llm_invocations")
