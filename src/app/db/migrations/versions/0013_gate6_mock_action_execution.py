"""Add Gate 6.4 irreversible execution, Mock Portal, and graph v6 metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_gate6_mock_action_execution"
down_revision: str | None = "0012_gate6_approval_decisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tool_invocations",
        sa.Column("action_intent_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.drop_constraint(op.f("ck_tool_invocations_gate4_effect"), "tool_invocations", type_="check")
    op.create_check_constraint(
        op.f("ck_tool_invocations_action_binding"),
        "tool_invocations",
        "(effect = 'irreversible' AND action_intent_id IS NOT NULL) OR "
        "(effect = 'read_only' AND action_intent_id IS NULL)",
    )
    op.create_foreign_key(
        "fk_tool_invocations_action_intent",
        "tool_invocations",
        "action_intents",
        ["workspace_id", "run_id", "action_intent_id"],
        ["workspace_id", "run_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_tool_invocations_action_intent_id",
        "tool_invocations",
        ["action_intent_id"],
    )

    op.create_table(
        "mock_submissions",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("originating_actor_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_intent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("payload_digest", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("external_ref", sa.Text(), nullable=False),
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
        sa.CheckConstraint(
            "idempotency_key ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'",
            name=op.f("ck_mock_submissions_idempotency_key"),
        ),
        sa.CheckConstraint(
            "payload_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_mock_submissions_payload_digest"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name=op.f("ck_mock_submissions_payload"),
        ),
        sa.CheckConstraint(
            "char_length(external_ref) BETWEEN 1 AND 200 AND btrim(external_ref) = external_ref",
            name=op.f("ck_mock_submissions_external_ref"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_mock_submissions_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_mock_submissions_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id", "action_intent_id"],
            ["action_intents.workspace_id", "action_intents.run_id", "action_intents.id"],
            name="fk_mock_submissions_action_intent",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_mock_submissions")),
        sa.UniqueConstraint("idempotency_key", name="uq_mock_submissions_idempotency_key"),
        sa.UniqueConstraint("action_intent_id", name="uq_mock_submissions_action_intent_id"),
    )
    op.create_index(
        "ix_mock_submissions_workspace_id",
        "mock_submissions",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_mock_submissions_workspace_run",
        "mock_submissions",
        ["workspace_id", "run_id"],
        unique=False,
    )

    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', "
        "'pathfinder-research-v5', 'pathfinder-research-v6')",
    )
    op.alter_column("runs", "graph_version", server_default=sa.text("'pathfinder-research-v6'"))


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs "
            "WHERE graph_version = 'pathfinder-research-v6') "
            "OR EXISTS (SELECT 1 FROM mock_submissions) "
            "OR EXISTS (SELECT 1 FROM tool_invocations WHERE action_intent_id IS NOT NULL)"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 6.4 execution data cannot be safely downgraded")

    op.alter_column("runs", "graph_version", server_default=sa.text("'pathfinder-research-v5'"))
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', "
        "'pathfinder-research-v5')",
    )
    op.drop_index("ix_mock_submissions_workspace_run", table_name="mock_submissions")
    op.drop_index("ix_mock_submissions_workspace_id", table_name="mock_submissions")
    op.drop_table("mock_submissions")
    op.drop_constraint("uq_tool_invocations_action_intent_id", "tool_invocations", type_="unique")
    op.drop_constraint("fk_tool_invocations_action_intent", "tool_invocations", type_="foreignkey")
    op.drop_constraint(
        op.f("ck_tool_invocations_action_binding"), "tool_invocations", type_="check"
    )
    op.create_check_constraint(
        op.f("ck_tool_invocations_gate4_effect"),
        "tool_invocations",
        "effect = 'read_only'",
    )
    op.drop_column("tool_invocations", "action_intent_id")
