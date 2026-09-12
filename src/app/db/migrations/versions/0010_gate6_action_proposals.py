"""Add Gate 6 action proposals, approval requests, and graph v3 history."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_gate6_action_proposals"
down_revision: str | None = "0009_gate5_graph_rag"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _id_column() -> sa.Column:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        server_default=sa.text("gen_random_uuid()"),
        nullable=False,
    )


def _timestamp_columns() -> tuple[sa.Column, sa.Column]:
    return (
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
    )


def upgrade() -> None:
    op.create_table(
        "action_intents",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("originating_actor_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_key", sa.Text(), nullable=False),
        sa.Column("action_revision", sa.BigInteger(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("effect", sa.Text(), nullable=False),
        sa.Column("args_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("canonicalization_version", sa.BigInteger(), nullable=False),
        sa.Column("args_digest", sa.Text(), nullable=False),
        sa.Column("target_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("target_canonicalization_version", sa.BigInteger(), nullable=False),
        sa.Column("target_digest", sa.Text(), nullable=False),
        sa.Column("approval_binding_version", sa.BigInteger(), nullable=False),
        sa.Column("approval_binding_digest", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'proposed'"), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column(
            "recovery_attempts",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        _id_column(),
        *_timestamp_columns(),
        sa.CheckConstraint(
            "action_key ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_action_intents_action_key"),
        ),
        sa.CheckConstraint("action_revision >= 1", name=op.f("ck_action_intents_action_revision")),
        sa.CheckConstraint(
            "tool_name ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_action_intents_tool_name"),
        ),
        sa.CheckConstraint(
            "effect IN ('read_only', 'reversible', 'irreversible')",
            name=op.f("ck_action_intents_effect"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(args_snapshot) = 'object'",
            name=op.f("ck_action_intents_args_snapshot"),
        ),
        sa.CheckConstraint(
            "canonicalization_version >= 1",
            name=op.f("ck_action_intents_canonicalization_version"),
        ),
        sa.CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_action_intents_args_digest"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(target_snapshot) = 'object'",
            name=op.f("ck_action_intents_target_snapshot"),
        ),
        sa.CheckConstraint(
            "target_canonicalization_version >= 1",
            name=op.f("ck_action_intents_target_canonicalization_version"),
        ),
        sa.CheckConstraint(
            "target_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_action_intents_target_digest"),
        ),
        sa.CheckConstraint(
            "approval_binding_version >= 1",
            name=op.f("ck_action_intents_approval_binding_version"),
        ),
        sa.CheckConstraint(
            "approval_binding_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_action_intents_approval_binding_digest"),
        ),
        sa.CheckConstraint(
            "status IN ('proposed', 'authorized', 'executing', 'succeeded', "
            "'failed', 'outcome_unknown', 'cancelled')",
            name=op.f("ck_action_intents_status"),
        ),
        sa.CheckConstraint(
            "idempotency_key ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'",
            name=op.f("ck_action_intents_idempotency_key"),
        ),
        sa.CheckConstraint(
            "recovery_attempts >= 0",
            name=op.f("ck_action_intents_recovery_attempts"),
        ),
        sa.CheckConstraint(
            "result IS NULL OR jsonb_typeof(result) = 'object'",
            name=op.f("ck_action_intents_result"),
        ),
        sa.CheckConstraint(
            "evidence IS NULL OR jsonb_typeof(evidence) = 'object'",
            name=op.f("ck_action_intents_evidence"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_action_intents_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_action_intents_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_action_intents_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_action_intents")),
        sa.UniqueConstraint("workspace_id", "id", name=op.f("uq_action_intents_workspace_id_id")),
        sa.UniqueConstraint(
            "workspace_id",
            "run_id",
            "id",
            name=op.f("uq_action_intents_workspace_id_run_id_id"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "run_id",
            "action_key",
            "action_revision",
            name="uq_action_intents_logical_revision",
        ),
    )
    op.create_index(
        "ix_action_intents_workspace_id",
        "action_intents",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_action_intents_workspace_run",
        "action_intents",
        ["workspace_id", "run_id"],
        unique=False,
    )

    op.create_table(
        "approval_requests",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action_intent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("args_digest", sa.Text(), nullable=False),
        sa.Column("target_digest", sa.Text(), nullable=False),
        sa.Column("approval_binding_version", sa.BigInteger(), nullable=False),
        sa.Column("approval_binding_digest", sa.Text(), nullable=False),
        sa.Column("policy_version", sa.BigInteger(), nullable=False),
        sa.Column("policy_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("version", sa.BigInteger(), server_default=sa.text("1"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        _id_column(),
        *_timestamp_columns(),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'consumed', 'expired')",
            name=op.f("ck_approval_requests_status"),
        ),
        sa.CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_approval_requests_args_digest"),
        ),
        sa.CheckConstraint(
            "target_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_approval_requests_target_digest"),
        ),
        sa.CheckConstraint(
            "approval_binding_version >= 1",
            name=op.f("ck_approval_requests_approval_binding_version"),
        ),
        sa.CheckConstraint(
            "approval_binding_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_approval_requests_approval_binding_digest"),
        ),
        sa.CheckConstraint("policy_version >= 1", name=op.f("ck_approval_requests_policy_version")),
        sa.CheckConstraint(
            "jsonb_typeof(policy_snapshot) = 'object'",
            name=op.f("ck_approval_requests_policy_snapshot"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_approval_requests_version")),
        sa.CheckConstraint(
            "(status = 'consumed' AND consumed_at IS NOT NULL) OR "
            "(status <> 'consumed' AND consumed_at IS NULL)",
            name=op.f("ck_approval_requests_consumed_at"),
        ),
        sa.CheckConstraint(
            "consumed_at IS NULL OR consumed_at >= created_at",
            name=op.f("ck_approval_requests_consumed_time_order"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_approval_requests_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id", "action_intent_id"],
            ["action_intents.workspace_id", "action_intents.run_id", "action_intents.id"],
            name="fk_approval_requests_action_intent",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_approval_requests")),
        sa.UniqueConstraint(
            "workspace_id",
            "action_intent_id",
            name=op.f("uq_approval_requests_workspace_id_action_intent_id"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "run_id",
            "id",
            name=op.f("uq_approval_requests_workspace_id_run_id_id"),
        ),
    )
    op.create_index(
        "ix_approval_requests_workspace_id",
        "approval_requests",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_approval_requests_workspace_run",
        "approval_requests",
        ["workspace_id", "run_id"],
        unique=False,
    )
    op.create_index(
        "ix_approval_requests_due",
        "approval_requests",
        ["workspace_id", "status", "expires_at"],
        unique=False,
    )

    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3')",
    )
    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v3'"),
    )
    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        "type IN ('run.created', 'run.status_changed', 'run.completed', "
        "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
        "'agent.plan.created', 'agent.research.started', 'source.discovered', "
        "'tool.started', 'tool.finished', 'report.completed', 'rag.retrieved', "
        "'action.proposed', 'approval.expired')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs "
            "WHERE graph_version = 'pathfinder-research-v3') "
            "OR EXISTS (SELECT 1 FROM action_intents) "
            "OR EXISTS (SELECT 1 FROM approval_requests) "
            "OR EXISTS (SELECT 1 FROM run_events "
            "WHERE type IN ('action.proposed', 'approval.expired'))"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 6 data cannot be safely downgraded to the Gate 5 contract")

    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        "type IN ('run.created', 'run.status_changed', 'run.completed', "
        "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
        "'agent.plan.created', 'agent.research.started', 'source.discovered', "
        "'tool.started', 'tool.finished', 'report.completed', 'rag.retrieved')",
    )
    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v2'"),
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2')",
    )
    op.drop_table("approval_requests")
    op.drop_table("action_intents")
