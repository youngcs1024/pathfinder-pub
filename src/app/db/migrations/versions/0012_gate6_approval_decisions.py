"""Add Gate 6 approval decisions, decision events, and graph v5 metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_gate6_approval_decisions"
down_revision: str | None = "0011_gate6_graph_interrupt"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        op.f("uq_approval_requests_workspace_id_id"),
        "approval_requests",
        ["workspace_id", "id"],
    )
    op.create_table(
        "approval_decisions",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("approval_request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision IN ('approve', 'reject')",
            name=op.f("ck_approval_decisions_decision"),
        ),
        sa.CheckConstraint(
            "reason IS NULL OR char_length(reason) <= 1000",
            name=op.f("ck_approval_decisions_reason"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_approval_decisions_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "approval_request_id"],
            ["approval_requests.workspace_id", "approval_requests.id"],
            name="fk_approval_decisions_request",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_approval_decisions_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_approval_decisions")),
        sa.UniqueConstraint(
            "workspace_id",
            "approval_request_id",
            "actor_user_id",
            name=op.f("uq_approval_decisions_workspace_id_approval_request_id_actor_user_id"),
        ),
    )
    op.create_index(
        "ix_approval_decisions_workspace_id",
        "approval_decisions",
        ["workspace_id"],
        unique=False,
    )
    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        "type IN ('run.created', 'run.status_changed', 'run.completed', "
        "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
        "'agent.plan.created', 'agent.research.started', 'source.discovered', "
        "'tool.started', 'tool.finished', 'report.completed', 'rag.retrieved', "
        "'action.proposed', 'approval.expired', 'approval.decided', "
        "'action.cancelled')",
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', "
        "'pathfinder-research-v5')",
    )
    op.alter_column("runs", "graph_version", server_default=sa.text("'pathfinder-research-v5'"))


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs "
            "WHERE graph_version = 'pathfinder-research-v5') "
            "OR EXISTS (SELECT 1 FROM approval_decisions) "
            "OR EXISTS (SELECT 1 FROM run_events "
            "WHERE type IN ('approval.decided', 'action.cancelled'))"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 6 approval decision data cannot be safely downgraded")
    op.alter_column("runs", "graph_version", server_default=sa.text("'pathfinder-research-v4'"))
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4')",
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
    op.drop_index("ix_approval_decisions_workspace_id", table_name="approval_decisions")
    op.drop_table("approval_decisions")
    op.drop_constraint(
        op.f("uq_approval_requests_workspace_id_id"),
        "approval_requests",
        type_="unique",
    )
