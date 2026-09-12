"""Add Gate 6 graph interrupt resume binding and graph v4 metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_gate6_graph_interrupt"
down_revision: str | None = "0010_gate6_action_proposals"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "run_jobs",
        sa.Column(
            "resume_approval_request_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_run_jobs_resume_approval_request",
        "run_jobs",
        "approval_requests",
        ["workspace_id", "run_id", "resume_approval_request_id"],
        ["workspace_id", "run_id", "id"],
        ondelete="RESTRICT",
    )

    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4')",
    )
    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v4'"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs "
            "WHERE graph_version = 'pathfinder-research-v4') "
            "OR EXISTS (SELECT 1 FROM run_jobs "
            "WHERE resume_approval_request_id IS NOT NULL)"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 6 graph interrupt data cannot be safely downgraded")

    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v3'"),
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3')",
    )
    op.drop_constraint(
        "fk_run_jobs_resume_approval_request",
        "run_jobs",
        type_="foreignkey",
    )
    op.drop_column("run_jobs", "resume_approval_request_id")
