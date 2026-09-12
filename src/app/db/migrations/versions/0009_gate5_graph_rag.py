"""Add Gate 5 application binding, graph v2 history, and RAG observations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_gate5_graph_rag"
down_revision: str | None = "0008_rag_document_storage"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column("resume_document_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_runs_resume_document"),
        "runs",
        "documents",
        ["workspace_id", "resume_document_id"],
        ["workspace_id", "id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint(op.f("ck_runs_mode"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_mode"),
        "runs",
        "mode IN ('research', 'application')",
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2')",
    )
    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v2'"),
    )
    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        "type IN ('run.created', 'run.status_changed', 'run.completed', "
        "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
        "'agent.plan.created', 'agent.research.started', 'source.discovered', "
        "'tool.started', 'tool.finished', 'report.completed', 'rag.retrieved')",
    )


def downgrade() -> None:
    bind = op.get_bind()
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs WHERE mode <> 'research' "
            "OR resume_document_id IS NOT NULL OR graph_version <> 'pathfinder-research-v1' "
            "OR result_json->>'schema_version' = '2') "
            "OR EXISTS (SELECT 1 FROM run_events WHERE type = 'rag.retrieved')"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("Gate 5 data cannot be safely downgraded to the Gate 4 contract")

    op.drop_constraint(op.f("ck_run_events_type"), "run_events", type_="check")
    op.create_check_constraint(
        op.f("ck_run_events_type"),
        "run_events",
        "type IN ('run.created', 'run.status_changed', 'run.completed', "
        "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
        "'agent.plan.created', 'agent.research.started', 'source.discovered', "
        "'tool.started', 'tool.finished', 'report.completed')",
    )
    op.alter_column(
        "runs",
        "graph_version",
        server_default=sa.text("'pathfinder-research-v1'"),
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"), "runs", "graph_version = 'pathfinder-research-v1'"
    )
    op.drop_constraint(op.f("ck_runs_mode"), "runs", type_="check")
    op.create_check_constraint(op.f("ck_runs_mode"), "runs", "mode = 'research'")
    op.drop_constraint(op.f("fk_runs_resume_document"), "runs", type_="foreignkey")
    op.drop_column("runs", "resume_document_id")
