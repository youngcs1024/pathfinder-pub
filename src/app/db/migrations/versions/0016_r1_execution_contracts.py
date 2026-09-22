"""Expand execution modes without rewriting historical rows or enabling a new graph."""

import sqlalchemy as sa
from alembic import op

revision: str = "0016_r1_execution_contracts"
down_revision: str | None = "0015_e3_run_request_identity"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.drop_constraint(op.f("ck_runs_mode"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_mode"),
        "runs",
        "mode IN ('research', 'application', 'material_preparation', "
        "'resume_generation', 'resume_revision')",
    )
    op.create_check_constraint(
        op.f("ck_runs_mode_graph_family"),
        "runs",
        "(mode IN ('research', 'application') "
        "AND graph_version LIKE 'pathfinder-research-v%') "
        "OR (mode IN ('material_preparation', 'resume_generation', 'resume_revision') "
        "AND graph_version LIKE 'pathfinder-resume-v%')",
    )
    op.create_check_constraint(
        op.f("ck_runs_resume_no_legacy_approval"),
        "runs",
        "mode IN ('research', 'application') OR status <> 'waiting_approval'",
    )
    op.alter_column("runs", "mode", server_default=None)
    op.alter_column("runs", "graph_version", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("LOCK TABLE runs IN ACCESS EXCLUSIVE MODE"))
    if bind.execute(
        sa.text("SELECT EXISTS (SELECT 1 FROM runs WHERE mode NOT IN ('research', 'application'))")
    ).scalar_one():
        raise RuntimeError("resume execution data cannot be safely downgraded")
    op.drop_constraint(op.f("ck_runs_resume_no_legacy_approval"), "runs", type_="check")
    op.drop_constraint(op.f("ck_runs_mode_graph_family"), "runs", type_="check")
    op.drop_constraint(op.f("ck_runs_mode"), "runs", type_="check")
    op.create_check_constraint(op.f("ck_runs_mode"), "runs", "mode IN ('research', 'application')")
    op.alter_column("runs", "mode", server_default=sa.text("'research'"))
    op.alter_column("runs", "graph_version", server_default=sa.text("'pathfinder-research-v6'"))
