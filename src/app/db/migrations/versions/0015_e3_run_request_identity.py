"""Store scoped Run creation request identities without backfilling legacy requests."""

import sqlalchemy as sa
from alembic import op

revision: str = "0015_e3_run_request_identity"
down_revision: str | None = "0014_gate6_action_recovery"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("client_request_id", sa.Uuid(), nullable=True))
    op.add_column("runs", sa.Column("create_request_digest", sa.Text(), nullable=True))
    op.add_column("runs", sa.Column("create_request_version", sa.Integer(), nullable=True))
    op.create_check_constraint(
        op.f("ck_runs_create_request_identity"),
        "runs",
        "(client_request_id IS NULL AND create_request_digest IS NULL "
        "AND create_request_version IS NULL) OR "
        "(client_request_id IS NOT NULL AND create_request_digest IS NOT NULL "
        "AND create_request_version IS NOT NULL AND create_request_version = 1 "
        "AND create_request_digest ~ '^[0-9a-f]{64}$')",
    )
    op.create_unique_constraint(
        "uq_runs_workspace_creator_request",
        "runs",
        ["workspace_id", "created_by_user_id", "client_request_id"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    # Hold the same lock through the check and DDL so no keyed insert can race it.
    bind.execute(sa.text("LOCK TABLE runs IN ACCESS EXCLUSIVE MODE"))
    unsafe = bind.execute(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM runs WHERE client_request_id IS NOT NULL "
            "OR create_request_digest IS NOT NULL OR create_request_version IS NOT NULL)"
        )
    ).scalar_one()
    if unsafe:
        raise RuntimeError("E3 request identity data cannot be safely downgraded")

    op.drop_constraint("uq_runs_workspace_creator_request", "runs", type_="unique")
    op.drop_constraint(op.f("ck_runs_create_request_identity"), "runs", type_="check")
    op.drop_column("runs", "create_request_version")
    op.drop_column("runs", "create_request_digest")
    op.drop_column("runs", "client_request_id")
