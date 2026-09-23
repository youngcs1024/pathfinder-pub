"""Persist scoped resume command receipts without changing legacy Run identities."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0017_r12_resume_commands"
down_revision: str | None = "0016_r1_execution_contracts"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "resume_commands",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("actor_user_id", sa.Uuid(), nullable=False),
        sa.Column("client_request_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("digest_version", sa.Integer(), nullable=False),
        sa.Column("request_digest", sa.Text(), nullable=False),
        sa.Column("receipt_version", sa.Integer(), nullable=False),
        sa.Column("receipt_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("kind ~ '^[a-z][a-z0-9_]{0,63}$'", name=op.f("ck_resume_commands_kind")),
        sa.CheckConstraint("digest_version = 1", name=op.f("ck_resume_commands_digest_version")),
        sa.CheckConstraint(
            "request_digest ~ '^[0-9a-f]{64}$'", name=op.f("ck_resume_commands_request_digest")
        ),
        sa.CheckConstraint("receipt_version = 1", name=op.f("ck_resume_commands_receipt_version")),
        sa.CheckConstraint(
            "jsonb_typeof(receipt_json) = 'object'", name=op.f("ck_resume_commands_receipt_json")
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "actor_user_id",
            "client_request_id",
            name="uq_resume_commands_actor_request",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_resume_commands_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_resume_commands_run",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_resume_commands_workspace_id", "resume_commands", ["workspace_id"])


def downgrade() -> None:
    bind = op.get_bind()
    bind.execute(sa.text("LOCK TABLE resume_commands IN ACCESS EXCLUSIVE MODE"))
    if bind.execute(sa.text("SELECT EXISTS (SELECT 1 FROM resume_commands)")).scalar_one():
        raise RuntimeError("resume command receipts cannot be safely downgraded")
    op.drop_index("ix_resume_commands_workspace_id", table_name="resume_commands")
    op.drop_table("resume_commands")
