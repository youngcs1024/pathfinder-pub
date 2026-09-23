"""Immutable deterministic TeX artifacts for resume rendering."""

import sqlalchemy as sa
from alembic import op

revision = "0022_r32_resume_tex_artifacts"
down_revision = "0021_r31_resume_profiles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "resume_tex_artifacts",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspaces.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("profile_version_id", sa.Uuid(), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("template_commit", sa.Text(), nullable=False),
        sa.Column("template_source_sha256", sa.Text(), nullable=False),
        sa.Column("preamble_sha256", sa.Text(), nullable=False),
        sa.Column("renderer_version", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("config_sha256", sa.Text(), nullable=False),
        sa.Column("tex_sha256", sa.Text(), nullable=False),
        sa.Column("tex_bytes", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint(
            "workspace_id",
            "profile_version_id",
            "template_source_sha256",
            "content_sha256",
            "config_sha256",
            "renderer_version",
            name="uq_resume_tex_artifacts_identity",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "profile_version_id"],
            ["resume_profile_versions.workspace_id", "resume_profile_versions.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "octet_length(tex_bytes) BETWEEN 1 AND 524288",
            name=op.f("ck_resume_tex_artifacts_tex_bytes"),
        ),
        *(
            sa.CheckConstraint(
                f"{field} ~ '^[0-9a-f]{{64}}$'",
                name=op.f(f"ck_resume_tex_artifacts_{field}"),
            )
            for field in (
                "template_source_sha256",
                "preamble_sha256",
                "content_sha256",
                "config_sha256",
                "tex_sha256",
            )
        ),
    )
    op.create_index(
        "ix_resume_tex_artifacts_workspace_profile_version",
        "resume_tex_artifacts",
        ["workspace_id", "profile_version_id"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text("SELECT EXISTS (SELECT 1 FROM resume_tex_artifacts)")).scalar_one():
        raise RuntimeError("resume artifact history cannot be safely downgraded")
    op.drop_table("resume_tex_artifacts")
