"""R5.2 immutable version confirmations."""

import sqlalchemy as sa
from alembic import op

revision = "0025_r52_resume_confirmations"
down_revision = "0024_r51_resume_revision"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text("""
        CREATE TABLE resume_confirmations (
            id uuid NOT NULL DEFAULT gen_random_uuid(),
            workspace_id uuid NOT NULL CONSTRAINT fk_resume_confirmations_workspace
                REFERENCES workspaces(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL,
            version_id uuid NOT NULL,
            artifact_id uuid NOT NULL,
            tex_sha256 text NOT NULL CONSTRAINT ck_resume_confirmations_tex_sha256
                CHECK (tex_sha256 ~ '^[0-9a-f]{64}$'),
            confirmed_by_user_id uuid NOT NULL,
            confirmed_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT pk_resume_confirmations PRIMARY KEY (id),
            CONSTRAINT uq_resume_confirmations_workspace_id_id UNIQUE (workspace_id,id),
            CONSTRAINT uq_resume_confirmations_workspace_id_session_id_version_id
                UNIQUE (workspace_id,session_id,version_id),
            CONSTRAINT fk_resume_confirmations_session
                FOREIGN KEY (workspace_id,session_id)
                REFERENCES resume_sessions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT fk_resume_confirmations_version
                FOREIGN KEY (workspace_id,version_id)
                REFERENCES resume_versions(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT fk_resume_confirmations_artifact
                FOREIGN KEY (workspace_id,artifact_id)
                REFERENCES resume_tex_artifacts(workspace_id,id) ON DELETE RESTRICT,
            CONSTRAINT fk_resume_confirmations_actor
                FOREIGN KEY (workspace_id,confirmed_by_user_id)
                REFERENCES workspace_memberships(workspace_id,user_id) ON DELETE RESTRICT
        )
    """)
    )
    op.create_index(
        "ix_resume_confirmations_workspace_session",
        "resume_confirmations",
        ["workspace_id", "session_id"],
    )


def downgrade() -> None:
    if (
        op.get_bind()
        .execute(sa.text("SELECT EXISTS (SELECT 1 FROM resume_confirmations)"))
        .scalar_one()
    ):
        raise RuntimeError("resume confirmations cannot be safely downgraded")
    op.drop_index("ix_resume_confirmations_workspace_session", table_name="resume_confirmations")
    op.drop_table("resume_confirmations")
