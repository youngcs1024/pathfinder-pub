"""Private resume source, versioned content/preferences and claim reviews."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0021_r31_resume_profiles"
down_revision = "0020_r22_project_facts"
branch_labels = None
depends_on = None


def _id():
    return sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()"))


def _ws():
    return sa.Column(
        "workspace_id",
        sa.Uuid(),
        sa.ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )


def _fk(columns, table, target):
    return sa.ForeignKeyConstraint(
        columns, [f"{table}.{part}" for part in target], ondelete="RESTRICT"
    )


def _created():
    return sa.Column(
        "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )


def upgrade() -> None:
    op.create_table(
        "resume_profiles",
        _id(),
        _ws(),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("current_preference_version", sa.Integer(), nullable=False),
        _created(),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "owner_user_id"),
        _fk(
            ["workspace_id", "owner_user_id"], "workspace_memberships", ["workspace_id", "user_id"]
        ),
        sa.CheckConstraint(
            "current_version >= 1 AND current_preference_version >= 1",
            name=op.f("ck_resume_profiles_versions"),
        ),
    )
    op.create_index(
        "ix_resume_profiles_workspace_owner", "resume_profiles", ["workspace_id", "owner_user_id"]
    )
    op.create_table(
        "resume_profile_imports",
        _id(),
        _ws(),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("template_commit", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=False),
        sa.Column("source_bytes", sa.LargeBinary(), nullable=False),
        _created(),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "profile_id", "id"),
        _fk(["workspace_id", "profile_id"], "resume_profiles", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "source_sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_resume_profile_imports_source_sha256")
        ),
        sa.CheckConstraint(
            "octet_length(source_bytes) BETWEEN 1 AND 131072",
            name=op.f("ck_resume_profile_imports_source_bytes"),
        ),
    )
    op.create_index(
        "ix_resume_profile_imports_workspace_profile",
        "resume_profile_imports",
        ["workspace_id", "profile_id"],
    )
    op.create_table(
        "resume_profile_versions",
        _id(),
        _ws(),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("source_import_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        _created(),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "profile_id", "version"),
        _fk(["workspace_id", "profile_id"], "resume_profiles", ["workspace_id", "id"]),
        _fk(
            ["workspace_id", "profile_id", "source_import_id"],
            "resume_profile_imports",
            ["workspace_id", "profile_id", "id"],
        ),
        _fk(
            ["workspace_id", "created_by_user_id"],
            "workspace_memberships",
            ["workspace_id", "user_id"],
        ),
        sa.CheckConstraint(
            "version >= 1 AND jsonb_typeof(content_json) = 'object'",
            name=op.f("ck_resume_profile_versions_content"),
        ),
    )
    op.create_index(
        "ix_resume_profile_versions_workspace_profile",
        "resume_profile_versions",
        ["workspace_id", "profile_id"],
    )
    op.create_table(
        "resume_preference_versions",
        _id(),
        _ws(),
        sa.Column("profile_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("preferences_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        _created(),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "profile_id", "version"),
        _fk(["workspace_id", "profile_id"], "resume_profiles", ["workspace_id", "id"]),
        _fk(
            ["workspace_id", "created_by_user_id"],
            "workspace_memberships",
            ["workspace_id", "user_id"],
        ),
        sa.CheckConstraint(
            "version >= 1 AND jsonb_typeof(preferences_json) = 'object'",
            name=op.f("ck_resume_preference_versions_preferences"),
        ),
    )
    op.create_index(
        "ix_resume_preference_versions_workspace_profile",
        "resume_preference_versions",
        ["workspace_id", "profile_id"],
    )
    op.create_table(
        "resume_source_claims",
        _id(),
        _ws(),
        sa.Column("source_import_id", sa.Uuid(), nullable=False),
        sa.Column("project_item_id", sa.Uuid(), nullable=False),
        sa.Column("item_id", sa.Uuid(), nullable=False),
        sa.Column("field", sa.Text(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("source_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("review_version", sa.Integer(), nullable=False),
        sa.UniqueConstraint("workspace_id", "id"),
        _fk(["workspace_id", "source_import_id"], "resume_profile_imports", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "review_version >= 0 AND char_length(claim_text) BETWEEN 1 AND 4000",
            name=op.f("ck_resume_source_claims_claim"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(source_json) = 'object'", name=op.f("ck_resume_source_claims_source")
        ),
    )
    op.create_index(
        "ix_resume_source_claims_workspace_import",
        "resume_source_claims",
        ["workspace_id", "source_import_id"],
    )
    op.create_table(
        "resume_claim_reviews",
        _id(),
        _ws(),
        sa.Column("claim_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("actor_user_id", sa.Uuid(), nullable=False),
        _created(),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "claim_id", "version"),
        _fk(["workspace_id", "claim_id"], "resume_source_claims", ["workspace_id", "id"]),
        _fk(["workspace_id", "project_id"], "material_projects", ["workspace_id", "id"]),
        _fk(
            ["workspace_id", "actor_user_id"], "workspace_memberships", ["workspace_id", "user_id"]
        ),
        sa.CheckConstraint(
            "version >= 1 AND decision IN ('linked', 'needs_evidence', 'excluded')",
            name=op.f("ck_resume_claim_reviews_decision"),
        ),
    )
    op.create_index(
        "ix_resume_claim_reviews_workspace_claim",
        "resume_claim_reviews",
        ["workspace_id", "claim_id"],
    )
    op.create_table(
        "resume_claim_fact_links",
        _id(),
        _ws(),
        sa.Column("review_id", sa.Uuid(), nullable=False),
        sa.Column("fact_version_id", sa.Uuid(), nullable=False),
        sa.UniqueConstraint("workspace_id", "review_id", "fact_version_id"),
        _fk(["workspace_id", "review_id"], "resume_claim_reviews", ["workspace_id", "id"]),
        _fk(["workspace_id", "fact_version_id"], "material_fact_versions", ["workspace_id", "id"]),
    )
    op.create_index(
        "ix_resume_claim_fact_links_workspace_review",
        "resume_claim_fact_links",
        ["workspace_id", "review_id"],
    )


def downgrade() -> None:
    tables = (
        "resume_claim_fact_links",
        "resume_claim_reviews",
        "resume_source_claims",
        "resume_preference_versions",
        "resume_profile_versions",
        "resume_profile_imports",
        "resume_profiles",
    )
    bind = op.get_bind()
    for table in tables:
        if bind.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar_one():
            raise RuntimeError("resume profile history cannot be safely downgraded")
    for table in tables:
        op.drop_table(table)
