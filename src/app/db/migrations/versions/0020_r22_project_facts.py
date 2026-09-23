"""Versioned, workspace-scoped facts bound to immutable material evidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0020_r22_project_facts"
down_revision = "0019_r21_material_line_ranges"
branch_labels = None
depends_on = None


def _id() -> sa.Column:
    return sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()"))


def _workspace() -> sa.Column:
    return sa.Column(
        "workspace_id",
        sa.Uuid(),
        sa.ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )


def _fk(columns: list[str], table: str, target: list[str]) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        columns, [f"{table}.{part}" for part in target], ondelete="RESTRICT"
    )


def upgrade() -> None:
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', 'pathfinder-research-v5', "
        "'pathfinder-research-v6', 'pathfinder-resume-v1', 'pathfinder-resume-v2')",
    )
    op.create_table(
        "material_fact_sets",
        _id(),
        _workspace(),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("import_id", sa.Uuid(), nullable=False),
        sa.Column("cache_digest", sa.Text(), nullable=False),
        sa.Column("extractor_digest", sa.Text(), nullable=False),
        sa.Column("complete", sa.Boolean(), nullable=False),
        sa.Column("issues_json", JSONB(none_as_null=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "project_id", "cache_digest", "extractor_digest"),
        _fk(["workspace_id", "project_id"], "material_projects", ["workspace_id", "id"]),
        _fk(["workspace_id", "import_id"], "material_imports", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "cache_digest ~ '^[0-9a-f]{64}$'", name=op.f("ck_material_fact_sets_cache_digest")
        ),
        sa.CheckConstraint(
            "extractor_digest ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_material_fact_sets_extractor_digest"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(issues_json) = 'array'", name=op.f("ck_material_fact_sets_issues_json")
        ),
    )
    op.create_index(
        "ix_material_fact_sets_workspace_project",
        "material_fact_sets",
        ["workspace_id", "project_id"],
    )
    op.create_table(
        "material_facts",
        _id(),
        _workspace(),
        sa.Column("fact_set_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "fact_set_id", "ordinal"),
        _fk(["workspace_id", "fact_set_id"], "material_fact_sets", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "ordinal >= 0 AND current_version > 0", name=op.f("ck_material_facts_version")
        ),
    )
    op.create_index(
        "ix_material_facts_workspace_set", "material_facts", ["workspace_id", "fact_set_id"]
    )
    op.create_table(
        "material_fact_versions",
        _id(),
        _workspace(),
        sa.Column("fact_id", sa.Uuid(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("conditions_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("review_status", sa.Text(), nullable=False),
        sa.Column("issues_json", JSONB(none_as_null=True), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "fact_id", "version"),
        _fk(["workspace_id", "fact_id"], "material_facts", ["workspace_id", "id"]),
        _fk(
            ["workspace_id", "created_by_user_id"],
            "workspace_memberships",
            ["workspace_id", "user_id"],
        ),
        sa.CheckConstraint(
            "version > 0 AND char_length(claim) BETWEEN 1 AND 2000",
            name=op.f("ck_material_fact_versions_claim"),
        ),
        sa.CheckConstraint(
            "kind IN ('implementation', 'plan', 'experiment', 'personal_statement')",
            name=op.f("ck_material_fact_versions_kind"),
        ),
        sa.CheckConstraint(
            "review_status IN ('pending', 'confirmed', 'rejected')",
            name=op.f("ck_material_fact_versions_review"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(conditions_json) = 'object' AND jsonb_typeof(issues_json) = 'array'",
            name=op.f("ck_material_fact_versions_json"),
        ),
    )
    op.create_index(
        "ix_material_fact_versions_workspace_fact",
        "material_fact_versions",
        ["workspace_id", "fact_id"],
    )
    op.create_table(
        "material_fact_evidence",
        _id(),
        _workspace(),
        sa.Column("fact_version_id", sa.Uuid(), nullable=False),
        sa.Column("snapshot_file_id", sa.Uuid(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "workspace_id", "fact_version_id", "snapshot_file_id", "start_line", "end_line"
        ),
        _fk(["workspace_id", "fact_version_id"], "material_fact_versions", ["workspace_id", "id"]),
        _fk(
            ["workspace_id", "snapshot_file_id"], "material_snapshot_files", ["workspace_id", "id"]
        ),
        sa.CheckConstraint(
            "start_line > 0 AND end_line >= start_line AND end_line - start_line < 80",
            name=op.f("ck_material_fact_evidence_lines"),
        ),
        sa.CheckConstraint(
            "char_length(quote) BETWEEN 1 AND 4000", name=op.f("ck_material_fact_evidence_quote")
        ),
    )
    op.create_index(
        "ix_material_fact_evidence_workspace_version",
        "material_fact_evidence",
        ["workspace_id", "fact_version_id"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    for table in (
        "material_fact_evidence",
        "material_fact_versions",
        "material_facts",
        "material_fact_sets",
    ):
        if bind.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar_one():
            raise RuntimeError("project fact history cannot be safely downgraded")
    for table in (
        "material_fact_evidence",
        "material_fact_versions",
        "material_facts",
        "material_fact_sets",
    ):
        op.drop_table(table)
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', 'pathfinder-research-v5', "
        "'pathfinder-research-v6', 'pathfinder-resume-v1')",
    )
