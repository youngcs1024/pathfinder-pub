"""Persist R2.1 material projects, imports, and immutable source evidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import BYTEA, JSONB

revision = "0018_r21_material_snapshots"
down_revision = "0017_r12_resume_commands"
branch_labels = None
depends_on = None


def _identity_columns() -> list[sa.Column]:
    return [
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "workspace_id",
            sa.Uuid(),
            sa.ForeignKey("workspaces.id", ondelete="RESTRICT"),
            nullable=False,
        ),
    ]


def _workspace_fk(columns: list[str], table: str, target: list[str]) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        columns, [f"{table}.{name}" for name in target], ondelete="RESTRICT"
    )


def upgrade() -> None:
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', 'pathfinder-research-v5', "
        "'pathfinder-research-v6', 'pathfinder-resume-v1')",
    )
    op.drop_constraint(op.f("ck_documents_source_type"), "documents", type_="check")
    op.create_check_constraint(
        op.f("ck_documents_source_type"), "documents", "source_type IN ('markdown', 'text', 'code')"
    )
    op.drop_constraint(op.f("ck_documents_content"), "documents", type_="check")
    op.create_check_constraint(
        op.f("ck_documents_content"), "documents", "octet_length(content) BETWEEN 1 AND 1048576"
    )
    op.add_column("document_chunks", sa.Column("start_line", sa.BigInteger(), nullable=True))
    op.add_column("document_chunks", sa.Column("end_line", sa.BigInteger(), nullable=True))
    op.create_check_constraint(
        op.f("ck_document_chunks_line_range"),
        "document_chunks",
        "(start_line IS NULL AND end_line IS NULL) OR (start_line > 0 AND end_line >= start_line)",
    )

    op.create_table(
        "material_projects",
        *_identity_columns(),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        _workspace_fk(
            ["workspace_id", "created_by_user_id"],
            "workspace_memberships",
            ["workspace_id", "user_id"],
        ),
        sa.CheckConstraint(
            "char_length(name) BETWEEN 1 AND 120", name=op.f("ck_material_projects_name")
        ),
    )
    op.create_index("ix_material_projects_workspace_id", "material_projects", ["workspace_id"])
    op.create_table(
        "material_sources",
        *_identity_columns(),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("alias_name", sa.Text(), nullable=False),
        sa.Column("alias_digest", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        _workspace_fk(["workspace_id", "project_id"], "material_projects", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "alias_digest ~ '^[0-9a-f]{64}$'", name=op.f("ck_material_sources_alias_digest")
        ),
        sa.CheckConstraint("kind IN ('git', 'file')", name=op.f("ck_material_sources_kind")),
    )
    op.create_index(
        "ix_material_sources_workspace_project", "material_sources", ["workspace_id", "project_id"]
    )
    op.create_table(
        "material_imports",
        *_identity_columns(),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("source_ids", JSONB(none_as_null=True), nullable=False),
        sa.Column("cache_digest", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "run_id"),
        _workspace_fk(["workspace_id", "project_id"], "material_projects", ["workspace_id", "id"]),
        _workspace_fk(["workspace_id", "run_id"], "runs", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "jsonb_typeof(source_ids) = 'array'", name=op.f("ck_material_imports_source_ids")
        ),
        sa.CheckConstraint(
            "cache_digest IS NULL OR cache_digest ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_material_imports_cache_digest"),
        ),
    )
    op.create_index(
        "ix_material_imports_workspace_project", "material_imports", ["workspace_id", "project_id"]
    )
    op.create_table(
        "material_snapshots",
        *_identity_columns(),
        sa.Column("import_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("source_revision", sa.Text(), nullable=False),
        sa.Column("manifest_digest", sa.Text(), nullable=False),
        sa.Column("cache_digest", sa.Text(), nullable=False),
        sa.Column("inventory_json", JSONB(none_as_null=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "id"),
        sa.UniqueConstraint("workspace_id", "import_id", "source_id"),
        _workspace_fk(["workspace_id", "import_id"], "material_imports", ["workspace_id", "id"]),
        _workspace_fk(["workspace_id", "source_id"], "material_sources", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "manifest_digest ~ '^[0-9a-f]{64}$'", name=op.f("ck_material_snapshots_manifest_digest")
        ),
        sa.CheckConstraint(
            "cache_digest ~ '^[0-9a-f]{64}$'", name=op.f("ck_material_snapshots_cache_digest")
        ),
        sa.CheckConstraint(
            "jsonb_typeof(inventory_json) = 'object'",
            name=op.f("ck_material_snapshots_inventory_json"),
        ),
    )
    op.create_index(
        "ix_material_snapshots_workspace_import",
        "material_snapshots",
        ["workspace_id", "import_id"],
    )
    op.create_table(
        "material_snapshot_files",
        *_identity_columns(),
        sa.Column("snapshot_id", sa.Uuid(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("content", BYTEA(), nullable=False),
        sa.Column("content_digest", sa.Text(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("workspace_id", "snapshot_id", "path"),
        _workspace_fk(
            ["workspace_id", "snapshot_id"], "material_snapshots", ["workspace_id", "id"]
        ),
        _workspace_fk(["workspace_id", "document_id"], "documents", ["workspace_id", "id"]),
        sa.CheckConstraint(
            "octet_length(content) <= 1048576", name=op.f("ck_material_snapshot_files_content")
        ),
        sa.CheckConstraint(
            "content_digest ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_material_snapshot_files_content_digest"),
        ),
    )
    op.create_index(
        "ix_material_snapshot_files_workspace_snapshot",
        "material_snapshot_files",
        ["workspace_id", "snapshot_id"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    for table in (
        "material_snapshot_files",
        "material_snapshots",
        "material_imports",
        "material_sources",
        "material_projects",
    ):
        if bind.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar_one():
            raise RuntimeError("material snapshot data cannot be safely downgraded")
    for name in (
        "material_snapshot_files",
        "material_snapshots",
        "material_imports",
        "material_sources",
        "material_projects",
    ):
        op.drop_table(name)
    op.drop_constraint(op.f("ck_document_chunks_line_range"), "document_chunks", type_="check")
    op.drop_column("document_chunks", "end_line")
    op.drop_column("document_chunks", "start_line")
    op.drop_constraint(op.f("ck_documents_content"), "documents", type_="check")
    op.create_check_constraint(
        op.f("ck_documents_content"), "documents", "octet_length(content) BETWEEN 1 AND 400000"
    )
    op.drop_constraint(op.f("ck_documents_source_type"), "documents", type_="check")
    op.create_check_constraint(
        op.f("ck_documents_source_type"), "documents", "source_type IN ('markdown', 'text')"
    )
    op.drop_constraint(op.f("ck_runs_graph_version"), "runs", type_="check")
    op.create_check_constraint(
        op.f("ck_runs_graph_version"),
        "runs",
        "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
        "'pathfinder-research-v3', 'pathfinder-research-v4', "
        "'pathfinder-research-v5', 'pathfinder-research-v6')",
    )
