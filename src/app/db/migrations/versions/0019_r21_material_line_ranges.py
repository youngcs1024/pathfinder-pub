"""Keep historical document chunks intact and bind line ranges to material files."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0019_r21_material_line_ranges"
down_revision = "0018_r21_material_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "material_snapshot_files",
        sa.Column("line_ranges_json", JSONB(none_as_null=True), nullable=True),
    )
    op.create_check_constraint(
        op.f("ck_material_snapshot_files_line_ranges_json"),
        "material_snapshot_files",
        "line_ranges_json IS NULL OR jsonb_typeof(line_ranges_json) = 'array'",
    )
    op.drop_constraint(op.f("ck_document_chunks_line_range"), "document_chunks", type_="check")
    op.drop_column("document_chunks", "end_line")
    op.drop_column("document_chunks", "start_line")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text("SELECT EXISTS (SELECT 1 FROM material_snapshot_files)")).scalar_one():
        raise RuntimeError("material line ranges cannot be safely downgraded")
    op.add_column("document_chunks", sa.Column("start_line", sa.BigInteger(), nullable=True))
    op.add_column("document_chunks", sa.Column("end_line", sa.BigInteger(), nullable=True))
    op.create_check_constraint(
        op.f("ck_document_chunks_line_range"),
        "document_chunks",
        "(start_line IS NULL AND end_line IS NULL) OR (start_line > 0 AND end_line >= start_line)",
    )
    op.drop_constraint(
        op.f("ck_material_snapshot_files_line_ranges_json"),
        "material_snapshot_files",
        type_="check",
    )
    op.drop_column("material_snapshot_files", "line_ranges_json")
