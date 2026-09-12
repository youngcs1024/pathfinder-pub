"""Add immutable pgvector document and chunk storage."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0008_rag_document_storage"
down_revision: str | None = "0007_persistent_run_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _id_column() -> sa.Column:
    return sa.Column(
        "id",
        postgresql.UUID(as_uuid=True),
        server_default=sa.text("gen_random_uuid()"),
        nullable=False,
    )


def _created_at_column() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("normalization_version", sa.Text(), nullable=False),
        sa.Column("chunking_version", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        _id_column(),
        _created_at_column(),
        sa.CheckConstraint(
            "source_type IN ('markdown', 'text')", name=op.f("ck_documents_source_type")
        ),
        sa.CheckConstraint("char_length(title) BETWEEN 1 AND 255", name=op.f("ck_documents_title")),
        sa.CheckConstraint(
            "char_length(source_name) BETWEEN 1 AND 255",
            name=op.f("ck_documents_source_name"),
        ),
        sa.CheckConstraint(
            "octet_length(content) BETWEEN 1 AND 400000",
            name=op.f("ck_documents_content"),
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_documents_content_hash")
        ),
        sa.CheckConstraint(
            "normalization_version ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name=op.f("ck_documents_normalization_version"),
        ),
        sa.CheckConstraint(
            "chunking_version ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name=op.f("ck_documents_chunking_version"),
        ),
        sa.CheckConstraint(
            "embedding_model ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name=op.f("ck_documents_embedding_model"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_documents_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_documents_creator_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("workspace_id", "id", name=op.f("uq_documents_workspace_id_id")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            "embedding_model",
            name=op.f("uq_documents_workspace_id_id_embedding_model"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "content_hash",
            "normalization_version",
            "chunking_version",
            "embedding_model",
            name="uq_documents_representation_identity",
        ),
    )
    op.create_index("ix_documents_workspace_id", "documents", ["workspace_id"], unique=False)

    op.create_table(
        "document_chunks",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.BigInteger(), nullable=False),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("token_count", sa.BigInteger(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        _id_column(),
        _created_at_column(),
        sa.CheckConstraint("ordinal >= 0", name=op.f("ck_document_chunks_ordinal")),
        sa.CheckConstraint(
            "section IS NULL OR char_length(section) BETWEEN 1 AND 800",
            name=op.f("ck_document_chunks_section"),
        ),
        sa.CheckConstraint(
            "octet_length(text) BETWEEN 1 AND 800", name=op.f("ck_document_chunks_text")
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_document_chunks_content_hash"),
        ),
        sa.CheckConstraint(
            "token_count BETWEEN 1 AND 800 AND token_count = octet_length(text)",
            name=op.f("ck_document_chunks_token_count"),
        ),
        sa.CheckConstraint(
            "embedding_model ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name=op.f("ck_document_chunks_embedding_model"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_document_chunks_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "document_id", "embedding_model"],
            ["documents.workspace_id", "documents.id", "documents.embedding_model"],
            name="fk_document_chunks_document_profile",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_chunks")),
        sa.UniqueConstraint(
            "workspace_id",
            "document_id",
            "ordinal",
            name=op.f("uq_document_chunks_workspace_id_document_id_ordinal"),
        ),
    )
    op.create_index(
        "ix_document_chunks_workspace_id", "document_chunks", ["workspace_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_workspace_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_workspace_id", table_name="documents")
    op.drop_table("documents")
