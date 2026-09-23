from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    auth_subject: Mapped[str] = mapped_column(Text, nullable=False, unique=True)


class Workspace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"
    __table_args__ = (
        CheckConstraint("kind IN ('personal', 'team')", name="kind"),
        Index(
            "uq_workspaces_personal_creator",
            "created_by_user_id",
            unique=True,
            postgresql_where=text("kind = 'personal'"),
        ),
    )

    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )


class WorkspaceMembership(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workspace_memberships"
    __table_args__ = (
        CheckConstraint("role IN ('member', 'reviewer', 'admin')", name="role"),
        UniqueConstraint("workspace_id", "user_id"),
        Index("ix_workspace_memberships_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class Document(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("source_type IN ('markdown', 'text', 'code')", name="source_type"),
        CheckConstraint("char_length(title) BETWEEN 1 AND 255", name="title"),
        CheckConstraint("char_length(source_name) BETWEEN 1 AND 255", name="source_name"),
        CheckConstraint("octet_length(content) BETWEEN 1 AND 1048576", name="content"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash"),
        CheckConstraint(
            "normalization_version ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name="normalization_version",
        ),
        CheckConstraint(
            "chunking_version ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name="chunking_version",
        ),
        CheckConstraint(
            "embedding_model ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name="embedding_model",
        ),
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "id", "embedding_model"),
        UniqueConstraint(
            "workspace_id",
            "content_hash",
            "normalization_version",
            "chunking_version",
            "embedding_model",
            name="uq_documents_representation_identity",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_documents_creator_membership",
            ondelete="RESTRICT",
        ),
        Index("ix_documents_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    normalization_version: Mapped[str] = mapped_column(Text, nullable=False)
    chunking_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentChunk(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint("ordinal >= 0", name="ordinal"),
        CheckConstraint(
            "section IS NULL OR char_length(section) BETWEEN 1 AND 800", name="section"
        ),
        CheckConstraint("octet_length(text) BETWEEN 1 AND 800", name="text"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash"),
        CheckConstraint(
            "token_count BETWEEN 1 AND 800 AND token_count = octet_length(text)",
            name="token_count",
        ),
        CheckConstraint(
            "embedding_model ~ '^[a-z0-9][a-z0-9._-]{0,99}$'",
            name="embedding_model",
        ),
        UniqueConstraint("workspace_id", "document_id", "ordinal"),
        ForeignKeyConstraint(
            ["workspace_id", "document_id", "embedding_model"],
            ["documents.workspace_id", "documents.id", "documents.embedding_model"],
            name="fk_document_chunks_document_profile",
            ondelete="RESTRICT",
        ),
        Index("ix_document_chunks_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    document_id: Mapped[UUID] = mapped_column(nullable=False)
    ordinal: Mapped[int] = mapped_column(BigInteger, nullable=False)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint(
            "char_length(title) BETWEEN 1 AND 500 AND btrim(title) = title",
            name="title",
        ),
        UniqueConstraint("workspace_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_conversations_creator_membership",
            ondelete="RESTRICT",
        ),
        Index("ix_conversations_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)


class Message(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role"),
        CheckConstraint(
            "(role = 'user' AND actor_user_id IS NOT NULL) OR "
            "(role = 'assistant' AND actor_user_id IS NULL)",
            name="role_actor",
        ),
        CheckConstraint("char_length(content) > 0", name="content"),
        UniqueConstraint("workspace_id", "conversation_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "conversation_id"],
            ["conversations.workspace_id", "conversations.id"],
            name="fk_messages_conversation",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_messages_actor_membership",
            ondelete="RESTRICT",
        ),
        Index("ix_messages_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Run(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "runs"
    __table_args__ = (
        CheckConstraint(
            "mode IN ('research', 'application', 'material_preparation', "
            "'resume_generation', 'resume_revision')",
            name="mode",
        ),
        CheckConstraint(
            "status IN "
            "('queued', 'running', 'waiting_approval', 'completed', 'failed', "
            "'cancelled')",
            name="status",
        ),
        CheckConstraint(
            "graph_version IN ('pathfinder-research-v1', 'pathfinder-research-v2', "
            "'pathfinder-research-v3', 'pathfinder-research-v4', "
            "'pathfinder-research-v5', 'pathfinder-research-v6', "
            "'pathfinder-resume-v1', 'pathfinder-resume-v2')",
            name="graph_version",
        ),
        CheckConstraint(
            "(mode IN ('research', 'application') "
            "AND graph_version LIKE 'pathfinder-research-v%') "
            "OR (mode IN ('material_preparation', 'resume_generation', 'resume_revision') "
            "AND graph_version LIKE 'pathfinder-resume-v%')",
            name="mode_graph_family",
        ),
        CheckConstraint(
            "mode IN ('research', 'application') OR status <> 'waiting_approval'",
            name="resume_no_legacy_approval",
        ),
        CheckConstraint("jsonb_typeof(input_json) = 'object'", name="input_json"),
        CheckConstraint("jsonb_typeof(limits_json) = 'object'", name="limits_json"),
        CheckConstraint(
            "result_json IS NULL OR jsonb_typeof(result_json) = 'object'",
            name="result_json",
        ),
        CheckConstraint("next_event_seq > 0", name="next_event_seq"),
        CheckConstraint(
            "error_category IS NULL OR error_category ~ '^[a-z][a-z0-9_]{0,99}$'",
            name="error_category",
        ),
        CheckConstraint(
            "status <> 'queued' OR (started_at IS NULL AND finished_at IS NULL "
            "AND result_json IS NULL AND error_category IS NULL "
            "AND cancel_requested_at IS NULL)",
            name="queued_fields",
        ),
        CheckConstraint(
            "((status IN ('completed', 'failed', 'cancelled') "
            "AND finished_at IS NOT NULL) OR "
            "(status NOT IN ('completed', 'failed', 'cancelled') "
            "AND finished_at IS NULL))",
            name="finished_at",
        ),
        CheckConstraint(
            "status NOT IN ('running', 'waiting_approval', 'completed', 'failed') "
            "OR started_at IS NOT NULL",
            name="started_at",
        ),
        CheckConstraint(
            "status <> 'completed' OR (result_json IS NOT NULL AND error_category IS NULL)",
            name="completed_fields",
        ),
        CheckConstraint(
            "status <> 'failed' OR error_category IS NOT NULL",
            name="failed_fields",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name="time_order",
        ),
        CheckConstraint(
            "finished_at IS NULL OR cancel_requested_at IS NULL "
            "OR finished_at >= cancel_requested_at",
            name="cancel_time_order",
        ),
        CheckConstraint(
            "(client_request_id IS NULL AND create_request_digest IS NULL "
            "AND create_request_version IS NULL) OR "
            "(client_request_id IS NOT NULL AND create_request_digest IS NOT NULL "
            "AND create_request_version IS NOT NULL AND create_request_version = 1 "
            "AND create_request_digest ~ '^[0-9a-f]{64}$')",
            name="create_request_identity",
        ),
        UniqueConstraint(
            "workspace_id",
            "created_by_user_id",
            "client_request_id",
            name="uq_runs_workspace_creator_request",
        ),
        UniqueConstraint("workspace_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_runs_creator_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "resume_document_id"],
            ["documents.workspace_id", "documents.id"],
            name="fk_runs_resume_document",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "conversation_id"],
            ["conversations.workspace_id", "conversations.id"],
            name="fk_runs_conversation",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "conversation_id", "request_message_id"],
            ["messages.workspace_id", "messages.conversation_id", "messages.id"],
            name="fk_runs_request_message",
            ondelete="RESTRICT",
        ),
        Index("ix_runs_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    client_request_id: Mapped[UUID | None] = mapped_column(nullable=True)
    create_request_digest: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_request_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    conversation_id: Mapped[UUID] = mapped_column(nullable=False)
    request_message_id: Mapped[UUID] = mapped_column(nullable=False)
    mode: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    resume_document_id: Mapped[UUID | None] = mapped_column(nullable=True)
    input_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True),
        nullable=False,
    )
    limits_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'queued'"),
    )
    graph_version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    next_event_seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("1"),
    )
    result_json: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    error_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class MaterialProject(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "material_projects"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("char_length(name) BETWEEN 1 AND 120", name="name"),
        Index("ix_material_projects_workspace_id", "workspace_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)


class MaterialSource(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "material_sources"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "project_id"],
            ["material_projects.workspace_id", "material_projects.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("alias_digest ~ '^[0-9a-f]{64}$'", name="alias_digest"),
        CheckConstraint("kind IN ('git', 'file')", name="kind"),
        Index("ix_material_sources_workspace_project", "workspace_id", "project_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    alias_name: Mapped[str] = mapped_column(Text, nullable=False)
    alias_digest: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)


class MaterialImport(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_imports"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "run_id"),
        ForeignKeyConstraint(
            ["workspace_id", "project_id"],
            ["material_projects.workspace_id", "material_projects.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"], ["runs.workspace_id", "runs.id"], ondelete="RESTRICT"
        ),
        CheckConstraint("jsonb_typeof(source_ids) = 'array'", name="source_ids"),
        CheckConstraint(
            "cache_digest IS NULL OR cache_digest ~ '^[0-9a-f]{64}$'",
            name="cache_digest",
        ),
        Index("ix_material_imports_workspace_project", "workspace_id", "project_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    source_ids: Mapped[list[str]] = mapped_column(JSONB(none_as_null=True), nullable=False)
    cache_digest: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MaterialSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_snapshots"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "import_id", "source_id"),
        ForeignKeyConstraint(
            ["workspace_id", "import_id"],
            ["material_imports.workspace_id", "material_imports.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "source_id"],
            ["material_sources.workspace_id", "material_sources.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("manifest_digest ~ '^[0-9a-f]{64}$'", name="manifest_digest"),
        CheckConstraint("cache_digest ~ '^[0-9a-f]{64}$'", name="cache_digest"),
        CheckConstraint("jsonb_typeof(inventory_json) = 'object'", name="inventory_json"),
        Index("ix_material_snapshots_workspace_import", "workspace_id", "import_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    import_id: Mapped[UUID] = mapped_column(nullable=False)
    source_id: Mapped[UUID] = mapped_column(nullable=False)
    source_revision: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_digest: Mapped[str] = mapped_column(Text, nullable=False)
    cache_digest: Mapped[str] = mapped_column(Text, nullable=False)
    inventory_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MaterialSnapshotFile(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_snapshot_files"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "snapshot_id", "path"),
        ForeignKeyConstraint(
            ["workspace_id", "snapshot_id"],
            ["material_snapshots.workspace_id", "material_snapshots.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "document_id"],
            ["documents.workspace_id", "documents.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("octet_length(content) <= 1048576", name="content"),
        CheckConstraint("content_digest ~ '^[0-9a-f]{64}$'", name="content_digest"),
        CheckConstraint(
            "line_ranges_json IS NULL OR jsonb_typeof(line_ranges_json) = 'array'",
            name="line_ranges_json",
        ),
        Index("ix_material_snapshot_files_workspace_snapshot", "workspace_id", "snapshot_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    snapshot_id: Mapped[UUID] = mapped_column(nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    content_digest: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[UUID | None] = mapped_column(nullable=True)
    line_ranges_json: Mapped[list[dict[str, int]] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MaterialFactSet(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_fact_sets"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "project_id", "cache_digest", "extractor_digest"),
        ForeignKeyConstraint(
            ["workspace_id", "project_id"],
            ["material_projects.workspace_id", "material_projects.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "import_id"],
            ["material_imports.workspace_id", "material_imports.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("cache_digest ~ '^[0-9a-f]{64}$'", name="cache_digest"),
        CheckConstraint("extractor_digest ~ '^[0-9a-f]{64}$'", name="extractor_digest"),
        CheckConstraint("jsonb_typeof(issues_json) = 'array'", name="issues_json"),
        Index("ix_material_fact_sets_workspace_project", "workspace_id", "project_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    project_id: Mapped[UUID] = mapped_column(nullable=False)
    import_id: Mapped[UUID] = mapped_column(nullable=False)
    cache_digest: Mapped[str] = mapped_column(Text, nullable=False)
    extractor_digest: Mapped[str] = mapped_column(Text, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False)
    issues_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MaterialFact(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_facts"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "fact_set_id", "ordinal"),
        ForeignKeyConstraint(
            ["workspace_id", "fact_set_id"],
            ["material_fact_sets.workspace_id", "material_fact_sets.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("ordinal >= 0 AND current_version > 0", name="version"),
        Index("ix_material_facts_workspace_set", "workspace_id", "fact_set_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    fact_set_id: Mapped[UUID] = mapped_column(nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)


class MaterialFactVersion(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_fact_versions"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "fact_id", "version"),
        ForeignKeyConstraint(
            ["workspace_id", "fact_id"],
            ["material_facts.workspace_id", "material_facts.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("version > 0 AND char_length(claim) BETWEEN 1 AND 2000", name="claim"),
        CheckConstraint(
            "kind IN ('implementation', 'plan', 'experiment', 'personal_statement')", name="kind"
        ),
        CheckConstraint("review_status IN ('pending', 'confirmed', 'rejected')", name="review"),
        CheckConstraint(
            "jsonb_typeof(conditions_json) = 'object' AND jsonb_typeof(issues_json) = 'array'",
            name="json",
        ),
        Index("ix_material_fact_versions_workspace_fact", "workspace_id", "fact_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    fact_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    conditions_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    review_status: Mapped[str] = mapped_column(Text, nullable=False)
    issues_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    created_by_user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class MaterialFactEvidence(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "material_fact_evidence"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "fact_version_id", "snapshot_file_id", "start_line", "end_line"
        ),
        ForeignKeyConstraint(
            ["workspace_id", "fact_version_id"],
            ["material_fact_versions.workspace_id", "material_fact_versions.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "snapshot_file_id"],
            ["material_snapshot_files.workspace_id", "material_snapshot_files.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "start_line > 0 AND end_line >= start_line AND end_line - start_line < 80", name="lines"
        ),
        CheckConstraint("char_length(quote) BETWEEN 1 AND 4000", name="quote"),
        Index("ix_material_fact_evidence_workspace_version", "workspace_id", "fact_version_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    fact_version_id: Mapped[UUID] = mapped_column(nullable=False)
    snapshot_file_id: Mapped[UUID] = mapped_column(nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)


class ResumeProfile(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "resume_profiles"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "owner_user_id"),
        ForeignKeyConstraint(
            ["workspace_id", "owner_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "current_version >= 1 AND current_preference_version >= 1", name="versions"
        ),
        Index("ix_resume_profiles_workspace_owner", "workspace_id", "owner_user_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    owner_user_id: Mapped[UUID] = mapped_column(nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False)
    current_preference_version: Mapped[int] = mapped_column(Integer, nullable=False)


class ResumeProfileImport(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_profile_imports"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "profile_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "profile_id"],
            ["resume_profiles.workspace_id", "resume_profiles.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("source_sha256 ~ '^[0-9a-f]{64}$'", name="source_sha256"),
        CheckConstraint("octet_length(source_bytes) BETWEEN 1 AND 131072", name="source_bytes"),
        Index("ix_resume_profile_imports_workspace_profile", "workspace_id", "profile_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    profile_id: Mapped[UUID] = mapped_column(nullable=False)
    template_commit: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    source_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumeProfileVersion(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_profile_versions"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "profile_id", "version"),
        ForeignKeyConstraint(
            ["workspace_id", "profile_id"],
            ["resume_profiles.workspace_id", "resume_profiles.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "profile_id", "source_import_id"],
            [
                "resume_profile_imports.workspace_id",
                "resume_profile_imports.profile_id",
                "resume_profile_imports.id",
            ],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("version >= 1 AND jsonb_typeof(content_json) = 'object'", name="content"),
        Index("ix_resume_profile_versions_workspace_profile", "workspace_id", "profile_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    profile_id: Mapped[UUID] = mapped_column(nullable=False)
    source_import_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumePreferenceVersion(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_preference_versions"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "profile_id", "version"),
        ForeignKeyConstraint(
            ["workspace_id", "profile_id"],
            ["resume_profiles.workspace_id", "resume_profiles.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "version >= 1 AND jsonb_typeof(preferences_json) = 'object'", name="preferences"
        ),
        Index("ix_resume_preference_versions_workspace_profile", "workspace_id", "profile_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    profile_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    preferences_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumeTexArtifact(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_tex_artifacts"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint(
            "workspace_id",
            "profile_version_id",
            "template_source_sha256",
            "content_sha256",
            "config_sha256",
            "renderer_version",
            name="uq_resume_tex_artifacts_identity",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "profile_version_id"],
            ["resume_profile_versions.workspace_id", "resume_profile_versions.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("octet_length(tex_bytes) BETWEEN 1 AND 524288", name="tex_bytes"),
        *(
            CheckConstraint(f"{field} ~ '^[0-9a-f]{{64}}$'", name=field)
            for field in (
                "template_source_sha256",
                "preamble_sha256",
                "content_sha256",
                "config_sha256",
                "tex_sha256",
            )
        ),
        Index(
            "ix_resume_tex_artifacts_workspace_profile_version",
            "workspace_id",
            "profile_version_id",
        ),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    profile_version_id: Mapped[UUID] = mapped_column(nullable=False)
    created_by_user_id: Mapped[UUID] = mapped_column(nullable=False)
    template_commit: Mapped[str] = mapped_column(Text, nullable=False)
    template_source_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    preamble_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    renderer_version: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    config_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    tex_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    tex_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumeSourceClaim(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_source_claims"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "source_import_id"],
            ["resume_profile_imports.workspace_id", "resume_profile_imports.id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "review_version >= 0 AND char_length(claim_text) BETWEEN 1 AND 4000", name="claim"
        ),
        CheckConstraint("jsonb_typeof(source_json) = 'object'", name="source"),
        Index("ix_resume_source_claims_workspace_import", "workspace_id", "source_import_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    source_import_id: Mapped[UUID] = mapped_column(nullable=False)
    project_item_id: Mapped[UUID] = mapped_column(nullable=False)
    item_id: Mapped[UUID] = mapped_column(nullable=False)
    field: Mapped[str] = mapped_column(Text, nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_json: Mapped[dict[str, object]] = mapped_column(JSONB(none_as_null=True), nullable=False)
    review_version: Mapped[int] = mapped_column(Integer, nullable=False)


class ResumeClaimReview(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_claim_reviews"
    __table_args__ = (
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "claim_id", "version"),
        ForeignKeyConstraint(
            ["workspace_id", "claim_id"],
            ["resume_source_claims.workspace_id", "resume_source_claims.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "project_id"],
            ["material_projects.workspace_id", "material_projects.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "version >= 1 AND decision IN ('linked', 'needs_evidence', 'excluded')", name="decision"
        ),
        Index("ix_resume_claim_reviews_workspace_claim", "workspace_id", "claim_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    claim_id: Mapped[UUID] = mapped_column(nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    project_id: Mapped[UUID | None] = mapped_column(nullable=True)
    actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ResumeClaimFactLink(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_claim_fact_links"
    __table_args__ = (
        UniqueConstraint("workspace_id", "review_id", "fact_version_id"),
        ForeignKeyConstraint(
            ["workspace_id", "review_id"],
            ["resume_claim_reviews.workspace_id", "resume_claim_reviews.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "fact_version_id"],
            ["material_fact_versions.workspace_id", "material_fact_versions.id"],
            ondelete="RESTRICT",
        ),
        Index("ix_resume_claim_fact_links_workspace_review", "workspace_id", "review_id"),
    )
    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    review_id: Mapped[UUID] = mapped_column(nullable=False)
    fact_version_id: Mapped[UUID] = mapped_column(nullable=False)


class ResumeCommand(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "resume_commands"
    __table_args__ = (
        CheckConstraint("kind ~ '^[a-z][a-z0-9_]{0,63}$'", name="kind"),
        CheckConstraint("digest_version = 1", name="digest_version"),
        CheckConstraint("request_digest ~ '^[0-9a-f]{64}$'", name="request_digest"),
        CheckConstraint("receipt_version = 1", name="receipt_version"),
        CheckConstraint("jsonb_typeof(receipt_json) = 'object'", name="receipt_json"),
        UniqueConstraint(
            "workspace_id",
            "actor_user_id",
            "client_request_id",
            name="uq_resume_commands_actor_request",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_resume_commands_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_resume_commands_run",
            ondelete="RESTRICT",
        ),
        Index("ix_resume_commands_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    client_request_id: Mapped[UUID] = mapped_column(nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    digest_version: Mapped[int] = mapped_column(Integer, nullable=False)
    request_digest: Mapped[str] = mapped_column(Text, nullable=False)
    receipt_version: Mapped[int] = mapped_column(Integer, nullable=False)
    receipt_json: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    run_id: Mapped[UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RunJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "run_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'leased', 'done', 'dead')",
            name="status",
        ),
        CheckConstraint(
            "attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts",
            name="attempts",
        ),
        CheckConstraint(
            "((status = 'leased' AND leased_by IS NOT NULL "
            "AND owner_token IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND attempt >= 1) OR "
            "(status <> 'leased' AND leased_by IS NULL AND owner_token IS NULL "
            "AND lease_expires_at IS NULL))",
            name="lease_fields",
        ),
        CheckConstraint(
            "status <> 'dead' OR "
            "(error_summary IS NOT NULL AND char_length(error_summary) BETWEEN 1 AND 1000 "
            "AND btrim(error_summary) = error_summary)",
            name="dead_error",
        ),
        CheckConstraint(
            "error_summary IS NULL OR char_length(error_summary) <= 1000",
            name="error_summary",
        ),
        UniqueConstraint("run_id"),
        ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_run_jobs_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_run_jobs_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id", "resume_approval_request_id"],
            [
                "approval_requests.workspace_id",
                "approval_requests.run_id",
                "approval_requests.id",
            ],
            name="fk_run_jobs_resume_approval_request",
            ondelete="RESTRICT",
        ),
        Index("ix_run_jobs_workspace_id", "workspace_id"),
        Index(
            "ix_run_jobs_due_claim",
            "status",
            "available_at",
            "created_at",
        ),
        Index("ix_run_jobs_stale_lease", "status", "lease_expires_at"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    originating_actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'queued'"),
    )
    attempt: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )
    max_attempts: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("3"),
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    leased_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_token: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    resume_approval_request_id: Mapped[UUID | None] = mapped_column(nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class ActionIntent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "action_intents"
    __table_args__ = (
        CheckConstraint(
            "action_key ~ '^[a-z][a-z0-9_]{0,99}$'",
            name="action_key",
        ),
        CheckConstraint("action_revision >= 1", name="action_revision"),
        CheckConstraint(
            "tool_name ~ '^[a-z][a-z0-9_]{0,99}$'",
            name="tool_name",
        ),
        CheckConstraint(
            "effect IN ('read_only', 'reversible', 'irreversible')",
            name="effect",
        ),
        CheckConstraint(
            "jsonb_typeof(args_snapshot) = 'object'",
            name="args_snapshot",
        ),
        CheckConstraint("canonicalization_version >= 1", name="canonicalization_version"),
        CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="args_digest",
        ),
        CheckConstraint(
            "jsonb_typeof(target_snapshot) = 'object'",
            name="target_snapshot",
        ),
        CheckConstraint(
            "target_canonicalization_version >= 1",
            name="target_canonicalization_version",
        ),
        CheckConstraint(
            "target_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="target_digest",
        ),
        CheckConstraint(
            "approval_binding_version >= 1",
            name="approval_binding_version",
        ),
        CheckConstraint(
            "approval_binding_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="approval_binding_digest",
        ),
        CheckConstraint(
            "status IN ('proposed', 'authorized', 'executing', 'succeeded', "
            "'failed', 'outcome_unknown', 'cancelled')",
            name="status",
        ),
        CheckConstraint(
            "idempotency_key ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'",
            name="idempotency_key",
        ),
        CheckConstraint("recovery_attempts >= 0", name="recovery_attempts"),
        CheckConstraint(
            "result IS NULL OR jsonb_typeof(result) = 'object'",
            name="result",
        ),
        CheckConstraint(
            "evidence IS NULL OR jsonb_typeof(evidence) = 'object'",
            name="evidence",
        ),
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "run_id", "id"),
        UniqueConstraint(
            "workspace_id",
            "run_id",
            "action_key",
            "action_revision",
            name="uq_action_intents_logical_revision",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_action_intents_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_action_intents_run",
            ondelete="RESTRICT",
        ),
        Index("ix_action_intents_workspace_id", "workspace_id"),
        Index("ix_action_intents_workspace_run", "workspace_id", "run_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    originating_actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    action_key: Mapped[str] = mapped_column(Text, nullable=False)
    action_revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    effect: Mapped[str] = mapped_column(Text, nullable=False)
    args_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    canonicalization_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    args_digest: Mapped[str] = mapped_column(Text, nullable=False)
    target_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    target_canonicalization_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    target_digest: Mapped[str] = mapped_column(Text, nullable=False)
    approval_binding_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    approval_binding_digest: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'proposed'"))
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    recovery_attempts: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0")
    )
    result: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    evidence: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )


class ApprovalRequest(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "approval_requests"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'consumed', 'expired')",
            name="status",
        ),
        CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="args_digest",
        ),
        CheckConstraint(
            "target_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="target_digest",
        ),
        CheckConstraint(
            "approval_binding_version >= 1",
            name="approval_binding_version",
        ),
        CheckConstraint(
            "approval_binding_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="approval_binding_digest",
        ),
        CheckConstraint("policy_version >= 1", name="policy_version"),
        CheckConstraint(
            "jsonb_typeof(policy_snapshot) = 'object'",
            name="policy_snapshot",
        ),
        CheckConstraint("version >= 1", name="version"),
        CheckConstraint(
            "(status = 'consumed' AND consumed_at IS NOT NULL) OR "
            "(status <> 'consumed' AND consumed_at IS NULL)",
            name="consumed_at",
        ),
        CheckConstraint(
            "consumed_at IS NULL OR consumed_at >= created_at",
            name="consumed_time_order",
        ),
        UniqueConstraint("workspace_id", "action_intent_id"),
        UniqueConstraint("workspace_id", "id"),
        UniqueConstraint("workspace_id", "run_id", "id"),
        ForeignKeyConstraint(
            ["workspace_id", "run_id", "action_intent_id"],
            ["action_intents.workspace_id", "action_intents.run_id", "action_intents.id"],
            name="fk_approval_requests_action_intent",
            ondelete="RESTRICT",
        ),
        Index("ix_approval_requests_workspace_id", "workspace_id"),
        Index("ix_approval_requests_workspace_run", "workspace_id", "run_id"),
        Index("ix_approval_requests_due", "workspace_id", "status", "expires_at"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    action_intent_id: Mapped[UUID] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    args_digest: Mapped[str] = mapped_column(Text, nullable=False)
    target_digest: Mapped[str] = mapped_column(Text, nullable=False)
    approval_binding_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    approval_binding_digest: Mapped[str] = mapped_column(Text, nullable=False)
    policy_version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    policy_snapshot: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True), nullable=False
    )
    version: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=text("1"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ApprovalDecision(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "approval_decisions"
    __table_args__ = (
        CheckConstraint("decision IN ('approve', 'reject')", name="decision"),
        CheckConstraint(
            "reason IS NULL OR char_length(reason) <= 1000",
            name="reason",
        ),
        UniqueConstraint("workspace_id", "approval_request_id", "actor_user_id"),
        ForeignKeyConstraint(
            ["workspace_id", "approval_request_id"],
            ["approval_requests.workspace_id", "approval_requests.id"],
            name="fk_approval_decisions_request",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_approval_decisions_actor_membership",
            ondelete="RESTRICT",
        ),
        Index("ix_approval_decisions_workspace_id", "workspace_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    approval_request_id: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class RunEvent(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "run_events"
    __table_args__ = (
        CheckConstraint("seq > 0", name="seq"),
        CheckConstraint("version = 1", name="version"),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="payload"),
        CheckConstraint(
            "type IN ('run.created', 'run.status_changed', 'run.completed', "
            "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
            "'agent.plan.created', 'agent.research.started', 'source.discovered', "
            "'tool.started', 'tool.finished', 'report.completed', 'rag.retrieved', "
            "'action.proposed', 'approval.expired', 'approval.decided', "
            "'action.cancelled', 'action.started', 'action.completed', "
            "'action.failed', 'action.outcome_unknown')",
            name="type",
        ),
        UniqueConstraint("run_id", "seq"),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_run_events_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_run_events_actor_membership",
            ondelete="RESTRICT",
        ),
        Index("ix_run_events_workspace_id", "workspace_id"),
        Index("ix_run_events_workspace_run_seq", "workspace_id", "run_id", "seq"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID | None] = mapped_column(nullable=True)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("1"),
    )
    payload: Mapped[dict[str, object]] = mapped_column(
        JSONB(none_as_null=True),
        nullable=False,
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ToolInvocation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "tool_invocations"
    __table_args__ = (
        CheckConstraint(
            "effect IN ('read_only', 'reversible', 'irreversible')",
            name="effect",
        ),
        CheckConstraint(
            "(effect = 'irreversible' AND action_intent_id IS NOT NULL) OR "
            "(effect = 'read_only' AND action_intent_id IS NULL)",
            name="action_binding",
        ),
        CheckConstraint(
            "status IN ('prepared', 'executing', 'succeeded', 'failed', 'outcome_unknown')",
            name="status",
        ),
        CheckConstraint(
            "tool_name ~ '^[a-z][a-z0-9_]{0,99}$'",
            name="tool_name",
        ),
        CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="args_digest",
        ),
        CheckConstraint("attempt >= 0", name="attempt"),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_ms"),
        CheckConstraint(
            "result_summary IS NULL OR "
            "(jsonb_typeof(result_summary) = 'object' "
            "AND octet_length(result_summary::text) <= 10000)",
            name="result_summary",
        ),
        CheckConstraint(
            "error_category IS NULL OR error_category ~ '^[a-z][a-z0-9_]{0,99}$'",
            name="error_category",
        ),
        CheckConstraint(
            "(status = 'prepared' AND started_at IS NULL AND finished_at IS NULL "
            "AND latency_ms IS NULL AND result_summary IS NULL "
            "AND error_category IS NULL) OR "
            "(status = 'executing' AND started_at IS NOT NULL "
            "AND finished_at IS NULL AND latency_ms IS NULL "
            "AND result_summary IS NULL AND error_category IS NULL) OR "
            "(status = 'succeeded' AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND latency_ms IS NOT NULL "
            "AND result_summary IS NOT NULL AND error_category IS NULL) OR "
            "(status = 'failed' AND finished_at IS NOT NULL "
            "AND latency_ms IS NOT NULL AND result_summary IS NULL "
            "AND error_category IS NOT NULL) OR "
            "(status = 'outcome_unknown' AND started_at IS NOT NULL "
            "AND finished_at IS NOT NULL AND latency_ms IS NOT NULL "
            "AND result_summary IS NULL "
            "AND error_category = 'external_outcome_unknown')",
            name="terminal_fields",
        ),
        CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name="time_order",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_tool_invocations_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_tool_invocations_run",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id", "action_intent_id"],
            ["action_intents.workspace_id", "action_intents.run_id", "action_intents.id"],
            name="fk_tool_invocations_action_intent",
            ondelete="RESTRICT",
        ),
        UniqueConstraint("action_intent_id", name="uq_tool_invocations_action_intent_id"),
        Index("ix_tool_invocations_workspace_id", "workspace_id"),
        Index("ix_tool_invocations_workspace_run", "workspace_id", "run_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        nullable=False,
    )
    originating_actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    action_intent_id: Mapped[UUID | None] = mapped_column(nullable=True)
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    effect: Mapped[str] = mapped_column(Text, nullable=False)
    args_digest: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    attempt: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default=text("0"),
    )
    latency_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    result_summary: Mapped[dict[str, object] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    error_category: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class MockSubmission(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "mock_submissions"
    __table_args__ = (
        CheckConstraint(
            "idempotency_key ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'",
            name="idempotency_key",
        ),
        CheckConstraint(
            "payload_digest ~ '^sha256:[0-9a-f]{64}$'",
            name="payload_digest",
        ),
        CheckConstraint("jsonb_typeof(payload) = 'object'", name="payload"),
        CheckConstraint(
            "char_length(external_ref) BETWEEN 1 AND 200 AND btrim(external_ref) = external_ref",
            name="external_ref",
        ),
        UniqueConstraint("idempotency_key", name="uq_mock_submissions_idempotency_key"),
        UniqueConstraint("action_intent_id", name="uq_mock_submissions_action_intent_id"),
        ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_mock_submissions_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id", "action_intent_id"],
            ["action_intents.workspace_id", "action_intents.run_id", "action_intents.id"],
            name="fk_mock_submissions_action_intent",
            ondelete="RESTRICT",
        ),
        Index("ix_mock_submissions_workspace_id", "workspace_id"),
        Index("ix_mock_submissions_workspace_run", "workspace_id", "run_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"), nullable=False
    )
    originating_actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID] = mapped_column(nullable=False)
    action_intent_id: Mapped[UUID] = mapped_column(nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    payload_digest: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB(none_as_null=True), nullable=False)
    external_ref: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LLMInvocation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "llm_invocations"
    __table_args__ = (
        CheckConstraint("invocation_kind IN ('chat', 'embedding')", name="invocation_kind"),
        CheckConstraint("provider IN ('fake', 'openai', 'qwen')", name="provider"),
        CheckConstraint("status IN ('started', 'succeeded', 'failed')", name="status"),
        CheckConstraint(
            "error_category IS NULL OR error_category IN "
            "('rate_limited', 'provider_timeout', 'provider_unavailable', "
            "'provider_authentication', 'provider_rejected', 'provider_error', "
            "'cancelled', 'invalid_provider_response')",
            name="error_category",
        ),
        CheckConstraint("graph_node ~ '^[a-z][a-z0-9_]{0,99}$'", name="graph_node"),
        CheckConstraint("request_hash ~ '^sha256:[0-9a-f]{64}$'", name="request_hash"),
        CheckConstraint(
            "provider_response_id IS NULL OR "
            "(char_length(provider_response_id) BETWEEN 1 AND 512 "
            "AND btrim(provider_response_id) = provider_response_id)",
            name="provider_response_id",
        ),
        CheckConstraint(
            "(provider = 'openai' AND (((invocation_kind = 'chat' "
            "AND model = 'gpt-5.6-terra' "
            "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
            "(invocation_kind = 'embedding' AND model = 'text-embedding-3-small' "
            "AND prompt_version IS NULL)))) OR "
            "(provider = 'qwen' AND (((invocation_kind = 'chat' "
            "AND model = 'qwen3.6-flash-2026-04-16' "
            "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
            "(invocation_kind = 'embedding' AND model = 'text-embedding-v4' "
            "AND prompt_version IS NULL)))) OR "
            "(provider = 'fake' AND ((((invocation_kind = 'chat' "
            "AND model = 'gpt-5.6-terra' "
            "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
            "(invocation_kind = 'embedding' AND model = 'text-embedding-3-small' "
            "AND prompt_version IS NULL))) OR (((invocation_kind = 'chat' "
            "AND model = 'qwen3.6-flash-2026-04-16' "
            "AND prompt_version ~ '^sha256:[0-9a-f]{64}$') OR "
            "(invocation_kind = 'embedding' AND model = 'text-embedding-v4' "
            "AND prompt_version IS NULL)))))",
            name="profile",
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_ms"),
        CheckConstraint(
            "pricing_version IS NULL OR "
            "(char_length(pricing_version) BETWEEN 1 AND 100 "
            "AND pricing_version ~ '^[a-z0-9][a-z0-9._-]*$')",
            name="pricing_version",
        ),
        CheckConstraint(
            "currency IS NULL OR currency IN ('USD', 'CNY')",
            name="currency",
        ),
        CheckConstraint(
            "estimated_cost IS NULL OR estimated_cost >= 0",
            name="estimated_cost",
        ),
        CheckConstraint(
            "trace_ids IS NULL OR (status IN ('succeeded', 'failed') "
            "AND jsonb_typeof(trace_ids) = 'object' "
            "AND trace_ids ? 'trace_id' AND trace_ids ? 'observation_id' "
            "AND trace_ids - 'trace_id' - 'observation_id' = '{}'::jsonb "
            "AND jsonb_typeof(trace_ids -> 'trace_id') = 'string' "
            "AND jsonb_typeof(trace_ids -> 'observation_id') = 'string' "
            "AND trace_ids ->> 'trace_id' ~ '^[0-9a-f]{32}$' "
            "AND trace_ids ->> 'trace_id' !~ '^0{32}$' "
            "AND trace_ids ->> 'observation_id' ~ '^[0-9a-f]{16}$' "
            "AND trace_ids ->> 'observation_id' !~ '^0{16}$')",
            name="trace_ids",
        ),
        CheckConstraint(
            "(pricing_version IS NULL AND currency IS NULL AND estimated_cost IS NULL) OR "
            "(pricing_version IS NOT NULL AND currency IS NOT NULL "
            "AND estimated_cost IS NOT NULL AND status = 'succeeded' "
            "AND token_usage IS NOT NULL AND "
            "((provider = 'openai' AND currency = 'USD') OR "
            "(provider = 'qwen' AND currency = 'CNY')))",
            name="cost_fields",
        ),
        CheckConstraint(
            "(status = 'started' AND token_usage IS NULL AND latency_ms IS NULL "
            "AND provider_response_id IS NULL AND error_category IS NULL) OR "
            "(status = 'succeeded' AND latency_ms IS NOT NULL AND error_category IS NULL) OR "
            "(status = 'failed' AND token_usage IS NULL AND latency_ms IS NOT NULL "
            "AND error_category IS NOT NULL)",
            name="terminal_fields",
        ),
        CheckConstraint(
            "status <> 'succeeded' OR "
            "(invocation_kind = 'chat' AND token_usage IS NOT NULL) OR "
            "invocation_kind = 'embedding'",
            name="success_usage",
        ),
        ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_llm_invocations_actor_membership",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_llm_invocations_run",
            ondelete="RESTRICT",
        ),
        Index("ix_llm_invocations_workspace_id", "workspace_id"),
        Index(
            "ix_llm_invocations_workspace_id_actor_user_id",
            "workspace_id",
            "actor_user_id",
        ),
        Index("ix_llm_invocations_workspace_run", "workspace_id", "run_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(nullable=False)
    actor_user_id: Mapped[UUID] = mapped_column(nullable=False)
    run_id: Mapped[UUID | None] = mapped_column(nullable=True)
    invocation_kind: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    graph_node: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)
    provider_response_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_usage: Mapped[dict[str, int] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    pricing_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    currency: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_cost: Mapped[Decimal | None] = mapped_column(
        Numeric(precision=20, scale=12),
        nullable=True,
    )
    trace_ids: Mapped[dict[str, str] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    latency_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_category: Mapped[str | None] = mapped_column(Text, nullable=True)
