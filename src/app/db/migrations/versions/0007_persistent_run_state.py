"""Add the persistent Gate 4.1 run-state baseline."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_persistent_run_state"
down_revision: str | None = "0006_qwen_provider_profile"
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


def _updated_at_column() -> sa.Column:
    return sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("CURRENT_TIMESTAMP"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        _id_column(),
        _created_at_column(),
        _updated_at_column(),
        sa.CheckConstraint(
            "char_length(title) BETWEEN 1 AND 500 AND btrim(title) = title",
            name=op.f("ck_conversations_title"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_conversations_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_conversations_creator_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name=op.f("uq_conversations_workspace_id_id"),
        ),
    )
    op.create_index(
        "ix_conversations_workspace_id",
        "conversations",
        ["workspace_id"],
        unique=False,
    )

    op.create_table(
        "messages",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        _id_column(),
        _created_at_column(),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name=op.f("ck_messages_role"),
        ),
        sa.CheckConstraint(
            "(role = 'user' AND actor_user_id IS NOT NULL) OR "
            "(role = 'assistant' AND actor_user_id IS NULL)",
            name=op.f("ck_messages_role_actor"),
        ),
        sa.CheckConstraint(
            "char_length(content) > 0",
            name=op.f("ck_messages_content"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_messages_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "conversation_id"],
            ["conversations.workspace_id", "conversations.id"],
            name="fk_messages_conversation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_messages_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "workspace_id",
            "conversation_id",
            "id",
            name=op.f("uq_messages_workspace_id_conversation_id_id"),
        ),
    )
    op.create_index(
        "ix_messages_workspace_id",
        "messages",
        ["workspace_id"],
        unique=False,
    )

    op.create_table(
        "runs",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "mode",
            sa.Text(),
            server_default=sa.text("'research'"),
            nullable=False,
        ),
        sa.Column(
            "input_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "limits_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'queued'"),
            nullable=False,
        ),
        sa.Column(
            "graph_version",
            sa.Text(),
            server_default=sa.text("'pathfinder-research-v1'"),
            nullable=False,
        ),
        sa.Column(
            "next_event_seq",
            sa.BigInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "result_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_category", sa.Text(), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _id_column(),
        _created_at_column(),
        _updated_at_column(),
        sa.CheckConstraint("mode = 'research'", name=op.f("ck_runs_mode")),
        sa.CheckConstraint(
            "status IN "
            "('queued', 'running', 'waiting_approval', 'completed', 'failed', "
            "'cancelled')",
            name=op.f("ck_runs_status"),
        ),
        sa.CheckConstraint(
            "graph_version = 'pathfinder-research-v1'",
            name=op.f("ck_runs_graph_version"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(input_json) = 'object'",
            name=op.f("ck_runs_input_json"),
        ),
        sa.CheckConstraint(
            "jsonb_typeof(limits_json) = 'object'",
            name=op.f("ck_runs_limits_json"),
        ),
        sa.CheckConstraint(
            "result_json IS NULL OR jsonb_typeof(result_json) = 'object'",
            name=op.f("ck_runs_result_json"),
        ),
        sa.CheckConstraint(
            "next_event_seq > 0",
            name=op.f("ck_runs_next_event_seq"),
        ),
        sa.CheckConstraint(
            "error_category IS NULL OR error_category ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_runs_error_category"),
        ),
        sa.CheckConstraint(
            "status <> 'queued' OR (started_at IS NULL AND finished_at IS NULL "
            "AND result_json IS NULL AND error_category IS NULL "
            "AND cancel_requested_at IS NULL)",
            name=op.f("ck_runs_queued_fields"),
        ),
        sa.CheckConstraint(
            "((status IN ('completed', 'failed', 'cancelled') "
            "AND finished_at IS NOT NULL) OR "
            "(status NOT IN ('completed', 'failed', 'cancelled') "
            "AND finished_at IS NULL))",
            name=op.f("ck_runs_finished_at"),
        ),
        sa.CheckConstraint(
            "status NOT IN ('running', 'waiting_approval', 'completed', 'failed') "
            "OR started_at IS NOT NULL",
            name=op.f("ck_runs_started_at"),
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR (result_json IS NOT NULL AND error_category IS NULL)",
            name=op.f("ck_runs_completed_fields"),
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR error_category IS NOT NULL",
            name=op.f("ck_runs_failed_fields"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name=op.f("ck_runs_time_order"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR cancel_requested_at IS NULL "
            "OR finished_at >= cancel_requested_at",
            name=op.f("ck_runs_cancel_time_order"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_runs_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "created_by_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_runs_creator_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "conversation_id"],
            ["conversations.workspace_id", "conversations.id"],
            name="fk_runs_conversation",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "conversation_id", "request_message_id"],
            ["messages.workspace_id", "messages.conversation_id", "messages.id"],
            name="fk_runs_request_message",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runs")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name=op.f("uq_runs_workspace_id_id"),
        ),
    )
    op.create_index("ix_runs_workspace_id", "runs", ["workspace_id"], unique=False)

    op.create_table(
        "run_jobs",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "originating_actor_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'queued'"),
            nullable=False,
        ),
        sa.Column(
            "attempt",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "max_attempts",
            sa.BigInteger(),
            server_default=sa.text("3"),
            nullable=False,
        ),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("leased_by", sa.Text(), nullable=True),
        sa.Column("owner_token", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        _id_column(),
        _created_at_column(),
        _updated_at_column(),
        sa.CheckConstraint(
            "status IN ('queued', 'leased', 'done', 'dead')",
            name=op.f("ck_run_jobs_status"),
        ),
        sa.CheckConstraint(
            "attempt >= 0 AND max_attempts > 0 AND attempt <= max_attempts",
            name=op.f("ck_run_jobs_attempts"),
        ),
        sa.CheckConstraint(
            "((status = 'leased' AND leased_by IS NOT NULL "
            "AND owner_token IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND attempt >= 1) OR "
            "(status <> 'leased' AND leased_by IS NULL AND owner_token IS NULL "
            "AND lease_expires_at IS NULL))",
            name=op.f("ck_run_jobs_lease_fields"),
        ),
        sa.CheckConstraint(
            "status <> 'dead' OR "
            "(error_summary IS NOT NULL AND char_length(error_summary) BETWEEN 1 AND 1000 "
            "AND btrim(error_summary) = error_summary)",
            name=op.f("ck_run_jobs_dead_error"),
        ),
        sa.CheckConstraint(
            "error_summary IS NULL OR char_length(error_summary) <= 1000",
            name=op.f("ck_run_jobs_error_summary"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_run_jobs_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_run_jobs_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_run_jobs_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_jobs")),
        sa.UniqueConstraint("run_id", name=op.f("uq_run_jobs_run_id")),
    )
    op.create_index(
        "ix_run_jobs_workspace_id",
        "run_jobs",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_run_jobs_due_claim",
        "run_jobs",
        ["status", "available_at", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_run_jobs_stale_lease",
        "run_jobs",
        ["status", "lease_expires_at"],
        unique=False,
    )

    op.create_table(
        "run_events",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column(
            "version",
            sa.BigInteger(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        _id_column(),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("seq > 0", name=op.f("ck_run_events_seq")),
        sa.CheckConstraint("version = 1", name=op.f("ck_run_events_version")),
        sa.CheckConstraint(
            "jsonb_typeof(payload) = 'object'",
            name=op.f("ck_run_events_payload"),
        ),
        sa.CheckConstraint(
            "type IN ('run.created', 'run.status_changed', 'run.completed', "
            "'run.failed', 'run.cancelled', 'job.lease_expired', 'job.dead', "
            "'agent.plan.created', 'agent.research.started', 'source.discovered', "
            "'tool.started', 'tool.finished', 'report.completed')",
            name=op.f("ck_run_events_type"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_run_events_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_run_events_run",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_run_events_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_events")),
        sa.UniqueConstraint(
            "run_id",
            "seq",
            name=op.f("uq_run_events_run_id_seq"),
        ),
    )
    op.create_index(
        "ix_run_events_workspace_id",
        "run_events",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_run_events_workspace_run_seq",
        "run_events",
        ["workspace_id", "run_id", "seq"],
        unique=False,
    )

    op.create_table(
        "tool_invocations",
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "originating_actor_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("effect", sa.Text(), nullable=False),
        sa.Column("args_digest", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "attempt",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("latency_ms", sa.BigInteger(), nullable=True),
        sa.Column(
            "result_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("error_category", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        _id_column(),
        _created_at_column(),
        _updated_at_column(),
        sa.CheckConstraint(
            "effect IN ('read_only', 'reversible', 'irreversible')",
            name=op.f("ck_tool_invocations_effect"),
        ),
        sa.CheckConstraint(
            "effect = 'read_only'",
            name=op.f("ck_tool_invocations_gate4_effect"),
        ),
        sa.CheckConstraint(
            "status IN ('prepared', 'executing', 'succeeded', 'failed', 'outcome_unknown')",
            name=op.f("ck_tool_invocations_status"),
        ),
        sa.CheckConstraint(
            "status <> 'outcome_unknown'",
            name=op.f("ck_tool_invocations_gate4_status"),
        ),
        sa.CheckConstraint(
            "tool_name ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_tool_invocations_tool_name"),
        ),
        sa.CheckConstraint(
            "args_digest ~ '^sha256:[0-9a-f]{64}$'",
            name=op.f("ck_tool_invocations_args_digest"),
        ),
        sa.CheckConstraint(
            "attempt >= 0",
            name=op.f("ck_tool_invocations_attempt"),
        ),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name=op.f("ck_tool_invocations_latency_ms"),
        ),
        sa.CheckConstraint(
            "result_summary IS NULL OR "
            "(jsonb_typeof(result_summary) = 'object' "
            "AND octet_length(result_summary::text) <= 10000)",
            name=op.f("ck_tool_invocations_result_summary"),
        ),
        sa.CheckConstraint(
            "error_category IS NULL OR error_category ~ '^[a-z][a-z0-9_]{0,99}$'",
            name=op.f("ck_tool_invocations_error_category"),
        ),
        sa.CheckConstraint(
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
            "AND error_category IS NOT NULL)",
            name=op.f("ck_tool_invocations_terminal_fields"),
        ),
        sa.CheckConstraint(
            "finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at",
            name=op.f("ck_tool_invocations_time_order"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_tool_invocations_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "originating_actor_user_id"],
            ["workspace_memberships.workspace_id", "workspace_memberships.user_id"],
            name="fk_tool_invocations_actor_membership",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "run_id"],
            ["runs.workspace_id", "runs.id"],
            name="fk_tool_invocations_run",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tool_invocations")),
    )
    op.create_index(
        "ix_tool_invocations_workspace_id",
        "tool_invocations",
        ["workspace_id"],
        unique=False,
    )
    op.create_index(
        "ix_tool_invocations_workspace_run",
        "tool_invocations",
        ["workspace_id", "run_id"],
        unique=False,
    )

    op.add_column(
        "llm_invocations",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_llm_invocations_run",
        "llm_invocations",
        "runs",
        ["workspace_id", "run_id"],
        ["workspace_id", "id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_llm_invocations_workspace_run",
        "llm_invocations",
        ["workspace_id", "run_id"],
        unique=False,
    )


def downgrade() -> None:
    connection = op.get_bind()
    gate4_row_count = connection.scalar(
        sa.text(
            "SELECT "
            "(SELECT count(*) FROM conversations) + "
            "(SELECT count(*) FROM messages) + "
            "(SELECT count(*) FROM runs) + "
            "(SELECT count(*) FROM run_jobs) + "
            "(SELECT count(*) FROM run_events) + "
            "(SELECT count(*) FROM tool_invocations)"
        )
    )
    linked_invocation_count = connection.scalar(
        sa.text("SELECT count(*) FROM llm_invocations WHERE run_id IS NOT NULL")
    )
    if gate4_row_count or linked_invocation_count:
        raise RuntimeError(
            "cannot downgrade 0007 while Gate 4 business facts or linked invocations exist"
        )

    op.drop_index(
        "ix_llm_invocations_workspace_run",
        table_name="llm_invocations",
    )
    op.drop_constraint(
        "fk_llm_invocations_run",
        "llm_invocations",
        type_="foreignkey",
    )
    op.drop_column("llm_invocations", "run_id")

    op.drop_index(
        "ix_tool_invocations_workspace_run",
        table_name="tool_invocations",
    )
    op.drop_index(
        "ix_tool_invocations_workspace_id",
        table_name="tool_invocations",
    )
    op.drop_table("tool_invocations")

    op.drop_index(
        "ix_run_events_workspace_run_seq",
        table_name="run_events",
    )
    op.drop_index("ix_run_events_workspace_id", table_name="run_events")
    op.drop_table("run_events")

    op.drop_index("ix_run_jobs_stale_lease", table_name="run_jobs")
    op.drop_index("ix_run_jobs_due_claim", table_name="run_jobs")
    op.drop_index("ix_run_jobs_workspace_id", table_name="run_jobs")
    op.drop_table("run_jobs")

    op.drop_index("ix_runs_workspace_id", table_name="runs")
    op.drop_table("runs")

    op.drop_index("ix_messages_workspace_id", table_name="messages")
    op.drop_table("messages")

    op.drop_index("ix_conversations_workspace_id", table_name="conversations")
    op.drop_table("conversations")
