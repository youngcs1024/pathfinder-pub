from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from app.db.models import WorkspaceMembership
from tests.integration.support import alembic_config, connect_database

pytestmark = pytest.mark.integration

APP_TABLES = {
    "users",
    "workspaces",
    "workspace_memberships",
    "documents",
    "document_chunks",
    "conversations",
    "messages",
    "runs",
    "run_jobs",
    "run_events",
    "action_intents",
    "approval_requests",
    "approval_decisions",
    "mock_submissions",
    "tool_invocations",
    "llm_invocations",
}


def _app_tables(engine: sa.Engine) -> set[str]:
    return APP_TABLES.intersection(sa.inspect(engine).get_table_names())


def test_postgres_major_and_pgvector_image_contract(database_url: str) -> None:
    with connect_database(database_url) as connection:
        server_version = connection.execute(
            "SELECT current_setting('server_version_num')::integer"
        ).fetchone()
        vector_extension = connection.execute(
            """
            SELECT default_version, installed_version
            FROM pg_available_extensions
            WHERE name = 'vector'
            """
        ).fetchone()

    assert server_version is not None
    assert server_version[0] // 10_000 == 16
    assert vector_extension is not None
    assert vector_extension[0] is not None
    assert vector_extension[1] is None


def test_migration_round_trip_constraints_and_metadata(database_url: str) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        assert _app_tables(engine) == set()

        command.upgrade(config, "head")

        inspector = sa.inspect(engine)
        assert _app_tables(engine) == APP_TABLES
        with engine.connect() as connection:
            revision = connection.scalar(sa.text("SELECT version_num FROM alembic_version"))
            migration_context = MigrationContext.configure(
                connection,
                opts={
                    "compare_server_default": True,
                    "compare_type": True,
                },
            )
            metadata_diff = compare_metadata(
                migration_context,
                WorkspaceMembership.metadata,
            )

            assert revision == "0016_r1_execution_contracts"
        assert metadata_diff == []
        with engine.connect() as connection:
            vector_extension = connection.scalar(
                sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            embedding_type = connection.scalar(
                sa.text(
                    """
                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                    FROM pg_attribute AS attribute
                    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
                    WHERE relation.relname = 'document_chunks'
                      AND attribute.attname = 'embedding'
                    """
                )
            )
        assert vector_extension is not None
        assert embedding_type == "vector(1536)"
        assert {item["name"] for item in inspector.get_columns("documents")} == {
            "id",
            "workspace_id",
            "created_by_user_id",
            "title",
            "source_type",
            "source_name",
            "content",
            "content_hash",
            "normalization_version",
            "chunking_version",
            "embedding_model",
            "created_at",
        }
        assert {item["name"] for item in inspector.get_columns("document_chunks")} == {
            "id",
            "workspace_id",
            "document_id",
            "ordinal",
            "section",
            "text",
            "content_hash",
            "token_count",
            "embedding_model",
            "embedding",
            "created_at",
        }
        assert {item["name"] for item in inspector.get_unique_constraints("documents")} == {
            "uq_documents_representation_identity",
            "uq_documents_workspace_id_id",
            "uq_documents_workspace_id_id_embedding_model",
        }
        assert {item["name"] for item in inspector.get_unique_constraints("document_chunks")} == {
            "uq_document_chunks_workspace_id_document_id_ordinal"
        }
        assert {item["name"] for item in inspector.get_foreign_keys("documents")} == {
            "fk_documents_creator_membership",
            "fk_documents_workspace_id_workspaces",
        }
        assert {item["name"] for item in inspector.get_foreign_keys("document_chunks")} == {
            "fk_document_chunks_document_profile",
            "fk_document_chunks_workspace_id_workspaces",
        }
        assert {item["name"] for item in inspector.get_indexes("documents")} >= {
            "ix_documents_workspace_id"
        }
        assert {item["name"] for item in inspector.get_indexes("document_chunks")} >= {
            "ix_document_chunks_workspace_id"
        }
        assert {item["name"] for item in inspector.get_columns("action_intents")} == {
            "id",
            "workspace_id",
            "originating_actor_user_id",
            "run_id",
            "action_key",
            "action_revision",
            "tool_name",
            "effect",
            "args_snapshot",
            "canonicalization_version",
            "args_digest",
            "target_snapshot",
            "target_canonicalization_version",
            "target_digest",
            "approval_binding_version",
            "approval_binding_digest",
            "status",
            "idempotency_key",
            "recovery_attempts",
            "result",
            "evidence",
            "created_at",
            "updated_at",
        }
        assert {item["name"] for item in inspector.get_columns("approval_requests")} == {
            "id",
            "workspace_id",
            "run_id",
            "action_intent_id",
            "status",
            "args_digest",
            "target_digest",
            "approval_binding_version",
            "approval_binding_digest",
            "policy_version",
            "policy_snapshot",
            "version",
            "expires_at",
            "consumed_at",
            "created_at",
            "updated_at",
        }
        assert {item["name"] for item in inspector.get_unique_constraints("action_intents")} == {
            "uq_action_intents_logical_revision",
            "uq_action_intents_workspace_id_id",
            "uq_action_intents_workspace_id_run_id_id",
        }
        assert {item["name"] for item in inspector.get_unique_constraints("approval_requests")} == {
            "uq_approval_requests_workspace_id_action_intent_id",
            "uq_approval_requests_workspace_id_id",
            "uq_approval_requests_workspace_id_run_id_id",
        }
        assert {item["name"] for item in inspector.get_columns("approval_decisions")} == {
            "id",
            "workspace_id",
            "approval_request_id",
            "actor_user_id",
            "decision",
            "reason",
            "decided_at",
        }
        assert {item["name"] for item in inspector.get_check_constraints("approval_decisions")} == {
            "ck_approval_decisions_decision",
            "ck_approval_decisions_reason",
        }
        assert {item["name"] for item in inspector.get_foreign_keys("approval_decisions")} == {
            "fk_approval_decisions_actor_membership",
            "fk_approval_decisions_request",
            "fk_approval_decisions_workspace_id_workspaces",
        }
        assert {item["name"] for item in inspector.get_foreign_keys("action_intents")} == {
            "fk_action_intents_actor_membership",
            "fk_action_intents_run",
            "fk_action_intents_workspace_id_workspaces",
        }
        assert {item["name"] for item in inspector.get_foreign_keys("approval_requests")} == {
            "fk_approval_requests_action_intent",
            "fk_approval_requests_workspace_id_workspaces",
        }
        assert {item["name"] for item in inspector.get_indexes("action_intents")} >= {
            "ix_action_intents_workspace_id",
            "ix_action_intents_workspace_run",
        }
        assert {item["name"] for item in inspector.get_indexes("approval_requests")} >= {
            "ix_approval_requests_due",
            "ix_approval_requests_workspace_id",
            "ix_approval_requests_workspace_run",
        }
        assert {item["name"] for item in inspector.get_check_constraints("action_intents")} == {
            "ck_action_intents_action_key",
            "ck_action_intents_action_revision",
            "ck_action_intents_approval_binding_digest",
            "ck_action_intents_approval_binding_version",
            "ck_action_intents_args_digest",
            "ck_action_intents_args_snapshot",
            "ck_action_intents_canonicalization_version",
            "ck_action_intents_effect",
            "ck_action_intents_evidence",
            "ck_action_intents_idempotency_key",
            "ck_action_intents_recovery_attempts",
            "ck_action_intents_result",
            "ck_action_intents_status",
            "ck_action_intents_target_canonicalization_version",
            "ck_action_intents_target_digest",
            "ck_action_intents_target_snapshot",
            "ck_action_intents_tool_name",
        }
        assert {item["name"] for item in inspector.get_check_constraints("approval_requests")} == {
            "ck_approval_requests_approval_binding_digest",
            "ck_approval_requests_approval_binding_version",
            "ck_approval_requests_args_digest",
            "ck_approval_requests_consumed_at",
            "ck_approval_requests_consumed_time_order",
            "ck_approval_requests_policy_snapshot",
            "ck_approval_requests_policy_version",
            "ck_approval_requests_status",
            "ck_approval_requests_target_digest",
            "ck_approval_requests_version",
        }
        run_graph_version = next(
            item for item in inspector.get_columns("runs") if item["name"] == "graph_version"
        )
        assert run_graph_version["default"] is None
        graph_check = next(
            item
            for item in inspector.get_check_constraints("runs")
            if item["name"] == "ck_runs_graph_version"
        )
        assert all(
            version in graph_check["sqltext"]
            for version in (
                "pathfinder-research-v1",
                "pathfinder-research-v2",
                "pathfinder-research-v3",
                "pathfinder-research-v4",
                "pathfinder-research-v5",
                "pathfinder-research-v6",
            )
        )
        event_type_check = next(
            item
            for item in inspector.get_check_constraints("run_events")
            if item["name"] == "ck_run_events_type"
        )
        assert "action.proposed" in event_type_check["sqltext"]
        assert "approval.expired" in event_type_check["sqltext"]
        assert "approval.decided" in event_type_check["sqltext"]
        assert "action.cancelled" in event_type_check["sqltext"]
        assert all(
            event_type in event_type_check["sqltext"]
            for event_type in (
                "action.started",
                "action.completed",
                "action.failed",
                "action.outcome_unknown",
            )
        )
        tool_checks = {item["name"] for item in inspector.get_check_constraints("tool_invocations")}
        assert "ck_tool_invocations_action_binding" in tool_checks
        assert "ck_tool_invocations_gate4_status" not in tool_checks
        terminal_fields_check = next(
            item
            for item in inspector.get_check_constraints("tool_invocations")
            if item["name"] == "ck_tool_invocations_terminal_fields"
        )
        assert "status = 'outcome_unknown'" in terminal_fields_check["sqltext"]
        assert "external_outcome_unknown" in terminal_fields_check["sqltext"]
        assert "action_intent_id" in {
            item["name"] for item in inspector.get_columns("tool_invocations")
        }
        assert "fk_tool_invocations_action_intent" in {
            item["name"] for item in inspector.get_foreign_keys("tool_invocations")
        }
        assert "uq_tool_invocations_action_intent_id" in {
            item["name"] for item in inspector.get_unique_constraints("tool_invocations")
        }
        assert {item["name"] for item in inspector.get_columns("mock_submissions")} == {
            "id",
            "workspace_id",
            "originating_actor_user_id",
            "run_id",
            "action_intent_id",
            "idempotency_key",
            "payload_digest",
            "payload",
            "external_ref",
            "created_at",
        }
        assert {item["name"] for item in inspector.get_check_constraints("workspaces")} == {
            "ck_workspaces_kind"
        }
        assert {
            item["name"] for item in inspector.get_check_constraints("workspace_memberships")
        } == {"ck_workspace_memberships_role"}
        assert {
            item["name"] for item in inspector.get_unique_constraints("workspace_memberships")
        } == {"uq_workspace_memberships_workspace_id_user_id"}
        assert {item["name"] for item in inspector.get_check_constraints("llm_invocations")} == {
            "ck_llm_invocations_error_category",
            "ck_llm_invocations_cost_fields",
            "ck_llm_invocations_currency",
            "ck_llm_invocations_estimated_cost",
            "ck_llm_invocations_graph_node",
            "ck_llm_invocations_invocation_kind",
            "ck_llm_invocations_latency_ms",
            "ck_llm_invocations_profile",
            "ck_llm_invocations_pricing_version",
            "ck_llm_invocations_provider",
            "ck_llm_invocations_provider_response_id",
            "ck_llm_invocations_request_hash",
            "ck_llm_invocations_status",
            "ck_llm_invocations_success_usage",
            "ck_llm_invocations_terminal_fields",
            "ck_llm_invocations_trace_ids",
        }
        invocation_columns = {item["name"] for item in inspector.get_columns("llm_invocations")}
        assert invocation_columns == {
            "id",
            "workspace_id",
            "actor_user_id",
            "run_id",
            "invocation_kind",
            "provider",
            "model",
            "graph_node",
            "prompt_version",
            "request_hash",
            "provider_response_id",
            "token_usage",
            "pricing_version",
            "currency",
            "estimated_cost",
            "trace_ids",
            "latency_ms",
            "status",
            "error_category",
            "created_at",
            "updated_at",
        }
        estimated_cost_column = next(
            item
            for item in inspector.get_columns("llm_invocations")
            if item["name"] == "estimated_cost"
        )
        assert estimated_cost_column["type"].precision == 20
        assert estimated_cost_column["type"].scale == 12

        workspace_index = next(
            item
            for item in inspector.get_indexes("workspaces")
            if item["name"] == "uq_workspaces_personal_creator"
        )
        assert workspace_index["unique"] is True
        assert "kind = 'personal'" in str(workspace_index["dialect_options"]["postgresql_where"])
        assert {item["name"] for item in inspector.get_indexes("workspace_memberships")} >= {
            "ix_workspace_memberships_workspace_id"
        }
        workspace_foreign_keys = inspector.get_foreign_keys("workspaces")
        assert {item["name"] for item in workspace_foreign_keys} == {
            "fk_workspaces_created_by_user_id_users"
        }
        membership_foreign_keys = inspector.get_foreign_keys("workspace_memberships")
        assert {item["name"] for item in membership_foreign_keys} == {
            "fk_workspace_memberships_user_id_users",
            "fk_workspace_memberships_workspace_id_workspaces",
        }
        assert all(
            item["options"].get("ondelete") == "RESTRICT"
            for item in (*workspace_foreign_keys, *membership_foreign_keys)
        )
        invocation_foreign_keys = inspector.get_foreign_keys("llm_invocations")
        assert {item["name"] for item in invocation_foreign_keys} == {
            "fk_llm_invocations_actor_membership",
            "fk_llm_invocations_run",
            "fk_llm_invocations_workspace_id_workspaces",
        }
        assert all(
            item["options"].get("ondelete") == "RESTRICT" for item in invocation_foreign_keys
        )
        assert {item["name"] for item in inspector.get_indexes("llm_invocations")} == {
            "ix_llm_invocations_workspace_id",
            "ix_llm_invocations_workspace_id_actor_user_id",
            "ix_llm_invocations_workspace_run",
        }
        expected_gate4_columns = {
            "conversations": {
                "id",
                "workspace_id",
                "created_by_user_id",
                "title",
                "created_at",
                "updated_at",
            },
            "messages": {
                "id",
                "workspace_id",
                "conversation_id",
                "actor_user_id",
                "role",
                "content",
                "created_at",
            },
            "runs": {
                "client_request_id",
                "create_request_digest",
                "create_request_version",
                "id",
                "workspace_id",
                "created_by_user_id",
                "conversation_id",
                "request_message_id",
                "mode",
                "resume_document_id",
                "input_json",
                "limits_json",
                "status",
                "graph_version",
                "next_event_seq",
                "result_json",
                "error_category",
                "cancel_requested_at",
                "started_at",
                "finished_at",
                "created_at",
                "updated_at",
            },
            "run_jobs": {
                "id",
                "workspace_id",
                "originating_actor_user_id",
                "run_id",
                "status",
                "attempt",
                "max_attempts",
                "available_at",
                "leased_by",
                "owner_token",
                "lease_expires_at",
                "resume_approval_request_id",
                "error_summary",
                "created_at",
                "updated_at",
            },
            "run_events": {
                "id",
                "workspace_id",
                "run_id",
                "actor_user_id",
                "seq",
                "type",
                "version",
                "payload",
                "recorded_at",
            },
            "tool_invocations": {
                "id",
                "workspace_id",
                "originating_actor_user_id",
                "run_id",
                "action_intent_id",
                "tool_name",
                "effect",
                "args_digest",
                "status",
                "attempt",
                "latency_ms",
                "result_summary",
                "error_category",
                "started_at",
                "finished_at",
                "created_at",
                "updated_at",
            },
        }
        expected_gate4_checks = {
            "conversations": {"ck_conversations_title"},
            "messages": {
                "ck_messages_content",
                "ck_messages_role",
                "ck_messages_role_actor",
            },
            "runs": {
                "ck_runs_create_request_identity",
                "ck_runs_cancel_time_order",
                "ck_runs_completed_fields",
                "ck_runs_error_category",
                "ck_runs_failed_fields",
                "ck_runs_finished_at",
                "ck_runs_graph_version",
                "ck_runs_input_json",
                "ck_runs_limits_json",
                "ck_runs_mode",
                "ck_runs_mode_graph_family",
                "ck_runs_resume_no_legacy_approval",
                "ck_runs_next_event_seq",
                "ck_runs_queued_fields",
                "ck_runs_result_json",
                "ck_runs_started_at",
                "ck_runs_status",
                "ck_runs_time_order",
            },
            "run_jobs": {
                "ck_run_jobs_attempts",
                "ck_run_jobs_dead_error",
                "ck_run_jobs_error_summary",
                "ck_run_jobs_lease_fields",
                "ck_run_jobs_status",
            },
            "run_events": {
                "ck_run_events_payload",
                "ck_run_events_seq",
                "ck_run_events_type",
                "ck_run_events_version",
            },
            "tool_invocations": {
                "ck_tool_invocations_args_digest",
                "ck_tool_invocations_attempt",
                "ck_tool_invocations_effect",
                "ck_tool_invocations_error_category",
                "ck_tool_invocations_action_binding",
                "ck_tool_invocations_latency_ms",
                "ck_tool_invocations_result_summary",
                "ck_tool_invocations_status",
                "ck_tool_invocations_terminal_fields",
                "ck_tool_invocations_time_order",
                "ck_tool_invocations_tool_name",
            },
        }
        expected_gate4_indexes = {
            "conversations": {"ix_conversations_workspace_id"},
            "messages": {"ix_messages_workspace_id"},
            "runs": {"ix_runs_workspace_id"},
            "run_jobs": {
                "ix_run_jobs_due_claim",
                "ix_run_jobs_stale_lease",
                "ix_run_jobs_workspace_id",
            },
            "run_events": {
                "ix_run_events_workspace_id",
                "ix_run_events_workspace_run_seq",
            },
            "tool_invocations": {
                "ix_tool_invocations_workspace_id",
                "ix_tool_invocations_workspace_run",
            },
        }
        expected_gate4_foreign_keys = {
            "conversations": {
                "fk_conversations_creator_membership",
                "fk_conversations_workspace_id_workspaces",
            },
            "messages": {
                "fk_messages_actor_membership",
                "fk_messages_conversation",
                "fk_messages_workspace_id_workspaces",
            },
            "runs": {
                "fk_runs_conversation",
                "fk_runs_creator_membership",
                "fk_runs_resume_document",
                "fk_runs_request_message",
                "fk_runs_workspace_id_workspaces",
            },
            "run_jobs": {
                "fk_run_jobs_actor_membership",
                "fk_run_jobs_resume_approval_request",
                "fk_run_jobs_run",
                "fk_run_jobs_workspace_id_workspaces",
            },
            "run_events": {
                "fk_run_events_actor_membership",
                "fk_run_events_run",
                "fk_run_events_workspace_id_workspaces",
            },
            "tool_invocations": {
                "fk_tool_invocations_action_intent",
                "fk_tool_invocations_actor_membership",
                "fk_tool_invocations_run",
                "fk_tool_invocations_workspace_id_workspaces",
            },
        }
        expected_gate4_unique_constraints = {
            "conversations": {"uq_conversations_workspace_id_id"},
            "messages": {"uq_messages_workspace_id_conversation_id_id"},
            "runs": {
                "uq_runs_workspace_id_id",
                "uq_runs_workspace_creator_request",
            },
            "run_jobs": {"uq_run_jobs_run_id"},
            "run_events": {"uq_run_events_run_id_seq"},
            "tool_invocations": {"uq_tool_invocations_action_intent_id"},
        }
        for table_name, expected_columns in expected_gate4_columns.items():
            assert {item["name"] for item in inspector.get_columns(table_name)} == (
                expected_columns
            )
            assert {
                item["name"] for item in inspector.get_check_constraints(table_name)
            } == expected_gate4_checks[table_name]
            assert {item["name"] for item in inspector.get_indexes(table_name)} == (
                expected_gate4_indexes[table_name] | expected_gate4_unique_constraints[table_name]
            )
            foreign_keys = inspector.get_foreign_keys(table_name)
            assert {item["name"] for item in foreign_keys} == (
                expected_gate4_foreign_keys[table_name]
            )
            assert all(item["options"].get("ondelete") == "RESTRICT" for item in foreign_keys)
            assert {
                item["name"] for item in inspector.get_unique_constraints(table_name)
            } == expected_gate4_unique_constraints[table_name]
        for table_name in APP_TABLES:
            id_column = next(
                item for item in inspector.get_columns(table_name) if item["name"] == "id"
            )
            assert id_column["nullable"] is False
            assert "gen_random_uuid()" in str(id_column["default"])

        command.downgrade(config, "base")
        inspector.clear_cache()
        assert _app_tables(engine) == set()

        command.upgrade(config, "head")
        inspector.clear_cache()
        assert _app_tables(engine) == APP_TABLES
    finally:
        engine.dispose()


def test_gate65_downgrade_fails_closed_for_recovery_event(
    migrated_database_url: str,
) -> None:
    engine = sa.create_engine(migrated_database_url)
    with engine.begin() as connection:
        user_id, workspace_id, run_id, _conversation_id = _seed_gate6_migration_parent(
            connection,
            subject="gate65-downgrade",
            graph_version="pathfinder-research-v6",
        )
        connection.execute(
            sa.text(
                "INSERT INTO run_events "
                "(workspace_id, run_id, actor_user_id, seq, type, payload) "
                "VALUES (:workspace_id, :run_id, :user_id, 1, 'action.started', '{}'::jsonb)"
            ),
            {"workspace_id": workspace_id, "run_id": run_id, "user_id": user_id},
        )

    with pytest.raises(RuntimeError, match=r"Gate 6\.5 recovery data cannot be safely downgraded"):
        command.downgrade(
            alembic_config(migrated_database_url),
            "0013_gate6_mock_action_execution",
        )

    with connect_database(migrated_database_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0016_r1_execution_contracts",
        )
    engine.dispose()


def test_gate65_clean_downgrade_restores_gate4_invocation_contract(database_url: str) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "head")
        command.downgrade(config, "0013_gate6_mock_action_execution")
        inspector = sa.inspect(engine)
        tool_checks = {
            item["name"]: item["sqltext"]
            for item in inspector.get_check_constraints("tool_invocations")
        }
        assert "ck_tool_invocations_gate4_status" in tool_checks
        assert "outcome_unknown" not in tool_checks["ck_tool_invocations_terminal_fields"]
        event_check = next(
            item
            for item in inspector.get_check_constraints("run_events")
            if item["name"] == "ck_run_events_type"
        )
        assert "action.started" not in event_check["sqltext"]
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0013_gate6_mock_action_execution"
            )
        command.upgrade(config, "head")
    finally:
        engine.dispose()


def test_gate65_database_enforces_unknown_terminal_shape_and_lifecycle_events(
    migrated_database_url: str,
) -> None:
    engine = sa.create_engine(migrated_database_url)
    try:
        with engine.begin() as connection:
            user_id, workspace_id, run_id, _conversation_id = _seed_gate6_migration_parent(
                connection,
                subject="gate65-database-contract",
                graph_version="pathfinder-research-v6",
            )
            action_id = connection.scalar(
                sa.text(
                    "INSERT INTO action_intents "
                    "(workspace_id, originating_actor_user_id, run_id, action_key, "
                    "action_revision, tool_name, effect, args_snapshot, "
                    "canonicalization_version, args_digest, target_snapshot, "
                    "target_canonicalization_version, target_digest, "
                    "approval_binding_version, approval_binding_digest, status, "
                    "idempotency_key, recovery_attempts, evidence) VALUES "
                    "(:workspace_id, :user_id, :run_id, 'submit_application', 1, "
                    "'submit_mock_application', 'irreversible', '{}'::jsonb, 1, "
                    ":args_digest, '{}'::jsonb, 1, :target_digest, 1, :binding_digest, "
                    "'outcome_unknown', gen_random_uuid()::text, 1, "
                    '\'{"classification": "recovery_exhausted"}\'::jsonb) RETURNING id'
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "run_id": run_id,
                    "args_digest": f"sha256:{'a' * 64}",
                    "target_digest": f"sha256:{'b' * 64}",
                    "binding_digest": f"sha256:{'c' * 64}",
                },
            )
            invocation_id = connection.scalar(
                sa.text(
                    "INSERT INTO tool_invocations "
                    "(workspace_id, originating_actor_user_id, run_id, action_intent_id, "
                    "tool_name, effect, args_digest, status, attempt, latency_ms, "
                    "error_category, started_at, finished_at) VALUES "
                    "(:workspace_id, :user_id, :run_id, :action_id, "
                    "'submit_mock_application', 'irreversible', :args_digest, "
                    "'outcome_unknown', 1, 0, 'external_outcome_unknown', now(), now()) "
                    "RETURNING id"
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "run_id": run_id,
                    "action_id": action_id,
                    "args_digest": f"sha256:{'a' * 64}",
                },
            )
            for sequence, event_type in enumerate(
                (
                    "action.started",
                    "action.completed",
                    "action.failed",
                    "action.outcome_unknown",
                ),
                start=1,
            ):
                connection.execute(
                    sa.text(
                        "INSERT INTO run_events "
                        "(workspace_id, run_id, actor_user_id, seq, type, payload) "
                        "VALUES (:workspace_id, :run_id, :user_id, :seq, :event_type, '{}'::jsonb)"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "run_id": run_id,
                        "user_id": user_id,
                        "seq": sequence,
                        "event_type": event_type,
                    },
                )

        with pytest.raises(sa.exc.IntegrityError), engine.begin() as connection:
            connection.execute(
                sa.text("UPDATE tool_invocations SET latency_ms = NULL WHERE id = :invocation_id"),
                {"invocation_id": invocation_id},
            )

        with pytest.raises(
            RuntimeError, match=r"Gate 6\.5 recovery data cannot be safely downgraded"
        ):
            command.downgrade(
                alembic_config(migrated_database_url),
                "0013_gate6_mock_action_execution",
            )
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0016_r1_execution_contracts"
            )
    finally:
        engine.dispose()


def test_gate6_upgrade_from_0009_and_empty_downgrade_reupgrade(database_url: str) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "0009_gate5_graph_rag")
        before = sa.inspect(engine)
        assert "action_intents" not in before.get_table_names()
        assert "approval_requests" not in before.get_table_names()
        graph_before = next(
            item for item in before.get_columns("runs") if item["name"] == "graph_version"
        )
        assert "pathfinder-research-v2" in str(graph_before["default"])

        command.upgrade(config, "0010_gate6_action_proposals")
        after = sa.inspect(engine)
        assert {"action_intents", "approval_requests"}.issubset(after.get_table_names())
        graph_after = next(
            item for item in after.get_columns("runs") if item["name"] == "graph_version"
        )
        assert "pathfinder-research-v3" in str(graph_after["default"])

        command.downgrade(config, "0009_gate5_graph_rag")
        downgraded = sa.inspect(engine)
        assert "action_intents" not in downgraded.get_table_names()
        assert "approval_requests" not in downgraded.get_table_names()
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0016_r1_execution_contracts"
            )
    finally:
        engine.dispose()


def _seed_gate6_migration_parent(
    connection: sa.Connection,
    *,
    subject: str,
    graph_version: str,
) -> tuple[object, object, object, object]:
    user_id = connection.scalar(
        sa.text("INSERT INTO users (auth_subject) VALUES (:subject) RETURNING id"),
        {"subject": subject},
    )
    workspace_id = connection.scalar(
        sa.text(
            "INSERT INTO workspaces (kind, name, created_by_user_id) "
            "VALUES ('team', :name, :user_id) RETURNING id"
        ),
        {"name": subject, "user_id": user_id},
    )
    connection.execute(
        sa.text(
            "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
            "VALUES (:workspace_id, :user_id, 'admin')"
        ),
        {"workspace_id": workspace_id, "user_id": user_id},
    )
    conversation_id = connection.scalar(
        sa.text(
            "INSERT INTO conversations (workspace_id, created_by_user_id, title) "
            "VALUES (:workspace_id, :user_id, 'Gate 6 downgrade') RETURNING id"
        ),
        {"workspace_id": workspace_id, "user_id": user_id},
    )
    message_id = connection.scalar(
        sa.text(
            "INSERT INTO messages "
            "(workspace_id, conversation_id, actor_user_id, role, content) "
            "VALUES (:workspace_id, :conversation_id, :user_id, 'user', 'request') "
            "RETURNING id"
        ),
        {
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
        },
    )
    run_id = connection.scalar(
        sa.text(
            "INSERT INTO runs "
            "(workspace_id, created_by_user_id, conversation_id, request_message_id, "
            "mode, input_json, limits_json, status, graph_version, started_at) "
            "VALUES (:workspace_id, :user_id, :conversation_id, :message_id, 'research', "
            "'{}'::jsonb, '{}'::jsonb, 'running', :graph_version, now()) RETURNING id"
        ),
        {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "graph_version": graph_version,
        },
    )
    return user_id, workspace_id, run_id, conversation_id


@pytest.mark.parametrize(
    "unsafe_case",
    ["v3_run", "action_intent", "approval_request", "action_event", "approval_event"],
)
def test_gate6_downgrade_fails_closed_for_every_gate6_business_fact(
    database_url: str,
    unsafe_case: str,
) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "head")
        with engine.begin() as connection:
            user_id, workspace_id, run_id, _conversation_id = _seed_gate6_migration_parent(
                connection,
                subject=f"gate6-downgrade-{unsafe_case}",
                graph_version=(
                    "pathfinder-research-v3"
                    if unsafe_case == "v3_run"
                    else "pathfinder-research-v2"
                ),
            )
            action_id = None
            if unsafe_case in {"action_intent", "approval_request"}:
                action_id = connection.scalar(
                    sa.text(
                        "INSERT INTO action_intents "
                        "(workspace_id, originating_actor_user_id, run_id, action_key, "
                        "action_revision, tool_name, effect, args_snapshot, "
                        "canonicalization_version, args_digest, target_snapshot, "
                        "target_canonicalization_version, target_digest, "
                        "approval_binding_version, approval_binding_digest, status, "
                        "idempotency_key, recovery_attempts) VALUES "
                        "(:workspace_id, :user_id, :run_id, 'submit_application', 1, "
                        "'submit_mock_application', 'irreversible', '{}'::jsonb, 1, "
                        ":args_digest, '{}'::jsonb, 1, :target_digest, 1, :binding_digest, "
                        "'proposed', gen_random_uuid()::text, 0) RETURNING id"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "run_id": run_id,
                        "args_digest": f"sha256:{'a' * 64}",
                        "target_digest": f"sha256:{'b' * 64}",
                        "binding_digest": f"sha256:{'c' * 64}",
                    },
                )
            if unsafe_case == "approval_request":
                connection.execute(
                    sa.text(
                        "INSERT INTO approval_requests "
                        "(workspace_id, run_id, action_intent_id, status, args_digest, "
                        "target_digest, approval_binding_version, approval_binding_digest, "
                        "policy_version, policy_snapshot, version, expires_at) VALUES "
                        "(:workspace_id, :run_id, :action_id, 'pending', :args_digest, "
                        ":target_digest, 1, :binding_digest, 1, '{}'::jsonb, 1, "
                        "now() + interval '1 hour')"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "run_id": run_id,
                        "action_id": action_id,
                        "args_digest": f"sha256:{'a' * 64}",
                        "target_digest": f"sha256:{'b' * 64}",
                        "binding_digest": f"sha256:{'c' * 64}",
                    },
                )
            if unsafe_case in {"action_event", "approval_event"}:
                connection.execute(
                    sa.text(
                        "INSERT INTO run_events "
                        "(workspace_id, run_id, actor_user_id, seq, type, payload) VALUES "
                        "(:workspace_id, :run_id, :user_id, 1, :event_type, '{}'::jsonb)"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "run_id": run_id,
                        "user_id": user_id,
                        "event_type": (
                            "action.proposed"
                            if unsafe_case == "action_event"
                            else "approval.expired"
                        ),
                    },
                )

        with pytest.raises(RuntimeError, match="Gate 6 data cannot be safely downgraded"):
            command.downgrade(config, "0009_gate5_graph_rag")
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0016_r1_execution_contracts"
            )
    finally:
        engine.dispose()


def test_step_32_upgrade_preserves_historical_embedding_null_usage(database_url: str) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "0002_llm_invocations")
        with engine.begin() as connection:
            user_id = connection.scalar(
                sa.text(
                    """
                    INSERT INTO users (auth_subject)
                    VALUES ('historical-embedding-user')
                    RETURNING id
                    """
                )
            )
            workspace_id = connection.scalar(
                sa.text(
                    """
                    INSERT INTO workspaces (kind, name, created_by_user_id)
                    VALUES ('team', 'Historical Embedding Workspace', :user_id)
                    RETURNING id
                    """
                ),
                {"user_id": user_id},
            )
            connection.execute(
                sa.text(
                    """
                    INSERT INTO workspace_memberships (workspace_id, user_id, role)
                    VALUES (:workspace_id, :user_id, 'admin')
                    """
                ),
                {"workspace_id": workspace_id, "user_id": user_id},
            )
            invocation_id = connection.scalar(
                sa.text(
                    """
                    INSERT INTO llm_invocations (
                        workspace_id, actor_user_id, invocation_kind, provider, model,
                        graph_node, prompt_version, request_hash, token_usage,
                        latency_ms, status, error_category
                    ) VALUES (
                        :workspace_id, :user_id, 'embedding', 'fake',
                        'text-embedding-3-small', 'ingest_documents', NULL,
                        :request_hash, NULL, 3, 'succeeded', NULL
                    )
                    RETURNING id
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "request_hash": f"sha256:{'d' * 64}",
                },
            )

        command.upgrade(config, "head")

        with engine.connect() as connection:
            row = connection.execute(
                sa.text(
                    """
                    SELECT token_usage, provider_response_id, pricing_version,
                           currency, estimated_cost, trace_ids
                    FROM llm_invocations
                    WHERE id = :invocation_id
                    """
                ),
                {"invocation_id": invocation_id},
            ).one()
        assert row.token_usage is None
        assert row.provider_response_id is None
        assert row.pricing_version is None
        assert row.currency is None
        assert row.estimated_cost is None
        assert row.trace_ids is None
    finally:
        engine.dispose()


def test_qwen_profile_migration_preserves_legacy_rows_and_downgrade_fails_closed(
    database_url: str,
) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "0005_langfuse_tracing")
        with engine.begin() as connection:
            user_id = connection.scalar(
                sa.text(
                    "INSERT INTO users (auth_subject) VALUES ('qwen-migration-user') RETURNING id"
                )
            )
            workspace_id = connection.scalar(
                sa.text(
                    "INSERT INTO workspaces (kind, name, created_by_user_id) "
                    "VALUES ('team', 'Qwen Migration Workspace', :user_id) RETURNING id"
                ),
                {"user_id": user_id},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
                    "VALUES (:workspace_id, :user_id, 'admin')"
                ),
                {"workspace_id": workspace_id, "user_id": user_id},
            )
            legacy_id = connection.scalar(
                sa.text(
                    """
                    INSERT INTO llm_invocations (
                        workspace_id, actor_user_id, invocation_kind, provider, model,
                        graph_node, prompt_version, request_hash, status
                    ) VALUES (
                        :workspace_id, :user_id, 'chat', 'openai', 'gpt-5.6-terra',
                        'plan', :prompt_version, :request_hash, 'started'
                    ) RETURNING id
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "prompt_version": f"sha256:{'a' * 64}",
                    "request_hash": f"sha256:{'b' * 64}",
                },
            )

        command.upgrade(config, "head")
        with engine.begin() as connection:
            legacy = connection.execute(
                sa.text("SELECT provider, model, status FROM llm_invocations WHERE id = :id"),
                {"id": legacy_id},
            ).one()
            assert legacy == ("openai", "gpt-5.6-terra", "started")
            connection.execute(
                sa.text(
                    """
                    INSERT INTO llm_invocations (
                        workspace_id, actor_user_id, invocation_kind, provider, model,
                        graph_node, prompt_version, request_hash, status
                    ) VALUES (
                        :workspace_id, :user_id, 'chat', 'qwen',
                        'qwen3.6-flash-2026-04-16', 'plan', :prompt_version,
                        :request_hash, 'started'
                    )
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "prompt_version": f"sha256:{'c' * 64}",
                    "request_hash": f"sha256:{'d' * 64}",
                },
            )

        with pytest.raises(RuntimeError, match="Qwen provider-profile evidence exists"):
            command.downgrade(config, "0005_langfuse_tracing")

        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT count(*) FROM llm_invocations")) == 2
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0016_r1_execution_contracts"
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "unsafe_case",
    ["application", "resume_binding", "v2_result", "rag_retrieved"],
)
def test_gate5_migration_downgrade_fails_closed_when_gate5_evidence_exists(
    database_url: str,
    unsafe_case: str,
) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "head")
        with engine.begin() as connection:
            user_id = connection.scalar(
                sa.text("INSERT INTO users (auth_subject) VALUES (:subject) RETURNING id"),
                {"subject": f"gate5-downgrade-{unsafe_case}"},
            )
            workspace_id = connection.scalar(
                sa.text(
                    "INSERT INTO workspaces (kind, name, created_by_user_id) "
                    "VALUES ('team', :name, :user_id) RETURNING id"
                ),
                {"name": f"Gate5 downgrade {unsafe_case}", "user_id": user_id},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
                    "VALUES (:workspace_id, :user_id, 'admin')"
                ),
                {"workspace_id": workspace_id, "user_id": user_id},
            )
            conversation_id = connection.scalar(
                sa.text(
                    "INSERT INTO conversations "
                    "(workspace_id, created_by_user_id, title) "
                    "VALUES (:workspace_id, :user_id, 'Downgrade evidence') RETURNING id"
                ),
                {"workspace_id": workspace_id, "user_id": user_id},
            )
            message_id = connection.scalar(
                sa.text(
                    "INSERT INTO messages "
                    "(workspace_id, conversation_id, actor_user_id, role, content) "
                    "VALUES (:workspace_id, :conversation_id, :user_id, 'user', 'request') "
                    "RETURNING id"
                ),
                {
                    "workspace_id": workspace_id,
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                },
            )
            run_id = connection.scalar(
                sa.text(
                    "INSERT INTO runs "
                    "(workspace_id, created_by_user_id, conversation_id, request_message_id, "
                    "mode, input_json, limits_json, graph_version) "
                    "VALUES (:workspace_id, :user_id, :conversation_id, :message_id, "
                    "'research', CAST(:input_json AS jsonb), CAST(:limits_json AS jsonb), "
                    "'pathfinder-research-v1') RETURNING id"
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "input_json": json.dumps({"query": "request"}),
                    "limits_json": json.dumps({"max_iterations": 1}),
                },
            )

            if unsafe_case == "application":
                connection.execute(
                    sa.text("UPDATE runs SET mode = 'application' WHERE id = :run_id"),
                    {"run_id": run_id},
                )
            elif unsafe_case == "resume_binding":
                document_id = connection.scalar(
                    sa.text(
                        "INSERT INTO documents "
                        "(workspace_id, created_by_user_id, title, source_type, source_name, "
                        "content, content_hash, normalization_version, chunking_version, "
                        "embedding_model) VALUES "
                        "(:workspace_id, :user_id, 'Resume', 'markdown', 'resume.md', "
                        "'resume', :content_hash, 'text-normalization-v1', "
                        "'document-chunking-v1', "
                        "'qwen-beijing-text-embedding-v4-1536-v1') RETURNING id"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "content_hash": "a" * 64,
                    },
                )
                connection.execute(
                    sa.text("UPDATE runs SET resume_document_id = :document_id WHERE id = :run_id"),
                    {"document_id": document_id, "run_id": run_id},
                )
            elif unsafe_case == "v2_result":
                connection.execute(
                    sa.text(
                        "UPDATE runs SET status = 'completed', started_at = now(), "
                        "finished_at = now(), result_json = CAST(:result AS jsonb) "
                        "WHERE id = :run_id"
                    ),
                    {"result": json.dumps({"schema_version": 2}), "run_id": run_id},
                )
            else:
                connection.execute(
                    sa.text(
                        "INSERT INTO run_events "
                        "(workspace_id, run_id, actor_user_id, seq, type, version, payload) "
                        "VALUES (:workspace_id, :run_id, :user_id, 1, 'rag.retrieved', 1, "
                        "CAST(:payload AS jsonb))"
                    ),
                    {
                        "workspace_id": workspace_id,
                        "run_id": run_id,
                        "user_id": user_id,
                        "payload": json.dumps({"result_count": 0}),
                    },
                )

        with pytest.raises(RuntimeError, match="Gate 5 data cannot be safely downgraded"):
            command.downgrade(config, "0008_rag_document_storage")

        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == (
                "0016_r1_execution_contracts"
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("status", "trace_ids"),
    [
        ("started", {"trace_id": "1" * 32, "observation_id": "2" * 16}),
        ("failed", {}),
        ("failed", {"trace_id": "1" * 32}),
        (
            "failed",
            {"trace_id": "1" * 32, "observation_id": "2" * 16, "extra": "forged"},
        ),
        ("failed", {"trace_id": 1, "observation_id": "2" * 16}),
        ("failed", {"trace_id": "A" * 32, "observation_id": "2" * 16}),
        ("failed", {"trace_id": "0" * 32, "observation_id": "2" * 16}),
        ("failed", {"trace_id": "1" * 32, "observation_id": "0" * 16}),
    ],
)
def test_trace_id_constraint_rejects_invalid_shape_identifiers_and_started_rows(
    database_url: str,
    status: str,
    trace_ids: dict[str, object],
) -> None:
    config = alembic_config(database_url)
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            user_id = connection.scalar(
                sa.text(
                    "INSERT INTO users (auth_subject) VALUES ('trace-constraint-user') RETURNING id"
                )
            )
            workspace_id = connection.scalar(
                sa.text(
                    "INSERT INTO workspaces (kind, name, created_by_user_id) "
                    "VALUES ('team', 'Trace Constraint', :user_id) RETURNING id"
                ),
                {"user_id": user_id},
            )
            connection.execute(
                sa.text(
                    "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
                    "VALUES (:workspace_id, :user_id, 'admin')"
                ),
                {"workspace_id": workspace_id, "user_id": user_id},
            )

        with pytest.raises(sa.exc.IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    sa.text(
                        """
                        INSERT INTO llm_invocations (
                            workspace_id, actor_user_id, invocation_kind, provider, model,
                            graph_node, prompt_version, request_hash, trace_ids,
                            latency_ms, status, error_category
                        ) VALUES (
                            :workspace_id, :user_id, 'chat', 'fake', 'gpt-5.6-terra',
                            'plan', :prompt_version, :request_hash,
                            CAST(:trace_ids AS jsonb), :latency_ms, :status, :error_category
                        )
                        """
                    ),
                    {
                        "workspace_id": workspace_id,
                        "user_id": user_id,
                        "prompt_version": f"sha256:{'a' * 64}",
                        "request_hash": f"sha256:{'b' * 64}",
                        "trace_ids": json.dumps(trace_ids),
                        "latency_ms": None if status == "started" else 1,
                        "status": status,
                        "error_category": None if status == "started" else "provider_error",
                    },
                )
    finally:
        engine.dispose()
