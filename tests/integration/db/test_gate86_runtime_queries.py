from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from psycopg import Connection
from psycopg.types.json import Jsonb

from tests.integration.support import connect_database

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[3]
METRICS_SQL = PROJECT_ROOT / "scripts" / "gate86_runtime_metrics.sql"
RECOVERY_SQL = PROJECT_ROOT / "scripts" / "gate86_recovery_state.sql"
OBSERVED_AT = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
PROMPT_VERSION = f"sha256:{'a' * 64}"
REQUEST_HASH = f"sha256:{'b' * 64}"
ARGS_DIGEST = f"sha256:{'c' * 64}"
TARGET_DIGEST = f"sha256:{'d' * 64}"
BINDING_DIGEST = f"sha256:{'e' * 64}"


def _runtime_metrics(database_url: str) -> dict[str, object]:
    sql_text = "\n".join(
        line
        for line in METRICS_SQL.read_text(encoding="utf-8").splitlines()
        if not line.startswith("\\")
    )
    with connect_database(database_url, autocommit=True) as connection:
        connection.execute(
            "SELECT set_config('pathfinder.gate86_observed_at', %s, false)",
            (OBSERVED_AT.isoformat(),),
        )
        connection.autocommit = False
        with connection.cursor() as cursor:
            cursor.execute(sql_text, prepare=False)
            row = None
            while True:
                if cursor.description is not None:
                    row = cursor.fetchone()
                if not cursor.nextset():
                    break
        connection.rollback()
    assert row is not None and isinstance(row[0], str)
    return json.loads(row[0])


def _empty_recovery_snapshot(database_url: str) -> dict[str, object]:
    fixture_id = "00000000-0000-4000-8000-000000000000"
    sql_text = "\n".join(
        line
        for line in RECOVERY_SQL.read_text(encoding="utf-8").splitlines()
        if not line.startswith("\\")
    )
    for variable in ("baseline", "waiting", "leased", "queued"):
        sql_text = sql_text.replace(f":'{variable}_run_id'", f"'{fixture_id}'")
    with connect_database(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_text, prepare=False)
            row = None
            while True:
                if cursor.description is not None:
                    row = cursor.fetchone()
                if not cursor.nextset():
                    break
        connection.rollback()
    assert row is not None and isinstance(row[0], str)
    return json.loads(row[0])


def _create_identity(connection: Connection[object]) -> tuple[UUID, UUID]:
    user_id = uuid4()
    workspace_id = uuid4()
    connection.execute(
        "INSERT INTO users (id, auth_subject) VALUES (%s, %s)",
        (user_id, f"gate86-{user_id}"),
    )
    connection.execute(
        "INSERT INTO workspaces (id, kind, name, created_by_user_id) "
        "VALUES (%s, 'personal', 'Gate 8.6 metrics', %s)",
        (workspace_id, user_id),
    )
    connection.execute(
        "INSERT INTO workspace_memberships "
        "(id, workspace_id, user_id, role) VALUES (%s, %s, %s, 'admin')",
        (uuid4(), workspace_id, user_id),
    )
    return user_id, workspace_id


def _create_run_and_job(
    connection: Connection[object],
    *,
    actor_id: UUID,
    workspace_id: UUID,
    run_status: str,
    job_status: str,
    created_at: datetime,
    available_at: datetime,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error_category: str | None = None,
    duration_result: bool = False,
    lease_expires_at: datetime | None = None,
) -> UUID:
    conversation_id = uuid4()
    message_id = uuid4()
    run_id = uuid4()
    connection.execute(
        "INSERT INTO conversations "
        "(id, workspace_id, created_by_user_id, title, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (
            conversation_id,
            workspace_id,
            actor_id,
            f"Gate 8.6 {run_id}",
            created_at,
            created_at,
        ),
    )
    connection.execute(
        "INSERT INTO messages "
        "(id, workspace_id, conversation_id, actor_user_id, role, content, created_at) "
        "VALUES (%s, %s, %s, %s, 'user', 'synthetic metrics fixture', %s)",
        (message_id, workspace_id, conversation_id, actor_id, created_at),
    )
    result_json = (
        Jsonb({"evidence_sufficient": False, "limitations": []}) if duration_result else None
    )
    connection.execute(
        "INSERT INTO runs "
        "(id, workspace_id, created_by_user_id, conversation_id, request_message_id, "
        "mode, input_json, limits_json, status, graph_version, next_event_seq, result_json, "
        "error_category, started_at, finished_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, 'research', %s, %s, %s, "
        "'pathfinder-research-v6', 1, %s, %s, %s, %s, %s, %s)",
        (
            run_id,
            workspace_id,
            actor_id,
            conversation_id,
            message_id,
            Jsonb({"query": "synthetic metrics fixture"}),
            Jsonb({}),
            run_status,
            result_json,
            error_category,
            started_at,
            finished_at,
            created_at,
            finished_at or created_at,
        ),
    )
    leased = job_status == "leased"
    connection.execute(
        "INSERT INTO run_jobs "
        "(id, workspace_id, originating_actor_user_id, run_id, status, attempt, "
        "max_attempts, available_at, leased_by, owner_token, lease_expires_at, "
        "error_summary, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, 3, %s, %s, %s, %s, %s, %s, %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            job_status,
            1 if leased or job_status == "dead" else 0,
            available_at,
            "gate86-metrics-worker" if leased else None,
            uuid4() if leased else None,
            lease_expires_at if leased else None,
            "synthetic_dead_job" if job_status == "dead" else None,
            created_at,
            finished_at or created_at,
        ),
    )
    return run_id


def _insert_llm(
    connection: Connection[object],
    *,
    workspace_id: UUID,
    actor_id: UUID,
    run_id: UUID,
    provider: str,
    status: str,
    latency_ms: int,
    estimated_cost: Decimal | None = None,
    error_category: str | None = None,
) -> None:
    succeeded = status == "succeeded"
    connection.execute(
        "INSERT INTO llm_invocations "
        "(id, workspace_id, actor_user_id, run_id, invocation_kind, provider, model, "
        "graph_node, prompt_version, request_hash, provider_response_id, token_usage, "
        "pricing_version, currency, estimated_cost, latency_ms, status, error_category, "
        "created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'chat', %s, 'qwen3.6-flash-2026-04-16', "
        "'write_report', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            provider,
            PROMPT_VERSION,
            REQUEST_HASH,
            f"response-{uuid4()}" if succeeded else None,
            Jsonb({"input_tokens": 2, "output_tokens": 3}) if succeeded else None,
            "qwen-cn-beijing-cny-2026-08-13-v1" if estimated_cost is not None else None,
            "CNY" if estimated_cost is not None else None,
            estimated_cost,
            latency_ms,
            status,
            error_category,
            OBSERVED_AT - timedelta(minutes=10),
            OBSERVED_AT - timedelta(minutes=10),
        ),
    )


def _insert_tool(
    connection: Connection[object],
    *,
    workspace_id: UUID,
    actor_id: UUID,
    run_id: UUID,
    status: str,
    latency_ms: int,
    action_intent_id: UUID | None = None,
) -> None:
    started_at = OBSERVED_AT - timedelta(minutes=5)
    connection.execute(
        "INSERT INTO tool_invocations "
        "(id, workspace_id, originating_actor_user_id, run_id, action_intent_id, "
        "tool_name, effect, args_digest, status, attempt, latency_ms, result_summary, "
        "error_category, started_at, finished_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            action_intent_id,
            "submit_mock_application" if action_intent_id else "search_web",
            "irreversible" if action_intent_id else "read_only",
            ARGS_DIGEST,
            status,
            latency_ms,
            Jsonb({"result": "bounded"}) if status == "succeeded" else None,
            (
                "external_outcome_unknown"
                if status == "outcome_unknown"
                else "provider_timeout"
                if status == "failed"
                else None
            ),
            started_at,
            started_at + timedelta(milliseconds=latency_ms),
            OBSERVED_AT - timedelta(minutes=5),
            OBSERVED_AT - timedelta(minutes=5),
        ),
    )


def test_metrics_query_empty_database_is_safe(migrated_database_url: str) -> None:
    metrics = _runtime_metrics(migrated_database_url)

    assert metrics["queue"] == {
        "dead_total": 0,
        "expired_leases_total": 0,
        "leased_total": 0,
        "oldest_queued_age_seconds": None,
        "oldest_ready_queued_age_seconds": None,
        "queued_total": 0,
        "ready_queued_total": 0,
    }
    assert metrics["runs"]["failure_sample_count"] == 0
    assert metrics["runs"]["failure_rate"] is None
    assert metrics["cost_24h"]["known_estimated_cny_total"] is None
    assert metrics["storage"]["database_bytes"] > 0


def test_recovery_snapshot_query_is_read_only_and_handles_missing_fixtures(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        connection.execute("CREATE SCHEMA pathfinder_checkpoint")
        connection.execute(
            "CREATE TABLE pathfinder_checkpoint.checkpoints "
            "(thread_id text NOT NULL, checkpoint_ns text NOT NULL, checkpoint_id text NOT NULL)"
        )
        connection.commit()

    snapshot = _empty_recovery_snapshot(migrated_database_url)

    assert set(snapshot["fixtures"]) == {"baseline", "leased", "queued", "waiting"}
    assert snapshot["checks"]["all_fixture_names_present"] is True
    assert snapshot["checks"]["all_fixture_runs_present"] is False


def test_metrics_query_reports_deterministic_runtime_signals(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        actor_id, workspace_id = _create_identity(connection)
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="queued",
            job_status="queued",
            created_at=OBSERVED_AT - timedelta(seconds=600),
            available_at=OBSERVED_AT - timedelta(seconds=10),
        )
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="queued",
            job_status="queued",
            created_at=OBSERVED_AT - timedelta(seconds=700),
            available_at=OBSERVED_AT + timedelta(seconds=300),
        )
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="queued",
            job_status="leased",
            created_at=OBSERVED_AT - timedelta(seconds=100),
            available_at=OBSERVED_AT - timedelta(seconds=100),
            lease_expires_at=OBSERVED_AT + timedelta(seconds=30),
        )
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="queued",
            job_status="leased",
            created_at=OBSERVED_AT - timedelta(seconds=200),
            available_at=OBSERVED_AT - timedelta(seconds=200),
            lease_expires_at=OBSERVED_AT - timedelta(seconds=1),
        )
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="failed",
            job_status="dead",
            created_at=OBSERVED_AT - timedelta(hours=3),
            available_at=OBSERVED_AT - timedelta(hours=3),
            started_at=OBSERVED_AT - timedelta(hours=3),
            finished_at=OBSERVED_AT - timedelta(hours=2),
            error_category="synthetic_dead_job",
        )
        completed_run = _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="completed",
            job_status="done",
            created_at=OBSERVED_AT - timedelta(minutes=30),
            available_at=OBSERVED_AT - timedelta(minutes=30),
            started_at=OBSERVED_AT - timedelta(minutes=20, seconds=1),
            finished_at=OBSERVED_AT - timedelta(minutes=20),
            duration_result=True,
        )
        failed_run = _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="failed",
            job_status="done",
            created_at=OBSERVED_AT - timedelta(minutes=20),
            available_at=OBSERVED_AT - timedelta(minutes=20),
            started_at=OBSERVED_AT - timedelta(minutes=10, seconds=2),
            finished_at=OBSERVED_AT - timedelta(minutes=10),
            error_category="provider_unavailable",
        )
        _create_run_and_job(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_status="cancelled",
            job_status="done",
            created_at=OBSERVED_AT - timedelta(minutes=10),
            available_at=OBSERVED_AT - timedelta(minutes=10),
            started_at=OBSERVED_AT - timedelta(minutes=5, seconds=3),
            finished_at=OBSERVED_AT - timedelta(minutes=5),
        )

        _insert_llm(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=completed_run,
            provider="qwen",
            status="succeeded",
            latency_ms=100,
            estimated_cost=Decimal("1.250000000000"),
        )
        _insert_llm(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=completed_run,
            provider="qwen",
            status="succeeded",
            latency_ms=300,
        )
        _insert_llm(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=completed_run,
            provider="fake",
            status="succeeded",
            latency_ms=500,
        )
        _insert_llm(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=failed_run,
            provider="qwen",
            status="failed",
            latency_ms=700,
            error_category="provider_timeout",
        )

        _insert_tool(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=completed_run,
            status="succeeded",
            latency_ms=20,
        )
        _insert_tool(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=failed_run,
            status="failed",
            latency_ms=40,
        )
        action_id = uuid4()
        connection.execute(
            "INSERT INTO action_intents "
            "(id, workspace_id, originating_actor_user_id, run_id, action_key, "
            "action_revision, tool_name, effect, args_snapshot, canonicalization_version, "
            "args_digest, target_snapshot, target_canonicalization_version, target_digest, "
            "approval_binding_version, approval_binding_digest, status, idempotency_key, "
            "recovery_attempts, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, 'submit_application', 1, "
            "'submit_mock_application', 'irreversible', %s, 1, %s, %s, 1, %s, 1, %s, "
            "'outcome_unknown', %s, 1, %s, %s)",
            (
                action_id,
                workspace_id,
                actor_id,
                failed_run,
                Jsonb({}),
                ARGS_DIGEST,
                Jsonb({"provider": "mock_portal"}),
                TARGET_DIGEST,
                BINDING_DIGEST,
                str(action_id),
                OBSERVED_AT - timedelta(minutes=5),
                OBSERVED_AT - timedelta(minutes=5),
            ),
        )
        _insert_tool(
            connection,
            workspace_id=workspace_id,
            actor_id=actor_id,
            run_id=failed_run,
            status="outcome_unknown",
            latency_ms=60,
            action_intent_id=action_id,
        )
        connection.commit()

    metrics = _runtime_metrics(migrated_database_url)

    assert metrics["queue"] == {
        "dead_total": 1,
        "expired_leases_total": 1,
        "leased_total": 2,
        "oldest_queued_age_seconds": 700,
        "oldest_ready_queued_age_seconds": 600,
        "queued_total": 2,
        "ready_queued_total": 1,
    }
    assert metrics["runs"] == {
        "cancelled": 1,
        "completed": 1,
        "failed": 1,
        "failure_rate": 0.5,
        "failure_sample_count": 2,
    }
    assert metrics["latency"] == {
        "llm_p50_ms": 400.0,
        "llm_p95_ms": 670.0,
        "run_p50_ms": 2000.0,
        "run_p95_ms": 2900.0,
        "terminal_run_count": 3,
        "tool_p50_ms": 40.0,
        "tool_p95_ms": 58.0,
    }
    assert metrics["llm"] == {
        "error_categories": {"provider_timeout": 1},
        "failed": 1,
        "succeeded": 3,
    }
    assert metrics["tools"] == {"failed": 1, "outcome_unknown": 1, "succeeded": 1}
    assert metrics["cost_24h"] == {
        "known_estimated_cny_total": 1.25,
        "priced_attempts": 1,
        "unpriced_non_fake_succeeded_attempts": 1,
    }
    assert metrics["storage"]["database_bytes"] > 0
