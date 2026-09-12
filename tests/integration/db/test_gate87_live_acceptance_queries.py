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
ACCEPTANCE_SQL = PROJECT_ROOT / "scripts" / "gate87_live_acceptance.sql"
NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
ARGS_DIGEST = f"sha256:{'a' * 64}"
TARGET_DIGEST = f"sha256:{'b' * 64}"
BINDING_DIGEST = f"sha256:{'c' * 64}"
PROMPT_VERSION = f"sha256:{'d' * 64}"
REQUEST_HASH = f"sha256:{'e' * 64}"
TRACE_IDS = {"trace_id": "1" * 32, "observation_id": "2" * 16}


def _query_text(
    *, actor_id: UUID, workspace_id: UUID, document_id: UUID, run_id: UUID, cap: Decimal
) -> str:
    source = ACCEPTANCE_SQL.read_text(encoding="utf-8")
    query = source[source.index("WITH\nparameters AS") : source.index("\n\\gset gate87_")]
    return (
        query.replace(":'actor_user_id'", f"'{actor_id}'")
        .replace(":'workspace_id'", f"'{workspace_id}'")
        .replace(":'document_id'", f"'{document_id}'")
        .replace(":'run_id'", f"'{run_id}'")
        .replace(":'cost_cap_cny'", f"'{cap}'")
    )


def _identity(connection: Connection[object]) -> tuple[UUID, UUID]:
    actor_id = uuid4()
    workspace_id = uuid4()
    connection.execute(
        "INSERT INTO users (id, auth_subject) VALUES (%s, %s)",
        (actor_id, f"gate87-{actor_id}"),
    )
    connection.execute(
        "INSERT INTO workspaces (id, kind, name, created_by_user_id) "
        "VALUES (%s, 'personal', 'Gate 8.7 acceptance', %s)",
        (workspace_id, actor_id),
    )
    connection.execute(
        "INSERT INTO workspace_memberships "
        "(id, workspace_id, user_id, role) VALUES (%s, %s, %s, 'admin')",
        (uuid4(), workspace_id, actor_id),
    )
    return actor_id, workspace_id


def _document(connection: Connection[object], *, actor_id: UUID, workspace_id: UUID) -> UUID:
    document_id = uuid4()
    connection.execute(
        "INSERT INTO documents "
        "(id, workspace_id, created_by_user_id, title, source_type, source_name, content, "
        "content_hash, normalization_version, chunking_version, embedding_model, created_at) "
        "VALUES (%s, %s, %s, 'Synthetic resume', 'markdown', 'synthetic.md', "
        "'synthetic acceptance text', %s, 'nfkc-lf-v1', 'heading-greedy-v1', "
        "'qwen-beijing-text-embedding-v4-1536-v1', %s)",
        (document_id, workspace_id, actor_id, "1" * 64, NOW),
    )
    embedding = "[" + ",".join("0" for _ in range(1536)) + "]"
    connection.execute(
        "INSERT INTO document_chunks "
        "(id, workspace_id, document_id, ordinal, section, text, content_hash, token_count, "
        "embedding_model, embedding, created_at) "
        "VALUES (%s, %s, %s, 0, 'Resume', 'synthetic acceptance text', %s, 25, "
        "'qwen-beijing-text-embedding-v4-1536-v1', %s::vector, %s)",
        (uuid4(), workspace_id, document_id, "2" * 64, embedding, NOW),
    )
    return document_id


def _run(
    connection: Connection[object], *, actor_id: UUID, workspace_id: UUID, document_id: UUID
) -> UUID:
    conversation_id = uuid4()
    message_id = uuid4()
    run_id = uuid4()
    connection.execute(
        "INSERT INTO conversations "
        "(id, workspace_id, created_by_user_id, title, created_at, updated_at) "
        "VALUES (%s, %s, %s, 'Gate 8.7 live acceptance', %s, %s)",
        (conversation_id, workspace_id, actor_id, NOW, NOW),
    )
    connection.execute(
        "INSERT INTO messages "
        "(id, workspace_id, conversation_id, actor_user_id, role, content, created_at) "
        "VALUES (%s, %s, %s, %s, 'user', 'synthetic live query', %s)",
        (message_id, workspace_id, conversation_id, actor_id, NOW),
    )
    connection.execute(
        "INSERT INTO runs "
        "(id, workspace_id, created_by_user_id, conversation_id, request_message_id, mode, "
        "resume_document_id, input_json, limits_json, status, graph_version, next_event_seq, "
        "result_json, started_at, finished_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, 'application', %s, %s, %s, 'completed', "
        "'pathfinder-research-v6', 5, %s, %s, %s, %s, %s)",
        (
            run_id,
            workspace_id,
            actor_id,
            conversation_id,
            message_id,
            document_id,
            Jsonb({"query": "synthetic live query"}),
            Jsonb({}),
            Jsonb({"evidence_sufficient": True}),
            NOW,
            NOW + timedelta(seconds=3),
            NOW,
            NOW + timedelta(seconds=3),
        ),
    )
    connection.execute(
        "INSERT INTO run_jobs "
        "(id, workspace_id, originating_actor_user_id, run_id, status, attempt, max_attempts, "
        "available_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'done', 1, 3, %s, %s, %s)",
        (uuid4(), workspace_id, actor_id, run_id, NOW, NOW, NOW + timedelta(seconds=3)),
    )
    return run_id


def _successful_tool(
    connection: Connection[object],
    *,
    actor_id: UUID,
    workspace_id: UUID,
    run_id: UUID,
    name: str,
    action_id: UUID | None = None,
) -> None:
    connection.execute(
        "INSERT INTO tool_invocations "
        "(id, workspace_id, originating_actor_user_id, run_id, action_intent_id, tool_name, "
        "effect, args_digest, status, attempt, latency_ms, result_summary, started_at, "
        "finished_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'succeeded', 1, 2, %s, %s, %s, %s, %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            action_id,
            name,
            "irreversible" if action_id else "read_only",
            ARGS_DIGEST,
            Jsonb({"schema_version": 1}),
            NOW,
            NOW + timedelta(milliseconds=2),
            NOW,
            NOW + timedelta(milliseconds=2),
        ),
    )


def _qwen_invocation(
    connection: Connection[object],
    *,
    actor_id: UUID,
    workspace_id: UUID,
    run_id: UUID,
    unknown_cost: bool,
    missing_trace: bool,
) -> None:
    priced = not unknown_cost
    connection.execute(
        "INSERT INTO llm_invocations "
        "(id, workspace_id, actor_user_id, run_id, invocation_kind, provider, model, "
        "graph_node, prompt_version, request_hash, provider_response_id, token_usage, "
        "pricing_version, currency, estimated_cost, trace_ids, latency_ms, status, "
        "created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'chat', 'qwen', 'qwen3.6-flash-2026-04-16', "
        "'write_report', %s, %s, 'response-synthetic', %s, %s, %s, %s, %s, 10, "
        "'succeeded', %s, %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            PROMPT_VERSION,
            REQUEST_HASH,
            Jsonb({"input_tokens": 10, "output_tokens": 10}),
            "qwen-cn-beijing-cny-2026-08-13-v1" if priced else None,
            "CNY" if priced else None,
            Decimal("0.20") if priced else None,
            None if missing_trace else Jsonb(TRACE_IDS),
            NOW,
            NOW + timedelta(milliseconds=10),
        ),
    )


def _action(
    connection: Connection[object],
    *,
    actor_id: UUID,
    workspace_id: UUID,
    run_id: UUID,
    binding_mismatch: bool,
    missing_decision: bool,
    outcome_unknown: bool,
    duplicate_mock: bool,
) -> UUID:
    action_id = uuid4()
    status = "outcome_unknown" if outcome_unknown else "succeeded"
    connection.execute(
        "INSERT INTO action_intents "
        "(id, workspace_id, originating_actor_user_id, run_id, action_key, action_revision, "
        "tool_name, effect, args_snapshot, canonicalization_version, args_digest, "
        "target_snapshot, target_canonicalization_version, target_digest, "
        "approval_binding_version, approval_binding_digest, status, idempotency_key, "
        "recovery_attempts, result, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'submit_application', 1, 'submit_mock_application', "
        "'irreversible', %s, 1, %s, %s, 1, %s, 1, %s, %s, %s, 0, %s, %s, %s)",
        (
            action_id,
            workspace_id,
            actor_id,
            run_id,
            Jsonb({"job_ref": "synthetic"}),
            ARGS_DIGEST,
            Jsonb({"provider": "mock_portal"}),
            TARGET_DIGEST,
            BINDING_DIGEST,
            status,
            str(action_id),
            Jsonb({"external_ref": "mock:synthetic"}) if not outcome_unknown else None,
            NOW,
            NOW + timedelta(seconds=2),
        ),
    )
    request_id = uuid4()
    connection.execute(
        "INSERT INTO approval_requests "
        "(id, workspace_id, run_id, action_intent_id, status, args_digest, target_digest, "
        "approval_binding_version, approval_binding_digest, policy_version, policy_snapshot, "
        "version, expires_at, consumed_at, created_at, updated_at) "
        "VALUES (%s, %s, %s, %s, 'consumed', %s, %s, 1, %s, 1, %s, 3, %s, %s, %s, %s)",
        (
            request_id,
            workspace_id,
            run_id,
            action_id,
            f"sha256:{'f' * 64}" if binding_mismatch else ARGS_DIGEST,
            TARGET_DIGEST,
            BINDING_DIGEST,
            Jsonb({"required_approvals": 1}),
            NOW + timedelta(hours=1),
            NOW + timedelta(seconds=1),
            NOW,
            NOW + timedelta(seconds=1),
        ),
    )
    if not missing_decision:
        connection.execute(
            "INSERT INTO approval_decisions "
            "(id, workspace_id, approval_request_id, actor_user_id, decision, decided_at) "
            "VALUES (%s, %s, %s, %s, 'approve', %s)",
            (uuid4(), workspace_id, request_id, actor_id, NOW + timedelta(seconds=1)),
        )
    _successful_tool(
        connection,
        actor_id=actor_id,
        workspace_id=workspace_id,
        run_id=run_id,
        name="submit_mock_application",
        action_id=action_id,
    )
    connection.execute(
        "INSERT INTO mock_submissions "
        "(id, workspace_id, originating_actor_user_id, run_id, action_intent_id, "
        "idempotency_key, payload_digest, payload, external_ref, created_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'mock:synthetic', %s)",
        (
            uuid4(),
            workspace_id,
            actor_id,
            run_id,
            action_id,
            str(action_id),
            f"sha256:{'9' * 64}",
            Jsonb({"action_intent_id": str(action_id)}),
            NOW + timedelta(seconds=2),
        ),
    )
    if duplicate_mock:
        second_id = uuid4()
        connection.execute(
            "INSERT INTO action_intents "
            "(id, workspace_id, originating_actor_user_id, run_id, action_key, action_revision, "
            "tool_name, effect, args_snapshot, canonicalization_version, args_digest, "
            "target_snapshot, target_canonicalization_version, target_digest, "
            "approval_binding_version, approval_binding_digest, status, idempotency_key, "
            "recovery_attempts, result, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, 'submit_application', 2, 'submit_mock_application', "
            "'irreversible', %s, 1, %s, %s, 1, %s, 1, %s, 'succeeded', %s, 0, %s, %s, %s)",
            (
                second_id,
                workspace_id,
                actor_id,
                run_id,
                Jsonb({"job_ref": "duplicate"}),
                ARGS_DIGEST,
                Jsonb({"provider": "mock_portal"}),
                TARGET_DIGEST,
                BINDING_DIGEST,
                str(second_id),
                Jsonb({"external_ref": "mock:duplicate"}),
                NOW,
                NOW,
            ),
        )
        connection.execute(
            "INSERT INTO mock_submissions "
            "(id, workspace_id, originating_actor_user_id, run_id, action_intent_id, "
            "idempotency_key, payload_digest, payload, external_ref, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'mock:duplicate', %s)",
            (
                uuid4(),
                workspace_id,
                actor_id,
                run_id,
                second_id,
                str(second_id),
                f"sha256:{'8' * 64}",
                Jsonb({"action_intent_id": str(second_id)}),
                NOW,
            ),
        )
    return action_id


def _events(
    connection: Connection[object],
    *,
    actor_id: UUID,
    workspace_id: UUID,
    document_id: UUID,
    run_id: UUID,
    missing_rag: bool,
    inconsistent: bool,
) -> None:
    event_types = ["approval.decided", "action.completed", "run.completed"]
    if not missing_rag:
        event_types.insert(0, "rag.retrieved")
    sequences = list(range(1, len(event_types) + 1))
    if inconsistent:
        sequences[-1] += 1
    for sequence, event_type in zip(sequences, event_types, strict=True):
        payload = (
            {"document_ids": [str(document_id)], "chunk_ids": []}
            if event_type == "rag.retrieved"
            else {"synthetic": True}
        )
        connection.execute(
            "INSERT INTO run_events "
            "(id, workspace_id, run_id, actor_user_id, seq, type, version, payload, recorded_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s)",
            (uuid4(), workspace_id, run_id, actor_id, sequence, event_type, Jsonb(payload), NOW),
        )
    next_sequence = max(sequences) + 1
    connection.execute("UPDATE runs SET next_event_seq = %s WHERE id = %s", (next_sequence, run_id))


def _checkpoint(connection: Connection[object], *, run_id: UUID) -> None:
    connection.execute("CREATE SCHEMA pathfinder_checkpoint")
    connection.execute(
        "CREATE TABLE pathfinder_checkpoint.checkpoints "
        "(thread_id text NOT NULL, checkpoint_ns text NOT NULL, checkpoint_id text NOT NULL)"
    )
    connection.execute(
        "INSERT INTO pathfinder_checkpoint.checkpoints VALUES (%s, '', 'checkpoint-synthetic')",
        (str(run_id),),
    )


def _seed(
    database_url: str,
    *,
    variant: str = "complete",
) -> tuple[UUID, UUID, UUID, UUID]:
    with connect_database(database_url) as connection:
        actor_id, workspace_id = _identity(connection)
        document_id = _document(connection, actor_id=actor_id, workspace_id=workspace_id)
        run_id = _run(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
        if variant != "missing_rag":
            _successful_tool(
                connection,
                actor_id=actor_id,
                workspace_id=workspace_id,
                run_id=run_id,
                name="retrieve_documents",
            )
        if variant != "missing_search":
            _successful_tool(
                connection,
                actor_id=actor_id,
                workspace_id=workspace_id,
                run_id=run_id,
                name="search_web",
            )
        _qwen_invocation(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_id=run_id,
            unknown_cost=variant == "unknown_cost",
            missing_trace=variant == "missing_trace",
        )
        _action(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            run_id=run_id,
            binding_mismatch=variant == "binding_mismatch",
            missing_decision=variant == "missing_decision",
            outcome_unknown=variant == "outcome_unknown",
            duplicate_mock=variant == "duplicate_mock",
        )
        _events(
            connection,
            actor_id=actor_id,
            workspace_id=workspace_id,
            document_id=document_id,
            run_id=run_id,
            missing_rag=variant == "missing_rag",
            inconsistent=variant == "event_inconsistent",
        )
        if variant != "missing_checkpoint":
            _checkpoint(connection, run_id=run_id)
        else:
            connection.execute("CREATE SCHEMA pathfinder_checkpoint")
            connection.execute(
                "CREATE TABLE pathfinder_checkpoint.checkpoints "
                "(thread_id text NOT NULL, checkpoint_ns text NOT NULL, "
                "checkpoint_id text NOT NULL)"
            )
        connection.commit()
    return actor_id, workspace_id, document_id, run_id


def _verify(
    database_url: str,
    *,
    actor_id: UUID,
    workspace_id: UUID,
    document_id: UUID,
    run_id: UUID,
    cap: Decimal = Decimal("1.00"),
) -> tuple[dict[str, object], bool]:
    query = _query_text(
        actor_id=actor_id,
        workspace_id=workspace_id,
        document_id=document_id,
        run_id=run_id,
        cap=cap,
    )
    with connect_database(database_url) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        row = connection.execute(query, prepare=False).fetchone()
        connection.rollback()
    assert row is not None and isinstance(row[0], str)
    return json.loads(row[0]), str(row[1]).lower() == "true"


def test_complete_synthetic_live_acceptance_snapshot_passes(
    migrated_database_url: str,
) -> None:
    actor_id, workspace_id, document_id, run_id = _seed(migrated_database_url)

    evidence, accepted = _verify(
        migrated_database_url,
        actor_id=actor_id,
        workspace_id=workspace_id,
        document_id=document_id,
        run_id=run_id,
    )

    assert accepted is True
    assert evidence["accepted"] is True
    assert all(evidence["checks"].values())
    serialized = json.dumps(evidence)
    for forbidden in ("synthetic acceptance text", "synthetic live query", "payload"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "variant",
    [
        "missing_rag",
        "missing_search",
        "missing_checkpoint",
        "missing_decision",
        "binding_mismatch",
        "duplicate_mock",
        "outcome_unknown",
        "unknown_cost",
        "missing_trace",
        "event_inconsistent",
    ],
)
def test_live_acceptance_query_fails_closed_for_missing_or_conflicting_evidence(
    migrated_database_url: str,
    variant: str,
) -> None:
    actor_id, workspace_id, document_id, run_id = _seed(migrated_database_url, variant=variant)

    evidence, accepted = _verify(
        migrated_database_url,
        actor_id=actor_id,
        workspace_id=workspace_id,
        document_id=document_id,
        run_id=run_id,
    )

    assert accepted is False
    assert evidence["accepted"] is False
    assert not all(evidence["checks"].values())


def test_live_acceptance_query_rejects_wrong_workspace_and_cost_over_cap(
    migrated_database_url: str,
) -> None:
    actor_id, workspace_id, document_id, run_id = _seed(migrated_database_url)

    _evidence, wrong_workspace = _verify(
        migrated_database_url,
        actor_id=actor_id,
        workspace_id=uuid4(),
        document_id=document_id,
        run_id=run_id,
    )
    _evidence, over_cap = _verify(
        migrated_database_url,
        actor_id=actor_id,
        workspace_id=workspace_id,
        document_id=document_id,
        run_id=run_id,
        cap=Decimal("0.10"),
    )

    assert wrong_workspace is False
    assert over_cap is False
