from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import psycopg
import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.support import alembic_config, connect_database

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _RunFacts:
    user_id: UUID
    workspace_id: UUID
    conversation_id: UUID
    message_id: UUID
    run_id: UUID
    job_id: UUID
    event_id: UUID
    tool_invocation_id: UUID


def _insert_run_graph(
    connection: psycopg.Connection,
    *,
    suffix: str,
) -> _RunFacts:
    user_id = uuid4()
    workspace_id = uuid4()
    conversation_id = uuid4()
    message_id = uuid4()
    run_id = uuid4()
    job_id = uuid4()
    event_id = uuid4()
    tool_invocation_id = uuid4()
    connection.execute(
        "INSERT INTO users (id, auth_subject) VALUES (%s, %s)",
        (user_id, f"gate4-user-{suffix}"),
    )
    connection.execute(
        """
        INSERT INTO workspaces (id, kind, name, created_by_user_id)
        VALUES (%s, 'team', %s, %s)
        """,
        (workspace_id, f"Gate 4 Workspace {suffix}", user_id),
    )
    connection.execute(
        """
        INSERT INTO workspace_memberships (workspace_id, user_id, role)
        VALUES (%s, %s, 'admin')
        """,
        (workspace_id, user_id),
    )
    connection.execute(
        """
        INSERT INTO conversations (id, workspace_id, created_by_user_id, title)
        VALUES (%s, %s, %s, %s)
        """,
        (conversation_id, workspace_id, user_id, f"Research {suffix}"),
    )
    connection.execute(
        """
        INSERT INTO messages (
            id, workspace_id, conversation_id, actor_user_id, role, content
        ) VALUES (%s, %s, %s, %s, 'user', 'Research this role')
        """,
        (message_id, workspace_id, conversation_id, user_id),
    )
    connection.execute(
        """
        INSERT INTO runs (
            id, workspace_id, created_by_user_id, conversation_id,
            request_message_id, input_json, limits_json, mode, graph_version
        ) VALUES (%s, %s, %s, %s, %s, '{}'::jsonb, '{}'::jsonb,
            'research', 'pathfinder-research-v6')
        """,
        (run_id, workspace_id, user_id, conversation_id, message_id),
    )
    connection.execute(
        """
        INSERT INTO run_jobs (id, workspace_id, originating_actor_user_id, run_id)
        VALUES (%s, %s, %s, %s)
        """,
        (job_id, workspace_id, user_id, run_id),
    )
    allocated_seq = connection.execute(
        """
        UPDATE runs
        SET next_event_seq = next_event_seq + 1
        WHERE id = %s
        RETURNING next_event_seq - 1
        """,
        (run_id,),
    ).fetchone()
    assert allocated_seq == (1,)
    connection.execute(
        """
        INSERT INTO run_events (
            id, workspace_id, run_id, actor_user_id, seq, type, payload
        ) VALUES (%s, %s, %s, %s, 1, 'run.created', '{}'::jsonb)
        """,
        (event_id, workspace_id, run_id, user_id),
    )
    connection.execute(
        """
        INSERT INTO tool_invocations (
            id, workspace_id, originating_actor_user_id, run_id, tool_name,
            effect, args_digest, status
        ) VALUES (
            %s, %s, %s, %s, 'web_search', 'read_only', %s, 'prepared'
        )
        """,
        (
            tool_invocation_id,
            workspace_id,
            user_id,
            run_id,
            f"sha256:{'a' * 64}",
        ),
    )
    return _RunFacts(
        user_id=user_id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        run_id=run_id,
        job_id=job_id,
        event_id=event_id,
        tool_invocation_id=tool_invocation_id,
    )


def test_valid_persistent_run_graph_closes_all_composite_keys(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="valid")
        row = connection.execute(
            """
            SELECT r.mode, r.status, r.graph_version, r.next_event_seq,
                   j.status, j.attempt, j.max_attempts, e.version,
                   t.effect, t.status
            FROM runs AS r
            JOIN run_jobs AS j ON j.run_id = r.id
            JOIN run_events AS e ON e.run_id = r.id AND e.seq = 1
            JOIN tool_invocations AS t ON t.run_id = r.id
            WHERE r.id = %s
            """,
            (facts.run_id,),
        ).fetchone()

    assert row == (
        "research",
        "queued",
        "pathfinder-research-v6",
        2,
        "queued",
        0,
        3,
        1,
        "read_only",
        "prepared",
    )


@pytest.mark.parametrize(
    "statement",
    [
        """
        INSERT INTO messages (
            workspace_id, conversation_id, actor_user_id, role, content
        ) VALUES (%s, %s, NULL, 'user', 'missing actor')
        """,
        """
        INSERT INTO messages (
            workspace_id, conversation_id, actor_user_id, role, content
        ) VALUES (%s, %s, %s, 'assistant', 'unexpected actor')
        """,
        """
        INSERT INTO messages (
            workspace_id, conversation_id, actor_user_id, role, content
        ) VALUES (%s, %s, %s, 'system', 'invalid role')
        """,
    ],
)
def test_message_role_actor_matrix_is_enforced(
    migrated_database_url: str,
    statement: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix=uuid4().hex)
        connection.commit()

        parameters = (facts.workspace_id, facts.conversation_id, facts.user_id)
        if "NULL, 'user'" in statement:
            parameters = parameters[:2]
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(statement, parameters)
        connection.rollback()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE runs SET status = 'pending' WHERE id = %s",
        "UPDATE runs SET graph_version = 'unknown-v9' WHERE id = %s",
        "UPDATE runs SET input_json = '[]'::jsonb WHERE id = %s",
        "UPDATE runs SET limits_json = 'null'::jsonb WHERE id = %s",
        "UPDATE runs SET next_event_seq = 0 WHERE id = %s",
        "UPDATE runs SET status = 'completed', finished_at = now() WHERE id = %s",
        "UPDATE runs SET status = 'failed', started_at = now(), finished_at = now() WHERE id = %s",
    ],
)
def test_run_value_and_lifecycle_constraints_reject_invalid_rows(
    migrated_database_url: str,
    statement: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix=uuid4().hex)
        connection.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(statement, (facts.run_id,))
        connection.rollback()


def test_request_message_must_belong_to_the_run_conversation(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="conversation-binding")
        other_conversation_id = uuid4()
        other_message_id = uuid4()
        connection.execute(
            """
            INSERT INTO conversations (id, workspace_id, created_by_user_id, title)
            VALUES (%s, %s, %s, 'Other conversation')
            """,
            (other_conversation_id, facts.workspace_id, facts.user_id),
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, workspace_id, conversation_id, actor_user_id, role, content
            ) VALUES (%s, %s, %s, %s, 'user', 'Other request')
            """,
            (
                other_message_id,
                facts.workspace_id,
                other_conversation_id,
                facts.user_id,
            ),
        )
        connection.commit()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                "UPDATE runs SET request_message_id = %s WHERE id = %s",
                (other_message_id, facts.run_id),
            )
        connection.rollback()


@pytest.mark.parametrize(
    "statement",
    [
        """
        INSERT INTO messages (
            workspace_id, conversation_id, actor_user_id, role, content
        ) VALUES (%s, %s, %s, 'user', 'cross workspace')
        """,
        """
        UPDATE run_jobs
        SET workspace_id = %s, originating_actor_user_id = %s
        WHERE id = %s
        """,
        """
        INSERT INTO run_events (
            workspace_id, run_id, actor_user_id, seq, type, payload
        ) VALUES (%s, %s, %s, 2, 'run.status_changed', '{}'::jsonb)
        """,
        """
        INSERT INTO tool_invocations (
            workspace_id, originating_actor_user_id, run_id, tool_name,
            effect, args_digest, status
        ) VALUES (%s, %s, %s, 'web_search', 'read_only', %s, 'prepared')
        """,
        """
        INSERT INTO llm_invocations (
            workspace_id, actor_user_id, run_id, invocation_kind, provider, model,
            graph_node, prompt_version, request_hash, status
        ) VALUES (
            %s, %s, %s, 'chat', 'fake', 'qwen3.6-flash-2026-04-16',
            'plan', %s, %s, 'started'
        )
        """,
    ],
)
def test_cross_workspace_parent_and_actor_combinations_are_rejected(
    migrated_database_url: str,
    statement: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        first = _insert_run_graph(connection, suffix=f"first-{uuid4().hex}")
        second = _insert_run_graph(connection, suffix=f"second-{uuid4().hex}")
        connection.commit()

        parameters: tuple[object, ...] = (
            first.workspace_id,
            first.user_id,
            second.run_id,
        )
        if "INSERT INTO messages" in statement:
            parameters = (
                first.workspace_id,
                second.conversation_id,
                first.user_id,
            )
        elif "UPDATE run_jobs" in statement:
            parameters = (first.workspace_id, first.user_id, second.job_id)
        elif "tool_invocations" in statement:
            parameters = (*parameters, f"sha256:{'b' * 64}")
        elif "llm_invocations" in statement:
            parameters = (
                *parameters,
                f"sha256:{'c' * 64}",
                f"sha256:{'d' * 64}",
            )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(statement, parameters)
        connection.rollback()


def test_missing_actor_membership_and_historical_revocation_semantics(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="membership")
        connection.commit()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                """
                INSERT INTO conversations (workspace_id, created_by_user_id, title)
                VALUES (%s, %s, 'Orphan creator')
                """,
                (facts.workspace_id, uuid4()),
            )
        connection.rollback()

        connection.execute(
            """
            UPDATE workspace_memberships SET revoked_at = now()
            WHERE workspace_id = %s AND user_id = %s
            """,
            (facts.workspace_id, facts.user_id),
        )
        connection.commit()
        assert connection.execute(
            "SELECT count(*) FROM runs WHERE id = %s",
            (facts.run_id,),
        ).fetchone() == (1,)


def test_job_uniqueness_attempt_and_lease_constraints(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="job-constraints")
        connection.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                """
                INSERT INTO run_jobs (
                    workspace_id, originating_actor_user_id, run_id
                ) VALUES (%s, %s, %s)
                """,
                (facts.workspace_id, facts.user_id, facts.run_id),
            )
        connection.rollback()

        invalid_updates = (
            "UPDATE run_jobs SET attempt = -1 WHERE id = %s",
            "UPDATE run_jobs SET attempt = 4, max_attempts = 3 WHERE id = %s",
            "UPDATE run_jobs SET status = 'leased' WHERE id = %s",
            "UPDATE run_jobs SET leased_by = 'worker-1' WHERE id = %s",
            "UPDATE run_jobs SET status = 'dead' WHERE id = %s",
        )
        for statement in invalid_updates:
            with pytest.raises(psycopg.errors.CheckViolation):
                connection.execute(statement, (facts.job_id,))
            connection.rollback()


def test_event_and_tool_gate4_constraints(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="event-tool")
        connection.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                """
                INSERT INTO run_events (
                    workspace_id, run_id, seq, type, payload
                ) VALUES (%s, %s, 1, 'run.created', '{}'::jsonb)
                """,
                (facts.workspace_id, facts.run_id),
            )
        connection.rollback()

        invalid_updates = (
            (
                "UPDATE run_events SET type = 'checkpoint.saved' WHERE id = %s",
                facts.event_id,
            ),
            ("UPDATE run_events SET version = 2 WHERE id = %s", facts.event_id),
            (
                "UPDATE run_events SET payload = '[]'::jsonb WHERE id = %s",
                facts.event_id,
            ),
            (
                "UPDATE tool_invocations SET effect = 'reversible' WHERE id = %s",
                facts.tool_invocation_id,
            ),
            (
                "UPDATE tool_invocations SET effect = 'irreversible' WHERE id = %s",
                facts.tool_invocation_id,
            ),
            (
                "UPDATE tool_invocations SET status = 'outcome_unknown' WHERE id = %s",
                facts.tool_invocation_id,
            ),
            (
                "UPDATE tool_invocations SET tool_name = 'WebSearch' WHERE id = %s",
                facts.tool_invocation_id,
            ),
            (
                "UPDATE tool_invocations SET args_digest = 'not-a-digest' WHERE id = %s",
                facts.tool_invocation_id,
            ),
            (
                "UPDATE tool_invocations SET status = 'succeeded' WHERE id = %s",
                facts.tool_invocation_id,
            ),
        )
        for statement, row_id in invalid_updates:
            with pytest.raises(psycopg.errors.CheckViolation):
                connection.execute(statement, (row_id,))
            connection.rollback()


def test_duplicate_event_failure_rolls_back_status_and_next_sequence(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        facts = _insert_run_graph(connection, suffix="event-rollback")
        connection.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                """
                UPDATE runs
                SET status = 'running', started_at = now(), next_event_seq = next_event_seq + 1
                WHERE id = %s
                """,
                (facts.run_id,),
            )
            connection.execute(
                """
                INSERT INTO run_events (
                    workspace_id, run_id, actor_user_id, seq, type, payload
                ) VALUES (%s, %s, %s, 1, 'run.status_changed', '{}'::jsonb)
                """,
                (facts.workspace_id, facts.run_id, facts.user_id),
            )
        connection.rollback()

        run_row = connection.execute(
            "SELECT status, started_at, next_event_seq FROM runs WHERE id = %s",
            (facts.run_id,),
        ).fetchone()
        event_count = connection.execute(
            "SELECT count(*) FROM run_events WHERE run_id = %s",
            (facts.run_id,),
        ).fetchone()

    assert run_row == ("queued", None, 2)
    assert event_count == (1,)


def test_historical_unlinked_invocation_survives_upgrade_and_safe_downgrade(
    database_url: str,
) -> None:
    config = alembic_config(database_url)
    engine = sa.create_engine(database_url)
    try:
        command.upgrade(config, "0006_qwen_provider_profile")
        with engine.begin() as connection:
            user_id = connection.scalar(
                sa.text(
                    "INSERT INTO users (auth_subject) "
                    "VALUES ('gate4-historical-invocation') RETURNING id"
                )
            )
            workspace_id = connection.scalar(
                sa.text(
                    "INSERT INTO workspaces (kind, name, created_by_user_id) "
                    "VALUES ('team', 'Historical Gate 3', :user_id) RETURNING id"
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
            invocation_id = connection.scalar(
                sa.text(
                    """
                    INSERT INTO llm_invocations (
                        workspace_id, actor_user_id, invocation_kind, provider, model,
                        graph_node, prompt_version, request_hash, status
                    ) VALUES (
                        :workspace_id, :user_id, 'chat', 'fake',
                        'qwen3.6-flash-2026-04-16', 'plan', :prompt_version,
                        :request_hash, 'started'
                    ) RETURNING id
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "prompt_version": f"sha256:{'c' * 64}",
                    "request_hash": f"sha256:{'d' * 64}",
                },
            )

        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert (
                connection.scalar(
                    sa.text("SELECT run_id FROM llm_invocations WHERE id = :id"),
                    {"id": invocation_id},
                )
                is None
            )

        command.downgrade(config, "0006_qwen_provider_profile")
        with engine.connect() as connection:
            assert (
                connection.scalar(
                    sa.text("SELECT count(*) FROM llm_invocations WHERE id = :id"),
                    {"id": invocation_id},
                )
                == 1
            )
        command.upgrade(config, "head")
    finally:
        engine.dispose()


def test_downgrade_fails_closed_when_gate4_business_facts_exist(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        _insert_run_graph(connection, suffix="downgrade-protection")
        connection.commit()

    with pytest.raises(
        RuntimeError,
        match=r"Gate 6\.4 execution data cannot be safely downgraded",
    ):
        command.downgrade(
            alembic_config(migrated_database_url),
            "0006_qwen_provider_profile",
        )

    with connect_database(migrated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM runs").fetchone() == (1,)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "0021_r31_resume_profiles",
        )
