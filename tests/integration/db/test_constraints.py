from decimal import Decimal
from uuid import UUID, uuid4

import psycopg
import pytest

from tests.integration.support import connect_database

pytestmark = pytest.mark.integration


def _insert_user(connection: psycopg.Connection, auth_subject: str) -> UUID:
    user_id = uuid4()
    connection.execute(
        "INSERT INTO users (id, auth_subject) VALUES (%s, %s)",
        (user_id, auth_subject),
    )
    return user_id


def _insert_workspace(
    connection: psycopg.Connection,
    *,
    creator_id: UUID,
    kind: str,
    name: str,
) -> UUID:
    workspace_id = uuid4()
    connection.execute(
        """
        INSERT INTO workspaces (id, kind, name, created_by_user_id)
        VALUES (%s, %s, %s, %s)
        """,
        (workspace_id, kind, name, creator_id),
    )
    return workspace_id


def test_locked_workspace_kinds_and_membership_roles(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        users = [_insert_user(connection, f"subject-{index}") for index in range(3)]
        personal_workspace = _insert_workspace(
            connection,
            creator_id=users[0],
            kind="personal",
            name="Personal Workspace",
        )
        team_workspace = _insert_workspace(
            connection,
            creator_id=users[0],
            kind="team",
            name="Team Fixture",
        )
        for user_id, role in zip(users, ("member", "reviewer", "admin"), strict=True):
            connection.execute(
                """
                INSERT INTO workspace_memberships (workspace_id, user_id, role)
                VALUES (%s, %s, %s)
                """,
                (team_workspace, user_id, role),
            )

        kinds = {row[0] for row in connection.execute("SELECT kind FROM workspaces").fetchall()}
        roles = {
            row[0]
            for row in connection.execute("SELECT role FROM workspace_memberships").fetchall()
        }

    assert personal_workspace != team_workspace
    assert kinds == {"personal", "team"}
    assert roles == {"member", "reviewer", "admin"}


@pytest.mark.parametrize(
    ("table", "value"),
    [
        ("workspaces", "organization"),
        ("workspace_memberships", "owner"),
    ],
)
def test_invalid_locked_values_are_rejected(
    migrated_database_url: str,
    table: str,
    value: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "invalid-value-user")
        workspace_id = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="Team Fixture",
        )
        connection.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            if table == "workspaces":
                _insert_workspace(
                    connection,
                    creator_id=user_id,
                    kind=value,
                    name="Invalid",
                )
            else:
                connection.execute(
                    """
                    INSERT INTO workspace_memberships (workspace_id, user_id, role)
                    VALUES (%s, %s, %s)
                    """,
                    (workspace_id, user_id, value),
                )
        connection.rollback()


def test_unique_identity_membership_and_personal_workspace_constraints(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "unique-subject")
        personal_workspace = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="personal",
            name="First Personal",
        )
        connection.execute(
            """
            INSERT INTO workspace_memberships (workspace_id, user_id, role)
            VALUES (%s, %s, 'admin')
            """,
            (personal_workspace, user_id),
        )
        connection.commit()

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_user(connection, "unique-subject")
        connection.rollback()

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert_workspace(
                connection,
                creator_id=user_id,
                kind="personal",
                name="Second Personal",
            )
        connection.rollback()

        with pytest.raises(psycopg.errors.UniqueViolation):
            connection.execute(
                """
                INSERT INTO workspace_memberships (workspace_id, user_id, role)
                VALUES (%s, %s, 'admin')
                """,
                (personal_workspace, user_id),
            )
        connection.rollback()

        first_team = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="First Team",
        )
        second_team = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="Second Team",
        )

    assert first_team != second_team


def test_orphan_creator_and_membership_foreign_keys_are_rejected(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "foreign-key-user")
        workspace_id = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="Team Fixture",
        )
        connection.commit()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            _insert_workspace(
                connection,
                creator_id=uuid4(),
                kind="team",
                name="Orphan",
            )
        connection.rollback()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                """
                INSERT INTO workspace_memberships (workspace_id, user_id, role)
                VALUES (%s, %s, 'member')
                """,
                (uuid4(), user_id),
            )
        connection.rollback()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                """
                INSERT INTO workspace_memberships (workspace_id, user_id, role)
                VALUES (%s, %s, 'member')
                """,
                (workspace_id, uuid4()),
            )
        connection.rollback()


def _insert_started_chat_invocation(
    connection: psycopg.Connection,
    *,
    workspace_id: UUID,
    actor_user_id: UUID,
    provider: str = "fake",
) -> UUID:
    invocation_id = uuid4()
    model = "gpt-5.6-terra" if provider == "openai" else "qwen3.6-flash-2026-04-16"
    connection.execute(
        """
        INSERT INTO llm_invocations (
            id, workspace_id, actor_user_id, invocation_kind, provider, model,
            graph_node, prompt_version, request_hash, status
        ) VALUES (
            %s, %s, %s, 'chat', %s, %s, 'plan', %s, %s, 'started'
        )
        """,
        (
            invocation_id,
            workspace_id,
            actor_user_id,
            provider,
            model,
            f"sha256:{'a' * 64}",
            f"sha256:{'b' * 64}",
        ),
    )
    return invocation_id


def test_llm_invocation_actor_must_belong_to_the_same_workspace(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        first_user = _insert_user(connection, "invocation-first-user")
        second_user = _insert_user(connection, "invocation-second-user")
        first_workspace = _insert_workspace(
            connection,
            creator_id=first_user,
            kind="team",
            name="First Team Fixture",
        )
        second_workspace = _insert_workspace(
            connection,
            creator_id=second_user,
            kind="team",
            name="Second Team Fixture",
        )
        connection.execute(
            """
            INSERT INTO workspace_memberships (workspace_id, user_id, role)
            VALUES (%s, %s, 'admin'), (%s, %s, 'admin')
            """,
            (first_workspace, first_user, second_workspace, second_user),
        )
        connection.commit()

        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            _insert_started_chat_invocation(
                connection,
                workspace_id=first_workspace,
                actor_user_id=second_user,
            )
        connection.rollback()


def test_llm_invocation_profile_and_terminal_state_are_database_enforced(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "invocation-state-user")
        workspace_id = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="State Team Fixture",
        )
        connection.execute(
            """
            INSERT INTO workspace_memberships (workspace_id, user_id, role)
            VALUES (%s, %s, 'admin')
            """,
            (workspace_id, user_id),
        )
        invocation_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
        )
        connection.commit()

        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                "UPDATE llm_invocations SET status = 'succeeded' WHERE id = %s",
                (invocation_id,),
            )
        connection.rollback()

        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                """
                INSERT INTO llm_invocations (
                    id, workspace_id, actor_user_id, invocation_kind, provider, model,
                    graph_node, prompt_version, request_hash, status
                ) VALUES (
                    %s, %s, %s, 'embedding', 'fake', 'request-overridden-model',
                    'ingest_documents', NULL, %s, 'started'
                )
                """,
                (uuid4(), workspace_id, user_id, f"sha256:{'c' * 64}"),
            )
        connection.rollback()


def test_llm_invocation_provider_response_id_and_new_error_categories_are_enforced(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "invocation-provider-metadata-user")
        workspace_id = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="Provider Metadata Team Fixture",
        )
        connection.execute(
            """
            INSERT INTO workspace_memberships (workspace_id, user_id, role)
            VALUES (%s, %s, 'admin')
            """,
            (workspace_id, user_id),
        )
        invocation_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
        )
        connection.execute(
            """
            UPDATE llm_invocations
            SET status = 'failed', provider_response_id = 'req_rate_limited_123',
                latency_ms = 5, error_category = 'rate_limited'
            WHERE id = %s
            """,
            (invocation_id,),
        )
        connection.commit()

        for invalid_response_id in (" ", " padded ", "x" * 513):
            next_id = _insert_started_chat_invocation(
                connection,
                workspace_id=workspace_id,
                actor_user_id=user_id,
            )
            with pytest.raises(psycopg.errors.CheckViolation):
                connection.execute(
                    """
                    UPDATE llm_invocations
                    SET status = 'failed', provider_response_id = %s,
                        latency_ms = 5, error_category = 'provider_rejected'
                    WHERE id = %s
                    """,
                    (invalid_response_id, next_id),
                )
            connection.rollback()


def test_llm_invocation_cost_fields_are_atomic_precise_and_provider_currency_bound(
    migrated_database_url: str,
) -> None:
    with connect_database(migrated_database_url) as connection:
        user_id = _insert_user(connection, "invocation-cost-user")
        workspace_id = _insert_workspace(
            connection,
            creator_id=user_id,
            kind="team",
            name="Cost Constraint Team Fixture",
        )
        connection.execute(
            """
            INSERT INTO workspace_memberships (workspace_id, user_id, role)
            VALUES (%s, %s, 'admin')
            """,
            (workspace_id, user_id),
        )
        valid_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
            provider="qwen",
        )
        connection.execute(
            """
            UPDATE llm_invocations
            SET status = 'succeeded',
                token_usage = '{"input_tokens": 11, "output_tokens": 13}'::jsonb,
                latency_ms = 7,
                pricing_version = 'qwen-cn-beijing-cny-2026-08-13-v1',
                currency = 'CNY',
                estimated_cost = %s
            WHERE id = %s
            """,
            (Decimal("0.000106800000"), valid_id),
        )
        connection.commit()

        stored_cost = connection.execute(
            "SELECT estimated_cost FROM llm_invocations WHERE id = %s",
            (valid_id,),
        ).fetchone()
        assert stored_cost == (Decimal("0.000106800000"),)

        legacy_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
            provider="openai",
        )
        connection.execute(
            """
            UPDATE llm_invocations
            SET status = 'succeeded',
                token_usage = '{"input_tokens": 1, "output_tokens": 1}'::jsonb,
                latency_ms = 1,
                pricing_version = 'openai-standard-usd-2026-08-13-v1',
                currency = 'USD',
                estimated_cost = 0.000014
            WHERE id = %s
            """,
            (legacy_id,),
        )
        connection.commit()

        partial_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
            provider="qwen",
        )
        connection.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                """
                UPDATE llm_invocations
                SET status = 'succeeded',
                    token_usage = '{"input_tokens": 1, "output_tokens": 1}'::jsonb,
                    latency_ms = 1,
                    pricing_version = 'qwen-cn-beijing-cny-2026-08-13-v1'
                WHERE id = %s
                """,
                (partial_id,),
            )
        connection.rollback()

        negative_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
            provider="qwen",
        )
        connection.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                """
                UPDATE llm_invocations
                SET status = 'succeeded',
                    token_usage = '{"input_tokens": 1, "output_tokens": 1}'::jsonb,
                    latency_ms = 1,
                    pricing_version = 'qwen-cn-beijing-cny-2026-08-13-v1',
                    currency = 'CNY',
                    estimated_cost = -0.000001
                WHERE id = %s
                """,
                (negative_id,),
            )
        connection.rollback()

        wrong_currency_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
            provider="qwen",
        )
        connection.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                """
                UPDATE llm_invocations
                SET status = 'succeeded',
                    token_usage = '{"input_tokens": 1, "output_tokens": 1}'::jsonb,
                    latency_ms = 1,
                    pricing_version = 'qwen-cn-beijing-cny-2026-08-13-v1',
                    currency = 'USD',
                    estimated_cost = 0.000014
                WHERE id = %s
                """,
                (wrong_currency_id,),
            )
        connection.rollback()

        fake_id = _insert_started_chat_invocation(
            connection,
            workspace_id=workspace_id,
            actor_user_id=user_id,
        )
        connection.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            connection.execute(
                """
                UPDATE llm_invocations
                SET status = 'succeeded',
                    token_usage = '{"input_tokens": 1, "output_tokens": 1}'::jsonb,
                    latency_ms = 1,
                    pricing_version = 'qwen-cn-beijing-cny-2026-08-13-v1',
                    currency = 'CNY',
                    estimated_cost = 0.000014
                WHERE id = %s
                """,
                (fake_id,),
            )
        connection.rollback()
