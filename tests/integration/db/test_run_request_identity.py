from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic import command
from sqlalchemy.orm import Session

from app.db.models import Run
from tests.integration.db.test_migrations import _seed_gate6_migration_parent
from tests.integration.support import alembic_config, connect_database

pytestmark = pytest.mark.integration

OLD_HEAD = "0014_gate6_action_recovery"
NEW_HEAD = "0019_r21_material_line_ranges"
IDENTITY_COLUMNS = {"client_request_id", "create_request_digest", "create_request_version"}


def _seed(connection: sa.Connection) -> tuple[object, object, object, object]:
    return _seed_gate6_migration_parent(
        connection, subject=f"e3-{uuid4()}", graph_version="pathfinder-research-v6"
    )


def _identity(connection: sa.Connection, run_id: object, **overrides: object) -> None:
    connection.execute(
        sa.text(
            "UPDATE runs SET client_request_id = :key, create_request_digest = :digest, "
            "create_request_version = :version WHERE id = :run_id"
        ),
        {"run_id": run_id, "key": uuid4(), "digest": "a" * 64, "version": 1, **overrides},
    )


def _clone(connection: sa.Connection, run_id: object, actor: object) -> object:
    return connection.scalar(
        sa.text(
            "INSERT INTO runs (workspace_id, created_by_user_id, conversation_id, "
            "request_message_id, input_json, limits_json, graph_version, mode) "
            "SELECT workspace_id, :actor, conversation_id, request_message_id, "
            "input_json, limits_json, graph_version, mode FROM runs WHERE id = :run_id RETURNING id"
        ),
        {"actor": actor, "run_id": run_id},
    )


def test_upgrade_preserves_legacy_run_and_legacy_downgrade(database_url: str) -> None:
    config = alembic_config(database_url)
    command.upgrade(config, OLD_HEAD)
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            _, _, run_id, _ = _seed(connection)
            before = connection.execute(sa.text("SELECT * FROM runs")).mappings().one()
        command.upgrade(config, "head")
        with Session(engine) as session:
            run = session.get(Run, run_id)
            assert run is not None
            assert all(getattr(run, field) is None for field in IDENTITY_COLUMNS)
            assert run.status == "running"
        columns = sa.inspect(engine).get_columns("runs")
        assert all(
            column["nullable"] and column["default"] is None
            for column in columns
            if column["name"] in IDENTITY_COLUMNS
        )
        command.downgrade(config, OLD_HEAD)
        assert IDENTITY_COLUMNS.isdisjoint(
            c["name"] for c in sa.inspect(engine).get_columns("runs")
        )
        with engine.connect() as connection:
            assert connection.execute(sa.text("SELECT * FROM runs")).mappings().one() == before
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == NEW_HEAD
    finally:
        engine.dispose()


def test_request_key_scope_and_null_distinct(migrated_database_url: str) -> None:
    engine = sa.create_engine(migrated_database_url)
    key = uuid4()
    try:
        with engine.begin() as connection:
            actor, workspace, run_id, _ = _seed(connection)
            second_actor, _, other_run, _ = _seed(connection)
            # Same actor in another workspace can reuse the key.
            other_workspace = connection.scalar(
                sa.text("SELECT workspace_id FROM runs WHERE id = :id"), {"id": other_run}
            )
            for workspace_id, user_id in ((workspace, second_actor), (other_workspace, actor)):
                connection.execute(
                    sa.text(
                        "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
                        "VALUES (:workspace, :actor, 'member')"
                    ),
                    {"workspace": workspace_id, "actor": user_id},
                )
            legacy_one = _clone(connection, run_id, actor)
            legacy_two = _clone(connection, run_id, actor)
            assert legacy_one != legacy_two
            other_actor_run = _clone(connection, run_id, second_actor)
            other_workspace_run = _clone(connection, other_run, actor)
            for scoped_run in (run_id, other_actor_run, other_workspace_run):
                _identity(connection, scoped_run, key=key)
            with pytest.raises(sa.exc.IntegrityError) as error, connection.begin_nested():
                _identity(connection, legacy_one, key=key)
            assert error.value.orig.diag.constraint_name == "uq_runs_workspace_creator_request"
            assert connection.scalar(sa.text("SELECT count(*) FROM runs")) == 6
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    ("key_present", "digest_present", "version_present"),
    [bits for bits in product((False, True), repeat=3) if any(bits) and not all(bits)],
)
def test_partial_identity_is_rejected(
    migrated_database_url: str,
    key_present: bool,
    digest_present: bool,
    version_present: bool,
) -> None:
    _assert_invalid_identity(
        migrated_database_url,
        key=uuid4() if key_present else None,
        digest="a" * 64 if digest_present else None,
        version=1 if version_present else None,
    )


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"])
def test_invalid_digest_is_rejected(migrated_database_url: str, digest: str) -> None:
    _assert_invalid_identity(migrated_database_url, digest=digest)


@pytest.mark.parametrize("version", [-1, 0, 2])
def test_unknown_version_is_rejected(migrated_database_url: str, version: int) -> None:
    _assert_invalid_identity(migrated_database_url, version=version)


def _assert_invalid_identity(database_url: str, **overrides: object) -> None:
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            _, _, run_id, _ = _seed(connection)
            with pytest.raises(sa.exc.IntegrityError) as error, connection.begin_nested():
                _identity(connection, run_id, **overrides)
            assert error.value.orig.diag.constraint_name == "ck_runs_create_request_identity"
    finally:
        engine.dispose()


@pytest.mark.parametrize("target", [OLD_HEAD, "base"])
def test_keyed_downgrade_preserves_identity(migrated_database_url: str, target: str) -> None:
    engine = sa.create_engine(migrated_database_url)
    key = uuid4()
    try:
        with engine.begin() as connection:
            _, _, run_id, _ = _seed(connection)
            _identity(connection, run_id, key=key)
        with pytest.raises(
            RuntimeError, match="E3 request identity data cannot be safely downgraded"
        ):
            command.downgrade(alembic_config(migrated_database_url), target)
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == NEW_HEAD
            assert connection.execute(
                sa.text(
                    "SELECT client_request_id, create_request_digest, create_request_version "
                    "FROM runs WHERE id = :id"
                ),
                {"id": run_id},
            ).one() == (key, "a" * 64, 1)
    finally:
        engine.dispose()


def _schema_snapshot(database_url: str) -> dict[str, object]:
    source = Path(__file__).resolve().parents[3] / "scripts" / "gate85_consistency.sql"
    query = "\n".join(line for line in source.read_text().splitlines() if not line.startswith("\\"))
    for variable in ("fixture_run_id", "fixture_document_id"):
        query = query.replace(f":'{variable}'", f"'{uuid4()}'")
    with connect_database(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(query, prepare=False)
        row = None
        while True:
            if cursor.description is not None:
                row = cursor.fetchone()
            if not cursor.nextset():
                break
        assert row is not None
        return json.loads(row[0])


@pytest.mark.parametrize(
    "damage",
    [
        "ALTER TABLE runs DROP CONSTRAINT uq_runs_workspace_creator_request",
        "ALTER TABLE runs DROP CONSTRAINT ck_runs_create_request_identity",
        "ALTER TABLE runs ALTER COLUMN create_request_version SET DEFAULT 1",
        "ALTER TABLE runs DROP COLUMN create_request_digest",
    ],
)
def test_operational_probe_checks_request_schema(migrated_database_url: str, damage: str) -> None:
    # Only the checkpoint catalog/identifier shape is needed by this schema probe test.
    with connect_database(migrated_database_url) as connection:
        connection.execute("CREATE SCHEMA pathfinder_checkpoint")
        connection.execute("CREATE TABLE pathfinder_checkpoint.checkpoints (thread_id text)")
        for table in ("checkpoint_blobs", "checkpoint_migrations", "checkpoint_writes"):
            connection.execute(f"CREATE TABLE pathfinder_checkpoint.{table} (id integer)")
    assert _schema_snapshot(migrated_database_url)["checks"]["run_request_identity_schema"] is True
    with connect_database(migrated_database_url) as connection:
        connection.execute(damage)
    assert _schema_snapshot(migrated_database_url)["checks"]["run_request_identity_schema"] is False
