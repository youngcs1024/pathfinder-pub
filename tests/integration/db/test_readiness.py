from __future__ import annotations

import pytest
from alembic import command
from pydantic import SecretStr

from app.db.readiness import REQUIRED_DATABASE_REVISION, DatabaseReadinessProbe
from app.db.session import create_database_engine
from tests.integration.support import alembic_config, connect_database

pytestmark = pytest.mark.integration


async def test_probe_tracks_database_migration_revision(database_url: str) -> None:
    engine = create_database_engine(SecretStr(database_url))
    probe = DatabaseReadinessProbe(engine)
    config = alembic_config(database_url)

    try:
        assert await probe.is_ready() is False

        command.upgrade(config, "0014_gate6_action_recovery")
        assert await probe.is_ready() is False

        command.upgrade(config, "head")
        assert await probe.is_ready() is True

        with connect_database(database_url) as connection:
            connection.execute("UPDATE alembic_version SET version_num = 'stale_revision'")
        assert await probe.is_ready() is False

        with connect_database(database_url) as connection:
            connection.execute(
                "UPDATE alembic_version SET version_num = %s",
                (REQUIRED_DATABASE_REVISION,),
            )
        assert await probe.is_ready() is True
    finally:
        await engine.dispose()


async def test_probe_rejects_wrong_or_missing_embedding_schema(database_url: str) -> None:
    engine = create_database_engine(SecretStr(database_url))
    probe = DatabaseReadinessProbe(engine)
    config = alembic_config(database_url)
    try:
        command.upgrade(config, "head")
        assert await probe.is_ready() is True

        with connect_database(database_url) as connection:
            connection.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(3)")
        assert await probe.is_ready() is False

        with connect_database(database_url) as connection:
            connection.execute("ALTER TABLE document_chunks DROP COLUMN embedding")
        assert await probe.is_ready() is False
    finally:
        await engine.dispose()
