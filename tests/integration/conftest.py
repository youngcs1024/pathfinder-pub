from collections.abc import Iterator
from uuid import uuid4

import pytest
from alembic import command
from psycopg import sql
from testcontainers.community.postgres import PostgresContainer

from tests.integration.support import (
    alembic_config,
    connect_database,
    database_url_with_name,
)

POSTGRES_IMAGE = "pgvector/pgvector:0.8.5-pg16"


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    with PostgresContainer(
        image=POSTGRES_IMAGE,
        username="pathfinder_test",
        password="pathfinder_test",
        dbname="postgres",
        driver="psycopg",
    ) as container:
        yield container.get_connection_url()


@pytest.fixture
def database_url(postgres_url: str) -> Iterator[str]:
    database_name = f"pathfinder_test_{uuid4().hex}"
    with connect_database(postgres_url, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))

    isolated_url = database_url_with_name(postgres_url, database_name)
    try:
        yield isolated_url
    finally:
        with connect_database(postgres_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database_name))
            )


@pytest.fixture
def migrated_database_url(database_url: str) -> str:
    command.upgrade(alembic_config(database_url), "head")
    return database_url
