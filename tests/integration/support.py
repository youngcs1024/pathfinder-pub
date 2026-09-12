from pathlib import Path
from typing import Any

import psycopg
from alembic.config import Config
from sqlalchemy import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def connect_database(
    database_url: str,
    *,
    autocommit: bool = False,
) -> psycopg.Connection[Any]:
    url = make_url(database_url)
    return psycopg.connect(
        host=url.host,
        port=url.port,
        user=url.username,
        password=url.password,
        dbname=url.database,
        autocommit=autocommit,
    )


def database_url_with_name(database_url: str, database_name: str) -> str:
    url: URL = make_url(database_url)
    return url.set(database=database_name).render_as_string(hide_password=False)


def alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    return config
