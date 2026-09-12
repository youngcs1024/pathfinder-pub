from alembic import context
from pydantic import SecretStr
from sqlalchemy import create_engine, pool

from app.config import Settings
from app.db.models import WorkspaceMembership

config = context.config
target_metadata = WorkspaceMembership.metadata


def _database_url() -> str:
    configured = config.attributes.get("database_url")
    if isinstance(configured, SecretStr):
        return configured.get_secret_value()
    if isinstance(configured, str):
        return configured
    return Settings().database_url.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(
        _database_url(),
        poolclass=pool.NullPool,
    )
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
