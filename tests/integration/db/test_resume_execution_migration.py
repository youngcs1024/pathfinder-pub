import pytest
import sqlalchemy as sa
from alembic import command

from tests.integration.db.test_migrations import _seed_gate6_migration_parent
from tests.integration.support import alembic_config

pytestmark = pytest.mark.integration
OLD_HEAD = "0015_e3_run_request_identity"
NEW_HEAD = "0016_r1_execution_contracts"


def test_upgrade_preserves_history_removes_defaults_and_can_reverse_when_empty_of_new_data(
    database_url,
):
    config = alembic_config(database_url)
    command.upgrade(config, OLD_HEAD)
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            _seed_gate6_migration_parent(
                connection, subject="r11-migration", graph_version="pathfinder-research-v6"
            )
            before = connection.execute(sa.text("SELECT * FROM runs")).mappings().one()
        command.upgrade(config, NEW_HEAD)
        with engine.connect() as connection:
            assert connection.execute(sa.text("SELECT * FROM runs")).mappings().one() == before
        columns = {c["name"]: c for c in sa.inspect(engine).get_columns("runs")}
        assert columns["mode"]["default"] is None and columns["graph_version"]["default"] is None
        with pytest.raises(sa.exc.IntegrityError), engine.begin() as connection:
            connection.execute(sa.text("UPDATE runs SET mode = 'material_preparation'"))
        command.downgrade(config, OLD_HEAD)
        with engine.connect() as connection:
            assert connection.execute(sa.text("SELECT * FROM runs")).mappings().one() == before
        command.upgrade(config, NEW_HEAD)
    finally:
        engine.dispose()


def test_downgrade_refuses_future_resume_data_without_erasing_it(database_url):
    command.upgrade(alembic_config(database_url), NEW_HEAD)
    engine = sa.create_engine(database_url)
    try:
        with engine.begin() as connection:
            _seed_gate6_migration_parent(
                connection, subject="r11-downgrade", graph_version="pathfinder-research-v6"
            )
            # Simulate a later step registering its real graph; R1.1 itself registers none.
            connection.execute(sa.text("ALTER TABLE runs DROP CONSTRAINT ck_runs_graph_version"))
            connection.execute(
                sa.text(
                    "UPDATE runs SET mode='material_preparation', "
                    "graph_version='pathfinder-resume-v1'"
                )
            )
        with pytest.raises(RuntimeError, match="cannot be safely downgraded"):
            command.downgrade(alembic_config(database_url), OLD_HEAD)
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == NEW_HEAD
            assert connection.scalar(sa.text("SELECT mode FROM runs")) == "material_preparation"
    finally:
        engine.dispose()
