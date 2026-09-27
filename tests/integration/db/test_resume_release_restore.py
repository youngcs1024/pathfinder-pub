"""R7.2 actual PostgreSQL migration and isolated restore; no deployment DSN."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import pytest
from alembic import command
from psycopg import sql

from tests.integration import resume_product
from tests.integration.restore_product import decide, product, seed_current
from tests.integration.restore_support import HEAD, IMAGE, Rehearsal, RestoreError
from tests.integration.support import alembic_config

pytestmark = pytest.mark.integration
OLD = "0015_e3_run_request_identity"


async def test_resume_upgrade_backup_restore_and_continue(tmp_path, monkeypatch, request):
    report = dict(
        status="IN_PROGRESS",
        stage="environment",
        checks={},
        old_revision=OLD,
        revision=HEAD,
        graph="pathfinder-resume-v4",
        postgres_image=IMAGE,
        elapsed_ms=0,
        resources_stopped=False,
        tables={},
        failure_category=None,
    )
    request.node.user_properties.append(("restore", report))
    rehearsal = Rehearsal(tmp_path / "r72")
    try:
        async with asyncio.timeout(600):
            source = rehearsal.start("source")
            target = rehearsal.start("target")
            assert source.image_id == target.image_id
            report["stage"] = "upgrade"
            source.upgrade(OLD)
            legacy = product(source, rehearsal.directory / "legacy")
            await seed_current(legacy, monkeypatch, rehearsal)
            drained = product(source, rehearsal.directory / "legacy-drain")
            drained.document_id, drained.runs = legacy.document_id, dict(legacy.runs)
            async with drained.open():
                for name in ("approve", "reject"):
                    await decide(drained, name, "reject")
                    await drained.work(name)
            before_upgrade = source.snapshot()
            # Real approval decisions, invocation accounting, and checkpoint rows exist at 0015.
            for table in ("approval_decisions", "llm_invocations", "run_events"):
                assert before_upgrade["public." + table]["rows"] > 0
            with source.connect() as connection:
                columns = connection.execute(
                    "SELECT table_schema, table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema IN ('public', 'pathfinder_checkpoint') "
                    "ORDER BY ordinal_position"
                ).fetchall()
            legacy_columns = {}
            for schema, table, column in columns:
                legacy_columns.setdefault((schema, table), []).append(column)
            old_rows = projected_history(source, legacy_columns)
            source.upgrade()
            upgraded = source.snapshot()
            assert projected_history(source, legacy_columns) == old_rows
            source.upgrade()
            assert source.snapshot() == upgraded
            report["checks"]["legacy_upgrade"] = True
            report["stage"] = "fixtures"
            async with resume_product.product(
                source.url, rehearsal.directory / "source-product"
            ) as rig:
                state = await resume_product.seed(rig)
                await resume_product.revise(rig, state)
                await resume_product.revise(rig, state, lock=True)
                await resume_product.confirm(rig, state)
                for run_id in legacy.runs.values():
                    url = f"/api/v1/workspaces/{rig.tenant.workspace_id}/runs/{run_id}"
                    assert (await rig.client.get(url)).status_code == 200
                old_write = f"/api/v1/workspaces/{rig.tenant.workspace_id}/runs"
                assert (
                    await rig.client.post(old_write, json={"query": "historical"})
                ).status_code == 410
                assert not await rig.reader.has_unsupported_pending_work()
            report["checks"].update(current_fixtures=True, historical_output=True)
            report["stage"] = "downgrade_guard"
            before = source.snapshot()
            with pytest.raises(RuntimeError, match="cannot be safely downgraded"):
                command.downgrade(alembic_config(source.maintenance_url), OLD)
            assert source.snapshot() == before
            report["checks"]["keyed_downgrade_refused"] = True
            report["stage"] = "backup"
            dump, checksum = source.dump()
            assert source.snapshot() == before
            report.update(dump_sha256=checksum, dump_bytes=dump.stat().st_size)
            report["checks"]["stable_backup"] = True
            report["postgres_image_id"] = source.image_id
            report["snapshot_sha256"] = hashlib.sha256(
                json.dumps(before, sort_keys=True).encode()
            ).hexdigest()
            source_probe = operational_snapshot(source, state["run_id"])
            assert all(source_probe["checks"].values())
            report["stage"] = "restore"
            corrupt = rehearsal.directory / "corrupt.dump"
            with corrupt.open("xb") as handle:
                handle.write(dump.read_bytes()[:32])
            with pytest.raises(RestoreError, match="command_failed"):
                target.validate_dump(corrupt)
            with pytest.raises(RestoreError, match="checksum_mismatch"):
                target.restore(dump, "0" * 64, before)
            report["checks"]["invalid_backup_refused"] = True
            report["tables"] = target.restore(dump, checksum, before)
            assert report["tables"] == before
            with pytest.raises(RestoreError, match="target_not_empty"):
                target.restore(dump, checksum, before)
            assert target.snapshot() == before
            assert operational_snapshot(target, state["run_id"]) == source_probe
            report["checks"].update(exact_restore=True, occupied_target_refused=True)
            report["stage"] = "resume"
            async with resume_product.product(
                target.url, rehearsal.directory / "target-product"
            ) as rig:
                assert (await rig.client.get("/readyz")).status_code == 200
                await resume_product.verify_history(rig, state)
                confirmed, _ = await resume_product.post(
                    rig,
                    resume_product.session_path(rig, state)
                    + "/versions/"
                    + state["confirmation"]["version_id"]
                    + "/confirm",
                    state["confirm_body"],
                    key=state["confirm_key"],
                )
                assert confirmed["replayed"]
                assert target.snapshot() == before
                await resume_product.revise(rig, state)
                await resume_product.verify_history(rig, state)
                await resume_product.check_sse(rig, state)
                assert len(state["versions"]) == 4
            # Historical rows remain byte-identical even after new resume work.
            after = target.snapshot()
            for name in (
                "approval_requests",
                "approval_decisions",
                "action_intents",
                "mock_submissions",
            ):
                assert after["public." + name] == before["public." + name]
            report["checks"].update(
                request_replay=True,
                replay_authorization=True,
                resume_history_download=True,
                resume_revision_after_restore=True,
            )
            report["business_digest"] = hashlib.sha256(
                json.dumps(after, sort_keys=True).encode()
            ).hexdigest()
            report.update(stage="complete", status="PASS")
    except BaseException:
        report.update(status="IN_PROGRESS", failure_category="unexpected_error")
        raise RestoreError("resume_restore_rehearsal_failed") from None
    finally:
        report["resources_stopped"] = rehearsal.stop()
        report["elapsed_ms"] = round((time.monotonic() - rehearsal.started) * 1000)
        if not report["resources_stopped"]:
            report.update(stage="cleanup", status="IN_PROGRESS", failure_category="cleanup_failed")
            raise RestoreError("cleanup_failed")


def test_empty_resume_upgrade_is_repeatable(database_url):
    config = alembic_config(database_url)
    command.upgrade(config, "head")
    command.upgrade(config, "head")
    import sqlalchemy as sa

    engine = sa.create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.scalar(sa.text("SELECT version_num FROM alembic_version")) == HEAD
            assert connection.scalar(sa.text("SELECT count(*) FROM resume_confirmations")) == 0
    finally:
        engine.dispose()


def projected_history(database, columns):
    result = {}
    with database.connect() as connection:
        for (schema, table), names in columns.items():
            if table == "alembic_version":
                continue
            query = sql.SQL(
                "SELECT to_jsonb(t)::text FROM (SELECT {} FROM {}.{}) t ORDER BY to_jsonb(t)::text"
            ).format(
                sql.SQL(",").join(map(sql.Identifier, names)),
                sql.Identifier(schema),
                sql.Identifier(table),
            )
            rows = connection.execute(query).fetchall()
            result[schema + "." + table] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    return result


def operational_snapshot(database, run_id):
    path = Path(__file__).resolve().parents[3] / "scripts/resume_restore_consistency.sql"
    with path.open("rb") as script:
        result = database.execute(
            [
                "psql",
                "-X",
                "-qAt",
                "-U",
                "pf_e84",
                "-d",
                "pathfinder",
                "--set",
                "ON_ERROR_STOP=1",
                "--set",
                f"fixture_run_id={run_id}",
            ],
            stdin=script,
        )
    return json.loads(result)
