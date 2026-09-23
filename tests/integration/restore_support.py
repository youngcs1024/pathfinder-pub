"""Owned, bounded E8.4 database rehearsal. No external DSN or deployment is accepted."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import psycopg
from alembic import command
from psycopg import sql

from tests.integration.support import alembic_config

ROOT = Path(__file__).resolve().parents[2]
OLD_HEAD = "0014_gate6_action_recovery"
HEAD = "0018_r21_material_snapshots"
IMAGE = "pgvector/pgvector:0.8.5-pg16"
LABEL = "pathfinder.e84.owner"
MAX_DUMP_BYTES = 64 * 1024 * 1024
DOCKER = ("docker", "--host", "unix:///var/run/docker.sock")


class RestoreError(Exception):
    """Only a fixed classification crosses the test boundary."""


def require(condition, category):
    if not condition:
        raise RestoreError(category)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Rehearsal:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(mode=0o700)
        self.owner = uuid4().hex
        self.started = time.monotonic()
        self.deadline = self.started + 600
        self.containers = []

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "deadline")
        return min(60, remaining)

    def command(self, args, *, stdin=None, stdout=subprocess.PIPE, environment=None):
        # Never inherit provider credentials or an alternate Docker endpoint/context.
        env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
        if environment:
            env.update(environment)
        try:
            result = subprocess.run(
                args,
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.DEVNULL,
                env=env,
                timeout=self.remaining(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise RestoreError("deadline") from None
        except OSError:
            raise RestoreError("command_unavailable") from None
        require(result.returncode == 0, "command_failed")
        return result.stdout

    def start(self, role):
        require(role in {"source", "target"}, "invalid_role")
        database = OwnedDatabase(self, role)
        self.containers.append(database)
        database.start()
        return database

    def stop(self):
        # Stop only verified owned containers, retaining their data and dump files.
        # GitHub later disposes of its runner. No rm/prune/volume deletion is issued here.
        stopped = True
        for database in reversed(self.containers):
            try:
                database.stop()
            except (RestoreError, ValueError, KeyError, TypeError):
                stopped = False
        return stopped


class OwnedDatabase:
    def __init__(self, rehearsal, role):
        self.rehearsal = rehearsal
        self.name = f"pf-e84-{rehearsal.owner}-{role}"
        self.identifier = None
        self.password = uuid4().hex
        self.port = None
        self.image_id = None
        self.created = False

    def inspect(self):
        raw = self.rehearsal.command([*DOCKER, "inspect", self.name])
        records = json.loads(raw)
        require(isinstance(records, list) and len(records) == 1, "unsafe_container")
        value = records[0]
        identifier = value["Id"]
        require(re.fullmatch(r"[0-9a-f]{64}", identifier) is not None, "unsafe_container")
        require(
            value["Name"] == "/" + self.name
            and value["Config"]["Labels"].get(LABEL) == self.rehearsal.owner
            and (self.identifier is None or identifier == self.identifier),
            "unsafe_container",
        )
        require(
            value["HostConfig"]["NanoCpus"] == 2_000_000_000
            and value["HostConfig"]["Memory"] == 1024**3
            and value["HostConfig"]["PidsLimit"] == 128,
            "unsafe_container",
        )
        return value

    def start(self):
        self.created = True
        self.rehearsal.command(
            [
                *DOCKER,
                "run",
                "--detach",
                "--name",
                self.name,
                "--label",
                f"{LABEL}={self.rehearsal.owner}",
                "--cpus",
                "2",
                "--memory",
                "1g",
                "--pids-limit",
                "128",
                "--publish",
                "127.0.0.1::5432",
                "--env",
                "POSTGRES_USER",
                "--env",
                "POSTGRES_PASSWORD",
                "--env",
                "POSTGRES_DB",
                IMAGE,
            ],
            environment={
                "POSTGRES_USER": "pf_e84",
                "POSTGRES_PASSWORD": self.password,
                "POSTGRES_DB": "pathfinder",
            },
        )
        value = self.inspect()
        self.identifier, self.image_id = value["Id"], value["Image"]
        ports = value["NetworkSettings"]["Ports"]
        require(set(ports) == {"5432/tcp"}, "unsafe_binding")
        bindings = ports["5432/tcp"]
        require(len(bindings) == 1 and bindings[0]["HostIp"] == "127.0.0.1", "unsafe_binding")
        self.port = int(bindings[0]["HostPort"])
        require(0 < self.port < 65536, "unsafe_binding")
        ready_deadline = min(self.rehearsal.deadline, time.monotonic() + 60)
        while time.monotonic() < ready_deadline:
            try:
                with self.connect() as connection:
                    connection.execute("SELECT 1")
                return
            except psycopg.OperationalError:
                time.sleep(0.1)
        raise RestoreError("database_not_ready")

    @property
    def url(self):
        require(self.port is not None, "database_not_ready")
        return f"postgresql+psycopg://pf_e84:{self.password}@127.0.0.1:{self.port}/pathfinder"

    @property
    def maintenance_url(self):
        # Application pools reject DSN options and own their E2 policy. DDL is independent.
        return self.url + "?connect_timeout=5&options=-c%20statement_timeout%3D30000"

    def connect(self):
        return psycopg.connect(
            host="127.0.0.1",
            port=self.port,
            user="pf_e84",
            password=self.password,
            dbname="pathfinder",
            connect_timeout=5,
            options="-c statement_timeout=10000 -c lock_timeout=1000",
        )

    def upgrade(self, revision=HEAD):
        self.rehearsal.remaining()
        command.upgrade(alembic_config(self.maintenance_url), revision)
        with self.connect() as connection:
            actual = connection.execute("SELECT version_num FROM alembic_version").fetchone()[0]
        require(actual == revision, "schema_mismatch")

    def execute(self, args, **kwargs):
        require(self.identifier is not None, "database_not_ready")
        self.inspect()
        return self.rehearsal.command([*DOCKER, "exec", "-i", self.identifier, *args], **kwargs)

    def snapshot(self):
        """Hash every row, including JSON, vectors and checkpoint bytes, without publishing it."""
        with self.connect() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            tables = connection.execute(
                "SELECT schemaname, tablename FROM pg_tables "
                "WHERE schemaname IN ('public', 'pathfinder_checkpoint') "
                "ORDER BY schemaname, tablename"
            ).fetchall()
            result = {}
            for schema, table in tables:
                rows = connection.execute(
                    sql.SQL(
                        "SELECT to_jsonb(t)::text FROM {}.{} t ORDER BY to_jsonb(t)::text"
                    ).format(sql.Identifier(schema), sql.Identifier(table))
                ).fetchall()
                # Length framing preserves boundaries, including empty tables and embedded newlines.
                hasher = hashlib.sha256()
                for (row,) in rows:
                    encoded = row.encode()
                    hasher.update(len(encoded).to_bytes(8, "big"))
                    hasher.update(encoded)
                result[f"{schema}.{table}"] = {"rows": len(rows), "sha256": hasher.hexdigest()}
        return result

    def dump(self):
        path = self.rehearsal.directory / "source.dump"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            self.execute(["pg_dump", "-U", "pf_e84", "-d", "pathfinder", "-Fc"], stdout=output)
        require(0 < path.stat().st_size <= MAX_DUMP_BYTES, "invalid_dump")
        self.validate_dump(path)
        return path, digest(path.read_bytes())

    def validate_dump(self, path):
        with path.open("rb") as source:
            self.execute(["pg_restore", "--list"], stdin=source, stdout=subprocess.DEVNULL)

    def restore(self, path, expected_digest, expected_snapshot):
        require(not path.is_symlink() and path.is_file(), "invalid_dump")
        require(0 < path.stat().st_size <= MAX_DUMP_BYTES, "invalid_dump")
        require(digest(path.read_bytes()) == expected_digest, "checksum_mismatch")
        self.validate_dump(path)
        with self.connect() as connection:
            occupied = connection.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_class c JOIN pg_namespace n "
                "ON n.oid=c.relnamespace WHERE n.nspname IN ('public','pathfinder_checkpoint') "
                "AND c.relkind IN ('r','p','v','m','S','f'))"
            ).fetchone()[0]
        require(not occupied, "target_not_empty")
        with path.open("rb") as source:
            self.execute(
                [
                    "pg_restore",
                    "--exit-on-error",
                    "--no-owner",
                    "--no-privileges",
                    "-U",
                    "pf_e84",
                    "-d",
                    "pathfinder",
                ],
                stdin=source,
                stdout=subprocess.DEVNULL,
            )
        self.upgrade()
        actual = self.snapshot()
        require(actual == expected_snapshot, "snapshot_mismatch")
        return actual

    def stop(self):
        if self.created:
            # Cleanup has its own short deadline even after the rehearsal deadline expires.
            previous = self.rehearsal.deadline
            self.rehearsal.deadline = time.monotonic() + 20
            try:
                value = self.inspect()
                self.rehearsal.command([*DOCKER, "stop", "--time", "5", value["Id"]])
                require(not self.inspect()["State"]["Running"], "cleanup_failed")
            finally:
                self.rehearsal.deadline = previous


def consistency_snapshot(database, run_id, document_id, pending_run_ids):
    query = "\n".join(
        line
        for line in (ROOT / "scripts/gate85_consistency.sql").read_text().splitlines()
        if not line.startswith("\\")
    )
    for key, value in (("fixture_run_id", run_id), ("fixture_document_id", document_id)):
        # UUID objects come from the trusted test fixture, not SQL input supplied by a caller.
        query = query.replace(f":'{key}'", "'" + str(value) + "'")
    with database.connect() as connection, connection.cursor() as cursor:
        cursor.execute(query, prepare=False)
        row = None
        while True:
            if cursor.description is not None:
                row = cursor.fetchone()
            if not cursor.nextset():
                break
    require(row is not None, "probe_failed")
    result = json.loads(row[0])
    # Gate 8.5's remote script still requires source_is_quiescent=true. This isolated fixture
    # intentionally retains exactly two waiting approvals, with all execution connections closed.
    # Check that exact alternative here; do not relax or change the original remote gate.
    expected = dict.fromkeys(result["checks"], True)
    expected["source_is_quiescent"] = False
    require(result["checks"] == expected, "probe_failed")
    require(len(set(pending_run_ids)) == 2, "probe_failed")
    with database.connect() as connection:
        active = connection.execute(
            "SELECT id, status FROM runs WHERE status IN ('queued','running','waiting_approval')"
        ).fetchall()
        executable = connection.execute(
            "SELECT count(*) FROM run_jobs WHERE status IN ('queued','leased')"
        ).fetchone()[0]
    require(
        set(active) == {(identifier, "waiting_approval") for identifier in pending_run_ids}
        and executable == 0,
        "probe_failed",
    )
    return result
