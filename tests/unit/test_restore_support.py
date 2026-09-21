"""Failure propagation and ownership boundaries for the E8.4 test-only driver."""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tests.integration.restore_support import (
    DOCKER,
    LABEL,
    OwnedDatabase,
    Rehearsal,
    RestoreError,
    digest,
)


@pytest.fixture
def rehearsal(tmp_path):
    return Rehearsal(tmp_path / "owned")


def test_command_has_bounded_timeout_and_excludes_credentials(rehearsal, monkeypatch):
    captured = {}

    def run(args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"safe")

    monkeypatch.setenv("DASHSCOPE_API_KEY", "PRIVATE-CANARY")
    monkeypatch.setenv("DOCKER_HOST", "tcp://unowned.invalid:2375")
    monkeypatch.setattr(subprocess, "run", run)
    assert rehearsal.command([*DOCKER, "version"]) == b"safe"
    assert "DASHSCOPE_API_KEY" not in captured["env"]
    assert "DOCKER_HOST" not in captured["env"]
    assert 0 < captured["timeout"] <= 60
    assert captured["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize("failure", ["exit", "timeout", "unavailable", "deadline"])
def test_command_failure_never_exposes_error_body(rehearsal, monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("PRIVATE-CANARY", 60, output=b"PRIVATE-CANARY")
        if failure == "unavailable":
            raise OSError("PRIVATE-CANARY")
        return SimpleNamespace(returncode=1, stdout=b"PRIVATE-CANARY")

    monkeypatch.setattr(subprocess, "run", run)
    if failure == "deadline":
        rehearsal.deadline = 0
    with pytest.raises(RestoreError) as error:
        rehearsal.command(["test-only"])
    assert "PRIVATE-CANARY" not in str(error.value)


def container_record(database):
    return {
        "Id": "a" * 64,
        "Name": "/" + database.name,
        "Config": {"Labels": {LABEL: database.rehearsal.owner}},
        "HostConfig": {"NanoCpus": 2_000_000_000, "Memory": 1024**3, "PidsLimit": 128},
        "State": {"Running": True},
    }


@pytest.mark.parametrize("damage", ["label", "name", "id", "cpu", "memory", "pids"])
def test_foreign_or_unbounded_container_is_never_stopped(rehearsal, monkeypatch, damage):
    database = OwnedDatabase(rehearsal, "source")
    database.created = True
    record = container_record(database)
    if damage == "label":
        record["Config"]["Labels"][LABEL] = "other-owner"
    elif damage == "name":
        record["Name"] = "/other-project"
    elif damage == "id":
        database.identifier = "b" * 64
    else:
        record["HostConfig"][
            {"cpu": "NanoCpus", "memory": "Memory", "pids": "PidsLimit"}[damage]
        ] = 0
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return json.dumps([record]).encode()

    monkeypatch.setattr(rehearsal, "command", command)
    with pytest.raises(RestoreError, match="unsafe_container"):
        database.stop()
    assert len(calls) == 1 and "stop" not in calls[0]


def test_cleanup_failure_is_reported_and_remaining_owned_resources_are_attempted(rehearsal):
    calls = []

    def bad():
        calls.append("bad")
        raise RestoreError("cleanup_failed")

    rehearsal.containers = [
        SimpleNamespace(stop=lambda: calls.append("good")),
        SimpleNamespace(stop=bad),
    ]
    assert rehearsal.stop() is False
    assert calls == ["bad", "good"]


@pytest.mark.parametrize(
    "failure,expected_calls",
    [
        ("checksum", []),
        ("invalid_list", ["list"]),
        ("occupied", ["list", "empty"]),
        ("restore", ["list", "empty", "restore"]),
        ("migration", ["list", "empty", "restore", "migration"]),
        ("snapshot", ["list", "empty", "restore", "migration", "snapshot"]),
    ],
)
def test_restore_failures_never_reach_activation(rehearsal, monkeypatch, failure, expected_calls):
    database = OwnedDatabase(rehearsal, "target")
    archive = rehearsal.directory / "synthetic.dump"
    archive.write_bytes(b"test-only-custom-dump")
    calls = []

    def validate(path):
        calls.append("list")
        if failure == "invalid_list":
            raise RestoreError("command_failed")

    @contextmanager
    def connect():
        calls.append("empty")
        yield SimpleNamespace(
            execute=lambda _: SimpleNamespace(fetchone=lambda: (failure == "occupied",))
        )

    def execute(*args, **kwargs):
        calls.append("restore")
        if failure == "restore":
            raise RestoreError("command_failed")

    def upgrade():
        calls.append("migration")
        if failure == "migration":
            raise RestoreError("schema_mismatch")

    def snapshot():
        calls.append("snapshot")
        return {"changed": True}

    monkeypatch.setattr(database, "validate_dump", validate)
    monkeypatch.setattr(database, "connect", connect)
    monkeypatch.setattr(database, "execute", execute)
    monkeypatch.setattr(database, "upgrade", upgrade)
    monkeypatch.setattr(database, "snapshot", snapshot)
    with pytest.raises(RestoreError):
        database.restore(
            archive, "0" * 64 if failure == "checksum" else digest(archive.read_bytes()), {}
        )
        calls.append("activation")
    assert calls == expected_calls
    assert archive.read_bytes() == b"test-only-custom-dump"


def test_snapshot_mismatch_detects_changes_even_when_counts_match(rehearsal, monkeypatch):
    database = OwnedDatabase(rehearsal, "source")

    @contextmanager
    def connect():
        def execute(query):
            if isinstance(query, str):
                return SimpleNamespace(fetchall=lambda: [("public", "runs")])
            return SimpleNamespace(fetchall=lambda: [('"PRIVATE-BODY-CANARY"',)])

        yield SimpleNamespace(execute=execute)

    monkeypatch.setattr(database, "connect", connect)
    value = database.snapshot()
    assert value["public.runs"]["rows"] == 1
    assert "PRIVATE-BODY-CANARY" not in json.dumps(value)
    assert value != {"public.runs": {"rows": 1, "sha256": "0" * 64}}
