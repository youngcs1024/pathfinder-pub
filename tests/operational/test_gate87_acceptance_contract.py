from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "gate87_acceptance.sh"
SQL = PROJECT_ROOT / "scripts" / "gate87_live_acceptance.sql"


def _fake_docker(path: Path) -> Path:
    executable = path / "docker"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + "\\n")

if args[0] == "inspect":
    template = args[2]
    container = args[3]
    if "NetworkSettings.Ports" in template:
        host = os.environ.get("FAKE_CADDY_HOST", "127.0.0.1")
        print(json.dumps({"80/tcp": [{"HostIp": host, "HostPort": "18080"}]}))
    else:
        image = "sha256:application"
        if container == "worker-id" and os.environ.get("FAKE_IMAGE_MISMATCH") == "1":
            image = "sha256:other"
        health = "" if container == "caddy-id" else "healthy"
        print(f"running|{health}|pathfinder@sha256:ref|{image}")
    raise SystemExit(0)

command_index = next(
    index for index, value in enumerate(args) if value in {"config", "ps", "exec"}
)
command = args[command_index]
tail = args[command_index + 1:]
if command == "config":
    if "--format" in tail:
        host = os.environ.get("FAKE_CADDY_HOST", "127.0.0.1")
        print(json.dumps({
            "services": {
                "postgres": {"image": "postgres:16"},
                "api": {"image": "pathfinder@sha256:ref"},
                "worker": {"image": "pathfinder@sha256:ref"},
                "caddy": {"image": "caddy:2", "ports": [{
                    "target": 80, "published": 18080, "host_ip": host, "protocol": "tcp"
                }]},
            }
        }))
    raise SystemExit(0)
if command == "ps":
    service = tail[-1]
    print(f"{service}-id")
    if service == "worker" and os.environ.get("FAKE_MULTIPLE_WORKERS") == "1":
        print("worker-second-id")
    raise SystemExit(0)

service = tail[tail.index("-T") + 1]
if service == "worker":
    print(os.environ.get("FAKE_WORKER_MODES", '["qwen","tavily","langfuse",1.0]'))
elif service == "api" and "alembic" in tail:
    print("0015_e3_run_request_identity (head)")
elif service == "api":
    print(os.environ.get("FAKE_API_AUTH", "supabase"))
elif service == "postgres":
    query = tail[-1]
    if "alembic_version" in query:
        print("0015_e3_run_request_identity")
    else:
        print(os.environ.get("FAKE_UNKNOWN_COUNT", "0"))
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_docker(bin_dir)
    compose = tmp_path / "compose.yaml"
    overlay = tmp_path / "private-overlay.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")
    overlay.write_text("services: {}\n", encoding="utf-8")
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    log = tmp_path / "docker.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "COMPOSE_PROJECT_NAME": "pathfinder-live",
            "PF_GATE87_COMPOSE_FILES": f"{compose}:{overlay}",
            "PF_GATE87_EVIDENCE_DIR": str(evidence),
            "PF_GATE87_COST_CAP_CNY": "1.00",
            "FAKE_DOCKER_LOG": str(log),
            "DASHSCOPE_API_KEY": "secret-canary-qwen",
            "TAVILY_API_KEY": "secret-canary-tavily",
            "LANGFUSE_SECRET_KEY": "secret-canary-langfuse",
        }
    )
    return env, log


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(SCRIPT), "preflight"),
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "missing",
    [
        "COMPOSE_PROJECT_NAME",
        "PF_GATE87_COMPOSE_FILES",
        "PF_GATE87_EVIDENCE_DIR",
        "PF_GATE87_COST_CAP_CNY",
    ],
)
def test_preflight_requires_explicit_project_files_evidence_and_cost_cap(
    tmp_path: Path, missing: str
) -> None:
    env, _log = _environment(tmp_path)
    env.pop(missing)

    result = _run(env)

    assert result.returncode != 0
    assert missing in result.stderr


def test_preflight_rejects_insecure_evidence_directory_permissions(tmp_path: Path) -> None:
    env, _log = _environment(tmp_path)
    Path(env["PF_GATE87_EVIDENCE_DIR"]).chmod(0o755)

    result = _run(env)

    assert result.returncode != 0
    assert "group or other permissions" in result.stderr


def test_preflight_requires_repository_external_private_overlay(tmp_path: Path) -> None:
    env, _log = _environment(tmp_path)
    env["PF_GATE87_COMPOSE_FILES"] = (
        f"{PROJECT_ROOT / 'compose.yaml'}:{PROJECT_ROOT / 'compose.dev.yaml'}"
    )

    result = _run(env)

    assert result.returncode != 0
    assert "repository-external private overlay" in result.stderr


def test_preflight_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    env, _log = _environment(tmp_path)
    evidence_path = Path(env["PF_GATE87_EVIDENCE_DIR"]) / "gate87-preflight.json"
    original = '{"sentinel":true}\n'
    evidence_path.write_text(original, encoding="utf-8")

    result = _run(env)

    assert result.returncode != 0
    assert "evidence already exists" in result.stderr
    assert evidence_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("override", "value", "message"),
    [
        ("FAKE_WORKER_MODES", '["fake","tavily","langfuse",1.0]', "qwen/tavily"),
        ("FAKE_API_AUTH", "fake", "auth mode"),
        ("FAKE_MULTIPLE_WORKERS", "1", "cardinality"),
        ("FAKE_CADDY_HOST", "0.0.0.0", "loopback-only"),
        ("FAKE_IMAGE_MISMATCH", "1", "image identity"),
    ],
)
def test_preflight_fails_closed_for_runtime_mismatch(
    tmp_path: Path, override: str, value: str, message: str
) -> None:
    env, _log = _environment(tmp_path)
    env[override] = value

    result = _run(env)

    assert result.returncode != 0
    assert message in result.stderr


def test_preflight_is_sanitized_read_only_and_writes_private_evidence(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "preflight PASS" in output
    assert "secret-canary" not in output
    evidence_path = Path(env["PF_GATE87_EVIDENCE_DIR"]) / "gate87-preflight.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["accepted"] is True
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    flattened = "\n".join(" ".join(record) for record in records)
    for forbidden in (
        " up ",
        " stop ",
        " start ",
        " restart ",
        " down ",
        " rm ",
        " prune ",
        "INSERT ",
        "UPDATE ",
        "DELETE ",
    ):
        assert forbidden not in f" {flattened} "


def test_preflight_error_stops_before_database_stages(tmp_path: Path) -> None:
    env, log = _environment(tmp_path)
    env["FAKE_WORKER_MODES"] = '["fake","fake","off",0.0]'

    result = _run(env)

    assert result.returncode != 0
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert not any("alembic" in record or "alembic_version" in record for record in records)


def test_operator_contract_contains_only_read_only_fail_closed_primitives() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    sql = SQL.read_text(encoding="utf-8")

    assert "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY" in sql
    assert "ROLLBACK" in sql and "\\quit 3" in sql
    assert "checkpoint.payload" not in sql
    assert "checkpoint.checkpoint AS" not in sql
    assert "document.content" not in sql
    assert "action.args_snapshot" not in sql
    assert "submission.payload" not in sql
    for forbidden in (
        "docker compose down",
        "docker prune",
        "down -v",
        "sudo ",
        "reboot",
        "shutdown",
        "systemctl",
        "rm -rf",
    ):
        assert forbidden not in script
