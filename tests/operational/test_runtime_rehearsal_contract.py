from __future__ import annotations

import json
import os
import secrets
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REHEARSAL_SCRIPT = PROJECT_ROOT / "scripts" / "runtime_rehearsal.sh"
METRICS_SQL = PROJECT_ROOT / "scripts" / "gate86_runtime_metrics.sql"
RECOVERY_SQL = PROJECT_ROOT / "scripts" / "gate86_recovery_state.sql"

PROJECT = "pathfinder-gate86-contract"
ACTOR_ID = "11111111-1111-4111-8111-111111111111"
WORKSPACE_ID = "22222222-2222-4222-8222-222222222222"
DOCUMENT_ID = "33333333-3333-4333-8333-333333333333"
BASELINE_RUN_ID = "44444444-4444-4444-8444-444444444444"
WAITING_RUN_ID = "55555555-5555-4555-8555-555555555555"
LEASED_RUN_ID = "66666666-6666-4666-8666-666666666666"
QUEUED_RUN_ID = "77777777-7777-4777-8777-777777777777"
BASELINE_ACTION_ID = "88888888-8888-4888-8888-888888888888"
WAITING_ACTION_ID = "99999999-9999-4999-8999-999999999999"
BASELINE_REQUEST_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
WAITING_REQUEST_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
POSTGRES_ID = "1" * 64
API_ID = "2" * 64
WORKER_ID = "3" * 64
CADDY_ID = "4" * 64
APPLICATION_IMAGE_ID = f"sha256:{'a' * 64}"
APPLICATION_IMAGE = f"pathfinder:gate86@sha256:{'b' * 64}"
POSTGRES_IMAGE = f"pgvector/pgvector:0.8.5-pg16@sha256:{'c' * 64}"
MIGRATION_REVISION = "0015_e3_run_request_identity"


def _event(seq: int, event_type: str) -> dict[str, object]:
    return {
        "id": f"event-{seq}-{event_type}",
        "seq": seq,
        "type": event_type,
        "created_at": f"2026-08-28T00:00:{seq:02d}+00:00",
    }


def _fixture(
    *,
    name: str,
    run_id: str,
    status: str,
    job_status: str,
    attempt: int,
    lease_expires_at: str | None = None,
    leased_by: str | None = None,
) -> dict[str, Any]:
    return {
        "run": {
            "id": run_id,
            "status": status,
            "graph_version": "pathfinder-research-v6",
            "next_event_seq": 2,
        },
        "job": {
            "status": job_status,
            "attempt": attempt,
            "lease_expires_at": lease_expires_at,
            "leased_by": leased_by,
            "owner_present": leased_by is not None,
        },
        "events": [_event(1, "run.created")],
        "approval_requests": [],
        "decision_count": 0,
        "approve_decision_count": 0,
        "actions": [],
        "mock_submissions": [],
        "tool_invocations": [],
        "llm_invocations": [],
        "checkpoints": [],
        "fixture_name": name,
    }


def _snapshots() -> tuple[str, str, str]:
    now = datetime.now(UTC)
    live_lease = (now + timedelta(seconds=3)).isoformat()
    expired_lease = (now - timedelta(seconds=3)).isoformat()
    approval_expiry = (now + timedelta(hours=2)).isoformat()
    leased_by = "gate86-contract-crash-window"

    baseline = _fixture(
        name="baseline",
        run_id=BASELINE_RUN_ID,
        status="completed",
        job_status="done",
        attempt=1,
    )
    baseline.update(
        {
            "events": [_event(1, "run.created"), _event(2, "run.completed")],
            "approval_requests": [
                {"id": BASELINE_REQUEST_ID, "status": "consumed", "expires_at": approval_expiry}
            ],
            "decision_count": 1,
            "approve_decision_count": 1,
            "actions": [
                {
                    "id": BASELINE_ACTION_ID,
                    "status": "succeeded",
                    "idempotency_key": BASELINE_ACTION_ID,
                }
            ],
            "mock_submissions": [
                {"idempotency_key": BASELINE_ACTION_ID, "payload_digest": "sha256:baseline"}
            ],
            "tool_invocations": [
                {
                    "id": "baseline-submit-tool",
                    "effect": "irreversible",
                    "status": "succeeded",
                }
            ],
            "checkpoints": [{"checkpoint_ns": "", "checkpoint_id": "baseline-checkpoint"}],
        }
    )
    waiting = _fixture(
        name="waiting",
        run_id=WAITING_RUN_ID,
        status="waiting_approval",
        job_status="done",
        attempt=1,
    )
    waiting.update(
        {
            "events": [_event(1, "run.created"), _event(2, "approval.requested")],
            "approval_requests": [
                {"id": WAITING_REQUEST_ID, "status": "pending", "expires_at": approval_expiry}
            ],
            "actions": [
                {
                    "id": WAITING_ACTION_ID,
                    "status": "proposed",
                    "idempotency_key": WAITING_ACTION_ID,
                }
            ],
            "llm_invocations": [{"id": "waiting-llm", "provider": "fake", "status": "succeeded"}],
            "tool_invocations": [
                {
                    "id": "waiting-read-tool",
                    "effect": "read_only",
                    "status": "succeeded",
                }
            ],
            "checkpoints": [{"checkpoint_ns": "", "checkpoint_id": "waiting-checkpoint"}],
        }
    )
    leased = _fixture(
        name="leased",
        run_id=LEASED_RUN_ID,
        status="queued",
        job_status="leased",
        attempt=1,
        lease_expires_at=live_lease,
        leased_by=leased_by,
    )
    queued = _fixture(
        name="queued",
        run_id=QUEUED_RUN_ID,
        status="queued",
        job_status="queued",
        attempt=0,
    )
    checks = {
        "fixture_runs_exactly_four": True,
        "jobs_exactly_four": True,
        "events_contiguous": True,
        "next_event_seq_consistent": True,
    }
    pre = {
        "fixtures": {
            "baseline": baseline,
            "waiting": waiting,
            "leased": leased,
            "queued": queued,
        },
        "checks": checks,
    }
    post = json.loads(json.dumps(pre))
    post["fixtures"]["leased"]["job"]["lease_expires_at"] = expired_lease

    final = json.loads(json.dumps(post))
    final["fixtures"]["queued"].update(
        {
            "run": {
                **final["fixtures"]["queued"]["run"],
                "status": "completed",
                "next_event_seq": 3,
            },
            "job": {
                **final["fixtures"]["queued"]["job"],
                "status": "done",
                "attempt": 1,
            },
            "events": [_event(1, "run.created"), _event(2, "run.completed")],
        }
    )
    final["fixtures"]["leased"].update(
        {
            "run": {
                **final["fixtures"]["leased"]["run"],
                "status": "completed",
                "next_event_seq": 4,
            },
            "job": {
                **final["fixtures"]["leased"]["job"],
                "status": "done",
                "attempt": 2,
            },
            "events": [
                _event(1, "run.created"),
                _event(2, "job.lease_expired"),
                _event(3, "run.completed"),
            ],
        }
    )
    final["fixtures"]["waiting"].update(
        {
            "run": {
                **final["fixtures"]["waiting"]["run"],
                "status": "completed",
                "next_event_seq": 4,
            },
            "job": {
                **final["fixtures"]["waiting"]["job"],
                "status": "done",
                "attempt": 2,
            },
            "events": [
                *final["fixtures"]["waiting"]["events"],
                _event(3, "run.completed"),
            ],
            "approval_requests": [
                {
                    "id": WAITING_REQUEST_ID,
                    "status": "consumed",
                    "expires_at": approval_expiry,
                }
            ],
            "decision_count": 1,
            "approve_decision_count": 1,
            "actions": [
                {
                    "id": WAITING_ACTION_ID,
                    "status": "succeeded",
                    "idempotency_key": WAITING_ACTION_ID,
                }
            ],
            "mock_submissions": [
                {"idempotency_key": WAITING_ACTION_ID, "payload_digest": "sha256:waiting"}
            ],
            "tool_invocations": [
                *final["fixtures"]["waiting"]["tool_invocations"],
                {
                    "id": "waiting-submit-tool",
                    "effect": "irreversible",
                    "status": "succeeded",
                },
            ],
        }
    )

    def compact(value: dict[str, Any]) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    return compact(pre), compact(post), compact(final)


METRICS = json.dumps(
    {
        "observed_at": "2026-08-28T00:00:00+00:00",
        "queue": {
            "queued_total": 0,
            "ready_queued_total": 0,
            "oldest_queued_age_seconds": None,
            "oldest_ready_queued_age_seconds": None,
            "leased_total": 0,
            "expired_leases_total": 0,
            "dead_total": 0,
        },
        "runs_recent_1h": {
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "failure_rate": None,
            "failure_rate_sample": 0,
        },
        "latency_ms": {},
        "llm_recent_1h": {},
        "tools_recent_1h": {},
        "cost_recent_24h": {"priced_cny": 0, "unpriced_succeeded": 0},
        "storage": {"postgres_database_bytes": 4096},
    },
    sort_keys=True,
    separators=(",", ":"),
)


FAKE_TOOL = f'''#!/usr/bin/env python3
import datetime as dt
import json
import os
import signal
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
log_path = Path(os.environ["FAKE_COMMAND_LOG"])

def log():
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{"tool": tool, "args": args}}, separators=(",", ":")) + "\\n")

def emit(value):
    sys.stdout.write(str(value) + "\\n")

def state():
    path = Path(os.environ["FAKE_STATE"])
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {{
        "phase": "initial",
        "baseline_approved": False,
        "waiting_approved": False,
        "clock": 1000,
    }}

def save(value):
    Path(os.environ["FAKE_STATE"]).write_text(json.dumps(value), encoding="utf-8")

def records():
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

failure = os.environ.get("FAKE_FAIL", "")
current = state()

if tool == "date":
    if args == ["+%s"]:
        current["clock"] += 10
        save(current)
        emit(current["clock"])
    elif args == ["-u", "+%Y-%m-%dT%H:%M:%SZ"]:
        emit("2026-08-28T00:00:00Z")
    elif args == ["-u", "+%Y%m%dT%H%M%SZ"]:
        emit("20260828T000000Z")
    else:
        raise SystemExit(98)
    raise SystemExit(0)

if tool == "sleep":
    raise SystemExit(0)

if tool == "stat":
    if "%u" in args:
        emit(os.getuid())
    elif "%a" in args:
        emit("700")
    else:
        raise SystemExit(98)
    raise SystemExit(0)

if tool != "docker":
    raise SystemExit(98)

log()

if args[:2] == ["volume", "ls"]:
    emit("{PROJECT}_caddy_config")
    emit("{PROJECT}_caddy_data")
    emit("{PROJECT}_postgres_data")
    raise SystemExit(0)
if args[:2] == ["volume", "inspect"]:
    if failure == "missing-volume" and args[-1].endswith("postgres_data"):
        raise SystemExit(1)
    emit("[]")
    raise SystemExit(0)
if args and args[0] == "inspect":
    template = args[args.index("--format") + 1]
    container_id = args[-1]
    service = {{
        "{POSTGRES_ID}": "postgres",
        "{API_ID}": "api",
        "{WORKER_ID}": "worker",
        "{CADDY_ID}": "caddy",
    }}.get(container_id, "unknown")
    if "NetworkSettings.Ports" in template:
        binding = {{
            "80/tcp": [{{"HostIp": "127.0.0.1", "HostPort": "18080"}}],
            "443/tcp": [{{"HostIp": "127.0.0.1", "HostPort": "18443"}}],
        }}
        if failure == "binding-mismatch" and current["phase"] == "initial":
            binding["80/tcp"][0]["HostPort"] = "28080"
        if failure == "single-http-binding":
            binding.pop("443/tcp")
        if failure == "unsafe-binding":
            binding["80/tcp"][0]["HostIp"] = "0.0.0.0"
            binding["443/tcp"][0]["HostIp"] = "0.0.0.0"
        if failure == "binding-drift" and current["phase"] == "caddy":
            binding["80/tcp"][0]["HostPort"] = "28080"
        emit(json.dumps(binding, separators=(",", ":")))
    elif template == "{{{{.Config.Image}}}}" and service == "postgres":
        emit("{POSTGRES_IMAGE}")
    else:
        health = "" if service == "caddy" else "healthy"
        if failure == f"unhealthy-{{service}}":
            health = "unhealthy"
        image_ref = "{POSTGRES_IMAGE}" if service == "postgres" else "{APPLICATION_IMAGE}"
        image_id = "sha256:" + "c" * 64 if service == "postgres" else "{APPLICATION_IMAGE_ID}"
        if failure == "image-mismatch" and service == "worker":
            image_id = "sha256:" + "d" * 64
        emit(f"running|{{health}}|{{image_ref}}|{{image_id}}")
    raise SystemExit(0)

if not args or args[0] != "compose":
    raise SystemExit(97)

operations = {{"config", "ps", "exec", "run", "stop", "down", "up"}}
operation_index = next((index for index, value in enumerate(args) if value in operations), None)
if operation_index is None:
    raise SystemExit(97)
command = args[operation_index:]

if command[:2] == ["config", "--quiet"]:
    raise SystemExit(0)
if command[:3] == ["config", "--format", "json"]:
    host_ip = "0.0.0.0" if failure == "unsafe-binding" else "127.0.0.1"
    services = {{
        "postgres": {{"image": "{POSTGRES_IMAGE}"}},
        "api": {{"image": "{APPLICATION_IMAGE}"}},
        "worker": {{"image": "{APPLICATION_IMAGE}"}},
        "caddy": {{
            "image": "caddy:2.10.2",
            "ports": [
                {{"target": 80, "published": "18080", "host_ip": host_ip, "protocol": "tcp"}},
                {{"target": 443, "published": "18443", "host_ip": host_ip, "protocol": "tcp"}},
            ],
        }},
    }}
    if failure == "single-http-binding":
        services["caddy"]["ports"] = services["caddy"]["ports"][:1]
    if failure == "partial-project":
        services.pop("worker")
    if failure == "candidate-mismatch":
        services["api"]["image"] = "pathfinder:other"
    emit(json.dumps({{
        "services": services,
        "volumes": {{"postgres_data": {{}}, "caddy_data": {{}}, "caddy_config": {{}}}},
    }}, separators=(",", ":")))
    raise SystemExit(0)

if command[0] == "ps":
    service = command[-1] if command[-1] in {{"postgres", "api", "worker", "caddy"}} else None
    if "--status" in command and service == "worker":
        if failure == "slow-worker-stop" and current["phase"] == "worker-stopped":
            polls = current.get("worker_stop_polls", 0)
            current["worker_stop_polls"] = polls + 1
            save(current)
            if polls == 0:
                emit("{WORKER_ID}")
            raise SystemExit(0)
        if current["phase"] not in {{"worker-stopped", "down"}}:
            emit("{WORKER_ID}")
        raise SystemExit(0)
    if service is None:
        if current["phase"] != "down":
            for value in ("{POSTGRES_ID}", "{API_ID}", "{WORKER_ID}", "{CADDY_ID}"):
                emit(value)
        raise SystemExit(0)
    ids = {{
        "postgres": "{POSTGRES_ID}",
        "api": "{API_ID}",
        "worker": "{WORKER_ID}",
        "caddy": "{CADDY_ID}",
    }}
    if failure == "partial-project" and service == "worker":
        raise SystemExit(0)
    emit(ids[service])
    if failure == "multiple-worker" and service == "worker":
        emit("5" * 64)
    raise SystemExit(0)

if command[0] == "stop":
    current["phase"] = "worker-stopped"
    save(current)
    raise SystemExit(0)

if command[0] == "down":
    if failure == "signal-down":
        os.kill(os.getppid(), signal.SIGTERM)
        raise SystemExit(143)
    if failure == "down":
        raise SystemExit(1)
    current["phase"] = "down"
    save(current)
    raise SystemExit(0)

if command[0] == "up":
    service = command[-1]
    if failure == f"up-{{service}}":
        raise SystemExit(1)
    current["phase"] = service
    save(current)
    raise SystemExit(0)

if command[0] == "run":
    if failure == "wrong-claim":
        run_id = "00000000-0000-4000-8000-000000000000"
    else:
        run_id = "{LEASED_RUN_ID}"
    emit(json.dumps({{
        "run_id": run_id,
        "attempt": 1,
        "lease_expires_at": "2099-01-01T00:00:00+00:00",
    }}))
    raise SystemExit(0)

if command[0] != "exec":
    raise SystemExit(97)
service = command[2]
joined = " ".join(command)

if service == "worker" and "from app.config import Settings" in joined:
    value = ["fake", "fake", "fake", "off"]
    if failure == "non-fake":
        value[0] = "qwen"
    emit(json.dumps(value))
    raise SystemExit(0)

if service == "postgres":
    if command[-2:] == ["postgres", "--version"]:
        emit("postgres (PostgreSQL) 16.10")
        raise SystemExit(0)
    if "df -Pk" in joined:
        emit("Filesystem 1024-blocks Used Available Capacity Mounted on")
        emit("/dev/fake 100000 25000 75000 25% /var/lib/postgresql/data")
        raise SystemExit(0)
    sql = sys.stdin.read()
    if "SELECT version_num FROM alembic_version" in joined:
        emit("{MIGRATION_REVISION}")
    elif "active_runs" in joined:
        active = 1 if failure == "active-state" else 0
        emit(json.dumps({{"active_runs": active, "active_jobs": active}}, separators=(",", ":")))
    elif "SELECT id FROM action_intents" in joined:
        emit("{BASELINE_ACTION_ID}" if "{BASELINE_RUN_ID}" in joined else "{WAITING_ACTION_ID}")
    elif "baseline_run_id=" in joined:
        if current["phase"] in {{"initial", "worker-stopped"}}:
            raw = os.environ["FAKE_PRE_SNAPSHOT"]
            snapshot_phase = "pre"
        elif current["phase"] in {{"postgres", "api", "caddy", "worker"}}:
            raw = os.environ["FAKE_POST_SNAPSHOT"]
            snapshot_phase = "post"
        else:
            raw = os.environ["FAKE_FINAL_SNAPSHOT"]
            snapshot_phase = "final"
        if current.get("waiting_approved"):
            raw = os.environ["FAKE_FINAL_SNAPSHOT"]
            snapshot_phase = "final"
        value = json.loads(raw)
        if snapshot_phase == "pre":
            value["fixtures"]["leased"]["job"]["lease_expires_at"] = (
                dt.datetime.now(dt.UTC) + dt.timedelta(seconds=5)
            ).isoformat()
        value["fixtures"]["leased"]["job"]["leased_by"] = os.environ["LEASED_BY"]
        if failure == "invalid-leased" and snapshot_phase == "pre":
            value["fixtures"]["leased"]["job"]["attempt"] = 0
        if failure == "invalid-queued" and snapshot_phase == "pre":
            value["fixtures"]["queued"]["job"]["attempt"] = 1
        if failure == "missing-waiting-checkpoint" and snapshot_phase == "pre":
            value["fixtures"]["waiting"]["checkpoints"] = []
        if failure == "baseline-mock-count" and snapshot_phase == "pre":
            value["fixtures"]["baseline"]["mock_submissions"] = []
        if failure == "pre-check" and snapshot_phase == "pre":
            value["checks"]["events_contiguous"] = False
        if failure == "post-mismatch" and snapshot_phase == "post":
            value["fixtures"]["queued"]["job"]["attempt"] = 1
        if (
            failure == "waiting-changed"
            and snapshot_phase == "post"
            and current["phase"] == "worker"
        ):
            value["fixtures"]["waiting"]["run"]["status"] = "running"
        if (
            failure == "prefix-mismatch"
            and snapshot_phase == "post"
            and current["phase"] == "worker"
        ):
            value["fixtures"]["waiting"]["events"][0]["type"] = "changed"
        if failure == "missing-expiry" and snapshot_phase == "final":
            value["fixtures"]["leased"]["events"] = [value["fixtures"]["leased"]["events"][0]]
        if failure == "leased-attempt" and snapshot_phase == "final":
            value["fixtures"]["leased"]["job"]["attempt"] = 1
        if failure == "baseline-duplicate" and snapshot_phase == "final":
            value["fixtures"]["baseline"]["mock_submissions"].append(
                value["fixtures"]["baseline"]["mock_submissions"][0]
            )
        if failure == "duplicate-llm" and snapshot_phase == "final":
            value["fixtures"]["waiting"]["llm_invocations"].append({{"id": "duplicate"}})
        if failure == "final-mock" and snapshot_phase == "final":
            value["fixtures"]["waiting"]["mock_submissions"] = []
        emit(json.dumps(value, sort_keys=True, separators=(",", ":")))
    else:
        emit(os.environ["FAKE_METRICS"])
    raise SystemExit(0)

if service == "api" and command[3:5] == ["alembic", "heads"]:
    emit("{MIGRATION_REVISION} (head)")
    raise SystemExit(0)
if service == "api" and "document_path=" in joined:
    emit("document_id={DOCUMENT_ID}")
    raise SystemExit(0)
if service == "api" and "readyz" in joined:
    raise SystemExit(1 if failure == "api-readiness" else 0)
if service == "api" and "terminal SSE returned no event IDs" in joined:
    if failure == "sse-duplicate":
        raise SystemExit(1)
    emit(json.dumps({{"last_event_id": 4, "replayed_ids": []}}, separators=(",", ":")))
    raise SystemExit(0)
if service != "api" or "urllib.request.Request" not in joined:
    raise SystemExit(97)

method, path, body = command[-3:]
if method == "GET" and path == "/api/v1/me":
    emit(json.dumps({{
        "user_id": "{ACTOR_ID}",
        "workspaces": [{{"workspace_id": "{WORKSPACE_ID}", "kind": "personal", "role": "admin"}}],
    }}))
elif method == "POST" and path.endswith("/runs"):
    if "baseline" in body:
        run_id = "{BASELINE_RUN_ID}"
    elif "waiting" in body:
        run_id = "{WAITING_RUN_ID}"
    elif "leased" in body:
        run_id = "{LEASED_RUN_ID}"
    else:
        run_id = "{QUEUED_RUN_ID}"
    emit(json.dumps({{"run_id": run_id, "status": "queued"}}))
elif method == "GET" and "/runs/" in path:
    run_id = path.rsplit("/", 1)[1]
    if run_id == "{BASELINE_RUN_ID}":
        status = "completed" if current.get("baseline_approved") else "waiting_approval"
    elif run_id == "{WAITING_RUN_ID}":
        status = "completed" if current.get("waiting_approved") else "waiting_approval"
    elif failure == "queue-timeout" and run_id == "{QUEUED_RUN_ID}":
        status = "failed"
    else:
        status = "completed" if current["phase"] == "worker" else "queued"
    emit(json.dumps({{"run_id": run_id, "status": status}}))
elif method == "GET" and "/action-intents/" in path:
    action_id = path.rsplit("/", 1)[1]
    approved = (
        current.get("baseline_approved")
        if action_id == "{BASELINE_ACTION_ID}"
        else current.get("waiting_approved")
    )
    emit(json.dumps({{
        "approval_request": {{"version": 1}},
        "decision": {{"decision": "approve"}} if approved else None,
    }}))
elif method == "POST" and path.endswith("/decision"):
    action_id = path.split("/")[-2]
    if action_id == "{BASELINE_ACTION_ID}":
        current["baseline_approved"] = True
    else:
        current["waiting_approved"] = True
        current["phase"] = "final"
    save(current)
    if failure == "decision-uncertain" and action_id == "{WAITING_ACTION_ID}":
        raise SystemExit(1)
    emit(json.dumps({{"decision": "approve"}}))
else:
    raise SystemExit(95)
raise SystemExit(0)
'''


def _write_fake_tools(fake_bin: Path) -> None:
    tool = fake_bin / "fake-gate86-tool"
    tool.write_text(FAKE_TOOL, encoding="utf-8")
    tool.chmod(0o755)
    for name in ("docker", "date", "sleep", "stat"):
        (fake_bin / name).symlink_to(tool)


def _base_environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o700)
    locks = tmp_path / "locks"
    locks.mkdir(mode=0o700)
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n", encoding="utf-8")
    private = tmp_path / "compose.private.yaml"
    private.write_text("services: {}\n", encoding="utf-8")
    pre, post, final = _snapshots()
    environment = os.environ.copy()
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": PROJECT,
            "FAKE_COMMAND_LOG": str(tmp_path / "commands.jsonl"),
            "FAKE_FINAL_SNAPSHOT": final,
            "FAKE_METRICS": METRICS,
            "FAKE_POST_SNAPSHOT": post,
            "FAKE_PRE_SNAPSHOT": pre,
            "FAKE_STATE": str(tmp_path / "state.json"),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PF_GATE86_COMPOSE_FILES": f"{compose}:{private}",
            "PF_GATE86_DEADLINE_SECONDS": "30",
            "PF_GATE86_EVIDENCE_DIR": str(evidence),
            "PF_GATE86_SHARED_HOST_OUTAGE_APPROVED": "1",
            "TMPDIR": str(locks),
        }
    )
    return environment


@pytest.fixture
def rehearsal_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_tools(fake_bin)
    environment = _base_environment(tmp_path, fake_bin)
    return environment, Path(environment["FAKE_COMMAND_LOG"])


def _run(
    environment: dict[str, str], mode: str = "shared-host"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(REHEARSAL_SCRIPT), mode),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _flat(records: list[dict[str, Any]]) -> str:
    return "\n".join(f"{record['tool']} {' '.join(record['args'])}" for record in records)


def _contains_mutation(records: list[dict[str, Any]], verb: str, service: str = "") -> bool:
    return any(
        record["tool"] == "docker"
        and "compose" in record["args"]
        and verb in record["args"]
        and (not service or record["args"][-1] == service)
        for record in records
    )


def _manifest(environment: dict[str, str]) -> Path:
    paths = list(Path(environment["PF_GATE86_EVIDENCE_DIR"]).glob("*/manifest.json"))
    assert len(paths) == 1
    return paths[0]


def test_successful_command_order_scope_and_manifest(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment

    result = _run(environment)

    assert result.returncode == 0, result.stderr
    records = _records(command_log)
    down = next(index for index, record in enumerate(records) if "down" in record["args"])
    postgres = next(
        index
        for index, record in enumerate(records)
        if "up" in record["args"] and record["args"][-1] == "postgres"
    )
    api = next(
        index
        for index, record in enumerate(records)
        if "up" in record["args"] and record["args"][-1] == "api"
    )
    caddy = next(
        index
        for index, record in enumerate(records)
        if "up" in record["args"] and record["args"][-1] == "caddy"
    )
    worker = next(
        index
        for index, record in enumerate(records)
        if "up" in record["args"] and record["args"][-1] == "worker"
    )
    decision = next(
        index
        for index, record in enumerate(records)
        if any(str(value).endswith("/decision") for value in record["args"])
        and WAITING_ACTION_ID in " ".join(record["args"])
    )
    assert down < postgres < api < caddy < worker < decision
    assert all(
        "--project-name" in record["args"]
        and record["args"][record["args"].index("--project-name") + 1] == PROJECT
        and record["args"].count("--file") == 2
        for record in records
        if record["args"] and record["args"][0] == "compose"
    )
    manifest_path = _manifest(environment)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["result"] == "PASS"
    assert manifest["scope"]["application_compose_recovery"] == "VERIFIED"
    assert manifest["scope"]["whole_host_recovery"] == "NOT_VERIFIED"
    assert manifest["fixtures"]["leased"]["job_attempt"] == 2
    assert manifest["fixtures"]["queued"]["job_attempt"] == 1
    assert "mkdir --mode 0700" in REHEARSAL_SCRIPT.read_text(encoding="utf-8")
    assert "chmod 0600" in REHEARSAL_SCRIPT.read_text(encoding="utf-8")
    assert "whole-host recovery=NOT VERIFIED" in result.stdout


def test_metrics_mode_is_read_only_and_emits_storage(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment

    result = _run(environment, "metrics")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["storage"]["filesystem"]["percent_used"] == 25
    records = _records(command_log)
    assert not any(
        any(verb in record["args"] for verb in ("up", "down", "stop", "run")) for record in records
    )


def test_single_loopback_http_binding_is_a_safe_current_deployment(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, _ = rehearsal_environment
    environment["FAKE_FAIL"] = "single-http-binding"

    result = _run(environment)

    assert result.returncode == 0, result.stderr


def test_worker_stop_waits_for_compose_status_convergence(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, _ = rehearsal_environment
    environment["FAKE_FAIL"] = "slow-worker-stop"

    result = _run(environment)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("COMPOSE_PROJECT_NAME", "COMPOSE_PROJECT_NAME is required"),
        ("PF_GATE86_EVIDENCE_DIR", "PF_GATE86_EVIDENCE_DIR is required"),
        ("PF_GATE86_SHARED_HOST_OUTAGE_APPROVED", "is required"),
    ],
)
def test_missing_required_scope_fails_before_mutation(
    rehearsal_environment: tuple[dict[str, str], Path], missing: str, message: str
) -> None:
    environment, command_log = rehearsal_environment
    environment.pop(missing)

    result = _run(environment)

    assert result.returncode != 0
    assert message in result.stderr
    records = _records(command_log)
    assert not any(_contains_mutation(records, verb) for verb in ("stop", "down", "up", "run"))


@pytest.mark.parametrize(
    "failure",
    [
        "partial-project",
        "multiple-worker",
        "unhealthy-postgres",
        "unhealthy-api",
        "unhealthy-worker",
        "non-fake",
        "active-state",
        "binding-mismatch",
        "unsafe-binding",
        "candidate-mismatch",
        "image-mismatch",
    ],
)
def test_preflight_failure_prevents_fixture_or_outage(
    rehearsal_environment: tuple[dict[str, str], Path], failure: str
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert not _contains_mutation(records, "stop")
    assert not _contains_mutation(records, "down")


def test_wrong_claimed_run_prevents_queued_fixture_and_outage(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = "wrong-claim"

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert not _contains_mutation(records, "down")
    assert QUEUED_RUN_ID not in _flat(records)


@pytest.mark.parametrize(
    "failure",
    [
        "invalid-leased",
        "invalid-queued",
        "missing-waiting-checkpoint",
        "baseline-mock-count",
        "pre-check",
    ],
)
def test_invalid_pre_snapshot_prevents_outage(
    rehearsal_environment: tuple[dict[str, str], Path], failure: str
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run(environment)

    assert result.returncode != 0
    assert not _contains_mutation(_records(command_log), "down")


def test_down_failure_starts_no_recovery(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = "down"

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert _contains_mutation(records, "down")
    assert not _contains_mutation(records, "up")


@pytest.mark.parametrize("failure", ["missing-volume", "up-postgres"])
def test_volume_or_postgres_failure_prevents_later_services(
    rehearsal_environment: tuple[dict[str, str], Path], failure: str
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert not _contains_mutation(records, "up", "api")
    assert not _contains_mutation(records, "up", "caddy")
    assert not _contains_mutation(records, "up", "worker")


def test_post_storage_mismatch_prevents_api_and_worker(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = "post-mismatch"

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert not _contains_mutation(records, "up", "api")
    assert not _contains_mutation(records, "up", "worker")


@pytest.mark.parametrize(
    ("failure", "forbidden_service"),
    [
        ("up-api", "worker"),
        ("api-readiness", "worker"),
        ("binding-drift", "worker"),
        ("up-worker", "decision"),
    ],
)
def test_service_barrier_failure_prevents_later_mutation(
    rehearsal_environment: tuple[dict[str, str], Path],
    failure: str,
    forbidden_service: str,
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run(environment)

    assert result.returncode != 0
    records = _records(command_log)
    if forbidden_service == "decision":
        assert f"/action-intents/{WAITING_ACTION_ID}/decision" not in _flat(records)
    else:
        assert not _contains_mutation(records, "up", forbidden_service)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-expiry",
        "leased-attempt",
        "queue-timeout",
        "baseline-duplicate",
        "waiting-changed",
        "prefix-mismatch",
        "duplicate-llm",
        "final-mock",
        "sse-duplicate",
    ],
)
def test_recovery_assertion_failure_never_writes_pass_manifest(
    rehearsal_environment: tuple[dict[str, str], Path], failure: str
) -> None:
    environment, _ = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run(environment)

    assert result.returncode != 0
    assert not list(Path(environment["PF_GATE86_EVIDENCE_DIR"]).glob("*/manifest.json"))
    assert "application/Compose recovery=VERIFIED" not in result.stdout


def test_uncertain_decision_reads_back_without_blind_retry(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = "decision-uncertain"

    result = _run(environment)

    assert result.returncode == 0, result.stderr
    records = _records(command_log)
    waiting_decisions = [
        record
        for record in records
        if record["tool"] == "docker"
        and f"/action-intents/{WAITING_ACTION_ID}/decision" in " ".join(record["args"])
    ]
    waiting_reads = [
        record
        for record in records
        if record["tool"] == "docker"
        and f"/action-intents/{WAITING_ACTION_ID}" in " ".join(record["args"])
        and "/decision" not in " ".join(record["args"])
    ]
    assert len(waiting_decisions) == 1
    assert len(waiting_reads) >= 2


def test_signal_during_outage_exits_130_and_starts_no_recovery(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = "signal-down"

    result = _run(environment)

    assert result.returncode == 130
    assert "interrupted by SIGTERM during stage=controlled-full-stack-outage" in result.stderr
    assert not _contains_mutation(_records(command_log), "up")


def test_secret_canary_is_absent_from_output_evidence_and_command_log(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    canary = f"gate86-secret-{secrets.token_hex(24)}"
    environment.update(
        {
            "DASHSCOPE_API_KEY": canary,
            "LANGFUSE_PUBLIC_KEY": canary,
            "LANGFUSE_SECRET_KEY": canary,
            "PF_DATABASE_URL": f"postgresql://user:{canary}@example.invalid/db",
            "PF_POSTGRES_PASSWORD": canary,
            "PF_QWEN_WORKSPACE_ID": canary,
            "TAVILY_API_KEY": canary,
        }
    )

    result = _run(environment)

    assert result.returncode == 0, result.stderr
    assert canary not in result.stdout
    assert canary not in result.stderr
    assert canary not in command_log.read_text(encoding="utf-8")
    for path in Path(environment["PF_GATE86_EVIDENCE_DIR"]).glob("*/*"):
        assert canary not in path.read_text(encoding="utf-8")


def test_script_static_safety_and_sql_redaction_contract() -> None:
    source = REHEARSAL_SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "sudo reboot",
        "shutdown",
        "systemctl restart docker",
        "docker system prune",
        "docker volume rm",
        "down -v",
        "down --volumes",
        "alembic downgrade",
        "rm -rf",
    )
    for value in forbidden:
        assert value not in source
    assert "compose down --timeout 30" in source
    assert "docker volume inspect" in source
    assert '"application_compose_recovery":"VERIFIED"' in source
    assert '"whole_host_recovery":"NOT_VERIFIED"' in source

    metrics = METRICS_SQL.read_text(encoding="utf-8")
    recovery = RECOVERY_SQL.read_text(encoding="utf-8")
    assert "REPEATABLE READ, READ ONLY" in metrics
    assert "REPEATABLE READ, READ ONLY" in recovery
    for sensitive in (
        "args_snapshot",
        "target_snapshot",
        "document.content",
        "chunk.text",
        "invocation.input",
        "invocation.output",
        "checkpoint_data",
    ):
        assert sensitive not in recovery
