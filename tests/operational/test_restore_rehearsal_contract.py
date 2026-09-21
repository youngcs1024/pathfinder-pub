from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REHEARSAL_SCRIPT = PROJECT_ROOT / "scripts" / "restore_rehearsal.sh"
CONSISTENCY_SQL = PROJECT_ROOT / "scripts" / "gate85_consistency.sql"

SOURCE_PROJECT = "pathfinder-source-contract"
RESTORE_PROJECT = "pathfinder-restore-contract"
SOURCE_RUN_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_DOCUMENT_ID = "22222222-2222-4222-8222-222222222222"
E2E_RUN_ID = "33333333-3333-4333-8333-333333333333"
E2E_ACTION_ID = "44444444-4444-4444-8444-444444444444"
ACTOR_ID = "55555555-5555-4555-8555-555555555555"
WORKSPACE_ID = "66666666-6666-4666-8666-666666666666"
REQUEST_ID = "77777777-7777-4777-8777-777777777777"
DUMP_CONTENT = b"PGDMP-gate85-contract-artifact"
DUMP_SHA256 = hashlib.sha256(DUMP_CONTENT).hexdigest()
IMAGE_ID = f"sha256:{'a' * 64}"
POSTGRES_ID = "1" * 64
API_ID = "2" * 64
WORKER_ID = "3" * 64
CADDY_ID = "4" * 64


def _snapshot(*, run_id: str, document_id: str, global_runs: int = 1) -> dict[str, object]:
    return {
        "schema": {
            "alembic_version": "0015_e3_run_request_identity",
            "public_tables": [{"name": "runs", "present": True}],
            "checkpoint_tables": [{"name": "checkpoints", "present": True}],
            "runs_has_thread_id": False,
        },
        "global_counts": {
            "users": 1,
            "workspaces": 1,
            "workspace_memberships": 1,
            "documents": 1,
            "document_chunks": 1,
            "runs": global_runs,
            "run_jobs": global_runs,
            "run_events": 12,
            "action_intents": global_runs,
            "approval_requests": global_runs,
            "approval_decisions": global_runs,
            "tool_invocations": global_runs,
            "mock_submissions": global_runs,
            "llm_invocations": global_runs,
            "checkpoints": global_runs,
        },
        "fixture": {
            "workspace": {
                "id": WORKSPACE_ID,
                "kind": "personal",
                "created_by_user_id": ACTOR_ID,
            },
            "memberships": [{"user_id": ACTOR_ID, "role": "admin", "revoked": False}],
            "document": {
                "id": document_id,
                "workspace_id": WORKSPACE_ID,
                "created_by_user_id": ACTOR_ID,
                "content_hash": "b" * 64,
                "normalization_version": "text-v1",
                "chunking_version": "section-v1",
                "embedding_model": "qwen-text-embedding-v4-1536-v1",
                "chunk_count": 1,
                "chunks": [
                    {
                        "id": "88888888-8888-4888-8888-888888888888",
                        "ordinal": 0,
                        "content_hash": "c" * 64,
                        "token_count": 20,
                        "embedding_model": "qwen-text-embedding-v4-1536-v1",
                    }
                ],
            },
            "run": {
                "id": run_id,
                "workspace_id": WORKSPACE_ID,
                "created_by_user_id": ACTOR_ID,
                "resume_document_id": document_id,
                "status": "completed",
                "graph_version": "pathfinder-research-v6",
                "next_event_seq": 13,
            },
            "job": {"status": "done"},
            "events": [{"seq": index, "type": "run.status_changed"} for index in range(1, 13)],
            "action_intents": [
                {
                    "id": E2E_ACTION_ID,
                    "status": "succeeded",
                    "args_digest": f"sha256:{'d' * 64}",
                    "target_digest": f"sha256:{'e' * 64}",
                    "approval_binding_digest": f"sha256:{'f' * 64}",
                    "idempotency_key": E2E_ACTION_ID,
                }
            ],
            "approval_requests": [{"id": REQUEST_ID, "status": "consumed"}],
            "approval_decisions": [{"decision": "approve", "actor_user_id": ACTOR_ID}],
            "mock_submissions": [
                {
                    "action_intent_id": E2E_ACTION_ID,
                    "idempotency_key": E2E_ACTION_ID,
                    "payload_digest": f"sha256:{'9' * 64}",
                }
            ],
            "tool_invocations": [{"provider": "fake", "status": "succeeded"}],
            "llm_invocations": [{"provider": "fake", "status": "succeeded"}],
            "checkpoint_count": 2,
        },
        "checks": {
            "all_public_tables_present": True,
            "all_checkpoint_tables_present": True,
            "runs_has_no_thread_id": True,
            "run_request_identity_schema": True,
            "fixture_run_exactly_one": True,
            "fixture_document_exactly_one": True,
            "fixture_document_matches_run": True,
            "fixture_document_has_chunks": True,
            "document_representation_identities_unique": True,
            "fixture_run_completed": True,
            "fixture_job_done": True,
            "fixture_events_contiguous": True,
            "fixture_next_event_seq_correct": True,
            "fixture_action_succeeded": True,
            "fixture_request_consumed": True,
            "fixture_decision_approved": True,
            "fixture_binding_consistent": True,
            "fixture_actor_audit_consistent": True,
            "fixture_mock_exactly_once": True,
            "fixture_irreversible_invocation_bound": True,
            "fixture_llm_invocations_fake_and_terminal": True,
            "fixture_checkpoint_present": True,
            "source_is_quiescent": True,
        },
    }


SOURCE_SNAPSHOT = json.dumps(
    _snapshot(run_id=SOURCE_RUN_ID, document_id=SOURCE_DOCUMENT_ID),
    sort_keys=True,
    separators=(",", ":"),
)
E2E_SNAPSHOT = json.dumps(
    _snapshot(run_id=E2E_RUN_ID, document_id=SOURCE_DOCUMENT_ID, global_runs=2),
    sort_keys=True,
    separators=(",", ":"),
)


FAKE_TOOL = f'''#!/usr/bin/env python3
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

tool = Path(sys.argv[0]).name
args = sys.argv[1:]
log_path = Path(os.environ["FAKE_COMMAND_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"tool": tool, "args": args}}, separators=(",", ":")) + "\\n")

failure = os.environ.get("FAKE_FAIL", "")
source_snapshot = os.environ["FAKE_SOURCE_SNAPSHOT"]
e2e_snapshot = os.environ["FAKE_E2E_SNAPSHOT"]

def emit(value):
    sys.stdout.write(str(value) + "\\n")

def prior_log():
    return log_path.read_text(encoding="utf-8")

if tool == "ssh":
    if failure == "remote-unreachable":
        raise SystemExit(255)
    remote = args[2:]
    if "gate85-remote-preflight" in remote:
        sys.stdin.read()
        if failure in {{"partial-project", "multiple-worker"}}:
            raise SystemExit(93)
        health = {{
            "postgres": "unhealthy" if failure == "unhealthy-postgres" else "healthy",
            "api": "unhealthy" if failure == "unhealthy-api" else "healthy",
            "worker": "unhealthy" if failure == "unhealthy-worker" else "healthy",
            "caddy": "",
        }}
        for service, container_id in (
            ("postgres", "{POSTGRES_ID}"),
            ("api", "{API_ID}"),
            ("worker", "{WORKER_ID}"),
            ("caddy", "{CADDY_ID}"),
        ):
            emit(f"service\\t{{service}}\\t{{container_id}}\\trunning\\t{{health[service]}}")
        emit("metadata\\tpostgres_image\\tpgvector/pgvector:0.8.5-pg16@sha256:" + "6" * 64)
        emit("metadata\\tpostgres_version\\tpostgres (PostgreSQL) 16.10")
        raise SystemExit(0)
    if "gate85-fixture-document" in remote:
        sys.stdin.read()
        emit("{SOURCE_DOCUMENT_ID}")
        raise SystemExit(0)
    if remote[:3] == ["docker", "ps", "--all"]:
        service_filter = next(
            value
            for value in remote
            if value.startswith("label=com.docker.compose.service=")
        )
        service = service_filter.rsplit("=", 1)[1]
        state = failure
        ids = {{
            "postgres": "{POSTGRES_ID}",
            "api": "{API_ID}",
            "worker": "{WORKER_ID}",
            "caddy": "{CADDY_ID}",
        }}
        if state == "partial-project" and service == "worker":
            raise SystemExit(0)
        emit(ids[service])
        if state == "multiple-worker" and service == "worker":
            emit("5" * 64)
        raise SystemExit(0)
    if remote[:2] == ["docker", "inspect"]:
        template = remote[remote.index("--format") + 1]
        container_id = remote[-1]
        if ".Config.Image" in template:
            emit("pgvector/pgvector:0.8.5-pg16@sha256:" + "6" * 64)
        else:
            services = {{
                "{POSTGRES_ID}": "postgres",
                "{API_ID}": "api",
                "{WORKER_ID}": "worker",
                "{CADDY_ID}": "caddy",
            }}
            service = services[container_id]
            health = "unhealthy" if failure == f"unhealthy-{{service}}" else "healthy"
            emit(f"running|{{health}}")
        raise SystemExit(0)
    if (
        remote[:3] == ["docker", "exec", "{POSTGRES_ID}"]
        and remote[-2:] == ["postgres", "--version"]
    ):
        emit("postgres (PostgreSQL) 16.10")
        raise SystemExit(0)
    if (
        remote[:3] == ["docker", "exec", "{POSTGRES_ID}"]
        and "resume_document_id" in " ".join(remote)
    ):
        emit("{SOURCE_DOCUMENT_ID}")
        raise SystemExit(0)
    if remote[:3] == ["docker", "exec", "-i"] and "psql" in remote:
        sys.stdin.read()
        snapshot_calls = sum(
            record["tool"] == "ssh"
            and "-i" in record["args"]
            and "psql" in record["args"]
            for record in map(json.loads, prior_log().splitlines())
        )
        value = json.loads(source_snapshot)
        if failure == "invalid-fixture":
            value["checks"]["fixture_run_completed"] = False
        if failure == "snapshot-changed" and snapshot_calls >= 2:
            value["global_counts"]["runs"] += 1
        emit(json.dumps(value, sort_keys=True, separators=(",", ":")))
        raise SystemExit(0)
    if "gate85-remote-dump" in remote:
        sys.stdin.read()
        if failure in {{"remote-dump", "invalid-dump"}}:
            raise SystemExit(97)
        staging = remote[-3]
        dump_name = remote[-2]
        path = f"{{staging}}/{{dump_name}}"
        emit("{DUMP_SHA256}\\t{len(DUMP_CONTENT)}\\t600\\t" + path)
        raise SystemExit(0)
    if "gate85-remote-staging-cleanup" in remote:
        sys.stdin.read()
        raise SystemExit(1 if failure == "remote-cleanup" else 0)
    raise SystemExit(97)

if tool == "scp":
    destination = Path(args[-1])
    content = b"{DUMP_CONTENT.decode()}"
    if failure == "checksum-mismatch":
        content += b"-corrupt"
    destination.write_bytes(content)
    raise SystemExit(1 if failure == "scp" else 0)

if tool != "docker":
    raise SystemExit(98)

if args[:3] == ["ps", "--all", "--quiet"]:
    if failure == "existing-restore-project" or '"up","--detach"' in prior_log():
        if "service=worker" in " ".join(args):
            emit("9" * 64)
        else:
            emit("7" * 64)
            emit("8" * 64)
            emit("9" * 64)
    raise SystemExit(0)

if args[:3] == ["volume", "ls", "--quiet"]:
    if failure == "existing-restore-project" or '"up","--detach"' in prior_log():
        emit("{RESTORE_PROJECT}_pathfinder_postgres_data")
    raise SystemExit(0)

if args[:3] == ["network", "ls", "--quiet"]:
    if failure == "existing-restore-project" or '"up","--detach"' in prior_log():
        emit("{RESTORE_PROJECT}_backend")
    raise SystemExit(0)

if args[:2] == ["image", "inspect"]:
    emit("{IMAGE_ID}")
    raise SystemExit(0)

if args and args[0] == "build":
    raise SystemExit(1 if failure == "image-build" else 0)

if args and args[0] == "inspect":
    container_id = args[-1]
    names = {{"7" * 64: "postgres", "8" * 64: "api", "9" * 64: "worker"}}
    stopped = any(
        record["tool"] == "docker" and "stop" in record["args"]
        for record in map(json.loads, prior_log().splitlines())
    )
    state = "exited" if stopped else "running"
    emit(f"{{container_id}}|/{RESTORE_PROJECT}-{{names.get(container_id, 'worker')}}-1|{{state}}")
    raise SystemExit(0)

if args[:1] != ["compose"]:
    raise SystemExit(97)

command = args[args.index("--file") + 2:]
if command[:2] == ["config", "--quiet"]:
    raise SystemExit(0)
if command[:6] == ["run", "--rm", "--no-deps", "--entrypoint", "python", "api"]:
    raise SystemExit(0)
if command[:6] == ["run", "--rm", "--no-deps", "--entrypoint", "python", "worker"]:
    raise SystemExit(0)
if command[:6] == ["run", "--rm", "--no-deps", "--entrypoint", "alembic", "api"]:
    emit("0015_e3_run_request_identity (head)")
    raise SystemExit(0)
if command[:2] == ["up", "--detach"]:
    service = command[-1]
    raise SystemExit(1 if failure == f"up-{{service}}" else 0)
if command[:5] == ["run", "--rm", "--no-deps", "api", "alembic"]:
    raise SystemExit(1 if failure == "migration" else 0)
if command[:2] == ["stop", "--timeout"]:
    raise SystemExit(0)
if command[:3] != ["exec", "-T", command[2]]:
    raise SystemExit(97)

service = command[2]
joined = " ".join(command)
if service == "postgres" and "pg_restore --exit-on-error" in joined:
    if os.environ.get("FAKE_BLOCK") == "pg-restore":
        Path(os.environ["FAKE_NOTIFY"]).write_text("pg-restore", encoding="utf-8")
        signal.pause()
    sys.stdin.buffer.read()
    raise SystemExit(1 if failure == "pg-restore" else 0)
if service == "postgres" and "SELECT version_num FROM alembic_version" in joined:
    emit("0015_e3_run_request_identity")
    raise SystemExit(0)
if service == "postgres" and "psql" in command and "--set" in command:
    sql = sys.stdin.read()
    normalized_sql = " ".join(sql.split())
    if normalized_sql.startswith("SELECT id FROM action_intents"):
        emit("{E2E_ACTION_ID}")
    elif normalized_sql.startswith("SELECT status FROM run_jobs"):
        emit("done")
    elif normalized_sql.startswith("SELECT count(*) FROM mock_submissions"):
        emit("1")
    else:
        run_setting = next(value for value in command if value.startswith("fixture_run_id="))
        run_id = run_setting.split("=", 1)[1]
        value = source_snapshot if run_id == "{SOURCE_RUN_ID}" else e2e_snapshot
        if failure == "consistency-mismatch" and run_id == "{SOURCE_RUN_ID}":
            parsed = json.loads(value)
            parsed["global_counts"]["runs"] += 1
            value = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        emit(value)
    raise SystemExit(0)
if service == "api" and "readyz" in joined:
    raise SystemExit(1 if failure == "api-readiness" else 0)
if service == "api" and "invalid initial SSE event sequence" in joined:
    emit('{{"last_event_id":12,"replayed_ids":[]}}')
    raise SystemExit(0)
if service == "api" and "urllib.request.Request" in joined:
    method, path, payload = command[-3:]
    if method == "GET" and path == "/api/v1/me":
        emit(json.dumps({{
            "user_id": "{ACTOR_ID}",
            "workspaces": [{{
                "workspace_id": "{WORKSPACE_ID}",
                "kind": "personal",
                "role": "admin",
                "name": "Personal",
            }}],
        }}))
    elif method == "POST" and path.endswith("/runs"):
        emit(json.dumps({{"run_id": "{E2E_RUN_ID}", "status": "queued", "events_url": "/events"}}))
    elif method == "GET" and path.endswith("/runs/{E2E_RUN_ID}"):
        status = "completed" if "/decision" in prior_log() else "waiting_approval"
        emit(json.dumps({{"run_id": "{E2E_RUN_ID}", "status": status}}))
    elif method == "GET" and "/action-intents/{E2E_ACTION_ID}" in path:
        decision = {{"decision": "approve"}} if "/decision" in prior_log() else None
        emit(json.dumps({{"approval_request": {{"version": 1}}, "decision": decision}}))
    elif method == "POST" and path.endswith("/decision"):
        if failure == "decision-uncertain":
            raise SystemExit(1)
        emit(json.dumps({{"decision": "approve"}}))
    else:
        raise SystemExit(95)
    raise SystemExit(0)
raise SystemExit(97)
'''


def _write_fake_tools(fake_bin: Path) -> None:
    for name in ("docker", "scp", "ssh"):
        path = fake_bin / name
        path.write_text(FAKE_TOOL, encoding="utf-8")
        path.chmod(0o755)


def _base_environment(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o700)
    lock_root = tmp_path / "locks"
    lock_root.mkdir(mode=0o700)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_COMMAND_LOG": str(tmp_path / "commands.jsonl"),
            "FAKE_E2E_SNAPSHOT": E2E_SNAPSHOT,
            "FAKE_SOURCE_SNAPSHOT": SOURCE_SNAPSHOT,
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PATHFINDER_IMAGE": "pathfinder-gate85:contract",
            "PF_GATE85_BACKUP_DIR": str(backup_dir),
            "PF_GATE85_DEADLINE_SECONDS": "30",
            "PF_GATE85_FIXTURE_RUN_ID": SOURCE_RUN_ID,
            "PF_GATE85_RESTORE_PROJECT": RESTORE_PROJECT,
            "PF_GATE85_SOURCE_PROJECT": SOURCE_PROJECT,
            "PF_GATE85_SOURCE_SSH": "source-test",
            "TMPDIR": str(lock_root),
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


def _run_rehearsal(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(REHEARSAL_SCRIPT),),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _index(records: list[dict[str, object]], predicate: object) -> int:
    assert callable(predicate)
    for index, record in enumerate(records):
        if predicate(record):
            return index
    raise AssertionError(f"record not found in {records!r}")


def _flat(records: list[dict[str, object]]) -> str:
    return "\n".join(f"{record['tool']} {' '.join(record['args'])}" for record in records)


def _has_compose_service(records: list[dict[str, object]], verb: str, service: str) -> bool:
    return any(
        record["tool"] == "docker"
        and "compose" in record["args"]
        and verb in record["args"]
        and record["args"][-1] == service
        for record in records
    )


def test_success_orders_backup_restore_migration_api_worker_and_e2e(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment

    result = _run_rehearsal(environment)

    assert result.returncode == 0, result.stderr
    records = _records(command_log)
    snapshot_a = _index(
        records,
        lambda record: (
            record["tool"] == "ssh" and "-i" in record["args"] and "psql" in record["args"]
        ),
    )
    remote_dump = _index(
        records,
        lambda record: record["tool"] == "ssh" and "gate85-remote-dump" in record["args"],
    )
    snapshot_b = next(
        index
        for index, record in enumerate(records[snapshot_a + 1 :], snapshot_a + 1)
        if record["tool"] == "ssh" and "-i" in record["args"] and "psql" in record["args"]
    )
    scp = _index(records, lambda record: record["tool"] == "scp")
    restore_postgres = _index(
        records,
        lambda record: (
            record["tool"] == "docker"
            and "up" in record["args"]
            and record["args"][-1] == "postgres"
        ),
    )
    restore = _index(
        records,
        lambda record: (
            record["tool"] == "docker" and "pg_restore --exit-on-error" in " ".join(record["args"])
        ),
    )
    migration = _index(
        records,
        lambda record: (
            record["tool"] == "docker" and record["args"][-3:] == ["alembic", "upgrade", "head"]
        ),
    )
    api = _index(
        records,
        lambda record: (
            record["tool"] == "docker" and "up" in record["args"] and record["args"][-1] == "api"
        ),
    )
    worker = _index(
        records,
        lambda record: (
            record["tool"] == "docker" and "up" in record["args"] and record["args"][-1] == "worker"
        ),
    )
    e2e = _index(
        records,
        lambda record: record["tool"] == "docker" and "/api/v1/me" in record["args"],
    )
    assert snapshot_a < remote_dump < snapshot_b < scp < restore_postgres
    assert restore_postgres < restore < migration < api < worker < e2e
    assert "source_snapshot_a_equals_b=true" in result.stdout
    assert "restored_consistency=exact-match" in result.stdout
    assert "Gate 8.5 validation: PASS" in result.stdout
    assert "TEMPORARY RESTORE ENVIRONMENT RETAINED" in result.stdout
    backups = list(Path(environment["PF_GATE85_BACKUP_DIR"]).glob("*.dump"))
    manifests = list(Path(environment["PF_GATE85_BACKUP_DIR"]).glob("*.manifest.json"))
    assert len(backups) == len(manifests) == 1
    assert backups[0].read_bytes() == DUMP_CONTENT
    assert backups[0].stat().st_mode & 0o077 == 0
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["result"] == "PASS"
    assert manifest["backup"]["sha256"] == DUMP_SHA256
    assert manifest["restore"]["consistency"] == "exact-match"
    assert all(
        container.endswith("|exited")
        for container in manifest["restore"]["resources_retained"]["containers"]
    )
    assert manifest["destructive_cleanup"] == "NOT PERFORMED"


@pytest.mark.parametrize(
    "failure",
    [
        "remote-unreachable",
        "partial-project",
        "multiple-worker",
        "unhealthy-postgres",
        "invalid-fixture",
        "snapshot-changed",
        "remote-dump",
        "scp",
        "checksum-mismatch",
        "existing-restore-project",
    ],
)
def test_source_backup_or_copy_failure_never_mutates_restore_project(
    rehearsal_environment: tuple[dict[str, str], Path],
    failure: str,
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run_rehearsal(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert not _has_compose_service(records, "up", "postgres")
    assert "pg_restore --exit-on-error" not in _flat(records)
    assert "restore resources are retained" in result.stderr


@pytest.mark.parametrize("failure", ["pg-restore", "migration", "consistency-mismatch"])
def test_restore_migration_or_consistency_failure_never_starts_api_worker_or_e2e(
    rehearsal_environment: tuple[dict[str, str], Path],
    failure: str,
) -> None:
    environment, command_log = rehearsal_environment
    environment["FAKE_FAIL"] = failure

    result = _run_rehearsal(environment)

    assert result.returncode != 0
    records = _records(command_log)
    assert _has_compose_service(records, "up", "postgres")
    assert not _has_compose_service(records, "up", "api")
    assert not _has_compose_service(records, "up", "worker")
    assert "/api/v1/me" not in _flat(records)
    assert "downgrade" not in _flat(records)
    assert "down" not in _flat(records)


def test_sigterm_during_restore_reports_stage_and_retains_resources(
    rehearsal_environment: tuple[dict[str, str], Path],
    tmp_path: Path,
) -> None:
    environment, command_log = rehearsal_environment
    notify = tmp_path / "restore-notify"
    environment["FAKE_BLOCK"] = "pg-restore"
    environment["FAKE_NOTIFY"] = str(notify)
    process = subprocess.Popen(
        (str(REHEARSAL_SCRIPT),),
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    for _ in range(200):
        if notify.exists():
            break
        import time

        time.sleep(0.02)
    assert notify.exists()

    os.killpg(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 130
    assert "interrupted by SIGTERM during stage=pg-restore" in stderr
    assert "restore resources are retained" in stderr
    records = _records(command_log)
    assert not any(record["args"][-3:] == ["alembic", "upgrade", "head"] for record in records)
    assert not _has_compose_service(records, "up", "api")
    assert "Gate 8.5 validation: PASS" not in stdout


def test_secret_canary_never_appears_in_output_command_log_or_manifest(
    rehearsal_environment: tuple[dict[str, str], Path],
) -> None:
    environment, command_log = rehearsal_environment
    canary = f"gate85-secret-{secrets.token_hex(24)}"
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

    result = _run_rehearsal(environment)

    assert result.returncode == 0, result.stderr
    assert canary not in result.stdout
    assert canary not in result.stderr
    assert canary not in command_log.read_text(encoding="utf-8")
    manifest = next(Path(environment["PF_GATE85_BACKUP_DIR"]).glob("*.manifest.json"))
    assert canary not in manifest.read_text(encoding="utf-8")


def test_script_has_no_source_restore_or_automatic_destruction() -> None:
    source = REHEARSAL_SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "alembic downgrade",
        "docker system prune",
        "docker volume prune",
        "docker network prune",
        "down -v",
        "rm -rf",
    )
    for value in forbidden:
        assert value not in source
    assert "pg_restore --list" in source
    assert "pg_restore --exit-on-error" in source
    assert "compose stop" in source
    assert "TEMPORARY RESTORE ENVIRONMENT RETAINED" in source


def test_consistency_sql_is_read_only_and_omits_sensitive_bodies() -> None:
    source = CONSISTENCY_SQL.read_text(encoding="utf-8")
    assert "REPEATABLE READ READ ONLY" in source
    assert "pathfinder_checkpoint.checkpoints" in source
    assert "thread_id = params.fixture_run_id::text" in source
    assert "args_snapshot" not in source
    assert "target_snapshot" not in source
    assert "document.content," not in source
    assert "chunk.text," not in source
    assert "invocation.input" not in source
    assert "invocation.output" not in source
    for field in ("client_request_id", "create_request_digest", "create_request_version"):
        assert f"'{field}', run.{field}" in source
    assert (
        "'result_sha256', encode(sha256(convert_to(run.result_json::text, 'UTF8')), 'hex')"
        in source
    )
