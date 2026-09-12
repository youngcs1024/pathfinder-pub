from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = PROJECT_ROOT / "scripts" / "release.sh"
CANDIDATE_DIGEST = f"registry.example.test/pathfinder@sha256:{'a' * 64}"
PREVIOUS_DIGEST = f"registry.example.test/pathfinder@sha256:{'b' * 64}"

FAKE_DOCKER = r"""#!/usr/bin/env python3
import json
import os
import signal
import sys
from pathlib import Path

args = sys.argv[1:]
log_path = Path(os.environ["FAKE_DOCKER_LOG"])
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")

state = os.environ.get("FAKE_STATE", "fresh")
failure = os.environ.get("FAKE_FAIL", "")
head = os.environ.get("FAKE_HEAD", "0015_e3_run_request_identity")
before = os.environ.get("FAKE_REVISION_BEFORE", head)
after = os.environ.get("FAKE_REVISION_AFTER", head)

def emit(value):
    sys.stdout.write(value + "\n")

if args[:2] == ["image", "inspect"]:
    if failure == "image-inspect":
        raise SystemExit(1)
    emit("sha256:" + "c" * 64)
    raise SystemExit(0)

if args[:3] == ["compose", "config", "--quiet"]:
    raise SystemExit(1 if failure == "compose-config" else 0)

if args[:6] == ["compose", "run", "--rm", "--no-deps", "--entrypoint", "python"]:
    service = args[6]
    if failure == f"settings-{service}":
        raise SystemExit(1)
    raise SystemExit(0)

if args[:6] == ["compose", "run", "--rm", "--no-deps", "--entrypoint", "alembic"]:
    if failure == "heads":
        raise SystemExit(1)
    if failure == "multiple-heads":
        emit("0014_one (head)")
        emit("0014_two (head)")
    else:
        emit(f"{head} (head)")
    raise SystemExit(0)

if args[:4] == ["compose", "ps", "--all", "--quiet"]:
    service = args[4]
    ids = {
        "fresh": {"api": [], "worker": [], "caddy": [], "postgres": []},
        "fresh-postgres": {
            "api": [], "worker": [], "caddy": [], "postgres": ["postgres-id"]
        },
        "existing": {
            "api": ["api-id"],
            "worker": ["worker-id"],
            "caddy": ["caddy-id"],
            "postgres": ["postgres-id"],
        },
        "partial": {"api": ["api-id"], "worker": [], "caddy": [], "postgres": []},
        "mismatch": {
            "api": ["api-id"],
            "worker": ["worker-id"],
            "caddy": ["caddy-id"],
            "postgres": ["postgres-id"],
        },
        "reference-mismatch": {
            "api": ["api-id"],
            "worker": ["worker-id"],
            "caddy": ["caddy-id"],
            "postgres": ["postgres-id"],
        },
        "multiworker": {
            "api": ["api-id"],
            "worker": ["worker-id", "worker-id-2"],
            "caddy": ["caddy-id"],
            "postgres": ["postgres-id"],
        },
    }[state][service]
    for container_id in ids:
        emit(container_id)
    raise SystemExit(0)

if args and args[0] == "inspect":
    container_id = args[-1]
    if container_id == "api-id":
        emit(f"{PREVIOUS_IMAGE}|sha256:previous|running|healthy")
    elif container_id.startswith("worker-id"):
        image_id = "sha256:different" if state == "mismatch" else "sha256:previous"
        image_ref = (
            "pathfinder:different-ref"
            if state == "reference-mismatch"
            else "{PREVIOUS_IMAGE}"
        )
        emit(f"{image_ref}|{image_id}|running|healthy")
    elif container_id == "caddy-id":
        emit("caddy:tested|sha256:caddy|running|")
    else:
        emit("postgres:tested|sha256:postgres|running|healthy")
    raise SystemExit(0)

if args[:5] == ["compose", "exec", "-T", "postgres", "sh"]:
    shell_command = args[-1]
    if "pg_dump" in shell_command:
        if failure == "pg-dump":
            raise SystemExit(1)
        sys.stdout.buffer.write(b"PGDMP-fake-release-backup")
        raise SystemExit(0)
    if "pg_restore" in shell_command:
        sys.stdin.buffer.read()
        raise SystemExit(1 if failure == "pg-restore-list" else 0)
    log_text = log_path.read_text(encoding="utf-8")
    migrated = '"api", "alembic", "upgrade", "head"' in log_text
    if migrated and failure == "migration":
        emit(os.environ.get("FAKE_REVISION_AFTER_FAILURE", before))
    else:
        emit(after if migrated else before)
    raise SystemExit(0)

if args[:6] == ["compose", "run", "--rm", "--no-deps", "api", "alembic"]:
    notify_path = os.environ.get("FAKE_MIGRATION_NOTIFY")
    release_path = os.environ.get("FAKE_MIGRATION_RELEASE")
    if notify_path:
        with open(notify_path, "w", encoding="utf-8") as handle:
            handle.write("migration-started")
        with open(release_path, "r", encoding="utf-8") as handle:
            handle.read(1)
    raise SystemExit(1 if failure == "migration" else 0)

if args[:2] == ["compose", "up"]:
    service = args[-1]
    if failure == f"up-{service}":
        raise SystemExit(1)
    raise SystemExit(0)

if args[:4] == ["compose", "exec", "-T", "api"]:
    raise SystemExit(1 if failure == "api-smoke" else 0)

raise SystemExit(97)
""".replace("{PREVIOUS_IMAGE}", PREVIOUS_DIGEST)

FAKE_CURL = r"""#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["FAKE_CURL_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\n")
raise SystemExit(1 if os.environ.get("FAKE_FAIL") == "edge-smoke" else 0)
"""


def _base_environment(tmp_path: Path, fake_bin: Path, docker_log: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "DASHSCOPE_API_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "PF_QWEN_WORKSPACE_ID",
        "PF_RELEASE_EDGE_SMOKE_URL",
        "PF_RELEASE_SCHEMA_COMPATIBLE",
        "PF_SUPABASE_PROJECT_REF",
        "PF_SUPABASE_PUBLISHABLE_KEY",
        "TAVILY_API_KEY",
    ):
        environment.pop(name, None)
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": "pathfinder-release-contract",
            "FAKE_DOCKER_LOG": str(docker_log),
            "FAKE_HEAD": "0015_e3_run_request_identity",
            "FAKE_REVISION_AFTER": "0015_e3_run_request_identity",
            "FAKE_REVISION_BEFORE": "0015_e3_run_request_identity",
            "FAKE_STATE": "fresh",
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PATHFINDER_IMAGE": CANDIDATE_DIGEST,
            "PF_AUTH_MODE": "fake",
            "PF_DATABASE_URL": (
                "postgresql+psycopg://pathfinder:test-password@postgres:5432/pathfinder"
            ),
            "PF_LLM_MODE": "fake",
            "PF_POSTGRES_PASSWORD": "test-password",
            "PF_PUBLIC_DOMAIN": "pathfinder.example.test",
            "PF_RELEASE_BACKUP_DIR": str(tmp_path / "backups"),
            "PF_SEARCH_MODE": "fake",
            "PF_TRACE_MODE": "off",
            "TMPDIR": str(lock_root),
        }
    )
    return environment


@pytest.fixture
def release_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8")
    docker.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)
    docker_log = tmp_path / "docker.log"
    curl_log = tmp_path / "curl.log"
    environment = _base_environment(tmp_path, fake_bin, docker_log)
    environment["FAKE_CURL_LOG"] = str(curl_log)
    return environment, docker_log, curl_log


def _run_release(
    environment: dict[str, str], action: str = "release"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (str(RELEASE_SCRIPT), action),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _commands(log_path: Path) -> list[list[str]]:
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def _find(commands: list[list[str]], expected: list[str]) -> int:
    for index, command in enumerate(commands):
        if command[: len(expected)] == expected:
            return index
    raise AssertionError(f"command prefix not found: {expected!r}\n{commands!r}")


def _service_ups(commands: list[list[str]]) -> list[str]:
    return [command[-1] for command in commands if command[:2] == ["compose", "up"]]


def _assert_no_migration_or_service_mutation(commands: list[list[str]]) -> None:
    assert ["compose", "run", "--rm", "--no-deps", "api", "alembic"] not in [
        command[:6] for command in commands
    ]
    assert _service_ups(commands) == []
    assert not any("pg_dump" in " ".join(command) for command in commands)


def test_fresh_release_orders_bootstrap_backup_migration_and_services(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_REVISION_BEFORE"] = "base"

    result = _run_release(environment)

    assert result.returncode == 0, result.stderr
    commands = _commands(docker_log)
    postgres = _find(commands, ["compose", "up", "--detach", "--no-deps"])
    dump = next(index for index, command in enumerate(commands) if "pg_dump" in " ".join(command))
    validate = next(
        index for index, command in enumerate(commands) if "pg_restore" in " ".join(command)
    )
    migration = _find(commands, ["compose", "run", "--rm", "--no-deps", "api", "alembic"])
    api = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["compose", "up"] and command[-1] == "api"
    )
    smoke = _find(commands, ["compose", "exec", "-T", "api", "python"])
    caddy = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["compose", "up"] and command[-1] == "caddy"
    )
    worker = next(
        index
        for index, command in enumerate(commands)
        if command[:2] == ["compose", "up"] and command[-1] == "worker"
    )
    assert postgres < dump < validate < migration < api < smoke < caddy < worker
    assert "deployment=fresh" in result.stdout
    assert "database_revision_before=base" in result.stdout
    assert "database_revision_after=0015_e3_run_request_identity" in result.stdout
    backups = list((Path(environment["PF_RELEASE_BACKUP_DIR"])).glob("*.dump"))
    assert len(backups) == 1
    assert backups[0].stat().st_size > 0
    assert backups[0].stat().st_mode & 0o077 == 0


def test_existing_release_keeps_postgres_and_caddy_and_converges(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_STATE"] = "existing"

    first = _run_release(environment)
    second = _run_release(environment)

    assert first.returncode == second.returncode == 0
    commands = _commands(docker_log)
    assert _service_ups(commands) == ["api", "worker", "api", "worker"]
    assert first.stdout.count(f"previous_api_image={PREVIOUS_DIGEST}") == 1
    assert "existing Caddy left unchanged" in first.stdout
    assert len(list(Path(environment["PF_RELEASE_BACKUP_DIR"]).glob("*.dump"))) == 2


@pytest.mark.parametrize("state", ["partial", "mismatch", "reference-mismatch", "multiworker"])
def test_inconsistent_deployment_refuses_before_backup_migration_or_mutation(
    release_environment: tuple[dict[str, str], Path, Path],
    state: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_STATE"] = state

    result = _run_release(environment)

    assert result.returncode != 0
    assert "deployment=inconsistent" in result.stderr
    _assert_no_migration_or_service_mutation(_commands(docker_log))


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("image-inspect", "candidate image is not available"),
        ("compose-config", "compose config validation failed"),
        ("settings-api", "candidate api settings validation failed"),
        ("settings-worker", "candidate worker settings validation failed"),
        ("multiple-heads", "exactly one Alembic head"),
    ],
)
def test_static_preflight_failure_has_no_backup_migration_or_service_mutation(
    release_environment: tuple[dict[str, str], Path, Path],
    failure: str,
    message: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_FAIL"] = failure

    result = _run_release(environment)

    assert result.returncode != 0
    assert message in result.stderr
    _assert_no_migration_or_service_mutation(_commands(docker_log))


@pytest.mark.parametrize(
    "variable_name",
    [
        "PATHFINDER_IMAGE",
        "PF_PUBLIC_DOMAIN",
        "PF_RELEASE_BACKUP_DIR",
        "PF_DATABASE_URL",
        "PF_POSTGRES_PASSWORD",
    ],
)
def test_required_release_input_fails_before_any_docker_command(
    release_environment: tuple[dict[str, str], Path, Path],
    variable_name: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment.pop(variable_name)

    result = _run_release(environment)

    assert result.returncode != 0
    assert f"{variable_name} is required" in result.stderr
    assert _commands(docker_log) == []


@pytest.mark.parametrize("failure", ["pg-dump", "pg-restore-list"])
def test_backup_failure_stops_before_migration_and_candidate_services(
    release_environment: tuple[dict[str, str], Path, Path],
    failure: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_FAIL"] = failure
    environment["FAKE_REVISION_BEFORE"] = "base"

    result = _run_release(environment)

    assert result.returncode != 0
    commands = _commands(docker_log)
    assert not any(command[-3:] == ["alembic", "upgrade", "head"] for command in commands)
    assert _service_ups(commands) == ["postgres"]


def test_migration_failure_is_precise_and_never_promotes_or_recovers_automatically(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, curl_log = release_environment
    environment["FAKE_FAIL"] = "migration"
    environment["FAKE_REVISION_BEFORE"] = "0013_previous"
    environment["FAKE_REVISION_AFTER_FAILURE"] = "0013_previous"
    environment["FAKE_STATE"] = "existing"
    environment["PF_RELEASE_SCHEMA_COMPATIBLE"] = "1"

    result = _run_release(environment)

    assert result.returncode != 0
    commands = _commands(docker_log)
    assert any(command[-3:] == ["alembic", "upgrade", "head"] for command in commands)
    assert _service_ups(commands) == []
    assert not curl_log.exists()
    flat = "\n".join(" ".join(command) for command in commands)
    assert "alembic downgrade" not in flat
    assert "pg_restore --list" in flat
    assert "migration failed; candidate API/worker/Caddy promotion was not started" in result.stderr
    assert "failure does not prove the database is unchanged" in result.stderr
    assert "revision_after=0013_previous" in result.stderr


def test_existing_schema_change_requires_explicit_compatibility_review(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_STATE"] = "existing"
    environment["FAKE_HEAD"] = "0015_candidate"
    environment["FAKE_REVISION_BEFORE"] = "0015_e3_run_request_identity"

    result = _run_release(environment)

    assert result.returncode != 0
    assert "PF_RELEASE_SCHEMA_COMPATIBLE=1" in result.stderr
    commands = _commands(docker_log)
    assert not any("pg_dump" in " ".join(command) for command in commands)
    assert not any(command[-3:] == ["alembic", "upgrade", "head"] for command in commands)


def test_fresh_base_does_not_require_schema_compatibility_confirmation(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, _docker_log, _curl_log = release_environment
    environment["FAKE_HEAD"] = "0015_candidate"
    environment["FAKE_REVISION_BEFORE"] = "base"
    environment["FAKE_REVISION_AFTER"] = "0015_candidate"

    result = _run_release(environment)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("failure", ["up-api", "api-smoke"])
def test_api_or_internal_smoke_failure_never_starts_caddy_or_worker(
    release_environment: tuple[dict[str, str], Path, Path],
    failure: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_FAIL"] = failure
    environment["FAKE_REVISION_BEFORE"] = "base"

    result = _run_release(environment)

    assert result.returncode != 0
    ups = _service_ups(_commands(docker_log))
    assert ups == ["postgres", "api"]
    assert "worker was not promoted" in result.stderr


def test_configured_edge_smoke_failure_never_starts_worker_or_rolls_back(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, curl_log = release_environment
    environment["FAKE_FAIL"] = "edge-smoke"
    environment["FAKE_REVISION_BEFORE"] = "base"
    environment["PF_RELEASE_EDGE_SMOKE_URL"] = "https://pathfinder.example.test/"

    result = _run_release(environment)

    assert result.returncode != 0
    assert _service_ups(_commands(docker_log)) == ["postgres", "api", "caddy"]
    assert curl_log.exists()
    flat = docker_log.read_text(encoding="utf-8")
    assert "downgrade" not in flat
    assert "rollback" not in flat


def test_worker_failure_is_terminal_and_has_no_hidden_retry_or_database_recovery(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_FAIL"] = "up-worker"
    environment["FAKE_REVISION_BEFORE"] = "base"

    result = _run_release(environment)

    assert result.returncode != 0
    commands = _commands(docker_log)
    assert _service_ups(commands) == ["postgres", "api", "caddy", "worker"]
    assert _service_ups(commands).count("worker") == 1
    flat = "\n".join(" ".join(command) for command in commands)
    assert "alembic downgrade" not in flat
    assert "candidate worker did not become healthy" in result.stderr


def test_safe_code_rollback_promotes_api_then_worker_without_database_mutation(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_STATE"] = "existing"

    result = _run_release(environment, "rollback")

    assert result.returncode == 0, result.stderr
    commands = _commands(docker_log)
    assert _service_ups(commands) == ["api", "worker"]
    assert not any("pg_dump" in " ".join(command) for command in commands)
    assert not any(command[-3:] == ["alembic", "upgrade", "head"] for command in commands)
    assert "no database downgrade or restore will run" in result.stdout


@pytest.mark.parametrize(
    ("state", "revision", "message"),
    [
        ("existing", "0015_newer", "schema has moved"),
        ("fresh", "base", "no previous application deployment"),
    ],
)
def test_unsafe_or_fresh_rollback_is_refused_before_promotion(
    release_environment: tuple[dict[str, str], Path, Path],
    state: str,
    revision: str,
    message: str,
) -> None:
    environment, docker_log, _curl_log = release_environment
    environment["FAKE_STATE"] = state
    environment["FAKE_REVISION_BEFORE"] = revision

    result = _run_release(environment, "rollback")

    assert result.returncode != 0
    assert message in result.stderr
    assert _service_ups(_commands(docker_log)) == []


def test_concurrent_release_is_refused_before_any_docker_command_and_sigterm_stops_flow(
    release_environment: tuple[dict[str, str], Path, Path],
    tmp_path: Path,
) -> None:
    first_environment, first_log, _curl_log = release_environment
    first_environment["FAKE_REVISION_BEFORE"] = "base"
    notify = tmp_path / "migration-notify"
    release = tmp_path / "migration-release"
    os.mkfifo(notify)
    os.mkfifo(release)
    first_environment["FAKE_MIGRATION_NOTIFY"] = str(notify)
    first_environment["FAKE_MIGRATION_RELEASE"] = str(release)
    second_log = tmp_path / "second-docker.log"
    second_environment = first_environment.copy()
    second_environment["FAKE_DOCKER_LOG"] = str(second_log)

    process = subprocess.Popen(
        (str(RELEASE_SCRIPT), "release"),
        cwd=PROJECT_ROOT,
        env=first_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with notify.open("r", encoding="utf-8") as handle:
        assert handle.read() == "migration-started"

    second = _run_release(second_environment)
    assert second.returncode != 0
    assert "another release holds lock" in second.stderr
    assert _commands(second_log) == []

    os.killpg(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 130
    assert "interrupted during migration" in stderr
    assert "revision_before=base" in stderr
    commands = _commands(first_log)
    assert _service_ups(commands) == ["postgres"]
    assert not any(command[:4] == ["compose", "exec", "-T", "api"] for command in commands)
    assert "success" not in stdout


def test_secret_canary_never_appears_in_output_diagnostics_or_command_log(
    release_environment: tuple[dict[str, str], Path, Path],
) -> None:
    environment, docker_log, _curl_log = release_environment
    canary = f"sensitive-{secrets.token_hex(24)}"
    environment.update(
        {
            "DASHSCOPE_API_KEY": canary,
            "LANGFUSE_PUBLIC_KEY": canary,
            "LANGFUSE_SECRET_KEY": canary,
            "PF_QWEN_WORKSPACE_ID": "valid-workspace",
            "PF_SUPABASE_PROJECT_REF": "valid-project",
            "PF_SUPABASE_PUBLISHABLE_KEY": f"sb_publishable_{canary}",
            "TAVILY_API_KEY": canary,
            "FAKE_STATE": "partial",
        }
    )

    result = _run_release(environment)

    assert result.returncode != 0
    assert canary not in result.stdout
    assert canary not in result.stderr
    assert canary not in docker_log.read_text(encoding="utf-8")


def test_release_script_does_not_name_or_create_business_execution_facts() -> None:
    source = RELEASE_SCRIPT.read_text(encoding="utf-8")

    for table_name in (
        "runs",
        "run_jobs",
        "run_events",
        "llm_invocations",
        "tool_invocations",
        "mock_submissions",
    ):
        assert table_name not in source
    assert "alembic downgrade" not in source
    assert "pg_restore --list" in source
    assert "pg_restore" in source
    assert "pg_dump" in source
