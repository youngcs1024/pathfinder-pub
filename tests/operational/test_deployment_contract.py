from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CADDYFILE = PROJECT_ROOT / "Caddyfile"
COMPOSE_FILE = PROJECT_ROOT / "compose.yaml"
CADDY_IMAGE = (
    "caddy:2.11.4-alpine@sha256:5f5c8640aae01df9654968d946d8f1a56c497f1dd5c5cda4cf95ab7c14d58648"
)
BUSINESS_SECRETS = (
    "DASHSCOPE_API_KEY",
    "TAVILY_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
)
DATABASE_URL = "postgresql+psycopg://pathfinder:deployment-test@postgres:5432/pathfinder"

DB_ENVIRONMENT = {
    "PF_DB_STATEMENT_TIMEOUT_MS": "7000",
    "PF_DB_LOCK_TIMEOUT_MS": "1200",
    "PF_DB_IDLE_IN_TRANSACTION_TIMEOUT_MS": "11000",
    "PF_DB_CONNECT_TIMEOUT_SECONDS": "6",
    "PF_DB_POOL_TIMEOUT_SECONDS": "3",
    "PF_DB_POOL_SIZE": "4",
    "PF_DB_MAX_OVERFLOW": "1",
}


def _render_compose(*files: str, database_defaults: bool = False) -> tuple[dict[str, object], str]:
    env = os.environ.copy()
    env.update(
        {
            **DB_ENVIRONMENT,
            "PATHFINDER_IMAGE": "pathfinder-ci:test-sha",
            "PF_AUTH_MODE": "supabase",
            "PF_DATABASE_URL": DATABASE_URL,
            "PF_LLM_MODE": "qwen",
            "PF_LOG_LEVEL": "WARNING",
            "PF_POSTGRES_PASSWORD": "deployment-test",
            "PF_PUBLIC_DOMAIN": "pathfinder.example.test",
            "PF_QWEN_WORKSPACE_ID": "beijing-workspace",
            "PF_SEARCH_MODE": "tavily",
            "PF_SUPABASE_JWT_AUDIENCE": "authenticated",
            "PF_SUPABASE_PROJECT_REF": "project-ref",
            "PF_SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test",
            "PF_TRACE_MODE": "langfuse",
        }
    )
    if database_defaults:
        for name in DB_ENVIRONMENT:
            env.pop(name, None)
    command = ["docker", "compose", "--env-file", "/dev/null"]
    for compose_file in files:
        command.extend(("-f", compose_file))
    command.extend(("config", "--format", "json"))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout), result.stdout


def _network_names(service: dict[str, object]) -> set[str]:
    return set(service["networks"])


def test_production_compose_has_only_the_approved_public_edge() -> None:
    config, _rendered = _render_compose()
    services = config["services"]

    assert set(services) == {"postgres", "api", "worker", "caddy"}
    for service_name in ("postgres", "api", "worker"):
        assert services[service_name].get("ports", []) == []

    caddy = services["caddy"]
    assert caddy["ports"] == [
        {"mode": "ingress", "target": 80, "published": "80", "protocol": "tcp"},
        {"mode": "ingress", "target": 443, "published": "443", "protocol": "tcp"},
    ]
    assert caddy["image"] == CADDY_IMAGE
    assert caddy["environment"] == {"PF_PUBLIC_DOMAIN": "pathfinder.example.test"}
    assert caddy["depends_on"]["api"]["condition"] == "service_healthy"

    assert _network_names(caddy) == {"edge"}
    assert _network_names(services["api"]) == {"edge", "backend"}
    assert _network_names(services["worker"]) == {"backend", "worker_egress"}
    assert _network_names(services["postgres"]) == {"backend"}
    assert config["networks"]["backend"]["internal"] is True
    assert config["networks"]["edge"].get("internal", False) is False
    assert config["networks"]["worker_egress"].get("internal", False) is False

    worker = services["worker"]
    assert worker["scale"] == 1
    assert worker["deploy"]["replicas"] == 1
    assert all(services[name]["restart"] == "unless-stopped" for name in services)

    api_environment = services["api"]["environment"]
    assert api_environment == {
        **DB_ENVIRONMENT,
        "PF_API_HOST": "0.0.0.0",
        "PF_AUTH_MODE": "supabase",
        "PF_DATABASE_URL": DATABASE_URL,
        "PF_LOG_LEVEL": "WARNING",
        "PF_MATERIAL_ALIASES_FILE": "",
        "PF_SUPABASE_JWT_AUDIENCE": "authenticated",
        "PF_SUPABASE_PROJECT_REF": "project-ref",
        "PF_SUPABASE_PUBLISHABLE_KEY": "sb_publishable_test",
    }
    assert not set(BUSINESS_SECRETS) & set(api_environment)

    assert worker["environment"] == {
        **DB_ENVIRONMENT,
        "DASHSCOPE_API_KEY": "",
        "LANGFUSE_BASE_URL": "",
        "LANGFUSE_PUBLIC_KEY": "",
        "LANGFUSE_SAMPLE_RATE": "1.0",
        "LANGFUSE_SECRET_KEY": "",
        "PF_DATABASE_URL": DATABASE_URL,
        "PF_LLM_MODE": "qwen",
        "PF_LOG_LEVEL": "WARNING",
        "PF_MATERIAL_ALIASES_FILE": "",
        "PF_MOCK_PORTAL_BASE_URL": "http://api:8000",
        "PF_QWEN_WORKSPACE_ID": "beijing-workspace",
        "PF_SEARCH_MODE": "tavily",
        "PF_TRACE_MODE": "langfuse",
        "TAVILY_API_KEY": "",
    }


def test_production_postgres_remains_private_healthy_and_persistent() -> None:
    config, _rendered = _render_compose()
    postgres = config["services"]["postgres"]

    assert postgres["image"] == "pgvector/pgvector:0.8.5-pg16"
    assert postgres["environment"] == {
        "POSTGRES_DB": "pathfinder",
        "POSTGRES_PASSWORD": "deployment-test",
        "POSTGRES_USER": "pathfinder",
    }
    assert postgres["healthcheck"] == {
        "test": ["CMD-SHELL", "pg_isready -U pathfinder -d pathfinder"],
        "timeout": "5s",
        "interval": "2s",
        "retries": 20,
        "start_period": "5s",
    }
    assert postgres["volumes"] == [
        {
            "type": "volume",
            "source": "pathfinder_postgres_data",
            "target": "/var/lib/postgresql/data",
            "volume": {},
        }
    ]


def test_production_compose_preserves_application_container_hardening() -> None:
    config, _rendered = _render_compose()
    services = config["services"]
    api = services["api"]
    worker = services["worker"]

    assert api["image"] == worker["image"] == "pathfinder-ci:test-sha"
    assert api["command"] == ["python", "-m", "app.main"]
    assert worker["command"] == ["python", "-m", "app.worker.main"]
    for service in (api, worker):
        assert "build" not in service
        assert service["read_only"] is True
        assert service["tmpfs"] == ["/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"]
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert int(service["mem_limit"]) > 0
        assert service["cpus"] > 0
        assert service["pids_limit"] > 0
        assert service["healthcheck"]["timeout"] == "2s"
        assert service["healthcheck"]["interval"] == "10s"
        assert service["healthcheck"]["retries"] == 3
        assert service["healthcheck"]["start_period"] == "10s"
        assert service["depends_on"]["postgres"]["condition"] == "service_healthy"

    api_health = api["healthcheck"]["test"]
    assert api_health[:3] == ["CMD", "python", "-c"]
    assert "http://127.0.0.1:8000/readyz" in api_health[3]
    assert "/healthz" not in api_health[3]
    worker_health = worker["healthcheck"]["test"]
    assert worker_health[:3] == ["CMD", "python", "-c"]
    assert "WORKER_READY_PATH.is_file()" in worker_health[3]
    assert "DatabaseComponent.WORKER_HEALTHCHECK" in worker_health[3]
    assert "await engine.dispose()" in worker_health[3]
    assert "DatabaseReadinessProbe(engine).is_ready()" in worker_health[3]
    assert worker["stop_grace_period"] == "30s"


def test_production_compose_hardens_caddy_and_persists_only_operational_state() -> None:
    config, _rendered = _render_compose()
    caddy = config["services"]["caddy"]

    assert caddy["read_only"] is True
    assert caddy["tmpfs"] == ["/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777"]
    assert caddy["cap_drop"] == ["ALL"]
    assert caddy["cap_add"] == ["NET_BIND_SERVICE"]
    assert caddy["security_opt"] == ["no-new-privileges:true"]
    assert int(caddy["mem_limit"]) > 0
    assert caddy["cpus"] > 0
    assert caddy["pids_limit"] > 0
    mounts_by_target = {mount["target"]: mount for mount in caddy["volumes"]}
    assert set(mounts_by_target) == {"/etc/caddy/Caddyfile", "/data", "/config"}

    caddyfile_mount = mounts_by_target["/etc/caddy/Caddyfile"]
    assert caddyfile_mount["type"] == "bind"
    assert caddyfile_mount["source"] == str(CADDYFILE)
    assert caddyfile_mount["target"] == "/etc/caddy/Caddyfile"
    assert caddyfile_mount["read_only"] is True

    data_mount = mounts_by_target["/data"]
    assert data_mount["type"] == "volume"
    assert data_mount["source"] == "pathfinder_caddy_data"
    assert data_mount["target"] == "/data"

    config_mount = mounts_by_target["/config"]
    assert config_mount["type"] == "volume"
    assert config_mount["source"] == "pathfinder_caddy_config"
    assert config_mount["target"] == "/config"
    assert set(config["volumes"]) == {
        "pathfinder_postgres_data",
        "pathfinder_caddy_data",
        "pathfinder_caddy_config",
    }


def test_production_compose_authors_fail_closed_caddyfile_bind() -> None:
    source = COMPOSE_FILE.read_text(encoding="utf-8")
    authored_mount = """\
      - type: bind
        source: ./Caddyfile
        target: /etc/caddy/Caddyfile
        read_only: true
        bind:
          create_host_path: false
"""

    assert source.count(authored_mount) == 1


def test_missing_bind_source_fails_service_start_without_creating_host_directory(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing-Caddyfile"
    image = f"pathfinder-ci:missing-bind-contract-{os.getpid()}"
    project = f"pathfinder-missing-bind-contract-{os.getpid()}"
    dockerfile = tmp_path / "Dockerfile"
    compose_file = tmp_path / "compose.yaml"
    dockerfile.write_text('FROM scratch\nCMD ["/missing"]\n', encoding="utf-8")
    compose_file.write_text(
        "\n".join(
            (
                "services:",
                "  caddy:",
                f"    image: {image}",
                "    volumes:",
                "      - type: bind",
                f"        source: {missing_source}",
                "        target: /etc/caddy/Caddyfile",
                "        read_only: true",
                "        bind:",
                "          create_host_path: false",
                "",
            )
        ),
        encoding="utf-8",
    )

    render = subprocess.run(
        (
            "docker",
            "compose",
            "--project-name",
            project,
            "-f",
            str(compose_file),
            "config",
            "--format",
            "json",
        ),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert render.returncode == 0, render.stderr
    assert not missing_source.exists()

    docker_info = subprocess.run(
        ("docker", "info", "--format", "{{.OperatingSystem}}"),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert docker_info.returncode == 0, docker_info.stderr
    if "docker desktop" in docker_info.stdout.lower():
        pytest.skip(
            "Docker Desktop's WSL path bridge creates a directory before the Linux "
            "Engine evaluates create_host_path=false"
        )

    build = subprocess.run(
        ("docker", "build", "--tag", image, str(tmp_path)),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    try:
        start = subprocess.run(
            (
                "docker",
                "compose",
                "--project-name",
                project,
                "-f",
                str(compose_file),
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
                "caddy",
            ),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        output = f"{start.stdout}\n{start.stderr}"
        assert start.returncode != 0
        assert "bind source path does not exist" in output.lower()
        assert not missing_source.exists()
    finally:
        subprocess.run(
            (
                "docker",
                "compose",
                "--project-name",
                project,
                "-f",
                str(compose_file),
                "down",
                "--timeout",
                "0",
            ),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ("docker", "image", "rm", image),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )


def test_dev_override_publishes_postgres_to_loopback_only() -> None:
    config, _rendered = _render_compose("compose.yaml", "compose.dev.yaml")
    postgres = config["services"]["postgres"]

    assert postgres["ports"] == [
        {
            "mode": "ingress",
            "host_ip": "127.0.0.1",
            "target": 5432,
            "published": "5432",
            "protocol": "tcp",
        }
    ]
    assert config["networks"]["backend"].get("internal", False) is False


def test_caddyfile_has_allowlisted_routes_and_bounded_transport_policy() -> None:
    source = CADDYFILE.read_text(encoding="utf-8")

    assert source.count("{$PF_PUBLIC_DOMAIN} {") == 1
    assert "protocols h1 h2" in source
    assert " h3" not in source
    assert "@public path / /static /static/* /api/v1/*" in source
    assert source.count("reverse_proxy") == 1
    assert "reverse_proxy api:8000" in source
    assert re.search(r"(?m)^\s*handle \{\n\s*respond 404\n\s*\}$", source)
    assert "max_size 1MB" in source
    assert "dial_timeout 5s" in source
    assert "response_header_timeout 30s" in source
    assert "health_uri /readyz" in source
    assert "health_interval 10s" in source
    assert "health_timeout 2s" in source

    for header in (
        '?X-Frame-Options "DENY"',
        '?X-Content-Type-Options "nosniff"',
        '?Referrer-Policy "no-referrer"',
        "-Server",
        'Strict-Transport-Security "max-age=86400"',
    ):
        assert header in source

    for forbidden in (
        "/internal/",
        "/healthz",
        "/docs",
        "/openapi.json",
        "lb_retries",
        "lb_try_duration",
        "stream_timeout",
        "flush_interval",
        "response_buffers",
        "header_up",
        "X-Actor",
        "X-Role",
        "X-Workspace",
        "X-Tenant",
    ):
        assert forbidden not in source


@pytest.mark.parametrize(
    "probe_result",
    [True, False, "failure", "dispose_failure", "cancelled", "missing_marker", "settings_failure"],
)
def test_worker_healthcheck_executes_policy_and_disposes(
    probe_result: bool | str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import asyncio
    import logging
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import app.config as config_module
    import app.db.readiness as readiness_module
    import app.db.session as session_module
    import app.worker.settings as worker_settings_module
    from app.config import Settings
    from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy

    config, _ = _render_compose()
    script = config["services"]["worker"]["healthcheck"]["test"][3]
    settings = Settings(db_pool_size=2, db_statement_timeout_ms=7500)
    engine = SimpleNamespace(dispose=AsyncMock())
    is_ready = AsyncMock(return_value=probe_result)
    canary = "DATABASE-QUERY-MODEL-SECRET-CANARY"
    if probe_result == "failure":
        is_ready.side_effect = RuntimeError(canary)
    elif probe_result == "dispose_failure":
        engine.dispose.side_effect = RuntimeError(canary)
    elif probe_result == "cancelled":
        is_ready.side_effect = asyncio.CancelledError()

    def create_engine(url, *, policy):
        logging.getLogger("sqlalchemy.pool").warning(canary)
        assert url == settings.database_url
        assert policy == DatabasePoolPolicy.from_settings(
            settings, DatabaseComponent.WORKER_HEALTHCHECK
        )
        return engine

    def load_settings():
        if probe_result == "settings_failure":
            raise ValueError(canary)
        return settings

    monkeypatch.setattr(config_module, "Settings", load_settings)
    monkeypatch.setattr(session_module, "create_database_engine", create_engine)
    monkeypatch.setattr(
        readiness_module,
        "DatabaseReadinessProbe",
        lambda _engine: SimpleNamespace(is_ready=is_ready),
    )
    monkeypatch.setattr(
        worker_settings_module,
        "WORKER_READY_PATH",
        SimpleNamespace(is_file=lambda: probe_result != "missing_marker"),
    )
    if probe_result == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            exec(compile(script, "<worker-healthcheck>", "exec"), {})
    else:
        with pytest.raises(SystemExit) as raised:
            exec(compile(script, "<worker-healthcheck>", "exec"), {})
        assert raised.value.code == (0 if probe_result is True else 1)
    if probe_result in ("missing_marker", "settings_failure"):
        is_ready.assert_not_awaited()
        engine.dispose.assert_not_awaited()
    else:
        is_ready.assert_awaited_once()
        engine.dispose.assert_awaited_once()
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_compose_database_defaults_match_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import Settings

    for name in DB_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    config, _ = _render_compose(database_defaults=True)
    settings = Settings()
    for service in ("api", "worker"):
        environment = config["services"][service]["environment"]
        for name in DB_ENVIRONMENT:
            assert float(environment[name]) == getattr(settings, name.removeprefix("PF_").lower())
