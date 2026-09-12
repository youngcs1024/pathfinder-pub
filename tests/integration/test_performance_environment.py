"""E5.2 real environment lifecycle only; no benchmark load or business smoke."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import httpx
import psycopg
import pytest
from sqlalchemy import make_url

from tests.performance.environment import PROFILE, EnvironmentProfile, IsolatedEnvironment

pytestmark = pytest.mark.integration

CANARY = "E52_ENVIRONMENT_SECRET_CANARY"


def test_owned_environment_migrates_serves_and_closes_without_business_load(tmp_path, monkeypatch):
    # The harness owns a NEW container, never the integration session's postgres_url fixture.
    # Testcontainers may prepare the pinned public image when this runner lacks it.
    for key in (
        "PF_DATABASE_URL",
        "PF_QWEN_API_KEY",
        "PF_TAVILY_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "DOCKER_HOST",
        "DOCKER_AUTH_CONFIG",
        "TESTCONTAINERS_HOST_OVERRIDE",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, CANARY)
    monkeypatch.setenv("PF_LLM_MODE", "qwen")
    monkeypatch.setenv("PF_AUTH_MODE", "supabase")
    marker = Path("/tmp/pathfinder-worker-ready")
    before = marker.read_bytes() if marker.exists() else None
    environment = IsolatedEnvironment(EnvironmentProfile(PROFILE), tmp_path / "environment")
    with environment:
        identity = environment.identity
        assert identity["api_origin"].startswith("http://127.0.0.1:")
        assert identity["database_name"] == "pf_e52_" + identity["owner"]
        assert len(identity["container_id"]) == 64
        assert set(identity["process_ids"]) == {"api", "worker", "migrate"}
        assert len(set(identity["process_ids"].values())) == 3
        assert "database_url" not in identity
        private_url = environment.database_url
        url = make_url(private_url)
        with psycopg.connect(
            host=url.host,
            port=url.port,
            dbname=url.database,
            user=url.username,
            password=url.password,
            connect_timeout=5,
            options="-c statement_timeout=5000",
        ) as connection:
            assert connection.execute("SELECT count(*) FROM alembic_version").fetchone() == (1,)
            assert connection.execute("SELECT count(*) FROM runs").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM run_jobs").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM llm_invocations").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM mock_submissions").fetchone() == (0,)
            assert (
                connection.execute(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = 'pathfinder_checkpoint'"
                ).fetchone()[0]
                > 0
            )
        with httpx.Client(timeout=2, trust_env=False) as client:
            assert client.get(environment.api_origin + "/readyz").status_code == 200
        assert (marker.read_bytes() if marker.exists() else None) == before
    result = environment.close()
    assert result["status"] == "PASS"
    assert result["resources_released"] is True
    for role in ("api", "worker"):
        # Inspect the recorded PID only; never signal a PID obtained from a report.
        assert not Path(f"/proc/{identity['process_ids'][role]}").exists()
    assert (marker.read_bytes() if marker.exists() else None) == before
    assert set(path.name for path in environment.output_dir.iterdir()) == {
        "config.json",
        "started.json",
        "resources.json",
        "ready.json",
        "result.json",
    }
    for path in environment.output_dir.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        content = path.read_text()
        if any(value in content for value in (CANARY, private_url, url.password)):
            pytest.fail("environment_artifact_leak", pytrace=False)
        json.loads(content)
    assert stat.S_IMODE(environment.output_dir.stat().st_mode) == 0o700
    assert os.getpid() not in identity["process_ids"].values()
