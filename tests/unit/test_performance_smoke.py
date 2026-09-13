"""Driver safety/failure tests use no real environment; actual flow is in integration core."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest

from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from tests.performance import _supervisor, environment, smoke
from tests.performance.adapters import Calls
from tests.performance.workload import CallRecord, profile, publish


def test_invalid_child_policy_is_rejected_before_directory_or_docker(tmp_path, monkeypatch):
    create = Mock(side_effect=AssertionError("must_not_create_resources"))
    monkeypatch.setattr(environment, "create_output_directory", create)
    env = environment.IsolatedEnvironment(
        environment.EnvironmentProfile(environment.PROFILE),
        tmp_path / "new",
        call_profile={"database_url": "production"},
    )
    with pytest.raises(environment.EnvironmentError, match="invalid_profile"):
        env.start()
    assert not env.output_created
    create.assert_not_called()
    with pytest.raises(environment.EnvironmentError, match="invalid_profile"):
        _supervisor.Supervisor(tmp_path, uuid4().hex, None, {"seed": "bad"})


def test_supervisor_passes_validated_profile_only_to_worker(tmp_path, monkeypatch):
    launched = []

    def launch(role, **kwargs):
        launched.append((role, kwargs))
        return SimpleNamespace(process=SimpleNamespace(pid=len(launched)))

    monkeypatch.setattr(_supervisor.OwnedProcess, "launch", launch)
    policy = profile("delayed-v1").model_dump(mode="json")
    supervisor = _supervisor.Supervisor(tmp_path, uuid4().hex, None, policy)
    for role in ("migrate", "api", "worker"):
        supervisor.launch(role, "private-runtime-capability")
    assert all("call_profile" not in kwargs["bootstrap"] for _, kwargs in launched[:2])
    assert launched[2][1]["bootstrap"]["call_profile"] == policy
    assert launched[2][1]["bootstrap"]["output_dir"] == str(tmp_path)
    assert all(kwargs["env"]["PF_LLM_MODE"] == "fake" for _, kwargs in launched)
    assert all(kwargs["env"]["PF_TRACE_MODE"] == "off" for _, kwargs in launched)


async def test_http_budget_stops_before_sending_next_request():
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1", trust_env=False
    ) as client:
        http = smoke.HTTP(client)
        http.count = smoke.MAX_REQUESTS - 1
        assert await http.request("GET", "/") == {"ok": True}
        with pytest.raises(smoke.SmokeFailure, match="request_limit"):
            await http.request("GET", "/")
        with pytest.raises(smoke.SmokeFailure, match="request_limit"):
            await http.events("/")
        assert len(sent) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="PRIVATE_BODY_CANARY"),
        httpx.Response(200, text="PRIVATE_BODY_CANARY"),
    ],
)
async def test_http_failure_does_not_echo_body(response):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: response),
        base_url="http://127.0.0.1",
        trust_env=False,
    ) as client:
        with pytest.raises(smoke.SmokeFailure, match=r"^http_failed$"):
            await smoke.HTTP(client).request("GET", "/")


async def test_stream_preserves_cursor_parses_heartbeat_and_rejects_oversize():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, text=': keepalive\n\nid: 2\nevent: run.completed\ndata: {"seq":2}\n\n'
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://127.0.0.1", trust_env=False
    ) as client:
        frames = await smoke.HTTP(client).events("/events", cursor=1)
    assert requests[0].headers["Last-Event-ID"] == "1"
    assert frames == [{"id": 2, "event": "run.completed", "data": {"seq": 2}}]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b"x" * (smoke.MAX_SSE_BYTES + 1))
        ),
        base_url="http://127.0.0.1",
        trust_env=False,
    ) as client:
        with pytest.raises(smoke.SmokeFailure):
            await smoke.HTTP(client).events("/events")


def test_exception_group_uses_fixed_underlying_category():
    error = ExceptionGroup("PRIVATE_EXCEPTION_CANARY", [smoke.SmokeFailure("request_limit")])
    assert smoke.failure_category(error) == "request_limit"
    assert smoke.failure_category(TimeoutError("PRIVATE_EXCEPTION_CANARY")) == "deadline"
    assert smoke.failure_category(RuntimeError("PRIVATE_EXCEPTION_CANARY")) == "business_failed"


def test_delayed_smoke_requires_explicit_opt_in_before_environment(tmp_path, monkeypatch):
    constructor = Mock(side_effect=AssertionError("must_not_start_environment"))
    monkeypatch.setattr(smoke, "IsolatedEnvironment", constructor)
    with pytest.raises(smoke.SmokeFailure):
        smoke.run_smoke(tmp_path / "new", mode="research", policy=profile("delayed-v1"))
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "error,category",
    [
        (smoke.SmokeFailure("request_limit"), "request_limit"),
        (TimeoutError(), "deadline"),
        (RuntimeError("PRIVATE_BODY_CANARY"), "business_failed"),
    ],
)
def test_driver_failure_closes_owned_environment_and_preserves_partial_evidence(
    tmp_path, monkeypatch, error, category
):
    instances = []

    class Environment:
        def __init__(self, _, directory, **kwargs):
            self.directory = directory
            self.output_created = False
            self.closed = False
            instances.append(self)

        def __enter__(self):
            self.directory.mkdir(mode=0o700)
            self.output_created = True
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closed = True
            return {"status": "PASS", "resources_released": True}

    async def fail(env, policy, mode, progress):
        progress["http_requests"] = 3
        record = CallRecord(
            process="worker",
            sequence=1,
            call="chat",
            ordinal=1,
            delay_seconds=0.0,
            phase="started",
            outcome="pending",
            elapsed_seconds=0.0,
        )
        publish(env.directory, "calls-worker-001-started.json", record)
        raise error

    monkeypatch.setattr(smoke, "IsolatedEnvironment", Environment)
    monkeypatch.setattr(smoke, "drive", fail)
    result = smoke.run_smoke(tmp_path / "report", mode="research")
    assert instances[0].closed and result.resources_released
    assert result.status == "IN_PROGRESS" and result.category == category
    assert result.http_requests == 3 and result.unfinished_calls == 1
    assert result.call_counts == {"chat": 1}
    assert result.mock_effects is None and result.model_attempts is None
    assert (tmp_path / "report" / "calls-worker-001-started.json").is_file()
    assert (
        json.loads((tmp_path / "report" / "smoke-result.json").read_text())["category"] == category
    )
    assert "PRIVATE_BODY_CANARY" not in result.model_dump_json()


def test_rejected_existing_directory_does_not_receive_smoke_artifacts(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    sentinel = directory / "sentinel.txt"
    sentinel.write_text("preserve")
    result = smoke.run_smoke(directory, mode="research")
    assert result.status == "IN_PROGRESS"
    assert list(directory.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserve"


async def test_evidence_missing_finish_is_unfinished_and_cannot_pass_reconciliation(tmp_path):
    record = CallRecord(
        process="worker",
        sequence=1,
        call="chat",
        ordinal=1,
        delay_seconds=0.0,
        phase="started",
        outcome="pending",
        elapsed_seconds=0.0,
    )
    publish(tmp_path, "calls-worker-001-started.json", record)
    assert smoke.partial_calls(tmp_path) == {"call_counts": {"chat": 1}, "unfinished_calls": 1}
    with pytest.raises(smoke.SmokeFailure):
        smoke.read_calls(tmp_path)


async def test_completed_call_evidence_identity_mismatch_is_rejected(tmp_path):
    calls = Calls(profile(), directory=tmp_path)
    await calls.invoke("search", lambda: asyncio.sleep(0))
    assert len(smoke.read_calls(tmp_path)) == 1
    # A different sequence in a correctly named, create-only new pair is still invalid.
    beginning = CallRecord(
        process="worker",
        sequence=3,
        call="chat",
        ordinal=1,
        delay_seconds=0.0,
        phase="started",
        outcome="pending",
        elapsed_seconds=0.0,
    )
    publish(tmp_path, "calls-worker-002-started.json", beginning)
    publish(
        tmp_path,
        "calls-worker-002-finished.json",
        beginning.model_copy(update={"phase": "finished", "outcome": "succeeded"}),
    )
    with pytest.raises(smoke.SmokeFailure):
        smoke.read_calls(tmp_path)


@pytest.mark.parametrize("field", ["workspace_id", "created_by_user_id", "mode", "status"])
async def test_synthetic_approval_rejects_foreign_or_ineligible_run_before_http(field):
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    run = SimpleNamespace(
        workspace_id=tenant.workspace_id,
        created_by_user_id=tenant.actor_user_id,
        mode="application",
        status="waiting_approval",
    )
    setattr(run, field, uuid4() if field.endswith("id") else "research")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args):
            return run

    http = Mock()
    with pytest.raises(smoke.SmokeFailure):
        await smoke.approve_synthetic(http, Session, tenant, uuid4())
    http.assert_not_called()
    assert http.mock_calls == []


def test_primary_failure_survives_cleanup_and_publication_failures(tmp_path, monkeypatch):
    class Environment:
        def __init__(self, _, directory, **kwargs):
            self.directory = directory
            self.output_created = False

        def __enter__(self):
            self.directory.mkdir(mode=0o700)
            self.output_created = True
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            return {"status": "IN_PROGRESS", "resources_released": False}

    async def fail(*args):
        raise smoke.SmokeFailure("http_failed")

    original = smoke.publish

    def publish_failure(directory, name, value):
        if name == "smoke-result.json":
            raise environment.EnvironmentError("report_failed")
        original(directory, name, value)

    monkeypatch.setattr(smoke, "IsolatedEnvironment", Environment)
    monkeypatch.setattr(smoke, "drive", fail)
    monkeypatch.setattr(smoke, "publish", publish_failure)
    result = smoke.run_smoke(tmp_path / "report", mode="research")
    assert result.category == "http_failed" and result.failure_stage == "drive"
    assert result.status == "IN_PROGRESS"
    assert set(result.diagnostic_errors) == {
        "cleanup_failed",
        "metrics_incomplete",
        "report_failed",
    }
    assert set(result.missing_roles) == {"api", "worker", "driver", "supervisor"}
    assert "missing_role" in result.metrics_reasons
