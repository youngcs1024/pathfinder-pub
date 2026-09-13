"""Driver safety/failure tests use no real environment; actual flow is in integration core."""

import asyncio
import json
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
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

        def start(self):
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

        def start(self):
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


@pytest.mark.parametrize(
    "primary", [None, smoke.SmokeFailure("http_failed"), asyncio.CancelledError()]
)
async def test_all_drive_finalizers_run_without_replacing_primary_failure(primary):
    sampler = asyncio.get_running_loop().create_future()
    sampler.set_exception(RuntimeError("PRIVATE_SAMPLER_CANARY"))
    metrics = SimpleNamespace(
        finish=Mock(side_effect=ValueError("PRIVATE_METRICS_CANARY")), write_failed=False
    )
    engine = SimpleNamespace(dispose=AsyncMock(side_effect=OSError("PRIVATE_DISPOSAL_CANARY")))
    progress = {}
    stop = asyncio.Event()

    async def execute():
        await smoke.finish_drive(
            engine, metrics, sampler, stop, progress, monotonic(), primary=primary
        )

    if primary is None:
        with pytest.raises(smoke.SmokeFailure, match="evidence_mismatch"):
            await execute()
    else:
        # Returning permits the enclosing finally to propagate its original exception.
        await execute()
    assert stop.is_set()
    metrics.finish.assert_called_once()
    engine.dispose.assert_awaited_once()
    assert set(progress["_diagnostic_errors"]) == {"metrics_incomplete", "cleanup_failed"}
    assert progress["_metrics_write_failed"]
    assert "CANARY" not in json.dumps(progress)


async def test_caller_cancellation_during_sampler_shutdown_still_disposes_engine():
    entered = asyncio.Event()

    async def sampling():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(60)

    sampler = asyncio.create_task(sampling())
    await entered.wait()
    engine = SimpleNamespace(dispose=AsyncMock())
    metrics = SimpleNamespace(finish=Mock(), write_failed=False)
    task = asyncio.create_task(
        smoke.finish_drive(engine, metrics, sampler, asyncio.Event(), {}, monotonic(), primary=None)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    engine.dispose.assert_awaited_once()
    metrics.finish.assert_called_once()


@pytest.mark.parametrize(
    "error", [smoke.SmokeFailure("http_failed"), KeyboardInterrupt(), asyncio.CancelledError()]
)
def test_cleanup_exception_cannot_replace_drive_failure_or_interrupt(tmp_path, monkeypatch, error):
    close = Mock(side_effect=OSError("PRIVATE_CLEANUP_CANARY"))

    class Environment:
        output_created = False

        def __init__(self, _, directory, **kwargs):
            self.directory = directory

        def start(self):
            self.directory.mkdir(mode=0o700)
            self.output_created = True

        def close(self):
            return close()

    monkeypatch.setattr(smoke, "IsolatedEnvironment", Environment)
    monkeypatch.setattr(smoke, "drive", AsyncMock(side_effect=error))
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        with pytest.raises(type(error)):
            smoke.run_smoke(tmp_path / "report", mode="research")
    else:
        result = smoke.run_smoke(tmp_path / "report", mode="research")
        assert result.category == "http_failed" and result.failure_stage == "drive"
    close.assert_called_once()
    data = json.loads((tmp_path / "report" / "smoke-result.json").read_text())
    assert data["status"] == "IN_PROGRESS" and not data["resources_released"]
    assert data["category"] == (
        "cancelled"
        if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
        else "http_failed"
    )
    assert "cleanup_failed" in data["diagnostic_errors"]
    assert "CANARY" not in json.dumps(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"--authorization": None},
        {"--authorization": "e59_local_faults_user_approved_v1"},
        {"--profile": "capacity-e56-v1"},
        {"--profile": "delayed-v1"},
        {"--mode": "invalid"},
        {"--output": "relative-output"},
    ],
)
def test_smoke_cli_invalid_arguments_never_create_environment(tmp_path, monkeypatch, changes):
    from tests.performance.__main__ import main

    constructor = Mock()
    monkeypatch.setattr(smoke, "IsolatedEnvironment", constructor)
    options = {
        "--profile": "instant-v1",
        "--authorization": "e510_local_smoke_user_approved_v1",
        "--output": str(tmp_path / "new"),
    }
    options.update(changes)
    args = [
        "smoke",
        *(part for key, value in options.items() if value is not None for part in (key, value)),
    ]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    constructor.assert_not_called()
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "error,code",
    [
        (RuntimeError("PRIVATE_CLI_CANARY"), 1),
        (KeyboardInterrupt(), 130),
        (asyncio.CancelledError(), 130),
    ],
)
def test_smoke_cli_sanitizes_exceptions_and_preserves_interrupt_exit(
    tmp_path, monkeypatch, capsys, error, code
):
    from tests.performance.__main__ import main

    monkeypatch.setattr(smoke, "run_smoke", Mock(side_effect=error))
    assert (
        main(
            [
                "smoke",
                "--profile",
                "instant-v1",
                "--authorization",
                "e510_local_smoke_user_approved_v1",
                "--output",
                str(tmp_path / "new"),
            ]
        )
        == code
    )
    assert "CANARY" not in capsys.readouterr().out


def test_smoke_cli_existing_output_fails_without_touching_evidence(tmp_path):
    from tests.performance.__main__ import main

    marker = tmp_path / "evidence.json"
    marker.write_text("retained")
    assert (
        main(
            [
                "smoke",
                "--profile",
                "instant-v1",
                "--authorization",
                "e510_local_smoke_user_approved_v1",
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == "retained"


def test_successful_drive_without_required_metrics_cannot_pass(tmp_path, monkeypatch):
    class Environment:
        output_created = False

        def __init__(self, _, directory, **kwargs):
            self.directory = directory

        def start(self):
            self.directory.mkdir(mode=0o700)
            self.output_created = True

        def close(self):
            return {"status": "PASS", "resources_released": True}

    monkeypatch.setattr(smoke, "IsolatedEnvironment", Environment)
    monkeypatch.setattr(smoke, "drive", AsyncMock())
    result = smoke.run_smoke(tmp_path / "new", mode="research")
    assert result.status == "IN_PROGRESS" and result.metrics_status == "IN_PROGRESS"
    assert result.category == "evidence_mismatch" and result.failure_stage == "metrics"
    assert result.resources_released and result.missing_roles
    assert "metrics_incomplete" in result.diagnostic_errors
