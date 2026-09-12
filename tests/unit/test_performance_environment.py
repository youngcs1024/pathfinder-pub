"""Fast environment boundary and failure tests, executed by the existing unit CI target."""

from __future__ import annotations

import json
import signal
import socket
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tests.performance import _supervisor as supervisor
from tests.performance import environment as env

OWNER = "a" * 32
CANARY = "E52_SECRET_CANARY_MUST_NOT_ESCAPE"


@pytest.fixture
def channel():
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        yield parent, child
    finally:
        parent.close()
        child.close()


def test_profile_has_no_target_or_load_options():
    assert env.EnvironmentProfile("environment-v1").name == env.PROFILE
    with pytest.raises(env.EnvironmentError, match="invalid_profile"):
        env.EnvironmentProfile(CANARY)
    with pytest.raises(TypeError):
        env.EnvironmentProfile("environment-v1", database_url="postgresql://production")
    with pytest.raises(TypeError):
        env.IsolatedEnvironment(
            env.EnvironmentProfile(env.PROFILE), Path("/tmp/new"), api_host="0.0.0.0"
        )


def test_child_environment_is_constructed_not_filtered(monkeypatch):
    for key in (
        "PF_DATABASE_URL",
        "PF_QWEN_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "PYTHONPATH",
        "PYTHONHOME",
        "DOCKER_HOST",
        "DOCKER_AUTH_CONFIG",
        "TESTCONTAINERS_HOST_OVERRIDE",
        "PF_LLM_MODE",
    ):
        monkeypatch.setenv(key, CANARY)
    child = env.child_environment()
    assert CANARY not in json.dumps(child)
    assert child["PF_LLM_MODE"] == child["PF_SEARCH_MODE"] == child["PF_AUTH_MODE"] == "fake"
    assert child["PF_TRACE_MODE"] == "off"
    assert child["PYTHONPATH"] == str(env.ROOT)
    assert "PF_DATABASE_URL" not in child
    assert set(child) == {
        "PATH",
        "LANG",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PF_LLM_MODE",
        "PF_SEARCH_MODE",
        "PF_AUTH_MODE",
        "PF_TRACE_MODE",
    }


def test_output_is_private_create_only_and_retained(tmp_path):
    destination = env.create_output_directory(tmp_path / "report")
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    env.write_record(destination, "result.json", {"status": "IN_PROGRESS"})
    original = (destination / "result.json").read_bytes()
    assert stat.S_IMODE((destination / "result.json").stat().st_mode) == 0o600
    with pytest.raises(env.EnvironmentError, match="report_failed"):
        env.write_record(destination, "result.json", {"status": "PASS"})
    with pytest.raises(env.EnvironmentError, match="unsafe_directory"):
        env.create_output_directory(destination)
    assert (destination / "result.json").read_bytes() == original


@pytest.mark.parametrize("path", ["/etc/new", "/proc/new", "/tmp/new", "/", "relative/path"])
def test_dangerous_directory_rejected(path):
    with pytest.raises(env.EnvironmentError, match="unsafe_directory"):
        env.create_output_directory(Path(path))


def test_repo_existing_file_symlink_and_writable_parent_rejected(tmp_path):
    existing = tmp_path / "existing"
    existing.write_text(CANARY)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    for destination in (existing, link / "new", env.ROOT / "new-report", tmp_path / ".." / "new"):
        with pytest.raises(env.EnvironmentError, match="unsafe_directory"):
            env.create_output_directory(destination)
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with pytest.raises(env.EnvironmentError, match="unsafe_directory"):
        env.create_output_directory(unsafe / "new")
    assert existing.read_text() == CANARY


def test_socket_requires_local_unix_socket_and_permission(tmp_path, monkeypatch):
    socket_path = tmp_path / "docker.sock"
    with socket.socket(socket.AF_UNIX) as local:
        local.bind(str(socket_path))
        assert env.validate_docker_socket(socket_path) == "unix://" + str(socket_path)
        monkeypatch.setattr(env.os, "access", lambda *_: False)
        with pytest.raises(env.EnvironmentError, match="docker_unavailable"):
            env.validate_docker_socket(socket_path)
    regular = tmp_path / "file"
    regular.touch()
    with pytest.raises(env.EnvironmentError, match="invalid_socket"):
        env.validate_docker_socket(regular)
    for value in ("tcp://example.com:2375", "ssh://production", "relative"):
        with pytest.raises(env.EnvironmentError, match="invalid_socket"):
            env.validate_docker_socket(Path(value))


def test_missing_docker_records_blocked_before_process_launch(tmp_path, monkeypatch):
    launch = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(env.OwnedProcess, "launch", launch)
    environment = env.IsolatedEnvironment(
        env.EnvironmentProfile(env.PROFILE), tmp_path / "report", tmp_path / "missing.sock"
    )
    with pytest.raises(env.EnvironmentError, match="docker_unavailable"):
        environment.start()
    assert not launch.called
    record = json.loads((environment.output_dir / "result.json").read_text())
    assert record["status"] == "BLOCKED"
    assert record["resources_released"] is True
    assert "missing.sock" not in json.dumps(record)


def test_packet_boundaries(channel):
    parent, child = channel
    env.send_packet(parent, {"kind": "stop"})
    assert env.receive_packet(child, 1) == {"kind": "stop"}
    with pytest.raises(env.EnvironmentError, match="protocol_failed"):
        env.send_packet(parent, {"value": "x" * env.PACKET_LIMIT})
    parent.send(b"not json " + CANARY.encode())
    with pytest.raises(env.EnvironmentError, match="protocol_failed") as error:
        env.receive_packet(child, 1)
    assert CANARY not in str(error.value)
    with pytest.raises(env.EnvironmentError, match="startup_timeout"):
        env.receive_packet(child, 0)


def container_fixture():
    return SimpleNamespace(
        id="b" * 64,
        reload=Mock(),
        remove=Mock(),
        attrs={
            "Name": "/pf-e52-" + OWNER,
            "Config": {"Labels": {supervisor.LABEL: OWNER}, "Image": env.POSTGRES_IMAGE},
            "NetworkSettings": {
                "Ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "12345"}]}
            },
            "HostConfig": {"NetworkMode": "bridge", "NanoCpus": 1_000_000_000, "Memory": 1024**3},
        },
    )


def test_container_identity_requires_id_name_label_and_image():
    container = container_fixture()
    arguments = {"owner": OWNER, "name": "pf-e52-" + OWNER, "identifier": container.id}
    supervisor.verify_container(container, **arguments)
    assert supervisor.verified_port(container) == 12345
    for key, value in (("owner", "wrong"), ("name", "production"), ("identifier", "foreign")):
        with pytest.raises(env.EnvironmentError, match="ownership_mismatch"):
            supervisor.verify_container(container, **{**arguments, key: value})
    container.attrs["Config"]["Image"] = "postgres:latest"
    with pytest.raises(env.EnvironmentError, match="ownership_mismatch"):
        supervisor.verify_container(container, **arguments)
    assert not container.remove.called


@pytest.mark.parametrize(
    "binding",
    [
        None,
        [],
        [{"HostIp": "0.0.0.0", "HostPort": "12345"}],
        [{"HostIp": "::", "HostPort": "12345"}],
        [{"HostIp": "127.0.0.1", "HostPort": "0"}],
        [{"HostIp": "127.0.0.1", "HostPort": CANARY}],
    ],
)
def test_non_loopback_or_invalid_mapping_is_rejected(binding):
    container = container_fixture()
    container.attrs["NetworkSettings"]["Ports"]["5432/tcp"] = binding
    with pytest.raises(env.EnvironmentError, match="unsafe_binding"):
        supervisor.verified_port(container)


def test_extra_port_and_missing_resource_limits_rejected():
    container = container_fixture()
    container.attrs["NetworkSettings"]["Ports"]["8080/tcp"] = []
    with pytest.raises(env.EnvironmentError, match="unsafe_binding"):
        supervisor.verified_port(container)
    container = container_fixture()
    container.attrs["HostConfig"]["Memory"] = 0
    with pytest.raises(env.EnvironmentError, match="unsafe_binding"):
        supervisor.verified_port(container)


def test_owned_process_uses_pidfd_and_bounded_escalation(monkeypatch):
    process = Mock()
    process.poll.side_effect = [None, -9]
    process.wait.side_effect = [subprocess.TimeoutExpired("fixed", 20), -9]
    signals = Mock()
    monkeypatch.setattr(env.signal, "pidfd_send_signal", signals)
    close_fd = Mock()
    monkeypatch.setattr(env.os, "close", close_fd)
    owned = env.OwnedProcess(process, 42)
    assert owned.close()
    assert owned.close()
    assert signals.call_args_list == [call(42, signal.SIGTERM), call(42, signal.SIGKILL)]
    assert process.wait.call_args_list == [call(timeout=20), call(timeout=5)]
    close_fd.assert_called_once_with(42)


def test_failed_process_cleanup_never_becomes_success_on_retry(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(env.signal, "pidfd_send_signal", Mock(side_effect=OSError(CANARY)))
    monkeypatch.setattr(env.os, "close", Mock())
    owned = env.OwnedProcess(process, 42)
    assert owned.close() is False
    assert owned.close() is False


def test_cleanup_continues_after_worker_failure_and_refuses_foreign_container(tmp_path, channel):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    worker, api = Mock(), Mock()
    worker.close.return_value = False
    api.close.return_value = True
    controller.processes = {"worker": worker, "api": api}
    foreign = container_fixture()
    foreign.attrs["Config"]["Labels"][supervisor.LABEL] = "foreign"
    client = Mock()
    controller.container = SimpleNamespace(_container=foreign, get_docker_client=lambda: client)
    assert not controller.cleanup()
    assert controller.cleanup_failures == ["worker", "database"]
    assert not foreign.remove.called
    api.close.assert_called_once()
    client.client.close.assert_called_once()


def test_cleanup_only_removes_verified_created_container(tmp_path, channel):
    from docker.errors import NotFound

    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    container = container_fixture()
    container.reload.side_effect = [None, NotFound("gone")]
    client = Mock()
    controller.container = SimpleNamespace(_container=container, get_docker_client=lambda: client)
    controller.container_id = container.id
    assert controller.cleanup()
    container.remove.assert_called_once_with(force=True, v=True)


def test_lost_create_response_looks_up_exact_owned_name(tmp_path, channel):
    from docker.errors import NotFound

    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    container = container_fixture()
    container.reload.side_effect = [None, NotFound("gone")]
    client = Mock()
    client.client.containers.get.return_value = container
    controller.container = SimpleNamespace(_container=None, get_docker_client=lambda: client)
    controller.create_attempted = True
    assert controller.cleanup()
    client.client.containers.get.assert_called_once_with("pf-e52-" + OWNER)
    container.remove.assert_called_once()


def test_no_object_after_uncertain_create_is_reported_as_residual(tmp_path, channel):
    from docker.errors import NotFound

    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    client = Mock()
    client.client.containers.get.side_effect = NotFound(CANARY)
    controller.container = SimpleNamespace(_container=None, get_docker_client=lambda: client)
    controller.create_attempted = True
    assert controller.cleanup() is False
    assert controller.cleanup_failures == ["database"]


@pytest.mark.parametrize(
    "failure",
    [
        env.EnvironmentError("startup_timeout"),
        RuntimeError(CANARY),
        supervisor.LifecycleInterrupted(),
    ],
)
def test_partial_start_always_cleans_and_sanitizes(tmp_path, channel, monkeypatch, failure):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    monkeypatch.setattr(supervisor.signal, "signal", Mock())
    monkeypatch.setattr(supervisor.signal, "setitimer", Mock())
    cleanup = Mock(return_value=True)
    monkeypatch.setattr(controller, "cleanup", cleanup)
    monkeypatch.setattr(controller, "start", Mock(side_effect=failure))
    assert controller.run() == 1
    cleanup.assert_called_once()
    record = json.loads((tmp_path / "result.json").read_text())
    packet = env.receive_packet(channel[1], 1)
    assert record == packet
    assert record["resources_released"] is True
    assert record["status"] != "PASS"
    assert CANARY not in json.dumps(record)
    assert "password" not in json.dumps(record)
    assert controller.password != controller.owner


def test_runtime_deadline_reserves_cleanup_time(tmp_path, channel, monkeypatch):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    monkeypatch.setattr(
        supervisor.time,
        "monotonic",
        lambda: controller.started + env.RUN_SECONDS - env.CLEANUP_SECONDS,
    )
    assert controller.serve() == "runtime_timeout"


def test_worker_crash_ends_environment(tmp_path, channel):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    controller.processes = {"worker": Mock(), "api": Mock()}
    controller.processes["worker"].process.poll.return_value = 1
    assert controller.serve() == "process_exited"


def test_parent_disconnect_requests_cleanup(tmp_path, channel):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    controller.processes = {"worker": Mock(), "api": Mock()}
    for child in controller.processes.values():
        child.process.poll.return_value = None
    channel[1].close()
    assert controller.serve() == "cancelled"


def test_second_worker_rejected_before_launch(tmp_path, channel, monkeypatch):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    controller.processes["worker"] = Mock()
    launch = Mock()
    monkeypatch.setattr(env.OwnedProcess, "launch", launch)
    with pytest.raises(env.EnvironmentError, match="already_started"):
        controller.launch("worker", CANARY)
    launch.assert_not_called()


def test_observer_reports_missing_cleanup_ack_and_is_idempotent(tmp_path, channel, monkeypatch):
    environment = env.IsolatedEnvironment(env.EnvironmentProfile(env.PROFILE), tmp_path)
    environment._channel = channel[0]
    environment._process = Mock()
    environment._process.close.return_value = True
    monkeypatch.setattr(
        env, "receive_packet", Mock(side_effect=env.EnvironmentError("protocol_failed"))
    )
    result = environment.close()
    assert result["resources_released"] is False
    assert environment.close() == result
    assert json.loads((tmp_path / "observer.json").read_text())["category"] == "cleanup_failed"


def test_parent_startup_failure_preserves_result_and_stops_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "validate_docker_socket", lambda _: "unix:///fixed.sock")
    process = Mock()
    process.close.return_value = True
    monkeypatch.setattr(env.OwnedProcess, "launch", Mock(return_value=process))
    packet = {"kind": "result", "category": "startup_failed", "resources_released": True}
    monkeypatch.setattr(env, "receive_packet", Mock(return_value=packet))
    environment = env.IsolatedEnvironment(env.EnvironmentProfile(env.PROFILE), tmp_path / "new")
    with pytest.raises(env.EnvironmentError, match="startup_failed"):
        environment.start()
    process.close.assert_called_once()
    with pytest.raises(env.EnvironmentError, match="already_started"):
        environment.start()


@pytest.mark.parametrize("safe_binding", [True, False])
def test_database_constructor_and_verification_precede_sql(
    tmp_path, channel, monkeypatch, safe_binding
):
    from testcontainers.community import postgres
    from testcontainers.core.config import testcontainers_config
    from testcontainers.core.container import DockerContainer

    monkeypatch.setattr(testcontainers_config, "tc_properties", {"tc.host": "tcp://production"})
    monkeypatch.setattr(testcontainers_config, "ryuk_disabled", False)
    wrapped = container_fixture()
    if not safe_binding:
        wrapped.attrs["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostIp"] = "0.0.0.0"
    container = SimpleNamespace(get_wrapped_container=lambda: wrapped, _connect=Mock())
    constructor = Mock(return_value=container)
    monkeypatch.setattr(postgres, "PostgresContainer", constructor)
    monkeypatch.setattr(DockerContainer, "start", Mock())
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    if safe_binding:
        private_url = controller.create_database()
        assert private_url.startswith("postgresql+psycopg://pf_e52:")
        container._connect.assert_called_once()
        record = (tmp_path / "resources.json").read_text()
        assert controller.password not in record
        assert private_url not in record
    else:
        with pytest.raises(env.EnvironmentError, match="unsafe_binding"):
            controller.create_database()
        container._connect.assert_not_called()
    assert container.ports == {"5432/tcp": ("127.0.0.1", 0)}
    assert container.tmpfs == {"/var/lib/postgresql/data": "rw,size=268435456"}
    assert constructor.call_args.args == (env.POSTGRES_IMAGE,)
    kwargs = constructor.call_args.kwargs
    assert kwargs["dbname"] == "pf_e52_" + OWNER
    assert kwargs["password"] == controller.password != OWNER
    assert kwargs["network_mode"] == "bridge"
    assert kwargs["nano_cpus"] == 1_000_000_000
    assert kwargs["mem_limit"] == 1024**3
    assert kwargs["labels"] == {supervisor.LABEL: OWNER}
    assert testcontainers_config.tc_properties == {}
    assert testcontainers_config.ryuk_disabled is True


def test_process_launch_failure_is_safe_and_closes_bootstrap(tmp_path, monkeypatch):
    process = Mock()
    process.pid = 123
    process.wait.return_value = 1
    popen = Mock(return_value=process)
    monkeypatch.setattr(env.subprocess, "Popen", popen)
    monkeypatch.setattr(env.os, "pidfd_open", Mock(side_effect=OSError(CANARY)))
    with pytest.raises(env.EnvironmentError, match="process_exited") as failure:
        env.OwnedProcess.launch("api", env=env.child_environment(), bootstrap={"owner": OWNER})
    assert CANARY not in str(failure.value)
    process.stdin.close.assert_called_once()
    kwargs = popen.call_args.kwargs
    assert kwargs["close_fds"] is True
    assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert "shell" not in kwargs
    assert kwargs["cwd"] == env.ROOT


def test_clean_shutdown_result_and_parent_ack(tmp_path, channel, monkeypatch):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    monkeypatch.setattr(supervisor.signal, "signal", Mock())
    monkeypatch.setattr(supervisor.signal, "setitimer", Mock())
    monkeypatch.setattr(controller, "start", Mock(return_value=CANARY))
    monkeypatch.setattr(controller, "serve", Mock(return_value=None))
    monkeypatch.setattr(controller, "cleanup", Mock(return_value=True))
    assert controller.run() == 0
    ready = env.receive_packet(channel[1], 1)
    assert ready["database_url"] == CANARY  # Private IPC only.
    result = env.receive_packet(channel[1], 1)
    assert result["status"] == "PASS"
    assert result["resources_released"] is True
    assert CANARY not in (tmp_path / "result.json").read_text()


def test_cleanup_deadline_preserves_failure_evidence(tmp_path, channel, monkeypatch):
    controller = supervisor.Supervisor(tmp_path, OWNER, channel[0])
    monkeypatch.setattr(supervisor.signal, "signal", Mock())
    monkeypatch.setattr(supervisor.signal, "setitimer", Mock())
    monkeypatch.setattr(controller, "start", Mock(side_effect=RuntimeError(CANARY)))
    monkeypatch.setattr(controller, "cleanup", Mock(side_effect=supervisor.LifecycleInterrupted()))
    assert controller.run() == 1
    record = json.loads((tmp_path / "result.json").read_text())
    assert record["resources_released"] is False
    assert record["category"] == "cleanup_failed"
    assert record["cleanup_failures"] == ["deadline"]
    assert CANARY not in json.dumps(record)


@pytest.mark.parametrize(
    "payload",
    [
        {"database_url": CANARY},
        {"category": CANARY},
        {"image": CANARY},
        {"identity": {"database_url": CANARY}},
        {"owner": CANARY},
        {"process_ids": {"api": CANARY}},
        {"api_origin": "http://example.com:8000"},
        {"elapsed_seconds": float("nan")},
        {"cleanup_failures": [CANARY]},
        {"primary_category": CANARY},
    ],
)
def test_reports_reject_unknown_fields_and_unsafe_values_before_writing(tmp_path, payload):
    with pytest.raises(env.EnvironmentError, match="report_failed") as failure:
        env.write_record(tmp_path, "result.json", payload)
    assert CANARY not in str(failure.value)
    assert not list(tmp_path.iterdir())


def test_record_cannot_escape_directory(tmp_path):
    with pytest.raises(env.EnvironmentError, match="report_failed"):
        env.write_record(tmp_path, "../outside.json", {"status": "PASS"})
    assert not list(tmp_path.iterdir())


def test_container_identifier_cannot_smuggle_body_into_report():
    container = container_fixture()
    container.id = CANARY
    with pytest.raises(env.EnvironmentError, match="ownership_mismatch"):
        supervisor.verify_container(container, owner=OWNER, name="pf-e52-" + OWNER, identifier=None)


def test_parent_stop_waits_past_racing_ready_for_final_ack(tmp_path, channel, monkeypatch):
    environment = env.IsolatedEnvironment(env.EnvironmentProfile(env.PROFILE), tmp_path)
    environment._channel = channel[0]
    environment._process = Mock()
    environment._process.close.return_value = True
    result = {"kind": "result", "status": "PASS", "category": None, "resources_released": True}
    monkeypatch.setattr(env, "receive_packet", Mock(side_effect=[{"kind": "ready"}, result]))
    assert environment.close() == result
    environment._process.request_stop.assert_called_once()
    assert not list(tmp_path.iterdir())


def test_parent_reads_queued_result_even_after_peer_closes(tmp_path, channel):
    environment = env.IsolatedEnvironment(env.EnvironmentProfile(env.PROFILE), tmp_path)
    environment._channel = channel[0]
    environment._process = Mock()
    environment._process.close.return_value = True
    result = {
        "kind": "result",
        "status": "IN_PROGRESS",
        "category": "runtime_timeout",
        "resources_released": True,
    }
    env.send_packet(channel[1], result)
    channel[1].close()
    assert environment.close() == result
    assert not list(tmp_path.iterdir())


def test_sensitive_configuration_directory_rejected(tmp_path):
    parent = tmp_path / ".ssh"
    parent.mkdir(mode=0o700)
    with pytest.raises(env.EnvironmentError, match="unsafe_directory"):
        env.create_output_directory(parent / "report")


def test_empty_docker_config_blocks_home_credentials_fallback(tmp_path, monkeypatch):
    from docker.utils import config

    home = tmp_path / "home"
    docker_home = home / ".docker"
    docker_home.mkdir(parents=True)
    (docker_home / "config.json").write_text(json.dumps({"auths": {"registry": CANARY}}))
    report = env.create_output_directory(tmp_path / "report")
    env.write_record(report, "config.json", {})
    monkeypatch.setattr(config, "home_dir", lambda: str(home))
    monkeypatch.setenv("DOCKER_CONFIG", str(report))
    assert config.find_config_file() == str(report / "config.json")
    assert config.load_general_config() == {}
