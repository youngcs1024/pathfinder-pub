"""E5.2 environment lifecycle, with no arbitrary DSN or load execution interface."""

from __future__ import annotations

import json
import os
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
POSTGRES_IMAGE = "pgvector/pgvector:0.8.5-pg16"
PROFILE = "environment-v1"
START_SECONDS = 120
RUN_SECONDS = 240
QUEUE_PROFILE = "environment-queue-v1"
QUEUE_RUN_SECONDS = 420
STOP_SECONDS = 20
CLEANUP_SECONDS = 70
PACKET_LIMIT = 8192
CATEGORIES = frozenset(
    {
        "invalid_profile",
        "unsafe_directory",
        "docker_unavailable",
        "invalid_socket",
        "already_started",
        "ownership_mismatch",
        "unsafe_binding",
        "startup_failed",
        "startup_timeout",
        "runtime_timeout",
        "process_exited",
        "cleanup_failed",
        "protocol_failed",
        "cancelled",
        "report_failed",
    }
)


class EnvironmentError(Exception):
    """Only fixed diagnostic categories may cross the harness boundary."""

    def __init__(self, category: str) -> None:
        self.category = category if category in CATEGORIES else "protocol_failed"
        super().__init__(self.category)


@dataclass(frozen=True)
class EnvironmentProfile:
    name: Literal["environment-v1", "environment-queue-v1"]

    def __post_init__(self) -> None:
        if self.name not in {PROFILE, QUEUE_PROFILE}:
            raise EnvironmentError("invalid_profile")


def create_output_directory(path: Path) -> Path:
    """Reject symlinks and existing destinations before creating any resource."""
    try:
        if not path.is_absolute() or ".." in path.parts:
            raise EnvironmentError("unsafe_directory")
        for part in (*reversed(path.parents), path):
            if part.is_symlink():
                raise EnvironmentError("unsafe_directory")
        if path.exists() or path.is_relative_to(ROOT) or ROOT.is_relative_to(path):
            raise EnvironmentError("unsafe_directory")
        parent = path.parent
        info = parent.stat()
        forbidden = (
            "/etc",
            "/proc",
            "/sys",
            "/dev",
            "/usr",
            "/bin",
            "/sbin",
            "/boot",
            "/run",
            "/var",
        )
        if (
            len(path.parts) < 4
            or any(
                part in {".git", ".ssh", ".aws", ".config", ".codex", ".agents", ".venv"}
                for part in path.parts
            )
            or any(path.is_relative_to(item) for item in forbidden)
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise EnvironmentError("unsafe_directory")
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
        return path
    except OSError:
        raise EnvironmentError("unsafe_directory") from None


def validate_docker_socket(path: Path) -> str:
    try:
        if not path.is_absolute() or ".." in path.parts or not stat.S_ISSOCK(path.stat().st_mode):
            raise EnvironmentError("invalid_socket")
        if not os.access(path, os.R_OK | os.W_OK):
            raise EnvironmentError("docker_unavailable")
        return "unix://" + str(path)
    except OSError:
        raise EnvironmentError("docker_unavailable") from None


def child_environment() -> dict[str, str]:
    # Construct from constants; no inherited PF_*, proxies, credentials or Python hooks.
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PF_LLM_MODE": "fake",
        "PF_SEARCH_MODE": "fake",
        "PF_AUTH_MODE": "fake",
        "PF_TRACE_MODE": "off",
    }


def send_packet(channel: socket.socket, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > PACKET_LIMIT or channel.send(data) != len(data):
        raise EnvironmentError("protocol_failed")


def receive_packet(channel: socket.socket, timeout: float) -> dict:
    if not select.select([channel], [], [], max(0, timeout))[0]:
        raise EnvironmentError("startup_timeout")
    data = channel.recv(PACKET_LIMIT + 1)
    try:
        payload = json.loads(data)
        if len(data) > PACKET_LIMIT or not isinstance(payload, dict):
            raise ValueError
        return payload
    except (ValueError, UnicodeError):
        raise EnvironmentError("protocol_failed") from None


@dataclass(repr=False)
class OwnedProcess:
    """A pidfd is bound to the actual child, including across PID reuse."""

    process: subprocess.Popen
    pidfd: int
    closed: bool = False
    released: bool = False

    @classmethod
    def launch(cls, role: str, *, env: dict[str, str], bootstrap: dict, fds: tuple[int, ...] = ()):
        if role not in {"supervisor", "api", "worker", "migrate"}:
            raise EnvironmentError("protocol_failed")
        module = (
            "tests.performance._supervisor"
            if role == "supervisor"
            else "tests.performance._process"
        )
        process = subprocess.Popen(
            [sys.executable, "-m", module, role],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=fds,
        )
        try:
            pidfd = os.pidfd_open(process.pid)
        except OSError:
            # Closing bootstrap stdin asks this fixed entrypoint to exit before resource use.
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                raise EnvironmentError("cleanup_failed") from None
            raise EnvironmentError("process_exited") from None
        owned = cls(process, pidfd)
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(bootstrap).encode())
            process.stdin.close()
        except (OSError, ValueError):
            released = owned.close()
            raise EnvironmentError("startup_failed" if released else "cleanup_failed") from None
        return owned

    def request_stop(self) -> None:
        if not self.closed and self.process.poll() is None:
            try:
                signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def close(self) -> bool:
        if self.closed:
            return self.released
        try:
            if self.process.poll() is None:
                try:
                    signal.pidfd_send_signal(self.pidfd, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Child exited between poll and signal; wait still reaps this child.
                try:
                    self.process.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
            self.released = self.process.poll() is not None
            return self.released
        except (OSError, subprocess.SubprocessError):
            return False
        finally:
            os.close(self.pidfd)
            self.closed = True


def validate_record(payload: dict) -> None:
    """E5.2 metadata only: allowlisted fields AND constrained values, never body redaction."""
    fixed = {
        "schema_version": {1},
        "profile": {PROFILE, QUEUE_PROFILE},
        "image": {POSTGRES_IMAGE},
        "status": {"IN_PROGRESS", "PASS", "BLOCKED"},
        "kind": {"result"},
        "category": CATEGORIES | {None},
        "primary_category": CATEGORIES | {None},
        "startup_limit_seconds": {START_SECONDS},
        "runtime_limit_seconds": {RUN_SECONDS, QUEUE_RUN_SECONDS},
        "database_cpu": {1},
        "database_memory_bytes": {1024**3},
    }
    patterns = {
        "owner": r"[0-9a-f]{32}",
        "container_id": r"[0-9a-f]{64}",
        "container_name": r"pf-e52-[0-9a-f]{32}",
        "database_name": r"pf_e52_[0-9a-f]{32}",
        "api_origin": r"http://127\.0\.0\.1:[1-9][0-9]{0,4}",
    }
    if not isinstance(payload, dict):
        raise ValueError
    for key, value in payload.items():
        if key in fixed:
            if type(value) not in {str, int, type(None)} or value not in fixed[key]:
                raise ValueError
        elif key in patterns:
            if not isinstance(value, str) or re.fullmatch(patterns[key], value) is None:
                raise ValueError
            if key == "api_origin" and int(value.rsplit(":", 1)[1]) > 65535:
                raise ValueError
        elif key == "identity":
            if not isinstance(value, dict) or set(value) - {
                "schema_version",
                "owner",
                "profile",
                "image",
                "container_name",
                "container_id",
                "database_name",
                "database_port",
                "api_origin",
                "process_ids",
            }:
                raise ValueError
            validate_record(value)
        elif key == "process_ids":
            if (
                not isinstance(value, dict)
                or set(value) - {"api", "worker", "migrate"}
                or any(type(pid) is not int or pid <= 0 for pid in value.values())
            ):
                raise ValueError
        elif key == "database_port":
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError
        elif key == "resources_released":
            if type(value) is not bool:
                raise ValueError
        elif key == "cleanup_failures":
            if not isinstance(value, list) or any(
                item
                not in {
                    "api",
                    "worker",
                    "migrate",
                    "database",
                    "docker_connection",
                    "listener",
                    "deadline",
                }
                for item in value
            ):
                raise ValueError
        elif key == "elapsed_seconds":
            if type(value) not in {int, float} or not isfinite(value) or value < 0:
                raise ValueError
        else:
            raise ValueError


def write_record(directory: Path, name: str, payload: dict) -> None:
    """Create once after validating the complete body-free record, including values."""
    try:
        if name not in {
            "started.json",
            "resources.json",
            "ready.json",
            "result.json",
            "observer.json",
            "config.json",
        }:
            raise ValueError
        if name == "config.json" and payload != {}:
            raise ValueError
        validate_record(payload)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor
            )
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
    except (OSError, ValueError, TypeError):
        raise EnvironmentError("report_failed") from None


@dataclass(repr=False)
class IsolatedEnvironment:
    profile: EnvironmentProfile
    output_dir: Path
    docker_socket: Path = Path("/var/run/docker.sock")
    call_profile: dict | None = None
    metrics: bool = False
    capacity: bool = False
    fault_config: dict | None = None
    _ipc_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _process: OwnedProcess | None = field(default=None, init=False)
    _channel: socket.socket | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)
    _identity: dict = field(default_factory=dict, init=False)
    _database_url: str | None = field(default=None, init=False)
    _result: dict | None = field(default=None, init=False)
    _output_created: bool = field(default=False, init=False)

    @property
    def output_created(self) -> bool:
        return self._output_created

    @property
    def identity(self) -> dict:
        return dict(self._identity)

    @property
    def api_origin(self) -> str:
        return self._identity["api_origin"]

    @property
    def database_url(self) -> str:
        """Private runtime capability; never serialized into public identity/reports."""
        if self._database_url is None:
            raise EnvironmentError("startup_failed")
        return self._database_url

    def start(self) -> IsolatedEnvironment:
        if self._started or self._closed:
            raise EnvironmentError("already_started")
        if type(self.profile) is not EnvironmentProfile or self.profile.name not in {
            PROFILE,
            QUEUE_PROFILE,
        }:
            raise EnvironmentError("invalid_profile")
        if type(self.metrics) is not bool or type(self.capacity) is not bool:
            raise EnvironmentError("invalid_profile")
        if self.fault_config is not None:
            from tests.performance.fault_contracts import parse_config

            self.fault_config = parse_config(self.fault_config).model_dump(mode="json")
            if (
                not (self.capacity and self.metrics)
                or self.profile.name != PROFILE
                or self.call_profile is None
            ):
                raise EnvironmentError("invalid_profile")
        if self.profile.name == QUEUE_PROFILE and not (self.capacity and self.metrics):
            raise EnvironmentError("invalid_profile")
        if self.call_profile is not None:
            from tests.performance.workload import parse_profile

            self.call_profile = parse_profile(self.call_profile).model_dump(mode="json")
        if self.profile.name == QUEUE_PROFILE and (
            self.call_profile is None or self.call_profile.get("schema_version") != 2
        ):
            raise EnvironmentError("invalid_profile")
        self._started = True
        directory = create_output_directory(self.output_dir)
        self._output_created = True
        try:
            docker_host = validate_docker_socket(self.docker_socket)
        except EnvironmentError as exc:
            self._result = {
                "profile": PROFILE,
                "status": "BLOCKED",
                "category": exc.category,
                "resources_released": True,
            }
            write_record(directory, "result.json", self._result)
            raise
        # docker-py falls back to ~/.docker when DOCKER_CONFIG has no config.json.
        # An existing empty config blocks auth/proxy/context inheritance without changing HOME.
        write_record(directory, "config.json", {})
        if self.profile.name == QUEUE_PROFILE and not (self.capacity and self.metrics):
            raise EnvironmentError("invalid_profile")
        if self.call_profile is not None:
            from tests.performance.workload import parse_profile, publish

            publish(directory, "call-profile.json", parse_profile(self.call_profile))
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self._channel = parent
        owner = uuid4().hex
        env = child_environment()
        env.update(
            {
                "DOCKER_HOST": docker_host,
                "DOCKER_CONFIG": str(directory),
                "TESTCONTAINERS_RYUK_DISABLED": "true",
            }
        )
        try:
            self._process = OwnedProcess.launch(
                "supervisor",
                env=env,
                bootstrap={
                    "owner": owner,
                    "output_dir": str(directory),
                    "channel_fd": child.fileno(),
                    "queue": self.profile.name == QUEUE_PROFILE,
                    "metrics": self.metrics,
                    "capacity": self.capacity,
                    **({"fault_config": self.fault_config} if self.fault_config else {}),
                    **(
                        {"call_profile": self.call_profile} if self.call_profile is not None else {}
                    ),
                },
                fds=(child.fileno(),),
            )
            child.close()
            packet = receive_packet(parent, START_SECONDS + CLEANUP_SECONDS + 5)
            if packet.get("kind") != "ready":
                self._result = packet
                raise EnvironmentError(packet.get("category", "startup_failed"))
            if packet.get("identity", {}).get("owner") != owner:
                raise EnvironmentError("ownership_mismatch")
            self._identity = packet["identity"]
            self._database_url = packet["database_url"]
            return self
        except BaseException as exc:
            child.close()
            self.close()
            if isinstance(exc, Exception) and not isinstance(exc, EnvironmentError):
                raise EnvironmentError("startup_failed") from None
            raise

    def capacity_command(self, kind: str) -> dict:
        if not self.capacity or not self._started or self._closed:
            raise EnvironmentError("protocol_failed")
        if (
            kind not in {"start_worker", "health"}
            and not (kind == "stop_worker" and self.profile.name == QUEUE_PROFILE)
            and not (
                self.fault_config is not None
                and kind in {"fault_poll", "fault_release", "fault_kill"}
            )
        ):
            raise EnvironmentError("protocol_failed")
        with self._ipc_lock:
            send_packet(self._channel, {"kind": kind})
            packet = receive_packet(self._channel, 30 if kind == "stop_worker" else 15)
            if packet.get("kind") != kind or packet.get("owner") != self._identity["owner"]:
                self._result = packet if packet.get("kind") == "result" else None
                raise EnvironmentError("protocol_failed")
            if kind == "start_worker":
                self._identity["process_ids"]["worker"] = packet["pid"]
            return packet

    def close(self) -> dict:
        if self._closed:
            return self._result or {"resources_released": False}
        self._closed = True
        if self._channel is None and self._process is None:
            self._result = self._result or {"resources_released": True}
            return dict(self._result)
        if self._channel is not None:
            try:
                if self._result is None:
                    try:
                        send_packet(self._channel, {"kind": "stop"})
                    except OSError:
                        pass  # A final packet can still be queued after the supervisor exits.
                    if not self._identity and self._process is not None:
                        self._process.request_stop()  # Interrupt startup, then allow full cleanup.
                    deadline = time.monotonic() + CLEANUP_SECONDS + 5
                    while True:
                        packet = receive_packet(self._channel, deadline - time.monotonic())
                        if packet.get("kind") == "ready":
                            continue  # Startup raced with stop; require the final cleanup ACK.
                        if packet.get("kind") != "result":
                            raise EnvironmentError("protocol_failed")
                        self._result = packet
                        break
            except (OSError, EnvironmentError):
                self._result = {
                    "kind": "result",
                    "status": "IN_PROGRESS",
                    "category": "cleanup_failed",
                    "resources_released": False,
                }
            finally:
                self._channel.close()
        process_stopped = self._process.close() if self._process is not None else True
        if not process_stopped or self._result is None:
            self._result = {
                "kind": "result",
                "status": "IN_PROGRESS",
                "category": "cleanup_failed",
                "resources_released": False,
            }
        self._database_url = None
        if not self._result.get("resources_released"):
            # Separate create-only observer record if the supervisor cannot confirm cleanup.
            write_record(
                self.output_dir,
                "observer.json",
                {
                    "status": "IN_PROGRESS",
                    "category": "cleanup_failed",
                    "resources_released": False,
                },
            )
        return dict(self._result)

    def __enter__(self) -> IsolatedEnvironment:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        result = self.close()
        if exc_type is None and (not result.get("resources_released") or result.get("category")):
            raise EnvironmentError(result.get("category", "cleanup_failed"))
