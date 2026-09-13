"""Private supervisor entrypoint: one owned DB, API and idle worker, bounded lifetime."""

from __future__ import annotations

import json
import re
import select
import signal
import socket
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

from tests.performance.environment import (
    CLEANUP_SECONDS,
    POSTGRES_IMAGE,
    PROFILE,
    QUEUE_PROFILE,
    QUEUE_RUN_SECONDS,
    RUN_SECONDS,
    START_SECONDS,
    EnvironmentError,
    OwnedProcess,
    child_environment,
    receive_packet,
    send_packet,
    write_record,
)

LABEL = "pathfinder.e52.owner"


class LifecycleInterrupted(BaseException):
    pass


def interrupt(signum, frame) -> None:
    raise LifecycleInterrupted


def verify_container(container, *, owner: str, name: str, identifier: str | None) -> None:
    container.reload()
    attrs = container.attrs
    if (
        not isinstance(container.id, str)
        or re.fullmatch(r"[0-9a-f]{64}", container.id) is None
        or (identifier is not None and container.id != identifier)
        or attrs.get("Name") != "/" + name
        or attrs.get("Config", {}).get("Labels", {}).get(LABEL) != owner
        or attrs.get("Config", {}).get("Image") != POSTGRES_IMAGE
    ):
        raise EnvironmentError("ownership_mismatch")


def verified_port(container) -> int:
    bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
    host_config = container.attrs.get("HostConfig", {})
    ports = bindings.get("5432/tcp")
    if (
        set(bindings) != {"5432/tcp"}
        or not isinstance(ports, list)
        or len(ports) != 1
        or ports[0].get("HostIp") != "127.0.0.1"
        or host_config.get("NetworkMode") != "bridge"
        or host_config.get("NanoCpus") != 1_000_000_000
        or host_config.get("Memory") != 1024**3
    ):
        raise EnvironmentError("unsafe_binding")
    try:
        port = int(ports[0]["HostPort"])
    except (ValueError, KeyError, TypeError):
        raise EnvironmentError("unsafe_binding") from None
    if not 1 <= port <= 65535:
        raise EnvironmentError("unsafe_binding")
    return port


class Supervisor:
    def __init__(
        self,
        directory: Path,
        owner: str,
        channel: socket.socket,
        call_profile=None,
        metrics=False,
        capacity=False,
        queue=False,
    ) -> None:
        from tests.performance.workload import parse_profile

        self.call_profile = (
            parse_profile(call_profile).model_dump(mode="json")
            if call_profile is not None
            else None
        )
        if type(metrics) is not bool:
            raise EnvironmentError("invalid_profile")
        from tests.performance.metrics import Collector

        if type(capacity) is not bool:
            raise EnvironmentError("invalid_profile")
        if type(queue) is not bool or (queue and not (capacity and metrics)):
            raise EnvironmentError("invalid_profile")
        self.queue = queue
        self.worker_stopped = False
        self.capacity = capacity
        self.database_url = None
        if queue:
            from tests.performance.queue_metrics import QueueCollector

            self.metrics = QueueCollector("supervisor", directory)
        elif capacity:
            from tests.performance.capacity_metrics import CapacityCollector

            self.metrics = CapacityCollector("supervisor", directory)
        else:
            self.metrics = Collector("supervisor", directory) if metrics else None
        self.serving = False
        self.previous_stats = None
        self.next_sample = 0.0
        self.directory = directory
        self.owner = UUID(owner).hex
        self.channel = channel
        self.name = "pf-e52-" + self.owner
        self.database_name = "pf_e52_" + self.owner
        self.password = uuid4().hex
        self.container = None
        self.container_id: str | None = None
        self.create_attempted = False
        self.processes: dict[str, OwnedProcess] = {}
        self.api_socket: socket.socket | None = None
        self.started = time.monotonic()
        self.identity: dict = {
            "schema_version": 1,
            "owner": self.owner,
            "profile": QUEUE_PROFILE if self.queue else PROFILE,
            "image": POSTGRES_IMAGE,
            "container_name": self.name,
        }
        self.cleanup_failures: list[str] = []

    @property
    def runtime_seconds(self):
        return QUEUE_RUN_SECONDS if self.queue else RUN_SECONDS

    def create_database(self) -> str:
        # Import only in the clean supervisor process; do not mutate pytest's TC configuration.
        from testcontainers.community.postgres import PostgresContainer
        from testcontainers.core.config import testcontainers_config
        from testcontainers.core.container import DockerContainer

        testcontainers_config.tc_properties = {}
        testcontainers_config.ryuk_disabled = True
        try:
            self.container = PostgresContainer(
                POSTGRES_IMAGE,
                username="pf_e52",
                password=self.password,
                dbname=self.database_name,
                driver="psycopg",
                name=self.name,
                docker_client_kw={"timeout": 5},
                network_mode="bridge",
                labels={LABEL: self.owner},
                nano_cpus=1_000_000_000,
                mem_limit=1024**3,
                pids_limit=128,
            )
        except Exception:
            raise EnvironmentError("docker_unavailable") from None
        # The installed TC version's with_bind_ports only accepts an integer. Its underlying
        # Docker port mapping supports (host IP, ephemeral port), so set that mapping explicitly.
        self.container.ports = {"5432/tcp": ("127.0.0.1", 0)}
        self.container.tmpfs = {"/var/lib/postgresql/data": "rw,size=268435456"}
        self.create_attempted = True
        DockerContainer.start(self.container)
        wrapped = self.container.get_wrapped_container()
        verify_container(wrapped, owner=self.owner, name=self.name, identifier=None)
        self.container_id = wrapped.id
        port = verified_port(wrapped)
        self.identity.update(
            {"container_id": wrapped.id, "database_name": self.database_name, "database_port": port}
        )
        write_record(self.directory, "resources.json", self.identity)
        # Wait for real SQL readiness only after ownership and host binding verification.
        self.container._connect()
        return f"postgresql+psycopg://pf_e52:{self.password}@127.0.0.1:{port}/{self.database_name}"

    def launch(self, role: str, database_url: str, **extra) -> OwnedProcess:
        if role not in {"api", "worker", "migrate"}:
            raise EnvironmentError("protocol_failed")
        if role in self.processes:
            raise EnvironmentError("already_started")
        env = child_environment()
        env["PF_DATABASE_URL"] = database_url
        if "api_origin" in self.identity:
            env["PF_MOCK_PORTAL_BASE_URL"] = self.identity["api_origin"]
        bootstrap = {"owner": self.owner, **extra}
        if role == "worker" and self.call_profile is not None:
            bootstrap.update(call_profile=self.call_profile, output_dir=str(self.directory))
        if self.metrics is not None and role in {"api", "worker"}:
            bootstrap.update(
                metrics=True,
                capacity=self.capacity,
                queue=self.queue,
                output_dir=str(self.directory),
            )
        fds = tuple(extra[key] for key in ("socket_fd", "ready_fd") if key in extra)
        process = OwnedProcess.launch(role, env=env, bootstrap=bootstrap, fds=fds)
        self.processes[role] = process
        self.identity["process_ids"] = {
            key: item.process.pid for key, item in self.processes.items()
        }
        return process

    def start(self) -> str:
        write_record(
            self.directory,
            "started.json",
            {
                **self.identity,
                "status": "IN_PROGRESS",
                "startup_limit_seconds": START_SECONDS,
                "runtime_limit_seconds": self.runtime_seconds,
                "database_cpu": 1,
                "database_memory_bytes": 1024**3,
            },
        )
        database_url = self.create_database()
        migration = self.launch("migrate", database_url)
        if migration.process.wait(
            timeout=max(1, START_SECONDS - (time.monotonic() - self.started))
        ):
            raise EnvironmentError("startup_failed")
        self.api_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.api_socket.bind(("127.0.0.1", 0))
        self.api_socket.listen(128)
        self.identity["api_origin"] = f"http://127.0.0.1:{self.api_socket.getsockname()[1]}"
        api = self.launch("api", database_url, socket_fd=self.api_socket.fileno())
        self.api_socket.close()
        self.api_socket = None
        import httpx

        with httpx.Client(timeout=1, trust_env=False, follow_redirects=False) as client:
            while True:
                if api.process.poll() is not None:
                    raise EnvironmentError("process_exited")
                try:
                    if client.get(self.identity["api_origin"] + "/readyz").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
        self.database_url = database_url
        if not self.capacity:
            self.start_worker()
        write_record(self.directory, "ready.json", self.identity)
        return database_url

    def start_worker(self):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            worker = self.launch("worker", self.database_url, ready_fd=child.fileno())
            child.close()
            timeout = (
                10 if self.capacity else max(1, START_SECONDS - (time.monotonic() - self.started))
            )
            packet = receive_packet(parent, timeout)
            if packet != {"owner": self.owner, "kind": "worker_ready"}:
                raise EnvironmentError("ownership_mismatch")
            if worker.process.poll() is not None:
                raise EnvironmentError("process_exited")
        finally:
            parent.close()
            child.close()
        return worker.process.pid

    def serve(self) -> str | None:
        deadline = self.started + self.runtime_seconds - CLEANUP_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "runtime_timeout"
            required = {"api"} if self.capacity else {"api", "worker"}
            if not required.issubset(self.processes):
                return "process_exited"
            if any(
                p.process.poll() is not None
                for r, p in self.processes.items()
                if r in {"api", "worker"} and not (r == "worker" and self.worker_stopped)
            ):
                return "process_exited"
            if self.metrics is not None and time.monotonic() >= self.next_sample:
                from tests.performance.metrics_runtime import sample_container

                sample_container(self)
                self.next_sample = time.monotonic() + 1.0
            if select.select([self.channel], [], [], min(remaining, 0.1))[0]:
                try:
                    packet = receive_packet(self.channel, 0)
                except EnvironmentError:
                    return "cancelled"  # Includes parent disconnect; still clean owned resources.
                if self.capacity and packet == {"kind": "start_worker"}:
                    pid = self.start_worker()
                    send_packet(
                        self.channel, {"kind": "start_worker", "owner": self.owner, "pid": pid}
                    )
                    continue
                if self.queue and packet == {"kind": "stop_worker"}:
                    child = self.processes.get("worker")
                    if child is None or self.worker_stopped:
                        raise EnvironmentError("protocol_failed")
                    if not child.close():
                        raise EnvironmentError("cleanup_failed")
                    self.worker_stopped = True
                    send_packet(
                        self.channel, {"kind": "stop_worker", "owner": self.owner, "stopped": True}
                    )
                    continue
                if self.capacity and packet == {"kind": "health"}:
                    from tests.performance.capacity_metrics import owned_health

                    send_packet(
                        self.channel, {"kind": "health", "owner": self.owner, **owned_health(self)}
                    )
                    continue
                return None if packet == {"kind": "stop"} else "protocol_failed"

    def cleanup(self) -> bool:
        for role in ("worker", "api", "migrate"):
            child = self.processes.get(role)
            try:
                if child is not None and not child.close():
                    self.cleanup_failures.append(role)
            except Exception:
                self.cleanup_failures.append(role)
        if self.api_socket is not None:
            try:
                self.api_socket.close()
            except OSError:
                self.cleanup_failures.append("listener")
        if self.container is not None:
            try:
                wrapped = self.container._container
                if wrapped is None and self.create_attempted:
                    # A failed create response may still have created a resource. Exact name
                    # lookup plus the unguessable label is required; absence is not proof.
                    wrapped = self.container.get_docker_client().client.containers.get(self.name)
                if wrapped is not None:
                    verify_container(
                        wrapped, owner=self.owner, name=self.name, identifier=self.container_id
                    )
                    wrapped.remove(force=True, v=True)
                    from docker.errors import NotFound

                    try:
                        wrapped.reload()
                    except NotFound:
                        pass
                    else:
                        raise EnvironmentError("cleanup_failed")
            except Exception:
                self.cleanup_failures.append("database")
            finally:
                try:
                    self.container.get_docker_client().client.close()
                except Exception:
                    self.cleanup_failures.append("docker_connection")
        return not self.cleanup_failures

    def run(self) -> int:
        category = None
        released = False
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGALRM, interrupt)
        signal.setitimer(signal.ITIMER_REAL, START_SECONDS)
        try:
            database_url = self.start()
            signal.setitimer(signal.ITIMER_REAL, 0)
            send_packet(
                self.channel,
                {"kind": "ready", "identity": self.identity, "database_url": database_url},
            )
            self.serving = True
            if self.metrics is not None:
                signal.setitimer(
                    signal.ITIMER_REAL,
                    max(
                        1,
                        self.runtime_seconds - CLEANUP_SECONDS - (time.monotonic() - self.started),
                    ),
                )
            category = self.serve()
        except LifecycleInterrupted:
            elapsed = time.monotonic() - self.started
            if self.serving and elapsed >= self.runtime_seconds - CLEANUP_SECONDS:
                category = "runtime_timeout"
            elif not self.serving and elapsed >= START_SECONDS:
                category = "startup_timeout"
            else:
                category = "cancelled"
        except EnvironmentError as exc:
            category = exc.category
        except Exception:
            category = "startup_failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.setitimer(signal.ITIMER_REAL, CLEANUP_SECONDS)
            try:
                released = self.cleanup()
            except LifecycleInterrupted:
                self.cleanup_failures.append("deadline")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
        if self.metrics is not None:
            self.metrics.finish()
            if self.metrics.write_failed and category is None:
                category = "report_failed"
        if category == "cleanup_failed":
            released = False
        result = {
            "kind": "result",
            "profile": QUEUE_PROFILE if self.queue else PROFILE,
            "status": "PASS" if released and category is None else "IN_PROGRESS",
            "category": category if released else "cleanup_failed",
            "primary_category": category,
            "resources_released": released,
            "cleanup_failures": self.cleanup_failures,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "identity": self.identity,
        }
        if category == "docker_unavailable" and released:
            result["status"] = "BLOCKED"
        try:
            write_record(self.directory, "result.json", result)
        except EnvironmentError:
            result["status"] = "IN_PROGRESS"
            result["category"] = "report_failed"
        try:
            send_packet(self.channel, result)
        except (OSError, EnvironmentError):
            pass  # On parent loss the durable result remains the cleanup record.
        finally:
            self.channel.close()
        return 0 if result["status"] == "PASS" else 1


def main() -> int:
    bootstrap = json.loads(sys.stdin.buffer.read(8193))
    with socket.socket(fileno=bootstrap["channel_fd"]) as channel:
        return Supervisor(
            Path(bootstrap["output_dir"]),
            bootstrap["owner"],
            channel,
            bootstrap.get("call_profile"),
            bootstrap.get("metrics", False),
            bootstrap.get("capacity", False),
            bootstrap.get("queue", False),
        ).run()


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        code = 1  # No raw provider/library/filesystem exceptions to stderr.
    raise SystemExit(code)
