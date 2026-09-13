"""Bounded fault IPC owned by the existing supervisor, never a new production root."""

from __future__ import annotations

import select
import signal
import socket
from time import monotonic

from tests.performance.environment import EnvironmentError, receive_packet, send_packet
from tests.performance.fault_contracts import CRASH, Lifecycle, parse_config
from tests.performance.workload import publish


class FaultControl:
    def __init__(self, supervisor, config):
        self.supervisor = supervisor
        self.config = parse_config(config)
        self.generation = 0
        self.channel = None
        self.child_channel = None
        self.barrier = False
        self.released = False
        self.killed = False

    def before_start(self):
        if "worker" in self.supervisor.processes or self.generation >= 2:
            raise EnvironmentError("already_started")
        if self.generation and not self.killed:
            raise EnvironmentError("protocol_failed")
        self.close()
        self.generation += 1
        self.channel, self.child_channel = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)

    def bootstrap(self):
        return {
            "fault_config": self.config.model_dump(mode="json"),
            "generation": self.generation,
            "fault_fd": self.child_channel.fileno(),
        }

    def record(self, kind, pid, exit_code=None):
        record = Lifecycle(
            generation=self.generation, pid=pid, kind=kind, at=monotonic(), exit_code=exit_code
        )
        publish(self.supervisor.directory, f"fault-life-{self.generation}-{kind}.json", record)
        return record

    def after_start(self, worker):
        self.child_channel.close()
        self.child_channel = None
        self.record("started", worker.process.pid)

    def command(self, packet):
        kind = packet.get("kind")
        if kind not in {"fault_poll", "fault_release", "fault_kill"}:
            return False
        if packet != {"kind": kind} or self.generation != 1 or self.channel is None:
            raise EnvironmentError("protocol_failed")
        worker = self.supervisor.processes.get("worker")
        if worker is None or worker.closed or worker.process.poll() is not None:
            raise EnvironmentError("process_exited")
        if not self.barrier and select.select([self.channel], [], [], 0)[0]:
            received = receive_packet(self.channel, 0)
            expected = {
                "kind": "barrier",
                "owner": self.supervisor.owner,
                "generation": 1,
                "scenario": self.config.scenario,
            }
            if received != expected:
                raise EnvironmentError("ownership_mismatch")
            self.barrier = True
            self.record("barrier", worker.process.pid)
        if kind != "fault_poll" and (not self.barrier or self.released or self.killed):
            raise EnvironmentError("protocol_failed")
        if kind == "fault_release":
            send_packet(
                self.channel, {"kind": "release", "owner": self.supervisor.owner, "generation": 1}
            )
            self.released = True
            self.record("released", worker.process.pid)
        if kind == "fault_kill":
            if self.config.scenario not in CRASH:
                raise EnvironmentError("protocol_failed")
            signal.pidfd_send_signal(worker.pidfd, signal.SIGKILL)
            code = worker.process.wait(timeout=5)
            if code != -signal.SIGKILL or not worker.close():
                raise EnvironmentError("cleanup_failed")
            self.record("killed", worker.process.pid, code)
            self.killed = True
            self.supervisor.processes.pop("worker")
        send_packet(
            self.supervisor.channel,
            {"kind": kind, "owner": self.supervisor.owner, "barrier": self.barrier},
        )
        return True

    def close(self):
        for channel in (self.channel, self.child_channel):
            if channel is not None:
                channel.close()
        self.channel = self.child_channel = None
