"""Capacity v2 observations; same-host monotonic windows, bounded private packets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from sqlalchemy import event

from tests.performance.metrics import Collector, Packet, Sample

CAPACITY_LIMIT = 16384
PACKET_BYTES = 32 * 1024 * 1024


class CapacitySample(Sample):
    ordinal: int = Field(ge=1, le=CAPACITY_LIMIT)


class CapacityPacket(Packet):
    schema_version: Literal[2] = 2
    samples: tuple[CapacitySample, ...] = Field(max_length=CAPACITY_LIMIT)
    pool_checkouts: int = Field(ge=0)
    pool_peak: int = Field(ge=0)
    pool_remaining: int = Field(ge=0)


class CapacityCollector(Collector):
    limit = CAPACITY_LIMIT
    sample_type = CapacitySample

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkouts = 0
        self.checked_out = 0
        self.peak = 0

    def attach_pool(self, engine):
        def checkout(*_args):
            self.checkouts += 1
            self.checked_out += 1
            self.peak = max(self.peak, self.checked_out)

        def checkin(*_args):
            self.checked_out -= 1

        event.listen(engine.sync_engine.pool, "checkout", checkout)
        event.listen(engine.sync_engine.pool, "checkin", checkin)

        def detach():
            event.remove(engine.sync_engine.pool, "checkout", checkout)
            event.remove(engine.sync_engine.pool, "checkin", checkin)

        return detach

    def packet(self):
        return CapacityPacket(
            role=self.role,
            process_id=self.process_id,
            started=self.started,
            finished=self.clock(),
            dropped=self.dropped,
            write_failed=self.write_failed,
            observer_seconds=self.observer_seconds,
            samples=tuple(self.samples),
            pool_checkouts=self.checkouts,
            pool_peak=self.peak,
            pool_remaining=self.checked_out,
        )


def owned_health(supervisor):
    """Only the already-owned container and process family; no arbitrary exec/target port."""
    from tests.performance._supervisor import verified_port, verify_container

    result = dict(memory=None, tmpfs=None, rss=None)
    wrapped = supervisor.container.get_wrapped_container()
    verify_container(
        wrapped, owner=supervisor.owner, name=supervisor.name, identifier=supervisor.container_id
    )
    verified_port(wrapped)
    if wrapped.attrs["HostConfig"].get("Tmpfs") != {
        "/var/lib/postgresql/data": "rw,size=268435456"
    }:
        return result
    stats = wrapped.stats(stream=False, one_shot=True)
    result["memory"] = stats.get("memory_stats", {}).get("usage")
    used = wrapped.exec_run(["df", "-B1", "--output=used", "/var/lib/postgresql/data"])
    if used.exit_code == 0 and len(used.output) < 128:
        result["tmpfs"] = int(used.output.splitlines()[-1].strip())
    pids = {os.getpid(), os.getppid()}
    for role, process in supervisor.processes.items():
        if role in {"api", "worker"}:
            if process.process.poll() is not None:
                return result
            pids.add(process.process.pid)
    result["rss"] = sum(
        int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        for pid in pids
    )
    return result
