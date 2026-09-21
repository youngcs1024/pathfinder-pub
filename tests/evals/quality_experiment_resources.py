"""Same Linux proc/Docker observer for both experiment arms; no new monitoring service."""

from __future__ import annotations

import asyncio
import os
from itertools import pairwise
from pathlib import Path
from time import monotonic
from typing import Literal

from pydantic import Field, model_validator

from tests.evals.contracts import EvalContractModel
from tests.evals.quality_experiment_binding import ExperimentError, fail, write_new

LIMIT = 2_147_483_648


def process_table(proc=Path("/proc")):
    rows = {}
    for item in proc.iterdir():
        if not item.name.isdecimal():
            continue
        try:
            text = (item / "stat").read_text()
            fields = text[text.rfind(")") + 2 :].split()
            rows[int(item.name)] = (int(fields[1]), int(fields[19]), int(fields[21]))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError):
            # An unreadable process is not silently assigned zero memory.
            fail("resource_measurement_missing")
    return rows


def process_memory(pid, parent_pid=None, *, proc=Path("/proc"), page_size=None):
    rows = process_table(proc)
    if pid not in rows or (parent_pid is not None and parent_pid not in rows):
        fail("resource_measurement_missing")
    included = {pid}
    while True:
        expanded = included | {key for key, row in rows.items() if row[0] in included}
        if expanded == included:
            break
        included = expanded
    # Attribute the same supervisor RSS to each arm; the other arm is observed separately.
    if parent_pid is not None:
        included.add(parent_pid)
    pages = sum(rows[key][2] for key in included)
    if pages <= 0:
        fail("resource_measurement_missing")
    return pages * (page_size or os.sysconf("SC_PAGE_SIZE")), rows[pid][1]


class ResourceSampleV1(EvalContractModel):
    elapsed_seconds: float = Field(ge=0)
    runner_rss_bytes: int = Field(gt=0)
    database_memory_bytes: int = Field(ge=0)
    combined_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def total(self):
        if self.combined_bytes != self.runner_rss_bytes + self.database_memory_bytes:
            raise ValueError("resource total mismatch")
        return self


class ResourceReportV1(EvalContractModel):
    artifact_kind: Literal["e7a7_resources_v1"] = "e7a7_resources_v1"
    arm: Literal["baseline", "candidate"]
    observer: Literal["e7a7-proc-docker-v1"] = "e7a7-proc-docker-v1"
    method: Literal["arm_process_tree_plus_supervisor_rss_and_own_database_usage"] = (
        "arm_process_tree_plus_supervisor_rss_and_own_database_usage"
    )
    samples: tuple[ResourceSampleV1, ...]
    elapsed_seconds: float = Field(ge=0)
    complete: bool
    failure: Literal["resource_limit", "resource_measurement_missing", "identity_drift"] | None
    peak_combined_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def checked(self):
        peak = max((s.combined_bytes for s in self.samples), default=None)
        if self.peak_combined_bytes != peak:
            raise ValueError("invalid resource peak")
        if self.complete != resource_complete(self.samples, self.elapsed_seconds, self.failure):
            raise ValueError("invalid resource completeness")
        return self


def resource_complete(samples, elapsed, failure):
    if failure is not None or len(samples) < 2:
        return False
    times = [s.elapsed_seconds for s in samples]
    if times[0] > 2.5 or times[-1] > elapsed or elapsed - times[-1] > 2.5:
        return False
    if any(not 0 < b - a <= 2.5 for a, b in pairwise(times)):
        return False
    return all(s.runner_rss_bytes <= LIMIT and s.database_memory_bytes <= LIMIT for s in samples)


class ResourceObserver:
    def __init__(self, owner, directory, arm, *, parent_pid=None, probe=None, clock=monotonic):
        self.owner, self.directory, self.arm = owner, directory, arm
        self.parent_pid, self.clock = parent_pid, clock
        self.probe = probe or self._probe
        self.samples = []
        self.failure = None
        self.failed = asyncio.Event()
        self.task = None
        self.started = None
        self.process_start = None
        self.closed = False

    def _probe(self):
        self.owner._verify_container()
        memory, start = process_memory(os.getpid(), self.parent_pid)
        if self.process_start is not None and start != self.process_start:
            fail("identity_drift")
        self.process_start = start
        container = self.owner._container.get_wrapped_container()
        identity = self.owner._container_id
        if container.id != identity:
            fail("identity_drift")
        stats = container.stats(stream=False, one_shot=True)
        # Moby 28.x one-shot stats omit ID (daemon/stats.go). The SDK request
        # targets the full owned ID; verify inspect identity on both sides.
        self.owner._verify_container()
        if self.owner._container_id != identity or ("id" in stats and stats["id"] != identity):
            fail("identity_drift")
        db = stats.get("memory_stats", {}).get("usage")
        if type(db) is not int or db < 0:
            fail("resource_measurement_missing")
        return memory, db

    def check(self):
        if self.failure:
            fail(self.failure)
        if self.closed:
            fail("resource_measurement_missing")

    async def sample(self):
        try:
            memory, db = await asyncio.to_thread(self.probe)
            sample = ResourceSampleV1(
                elapsed_seconds=self.clock() - self.started,
                runner_rss_bytes=memory,
                database_memory_bytes=db,
                combined_bytes=memory + db,
            )
            write_new(self.directory / f"sample-{len(self.samples):05d}.json", sample)
            self.samples.append(sample)
            if memory > LIMIT or db > LIMIT:
                fail("resource_limit")
            if (len(self.samples) == 1 and sample.elapsed_seconds > 2.5) or (
                len(self.samples) > 1
                and sample.elapsed_seconds - self.samples[-2].elapsed_seconds > 2.5
            ):
                fail("resource_measurement_missing")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failure = (
                error.category
                if isinstance(error, ExperimentError)
                and error.category in {"resource_limit", "identity_drift"}
                else "resource_measurement_missing"
            )
            self.failed.set()

    async def start(self):
        if self.started is not None:
            fail("resource_observer_reused")
        self.started = self.clock()
        await self.sample()
        self.check()
        self.task = asyncio.create_task(self._loop())

    async def _loop(self):
        deadline = self.clock() + 1
        while not self.closed and not self.failure:
            await asyncio.sleep(max(0, deadline - self.clock()))
            await self.sample()
            deadline += 1

    async def protect(self, awaitable):
        work = asyncio.ensure_future(awaitable)
        stopped = asyncio.create_task(self.failed.wait())
        try:
            done, _ = await asyncio.wait({work, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                fail(self.failure or "resource_measurement_missing")
            return await work
        finally:
            stopped.cancel()
            if not work.done():
                work.cancel()
            await asyncio.gather(stopped, work, return_exceptions=True)

    async def finish(self):
        if self.closed or self.started is None:
            fail("resource_observer_reused")
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if not self.failure:
            await self.sample()
        elapsed = self.clock() - self.started
        report = ResourceReportV1(
            arm=self.arm,
            samples=tuple(self.samples),
            elapsed_seconds=elapsed,
            failure=self.failure,
            peak_combined_bytes=max((s.combined_bytes for s in self.samples), default=None),
            complete=resource_complete(self.samples, elapsed, self.failure),
        )
        write_new(self.directory / "report.json", report)
        return report
