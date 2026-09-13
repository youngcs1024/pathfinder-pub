"""Bounded open-arrival scheduling only; no HTTP, DB, Docker or production runtime root.

Execute and guard adapters must be cancellation-cooperative. A broken adapter that suppresses
cancellation is reported as unreleased; a future owned runtime must terminate its own process.
This module cannot forcibly stop an arbitrary coroutine or claim that its external work stopped.
"""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable
from pathlib import Path
from random import Random
from time import monotonic
from typing import Protocol

from tests.performance.contracts import (
    GuardState,
    Manifest,
    Receipt,
    RequestRecord,
    Result,
    Slot,
    Started,
    StopReason,
    count,
    digest,
)
from tests.performance.environment import ROOT, EnvironmentError
from tests.performance.workload import publish

# Separate from the profile's workload/drain window; fits within existing environment reserves.
CANCEL_GRACE_SECONDS = 1.0


class Clock(Protocol):
    def now(self) -> float: ...

    async def sleep_until(self, when: float) -> None: ...


class MonotonicClock:
    def now(self) -> float:
        return monotonic()

    async def sleep_until(self, when: float) -> None:
        await asyncio.sleep(max(0.0, when - self.now()))


class Execute(Protocol):
    async def __call__(self, slot: Slot) -> Receipt: ...


class Guard(Protocol):
    async def __call__(self) -> GuardState: ...


def consume(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


class Driver:
    def __init__(self, manifest: Manifest, execute: Execute, guard: Guard, clock: Clock):
        self.manifest = manifest
        self.profile = manifest.profile
        self.execute = execute
        self.guard = guard
        self.clock = clock
        self.origin = clock.now()
        self.deadline = self.origin + self.profile.max_run_seconds
        self.pending: set[asyncio.Task] = set()
        self.records: dict[int, RequestRecord] = {}
        self.sealed = False
        self.reason: StopReason = "window_complete"
        self.drain_timed_out = False
        self.submission_stopped_at = None

    def elapsed(self) -> float:
        return max(0.0, self.clock.now() - self.origin)

    async def request(self, slot: Slot) -> None:
        record = RequestRecord(
            slot=slot,
            started_at=self.elapsed(),
            outcome="unfinished",
            receipt=Receipt(sent=None),
        )
        self.records[slot.ordinal] = record
        receipt = record.receipt
        outcome = "failed"
        try:
            received = await self.execute(slot)
            # Revalidate even a model_construct/model_copy result crossing the port.
            receipt = Receipt.model_validate_json(received.model_dump_json())
            outcome = "finished"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            # An exception does not establish whether the transport sent a request.
            outcome = "failed"
        finally:
            if not self.sealed:
                self.records[slot.ordinal] = RequestRecord(
                    slot=slot,
                    started_at=record.started_at,
                    finished_at=self.elapsed(),
                    outcome=outcome,
                    receipt=receipt,
                )

    async def bounded(self, operation: Awaitable, deadline: float):
        task = asyncio.create_task(operation)
        timer = asyncio.create_task(self.clock.sleep_until(deadline))
        self.pending.update((task, timer))
        try:
            await asyncio.wait((task, timer), return_when=asyncio.FIRST_COMPLETED)
            if self.clock.now() >= deadline:
                raise TimeoutError
            return task.result()
        finally:
            for item in (task, timer):
                if not item.done():
                    item.cancel()
                else:
                    consume(item)
                    self.pending.discard(item)

    async def schedule(self) -> None:
        random = Random(self.profile.seed)
        window = self.profile.warmup_seconds + self.profile.measurement_seconds
        active: set[asyncio.Task] = set()
        for ordinal in range(self.profile.max_requests):
            offset = ordinal / self.profile.arrival_rate
            if offset >= window:
                self.reason = "window_complete"
                break
            await self.clock.sleep_until(self.origin + offset)
            if self.clock.now() >= self.deadline:
                self.reason = "deadline"
                break
            slot = Slot(
                ordinal=ordinal,
                fixture_seed=random.getrandbits(32),
                phase="warmup" if offset < self.profile.warmup_seconds else "measurement",
                scheduled_at=offset,
            )
            self.records[ordinal] = RequestRecord(
                slot=slot, outcome="stopped", receipt=Receipt(sent=False)
            )
            active = {task for task in active if not task.done()}
            # Drop missed arrivals instead of releasing a catch-up burst after scheduler lag.
            late = self.elapsed() >= offset + 1 / self.profile.arrival_rate
            if len(active) >= self.profile.max_inflight or late:
                self.records[ordinal] = RequestRecord(
                    slot=slot, outcome="generator_dropped", receipt=Receipt(sent=False)
                )
                continue
            if self.elapsed() >= window:
                self.reason = "window_complete"
                break
            try:
                guard = await self.bounded(self.guard(), min(self.deadline, self.origin + window))
                guard = GuardState.model_validate_json(guard.model_dump_json())
            except asyncio.CancelledError:
                raise
            except Exception:
                if self.clock.now() >= self.deadline:
                    self.reason = "deadline"
                elif self.elapsed() >= window:
                    self.reason = "window_complete"
                else:
                    self.reason = "guard_failed"
                break
            if not guard.resources_within_limits:
                self.reason = "resource_limit"
                break
            if guard.queue_depth >= self.profile.max_queue_depth:
                self.reason = "queue_limit"
                break
            # A slow guard must not turn an obsolete slot into a catch-up request either.
            if self.elapsed() >= offset + 1 / self.profile.arrival_rate:
                self.records[ordinal] = RequestRecord(
                    slot=slot, outcome="generator_dropped", receipt=Receipt(sent=False)
                )
                continue
            self.records[ordinal] = RequestRecord(
                slot=slot, outcome="unfinished", receipt=Receipt(sent=None)
            )
            task = asyncio.create_task(self.request(slot))
            active.add(task)
            self.pending.add(task)
            # Let requests enter the port before examining a later scheduled arrival.
            await asyncio.sleep(0)
            self.reap()
        else:
            self.reason = "request_limit"
        self.submission_stopped_at = self.elapsed()
        # Drain includes request tasks and cancelled guard timers still finishing cleanup.
        self.reap()
        if self.pending:
            end = min(self.deadline, self.clock.now() + self.profile.drain_seconds)
            timer = asyncio.create_task(self.clock.sleep_until(end))
            self.pending.add(timer)
            try:
                while self.pending - {timer}:
                    await asyncio.wait(self.pending, return_when=asyncio.FIRST_COMPLETED)
                    self.reap(exclude=timer)
                    if timer.done():
                        self.drain_timed_out = bool(self.pending - {timer})
                        break
            finally:
                timer.cancel()

    def reap(self, *, exclude=None) -> None:
        for task in tuple(self.pending):
            if task is not exclude and task.done():
                consume(task)
                self.pending.discard(task)

    async def release(self) -> bool:
        self.reap()
        for task in self.pending:
            task.cancel()
        if self.pending:
            # A real bound even when a fake clock or an injected port is broken.
            await asyncio.wait(self.pending, timeout=CANCEL_GRACE_SECONDS)
        self.reap()
        self.sealed = True
        for task in self.pending:
            task.add_done_callback(consume)
        return not self.pending

    def result(self, released: bool) -> Result:
        records = tuple(self.records[key] for key in sorted(self.records))
        return Result(
            experiment_id=self.manifest.experiment_id,
            manifest_digest=digest(self.manifest),
            stop_reason=self.reason,
            drain_timed_out=self.drain_timed_out,
            tasks_released=released,
            elapsed_seconds=self.elapsed(),
            submission_stopped_at=(
                self.submission_stopped_at
                if self.submission_stopped_at is not None
                else self.elapsed()
            ),
            records=records,
            warmup=count(tuple(r for r in records if r.slot.phase == "warmup")),
            measurement=count(tuple(r for r in records if r.slot.phase == "measurement")),
        )


def validate_directory(directory: Path) -> None:
    """Existing private artifact directory, normally created by the owned environment."""
    try:
        info = directory.stat()
        if (
            not directory.is_absolute()
            or ".." in directory.parts
            or directory.is_relative_to(ROOT)
            or ROOT.is_relative_to(directory)
            or any(p.is_symlink() for p in (directory, *directory.parents))
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise EnvironmentError("unsafe_directory")
    except OSError:
        raise EnvironmentError("unsafe_directory") from None


async def run(
    manifest: Manifest,
    *,
    execute: Execute,
    guard: Guard,
    output_dir: Path,
    clock: Clock | None = None,
) -> Result:
    """Call explicitly with typed ports and an owned, private evidence directory.

    This is not an environment launcher or authorization for a capacity experiment. Future
    runtime adapters must enforce ownership, connection and provider-call limits themselves.
    """
    try:
        manifest = Manifest.model_validate_json(manifest.model_dump_json())
    except (AttributeError, TypeError, ValueError):
        raise EnvironmentError("invalid_profile") from None
    validate_directory(output_dir)
    publish(output_dir, "load-manifest.json", manifest)
    publish(
        output_dir,
        "load-started.json",
        Started(experiment_id=manifest.experiment_id, manifest_digest=digest(manifest)),
    )
    driver = Driver(manifest, execute, guard, clock or MonotonicClock())
    cancellation = None
    released = False
    try:
        await driver.schedule()
    except asyncio.CancelledError as exc:
        driver.reason = "cancelled"
        cancellation = exc
    except Exception:
        driver.reason = "driver_failed"
    finally:
        try:
            if driver.submission_stopped_at is None:
                driver.submission_stopped_at = driver.elapsed()
            released = await driver.release()
            result = driver.result(released)
            publish(output_dir, "load-result.json", result)
        except asyncio.CancelledError:
            # Repeated cancellation still must not leave newly submitted tasks running.
            for task in driver.pending:
                task.cancel()
                task.add_done_callback(consume)
            driver.sealed = True
            raise
        except Exception:
            if cancellation is not None:
                raise cancellation from None
            raise EnvironmentError("report_failed") from None
    if cancellation is not None:
        raise cancellation
    return result
