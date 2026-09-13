"""Virtual-clock E5.4 tests. No real HTTP, provider, database or load environment."""

import asyncio
import json

import pytest

from tests.performance import driver
from tests.performance.contracts import GuardState, Receipt
from tests.performance.environment import EnvironmentError
from tests.unit.test_performance_contracts import manifest


class VirtualClock:
    def __init__(self):
        self.value = 1000.0
        self.waiters = []

    def now(self):
        return self.value

    async def sleep_until(self, when):
        if when <= self.value:
            await asyncio.sleep(0)
            return
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((when, future))
        try:
            await future
        finally:
            self.waiters.remove((when, future))

    def advance(self, seconds):
        self.value += seconds
        for when, future in self.waiters:
            if when <= self.value and not future.done():
                future.set_result(None)


async def settle():
    for _ in range(30):
        await asyncio.sleep(0)


async def safe_guard():
    return GuardState(queue_depth=0, resources_within_limits=True)


async def accepted(slot):
    return Receipt(sent=True, http_status=202, replayed=False)


async def simulate(tmp_path, *, item=None, execute=accepted, guard=safe_guard):
    clock = VirtualClock()
    item = item or manifest()
    task = asyncio.create_task(
        driver.run(item, execute=execute, guard=guard, output_dir=tmp_path, clock=clock)
    )
    await settle()
    for _ in range(int(item.profile.max_run_seconds * 4) + 8):
        if task.done():
            break
        clock.advance(0.25)
        await settle()
    result = await asyncio.wait_for(task, timeout=2)
    assert not clock.waiters
    return result


async def test_scheduled_times_and_request_limit(tmp_path):
    result = await simulate(tmp_path)
    assert [r.slot.scheduled_at for r in result.records] == [0, 0.5]
    assert [r.started_at for r in result.records] == [0, 0.5]
    assert result.measurement.offered == result.measurement.sent == 2
    assert result.stop_reason == "request_limit"
    assert result.tasks_released
    assert {p.name for p in tmp_path.iterdir()} == {
        "load-manifest.json",
        "load-started.json",
        "load-result.json",
    }


async def test_slow_requests_drop_without_lowering_planned_arrival_rate(tmp_path):
    clock = VirtualClock()
    concurrency = 0
    peak = 0

    async def slow(slot):
        nonlocal concurrency, peak
        concurrency += 1
        peak = max(peak, concurrency)
        try:
            await clock.sleep_until(clock.now() + 0.75)
            return Receipt(sent=True, http_status=202)
        finally:
            concurrency -= 1

    task = asyncio.create_task(
        driver.run(manifest(), execute=slow, guard=safe_guard, output_dir=tmp_path, clock=clock)
    )
    await settle()
    clock.advance(0.5)
    await settle()
    clock.advance(0.25)
    await settle()
    result = await task
    assert peak == 1 and concurrency == 0
    assert [r.slot.scheduled_at for r in result.records] == [0, 0.5]
    assert result.records[1].outcome == "generator_dropped"
    assert result.records[1].started_at is None
    assert result.records[0].finished_at == 0.75
    assert result.measurement.sent == result.measurement.generator_dropped == 1
    assert not clock.waiters


async def test_drain_timeout_cancels_requests_and_preserves_unknown(tmp_path):
    stopped = asyncio.Event()

    async def never(slot):
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    result = await simulate(tmp_path, execute=never)
    assert result.drain_timed_out
    assert result.tasks_released and stopped.is_set()
    assert result.measurement.cancelled == result.measurement.sent_unknown == 1
    assert result.measurement.generator_dropped == 1
    assert result.measurement.http_accepted == 0


async def test_seed_and_warmup_replay(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(mode=0o700)
    right.mkdir(mode=0o700)
    a = await simulate(left, item=manifest("low-load-v1"))
    b = await simulate(right, item=manifest("low-load-v1"))
    assert [r.slot for r in a.records] == [r.slot for r in b.records]
    assert [r.slot.scheduled_at for r in a.records] == [0, 10, 20]
    assert a.warmup.offered == 1 and a.measurement.offered == 2


async def test_warmup_boundary_belongs_to_measurement(tmp_path):
    result = await simulate(tmp_path, item=manifest(warmup_seconds=0.5, measurement_seconds=0.5))
    assert [r.slot.phase for r in result.records] == ["warmup", "measurement"]


async def test_window_complete_does_not_schedule_boundary_slot(tmp_path):
    result = await simulate(tmp_path, item=manifest(max_requests=3))
    assert result.stop_reason == "window_complete"
    assert len(result.records) == 2


@pytest.mark.parametrize(
    "guard_result,reason",
    [
        (GuardState(queue_depth=2, resources_within_limits=True), "queue_limit"),
        (GuardState(queue_depth=0, resources_within_limits=False), "resource_limit"),
        (None, "guard_failed"),
        (GuardState.model_construct(queue_depth=-1, resources_within_limits=True), "guard_failed"),
    ],
)
async def test_protection_stops_before_sending(tmp_path, guard_result, reason):
    async def guard():
        return guard_result

    async def must_not_send(slot):
        raise AssertionError("should not execute")

    result = await simulate(tmp_path, execute=must_not_send, guard=guard)
    assert result.stop_reason == reason
    assert result.measurement.offered == result.measurement.stopped_before_send == 1
    assert result.measurement.sent == 0
    assert result.tasks_released


async def test_guard_failure_does_not_publish_exception_text(tmp_path):
    async def guard():
        raise RuntimeError("PRIVATE_CANARY")

    result = await simulate(tmp_path, guard=guard)
    assert result.stop_reason == "guard_failed"
    assert all("PRIVATE_CANARY" not in p.read_text() for p in tmp_path.iterdir())


@pytest.mark.parametrize("jump,reason", [(1.0, "window_complete"), (2.0, "deadline")])
async def test_hanging_guard_cannot_send_after_window_or_deadline(tmp_path, jump, reason):
    finished = asyncio.Event()
    clock = VirtualClock()

    async def guard():
        try:
            await asyncio.Future()
        finally:
            finished.set()

    task = asyncio.create_task(
        driver.run(manifest(), execute=accepted, guard=guard, output_dir=tmp_path, clock=clock)
    )
    await settle()
    clock.advance(jump)
    await settle()
    result = await task
    assert result.stop_reason == reason
    assert result.elapsed_seconds == jump
    assert result.tasks_released and finished.is_set()
    assert not clock.waiters
    assert len(result.records) == 1
    assert result.records[0].outcome == "stopped"
    assert result.measurement.sent == 0


async def test_transport_exception_keeps_unknown_sent_and_failed_count(tmp_path):
    async def execute(slot):
        raise RuntimeError("PRIVATE_CANARY")

    result = await simulate(tmp_path, execute=execute)
    assert result.measurement.failed == result.measurement.sent_unknown == 2
    assert result.measurement.sent == 0
    assert all("PRIVATE_CANARY" not in p.read_text() for p in tmp_path.iterdir())


async def test_invalid_transport_receipt_fails_without_copying_payload(tmp_path):
    async def execute(slot):
        return Receipt.model_construct(sent=True, http_status=999)

    result = await simulate(tmp_path, execute=execute)
    assert result.measurement.failed == 2
    assert result.measurement.http_accepted == 0


async def test_external_cancellation_propagates_after_partial_result(tmp_path):
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def execute(slot):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    clock = VirtualClock()
    task = asyncio.create_task(
        driver.run(manifest(), execute=execute, guard=safe_guard, output_dir=tmp_path, clock=clock)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and not clock.waiters
    saved = json.loads((tmp_path / "load-result.json").read_text())
    assert saved["stop_reason"] == "cancelled"
    assert saved["tasks_released"]
    assert saved["measurement"]["cancelled"] == 1


async def test_cancellation_is_not_replaced_by_report_failure(tmp_path, monkeypatch):
    entered = asyncio.Event()
    original = driver.publish

    def publish(directory, name, record):
        if name == "load-result.json":
            raise EnvironmentError("report_failed")
        original(directory, name, record)

    async def execute(slot):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(driver, "publish", publish)
    task = asyncio.create_task(
        driver.run(
            manifest(), execute=execute, guard=safe_guard, output_dir=tmp_path, clock=VirtualClock()
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (tmp_path / "load-started.json").exists()
    assert not (tmp_path / "load-result.json").exists()


async def test_existing_manifest_prevents_any_port_calls(tmp_path):
    (tmp_path / "load-manifest.json").write_text("existing")
    calls = []

    async def guard():
        calls.append(1)
        return await safe_guard()

    with pytest.raises(EnvironmentError, match="report_failed"):
        await driver.run(manifest(), execute=accepted, guard=guard, output_dir=tmp_path)
    assert not calls
    assert (tmp_path / "load-manifest.json").read_text() == "existing"


async def test_forged_manifest_rejected_before_artifact_or_ports(tmp_path):
    forged = manifest().model_copy(update={"lock_digest": "PRIVATE_CANARY"})
    with pytest.raises(EnvironmentError, match="invalid_profile"):
        await driver.run(forged, execute=accepted, guard=safe_guard, output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


async def test_scheduler_lag_is_recorded_without_catchup_burst(tmp_path):
    clock = VirtualClock()
    sent = []

    async def execute(slot):
        sent.append(slot.ordinal)
        return await accepted(slot)

    item = manifest(max_requests=4, measurement_seconds=2.0, max_run_seconds=3.0)
    task = asyncio.create_task(
        driver.run(item, execute=execute, guard=safe_guard, output_dir=tmp_path, clock=clock)
    )
    await settle()
    clock.advance(1.5)
    await settle()
    result = await task
    assert sent == [0, 3]
    assert [r.slot.scheduled_at for r in result.records] == [0, 0.5, 1, 1.5]
    assert result.measurement.generator_dropped == 2


async def test_slow_guard_drops_expired_slot(tmp_path):
    clock = VirtualClock()
    calls = 0

    async def guard():
        nonlocal calls
        calls += 1
        if calls == 1:
            await clock.sleep_until(clock.now() + 0.5)
        return await safe_guard()

    task = asyncio.create_task(
        driver.run(manifest(), execute=accepted, guard=guard, output_dir=tmp_path, clock=clock)
    )
    await settle()
    clock.advance(0.5)
    await settle()
    result = await task
    assert result.records[0].outcome == "generator_dropped"
    assert result.records[1].started_at == 0.5
    assert result.measurement.sent == 1


async def test_two_slots_are_the_actual_concurrency_bound(tmp_path):
    current = 0
    peak = 0

    async def execute(slot):
        nonlocal current, peak
        current += 1
        peak = max(current, peak)
        try:
            await asyncio.Future()
        finally:
            current -= 1

    item = manifest(
        connections=2, max_inflight=2, max_requests=4, measurement_seconds=2.0, max_run_seconds=3.0
    )
    result = await simulate(tmp_path, item=item, execute=execute)
    assert peak == 2 and current == 0
    assert result.measurement.cancelled == result.measurement.generator_dropped == 2
    assert result.tasks_released


async def test_different_seed_changes_fixture_selection_not_arrival_times(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(mode=0o700)
    right.mkdir(mode=0o700)
    a = await simulate(left)
    b = await simulate(right, item=manifest(seed=55))
    assert [r.slot.scheduled_at for r in a.records] == [r.slot.scheduled_at for r in b.records]
    assert [r.slot.fixture_seed for r in a.records] != [r.slot.fixture_seed for r in b.records]


async def test_uncooperative_port_is_not_reported_as_released(tmp_path, monkeypatch):
    # No long wait: explicitly let the broken test port finish after inspecting the report.
    monkeypatch.setattr(driver, "CANCEL_GRACE_SECONDS", 0.0)
    blocked = asyncio.Event()
    finished = asyncio.Event()

    async def execute(slot):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            try:
                await blocked.wait()
                return Receipt(sent=None)
            finally:
                finished.set()

    try:
        result = await simulate(tmp_path, execute=execute)
        assert not result.tasks_released
        assert result.measurement.unfinished == 1
        saved = (tmp_path / "load-result.json").read_bytes()
    finally:
        blocked.set()
        await settle()
    assert finished.is_set()
    assert (tmp_path / "load-result.json").read_bytes() == saved


async def test_result_write_failure_is_not_success(tmp_path, monkeypatch):
    original = driver.publish

    def publish(directory, name, record):
        if name == "load-result.json":
            raise EnvironmentError("report_failed")
        original(directory, name, record)

    monkeypatch.setattr(driver, "publish", publish)
    with pytest.raises(EnvironmentError, match="report_failed"):
        await simulate(tmp_path)
    assert (tmp_path / "load-started.json").exists()


async def test_public_directory_rejected_before_manifest(tmp_path):
    path = tmp_path / "public"
    path.mkdir(mode=0o755)
    with pytest.raises(EnvironmentError, match="unsafe_directory"):
        await driver.run(manifest(), execute=accepted, guard=safe_guard, output_dir=path)
    assert not list(path.iterdir())


async def test_real_receipt_replay_header_facts_are_not_new_run_throughput(tmp_path):
    from uuid import uuid4

    run_id = uuid4()

    async def execute(slot):
        return Receipt(sent=True, http_status=202, replayed=bool(slot.ordinal), run_id=run_id)

    result = await simulate(tmp_path, execute=execute)
    assert result.measurement.http_accepted == 2
    assert result.measurement.replays == 1
    assert result.measurement.observed_unique_runs == result.measurement.observed_new_runs == 1


async def test_protection_after_first_request_retains_earlier_results(tmp_path):
    calls = 0

    async def guard():
        nonlocal calls
        calls += 1
        return GuardState(queue_depth=0 if calls == 1 else 2, resources_within_limits=True)

    result = await simulate(tmp_path, guard=guard)
    assert result.stop_reason == "queue_limit"
    assert result.measurement.offered == 2
    assert result.measurement.sent == result.measurement.stopped_before_send == 1
    assert result.submission_stopped_at == 0.5


async def test_symlink_directory_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(EnvironmentError, match="unsafe_directory"):
        await driver.run(manifest(), execute=accepted, guard=safe_guard, output_dir=link)
    assert not list(actual.iterdir())
