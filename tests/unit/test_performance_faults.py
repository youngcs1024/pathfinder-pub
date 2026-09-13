"""Fault control/identity/retention regressions; no Docker or benchmark run."""

import asyncio
import signal
import socket
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.performance import faults
from tests.performance.environment import EnvironmentError, EnvironmentProfile, IsolatedEnvironment
from tests.performance.fault_contracts import (
    AUTHORIZATION,
    CRASH,
    SCENARIOS,
    FaultCall,
    Lifecycle,
    Point,
    Result,
    load_profile,
    points,
)
from tests.performance.fault_control import FaultControl
from tests.performance.fault_runtime import FaultCalls
from tests.performance.workload import profile, publish


def supervisor(tmp_path, scenario="claim"):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    owner = uuid4().hex
    s = SimpleNamespace(directory=tmp_path, owner=owner, processes={}, channel=child)
    c = FaultControl(s, {"scenario": scenario, "instant": True})
    return s, c, parent, child


def test_fixed_matrix_and_authorization(tmp_path):
    p = load_profile("faults-e59-v1")
    assert len(points(p)) == 30
    assert len({x.scenario for x in points(p)}) == 10
    assert all(sum(x.scenario == s for x in points(p)) == 3 for s in SCENARIOS)
    with pytest.raises(ValueError, match="invalid_authorization"):
        faults.run_point(
            tmp_path / "never",
            point=Point(scenario="claim", repetition=1),
            profile=p,
            authorization="ci_instant",
        )
    assert not (tmp_path / "never").exists()


def test_no_success_without_evidence():
    with pytest.raises(ValueError):
        Result(point=Point(scenario="claim", repetition=1), status="PASS", stop="complete")


def test_publication_create_only_name_and_body(tmp_path):
    life = Lifecycle(generation=1, pid=1, kind="started", at=1.0)
    publish(tmp_path, "fault-life-1-started.json", life)
    for name in ("fault-life-1-started.json", "fault-../escape.json", "fault-result.json"):
        with pytest.raises(EnvironmentError):
            publish(tmp_path, name, life)
    with pytest.raises(ValueError):
        Lifecycle.model_validate({**life.model_dump(), "body": "PRIVATE_CANARY"})
    assert "PRIVATE_CANARY" not in (tmp_path / "fault-life-1-started.json").read_text()


def test_controller_refuses_overlap_or_third_generation(tmp_path):
    s, c, parent, channel = supervisor(tmp_path)
    try:
        s.processes["worker"] = object()
        with pytest.raises(EnvironmentError, match="already_started"):
            c.before_start()
        s.processes.clear()
        c.generation = 1
        with pytest.raises(EnvironmentError, match="protocol_failed"):
            c.before_start()
        c.generation = 2
        c.killed = True
        with pytest.raises(EnvironmentError, match="already_started"):
            c.before_start()
    finally:
        c.close()
        parent.close()
        channel.close()


@pytest.mark.parametrize("scenario", CRASH)
def test_owned_kill_waits_and_reaps_before_restart(tmp_path, monkeypatch, scenario):
    s, c, parent, channel = supervisor(tmp_path, scenario)
    steps = []

    class Process:
        pid = 42
        code = None

        def poll(self):
            return self.code

        def wait(self, timeout):
            steps.append("wait")
            self.code = -9
            return self.code

    process = Process()

    def close():
        steps.append("close")
        return True

    worker = SimpleNamespace(process=process, pidfd=77, closed=False, close=close)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: steps.append((fd, sig)))
    try:
        c.before_start()
        s.processes["worker"] = worker
        c.barrier = True
        assert c.command({"kind": "fault_kill"})
        assert steps == [(77, signal.SIGKILL), "wait", "close"]
        assert "worker" not in s.processes and c.killed
        c.before_start()
        assert c.generation == 2
    finally:
        c.close()
        parent.close()
        channel.close()


@pytest.mark.parametrize("kind", ["fault_release", "fault_kill"])
def test_no_mutation_before_verified_barrier(tmp_path, kind):
    s, c, parent, channel = supervisor(tmp_path)
    try:
        c.before_start()
        s.processes["worker"] = SimpleNamespace(
            closed=False, process=SimpleNamespace(poll=lambda: None)
        )
        with pytest.raises(EnvironmentError, match="protocol_failed"):
            c.command({"kind": kind})
        c.child_channel.send(b'{"kind":"barrier","owner":"wrong"}')
        with pytest.raises(EnvironmentError, match="ownership_mismatch"):
            c.command({"kind": "fault_poll"})
    finally:
        c.close()
        parent.close()
        channel.close()


def calls(tmp_path, generation=1):
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    bootstrap = {
        "owner": uuid4().hex,
        "fault_config": {"scenario": "claim", "instant": True},
        "generation": generation,
        "fault_fd": child.detach(),
        "call_profile": profile().model_dump(mode="json"),
        "output_dir": str(tmp_path),
    }
    return FaultCalls(bootstrap), parent


async def test_cross_generation_budget_and_unfinished_records(tmp_path):
    for seq in range(1, 64):
        record = FaultCall(
            generation=1,
            process="worker",
            sequence=seq,
            call="chat",
            ordinal=seq,
            invocation_id=uuid4(),
            delay_seconds=0.0,
            phase="started",
            outcome="pending",
            elapsed_seconds=0.0,
        )
        publish(tmp_path, f"fault-call-1-{seq:03}-started.json", record)
    c, parent = calls(tmp_path, 2)
    try:
        with pytest.raises(asyncio.CancelledError, match="call_limit"):
            await c.invoke("chat", lambda: asyncio.sleep(0))
        assert len(faults.read_evidence(tmp_path)[1]) == 63
        assert not list(tmp_path.glob("fault-call-2-*"))
    finally:
        c.channel.close()
        parent.close()


async def test_restart_keeps_generation_and_global_sequence(tmp_path):
    first, parent = calls(tmp_path)
    try:
        await first.invoke("search", lambda: asyncio.sleep(0))
    finally:
        first.channel.close()
        parent.close()
    second, parent = calls(tmp_path, 2)
    try:
        await second.invoke("search", lambda: asyncio.sleep(0))
        records = faults.read_evidence(tmp_path)[1]
        assert {(r.generation, r.sequence) for r in records} == {(1, 1), (2, 2)}
    finally:
        second.channel.close()
        parent.close()


async def test_barrier_cancellation_propagates(tmp_path):
    c, parent = calls(tmp_path)
    try:
        task = asyncio.create_task(c.barrier())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        c.channel.close()
        parent.close()


def test_fault_commands_unavailable_to_normal_environment(tmp_path):
    env = IsolatedEnvironment(
        EnvironmentProfile("environment-v1"), tmp_path / "unused", capacity=True
    )
    env._started = True
    with pytest.raises(EnvironmentError, match="protocol_failed"):
        env.capacity_command("fault_kill")


def test_suite_stops_and_preserves_full_denominator(tmp_path, monkeypatch):
    invoked = []

    def failed(output, *, point, **kwargs):
        invoked.append(point)
        return Result(point=point, status="IN_PROGRESS", stop="correctness_failed")

    monkeypatch.setattr(faults, "run_point", failed)
    suite = faults.run_suite(
        tmp_path / "suite", authorization=AUTHORIZATION, selected_profile="faults-e59-v1"
    )
    assert len(invoked) == 1 and len(suite.results) == 30
    assert all(r.status == "NOT_STARTED" for r in suite.results[1:])
    assert (tmp_path / "suite" / "fault-suite.json").exists()


def test_cli_nonzero_for_partial(monkeypatch, tmp_path):
    from tests.performance.__main__ import main

    monkeypatch.setattr(
        faults,
        "run_suite",
        lambda *a, **kw: SimpleNamespace(results=[SimpleNamespace(status="IN_PROGRESS")]),
    )
    assert (
        main(
            [
                "faults",
                "--profile",
                "faults-e59-v1",
                "--authorization",
                AUTHORIZATION,
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
