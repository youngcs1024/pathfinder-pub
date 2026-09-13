"""Child-local wrappers pause after real commits; all transitions remain production-owned."""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from tests.performance.adapters import Calls
from tests.performance.capacity_contracts import safe_read
from tests.performance.environment import EnvironmentError, send_packet
from tests.performance.fault_contracts import FaultCall, parse_config
from tests.performance.workload import CallRecord, parse_profile, publish


class FaultCalls(Calls):
    def __init__(self, bootstrap):
        self.config = parse_config(bootstrap["fault_config"])
        self.generation = bootstrap["generation"]
        if type(self.generation) is not int or self.generation not in {1, 2}:
            raise EnvironmentError("invalid_profile")
        self.owner = bootstrap["owner"]
        self.channel = socket.socket(fileno=bootstrap["fault_fd"])
        self.channel.setblocking(False)
        self.entered = False
        super().__init__(
            parse_profile(bootstrap["call_profile"]), directory=Path(bootstrap["output_dir"])
        )
        # A single live worker writes these create-only records. Never reset the budget on restart.
        if self.generation == 2:
            records = [
                safe_read(p, FaultCall, 8192)
                for p in sorted(self.directory.glob("fault-call-1-*-started.json"))
            ]
            if [r.sequence for r in records] != list(range(1, len(records) + 1)):
                raise EnvironmentError("protocol_failed")
            for record in records:
                self.counts[record.call] += 1

    def record(self, record: CallRecord):
        fact = FaultCall(**record.model_dump(), generation=self.generation)
        publish(
            self.directory,
            f"fault-call-{self.generation}-{fact.sequence:03}-{fact.phase}.json",
            fact,
        )
        self.records.append(fact)

    async def barrier(self):
        if self.generation != 1 or self.entered:
            return
        self.entered = True
        send_packet(
            self.channel,
            {
                "kind": "barrier",
                "owner": self.owner,
                "generation": 1,
                "scenario": self.config.scenario,
            },
        )
        async with asyncio.timeout(60):
            raw = await asyncio.get_running_loop().sock_recv(self.channel, 8193)
        try:
            packet = json.loads(raw)
            if len(raw) > 8192 or packet != {
                "kind": "release",
                "owner": self.owner,
                "generation": 1,
            }:
                raise ValueError
        except (ValueError, UnicodeError):
            raise EnvironmentError("protocol_failed") from None

    async def invoke(self, kind, operation, **kwargs):
        from app.tools.adapters.mock_portal import MockPortalTransportError

        async def controlled():
            if kind == "mock_lookup" and self.config.scenario == "lookup_unavailable":
                raise MockPortalTransportError
            result = await operation()
            if kind == "mock_submit" and self.config.scenario in {
                "response_lost",
                "lookup_unavailable",
                "cancel_after",
                "revoke_after",
            }:
                await self.barrier()
                raise MockPortalTransportError
            return result

        return await super().invoke(kind, controlled, **kwargs)


@contextmanager
def instrument_faults(worker, calls):
    from app.db.jobs import SqlAlchemyWorkerJobStore
    from app.db.llm_invocations import SqlAlchemyInvocationRecorder
    from app.tools.registry import ToolRegistry
    from app.worker.runner import WorkerRunner
    from app.worker.settings import WorkerRuntimeSettings

    scenario = calls.config.scenario

    def after(original, predicate):
        async def wrapped(*args, **kwargs):
            result = await original(*args, **kwargs)
            if predicate(result):
                await calls.barrier()
            return result

        return wrapped

    with ExitStack() as stack:
        if calls.config.instant:
            stack.enter_context(
                patch.object(
                    worker,
                    "WorkerRuntimeSettings",
                    lambda: WorkerRuntimeSettings(
                        lease_seconds=3.0,
                        heartbeat_seconds=0.5,
                        poll_seconds=0.1,
                    ),
                )
            )
        if scenario == "claim":
            stack.enter_context(
                patch.object(
                    SqlAlchemyWorkerJobStore,
                    "claim_due_job",
                    after(
                        SqlAlchemyWorkerJobStore.claim_due_job,
                        lambda r: r is not None,
                    ),
                )
            )
        elif scenario == "provider":
            stack.enter_context(
                patch.object(
                    SqlAlchemyInvocationRecorder,
                    "finalize",
                    after(
                        SqlAlchemyInvocationRecorder.finalize,
                        lambda r: True,
                    ),
                )
            )
        elif scenario == "checkpoint":
            original = WorkerRunner._finalize

            async def finalize(*args, **kwargs):
                if kwargs["result"].status.value == "completed":
                    await calls.barrier()
                return await original(*args, **kwargs)

            stack.enter_context(patch.object(WorkerRunner, "_finalize", finalize))
        elif scenario == "approval":
            stack.enter_context(
                patch.object(
                    SqlAlchemyWorkerJobStore,
                    "wait_for_approval",
                    after(
                        SqlAlchemyWorkerJobStore.wait_for_approval,
                        bool,
                    ),
                )
            )
        elif scenario in {"cancel_before", "revoke_before"}:
            original = ToolRegistry._inject_approved_action

            async def inject(self, point):
                await original(self, point)
                if point == "after_consume_commit_before_send_cas":
                    await calls.barrier()

            stack.enter_context(patch.object(ToolRegistry, "_inject_approved_action", inject))
        try:
            yield
        finally:
            calls.channel.close()
