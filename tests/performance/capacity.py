"""E5.6 explicit capacity execution. Never imported as a production composition root."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    LLMInvocation,
    MockSubmission,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
)
from app.db.session import create_database_engine, create_session_factory
from app.domain.tenancy import TenantContext, WorkspaceRole
from tests.performance.adapters import QUERY
from tests.performance.capacity_contracts import (
    Health,
    Manifest,
    Point,
    Profile,
    Result,
    RunResult,
    Suite,
    load_profile,
    lock_digest,
    points,
    safe_read,
)
from tests.performance.capacity_metrics import CapacityCollector, CapacityPacket
from tests.performance.capacity_transport import CapacityHTTP, Consumer, close_consumers
from tests.performance.contracts import GuardState, LoadProfile, Receipt, digest
from tests.performance.contracts import Manifest as LoadManifest
from tests.performance.driver import run as run_schedule
from tests.performance.environment import (
    POSTGRES_IMAGE,
    PROFILE,
    ROOT,
    EnvironmentError,
    EnvironmentProfile,
    IsolatedEnvironment,
    create_output_directory,
)
from tests.performance.metrics import Summary, Value, difference, distribution, utc_difference
from tests.performance.metrics_runtime import sample_database
from tests.performance.smoke import (
    SmokeFailure,
    approve_synthetic,
    ingest_resume,
    read_calls,
    require,
)
from tests.performance.workload import profile as call_profile
from tests.performance.workload import publish


class CapacityStop(Exception):
    def __init__(self, category):
        self.category = category
        super().__init__(category)


def source_sha():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()


class Experiment:
    def __init__(self, environment, manifest):
        self.env = environment
        self.manifest = manifest
        self.profile = manifest.profile
        self.point = manifest.point
        self.metrics = CapacityCollector("driver", environment.output_dir)
        self.records = []
        self.health = []
        self.created = {}
        self.facts = []
        self.tasks = []
        self.stop_event = asyncio.Event()
        self.reason = None
        self.measurement_start = None
        self.measurement_end = None
        self.observation_end = None
        self.offered = 0
        self.dropped = 0
        self.worker_started = False
        self.http = None
        self.diagnostics = []

    def check(self):
        if self.reason:
            raise CapacityStop(self.reason)

    async def sleep_until(self, deadline):
        self.check()
        try:
            await asyncio.wait_for(self.stop_event.wait(), max(0, deadline - monotonic()))
        except TimeoutError:
            pass
        self.check()

    async def guard(self):
        started = monotonic()
        try:
            packet = await asyncio.to_thread(self.env.capacity_command, "health")
            meminfo = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            async with asyncio.timeout(2), self.sessions() as session, session.begin():
                await session.execute(text("SET TRANSACTION READ ONLY"))
                connections = await session.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                        "AND pid <> pg_backend_pid()"
                    )
                )
                queue = await session.scalar(
                    select(func.count())
                    .select_from(RunJob)
                    .where(RunJob.status.in_(("queued", "leased")))
                )
            result = Health(
                at=monotonic(),
                memory=packet["memory"],
                tmpfs=packet["tmpfs"],
                rss=packet["rss"],
                available=int(meminfo["MemAvailable"].split()[0]) * 1024,
                disk_free=shutil.disk_usage(self.env.output_dir).free,
                connections=connections,
                queue=queue,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            result = Health(at=monotonic())
        self.metrics.observer_seconds += monotonic() - started
        self.health.append(result)
        if reason := result.stop(self.profile):
            self.reason = reason
            self.stop_event.set()
        return result

    async def monitor(self):
        for _ in range(120):
            await self.guard()
            if self.reason:
                return
            await sample_database(self.sessions, self.metrics)
            await asyncio.sleep(1)
        self.reason = "deadline"
        self.stop_event.set()

    async def start_worker(self):
        self.check()
        require(not self.worker_started)
        await asyncio.to_thread(self.env.capacity_command, "start_worker")
        self.worker_started = True

    def path(self, run_id):
        return f"/api/v1/workspaces/{self.tenant.workspace_id}/runs/{run_id}"

    async def create(self, mode, phase):
        self.check()
        require(len(self.created) < 128, "request_limit")
        payload = {"mode": mode, "query": QUERY}
        if mode == "application":
            payload["resume_document_id"] = str(self.document)
        result = await self.http.request(
            "POST",
            f"/api/v1/workspaces/{self.tenant.workspace_id}/runs",
            expected=202,
            headers={"Idempotency-Key": str(uuid4())},
            json=payload,
        )
        run_id = UUID(result["run_id"])
        require(run_id not in self.created)
        self.created[run_id] = (mode, phase)
        return run_id

    async def wait_run(self, run_id):
        approved = False
        while True:
            self.check()
            state = await self.http.request("GET", self.path(run_id))
            require(state["status"] not in {"failed", "cancelled"}, "business_failed")
            if state["status"] == "waiting_approval" and not approved:
                require(self.created[run_id][0] == "application")
                approved = await approve_synthetic(self.http, self.sessions, self.tenant, run_id)
            if state["status"] == "completed":
                async with self.sessions() as session:
                    job = await session.scalar(
                        select(RunJob.status).where(
                            RunJob.run_id == run_id, RunJob.workspace_id == self.tenant.workspace_id
                        )
                    )
                if job == "done":
                    return
            await self.sleep_until(monotonic() + 0.25)

    async def api(self):
        p = self.profile
        load = LoadProfile(
            name="capacity-api-v1",
            warmup_seconds=p.warmup_seconds,
            measurement_seconds=p.measurement_seconds,
            drain_seconds=p.drain_seconds,
            arrival_rate=float(self.point.level),
            connections=10,
            max_requests=128,
            max_inflight=10,
            max_run_seconds=p.warmup_seconds + p.measurement_seconds + p.drain_seconds,
            max_queue_depth=p.max_queue,
        )
        manifest = LoadManifest(
            experiment_id=self.manifest.experiment_id,
            source_sha=self.manifest.source_sha,
            lock_digest=self.manifest.lock_digest,
            database_image=POSTGRES_IMAGE,
            database_version=self.manifest.database_version,
            environment_profile=PROFILE,
            profile=load,
            profile_digest=digest(load),
            call_profile=self.manifest.call_profile,
            call_profile_digest=self.manifest.call_profile_digest,
        )

        async def execute(slot):
            if slot.ordinal % 2 == 0 or not self.created:
                run_id = await self.create("research", slot.phase)
                return Receipt(sent=True, http_status=202, replayed=False, run_id=run_id)
            self.check()
            await self.http.request("GET", self.path(next(iter(self.created))))
            return Receipt(sent=True, http_status=200)

        async def guard():
            self.check()
            # The independent 1s observer owns slow Docker inspection; arrival slots never do it.
            require(bool(self.health) and monotonic() - self.health[-1].at < 5)
            return GuardState(queue_depth=self.health[-1].queue, resources_within_limits=True)

        origin = monotonic()
        self.measurement_start = origin + p.warmup_seconds
        self.measurement_end = self.measurement_start + p.measurement_seconds
        result = await run_schedule(
            manifest, execute=execute, guard=guard, output_dir=self.env.output_dir
        )
        self.offered = result.warmup.offered + result.measurement.offered
        self.dropped = result.warmup.generator_dropped + result.measurement.generator_dropped
        if result.stop_reason not in {"window_complete", "request_limit"}:
            raise CapacityStop(self.reason or result.stop_reason)
        require(result.tasks_released)
        await self.sleep_until(self.measurement_end)

    async def consumers(self, run_id, cursors=None, delay=0.0):
        group = [
            Consumer(
                self.http,
                run_id,
                self.path(run_id) + "/events",
                self.records,
                cursor=cursor,
                delay=delay,
            )
            for cursor in (cursors if cursors is not None else [0] * self.point.level)
        ]
        tasks = [asyncio.create_task(c.run()) for c in group]
        self.tasks.extend(tasks)
        async with asyncio.timeout(10):
            while not all(c.established.is_set() for c in group):
                self.check()
                for task in tasks:
                    if task.done():
                        task.result()
                        raise CapacityStop("http_failed")
                await asyncio.sleep(0.01)
        return group, tasks

    async def sse(self):
        run_id = await self.create("research", "setup")
        delay = self.profile.slow_seconds if self.point.scenario == "slow" else 0.0
        group, tasks = await self.consumers(run_id, delay=delay)
        self.measurement_start = monotonic() + self.profile.warmup_seconds
        self.measurement_end = self.measurement_start + self.profile.measurement_seconds
        await self.sleep_until(self.measurement_start)
        if self.point.scenario in {"reconnect", "slow"}:
            if self.point.scenario == "slow":
                await self.sleep_until(
                    self.measurement_start + self.profile.measurement_seconds / 2
                )
            await close_consumers(tasks)
            cursors = [c.cursor for c in group]
            require(all(cursor > 0 for cursor in cursors))
            group, tasks = await self.consumers(run_id, cursors, delay)
        if self.point.scenario != "idle":
            await self.start_worker()
        await self.sleep_until(self.measurement_end)
        if self.point.scenario != "idle":
            async with asyncio.timeout(self.profile.drain_seconds):
                await self.wait_run(run_id)
                await asyncio.gather(*tasks)
        await close_consumers(tasks)

    async def mixed(self):
        self.document = await ingest_resume(
            self.sessions, self.tenant, self.manifest.call_profile, self.env.output_dir
        )
        await self.start_worker()
        self.measurement_start = monotonic() + self.profile.warmup_seconds
        self.measurement_end = self.measurement_start + self.profile.measurement_seconds

        async def pair(phase):
            ids = [await self.create(mode, phase) for mode in ("research", "application")]
            for run_id in ids:
                # A single stream per Run, not another worker or another fake executor.
                consumer = Consumer(self.http, run_id, self.path(run_id) + "/events", self.records)
                task = asyncio.create_task(consumer.run())
                self.tasks.append(task)
                await self.wait_run(run_id)
                await task

        await pair("warmup")
        require(monotonic() <= self.measurement_start, "deadline")
        await self.sleep_until(self.measurement_start)
        # If warmup exceeded its fixed window, do not quietly move the measurement window.
        require(monotonic() < self.measurement_end, "deadline")
        await pair("measurement")
        await self.sleep_until(self.measurement_end)

    async def capture(self):
        facts = []
        if self.created:
            await self.http.request(
                "GET",
                f"/api/v1/workspaces/{uuid4()}/runs/{next(iter(self.created))}/events",
                expected=404,
            )
        async with asyncio.timeout(5), self.sessions() as session:
            for run_id, (mode, phase) in self.created.items():
                run = await session.get(Run, run_id)
                jobs = list(
                    await session.scalars(
                        select(RunJob).where(
                            RunJob.workspace_id == self.tenant.workspace_id, RunJob.run_id == run_id
                        )
                    )
                )
                require(
                    run is not None
                    and run.workspace_id == self.tenant.workspace_id
                    and run.created_by_user_id == self.tenant.actor_user_id
                    and len(jobs) == 1
                )
                events = tuple(
                    await session.scalars(
                        select(RunEvent.seq)
                        .where(
                            RunEvent.workspace_id == self.tenant.workspace_id,
                            RunEvent.run_id == run_id,
                        )
                        .order_by(RunEvent.seq)
                    )
                )
                require(events == tuple(range(1, len(events) + 1)))
                models = await session.scalar(
                    select(func.count())
                    .select_from(LLMInvocation)
                    .where(
                        LLMInvocation.workspace_id == self.tenant.workspace_id,
                        LLMInvocation.run_id == run_id,
                    )
                )
                effects = list(
                    await session.scalars(
                        select(MockSubmission).where(
                            MockSubmission.workspace_id == self.tenant.workspace_id,
                            MockSubmission.run_id == run_id,
                        )
                    )
                )
                if mode == "research":
                    require(not effects)
                elif run.status == "completed":
                    actions = list(
                        await session.scalars(
                            select(ActionIntent).where(
                                ActionIntent.workspace_id == self.tenant.workspace_id,
                                ActionIntent.run_id == run_id,
                            )
                        )
                    )
                    requests = list(
                        await session.scalars(
                            select(ApprovalRequest).where(
                                ApprovalRequest.workspace_id == self.tenant.workspace_id,
                                ApprovalRequest.run_id == run_id,
                            )
                        )
                    )
                    decisions = list(
                        await session.scalars(
                            select(ApprovalDecision).where(
                                ApprovalDecision.workspace_id == self.tenant.workspace_id,
                                ApprovalDecision.approval_request_id.in_([r.id for r in requests]),
                            )
                        )
                    )
                    require(len(effects) == len(actions) == len(requests) == len(decisions) == 1)
                    require(
                        actions[0].status == "succeeded"
                        and requests[0].status == "consumed"
                        and decisions[0].decision == "approve"
                        and effects[0].idempotency_key == actions[0].idempotency_key
                    )
                    for field in (
                        "args_digest",
                        "target_digest",
                        "approval_binding_digest",
                        "approval_binding_version",
                    ):
                        require(getattr(actions[0], field) == getattr(requests[0], field))
                facts.append(
                    RunResult(
                        run_id=run_id,
                        mode=mode,
                        phase=phase,
                        status=run.status,
                        job_status=jobs[0].status,
                        events=events,
                        model_attempts=models,
                        mock_effects=len(effects),
                    )
                )
            self.facts = facts
            if self.worker_started and all(f.status == "completed" for f in facts):
                calls = read_calls(self.env.output_dir)
                providers = [call for call in calls if call.call in {"chat", "embedding"}]
                invocations = list(
                    await session.scalars(
                        select(LLMInvocation).where(
                            LLMInvocation.workspace_id == self.tenant.workspace_id
                        )
                    )
                )
                require(
                    {call.invocation_id for call in providers} == {row.id for row in invocations}
                    and len(providers) == len(invocations)
                )
                require(
                    all(row.status == "succeeded" and row.provider == "fake" for row in invocations)
                )
                tools = list(
                    await session.scalars(
                        select(ToolInvocation).where(
                            ToolInvocation.workspace_id == self.tenant.workspace_id
                        )
                    )
                )
                for call_kind, tool in (
                    ("search", "search_web"),
                    ("mock_submit", "submit_mock_application"),
                ):
                    require(
                        sum(call.call == call_kind for call in calls)
                        == sum(row.tool_name == tool for row in tools)
                    )
                require(all(row.status == "succeeded" for row in tools))
            # The full, owned database must reconcile to driver-created IDs (including unfinished).
            require(set(await session.scalars(select(Run.id))) == set(self.created))
            require(
                await session.scalar(select(func.count()).select_from(RunJob)) == len(self.created)
            )

    async def drive(self):
        engine = create_database_engine(SecretStr(self.env.database_url))
        self.sessions = create_session_factory(engine)
        monitor = None
        try:
            async with (
                asyncio.timeout(45),
                httpx.AsyncClient(
                    base_url=self.env.api_origin,
                    trust_env=False,
                    follow_redirects=False,
                    limits=httpx.Limits(max_connections=110, max_keepalive_connections=10),
                    timeout=httpx.Timeout(5, read=40),
                ) as client,
            ):
                self.http = CapacityHTTP(client, self.metrics)
                identity = await self.http.request("GET", "/api/v1/me")
                self.tenant = TenantContext(
                    UUID(identity["workspaces"][0]["workspace_id"]),
                    UUID(identity["user_id"]),
                    WorkspaceRole.ADMIN,
                )
                # Probe a real Run through a foreign workspace path later in capture;
                # no actor/role override enters the request body.
                await self.guard()
                self.check()
                monitor = asyncio.create_task(self.monitor())
                primary = None
                try:
                    if self.point.scenario == "api":
                        await self.api()
                    elif self.point.scenario == "mixed":
                        await self.mixed()
                    else:
                        await self.sse()
                    self.check()
                except BaseException as error:
                    primary = error
                finally:
                    for operation in (lambda: close_consumers(self.tasks), self.capture):
                        try:
                            await operation()
                        except Exception as error:
                            self.diagnostics.append(classify(error))
                            primary = primary or error
                    self.observation_end = monotonic()
                if primary is not None:
                    raise primary
        finally:
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            self.metrics.finish()
            await asyncio.wait_for(engine.dispose(), 5)


def summarize(experiment, source, reason, cleanup):
    p = experiment
    roles = ("api", "driver", "supervisor") + (("worker",) if p.worker_started else ())
    packets, missing = [], []
    for role in roles:
        path = p.env.output_dir / f"metrics-{role}.json"
        if path.exists():
            packets.append(safe_read(path, CapacityPacket))
        else:
            missing.append(role)
    diagnostics = list(p.diagnostics)
    if missing or any(q.dropped or q.write_failed for q in packets):
        diagnostics.append("metrics_incomplete")
    if len({q.process_id for q in packets}) != len(packets):
        diagnostics.append("correctness_failed")
    if cleanup.get("category") or not cleanup.get("resources_released"):
        diagnostics.append("cleanup_failed")
    start, end = p.measurement_start, p.measurement_end
    if end is not None and p.observation_end is not None:
        end = min(end, p.observation_end)
    groups = []
    cross = 0
    event_values = []
    all_http = []
    api = next((q for q in packets if q.role == "api"), None)
    for packet in packets:
        all_http.extend(s for s in packet.samples if s.metric == "http")
        measured = [
            s
            for s in packet.samples
            if start is not None
            and end is not None
            and start <= s.started < end
            and s.finished is not None
            and s.finished <= end
        ]
        cross += sum(
            start is not None
            and end is not None
            and (
                (s.started < end and (s.finished is None or s.finished > end))
                or (s.started < start and (s.finished is None or s.finished > start))
            )
            for s in packet.samples
        )
        for metric, outcome, status in sorted(
            {(s.metric, s.outcome, s.expected_status) for s in measured},
            key=lambda k: (k[0], k[1], k[2] or 0),
        ):
            samples = [
                s
                for s in measured
                if (s.metric, s.outcome, s.expected_status) == (metric, outcome, status)
            ]
            duration = end - start
            groups.append(
                Summary(
                    role=packet.role,
                    metric=metric,
                    outcome=outcome,
                    expected_status=status,
                    count=len(samples),
                    duration_seconds=distribution(
                        [difference(s.started, s.finished) for s in samples]
                    ),
                    sql_started=sum(s.sql_started for s in samples),
                    sql_finished=sum(s.sql_finished for s in samples),
                    sql_failed=sum(s.sql_failed for s in samples),
                    sql_duration_seconds=Value(value=sum(s.sql_seconds for s in samples)),
                    process_observation_window_seconds=Value(value=duration),
                    calls_per_second=Value(value=len(samples) / duration),
                    sql_executions_per_second=Value(
                        value=sum(s.sql_started for s in samples) / duration
                    ),
                )
            )
        event_values.extend(
            utc_difference(s.event_recorded_at, s.recorded_at)
            for s in measured
            if s.metric == "sse_event"
        )
    correct = True
    if reason == "window_complete":
        if p.point.scenario in {"api", "idle"}:
            correct &= all(
                f.status == "queued" and f.job_status == "queued" and f.model_attempts == 0
                for f in p.facts
            )
        else:
            correct &= bool(p.facts) and all(
                f.status == "completed" and f.job_status == "done" and f.model_attempts > 0
                for f in p.facts
            )
        for run in p.facts:
            records = [r for r in p.records if r.run_id == run.run_id]
            for record in records:
                correct &= (
                    record.received
                    == run.events[record.cursor : record.cursor + len(record.received)]
                )
                correct &= record.processed == record.received[: len(record.processed)]
                if record.outcome == "closed":
                    correct &= record.processed == run.events[record.cursor :]
            if p.point.scenario not in {"api", "idle"}:
                correct &= any(r.terminal for r in records)
        if p.point.scenario not in {"api", "mixed"}:
            first = p.records[: p.point.level]
            correct &= len(first) == p.point.level and all(
                r.established_at is not None for r in first
            )
        correct &= api is not None and api.pool_remaining == 0
        try:
            calls = read_calls(p.env.output_dir)
            model_calls = [c for c in calls if c.call in {"chat", "embedding"}]
            # Ingest embedding has no run; the remainder must match run-scoped Factory attempts.
            correct &= len(model_calls) == sum(f.model_attempts for f in p.facts) + (
                1 if p.point.scenario == "mixed" else 0
            )
        except Exception:
            correct = False
    if not correct:
        diagnostics.append("correctness_failed")
    timeline = []
    seconds = 0.0
    for record in p.records:
        if record.established_at is not None and record.closed_at is not None:
            timeline += [(record.established_at, 1), (record.closed_at, -1)]
            if start is not None and end is not None:
                seconds += max(0.0, min(end, record.closed_at) - max(start, record.established_at))
    active = peak = 0
    for _, delta in sorted(timeline):
        active += delta
        peak = max(peak, active)
    final_reason = (
        reason if reason != "window_complete" else (diagnostics[0] if diagnostics else reason)
    )
    return Result(
        point=p.point,
        source_sha=source,
        status="PASS" if final_reason == "window_complete" else "IN_PROGRESS",
        stop=final_reason,
        diagnostics=tuple(diagnostics),
        resources_released=cleanup.get("resources_released") is True,
        completeness=not missing and "metrics_incomplete" not in diagnostics,
        correctness=correct if reason == "window_complete" else None,
        measurement_start=start,
        measurement_end=end,
        actual_observation_end=p.observation_end,
        offered=p.offered,
        generator_dropped=p.dropped,
        http_requests=p.http.count if p.http else 0,
        runs=tuple(p.facts),
        connections=tuple(p.records),
        health=tuple(p.health),
        established_peak=peak,
        connection_seconds=seconds,
        summaries=tuple(groups),
        cross_window_samples=cross,
        missing_roles=tuple(missing),
        sse_latency=distribution(event_values),
        counts={
            "http_202": sum(s.http_status == 202 for s in all_http),
            "http_failed": sum(s.outcome == "failed" for s in all_http),
            "http_unknown": sum(s.http_status is None for s in all_http),
            "replayed": sum(s.replayed is True for s in all_http),
            "unique_runs": len(p.facts),
            "unfinished_runs": sum(
                f.status not in {"completed", "failed", "cancelled"} for f in p.facts
            ),
            "unprocessed_events": sum(len(r.received) - len(r.processed) for r in p.records),
            "pool_peak": api.pool_peak if api else 0,
            "pool_remaining": api.pool_remaining if api else 0,
        },
    )


def classify(error):
    if isinstance(error, CapacityStop):
        return error.category
    if isinstance(error, TimeoutError):
        return "deadline"
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        return "cancelled"
    if isinstance(error, SmokeFailure):
        return {
            "evidence_mismatch": "correctness_failed",
            "business_failed": "correctness_failed",
        }.get(error.category, error.category)
    if isinstance(error, EnvironmentError):
        return (
            "ownership_mismatch" if error.category == "ownership_mismatch" else "environment_failed"
        )
    return "correctness_failed"


def run_point(output_dir, *, point, profile, authorization):
    profile = Profile.model_validate_json(profile.model_dump_json())
    point = Point.model_validate_json(point.model_dump_json())
    if authorization == "ci_instant":
        require(profile.name == "capacity-instant-ci-v1" and point.level == 1)
    else:
        require(
            authorization == "e56_local_capacity_user_approved_v1"
            and profile.name == "capacity-e56-v1"
        )
    policy = call_profile("instant-v1" if authorization == "ci_instant" else "delayed-v1")
    sha = source_sha()
    environment = IsolatedEnvironment(
        EnvironmentProfile(PROFILE),
        output_dir,
        call_profile=policy.model_dump(mode="json"),
        metrics=True,
        capacity=True,
    )
    experiment = None
    reason = "window_complete"
    interrupted = None
    try:
        environment.start()
        # Runtime identity is read before any workload and written once with the fixed policy.
        import psycopg

        with psycopg.connect(
            environment.database_url.replace("postgresql+psycopg", "postgresql"), connect_timeout=2
        ) as connection:
            version = connection.execute("SHOW server_version").fetchone()[0].split()[0]
        manifest = Manifest(
            experiment_id=uuid4(),
            authorization=authorization,
            source_sha=sha,
            lock_digest=lock_digest(ROOT),
            profile=profile,
            profile_digest=digest(profile),
            call_profile=policy,
            call_profile_digest=digest(policy),
            point=point,
            database_image=POSTGRES_IMAGE,
            database_version=version,
        )
        publish(output_dir, "capacity-manifest.json", manifest)
        experiment = Experiment(environment, manifest)
        asyncio.run(experiment.drive())
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        reason, interrupted = "cancelled", error
    except Exception as error:
        reason = classify(error)
    finally:
        cleanup = environment.close()
    if experiment is not None:
        try:
            result = summarize(experiment, sha, reason, cleanup)
        except Exception:
            result = Result(
                point=point,
                source_sha=sha,
                status="IN_PROGRESS",
                stop=reason if reason != "window_complete" else "report_failed",
                diagnostics=("report_failed",),
                resources_released=cleanup.get("resources_released") is True,
            )
    else:
        result = Result(
            point=point,
            source_sha=sha,
            status="IN_PROGRESS",
            stop=reason,
            resources_released=cleanup.get("resources_released") is True,
        )
    if environment.output_created:
        try:
            publish(output_dir, "capacity-result.json", result)
        except Exception:
            result = result.model_copy(
                update={
                    "status": "IN_PROGRESS",
                    "diagnostics": (*result.diagnostics, "report_failed"),
                }
            )
    if interrupted:
        raise interrupted
    return result


def run_suite(directory, *, authorization, selected_profile):
    require(authorization == "e56_local_capacity_user_approved_v1")
    require(selected_profile == "capacity-e56-v1")
    profile = load_profile(selected_profile)
    create_output_directory(directory)
    started = monotonic()
    sha = source_sha()
    results = []
    stopped = set()
    abort = False
    for index, point in enumerate(points()):
        reason = None
        if monotonic() - started + profile.max_seconds > profile.suite_seconds:
            reason = "suite_deadline"
        elif abort or point.scenario in stopped:
            reason = "previous_stop"
        if reason:
            result = Result(point=point, source_sha=sha, status="NOT_RUN", stop=reason)
        else:
            result = run_point(
                directory / f"point-{index:02d}",
                point=point,
                profile=profile,
                authorization=authorization,
            )
            if result.status != "PASS":
                stopped.add(point.scenario)
                abort = not result.resources_released or result.stop not in {
                    "resource_limit",
                    "queue_limit",
                    "request_limit",
                    "deadline",
                    "http_failed",
                }
        results.append(result)
        print(f"{point.scenario}/{point.level}: {result.status} {result.stop}", flush=True)
    suite = Suite(
        source_sha=sha, profile=profile, authorization=authorization, results=tuple(results)
    )
    publish(directory, "capacity-suite.json", suite)
    return suite
