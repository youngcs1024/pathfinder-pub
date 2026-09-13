"""Explicit E5.7 single-worker experiments over the existing owned runtime."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from time import monotonic
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr

from app.db.session import create_database_engine, create_session_factory
from app.domain.tenancy import TenantContext, WorkspaceRole
from tests.performance.adapters import QUERY
from tests.performance.capacity import Experiment, source_sha
from tests.performance.capacity import classify as capacity_classify
from tests.performance.capacity_contracts import lock_digest
from tests.performance.contracts import (
    GuardState,
    LoadProfile,
    Receipt,
    digest,
)
from tests.performance.contracts import (
    Manifest as LoadManifest,
)
from tests.performance.driver import MonotonicClock
from tests.performance.driver import run as run_schedule
from tests.performance.environment import (
    POSTGRES_IMAGE,
    QUEUE_PROFILE,
    ROOT,
    EnvironmentProfile,
    IsolatedEnvironment,
    create_output_directory,
)
from tests.performance.metrics_runtime import sample_database
from tests.performance.queue_contracts import (
    SAFE_STOPS,
    TERMINAL,
    Control,
    Manifest,
    Point,
    Profile,
    Request,
    Result,
    Suite,
    load_profile,
    points,
)
from tests.performance.queue_facts import reconcile_calls, sample, snapshot, state_rows
from tests.performance.queue_metrics import QueueCollector
from tests.performance.queue_report import summarize
from tests.performance.smoke import HTTP, SmokeFailure, approve_synthetic, ingest_resume, require
from tests.performance.workload import publish, queue_profile


class QueueHTTP(HTTP):
    def reserve(self):
        require(self.count < 2048, "request_limit")
        self.count += 1


class QueueExperiment(Experiment):
    def __init__(self, environment, manifest):
        super().__init__(environment, manifest)
        self.metrics = QueueCollector("driver", environment.output_dir)
        self.requests = []
        self.snapshots = []
        self.queue = []
        self.approval_due = {}
        self.approved = set()
        self.automatic = manifest.point.kind != "control"
        self.measurement_start_utc = None
        self.measurement_end_utc = None
        self.actual_end_utc = None
        self.load_result = None
        self.control_result = None
        self.unfinished_calls = 0
        self.worker_stopped = False
        self.monitor_done = asyncio.Event()

    async def rows(self):
        started = monotonic()
        try:
            return await state_rows(self.sessions, self.tenant)
        finally:
            self.metrics.observer_seconds += monotonic() - started

    async def monitor_queue(self):
        for _ in range(350):
            if self.monitor_done.is_set():
                return
            try:
                await self.guard()
                rows, now = await self.rows()
                self.queue.append(sample(rows, now, monotonic()))
                await sample_database(self.sessions, self.metrics)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.reason = self.reason or "guard_failed"
                self.stop_event.set()
                return
            try:
                await asyncio.wait_for(self.monitor_done.wait(), 1.0)
                return
            except TimeoutError:
                pass
        self.reason = self.reason or "deadline"
        self.stop_event.set()

    async def submit(self, mode, phase):
        self.check()
        require(len(self.requests) < self.profile.max_runs, "request_limit")
        self.http.reserve()
        index = len(self.requests)
        request = Request(
            ordinal=index, request_id=uuid4(), mode=mode, phase=phase, started_at=datetime.now(UTC)
        )
        self.requests.append(request)
        token = self.metrics.begin("http", expected_status=202)
        payload = {"mode": mode, "query": QUERY}
        if mode == "application":
            payload["resume_document_id"] = str(self.document)
        receipt = Receipt(sent=None)
        outcome = "failed"
        try:
            response = await self.http.client.post(
                f"/api/v1/workspaces/{self.tenant.workspace_id}/runs",
                json=payload,
                headers={"Idempotency-Key": str(request.request_id)},
            )
            flag = {"true": True, "false": False}.get(response.headers.get("Idempotency-Replayed"))
            self.metrics.end(
                token,
                "succeeded" if response.status_code == 202 else "failed",
                http_status=response.status_code,
                replayed=flag,
            )
            receipt = Receipt(sent=True, http_status=response.status_code)
            if response.status_code == 202:
                run_id = UUID(response.json()["run_id"])
                receipt = Receipt(sent=True, http_status=202, replayed=flag, run_id=run_id)
                self.metrics.update(token, run_id=run_id)
                outcome = "succeeded"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except httpx.TimeoutException:
            outcome = "timeout"
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            pass
        finally:
            if token is not None and self.metrics.samples[token - 1].finished is None:
                self.metrics.end(token, outcome)
            self.requests[index] = request.model_copy(
                update={
                    "finished_at": datetime.now(UTC),
                    "http_status": receipt.http_status,
                    "replayed": receipt.replayed,
                    "run_id": receipt.run_id,
                    "outcome": outcome,
                }
            )
        return receipt

    async def create_one(self, mode, phase):
        receipt = await self.submit(mode, phase)
        require(receipt.run_id is not None, "http_failed")
        return receipt.run_id

    async def approve_ready(self, rows):
        if not self.automatic:
            return
        for row in rows:
            run_id = row["id"]
            if (
                row["status"] != "waiting_approval"
                or row["job_status"] != "done"
                or run_id in self.approved
            ):
                continue
            self.approval_due.setdefault(run_id, monotonic() + self.profile.approval_seconds)
            if monotonic() >= self.approval_due[run_id]:
                if await approve_synthetic(self.http, self.sessions, self.tenant, run_id):
                    self.approved.add(run_id)

    async def approvals(self):
        while not self.monitor_done.is_set():
            rows, _ = await self.rows()
            await self.approve_ready(rows)
            await asyncio.sleep(0.1)

    async def wait_complete(self, run_id):
        while True:
            self.check()
            rows, _ = await self.rows()
            row = next(r for r in rows if r["id"] == run_id)
            require(row["status"] not in {"failed", "cancelled"}, "business_failed")
            if row["status"] == "completed" and row["job_status"] == "done":
                return
            await asyncio.sleep(0.05)

    async def baseline(self):
        for i in range(self.profile.baseline_warmup + self.profile.baseline_samples):
            phase = "warmup" if i < self.profile.baseline_warmup else "measurement"
            run_id = await self.create_one(self.point.mode, phase)
            await self.wait_complete(run_id)

    async def control(self):
        application = await self.create_one("application", "setup")
        while True:
            self.check()
            rows, _ = await self.rows()
            row = next(r for r in rows if r["id"] == application)
            if row["status"] == "waiting_approval" and row["job_status"] == "done":
                break
            await asyncio.sleep(0.02)
        research = await asyncio.gather(*(self.create_one("research", "setup") for _ in range(4)))
        while True:
            self.check()
            rows, _ = await self.rows()
            waiting = next(r for r in rows if r["id"] == application)
            pending = [r for r in rows if r["id"] in research and r["job_status"] == "queued"]
            progressed = any(r["id"] in research and r["status"] != "queued" for r in rows)
            if waiting["status"] == "waiting_approval" and pending and progressed:
                break
            if all(r["status"] in TERMINAL for r in rows if r["id"] in research):
                raise SmokeFailure("evidence_mismatch")
            await asyncio.sleep(0.01)
        target = pending[-1]["id"]
        self.control_result = Control(
            worker_released=True,
            backlog=len(pending),
            approval_run=application,
            cancelled_run=target,
        )

        async def approve():
            accepted = await approve_synthetic(self.http, self.sessions, self.tenant, application)
            self.control_result = self.control_result.model_copy(
                update={
                    "approval_http_accepted": accepted,
                    "approval_accepted_at": datetime.now(UTC),
                }
            )

        async def cancel():
            await self.http.request("POST", self.path(target) + "/cancel")
            self.control_result = self.control_result.model_copy(
                update={"cancel_http_accepted": True, "cancel_accepted_at": datetime.now(UTC)}
            )

        await asyncio.gather(approve(), cancel())
        self.approved.add(application)
        self.automatic = True

    async def load(self):
        p = self.profile
        profile = LoadProfile(
            name="queue-arrivals-v1",
            seed=57,
            warmup_seconds=p.warmup_seconds,
            measurement_seconds=p.measurement_seconds,
            drain_seconds=5.0,
            arrival_rate=self.manifest.arrival_rate,
            connections=8,
            max_requests=128,
            max_inflight=8,
            max_run_seconds=p.warmup_seconds + p.measurement_seconds + 5,
            max_queue_depth=p.max_queue,
        )
        manifest = LoadManifest(
            experiment_id=self.manifest.experiment_id,
            source_sha=self.manifest.source_sha,
            lock_digest=self.manifest.lock_digest,
            database_image=POSTGRES_IMAGE,
            database_version=self.manifest.database_version,
            environment_profile=QUEUE_PROFILE,
            profile=profile,
            profile_digest=digest(profile),
            call_profile=self.manifest.call_profile,
            call_profile_digest=self.manifest.call_profile_digest,
        )
        clock = MonotonicClock()
        # Capture the exact driver's origin; window labels cannot be guessed before startup I/O.
        origin = clock.now()

        class OriginClock:
            first = True

            def now(inner):
                if inner.first:
                    inner.first = False
                    return origin
                return clock.now()

            async def sleep_until(inner, when):
                await clock.sleep_until(when)

        self.measurement_start = origin + p.warmup_seconds
        self.measurement_end = self.measurement_start + p.measurement_seconds
        self.measurement_start_utc = datetime.now(UTC) + timedelta(seconds=p.warmup_seconds)
        self.measurement_end_utc = self.measurement_start_utc + timedelta(
            seconds=p.measurement_seconds
        )

        async def execute(slot):
            return await self.submit(self.point.mode, slot.phase)

        async def guard():
            self.check()
            require(bool(self.health) and monotonic() - self.health[-1].at < 5)
            return GuardState(queue_depth=self.health[-1].queue, resources_within_limits=True)

        self.load_result = await run_schedule(
            manifest,
            execute=execute,
            guard=guard,
            output_dir=self.env.output_dir,
            clock=OriginClock(),
        )
        require(self.load_result.tasks_released)
        if self.load_result.stop_reason not in SAFE_STOPS:
            self.reason = self.reason or self.load_result.stop_reason
        elif self.load_result.stop_reason != "window_complete":
            self.reason = self.reason or self.load_result.stop_reason
        if not self.reason:
            await self.sleep_until(self.measurement_end)
        self.actual_end_utc = (
            self.measurement_start_utc
            - timedelta(seconds=p.warmup_seconds)
            + timedelta(seconds=self.load_result.submission_stopped_at)
            if self.reason
            else self.measurement_end_utc
        )
        self.actual_end_utc = min(self.actual_end_utc, self.measurement_end_utc)

    async def drain(self):
        deadline = monotonic() + self.profile.drain_seconds
        while monotonic() < deadline:
            if self.reason is not None and self.reason not in SAFE_STOPS:
                self.check()
            rows, _ = await self.rows()
            if all(r["status"] in TERMINAL and r["job_status"] in {"done", "dead"} for r in rows):
                return
            await asyncio.sleep(min(0.1, max(0, deadline - monotonic())))

    async def drive(self):
        engine = create_database_engine(SecretStr(self.env.database_url))
        self.sessions = create_session_factory(engine)
        tasks = []
        primary = None
        try:
            async with httpx.AsyncClient(
                base_url=self.env.api_origin,
                trust_env=False,
                follow_redirects=False,
                timeout=5,
                limits=httpx.Limits(max_connections=12),
            ) as client:
                self.http = QueueHTTP(client, self.metrics)
                identity = await self.http.request("GET", "/api/v1/me")
                self.tenant = TenantContext(
                    UUID(identity["workspaces"][0]["workspace_id"]),
                    UUID(identity["user_id"]),
                    WorkspaceRole.ADMIN,
                )
                if self.point.mode == "application":
                    self.document = await ingest_resume(
                        self.sessions, self.tenant, self.manifest.call_profile, self.env.output_dir
                    )
                await self.guard()
                await self.start_worker()
                tasks = [
                    asyncio.create_task(self.monitor_queue()),
                    asyncio.create_task(self.approvals()),
                ]
                try:
                    async with asyncio.timeout(230):
                        operation = {
                            "baseline": self.baseline,
                            "load": self.load,
                            "control": self.control,
                        }[self.point.kind]
                        # A failed observer or approval loop must interrupt the workload promptly.
                        work = asyncio.create_task(operation())
                        tasks.append(work)
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            task.result()
                        if work not in done:
                            self.check()
                            raise SmokeFailure("evidence_mismatch")
                        await work
                except BaseException as error:
                    primary = error
                    self.reason = self.reason or classify(error)
                finally:
                    if len(tasks) == 3 and not tasks[-1].done():
                        tasks[-1].cancel()
                        await asyncio.gather(tasks[-1], return_exceptions=True)
                    if self.measurement_end_utc is not None and self.actual_end_utc is None:
                        self.actual_end_utc = min(datetime.now(UTC), self.measurement_end_utc)
                    try:
                        await snapshot(self, "submission_stopped")
                        if self.reason is None or self.reason in SAFE_STOPS:
                            await self.drain()
                        await snapshot(self, "drain_cutoff")
                    except Exception as error:
                        self.diagnostics.append(classify(error))
                        primary = primary or error
                    self.monitor_done.set()
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            primary = primary or result
                    try:
                        ack = await asyncio.to_thread(self.env.capacity_command, "stop_worker")
                        require(ack.get("stopped") is True)
                        self.worker_stopped = True
                        final = await snapshot(self, "worker_stopped")
                        self.unfinished_calls = await reconcile_calls(self)
                        if self.control_result:
                            statuses = {r.run_id: r.status for r in final.runs}
                            self.control_result = self.control_result.model_copy(
                                update={
                                    "converged": statuses.get(self.control_result.approval_run)
                                    == "completed"
                                    and statuses.get(self.control_result.cancelled_run)
                                    == "cancelled"
                                }
                            )
                            require(self.control_result.converged)
                    except Exception as error:
                        self.diagnostics.append(classify(error))
                        primary = primary or error
                if primary is not None and classify(primary) not in SAFE_STOPS:
                    raise primary
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.wait_for(engine.dispose(), 5)
            self.metrics.finish()


def classify(error):
    category = capacity_classify(error)
    return "environment_failed" if category == "ownership_mismatch" else category


def run_point(output_dir, *, point, profile, authorization, baseline=None):
    profile = Profile.model_validate_json(profile.model_dump_json())
    point = Point.model_validate_json(point.model_dump_json())
    instant = authorization == "ci_instant"
    require(authorization in {"ci_instant", "e57_local_queue_user_approved_v1"})
    require(profile.name == ("queue-instant-ci-v1" if instant else "queue-e57-v1"))
    policy = queue_profile(instant=instant)
    if point.kind == "load":
        require(baseline is not None and len(baseline.service_seconds) == profile.baseline_samples)
    env = IsolatedEnvironment(
        EnvironmentProfile(QUEUE_PROFILE),
        output_dir,
        call_profile=policy.model_dump(mode="json"),
        metrics=True,
        capacity=True,
    )
    sha = source_sha()
    experiment = None
    reason = "window_complete"
    interrupted = None
    try:
        env.start()
        import psycopg

        with psycopg.connect(
            env.database_url.replace("postgresql+psycopg", "postgresql"), connect_timeout=2
        ) as connection:
            version = connection.execute("SHOW server_version").fetchone()[0].split()[0]
        manifest = Manifest(
            experiment_id=uuid4(),
            source_sha=sha,
            lock_digest=lock_digest(ROOT),
            profile=profile,
            profile_digest=digest(profile),
            call_profile=policy,
            call_profile_digest=digest(policy),
            point=point,
            authorization=authorization,
            database_image=POSTGRES_IMAGE,
            database_version=version,
            baseline=baseline,
            arrival_rate=point.factor / baseline.mean_seconds if baseline is not None else None,
        )
        publish(output_dir, "queue-manifest.json", manifest)
        experiment = QueueExperiment(env, manifest)
        asyncio.run(experiment.drive())
        reason = experiment.reason or "window_complete"
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        reason, interrupted = "cancelled", error
    except Exception as error:
        reason = classify(error)
    finally:
        cleanup = env.close()
    result = Result(
        point=point,
        source_sha=sha,
        stop=reason,
        resources_released=cleanup.get("resources_released") is True,
    )
    if experiment is not None:
        try:
            result = summarize(experiment, reason, cleanup)
        except Exception:
            result = result.model_copy(update={"diagnostics": ("report_failed",)})
    if env.output_created:
        publish(output_dir, "queue-result.json", result)
    if interrupted:
        raise interrupted
    return result


def run_suite(directory, *, authorization, selected_profile):
    require(
        authorization == "e57_local_queue_user_approved_v1" and selected_profile == "queue-e57-v1"
    )
    profile = load_profile(selected_profile)
    create_output_directory(directory)
    started = monotonic()
    sha = source_sha()
    baselines = {}
    stopped = set()
    abort = False
    results = []
    for index, point in enumerate(points()):
        reason = None
        if monotonic() - started + profile.max_seconds > profile.suite_seconds:
            reason = "suite_deadline"
        elif abort or point.mode in stopped:
            reason = "previous_stop"
        elif point.kind == "load" and point.mode not in baselines:
            reason = "baseline_incomplete"
        if reason:
            result = Result(point=point, source_sha=sha, status="NOT_RUN", stop=reason)
        else:
            result = run_point(
                directory / f"point-{index:02d}",
                point=point,
                profile=profile,
                authorization=authorization,
                baseline=baselines.get(point.mode) if point.kind == "load" else None,
            )
            if result.baseline:
                baselines[point.mode] = result.baseline
            if result.status != "PASS":
                abort = True
            elif result.stop != "window_complete":
                stopped.add(point.mode)
        results.append(result)
        print(
            f"queue/{point.kind}/{point.mode}/{point.factor}: {result.status} {result.stop}",
            flush=True,
        )
    suite = Suite(
        source_sha=sha, authorization=authorization, profile=profile, results=tuple(results)
    )
    publish(directory, "queue-suite.json", suite)
    return suite
