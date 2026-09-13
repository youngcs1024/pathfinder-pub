"""Explicit bounded E5.9 matrix using the production chain and owned fault controller."""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, time
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr
from sqlalchemy import func, select, text, update

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
    WorkspaceMembership,
)
from app.db.session import create_database_engine, create_session_factory
from app.domain.tenancy import TenantContext, WorkspaceRole
from tests.performance.adapters import QUERY
from tests.performance.capacity import source_sha
from tests.performance.capacity_contracts import Health, lock_digest, safe_read
from tests.performance.capacity_metrics import CapacityCollector, CapacityPacket
from tests.performance.contracts import digest
from tests.performance.environment import (
    PROFILE,
    ROOT,
    EnvironmentError,
    EnvironmentProfile,
    IsolatedEnvironment,
    create_output_directory,
)
from tests.performance.fault_contracts import (
    AUTHORIZATION,
    CRASH,
    FaultCall,
    Lifecycle,
    Manifest,
    ModelFact,
    Result,
    Snapshot,
    Suite,
    load_profile,
    points,
)
from tests.performance.smoke import HTTP, approve_synthetic, ingest_resume
from tests.performance.workload import profile as call_profile
from tests.performance.workload import publish


class FaultStop(Exception):
    def __init__(self, category):
        self.category = category
        super().__init__(category)


def require(value):
    if not value:
        raise FaultStop("correctness_failed")


async def snapshot(sessions, tenant, run_id):
    async with sessions() as session:

        async def rows(model):
            return list(
                await session.scalars(
                    select(model).where(
                        model.workspace_id == tenant.workspace_id,
                        model.run_id == run_id,
                    )
                )
            )

        run = await session.get(Run, run_id)
        jobs, events, models, tools, actions, requests, effects = [
            await rows(m)
            for m in (
                RunJob,
                RunEvent,
                LLMInvocation,
                ToolInvocation,
                ActionIntent,
                ApprovalRequest,
                MockSubmission,
            )
        ]
        require(run is not None and run.workspace_id == tenant.workspace_id)
        require(len(jobs) == 1 and len(actions) <= 1 and len(requests) <= 1)
        require(await session.scalar(select(func.count()).select_from(Run)) == 1)
        require(await session.scalar(select(func.count()).select_from(RunJob)) == 1)
        decisions = list(
            await session.scalars(
                select(ApprovalDecision).where(
                    ApprovalDecision.workspace_id == tenant.workspace_id,
                )
            )
        )
    job = jobs[0]
    action = actions[0] if actions else None
    request = requests[0] if requests else None
    invocations = [t for t in tools if t.action_intent_id is not None]
    require(len(invocations) <= 1)
    invocation = invocations[0] if invocations else None
    if action is not None and request is not None:
        require(request.action_intent_id == action.id)
        require(
            all(
                getattr(action, k) == getattr(request, k)
                for k in (
                    "args_digest",
                    "target_digest",
                    "approval_binding_digest",
                )
            )
        )
        require(action.idempotency_key == str(action.id))
        require(all(d.approval_request_id == request.id for d in decisions))
        require(
            all(
                e.action_intent_id == action.id and e.idempotency_key == action.idempotency_key
                for e in effects
            )
        )
    events.sort(key=lambda e: e.seq)
    return Snapshot(
        run_id=run_id,
        job_id=job.id,
        run_status=run.status,
        job_status=job.status,
        job_attempt=job.attempt,
        lease_expires_at=job.lease_expires_at.timestamp() if job.lease_expires_at else None,
        events=tuple(e.type for e in events),
        seq=tuple(e.seq for e in events),
        models=tuple(
            ModelFact(
                id=m.id,
                node=m.graph_node,
                request_hash=m.request_hash,
                status=m.status,
                cost_known=m.estimated_cost is not None,
            )
            for m in sorted(models, key=lambda m: str(m.id))
        ),
        tools=tuple(sorted((t.id for t in tools), key=str)),
        action_id=action.id if action else None,
        request_id=request.id if request else None,
        invocation_id=invocation.id if invocation else None,
        action_status=action.status if action else None,
        invocation_status=invocation.status if invocation else None,
        approval_status=request.status if request else None,
        args_digest=action.args_digest if action else None,
        target_digest=action.target_digest if action else None,
        binding_digest=action.approval_binding_digest if action else None,
        decision_count=len(decisions),
        recovery_attempts=action.recovery_attempts if action else 0,
        effects=len(effects),
    )


def read_evidence(directory):
    life = tuple(
        sorted(
            (safe_read(p, Lifecycle, 8192) for p in directory.glob("fault-life-*.json")),
            key=lambda r: r.at,
        )
    )
    calls = tuple(
        sorted(
            (safe_read(p, FaultCall, 8192) for p in directory.glob("fault-call-*.json")),
            key=lambda r: (r.sequence, r.phase),
        )
    )
    started = [c for c in calls if c.phase == "started"]
    require([c.sequence for c in started] == list(range(1, len(started) + 1)))
    require(len(started) <= 63)
    for end in (c for c in calls if c.phase == "finished"):
        require(
            sum(
                (s.generation, s.sequence, s.call, s.invocation_id, s.action_id)
                == (end.generation, end.sequence, end.call, end.invocation_id, end.action_id)
                for s in started
            )
            == 1
        )
    return life, calls


def check_facts(point, before, final, life, calls):
    require(before is not None and final is not None)
    require(final.run_status == point.expected_run and final.job_status == "done")
    require(final.lease_expires_at is None and final.job_id == before.job_id)
    require(final.events.count("run." + point.expected_run) == 1)
    require(all(m.status == "succeeded" for m in final.models))
    started = [c for c in calls if c.phase == "started"]
    provider_ids = {c.invocation_id for c in started if c.call in {"chat", "embedding"}}
    require(provider_ids == {m.id for m in final.models})
    barrier = [e for e in life if e.kind == "barrier"]
    require(len(barrier) == 1)
    if point.scenario in CRASH:
        require([e.kind for e in life] == ["started", "barrier", "killed", "started"])
        require(life[2].exit_code == -9 and life[2].generation == 1 and life[3].generation == 2)
        require(life[2].at <= life[3].at)
        if point.scenario in CRASH[:3]:
            require(final.events.count("job.lease_expired") == 1 and final.job_attempt == 2)
    else:
        require([e.kind for e in life] == ["started", "barrier", "released"])
    if point.scenario == "claim":
        require(
            before.run_status == "queued" and before.job_status == "leased" and not before.models
        )
    if point.scenario == "provider":
        require(len(before.models) == 1 and before.models[0].status == "succeeded")
        first = before.models[0]
        require(
            sum(m.node == first.node and m.request_hash == first.request_hash for m in final.models)
            == 2
        )
    if point.scenario == "checkpoint":
        require(before.run_status == "running" and before.job_status == "leased")
        require(before.models == final.models and before.tools == final.tools)
    if point.mode == "research":
        require(final.effects == 0 and final.action_id is None)
        return
    require(final.action_id is not None and final.decision_count == 1)
    require(final.approval_status == "consumed")
    for key in ("action_id", "request_id", "args_digest", "target_digest", "binding_digest"):
        require(getattr(before, key) == getattr(final, key))
    require(all(c.action_id == final.action_id for c in started if c.call.startswith("mock_")))
    submits = sum(c.call == "mock_submit" for c in started)
    lookups = sum(c.call == "mock_lookup" for c in started)
    if point.scenario.endswith("before"):
        require(submits == lookups == final.effects == 0)
        require(final.action_status == "cancelled" and final.invocation_status == "failed")
    elif point.scenario == "lookup_unavailable":
        require(submits == final.effects == lookups == 1)
        require(final.action_status == final.invocation_status == "outcome_unknown")
        require(final.recovery_attempts == 1)
    else:
        require(
            submits == final.effects == 1
            and final.action_status == final.invocation_status == "succeeded"
        )
        require(lookups == (0 if point.scenario == "approval" else 1))
    if point.scenario == "approval":
        require(before.approval_status == "approved" and before.job_status == "queued")
        require(before.invocation_id is None and final.invocation_id is not None)


class Experiment:
    def __init__(self, env, profile, point, authorization, started):
        self.env, self.profile, self.point = env, profile, point
        self.authorization, self.started = authorization, started
        self.deadline = started + profile.max_seconds - profile.cleanup_seconds
        self.before = self.final = self.manifest = None
        self.injection = self.restart = self.converged = None
        self.health = []
        self.last_health = 0.0
        self.observation = 0.0
        self.pool_remaining = None
        self.stage = "environment"

    async def command(self, kind):
        return await asyncio.to_thread(self.env.capacity_command, kind)

    async def guard(self):
        if monotonic() >= self.deadline:
            raise FaultStop("deadline")
        if monotonic() - self.last_health < 1:
            return
        start = monotonic()
        try:
            packet = await self.command("health")
            meminfo = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            async with self.sessions() as session:
                connections = await session.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                        "AND pid <> pg_backend_pid()"
                    )
                )
                queue = await session.scalar(
                    select(func.count())
                    .select_from(RunJob)
                    .where(RunJob.status.in_(["queued", "leased"]))
                )
            h = Health(
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
            h = Health(at=monotonic())
        self.observation += monotonic() - start
        self.health.append(h)
        self.last_health = monotonic()
        if any(v is None for k, v in h.model_dump().items() if k != "at"):
            raise FaultStop("guard_failed")
        if h.queue > 1:
            raise FaultStop("queue_limit")
        if (
            h.memory >= self.profile.memory_limit
            or h.tmpfs >= self.profile.tmpfs_limit
            or h.rss >= self.profile.rss_limit
            or h.available < self.profile.minimum_free
            or h.disk_free < self.profile.minimum_free
            or h.connections > self.profile.connection_limit
        ):
            raise FaultStop("resource_limit")

    async def observe(self):
        started = monotonic()
        try:
            return await snapshot(self.sessions, self.tenant, self.run_id)
        finally:
            self.observation += monotonic() - started

    async def run(self):
        self.stage = "setup"
        engine = create_database_engine(SecretStr(self.env.database_url))
        self.sessions = create_session_factory(engine)
        collector = CapacityCollector("driver")
        detach = collector.attach_pool(engine)
        try:
            async with (
                asyncio.timeout(max(0.01, self.deadline - monotonic())),
                httpx.AsyncClient(
                    base_url=self.env.api_origin,
                    timeout=5,
                    trust_env=False,
                    follow_redirects=False,
                ) as client,
            ):
                http = HTTP(client)
                self.http = http
                async with self.sessions() as session:
                    version = await session.scalar(text("SHOW server_version"))
                instant = self.authorization == "ci_instant"
                self.manifest = Manifest(
                    experiment_id=uuid4(),
                    source_sha=source_sha(),
                    lock_digest=lock_digest(ROOT),
                    profile=self.profile,
                    profile_digest=digest(self.profile),
                    point=self.point,
                    authorization=self.authorization,
                    database_version=version.split()[0],
                    lease_seconds=3.0 if instant else 30.0,
                    heartbeat_seconds=0.5 if instant else 10.0,
                    call_profile_digest=digest(call_profile()),
                )
                publish(self.env.output_dir, "fault-manifest.json", self.manifest)
                await self.guard()
                me = await http.request("GET", "/api/v1/me")
                require(len(me["workspaces"]) == 1 and me["workspaces"][0]["role"] == "admin")
                self.tenant = TenantContext(
                    UUID(me["workspaces"][0]["workspace_id"]),
                    UUID(me["user_id"]),
                    WorkspaceRole.ADMIN,
                )
                doc = (
                    await ingest_resume(
                        self.sessions, self.tenant, call_profile(), self.env.output_dir
                    )
                    if self.point.mode == "application"
                    else None
                )
                payload = {"mode": self.point.mode, "query": QUERY}
                if doc is not None:
                    payload["resume_document_id"] = str(doc)
                root = f"/api/v1/workspaces/{self.tenant.workspace_id}/runs"
                key = str(uuid4())
                created = await http.request(
                    "POST", root, expected=202, headers={"Idempotency-Key": key}, json=payload
                )
                require(
                    created
                    == await http.request(
                        "POST", root, expected=202, headers={"Idempotency-Key": key}, json=payload
                    )
                )
                self.run_id = UUID(created["run_id"])
                self.path = root + "/" + str(self.run_id)
                await self.command("start_worker")
                self.stage = "barrier"
                barrier_deadline = min(self.deadline, monotonic() + self.profile.barrier_seconds)
                approved = False
                while True:
                    await self.guard()
                    s = await self.observe()
                    if (
                        self.point.mode == "application"
                        and self.point.scenario != "approval"
                        and s.run_status == "waiting_approval"
                        and s.job_status == "done"
                        and not approved
                    ):
                        approved = await approve_synthetic(
                            http, self.sessions, self.tenant, self.run_id
                        )
                    if (await self.command("fault_poll"))["barrier"]:
                        break
                    if monotonic() >= barrier_deadline:
                        raise FaultStop("barrier_timeout")
                    await asyncio.sleep(0.1)
                if self.point.scenario == "approval":
                    require(await approve_synthetic(http, self.sessions, self.tenant, self.run_id))
                self.before = await self.observe()
                self.stage = "injection"
                self.injection = monotonic()
                if self.point.scenario in CRASH:
                    await self.command("fault_kill")
                    # Observe the durable old lease before any successor can claim it.
                    killed = await self.observe()
                    require(
                        killed.job_id == self.before.job_id
                        and killed.job_attempt == self.before.job_attempt
                    )
                    self.restart = monotonic()
                    await self.command("start_worker")
                else:
                    if self.point.scenario.startswith("cancel_"):
                        await http.request("POST", self.path + "/cancel")
                    elif self.point.scenario.startswith("revoke_"):
                        async with self.sessions() as session, session.begin():
                            result = await session.execute(
                                update(WorkspaceMembership)
                                .where(
                                    WorkspaceMembership.workspace_id == self.tenant.workspace_id,
                                    WorkspaceMembership.user_id == self.tenant.actor_user_id,
                                    WorkspaceMembership.revoked_at.is_(None),
                                )
                                .values(revoked_at=datetime.now(UTC))
                            )
                            require(result.rowcount == 1)
                    await self.command("fault_release")
                self.stage = "recovery"
                while True:
                    await self.guard()
                    s = await self.observe()
                    if (
                        self.point.scenario in CRASH[:3]
                        and self.before.lease_expires_at is not None
                    ):
                        # Real UTC, without editing lease timestamps or advancing a fake clock.
                        if time() < self.before.lease_expires_at:
                            require(s.job_attempt == self.before.job_attempt)
                    if s.run_status in {"completed", "failed", "cancelled"} and s.job_status in {
                        "done",
                        "dead",
                    }:
                        self.final = s
                        self.converged = monotonic()
                        break
                    await asyncio.sleep(0.2)
                life, calls = read_evidence(self.env.output_dir)
                self.stage = "verification"
                check_facts(self.point, self.before, self.final, life, calls)
                # Terminal state must remain stable; no subsequent POST or business rewrite.
                await asyncio.sleep(0.6)
                require(await self.observe() == self.final)
                require(read_evidence(self.env.output_dir)[1] == calls)
                self.stage = "complete"
        finally:
            self.pool_remaining = collector.checked_out
            detach()
            await asyncio.wait_for(engine.dispose(), 5)


def classify(error):
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        return "cancelled"
    if isinstance(error, FaultStop):
        return error.category
    if isinstance(error, TimeoutError):
        return "deadline"
    if isinstance(error, EnvironmentError):
        return (
            error.category
            if error.category in {"cleanup_failed", "report_failed", "ownership_mismatch"}
            else "environment_failed"
        )
    return "correctness_failed"


def run_point(output, *, point, profile, authorization):
    instant = authorization == "ci_instant"
    if (
        authorization not in {AUTHORIZATION, "ci_instant"}
        or (profile.name == "faults-instant-ci-v1") != instant
        or point.repetition > profile.repeats
    ):
        raise ValueError("invalid_authorization")
    started = monotonic()
    env = IsolatedEnvironment(
        EnvironmentProfile(PROFILE),
        output,
        call_profile=call_profile().model_dump(mode="json"),
        metrics=True,
        capacity=True,
        fault_config={"scenario": point.scenario, "instant": instant},
    )
    experiment = Experiment(env, profile, point, authorization, started)
    stop = "complete"
    correct = False
    life, calls = (), ()
    try:
        env.start()
        asyncio.run(experiment.run())
        correct = True
    except BaseException as error:
        if not isinstance(error, (Exception, KeyboardInterrupt, asyncio.CancelledError)):
            raise
        stop = classify(error)
    finally:
        cleanup = env.close()
    if not cleanup.get("resources_released") or cleanup.get("category"):
        stop = "cleanup_failed" if not cleanup.get("resources_released") else "environment_failed"
    if env.output_created:
        try:
            life, calls = read_evidence(output)
            for role in ("api", "supervisor"):
                packet = safe_read(output / f"metrics-{role}.json", CapacityPacket)
                require(not packet.write_failed and packet.pool_remaining == 0)
        except Exception:
            if stop == "complete":
                stop = "report_failed"
    e = experiment
    final = e.final
    result = Result(
        point=point,
        manifest_digest=digest(e.manifest) if e.manifest else None,
        status="PASS" if stop == "complete" else "IN_PROGRESS",
        stop=stop,
        stage=e.stage,
        complete=correct,
        expected_state_correct=correct,
        automatic_business_complete=final is not None and final.run_status == "completed",
        manual_verification_required=final is not None and final.action_status == "outcome_unknown",
        resources_released=cleanup.get("resources_released", False),
        pool_remaining=e.pool_remaining,
        before=e.before,
        final=final,
        lifecycle=life,
        calls=calls,
        health=tuple(e.health),
        injection_at=e.injection,
        restart_at=e.restart,
        converged_at=e.converged,
        fault_to_convergence=e.converged - e.injection
        if e.converged is not None and e.injection is not None
        else None,
        restart_to_convergence=e.converged - e.restart
        if e.converged is not None and e.restart is not None
        else None,
        elapsed_seconds=monotonic() - started,
        observation_seconds=e.observation,
    )
    if env.output_created:
        publish(output, "fault-result.json", result)
    return result


def run_suite(output, *, authorization, selected_profile):
    if authorization != AUTHORIZATION or selected_profile != "faults-e59-v1":
        raise ValueError("invalid_authorization")
    profile = load_profile(selected_profile)
    directory = create_output_directory(output)
    started = monotonic()
    results = []
    stop = None
    for ordinal, point in enumerate(points(profile)):
        if stop is None and monotonic() - started + profile.max_seconds > profile.suite_seconds:
            stop = "suite_deadline"
        if stop:
            result = Result(point=point, status="NOT_STARTED", stop=stop)
        else:
            result = run_point(
                directory / f"point-{ordinal:02}",
                point=point,
                profile=profile,
                authorization=authorization,
            )
        results.append(result)
        if result.status != "PASS":
            stop = "previous_stop"
    suite = Suite(profile=profile, source_sha=source_sha(), results=tuple(results))
    publish(directory, "fault-suite.json", suite)
    return suite
