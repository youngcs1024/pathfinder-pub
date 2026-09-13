"""Explicit E5.8 driver. No arbitrary DSN, production worker or provider is accepted."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from time import monotonic
from uuid import uuid4

from pydantic import SecretStr
from sqlalchemy import select, text

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import LLMInvocation
from app.db.session import create_database_engine, create_session_factory
from app.llm.factory import LLMFactory, LLMRetryPolicy
from app.llm.fake import FakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.retrieval.documents import DocumentRetrievalService
from tests.performance.capacity import source_sha
from tests.performance.capacity_contracts import Health, lock_digest, safe_read
from tests.performance.capacity_metrics import CapacityCollector, CapacityPacket
from tests.performance.contracts import digest
from tests.performance.environment import (
    POSTGRES_IMAGE,
    PROFILE,
    ROOT,
    EnvironmentError,
    EnvironmentProfile,
    IsolatedEnvironment,
    create_output_directory,
)
from tests.performance.retrieval_contracts import (
    AUTHORIZATION,
    Manifest,
    Point,
    Profile,
    Result,
    Suite,
    load_profile,
    points,
)
from tests.performance.retrieval_data import QUERIES, QueryEmbedding, create_dataset
from tests.performance.retrieval_metrics import (
    ObservedRepository,
    QueryObserver,
    check_layout,
    explain,
    hit_digest,
    layout,
    sample_schedule,
    summaries,
)
from tests.performance.workload import publish


class StopMeasurement(Exception):
    def __init__(self, category):
        self.category = category
        super().__init__(category)


def require(condition):
    if not condition:
        raise StopMeasurement("correctness_failed")


def classify(error):
    if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
        return "cancelled"
    if isinstance(error, StopMeasurement):
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


class Experiment:
    def __init__(self, environment, manifest, deadline):
        self.env, self.manifest, self.deadline = environment, manifest, deadline
        self.profile = manifest.profile
        self.dataset = None
        self.samples = sample_schedule(self.profile)
        self.plans = []
        self.before_layout = self.final_layout = None
        self.health = []
        self.last_health = 0.0
        self.observer_seconds = 0.0
        self.attempts = None
        self.adapter = QueryEmbedding(self.profile.max_attempts)
        self.metrics = CapacityCollector("driver", self.env.output_dir)
        self.observer = QueryObserver()
        self.complete = False
        self.pool_remaining = None

    async def observe(self, operation, *args):
        started = monotonic()
        try:
            return await operation(*args)
        finally:
            self.observer_seconds += monotonic() - started

    async def guard(self, force=False):
        if monotonic() >= self.deadline:
            raise StopMeasurement("deadline")
        if not force and monotonic() - self.last_health < 1:
            return
        started = monotonic()
        try:
            packet = await asyncio.to_thread(self.env.capacity_command, "health")
            meminfo = dict(
                line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
            )
            async with self.sessions() as session, session.begin():
                connections = await session.scalar(
                    text(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() "
                        "AND pid <> pg_backend_pid()"
                    )
                )
                queue = await session.scalar(
                    text("SELECT count(*) FROM run_jobs WHERE status IN ('queued','leased')")
                )
            health = Health(
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
            health = Health(at=monotonic())
        self.observer_seconds += monotonic() - started
        self.last_health = monotonic()
        self.health.append(health)
        if reason := health.stop(self.profile):
            raise StopMeasurement(reason)

    async def measure(self, service):
        captured = {}
        expected_hits = {}
        try:
            for i, planned in enumerate(self.samples):
                await self.guard()
                self.observer.repository_seconds = None
                self.observer.sql_seconds = None
                self.observer.sql_count = 0
                started = monotonic()
                outcome, count, fingerprint = "failed", None, None
                try:
                    hits = await service.retrieve(
                        tenant=self.dataset.tenants[0],
                        query=QUERIES[planned.query],
                        allowed_document_ids=(self.dataset.documents[0],),
                    )
                    elapsed = monotonic() - started
                    fingerprint = hit_digest(hits, self.dataset)
                    count = len(hits)
                    require(self.observer.sql_count == 1 and self.observer.captured is not None)
                    require(self.observer.sql_seconds is not None)
                    if planned.query in expected_hits:
                        require(expected_hits[planned.query] == fingerprint)
                    expected_hits[planned.query] = fingerprint
                    captured[planned.query] = self.observer.captured
                    outcome = "succeeded"
                except asyncio.CancelledError:
                    outcome = "cancelled"
                    raise
                except TimeoutError:
                    outcome = "timeout"
                    raise
                finally:
                    self.samples[i] = planned.model_copy(
                        update=dict(
                            outcome=outcome,
                            total_seconds=(
                                elapsed if outcome == "succeeded" else monotonic() - started
                            ),
                            repository_seconds=self.observer.repository_seconds,
                            sql_seconds=self.observer.sql_seconds,
                            sql_count=self.observer.sql_count,
                            returned=count,
                            result_digest=fingerprint,
                        )
                    )
            for query in range(self.profile.query_count):
                await self.guard()
                self.plans.append(await self.observe(explain, self.engine, query, captured[query]))
        finally:
            captured.clear()
            self.observer.captured = None

    async def facts(self):
        current = await layout(self.sessions, self.dataset)
        async with self.sessions() as session:
            rows = (
                await session.execute(
                    select(
                        LLMInvocation.workspace_id,
                        LLMInvocation.actor_user_id,
                        LLMInvocation.provider,
                        LLMInvocation.invocation_kind,
                        LLMInvocation.status,
                    )
                )
            ).all()
        self.attempts = len(rows)
        tenant = self.dataset.tenants[0]
        require(
            all(
                tuple(row)
                == (tenant.workspace_id, tenant.actor_user_id, "fake", "embedding", "succeeded")
                for row in rows
            )
        )
        return current

    async def drive(self):
        self.engine = create_database_engine(SecretStr(self.env.database_url))
        self.sessions = create_session_factory(self.engine)
        detach = self.metrics.attach_pool(self.engine)
        try:
            async with asyncio.timeout(max(0, self.deadline - monotonic())):
                await self.guard(force=True)
                self.dataset = await create_dataset(
                    self.sessions,
                    self.manifest.point,
                    self.guard,
                    lambda value: setattr(self, "dataset", value),
                )
                await self.guard(force=True)
                # Statistics preparation is outside all latency samples and only touches
                # this already-owned database. Keep the production query planner settings.
                async with self.engine.connect() as connection, connection.begin():
                    await connection.exec_driver_sql("ANALYZE")
                self.before_layout = await self.observe(layout, self.sessions, self.dataset)
                check_layout(self.manifest.point, self.before_layout)
                tenant = self.dataset.tenants[0]
                factory = LLMFactory(
                    recorder=SqlAlchemyInvocationRecorder(self.sessions),
                    chat_adapter=FakeChatModel(),
                    embedding_adapter=self.adapter,
                    retry_policy=LLMRetryPolicy(max_attempts=1),
                )
                service = DocumentRetrievalService(
                    ObservedRepository(SqlAlchemyDocumentRepository(self.sessions), self.observer),
                    factory.create_embedding_model(
                        LLMInvocationContext(
                            tenant.workspace_id,
                            tenant.actor_user_id,
                        )
                    ),
                )
                with self.observer.attach(self.engine):
                    await self.measure(service)
                self.final_layout = await self.observe(self.facts)
                check_layout(self.manifest.point, self.final_layout)
                require(self.attempts == self.adapter.calls == self.profile.max_attempts)
                require(len(self.plans) == self.profile.query_count)
                await self.guard(force=True)
                self.complete = True
        finally:
            # Retain partial DB facts only while the original work deadline still permits
            # the read. Never steal the supervisor's cleanup reserve or replace an error.
            try:
                if self.final_layout is None and self.dataset and self.dataset.dataset_digest:
                    try:
                        async with asyncio.timeout(max(0, min(2, self.deadline - monotonic()))):
                            self.final_layout = await self.observe(self.facts)
                    except Exception:
                        pass
            finally:
                await self.engine.dispose()
                self.pool_remaining = self.metrics.checked_out
                detach()
                self.metrics.finish()


def make_manifest(environment, profile, point, authorization, sha):
    import psycopg

    with psycopg.connect(
        environment.database_url.replace("postgresql+psycopg", "postgresql"),
        connect_timeout=2,
        options="-c statement_timeout=2000",
    ) as connection:
        version = connection.execute("SHOW server_version").fetchone()[0].split()[0]
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname='vector'"
        ).fetchone()[0]
    generator = Path(__file__).with_name("retrieval_data.py").read_bytes()
    return Manifest(
        experiment_id=uuid4(),
        source_sha=sha,
        lock_digest=lock_digest(ROOT),
        profile=profile,
        profile_digest=digest(profile),
        point=point,
        authorization=authorization,
        database_image=POSTGRES_IMAGE,
        database_version=version,
        vector_extension_version=extension,
        generator_digest=hashlib.sha256(generator).hexdigest(),
        query_set_digest=hashlib.sha256(json.dumps(QUERIES).encode()).hexdigest(),
    )


def run_point(output_dir, *, point, profile, authorization):
    profile = Profile.model_validate_json(profile.model_dump_json())
    point = Point.model_validate_json(point.model_dump_json())
    instant = authorization == "ci_instant"
    require(authorization in {"ci_instant", AUTHORIZATION})
    require((profile.name == "retrieval-instant-ci-v1") == instant)
    require((point.kind == "instant") == instant)
    environment = IsolatedEnvironment(
        EnvironmentProfile(PROFILE), output_dir, capacity=True, metrics=True
    )
    sha = source_sha()
    started = monotonic()
    deadline = started + profile.max_seconds - profile.cleanup_seconds
    experiment = manifest = None
    reason, diagnostics = "window_complete", []
    interrupted = None
    try:
        environment.start()
        require("worker" not in environment.identity["process_ids"])
        manifest = make_manifest(environment, profile, point, authorization, sha)
        publish(output_dir, "retrieval-manifest.json", manifest)
        experiment = Experiment(environment, manifest, deadline)
        asyncio.run(experiment.drive())
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
        reason = classify(error)
        if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
            interrupted = error
    finally:
        cleanup = environment.close()
    released = cleanup.get("resources_released") is True and not cleanup.get("category")
    if not released:
        diagnostics.append("cleanup_failed")
    packets_valid = True
    if experiment is not None:
        for role in ("driver", "api", "supervisor"):
            try:
                packet = safe_read(output_dir / f"metrics-{role}.json", CapacityPacket)
                require(packet.role == role and not packet.write_failed and packet.dropped == 0)
                require(packet.pool_remaining == 0)
                expected_pid = (
                    experiment.metrics.process_id
                    if role == "driver"
                    else environment.identity["process_ids"]["api"]
                    if role == "api"
                    else environment._process.process.pid
                )
                require(packet.process_id == expected_pid)
            except Exception:
                packets_valid = False
        if not packets_valid:
            diagnostics.append("metrics_incomplete")
    complete = bool(experiment and experiment.complete and packets_valid)
    status = "PASS" if complete and released and reason == "window_complete" else "IN_PROGRESS"
    result = Result(
        point=point,
        source_sha=sha,
        manifest_digest=digest(manifest) if manifest else None,
        status=status,
        stop=reason,
        diagnostics=tuple(diagnostics),
        resources_released=released,
        samples=tuple(experiment.samples) if experiment else (),
        summaries=summaries(experiment.samples) if experiment else (),
        plans=tuple(experiment.plans) if experiment else (),
        layout=experiment.before_layout if experiment else None,
        final_layout=experiment.final_layout if experiment else None,
        health=tuple(experiment.health) if experiment else (),
        attempts=experiment.attempts if experiment else None,
        adapter_calls=experiment.adapter.calls if experiment else 0,
        observer_seconds=experiment.observer_seconds if experiment else 0.0,
        pool_remaining=experiment.pool_remaining if experiment else None,
        completeness=complete,
        correctness=bool(experiment and experiment.complete),
    )
    if environment.output_created:
        publish(output_dir, "retrieval-result.json", result)
    if interrupted is not None:
        raise interrupted
    return result


def run_suite(directory, *, authorization, selected_profile):
    require(authorization == AUTHORIZATION and selected_profile == "retrieval-e58-v1")
    profile = load_profile(selected_profile)
    create_output_directory(directory)
    started, sha = monotonic(), source_sha()
    results = []
    aborted = False
    interrupted = None
    for index, point in enumerate(points()):
        reason = (
            "previous_stop"
            if aborted
            else "suite_deadline"
            if monotonic() - started + profile.max_seconds > profile.suite_seconds
            else None
        )
        if reason:
            result = Result(point=point, source_sha=sha, status="NOT_RUN", stop=reason)
        else:
            point_directory = directory / f"point-{index:02d}"
            try:
                result = run_point(
                    point_directory,
                    point=point,
                    profile=profile,
                    authorization=authorization,
                )
            except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
                try:
                    result = safe_read(point_directory / "retrieval-result.json", Result)
                    require(result.source_sha == sha and result.point == point)
                    require(result.status != "PASS")
                except Exception:
                    result = Result(point=point, source_sha=sha, stop=classify(error))
                if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)):
                    interrupted = error
            aborted = result.status != "PASS"
        results.append(result)
        print(f"retrieval/{point.kind}: {result.status} {result.stop}", flush=True)
    suite = Suite(
        source_sha=sha, authorization=authorization, profile=profile, results=tuple(results)
    )
    publish(directory, "retrieval-suite.json", suite)
    if interrupted is not None:
        raise interrupted
    return suite
