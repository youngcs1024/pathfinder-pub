"""Narrow test-process wrappers and read-only owned-environment resource sampling."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from time import monotonic
from unittest.mock import patch

from sqlalchemy import text

from tests.performance.metrics import (
    SEGMENT_LIMIT,
    Collector,
    Resources,
    Value,
    count_sql,
    difference,
    missing,
)


@contextmanager
def instrument(root, collector: Collector):
    """Only installed inside the owned API/worker child; restores every patched reference."""
    from app.db.documents import SqlAlchemyDocumentRepository
    from app.db.events import SqlAlchemyRunEventReader
    from app.retrieval.documents import DocumentRetrievalService

    with ExitStack() as stack:
        original_engine = root.create_database_engine

        def engine(*args, **kwargs):
            value = original_engine(*args, **kwargs)
            stack.enter_context(count_sql(value, collector))
            return value

        stack.enter_context(patch.object(root, "create_database_engine", engine))

        def wrap(cls, name, metric):
            original = getattr(cls, name)

            async def observed(self, *args, **kwargs):
                fields = {"run_id": kwargs["run_id"]} if "run_id" in kwargs else {}
                with collector.observe(metric, **fields):
                    return await original(self, *args, **kwargs)

            stack.enter_context(patch.object(cls, name, observed))

        if collector.role == "api":
            wrap(SqlAlchemyRunEventReader, "read_after", "read_after")
        elif collector.role == "worker":
            wrap(SqlAlchemyDocumentRepository, "search", "repository")
            wrap(DocumentRetrievalService, "retrieve", "retrieval")
            original_store = root.SqlAlchemyWorkerJobStore
            original_runner = root.WorkerRunner
            active = None

            class Store(original_store):
                async def claim_due_job(self, **kwargs):
                    nonlocal active
                    job = await super().claim_due_job(**kwargs)
                    if job is not None:
                        collector.claims += 1
                        if collector.claims > SEGMENT_LIMIT:
                            collector.dropped += 1
                            active = None
                        else:
                            active = collector.begin(
                                "segment",
                                run_id=job.run_id,
                                job_id=job.job_id,
                                approval_request_id=job.resume_approval_request_id,
                                attempt=job.attempt,
                                claim_ordinal=collector.claims,
                            )
                            collector.segment_record(active, "started")
                    return job

                async def observe_requeue(self, job):
                    from sqlalchemy import select

                    from app.db.models import RunJob

                    token = collector.begin("requeue", run_id=job.run_id)
                    outcome = "failed"
                    try:
                        async with asyncio.timeout(1.0), self._session_factory() as session:
                            status = await session.scalar(
                                select(RunJob.status).where(
                                    RunJob.id == job.job_id,
                                    RunJob.workspace_id == job.workspace_id,
                                    RunJob.run_id == job.run_id,
                                )
                            )
                        outcome = "succeeded" if status == "queued" else "failed"
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                        raise
                    except Exception:
                        pass  # A failed observer is never evidence of requeue or business failure.
                    finally:
                        collector.end(token, outcome)
                        if token is not None:
                            sample = collector.samples[token - 1]
                            collector.observer_seconds += (
                                difference(sample.started, sample.finished).value or 0.0
                            )

                async def requeue(self, **kwargs):
                    result = await super().requeue(**kwargs)
                    if result:
                        await self.observe_requeue(kwargs["job"])
                    return result

                async def release(self, **kwargs):
                    result = await super().release(**kwargs)
                    if result:
                        await self.observe_requeue(kwargs["job"])
                    return result

            class Runner(original_runner):
                async def run_once(self, stop_requested):
                    nonlocal active
                    active = None
                    outcome = "failed"
                    try:
                        result = await super().run_once(stop_requested)
                        # This means runner returned, NOT that the business Run succeeded.
                        outcome = "succeeded"
                        return result
                    except asyncio.CancelledError:
                        outcome = "cancelled"
                        raise
                    finally:
                        collector.end(active, outcome)
                        collector.segment_record(active, "finished")
                        active = None

            stack.enter_context(patch.object(root, "SqlAlchemyWorkerJobStore", Store))
            stack.enter_context(patch.object(root, "WorkerRunner", Runner))
        try:
            yield
        finally:
            collector.finish()


# Aggregate only necessary state fields; never read query text, client addresses or parameters.
DB_STATE_SQL = """
SELECT count(*) AS connections,
       count(*) FILTER (WHERE state = 'active') AS active,
       count(*) FILTER (WHERE wait_event_type IS NOT NULL) AS waiting
FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()
"""
DB_SIZE_SQL = """
SELECT coalesce(sum(pg_table_size(c.oid)), 0) AS table_bytes,
       coalesce(sum(pg_indexes_size(c.oid)), 0) AS index_bytes
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND n.nspname IN ('public', 'pathfinder_checkpoint')
"""


def unavailable_resources(kind, reason="unavailable"):
    fields = (
        ("cpu_percent", "memory_bytes")
        if kind == "container"
        else ("connections", "active", "waiting", "table_bytes", "index_bytes", "events")
    )
    return Resources(**{field: missing(reason) for field in fields})


async def sample_database(sessions, collector: Collector):
    start = monotonic()
    token = collector.begin("resource_db")
    resources = unavailable_resources("db")
    outcome = "failed"
    try:
        async with asyncio.timeout(1.0), sessions() as session, session.begin():
            await session.execute(text("SET TRANSACTION READ ONLY"))
            states = (await session.execute(text(DB_STATE_SQL))).mappings().one()
            sizes = (await session.execute(text(DB_SIZE_SQL))).mappings().one()
            events = await session.scalar(text("SELECT count(*) FROM run_events"))
            resources = Resources(
                **{
                    key: Value(value=float(value))
                    for key, value in {**states, **sizes, "events": events}.items()
                }
            )
            outcome = "succeeded"
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except TimeoutError:
        outcome = "timeout"
    except Exception as error:
        original = getattr(error, "orig", error)
        if getattr(original, "sqlstate", None) == "42501":
            resources = unavailable_resources("db", "permission_denied")
    finally:
        collector.end(token, outcome, resources=resources)
        collector.observer_seconds += monotonic() - start


async def database_sampler(sessions, collector, stop: asyncio.Event):
    # No catch-up burst. Every sample has a 1s bound; stop/outer cancellation is propagated.
    for _ in range(40):
        await sample_database(sessions, collector)
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
            return
        except TimeoutError:
            pass
    collector.dropped += 1


def container_resources(current, previous):
    memory = current.get("memory_stats", {}).get("usage")
    memory_value = Value(value=float(memory)) if type(memory) is int and memory >= 0 else missing()
    cpu = missing()
    if previous is not None:
        try:
            cpu_delta = difference(
                float(previous["cpu_stats"]["cpu_usage"]["total_usage"]),
                float(current["cpu_stats"]["cpu_usage"]["total_usage"]),
            )
            system_delta = difference(
                float(previous["cpu_stats"]["system_cpu_usage"]),
                float(current["cpu_stats"]["system_cpu_usage"]),
            )
            cpus = current["cpu_stats"]["online_cpus"]
            if cpu_delta.value is None or system_delta.value is None:
                cpu = missing("counter_reset")
            elif system_delta.value and type(cpus) is int and cpus > 0:
                cpu = Value(value=100 * cpu_delta.value / system_delta.value * cpus)
        except (KeyError, TypeError, ValueError):
            cpu = missing()
    return Resources(cpu_percent=cpu, memory_bytes=memory_value)


def sample_container(supervisor):
    collector = supervisor.metrics
    token = collector.begin("resource_container")
    start = monotonic()
    resources = unavailable_resources("container")
    outcome = "failed"
    try:
        from tests.performance._supervisor import verify_container

        wrapped = supervisor.container.get_wrapped_container()
        verify_container(
            wrapped,
            owner=supervisor.owner,
            name=supervisor.name,
            identifier=supervisor.container_id,
        )
        # Docker client timeout is 5s, bounded by the supervisor lifetime alarm as well.
        current = wrapped.stats(stream=False, one_shot=True)
        resources = container_resources(current, supervisor.previous_stats)
        supervisor.previous_stats = current
        outcome = "succeeded"
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) in {401, 403}:
            resources = unavailable_resources("container", "permission_denied")
    finally:
        collector.end(token, outcome, resources=resources)
        collector.observer_seconds += monotonic() - start
