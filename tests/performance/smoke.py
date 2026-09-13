"""Bounded single-run driver over the real owned API/worker/graph/DB/SSE chain."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

import httpx
from pydantic import Field, SecretStr
from sqlalchemy import func, select, text

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
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
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory
from app.llm.invocations import LLMInvocationContext
from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.documents import DocumentIngestionService
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.performance.adapters import QUERY, RESUME, Calls, Chat, Embedding
from tests.performance.environment import (
    PROFILE,
    ROOT,
    EnvironmentError,
    EnvironmentProfile,
    IsolatedEnvironment,
)
from tests.performance.metrics import Collector, DecisionFact, RunFact, build_report
from tests.performance.metrics_runtime import database_sampler, sample_database
from tests.performance.workload import (
    KINDS,
    CallProfile,
    CallRecord,
    Contract,
    parse_profile,
    profile,
    publish,
)

MAX_SECONDS = 40
MAX_REQUESTS = 128
MAX_SSE_BYTES = 2 * 1024 * 1024
type SmokeCategory = Literal[
    "http_failed",
    "request_limit",
    "deadline",
    "cancelled",
    "business_failed",
    "evidence_mismatch",
    "cleanup_failed",
    "report_failed",
    "environment_failed",
]


class SmokeFailure(Exception):
    def __init__(self, category: SmokeCategory):
        self.category = category
        super().__init__(category)


def require(condition: bool, category: SmokeCategory = "evidence_mismatch") -> None:
    if not condition:
        raise SmokeFailure(category)


class SmokeRecord(Contract):
    schema_version: Literal[1] = 1
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    mode: Literal["research", "application"]
    profile: Literal["instant-v1", "delayed-v1"]
    approval_mode: Literal["none", "synthetic_driver"]
    status: Literal["IN_PROGRESS", "PASS"] = "IN_PROGRESS"
    category: SmokeCategory | None = None
    run_id: UUID | None = None
    max_requests: Literal[128] = MAX_REQUESTS
    max_seconds: Literal[40] = MAX_SECONDS
    max_calls: Literal[64] = 64
    http_requests: int = Field(default=0, ge=0, le=MAX_REQUESTS)
    elapsed_seconds: float = Field(default=0.0, ge=0)
    event_count: int | None = Field(default=None, ge=0)
    model_attempts: int | None = Field(default=None, ge=0, le=64)
    tool_invocations: int | None = Field(default=None, ge=0)
    mock_effects: int | None = Field(default=None, ge=0)
    unfinished_calls: int = Field(default=0, ge=0, le=64)
    call_counts: dict[Literal["chat", "embedding", "search", "mock_submit", "mock_lookup"], int] = (
        Field(default_factory=dict)
    )
    resources_released: bool = False
    failure_stage: Literal["environment", "drive", "cleanup", "metrics", "report"] | None = None
    diagnostic_errors: tuple[
        Literal["cleanup_failed", "report_failed", "metrics_incomplete"], ...
    ] = ()
    missing_roles: tuple[Literal["driver", "api", "worker", "supervisor"], ...] = ()
    metrics_reasons: tuple[
        Literal[
            "missing_role",
            "missing_sample",
            "missing_segment",
            "missing_fact",
            "write_failed",
            "sample_limit",
            "unfinished",
            "clock_invalid",
        ],
        ...,
    ] = ()
    metrics_status: Literal["PASS", "IN_PROGRESS", "NOT_RUN"] = "NOT_RUN"


class HTTP:
    def __init__(self, client: httpx.AsyncClient, metrics: Collector | None = None):
        self.client = client
        self.count = 0
        self.metrics = metrics

    def reserve(self):
        require(self.count < MAX_REQUESTS, "request_limit")
        self.count += 1

    async def request(self, method, path, *, expected=200, **kwargs):
        self.reserve()
        token = self.metrics.begin("http", expected_status=expected) if self.metrics else None
        outcome = "failed"
        try:
            response = await self.client.request(method, path, **kwargs)
            # HTTPX request reads the complete body. Stop latency before JSON decoding.
            if self.metrics:
                flag = response.headers.get("Idempotency-Replayed")
                self.metrics.end(
                    token,
                    "succeeded" if response.status_code == expected else "failed",
                    http_status=response.status_code,
                    replayed={"true": True, "false": False}.get(flag),
                )
            require(response.status_code == expected, "http_failed")
            data = response.json()
            if self.metrics and expected == 202:
                self.metrics.update(token, run_id=UUID(data["run_id"]))
            outcome = "succeeded"
            return data
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except httpx.TimeoutException:
            outcome = "timeout"
            raise SmokeFailure("http_failed") from None
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise SmokeFailure("http_failed") from None
        finally:
            if self.metrics and token is not None:
                sample = self.metrics.samples[token - 1]
                if sample.finished is None:
                    self.metrics.end(token, outcome)
                elif outcome != "succeeded":
                    self.metrics.update(token, outcome=outcome)

    async def events(self, path, *, cursor=0):
        self.reserve()
        token = self.metrics.begin("sse_connection", cursor=cursor) if self.metrics else None
        pending = bytearray()
        frames = []
        received_bytes = 0
        outcome = "failed"
        try:
            async with self.client.stream(
                "GET", path, headers={"Last-Event-ID": str(cursor)}
            ) as response:
                if self.metrics:
                    self.metrics.update(token, http_status=response.status_code)
                require(response.status_code == 200, "http_failed")
                async for chunk in response.aiter_bytes():
                    received_bytes += len(chunk)
                    require(received_bytes <= MAX_SSE_BYTES)
                    pending.extend(chunk)
                    while b"\n\n" in pending:
                        raw, _, remainder = pending.partition(b"\n\n")
                        pending = bytearray(remainder)
                        parsed = parse_events(raw.decode())
                        for frame in parsed:
                            if self.metrics:
                                data = frame["data"]
                                received = self.metrics.begin(
                                    "sse_event",
                                    run_id=UUID(data["run_id"]),
                                    seq=frame["id"],
                                    cursor=cursor,
                                    event_recorded_at=datetime.fromisoformat(data["occurred_at"]),
                                )
                                self.metrics.end(received)
                        frames.extend(parsed)
            require(not pending.strip())
            outcome = "succeeded"
            return frames
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except httpx.TimeoutException:
            outcome = "timeout"
            raise SmokeFailure("http_failed") from None
        except (httpx.HTTPError, ValueError, UnicodeError, KeyError, TypeError):
            raise SmokeFailure("http_failed") from None
        finally:
            if self.metrics:
                self.metrics.end(token, outcome)


def parse_events(body):
    frames = []
    for raw in body.split("\n\n"):
        lines = [line for line in raw.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        require(
            len(lines) == 3
            and lines[0].startswith("id: ")
            and lines[1].startswith("event: ")
            and lines[2].startswith("data: ")
        )
        frames.append(
            {"id": int(lines[0][4:]), "event": lines[1][7:], "data": json.loads(lines[2][6:])}
        )
    return frames


async def ingest_resume(sessions, tenant, policy, directory):
    calls = Calls(policy, directory=directory, process="ingest")
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=Chat(calls),
        embedding_adapter=Embedding(calls),
    )
    batch = normalize_and_chunk_batch(
        ValidatedIngestionBatch(
            (
                ValidatedIngestionSource(
                    source_name="e53-synthetic-resume.md",
                    source_type="markdown",
                    title="Synthetic Resume",
                    raw_text=RESUME,
                    character_count=len(RESUME),
                ),
            )
        )
    )
    documents = await DocumentIngestionService(
        SqlAlchemyDocumentRepository(sessions),
        factory.create_embedding_model(
            LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
        ),
    ).ingest(tenant=tenant, batch=batch)
    require(len(documents) == 1)
    return documents[0]


async def approve_synthetic(http, sessions, tenant, run_id):
    # Only the one driver-created Run in its provisioned personal workspace is eligible.
    async with sessions() as session:
        run = await session.get(Run, run_id)
        require(
            run is not None
            and run.workspace_id == tenant.workspace_id
            and run.created_by_user_id == tenant.actor_user_id
            and run.mode == "application"
            and run.status == "waiting_approval"
        )
        actions = list(
            await session.scalars(
                select(ActionIntent).where(
                    ActionIntent.workspace_id == tenant.workspace_id, ActionIntent.run_id == run_id
                )
            )
        )
        submissions = list(
            await session.scalars(
                select(MockSubmission).where(
                    MockSubmission.workspace_id == tenant.workspace_id,
                    MockSubmission.run_id == run_id,
                )
            )
        )
        jobs = list(
            await session.scalars(
                select(RunJob).where(
                    RunJob.workspace_id == tenant.workspace_id, RunJob.run_id == run_id
                )
            )
        )
    require(len(actions) == 1 and not submissions and len(jobs) == 1)
    if jobs[0].status == "leased":
        return False  # The graph paused; the real runner still owns its short finalization.
    require(jobs[0].status == "done")
    action = actions[0]
    require(
        action.status == "proposed" and action.originating_actor_user_id == tenant.actor_user_id
    )
    path = f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{action.id}"
    review = await http.request("GET", path)
    request = review["approval_request"]
    require(
        review["action_intent_id"] == str(action.id)
        and review["tool_name"] == "submit_mock_application"
        and review["effect"] == "irreversible"
        and request["status"] == "pending"
    )
    for field in (
        "args_digest",
        "target_digest",
        "approval_binding_digest",
        "approval_binding_version",
    ):
        require(review[field] == request[field] == getattr(action, field))
    decision = await http.request(
        "POST",
        path + "/decision",
        json={
            "decision": "approve",
            "expected_version": request["version"],
            "reason": "E5.3 synthetic fixture driver",
        },
    )
    require(
        decision["action_intent_id"] == str(action.id)
        and decision["approval_request_id"] == request["request_id"]
        and decision["decision"] == "approve"
    )
    return True


def read_calls(directory):
    # Fixed bounded names, no recursive scanning or arbitrary output paths.
    records = []
    for process, limit in (("worker", 63), ("ingest", 1)):
        for sequence in range(1, limit + 1):
            start = directory / f"calls-{process}-{sequence:03}-started.json"
            finish = directory / f"calls-{process}-{sequence:03}-finished.json"
            if not start.exists():
                require(not finish.exists())
                continue
            require(finish.is_file() and not start.is_symlink() and not finish.is_symlink())
            beginning = CallRecord.model_validate_json(start.read_bytes())
            ending = CallRecord.model_validate_json(finish.read_bytes())
            require(
                beginning.phase == "started"
                and ending.phase == "finished"
                and beginning.process == ending.process == process
                and beginning.sequence == ending.sequence == sequence
            )
            require(
                beginning.model_dump(exclude={"phase", "outcome", "elapsed_seconds"})
                == ending.model_dump(exclude={"phase", "outcome", "elapsed_seconds"})
            )
            records.append(ending)
    require(len(records) <= 64)
    return records


async def verify_facts(sessions, tenant, run_id, mode, document_id, frames, directory):
    async with sessions() as session:

        async def rows(model):
            return list(
                await session.scalars(
                    select(model).where(
                        model.workspace_id == tenant.workspace_id, model.run_id == run_id
                    )
                )
            )

        run = await session.get(Run, run_id)
        jobs, tools, events, actions, requests, submissions = [
            await rows(model)
            for model in (
                RunJob,
                ToolInvocation,
                RunEvent,
                ActionIntent,
                ApprovalRequest,
                MockSubmission,
            )
        ]
        llm = list(
            await session.scalars(
                select(LLMInvocation).where(LLMInvocation.workspace_id == tenant.workspace_id)
            )
        )
        decisions = list(
            await session.scalars(
                select(ApprovalDecision).where(ApprovalDecision.workspace_id == tenant.workspace_id)
            )
        )
        # Only the dev-only observer reads Checkpoint; product API still reads business facts.
        checkpoints = await session.scalar(
            text(
                "SELECT count(*) FROM pathfinder_checkpoint.checkpoints WHERE thread_id = :thread"
            ),
            {"thread": str(run_id)},
        )
        require(await session.scalar(select(func.count()).select_from(Run)) == 1)
        require(await session.scalar(select(func.count()).select_from(RunJob)) == 1)
    require(
        run is not None
        and run.status == "completed"
        and run.created_by_user_id == tenant.actor_user_id
        and run.resume_document_id == document_id
    )
    require(
        len(jobs) == 1
        and jobs[0].status == "done"
        and jobs[0].owner_token is None
        and jobs[0].lease_expires_at is None
        and checkpoints > 0
    )
    events.sort(key=lambda event: event.seq)
    require(
        [event.seq for event in events] == list(range(1, len(events) + 1))
        and events[-1].type == "run.completed"
    )
    require(
        frames
        == [
            {
                "id": event.seq,
                "event": event.type,
                "data": {
                    "version": event.version,
                    "run_id": str(run_id),
                    "seq": event.seq,
                    "occurred_at": event.recorded_at.isoformat().replace("+00:00", "Z"),
                    "payload": event.payload,
                },
            }
            for event in events
        ]
    )
    calls = read_calls(directory)
    counts = Counter(item.call for item in calls)
    provider_calls = [item for item in calls if item.call in {"chat", "embedding"}]
    require(
        len(provider_calls) == len(llm)
        and {item.invocation_id for item in provider_calls} == {item.id for item in llm}
    )
    require(
        all(
            item.status == "succeeded"
            and item.provider == "fake"
            and item.actor_user_id == tenant.actor_user_id
            for item in llm
        )
    )
    require(
        all(item.outcome == "succeeded" for item in calls)
        and all(item.status == "succeeded" for item in tools)
    )
    tool_counts = Counter(item.tool_name for item in tools)
    require(tool_counts["search_web"] == counts["search"] == 1)
    if mode == "application":
        require(tool_counts["retrieve_documents"] == 1 and counts["embedding"] == 2)
        require(
            len(actions)
            == len(requests)
            == len(decisions)
            == len(submissions)
            == counts["mock_submit"]
            == tool_counts["submit_mock_application"]
            == 1
        )
        require(
            counts["mock_lookup"] == 0
            and actions[0].status == "succeeded"
            and requests[0].status == "consumed"
        )
        require(
            decisions[0].approval_request_id == requests[0].id
            and decisions[0].decision == "approve"
        )
        require(
            submissions[0].action_intent_id == actions[0].id
            and submissions[0].idempotency_key == actions[0].idempotency_key
        )
        for field in (
            "args_digest",
            "target_digest",
            "approval_binding_digest",
            "approval_binding_version",
        ):
            require(getattr(actions[0], field) == getattr(requests[0], field))
        require(
            any(
                event.type == "rag.retrieved" and str(document_id) in event.payload["document_ids"]
                for event in events
            )
        )
    else:
        require(
            not actions
            and not requests
            and not decisions
            and not submissions
            and not counts["embedding"]
            and not counts["mock_submit"]
            and not counts["mock_lookup"]
        )
    return {
        "event_count": len(events),
        "model_attempts": len(llm),
        "tool_invocations": len(tools),
        "mock_effects": len(submissions),
        "call_counts": {kind: counts[kind] for kind in KINDS},
        "_metrics_facts": (
            RunFact(
                run_id=run.id,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                status=run.status,
                approval_mode="synthetic_driver" if mode == "application" else "none",
            ),
        ),
        "_metrics_decisions": tuple(
            DecisionFact(
                run_id=run.id,
                request_id=d.approval_request_id,
                decided_at=d.decided_at,
            )
            for d in decisions
        ),
    }


async def capture_timing_facts(sessions, tenant, run_id, mode, progress):
    async with sessions() as session:
        row = (
            await session.execute(
                select(
                    Run.id,
                    Run.created_at,
                    Run.started_at,
                    Run.finished_at,
                    Run.status,
                ).where(Run.workspace_id == tenant.workspace_id, Run.id == run_id)
            )
        ).one()
        decisions = (
            await session.execute(
                select(
                    ApprovalDecision.approval_request_id,
                    ApprovalDecision.decided_at,
                )
                .join(ApprovalRequest, ApprovalRequest.id == ApprovalDecision.approval_request_id)
                .where(
                    ApprovalDecision.workspace_id == tenant.workspace_id,
                    ApprovalRequest.workspace_id == tenant.workspace_id,
                    ApprovalRequest.run_id == run_id,
                )
            )
        ).all()
    progress["_metrics_facts"] = (
        RunFact(
            run_id=row.id,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            status=row.status,
            approval_mode="synthetic_driver" if mode == "application" else "none",
        ),
    )
    progress["_metrics_decisions"] = tuple(
        DecisionFact(
            run_id=run_id,
            request_id=d.approval_request_id,
            decided_at=d.decided_at,
        )
        for d in decisions
    )


async def drive(environment, policy, mode, progress):
    engine = create_database_engine(SecretStr(environment.database_url))
    sessions = create_session_factory(engine)
    started = monotonic()
    metrics = Collector("driver", environment.output_dir)
    sampler = None
    stop_sampling = asyncio.Event()
    try:
        async with (
            asyncio.timeout(MAX_SECONDS),
            httpx.AsyncClient(
                base_url=environment.api_origin,
                timeout=httpx.Timeout(5.0, read=MAX_SECONDS),
                trust_env=False,
                follow_redirects=False,
            ) as client,
        ):
            http = HTTP(client, metrics)
            await sample_database(sessions, metrics)
            sampler = asyncio.create_task(database_sampler(sessions, metrics, stop_sampling))
            try:
                identity = await http.request("GET", "/api/v1/me")
                require(
                    len(identity["workspaces"]) == 1
                    and identity["workspaces"][0]["role"] == "admin"
                )
                tenant = TenantContext(
                    UUID(identity["workspaces"][0]["workspace_id"]),
                    UUID(identity["user_id"]),
                    WorkspaceRole.ADMIN,
                )
                document_id = (
                    await ingest_resume(sessions, tenant, policy, environment.output_dir)
                    if mode == "application"
                    else None
                )
                payload = {"mode": mode, "query": QUERY}
                if document_id is not None:
                    payload["resume_document_id"] = str(document_id)
                key = str(uuid4())
                created = await http.request(
                    "POST",
                    f"/api/v1/workspaces/{tenant.workspace_id}/runs",
                    expected=202,
                    headers={"Idempotency-Key": key},
                    json=payload,
                )
                run_id = UUID(created["run_id"])
                progress["run_id"] = run_id
                path = f"/api/v1/workspaces/{tenant.workspace_id}/runs/{run_id}"
                # Explicitly exercise E3 replay without creating a second execution.
                replay = await http.request(
                    "POST",
                    f"/api/v1/workspaces/{tenant.workspace_id}/runs",
                    expected=202,
                    headers={"Idempotency-Key": key},
                    json=payload,
                )
                require(replay == created and created["status"] == "queued")
                async with asyncio.TaskGroup() as group:
                    stream = group.create_task(http.events(path + "/events"))
                    approved = False
                    while True:
                        state = await http.request("GET", path)
                        observation_started = monotonic()
                        await capture_timing_facts(sessions, tenant, run_id, mode, progress)
                        metrics.observer_seconds += monotonic() - observation_started
                        if state["status"] == "completed":
                            async with sessions() as session:
                                job_status = await session.scalar(
                                    select(RunJob.status).where(
                                        RunJob.workspace_id == tenant.workspace_id,
                                        RunJob.run_id == run_id,
                                    )
                                )
                            if job_status == "done":
                                break
                        require(state["status"] not in {"failed", "cancelled"}, "business_failed")
                        if state["status"] == "waiting_approval" and not approved:
                            require(mode == "application")
                            approved = await approve_synthetic(http, sessions, tenant, run_id)
                        await asyncio.sleep(0.25)
                frames = stream.result()
                require(bool(frames))
                midpoint = frames[len(frames) // 2]["id"]
                suffix = await http.events(path + "/events", cursor=midpoint)
                require(suffix == [frame for frame in frames if frame["id"] > midpoint])
                require(await http.events(path + "/events", cursor=frames[-1]["id"]) == [])
                progress.update(
                    await verify_facts(
                        sessions, tenant, run_id, mode, document_id, frames, environment.output_dir
                    )
                )
                await sample_database(sessions, metrics)
            finally:
                progress["http_requests"] = http.count
    finally:
        await finish_drive(
            engine, metrics, sampler, stop_sampling, progress, started, primary=sys.exception()
        )


async def finish_drive(engine, metrics, sampler, stop_sampling, progress, started, *, primary):
    """Attempt every bounded finalizer without replacing a primary failure or cancellation."""
    diagnostics = progress.setdefault("_diagnostic_errors", [])
    failure = None
    interrupted = None
    if sampler is not None:
        stop_sampling.set()
        cancellation_count = asyncio.current_task().cancelling()
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError as exc:
            # Cancellation requested by this finalizer is expected; caller cancellation is not.
            if asyncio.current_task().cancelling() > cancellation_count:
                interrupted = exc
        except KeyboardInterrupt as exc:
            interrupted = exc
        except Exception:
            diagnostics.append("metrics_incomplete")
            failure = SmokeFailure("evidence_mismatch")
    try:
        metrics.finish()
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        interrupted = interrupted or exc
        diagnostics.append("metrics_incomplete")
    except Exception:
        diagnostics.append("metrics_incomplete")
        failure = failure or SmokeFailure("evidence_mismatch")
    progress["_metrics_write_failed"] = metrics.write_failed or "metrics_incomplete" in diagnostics
    progress["elapsed_seconds"] = monotonic() - started
    try:
        await asyncio.wait_for(engine.dispose(), timeout=5)
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        interrupted = interrupted or exc
        diagnostics.append("cleanup_failed")
    except Exception:
        diagnostics.append("cleanup_failed")
        failure = failure or SmokeFailure("cleanup_failed")
    if interrupted is not None and not isinstance(
        primary, (KeyboardInterrupt, asyncio.CancelledError)
    ):
        raise interrupted
    if primary is None and failure is not None:
        raise failure


def failure_category(error) -> SmokeCategory:
    if isinstance(error, BaseExceptionGroup):
        return failure_category(error.exceptions[0])
    if isinstance(error, TimeoutError):
        return "deadline"
    if isinstance(error, SmokeFailure):
        return error.category
    if isinstance(error, EnvironmentError):
        return "report_failed" if error.category == "report_failed" else "environment_failed"
    return "business_failed"


def partial_calls(directory):
    counts = Counter()
    unfinished = 0
    for process, limit in (("worker", 63), ("ingest", 1)):
        for sequence in range(1, limit + 1):
            path = directory / f"calls-{process}-{sequence:03}-started.json"
            if path.exists():
                require(path.is_file() and not path.is_symlink())
                record = CallRecord.model_validate_json(path.read_bytes())
                counts[record.call] += 1
                unfinished += not (
                    directory / f"calls-{process}-{sequence:03}-finished.json"
                ).is_file()
    return {"call_counts": dict(counts), "unfinished_calls": unfinished}


def run_smoke(
    output_dir: Path,
    *,
    mode: Literal["research", "application"],
    policy: CallProfile | None = None,
    authorize_delayed: bool = False,
) -> SmokeRecord:
    """Manual delayed entry requires an explicit opt-in; ordinary CI selects only instant."""
    require(mode in {"research", "application"})
    selected = parse_profile((policy or profile()).model_dump(mode="json"))
    require(not selected.faults)  # Fault matrices use adapter ports, not this success-only smoke.
    require(selected.name == "instant-v1" or authorize_delayed is True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    initial = SmokeRecord(
        source_commit=commit,
        mode=mode,
        profile=selected.name,
        approval_mode="synthetic_driver" if mode == "application" else "none",
    )
    environment = IsolatedEnvironment(
        EnvironmentProfile(PROFILE),
        output_dir,
        call_profile=selected.model_dump(mode="json"),
        metrics=True,
    )
    progress = {}
    category = None
    interrupted = None
    stage = "environment"
    diagnostic_errors = []
    try:
        environment.start()
        publish(output_dir, "smoke-started.json", initial)
        stage = "drive"
        asyncio.run(drive(environment, selected, mode, progress))
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        category = "cancelled"
        interrupted = exc
    except Exception as exc:
        category = failure_category(exc)
    diagnostic_errors.extend(progress.pop("_diagnostic_errors", ()))
    try:
        cleanup = environment.close()
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        interrupted = interrupted or exc
        category = "cancelled"
        cleanup = {"resources_released": False}
    except Exception:
        cleanup = {"resources_released": False}
    released = cleanup.get("resources_released") is True
    if not released or cleanup.get("status") != "PASS" or cleanup.get("category"):
        diagnostic_errors.append("cleanup_failed")
        if category is None:
            category, stage = "cleanup_failed", "cleanup"
    if environment.output_created:
        try:
            progress.update(partial_calls(output_dir))
        except Exception:
            diagnostic_errors.append("report_failed")
            if category is None:
                category, stage = "report_failed", "report"
    facts = progress.pop("_metrics_facts", ())
    decisions = progress.pop("_metrics_decisions", ())
    metrics_write_failed = progress.pop("_metrics_write_failed", False)
    if environment.output_created:
        try:
            report = build_report(
                output_dir,
                source_commit=commit,
                lock_digest=hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
                facts=facts,
                decisions=decisions,
            )
            if metrics_write_failed:
                report = report.model_copy(
                    update={
                        "status": "IN_PROGRESS",
                        "incomplete_reasons": tuple(
                            dict.fromkeys((*report.incomplete_reasons, "write_failed"))
                        ),
                    }
                )
            publish(output_dir, "metrics-result.json", report)
            progress["metrics_status"] = report.status
            progress["missing_roles"] = report.missing_roles
            progress["metrics_reasons"] = report.incomplete_reasons
            if report.status != "PASS":
                diagnostic_errors.append("metrics_incomplete")
                if category is None:
                    category, stage = "evidence_mismatch", "metrics"
        except Exception:
            progress["metrics_status"] = "IN_PROGRESS"
            diagnostic_errors.append("report_failed")
            if category is None:
                category, stage = "report_failed", "metrics"
    result = SmokeRecord.model_validate(
        {
            **initial.model_dump(),
            **progress,
            "resources_released": released,
            "category": category,
            "failure_stage": stage if category else None,
            "diagnostic_errors": tuple(dict.fromkeys(diagnostic_errors)),
            "status": "PASS"
            if category is None
            and not diagnostic_errors
            and released
            and progress.get("metrics_status") == "PASS"
            and cleanup.get("status") == "PASS"
            else "IN_PROGRESS",
        }
    )
    if environment.output_created:
        try:
            publish(output_dir, "smoke-result.json", result)
        except Exception:
            result = result.model_copy(
                update={
                    "status": "IN_PROGRESS",
                    "category": result.category or "report_failed",
                    "failure_stage": result.failure_stage or "report",
                    "diagnostic_errors": tuple(
                        dict.fromkeys((*result.diagnostic_errors, "report_failed"))
                    ),
                }
            )
    if interrupted is not None:
        raise interrupted
    return result
