"""Bounded single-run driver over the real owned API/worker/graph/DB/SSE chain."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections import Counter
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


class HTTP:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.count = 0

    def reserve(self):
        require(self.count < MAX_REQUESTS, "request_limit")
        self.count += 1

    async def request(self, method, path, *, expected=200, **kwargs):
        self.reserve()
        try:
            response = await self.client.request(method, path, **kwargs)
            require(response.status_code == expected, "http_failed")
            return response.json()
        except (httpx.HTTPError, ValueError):
            raise SmokeFailure("http_failed") from None

    async def events(self, path, *, cursor=0):
        self.reserve()
        body = bytearray()
        try:
            async with self.client.stream(
                "GET", path, headers={"Last-Event-ID": str(cursor)}
            ) as response:
                require(response.status_code == 200, "http_failed")
                async for chunk in response.aiter_bytes():
                    require(len(body) + len(chunk) <= MAX_SSE_BYTES)
                    body.extend(chunk)
            return parse_events(body.decode())
        except (httpx.HTTPError, ValueError, UnicodeError):
            raise SmokeFailure("http_failed") from None


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
    }


async def drive(environment, policy, mode, progress):
    engine = create_database_engine(SecretStr(environment.database_url))
    sessions = create_session_factory(engine)
    started = monotonic()
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
            http = HTTP(client)
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
            finally:
                progress["http_requests"] = http.count
    finally:
        progress["elapsed_seconds"] = monotonic() - started
        await asyncio.wait_for(engine.dispose(), timeout=5)


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
        EnvironmentProfile(PROFILE), output_dir, call_profile=selected.model_dump(mode="json")
    )
    progress = {}
    category = None
    interrupted = None
    try:
        with environment:
            publish(output_dir, "smoke-started.json", initial)
            asyncio.run(drive(environment, selected, mode, progress))
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        category = "cancelled"
        interrupted = exc
    except Exception as exc:
        category = failure_category(exc)
    cleanup = environment.close()
    released = cleanup.get("resources_released") is True
    if not released:
        category = "cleanup_failed"
    if environment.output_created:
        try:
            progress.update(partial_calls(output_dir))
        except Exception:
            category = "report_failed"
    result = SmokeRecord.model_validate(
        {
            **initial.model_dump(),
            **progress,
            "resources_released": released,
            "category": category,
            "status": "PASS"
            if category is None and cleanup.get("status") == "PASS"
            else "IN_PROGRESS",
        }
    )
    if environment.output_created:
        publish(output_dir, "smoke-result.json", result)
    if interrupted is not None:
        raise interrupted
    return result
