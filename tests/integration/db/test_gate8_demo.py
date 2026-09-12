from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from random import Random
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select

from app.auth.fake import FAKE_ACTOR_SUBJECT
from app.config import Settings
from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.actions import SqlAlchemyActionStore
from app.db.approval_expiry import SqlAlchemyApprovalRequestExpirySweeper
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.checkpoints import CHECKPOINT_SCHEMA, open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    Document,
    DocumentChunk,
    LLMInvocation,
    MockSubmission,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
)
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext, TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import (
    ChatMessage,
    ChatModelResult,
    ModelToolCall,
    ModelToolSchema,
    ProviderAttemptContext,
)
from app.main import create_app
from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.documents import DocumentIngestionService
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.fake_search import FakeSearch
from app.tools.mock_application import create_approved_action_registry
from app.tools.search import normalize_search_result
from app.worker.backoff import ExponentialBackoff
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.support import connect_database

pytestmark = pytest.mark.integration

_QUERY = "Compare a synthetic backend role and draft a grounded application"
_FAILURE_MODES = {"", "rag", "approval", "mock", "sse"}


class _Gate8FakeChatAdapter(DeterministicResearchFakeChatAdapter):
    """Exercise both production research tools in one deterministic agent turn."""

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext | None = None,
    ) -> ChatModelResult:
        if metadata.get("graph_node") == "research_agent" and not any(
            message.role == "tool" for message in messages
        ):
            payload = json.loads(
                (messages[-1].content or "")
                .split("<untrusted_research_state>\n", 1)[1]
                .split("\n</untrusted_research_state>", 1)[0]
            )
            query = payload["plan"]["queries"][0]
            pass_number = metadata.get("research_pass_number", "1")
            return self._result(
                tool_calls=(
                    ModelToolCall(
                        call_id=f"gate8-search-{pass_number}",
                        name="search_web",
                        arguments={"query": query, "max_results": 8},
                    ),
                    ModelToolCall(
                        call_id=f"gate8-rag-{pass_number}",
                        name="retrieve_documents",
                        arguments={"query": query},
                    ),
                )
            )
        return await super().invoke(messages, tools, metadata, attempt=attempt)


def _sse_frames(body: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for raw_frame in body.strip().split("\n\n") if body.strip() else ():
        lines = raw_frame.splitlines()
        frames.append(
            {
                "id": int(lines[0].removeprefix("id: ")),
                "event": lines[1].removeprefix("event: "),
                "data": json.loads(lines[2].removeprefix("data: ")),
            }
        )
    return frames


def _checkpoint_count(database_url: str, run_id: UUID) -> int:
    with connect_database(database_url) as connection:
        row = connection.execute(
            f'SELECT count(*) FROM "{CHECKPOINT_SCHEMA}".checkpoints WHERE thread_id = %s',
            (str(run_id),),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _observed(ok: bool, *, subsystem: str, failure_mode: str) -> bool:
    return ok and failure_mode != subsystem


async def test_gate8_demo_complete_application_flow(
    migrated_database_url: str,
) -> None:
    failure_mode = os.getenv("PF_GATE87_DEMO_FAILURE", "")
    assert failure_mode in _FAILURE_MODES, "unsupported Gate 8.7 demo failure selector"
    settings = Settings(
        database_url=SecretStr(migrated_database_url),
        llm_mode="fake",
        search_mode="fake",
        auth_mode="fake",
        trace_mode="off",
        log_level="ERROR",
    )
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            me = await client.get("/api/v1/me")
            assert me.status_code == 200
            identity = me.json()
            assert len(identity["workspaces"]) == 1
            actor_id = UUID(identity["user_id"])
            workspace_id = UUID(identity["workspaces"][0]["workspace_id"])
            assert identity["workspaces"][0]["role"] == "admin"
            sessions = application.state.database_session_factory
            tenant = TenantContext(workspace_id, actor_id, WorkspaceRole.ADMIN)

            resume_text = (
                "# Synthetic Pathfinder Resume\n\n"
                "Candidate builds Python services, PostgreSQL systems, and reliable APIs.\n\n"
                "All names and experience in this document are synthetic."
            )
            batch = normalize_and_chunk_batch(
                ValidatedIngestionBatch(
                    (
                        ValidatedIngestionSource(
                            source_name="gate87-synthetic-resume.md",
                            source_type="markdown",
                            title="Synthetic Pathfinder Resume",
                            raw_text=resume_text,
                            character_count=len(resume_text),
                        ),
                    )
                )
            )
            (document_id,) = await DocumentIngestionService(
                SqlAlchemyDocumentRepository(sessions),
                LLMFactory(
                    recorder=SqlAlchemyInvocationRecorder(sessions),
                    chat_adapter=_Gate8FakeChatAdapter(),
                    embedding_adapter=FakeEmbeddingModel(),
                ).create_embedding_model(LLMInvocationContext(workspace_id, actor_id)),
            ).ingest(tenant=tenant, batch=batch)
            async with sessions() as session:
                document = await session.get(Document, document_id)
                chunks = tuple(
                    await session.scalars(
                        select(DocumentChunk)
                        .where(
                            DocumentChunk.workspace_id == workspace_id,
                            DocumentChunk.document_id == document_id,
                        )
                        .order_by(DocumentChunk.ordinal)
                    )
                )
            assert document is not None and chunks
            document_snapshot = (
                document.workspace_id,
                document.content,
                document.content_hash,
                document.normalization_version,
                document.chunking_version,
                document.embedding_model,
            )
            chunk_snapshot = tuple(
                (row.id, row.ordinal, row.content_hash, row.text, row.embedding_model)
                for row in chunks
            )

            created = await client.post(
                f"/api/v1/workspaces/{workspace_id}/runs",
                json={
                    "mode": "application",
                    "query": _QUERY,
                    "resume_document_id": str(document_id),
                },
            )
            assert created.status_code == 202, created.text
            run_id = UUID(created.json()["run_id"])

            runtime = WorkerRuntimeSettings()
            store = SqlAlchemyWorkerJobStore(
                sessions,
                ExponentialBackoff(runtime, Random(0)),
                action_recovery_max_attempts=runtime.action_recovery_max_attempts,
            )
            approval_store = SqlAlchemyApprovalStore(sessions)
            search = FakeSearch(
                {
                    _QUERY: (
                        normalize_search_result(
                            title="Synthetic backend role",
                            url="https://example.test/jobs/backend",
                            snippet=(
                                "The synthetic role values Python, PostgreSQL, and API reliability."
                            ),
                            published_at="2026-08-28",
                        ),
                    )
                }
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application),
                base_url="http://testserver",
            ) as mock_client:
                approved_registry = create_approved_action_registry(
                    adapter=MockPortalHTTPAdapter(mock_client),
                    action_execution_store=SqlAlchemyActionExecutionStore(sessions),
                    action_recovery_max_attempts=runtime.action_recovery_max_attempts,
                )
                async with open_postgres_checkpointer(
                    SecretStr(migrated_database_url)
                ) as checkpointer:
                    executor = LangGraphRunExecutor(
                        reader=SqlAlchemyRunExecutionReader(sessions),
                        checkpointer=checkpointer,
                        llm_factory=LLMFactory(
                            recorder=SqlAlchemyInvocationRecorder(sessions),
                            chat_adapter=_Gate8FakeChatAdapter(),
                            embedding_adapter=FakeEmbeddingModel(),
                        ),
                        search_port=search,
                        tool_recorder=SqlAlchemyToolInvocationRecorder(sessions),
                        document_repository=SqlAlchemyDocumentRepository(sessions),
                        retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(sessions),
                        action_store=SqlAlchemyActionStore(sessions),
                        approval_resume_resolver=approval_store,
                        approved_action_executor=approved_registry,
                        execution_timeout_seconds=300.0,
                        execution_guard_poll_seconds=0.01,
                    )
                    runner = WorkerRunner(
                        worker_id="gate87-demo-worker",
                        store=store,
                        tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
                        executor=executor,
                        settings=runtime,
                        approval_expiry_sweeper=SqlAlchemyApprovalRequestExpirySweeper(sessions),
                        approved_action_executor=approved_registry,
                    )
                    assert await runner.run_once(asyncio.Event()) is True
                    pre_approval_checkpoints = _checkpoint_count(migrated_database_url, run_id)
                    assert pre_approval_checkpoints > 0

                    async with sessions() as session:
                        waiting_run = await session.get(Run, run_id)
                        waiting_job = await session.scalar(
                            select(RunJob).where(
                                RunJob.workspace_id == workspace_id,
                                RunJob.run_id == run_id,
                            )
                        )
                        actions = list(
                            await session.scalars(
                                select(ActionIntent).where(
                                    ActionIntent.workspace_id == workspace_id,
                                    ActionIntent.run_id == run_id,
                                )
                            )
                        )
                        requests = list(
                            await session.scalars(
                                select(ApprovalRequest).where(
                                    ApprovalRequest.workspace_id == workspace_id,
                                    ApprovalRequest.run_id == run_id,
                                )
                            )
                        )
                        pre_mock_count = await session.scalar(
                            select(func.count())
                            .select_from(MockSubmission)
                            .where(
                                MockSubmission.workspace_id == workspace_id,
                                MockSubmission.run_id == run_id,
                            )
                        )
                    assert waiting_run is not None and waiting_run.status == "waiting_approval"
                    assert waiting_job is not None and waiting_job.status == "done"
                    assert waiting_job.owner_token is None and waiting_job.lease_expires_at is None
                    assert len(actions) == 1 and actions[0].status == "proposed"
                    assert len(requests) == 1 and requests[0].status == "pending"
                    assert pre_mock_count == 0

                    review = await client.get(
                        f"/api/v1/workspaces/{workspace_id}/action-intents/{actions[0].id}"
                    )
                    assert review.status_code == 200
                    review_json = review.json()
                    request_json = review_json["approval_request"]
                    assert review_json["args_digest"] == request_json["args_digest"]
                    assert review_json["target_digest"] == request_json["target_digest"]
                    assert (
                        review_json["approval_binding_digest"]
                        == request_json["approval_binding_digest"]
                    )
                    assert (
                        review_json["approval_binding_version"]
                        == request_json["approval_binding_version"]
                    )
                    decision = await client.post(
                        f"/api/v1/workspaces/{workspace_id}/action-intents/{actions[0].id}/decision",
                        json={
                            "decision": "approve",
                            "expected_version": request_json["version"],
                            "reason": "Gate 8.7 synthetic acceptance",
                        },
                    )
                    assert decision.status_code == 200, decision.text
                    assert decision.json()["decision"] == "approve"
                    assert await runner.run_once(asyncio.Event()) is True
                    post_approval_checkpoints = _checkpoint_count(migrated_database_url, run_id)
                    assert post_approval_checkpoints > pre_approval_checkpoints

            terminal = await client.get(f"/api/v1/workspaces/{workspace_id}/runs/{run_id}")
            assert terminal.status_code == 200
            assert terminal.json()["status"] == "completed"
            assert terminal.json()["result"] is not None

            async with sessions() as session:
                run = await session.get(Run, run_id)
                job = await session.scalar(
                    select(RunJob).where(
                        RunJob.workspace_id == workspace_id,
                        RunJob.run_id == run_id,
                    )
                )
                action = await session.get(ActionIntent, actions[0].id)
                request = await session.get(ApprovalRequest, requests[0].id)
                decisions = list(
                    await session.scalars(
                        select(ApprovalDecision).where(
                            ApprovalDecision.workspace_id == workspace_id,
                            ApprovalDecision.approval_request_id == requests[0].id,
                        )
                    )
                )
                tools = list(
                    await session.scalars(
                        select(ToolInvocation)
                        .where(
                            ToolInvocation.workspace_id == workspace_id,
                            ToolInvocation.run_id == run_id,
                        )
                        .order_by(ToolInvocation.created_at, ToolInvocation.id)
                    )
                )
                llm = list(
                    await session.scalars(
                        select(LLMInvocation).where(
                            LLMInvocation.workspace_id == workspace_id,
                            LLMInvocation.run_id == run_id,
                        )
                    )
                )
                events = list(
                    await session.scalars(
                        select(RunEvent)
                        .where(
                            RunEvent.workspace_id == workspace_id,
                            RunEvent.run_id == run_id,
                        )
                        .order_by(RunEvent.seq)
                    )
                )
                submissions = list(
                    await session.scalars(
                        select(MockSubmission).where(
                            MockSubmission.workspace_id == workspace_id,
                            MockSubmission.run_id == run_id,
                        )
                    )
                )
                persisted_document = await session.get(Document, document_id)
                persisted_chunks = tuple(
                    await session.scalars(
                        select(DocumentChunk)
                        .where(
                            DocumentChunk.workspace_id == workspace_id,
                            DocumentChunk.document_id == document_id,
                        )
                        .order_by(DocumentChunk.ordinal)
                    )
                )

            assert run is not None and run.workspace_id == workspace_id
            assert run.created_by_user_id == actor_id and run.status == "completed"
            assert run.resume_document_id == document_id and run.error_category is None
            assert job is not None and job.status == "done" and job.leased_by is None
            assert job.owner_token is None and job.lease_expires_at is None
            assert action is not None and action.status == "succeeded"
            assert (
                action.workspace_id == workspace_id and action.originating_actor_user_id == actor_id
            )
            assert request is not None and request.status == "consumed"
            assert request.args_digest == action.args_digest
            assert request.target_digest == action.target_digest
            assert request.approval_binding_digest == action.approval_binding_digest
            assert request.approval_binding_version == action.approval_binding_version
            assert _observed(
                len(decisions) == 1 and decisions[0].decision == "approve",
                subsystem="approval",
                failure_mode=failure_mode,
            ), "Gate 8.7 approval acceptance failed"

            by_name = {
                name: [row for row in tools if row.tool_name == name]
                for name in {
                    "search_web",
                    "retrieve_documents",
                    "submit_mock_application",
                }
            }
            assert len(by_name["search_web"]) == 1
            assert by_name["search_web"][0].status == "succeeded"
            assert _observed(
                len(by_name["retrieve_documents"]) == 1
                and by_name["retrieve_documents"][0].status == "succeeded"
                and any(event.type == "rag.retrieved" for event in events),
                subsystem="rag",
                failure_mode=failure_mode,
            ), "Gate 8.7 RAG acceptance failed"
            irreversible = by_name["submit_mock_application"]
            assert len(irreversible) == 1 and irreversible[0].status == "succeeded"
            assert irreversible[0].action_intent_id == action.id
            assert all(row.status in {"succeeded", "failed"} for row in llm)
            assert llm and all(
                row.workspace_id == workspace_id
                and row.actor_user_id == actor_id
                and row.provider == "fake"
                for row in llm
            )
            assert not [
                row for row in tools if row.status in {"prepared", "executing", "outcome_unknown"}
            ]
            assert not [row for row in llm if row.status not in {"succeeded", "failed"}]
            assert _observed(
                len(submissions) == 1
                and submissions[0].action_intent_id == action.id
                and submissions[0].idempotency_key == action.idempotency_key
                and action.idempotency_key == str(action.id),
                subsystem="mock",
                failure_mode=failure_mode,
            ), "Gate 8.7 Mock exactly-once acceptance failed"

            assert persisted_document is not None
            assert (
                persisted_document.workspace_id,
                persisted_document.content,
                persisted_document.content_hash,
                persisted_document.normalization_version,
                persisted_document.chunking_version,
                persisted_document.embedding_model,
            ) == document_snapshot
            assert (
                tuple(
                    (row.id, row.ordinal, row.content_hash, row.text, row.embedding_model)
                    for row in persisted_chunks
                )
                == chunk_snapshot
            )
            rag_events = [event for event in events if event.type == "rag.retrieved"]
            assert rag_events
            assert str(document_id) in rag_events[0].payload["document_ids"]
            assert set(rag_events[0].payload["chunk_ids"]).issubset(
                {str(row.id) for row in persisted_chunks}
            )

            sequences = [event.seq for event in events]
            assert sequences == list(range(1, len(events) + 1))
            assert run.next_event_seq == len(events) + 1
            assert events[-1].type == "run.completed"
            all_events = await client.get(f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events")
            assert all_events.status_code == 200
            frames = _sse_frames(all_events.text)
            database_projection = [
                {
                    "id": event.seq,
                    "event": event.type,
                    "data": {
                        "version": event.version,
                        "run_id": str(event.run_id),
                        "seq": event.seq,
                        "occurred_at": event.recorded_at.isoformat().replace("+00:00", "Z"),
                        "payload": event.payload,
                    },
                }
                for event in events
            ]
            midpoint = frames[len(frames) // 2]["id"]
            suffix = await client.get(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
                headers={"Last-Event-ID": str(midpoint)},
            )
            terminal_reconnect = await client.get(
                f"/api/v1/workspaces/{workspace_id}/runs/{run_id}/events",
                headers={"Last-Event-ID": str(frames[-1]["id"])},
            )
            assert _observed(
                frames == database_projection
                and _sse_frames(suffix.text)
                == [frame for frame in frames if frame["id"] > midpoint]
                and _sse_frames(terminal_reconnect.text) == [],
                subsystem="sse",
                failure_mode=failure_mode,
            ), "Gate 8.7 SSE acceptance failed"

    assert FAKE_ACTOR_SUBJECT not in json.dumps(identity)
