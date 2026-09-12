from __future__ import annotations

import asyncio
import re
from random import Random
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from app.auth.fake import FAKE_ACTOR_SUBJECT
from app.config import Settings
from app.db.checkpoints import CHECKPOINT_SCHEMA, open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.main import create_app
from app.tools.fake_search import FakeSearch
from app.worker.backoff import ExponentialBackoff
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.support import connect_database

pytestmark = pytest.mark.integration


def _sse_facts(body: str) -> list[tuple[int, str]]:
    return [
        (int(sequence), event_type)
        for sequence, event_type in re.findall(r"id: (\d+)\nevent: ([a-z_.]+)\n", body)
    ]


async def test_gate4_http_worker_checkpoint_and_terminal_sse_with_reconnect(
    migrated_database_url: str,
) -> None:
    bootstrap_engine = create_database_engine(SecretStr(migrated_database_url))
    bootstrap_sessions = create_session_factory(bootstrap_engine)
    identity = await ProvisioningService(
        SqlAlchemyProvisioningStore(bootstrap_sessions)
    ).provision_personal_workspace(FAKE_ACTOR_SUBJECT)
    await bootstrap_engine.dispose()

    application = create_app(
        Settings(database_url=SecretStr(migrated_database_url), log_level="ERROR")
    )
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            created = await client.post(
                f"/api/v1/workspaces/{identity.workspace_id}/runs",
                json={"mode": "research", "query": "Gate 4 vertical fake E2E"},
            )
            assert created.status_code == 202
            run_id = UUID(created.json()["run_id"])
            queued = await client.get(f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}")
            assert queued.status_code == 200 and queued.json()["status"] == "queued"

            session_factory = application.state.database_session_factory
            runtime_settings = WorkerRuntimeSettings()
            store = SqlAlchemyWorkerJobStore(
                session_factory,
                ExponentialBackoff(runtime_settings, Random(0)),
            )
            async with open_postgres_checkpointer(SecretStr(migrated_database_url)) as checkpointer:
                runner = WorkerRunner(
                    worker_id="gate-4-e2e-worker",
                    store=store,
                    tenant_service=TenantService(SqlAlchemyTenantResolver(session_factory)),
                    executor=LangGraphRunExecutor(
                        reader=SqlAlchemyRunExecutionReader(session_factory),
                        checkpointer=checkpointer,
                        llm_factory=LLMFactory(
                            recorder=SqlAlchemyInvocationRecorder(session_factory),
                            chat_adapter=DeterministicResearchFakeChatAdapter(),
                            embedding_adapter=FakeEmbeddingModel(),
                        ),
                        search_port=FakeSearch({}),
                        tool_recorder=SqlAlchemyToolInvocationRecorder(session_factory),
                        document_repository=SqlAlchemyDocumentRepository(session_factory),
                        retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(session_factory),
                        execution_timeout_seconds=300.0,
                        execution_guard_poll_seconds=0.01,
                    ),
                    settings=runtime_settings,
                )
                assert await runner.run_once(asyncio.Event()) is True

            completed = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}"
            )
            assert completed.status_code == 200
            assert completed.json()["status"] == "completed"
            assert completed.json()["result"] is not None

            all_events = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}/events"
            )
            assert all_events.status_code == 200
            facts = _sse_facts(all_events.text)
            assert facts[-1][1] == "run.completed"
            assert [sequence for sequence, _event_type in facts] == list(range(1, len(facts) + 1))

            reconnect_cursor = 2
            resumed_events = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}/events",
                headers={"Last-Event-ID": str(reconnect_cursor)},
            )
            resumed_facts = _sse_facts(resumed_events.text)
            assert resumed_facts == [fact for fact in facts if fact[0] > reconnect_cursor]

    with connect_database(migrated_database_url) as connection:
        persisted_events = connection.execute(
            "SELECT seq, type FROM run_events WHERE run_id = %s ORDER BY seq",
            (run_id,),
        ).fetchall()
        checkpoint_count = connection.execute(
            f'SELECT count(*) FROM "{CHECKPOINT_SCHEMA}".checkpoints WHERE thread_id = %s',
            (str(run_id),),
        ).fetchone()
    assert facts == [(row[0], row[1]) for row in persisted_events]
    assert checkpoint_count is not None and checkpoint_count[0] > 0
