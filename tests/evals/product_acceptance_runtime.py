"""Two serial product cases using production HTTP, worker, checkpoint and Mock ports."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from random import Random
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.config import Settings
from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.actions import SqlAlchemyActionStore
from app.db.approval_expiry import SqlAlchemyApprovalRequestExpirySweeper
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.checkpoints import open_postgres_checkpointer
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
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
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ModelToolCall
from app.main import create_app
from app.retrieval.chunking import PreparedIngestionBatch
from app.retrieval.documents import DocumentIngestionService
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter
from app.tools.mock_application import create_approved_action_registry
from app.worker.backoff import ExponentialBackoff
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from app.worker.langgraph_executor import LangGraphRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.evals.product_acceptance_budget import DatabaseBudgetRecorder, admit
from tests.evals.product_acceptance_contracts import (
    CASES,
    DATASET,
    Decision,
    encoded,
    publish,
    read_private_json,
    require,
    review_digest,
    validate_review,
)
from tests.evals.quality_contracts import QualityPrivateSourceV1
from tests.evals.quality_dataset import (
    load_quality_dataset,
    prepare_quality_mapping_sources,
    project_model_payload,
    quality_digest,
)
from tests.evals.quality_generation import FrozenQualityWeb


class AcceptanceFakeChat(DeterministicResearchFakeChatAdapter):
    """CI requires both governed tools; live execution never uses this adapter."""

    async def invoke(self, messages, tools, metadata, *, attempt=None):
        if metadata.get("graph_node") == "research_agent" and not any(
            m.role == "tool" for m in messages
        ):
            payload = json.loads(
                (messages[-1].content or "")
                .split("<untrusted_research_state>\n", 1)[1]
                .split("\n</untrusted_research_state>", 1)[0]
            )
            query = payload["plan"]["queries"][0]
            return self._result(
                tool_calls=(
                    ModelToolCall(
                        call_id="acceptance-web",
                        name="search_web",
                        arguments={"query": query, "max_results": 8},
                    ),
                    ModelToolCall(
                        call_id="acceptance-rag",
                        name="retrieve_documents",
                        arguments={"query": query},
                    ),
                )
            )
        return await super().invoke(messages, tools, metadata, attempt=attempt)


class ProductSession:
    def __init__(
        self,
        database_url,
        root,
        chat,
        embedding,
        *,
        provider,
        source_check=lambda: None,
        markers=(),
    ):
        self.database_url, self.root = database_url, root
        self.chat, self.embedding, self.provider = chat, embedding, provider
        self.source_check, self.markers = source_check, markers
        self.dataset = load_quality_dataset(DATASET)
        self.cases = {c.case_id: c for c in self.dataset.cases if c.case_id in CASES}
        self.recorder = None
        self.runs = {}

    def publish(self, name, value):
        return publish(self.root / name, value, self.markers)

    @asynccontextmanager
    async def open(self):
        settings = Settings(
            _env_file=None,
            database_url=SecretStr(self.database_url),
            llm_mode="fake",
            search_mode="fake",
            auth_mode="fake",
            trace_mode="off",
            log_level="ERROR",
        )
        self.app = create_app(settings)
        async with self.app.router.lifespan_context(self.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=self.app), base_url="http://testserver"
            ) as self.client:
                me = await self.client.get("/api/v1/me")
                require(me.status_code == 200, "identity_failed")
                identity = me.json()
                self.tenant = TenantContext(
                    UUID(identity["workspaces"][0]["workspace_id"]),
                    UUID(identity["user_id"]),
                    WorkspaceRole.ADMIN,
                )
                self.base = f"/api/v1/workspaces/{self.tenant.workspace_id}"
                self.sessions = self.app.state.database_session_factory
                self.recorder = DatabaseBudgetRecorder(
                    self.sessions,
                    self.tenant,
                    provider=self.provider,
                    source_check=self.source_check,
                )
                try:
                    yield self
                finally:
                    rows = await self.recorder.rows()
                    self.publish(
                        "invocations.json",
                        {
                            "usage": await self.recorder.usage(),
                            "attempts": [
                                {
                                    "id": str(r.id),
                                    "run_id": str(r.run_id) if r.run_id else None,
                                    "kind": r.invocation_kind,
                                    "model": r.model,
                                    "status": r.status,
                                    "usage": r.token_usage,
                                    "cost": str(r.estimated_cost)
                                    if r.estimated_cost is not None
                                    else None,
                                    "pricing_version": r.pricing_version,
                                    "latency_ms": r.latency_ms,
                                    "error_category": r.error_category,
                                }
                                for r in rows
                            ],
                        },
                    )

    def factory(self):
        return LLMFactory(
            recorder=self.recorder,
            chat_adapter=self.chat,
            embedding_adapter=self.embedding,
            provider=self.provider,
        )

    async def ingest(self):
        prepared = prepare_quality_mapping_sources(DATASET)["resume_alpha"]
        (self.document_id,) = await DocumentIngestionService(
            SqlAlchemyDocumentRepository(self.sessions),
            self.factory().create_embedding_model(
                LLMInvocationContext(self.tenant.workspace_id, self.tenant.actor_user_id)
            ),
        ).ingest(tenant=self.tenant, batch=PreparedIngestionBatch((prepared,)))
        self.publish(
            "document.json",
            {
                "document_id": str(self.document_id),
                "content_hash": prepared.content_hash,
                "normalization_version": prepared.normalization_version,
                "chunking_version": prepared.chunking_version,
                "embedding_model": self.embedding.model,
            },
        )

    async def create(self, case_id):
        payload = project_model_payload(self.cases[case_id]).model_dump(mode="json")
        payload["resume_document_id"] = str(self.document_id)
        key = str(uuid4())
        headers = {"Idempotency-Key": key}
        first = await self.client.post(f"{self.base}/runs", json=payload, headers=headers)
        require(first.status_code == 202, "create_failed")
        # The observer retains this receipt, while the simulated client loses it.
        replay = await self.client.post(f"{self.base}/runs", json=payload, headers=headers)
        require(
            replay.status_code == 202
            and replay.json() == first.json()
            and replay.headers.get("Idempotency-Replayed") == "true",
            "replay_failed",
        )
        conflict = await self.client.post(
            f"{self.base}/runs",
            json={**payload, "query": payload["query"] + " 再次核对。"},
            headers=headers,
        )
        require(conflict.status_code == 409, "conflict_not_rejected")
        run_id = UUID(replay.json()["run_id"])
        self.runs[case_id] = run_id
        async with self.sessions() as session:
            runs = await session.scalar(
                select(func.count())
                .select_from(Run)
                .where(
                    Run.workspace_id == self.tenant.workspace_id,
                    Run.client_request_id == UUID(key),
                )
            )
            jobs = await session.scalar(
                select(func.count())
                .select_from(RunJob)
                .where(
                    RunJob.workspace_id == self.tenant.workspace_id,
                    RunJob.run_id == run_id,
                )
            )
        require(runs == jobs == 1, "duplicate_intent")
        self.publish(
            f"{case_id}-created.json",
            {
                "receipt": first.json(),
                "key": key,
                "replayed": True,
                "response_loss": "client_discard_after_server_commit",
                "changed_payload_status": conflict.status_code,
                "run_count": runs,
                "job_count": jobs,
            },
        )
        return run_id

    def search(self, case_id):
        alias = self.cases[case_id].web_scenario_alias
        source = next(s for s in self.dataset.manifest.sources if s.alias == alias)
        content = (DATASET / source.path).read_text()
        return FrozenQualityWeb(
            QualityPrivateSourceV1(
                source_alias=alias,
                kind="web",
                text=content,
                digest=quality_digest(content.encode()),
            )
        )

    async def work(self, case_id):
        self.source_check()
        require(not self.recorder.stopped, "execution_stopped")
        admit(await self.recorder.usage(), self.recorder.budget, after=True)
        runtime = WorkerRuntimeSettings()
        approvals = SqlAlchemyApprovalStore(self.sessions)
        registry = create_approved_action_registry(
            adapter=MockPortalHTTPAdapter(self.client),
            action_execution_store=SqlAlchemyActionExecutionStore(self.sessions),
            action_recovery_max_attempts=runtime.action_recovery_max_attempts,
        )
        # A fresh connection, executor and runner for EVERY execution segment.
        async with open_postgres_checkpointer(SecretStr(self.database_url)) as saver:
            executor = LangGraphRunExecutor(
                reader=SqlAlchemyRunExecutionReader(self.sessions),
                checkpointer=saver,
                llm_factory=self.factory(),
                search_port=self.search(case_id),
                tool_recorder=SqlAlchemyToolInvocationRecorder(self.sessions),
                document_repository=SqlAlchemyDocumentRepository(self.sessions),
                retrieval_event_recorder=SqlAlchemyRetrievalEventRecorder(self.sessions),
                action_store=SqlAlchemyActionStore(self.sessions),
                approval_resume_resolver=approvals,
                approved_action_executor=registry,
                execution_timeout_seconds=runtime.execution_timeout_seconds,
            )
            runner = WorkerRunner(
                worker_id=f"e83-{uuid4().hex}",
                store=SqlAlchemyWorkerJobStore(
                    self.sessions,
                    ExponentialBackoff(runtime, Random(0)),
                    action_recovery_max_attempts=runtime.action_recovery_max_attempts,
                ),
                tenant_service=TenantService(SqlAlchemyTenantResolver(self.sessions)),
                executor=executor,
                settings=runtime,
                approval_expiry_sweeper=SqlAlchemyApprovalRequestExpirySweeper(self.sessions),
                approved_action_executor=registry,
            )
            require(await runner.run_once(asyncio.Event()), "job_not_claimed")

    async def facts(self, case_id):
        run_id = self.runs[case_id]
        async with self.sessions() as session:

            async def rows(model):
                return list(
                    await session.scalars(
                        select(model).where(
                            model.workspace_id == self.tenant.workspace_id,
                            model.run_id == run_id,
                        )
                    )
                )

            actions, requests, jobs, tools, events, submissions = [
                await rows(model)
                for model in (
                    ActionIntent,
                    ApprovalRequest,
                    RunJob,
                    ToolInvocation,
                    RunEvent,
                    MockSubmission,
                )
            ]
            decisions = list(
                await session.scalars(
                    select(ApprovalDecision).where(
                        ApprovalDecision.workspace_id == self.tenant.workspace_id,
                        ApprovalDecision.approval_request_id.in_([r.id for r in requests]),
                    )
                )
            )
            checkpoints = await session.scalar(
                text("SELECT count(*) FROM pathfinder_checkpoint.checkpoints WHERE thread_id=:run"),
                {"run": str(run_id)},
            )
            run = await session.get(Run, run_id)
            return {
                "run_status": run.status,
                "graph_version": run.graph_version,
                "action_statuses": [a.status for a in actions],
                "request_statuses": [r.status for r in requests],
                "job_statuses": [j.status for j in jobs],
                "decisions": [d.decision for d in decisions],
                "mock_count": len(submissions),
                "checkpoint_count": checkpoints,
                "web_calls": sum(
                    t.tool_name == "search_web" and t.status == "succeeded" for t in tools
                ),
                "retrieval_calls": sum(
                    t.tool_name == "retrieve_documents" and t.status == "succeeded" for t in tools
                ),
                "rag_events": sum(e.type == "rag.retrieved" for e in events),
            }

    async def review(self, case_id):
        async with self.sessions() as session:
            actions = list(
                await session.scalars(
                    select(ActionIntent.id).where(
                        ActionIntent.workspace_id == self.tenant.workspace_id,
                        ActionIntent.run_id == self.runs[case_id],
                    )
                )
            )
        if not actions:
            return None
        require(len(actions) == 1, "unexpected_actions")
        response = await self.client.get(f"{self.base}/action-intents/{actions[0]}")
        require(response.status_code == 200, "review_failed")
        value = response.json()
        validate_review(value)
        self.publish(f"{case_id}-review.json", value)
        return value

    async def decide(self, case_id, decision):
        require(decision.case_id == case_id, "decision_case_mismatch")
        saved = read_private_json(self.root / f"{case_id}-review.json")
        require(decision.review_digest == review_digest(saved), "review_changed")
        action_id = saved["action_intent_id"]
        response = await self.client.get(f"{self.base}/action-intents/{action_id}")
        require(response.status_code == 200, "review_failed")
        current = response.json()
        validate_review(current)
        require(review_digest(current) == decision.review_digest, "review_changed")
        require(decision.acknowledged_at <= datetime.now(UTC), "future_decision")
        self.source_check()
        result = await self.client.post(
            f"{self.base}/action-intents/{action_id}/decision",
            json={
                "decision": decision.decision,
                "expected_version": current["approval_request"]["version"],
                "reason": "E8.3 exact content reviewed by operator",
            },
        )
        require(result.status_code == 200, "decision_failed")
        replay = await self.client.post(
            f"{self.base}/action-intents/{action_id}/decision",
            json={
                "decision": decision.decision,
                "expected_version": current["approval_request"]["version"],
                "reason": "E8.3 exact content reviewed by operator",
            },
        )
        require(
            replay.status_code == 200 and replay.json() == result.json(), "decision_replay_failed"
        )
        self.publish(f"{case_id}-decision-result.json", result.json())

    async def outcome(self, case_id):
        response = await self.client.get(f"{self.base}/runs/{self.runs[case_id]}")
        require(response.status_code == 200, "result_failed")
        self.publish(f"{case_id}-result.json", response.json())


async def wait_decision(root, case_id, review):
    expires = validate_review(review).approval_request.expires_at
    path = root / f"{case_id}-decision.json"
    while datetime.now(UTC) < expires:
        if path.exists():
            return Decision.model_validate_json(encoded(read_private_json(path)))
        await asyncio.sleep(1)
    require(False, "approval_expired")


def accepted_facts(facts, decision):
    return (
        facts["run_status"] == "completed"
        and facts["job_statuses"] == ["done"]
        and facts["graph_version"] == "pathfinder-research-v6"
        and facts["checkpoint_count"] > 0
        and facts["decisions"] == [decision]
        and facts["action_statuses"] == (["succeeded"] if decision == "approve" else ["cancelled"])
        and facts["request_statuses"] == (["consumed"] if decision == "approve" else ["rejected"])
        and facts["mock_count"] == (1 if decision == "approve" else 0)
        and facts["web_calls"] > 0
        and facts["retrieval_calls"] > 0
        and facts["rag_events"] > 0
    )


async def run_cases(product, *, decision_reader=wait_decision):
    records = [{"case_id": case, "status": "NOT_RUN", "checks_passed": False} for case in CASES]
    try:
        await product.ingest()
        for item in records:
            case_id = item["case_id"]
            item["status"] = "IN_PROGRESS"
            await product.create(case_id)
            await product.work(case_id)
            before = await product.facts(case_id)
            product.publish(f"{case_id}-before.json", before)
            require(before["mock_count"] == 0, "effect_before_approval")
            review = await product.review(case_id)
            if review is None:
                item.update(status="NO_APPROVAL", facts=before)
                await product.outcome(case_id)
                require(
                    before["run_status"] in {"completed", "failed", "cancelled"}
                    and not product.recorder.stopped,
                    "incomplete_execution",
                )
                continue
            require(
                before["run_status"] == "waiting_approval" and before["checkpoint_count"] > 0,
                "not_waiting_approval",
            )
            decision = await decision_reader(product.root, case_id, review)
            await product.decide(case_id, decision)
            await product.work(case_id)
            facts = await product.facts(case_id)
            product.publish(f"{case_id}-after.json", facts)
            await product.outcome(case_id)
            expected = "approve" if case_id == "mixed_alpha" else "reject"
            checks = (
                decision.decision == expected
                and accepted_facts(facts, expected)
                and facts["checkpoint_count"] > before["checkpoint_count"]
            )
            item.update(status="PASS" if checks else "PARTIAL", checks_passed=checks, facts=facts)
    finally:
        product.publish("cases.json", {"cases": records, "usage": await product.recorder.usage()})
    return records
