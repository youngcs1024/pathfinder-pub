from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError

from app.api.routes.events import stream_run_events
from app.auth.fake import FAKE_ACTOR_SUBJECT
from app.config import Settings
from app.db.events import SqlAlchemyRunEventReader
from app.db.models import (
    Conversation,
    Document,
    LLMInvocation,
    Message,
    Run,
    RunEvent,
    RunJob,
    Workspace,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import (
    AsyncSessionFactory,
    create_database_engine,
    create_session_factory,
    transaction,
)
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import (
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
    InvalidEventCursorError,
)
from app.domain.jobs import JobStatus
from app.domain.provisioning import ProvisioningService, WorkspaceKind, WorkspaceRole
from app.domain.run_execution import RunExecutionInvalidError
from app.domain.runs import DEFAULT_RUN_LIMITS, RunMode, RunService, RunStatus
from app.domain.tenancy import TenantContext, TenantService
from app.events.contracts import RunEventType
from tests.integration.support import connect_database
from tests.legacy_app import create_app
from tests.legacy_runtime import SqlAlchemyRunExecutionReader, SqlAlchemyRunStore

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _Runtime:
    session_factory: AsyncSessionFactory
    provisioning: ProvisioningService
    tenant_service: TenantService
    run_service: RunService


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        yield _Runtime(
            session_factory=session_factory,
            provisioning=ProvisioningService(SqlAlchemyProvisioningStore(session_factory)),
            tenant_service=TenantService(SqlAlchemyTenantResolver(session_factory)),
            run_service=RunService(SqlAlchemyRunStore(session_factory)),
        )
    finally:
        await engine.dispose()


async def _personal_tenant(runtime: _Runtime, subject: str) -> TenantContext:
    identity = await runtime.provisioning.provision_personal_workspace(subject)
    return await runtime.tenant_service.resolve_tenant(
        workspace_id=identity.workspace_id,
        actor_user_id=identity.user_id,
    )


async def _business_counts(session_factory: AsyncSessionFactory) -> tuple[int, ...]:
    async with session_factory() as session:
        counts: list[int] = []
        for model in (Conversation, Message, Run, RunJob, RunEvent):
            counts.append((await session.scalar(select(func.count()).select_from(model))) or 0)
    return tuple(counts)


async def _resume(runtime: _Runtime, tenant: TenantContext, *, suffix: str = "a") -> UUID:
    document_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add(
            Document(
                id=document_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                title="resume",
                source_type="markdown",
                source_name="resume.md",
                content=f"Resume {suffix}",
                content_hash=(suffix * 64)[:64],
                normalization_version="text-normalization-v1",
                chunking_version="document-chunking-v1",
                embedding_model="qwen-beijing-text-embedding-v4-1536-v1",
            )
        )
    return document_id


async def test_application_resume_binding_is_required_workspace_safe_and_persisted(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "application-run-user")
    foreign = await _personal_tenant(runtime, "foreign-resume-user")
    resume_id = await _resume(runtime, tenant, suffix="a")
    foreign_id = await _resume(runtime, foreign, suffix="b")

    with pytest.raises(DomainValidationError, match="application requires"):
        await runtime.run_service.create_run(
            tenant=tenant,
            mode=RunMode.APPLICATION,
            query="Draft an application",
        )
    for unavailable_id in (uuid4(), foreign_id):
        with pytest.raises(DomainNotFoundError):
            await runtime.run_service.create_run(
                tenant=tenant,
                mode=RunMode.APPLICATION,
                query="Draft an application",
                resume_document_id=unavailable_id,
            )

    accepted = await runtime.run_service.create_run(
        tenant=tenant,
        mode=RunMode.APPLICATION,
        query="Draft an application",
        resume_document_id=resume_id,
    )
    async with runtime.session_factory() as session:
        run = await session.scalar(select(Run).where(Run.id == accepted.run_id))
    assert run is not None
    assert run.mode == "application"
    assert run.resume_document_id == resume_id
    assert run.input_json["include_application_draft"] is True


async def test_research_optional_resume_and_safe_unknown_foreign_rejection(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "research-resume-user")
    foreign = await _personal_tenant(runtime, "research-foreign-resume-user")
    resume_id = await _resume(runtime, tenant, suffix="c")
    foreign_id = await _resume(runtime, foreign, suffix="d")

    without_resume = await runtime.run_service.create_run(
        tenant=tenant, mode=RunMode.RESEARCH, query="Research without resume"
    )
    with_resume = await runtime.run_service.create_run(
        tenant=tenant,
        mode=RunMode.RESEARCH,
        query="Research with resume",
        resume_document_id=resume_id,
    )
    for unavailable_id in (uuid4(), foreign_id):
        with pytest.raises(DomainNotFoundError):
            await runtime.run_service.create_run(
                tenant=tenant,
                mode=RunMode.RESEARCH,
                query="Unsafe resume",
                resume_document_id=unavailable_id,
            )

    async with runtime.session_factory() as session:
        rows = {
            row.id: row
            for row in await session.scalars(
                select(Run).where(Run.id.in_((without_resume.run_id, with_resume.run_id)))
            )
        }
    assert rows[without_resume.run_id].resume_document_id is None
    assert rows[with_resume.run_id].resume_document_id == resume_id
    assert rows[with_resume.run_id].input_json["include_application_draft"] is False


async def test_resume_composite_fk_rejects_manual_cross_workspace_splice(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "resume-fk-user")
    foreign = await _personal_tenant(runtime, "resume-fk-foreign-user")
    resume_id = await _resume(runtime, tenant, suffix="e")
    foreign_id = await _resume(runtime, foreign, suffix="f")
    accepted = await runtime.run_service.create_run(
        tenant=tenant,
        mode=RunMode.APPLICATION,
        query="Bound application",
        resume_document_id=resume_id,
    )

    with pytest.raises(DBAPIError):
        async with transaction(runtime.session_factory) as session:
            await session.execute(
                update(Run).where(Run.id == accepted.run_id).values(resume_document_id=foreign_id)
            )


async def test_revoked_actor_cannot_create_resume_bound_run(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "revoked-resume-creator")
    resume_id = await _resume(runtime, tenant, suffix="7")
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == tenant.workspace_id,
                WorkspaceMembership.user_id == tenant.actor_user_id,
            )
            .values(revoked_at=datetime.now(UTC))
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.run_service.create_run(
            tenant=tenant,
            mode=RunMode.APPLICATION,
            query="Rejected application",
            resume_document_id=resume_id,
        )


async def test_terminal_v1_run_and_output_remain_readable_after_v2_activation(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "historical-v1-reader")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Historical")
    result = {
        "schema_version": 1,
        "evidence_sufficient": False,
        "summary": [],
        "findings": [],
        "evidence": [],
        "limitations": [
            {"code": "insufficient_evidence", "detail": "Historical bounded evidence."}
        ],
        "sources": [],
        "application_draft": None,
    }
    now = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(
                graph_version="pathfinder-research-v1",
                status="completed",
                started_at=now,
                finished_at=now,
                result_json=result,
            )
        )
    record = await runtime.run_service.get_run(tenant=tenant, run_id=accepted.run_id)
    assert record.graph_version == "pathfinder-research-v1"
    assert record.result is not None and record.result.schema_version == 1


async def test_execution_reader_refuses_historical_v1_even_when_argument_matches(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "historical-v1-execution-reader")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Historical")
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(
                graph_version="pathfinder-research-v1",
                status="running",
                started_at=datetime.now(UTC),
            )
        )
    with pytest.raises(RunExecutionInvalidError) as captured:
        await SqlAlchemyRunExecutionReader(runtime.session_factory).read_for_execution(
            run_id=accepted.run_id,
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            graph_version="pathfinder-research-v1",
        )
    assert captured.value.category == "unknown_graph_version"


async def test_create_run_commits_exact_atomic_intent_graph(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "run-create-user")

    accepted = await runtime.run_service.create_research_run(
        tenant=tenant,
        query=" Senior Python\tEngineer \n",
    )

    assert accepted.status is RunStatus.QUEUED
    assert await _business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)
    async with runtime.session_factory() as session:
        conversation = (await session.scalars(select(Conversation))).one()
        message = (await session.scalars(select(Message))).one()
        run = (await session.scalars(select(Run))).one()
        job = (await session.scalars(select(RunJob))).one()
        event = (await session.scalars(select(RunEvent))).one()

    assert conversation.title == "Research request"
    assert message.content == "Senior Python Engineer"
    assert message.actor_user_id == tenant.actor_user_id
    assert run.id == accepted.run_id
    assert run.input_json == {
        "schema_version": 1,
        "query": "Senior Python Engineer",
        "include_application_draft": False,
    }
    assert run.limits_json == DEFAULT_RUN_LIMITS
    assert run.status == "queued"
    assert run.graph_version == "pathfinder-research-v6"
    assert run.next_event_seq == 2
    assert job.run_id == run.id
    assert job.status == "queued"
    assert event.seq == 1
    assert event.type == "run.created"
    assert event.payload == {
        "mode": "research",
        "status": "queued",
        "graph_version": "pathfinder-research-v6",
    }
    assert "query" not in event.payload


async def test_duplicate_http_intents_create_distinct_runs_with_one_job_each(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "duplicate-run-user")

    first, second = await asyncio.gather(
        runtime.run_service.create_research_run(tenant=tenant, query="Same request"),
        runtime.run_service.create_research_run(tenant=tenant, query="Same request"),
    )

    assert first.run_id != second.run_id
    assert await _business_counts(runtime.session_factory) == (2, 2, 2, 2, 2)
    async with runtime.session_factory() as session:
        job_counts = (
            await session.execute(
                select(RunJob.run_id, func.count(RunJob.id)).group_by(RunJob.run_id)
            )
        ).all()
    assert {count for _run_id, count in job_counts} == {1}


@pytest.mark.parametrize("table_name", ["messages", "run_jobs", "run_events"])
async def test_insert_failure_rolls_back_all_run_intent_facts(
    runtime: _Runtime,
    migrated_database_url: str,
    table_name: str,
) -> None:
    tenant = await _personal_tenant(runtime, "rollback-run-user")
    function_name = f"fail_{table_name}_insert"
    trigger_name = f"fail_{table_name}_insert_trigger"
    with connect_database(migrated_database_url) as connection:
        connection.execute(
            f"""
            CREATE FUNCTION {function_name}() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'injected run intent failure';
            END;
            $$
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION {function_name}()
            """
        )

    with pytest.raises(DBAPIError):
        await runtime.run_service.create_research_run(
            tenant=tenant,
            query="This transaction must roll back",
        )

    assert await _business_counts(runtime.session_factory) == (0, 0, 0, 0, 0)


async def _seed_team_tenants(runtime: _Runtime) -> tuple[TenantContext, TenantContext, UUID]:
    owner = await runtime.provisioning.provision_personal_workspace("team-owner")
    second = await runtime.provisioning.provision_personal_workspace("team-admin")
    workspace_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add(
            Workspace(
                id=workspace_id,
                kind=WorkspaceKind.TEAM.value,
                name="Authorization fixture",
                created_by_user_id=owner.user_id,
            )
        )
        session.add_all(
            [
                WorkspaceMembership(
                    workspace_id=workspace_id,
                    user_id=owner.user_id,
                    role=WorkspaceRole.MEMBER.value,
                ),
                WorkspaceMembership(
                    workspace_id=workspace_id,
                    user_id=second.user_id,
                    role=WorkspaceRole.ADMIN.value,
                ),
            ]
        )
    owner_tenant = await runtime.tenant_service.resolve_tenant(
        workspace_id=workspace_id,
        actor_user_id=owner.user_id,
    )
    admin_tenant = await runtime.tenant_service.resolve_tenant(
        workspace_id=workspace_id,
        actor_user_id=second.user_id,
    )
    return owner_tenant, admin_tenant, second.user_id


async def test_workspace_read_and_cancel_rbac_are_enforced_in_store(runtime: _Runtime) -> None:
    owner_tenant, admin_tenant, admin_user_id = await _seed_team_tenants(runtime)
    owner_run = await runtime.run_service.create_research_run(
        tenant=owner_tenant,
        query="Owner cancellation",
    )
    owner_cancellation = await runtime.run_service.cancel_run(
        tenant=owner_tenant,
        run_id=owner_run.run_id,
    )
    assert owner_cancellation.status is RunStatus.CANCELLED

    accepted = await runtime.run_service.create_research_run(
        tenant=owner_tenant,
        query="Shared workspace research",
    )

    shared = await runtime.run_service.get_run(
        tenant=admin_tenant,
        run_id=accepted.run_id,
    )
    assert shared.run_id == accepted.run_id

    downgraded_admin = TenantContext(
        workspace_id=admin_tenant.workspace_id,
        actor_user_id=admin_user_id,
        role=WorkspaceRole.MEMBER,
    )
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == admin_tenant.workspace_id,
                WorkspaceMembership.user_id == admin_user_id,
            )
            .values(role=WorkspaceRole.MEMBER.value)
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.run_service.cancel_run(
            tenant=admin_tenant,
            run_id=accepted.run_id,
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.run_service.cancel_run(
            tenant=downgraded_admin,
            run_id=accepted.run_id,
        )

    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == admin_tenant.workspace_id,
                WorkspaceMembership.user_id == admin_user_id,
            )
            .values(role=WorkspaceRole.ADMIN.value)
        )
    refreshed_admin = await runtime.tenant_service.resolve_tenant(
        workspace_id=admin_tenant.workspace_id,
        actor_user_id=admin_user_id,
    )
    first, second = await asyncio.gather(
        runtime.run_service.cancel_run(tenant=refreshed_admin, run_id=accepted.run_id),
        runtime.run_service.cancel_run(tenant=refreshed_admin, run_id=accepted.run_id),
    )
    assert {first.status, second.status} == {RunStatus.CANCELLED}
    async with runtime.session_factory() as session:
        run = await session.get(Run, accepted.run_id)
        job = await session.scalar(select(RunJob).where(RunJob.run_id == accepted.run_id))
        events = list(
            await session.scalars(
                select(RunEvent).where(RunEvent.run_id == accepted.run_id).order_by(RunEvent.seq)
            )
        )
    assert run is not None and run.status == "cancelled" and run.finished_at is not None
    assert run.next_event_seq == 3
    assert job is not None and job.status == "done"
    assert [(event.seq, event.type) for event in events] == [
        (1, "run.created"),
        (2, "run.cancelled"),
    ]
    assert events[1].payload == {
        "previous_status": "queued",
        "status": "cancelled",
        "reason": "user_requested",
    }


async def test_reviewer_can_cancel_only_own_persisted_creator_run(runtime: _Runtime) -> None:
    owner_tenant, admin_tenant, reviewer_user_id = await _seed_team_tenants(runtime)
    owner_run = await runtime.run_service.create_research_run(
        tenant=owner_tenant,
        query="Owner run hidden from reviewer cancellation",
    )
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == admin_tenant.workspace_id,
                WorkspaceMembership.user_id == reviewer_user_id,
            )
            .values(role=WorkspaceRole.REVIEWER.value)
        )
    reviewer = await runtime.tenant_service.resolve_tenant(
        workspace_id=admin_tenant.workspace_id,
        actor_user_id=reviewer_user_id,
    )
    assert reviewer.role is WorkspaceRole.REVIEWER

    with pytest.raises(DomainNotFoundError):
        await runtime.run_service.cancel_run(tenant=reviewer, run_id=owner_run.run_id)

    reviewer_run = await runtime.run_service.create_research_run(
        tenant=reviewer,
        query="Reviewer-owned run",
    )
    cancellation = await runtime.run_service.cancel_run(
        tenant=reviewer,
        run_id=reviewer_run.run_id,
    )
    assert cancellation.status is RunStatus.CANCELLED


async def test_revoked_membership_is_hidden_before_run_read(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "revoked-run-reader")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Research")
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == tenant.workspace_id,
                WorkspaceMembership.user_id == tenant.actor_user_id,
            )
            .values(revoked_at=datetime.now(UTC))
        )

    with pytest.raises(DomainNotFoundError):
        await runtime.tenant_service.resolve_tenant(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.run_service.get_run(tenant=tenant, run_id=accepted.run_id)


async def test_running_cancel_is_idempotent_and_does_not_finalize_job(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "running-cancel-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Research")
    owner_token = uuid4()
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(status="running", started_at=datetime.now(UTC))
        )
        await session.execute(
            update(RunJob)
            .where(RunJob.run_id == accepted.run_id)
            .values(
                status="leased",
                attempt=1,
                leased_by="test-worker",
                owner_token=owner_token,
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=1),
            )
        )

    first = await runtime.run_service.cancel_run(tenant=tenant, run_id=accepted.run_id)
    second = await runtime.run_service.cancel_run(tenant=tenant, run_id=accepted.run_id)

    assert first.status is second.status is RunStatus.RUNNING
    assert first.cancel_requested_at == second.cancel_requested_at
    assert first.cancel_requested_at is not None
    async with runtime.session_factory() as session:
        job = await session.scalar(select(RunJob).where(RunJob.run_id == accepted.run_id))
        event_count = await session.scalar(
            select(func.count()).select_from(RunEvent).where(RunEvent.run_id == accepted.run_id)
        )
    assert job is not None and job.status == JobStatus.LEASED.value
    assert job.owner_token == owner_token
    assert event_count == 1


async def test_waiting_approval_cancel_fails_closed_in_gate4(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "waiting-cancel-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Research")
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(status="waiting_approval", started_at=datetime.now(UTC))
        )

    with pytest.raises(DomainInvariantError):
        await runtime.run_service.cancel_run(tenant=tenant, run_id=accepted.run_id)


def _qwen_invocation(
    *,
    tenant: TenantContext,
    run_id: UUID,
    kind: str,
    usage: dict[str, int],
    cost: Decimal | None,
) -> LLMInvocation:
    is_chat = kind == "chat"
    return LLMInvocation(
        id=uuid4(),
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
        run_id=run_id,
        invocation_kind=kind,
        provider="qwen",
        model="qwen3.6-flash-2026-04-16" if is_chat else "text-embedding-v4",
        graph_node="research_agent" if is_chat else "document_embedding",
        prompt_version=f"sha256:{'a' * 64}" if is_chat else None,
        request_hash=f"sha256:{uuid4().hex * 2}",
        provider_response_id=str(uuid4()),
        token_usage=usage,
        pricing_version="qwen-cn-beijing-cny-2026-08-13-v1" if cost is not None else None,
        currency="CNY" if cost is not None else None,
        estimated_cost=cost,
        trace_ids=None,
        latency_ms=10,
        status="succeeded",
        error_category=None,
    )


async def test_usage_and_cost_are_aggregated_only_for_requested_workspace_run(
    runtime: _Runtime,
) -> None:
    first = await _personal_tenant(runtime, "usage-first")
    second = await _personal_tenant(runtime, "usage-second")
    first_run = await runtime.run_service.create_research_run(tenant=first, query="First")
    second_run = await runtime.run_service.create_research_run(tenant=second, query="Second")
    async with transaction(runtime.session_factory) as session:
        session.add_all(
            [
                _qwen_invocation(
                    tenant=first,
                    run_id=first_run.run_id,
                    kind="chat",
                    usage={
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "cached_input_tokens": 2,
                    },
                    cost=Decimal("0.000100000000"),
                ),
                _qwen_invocation(
                    tenant=first,
                    run_id=first_run.run_id,
                    kind="chat",
                    usage={"input_tokens": 3, "output_tokens": 4},
                    cost=Decimal("0.000020000000"),
                ),
                _qwen_invocation(
                    tenant=first,
                    run_id=first_run.run_id,
                    kind="embedding",
                    usage={"input_tokens": 6, "output_tokens": 0},
                    cost=None,
                ),
                _qwen_invocation(
                    tenant=second,
                    run_id=second_run.run_id,
                    kind="chat",
                    usage={"input_tokens": 999, "output_tokens": 999},
                    cost=Decimal("0.999000000000"),
                ),
            ]
        )

    record = await runtime.run_service.get_run(tenant=first, run_id=first_run.run_id)

    assert record.usage.chat.attempt_count == 2
    assert record.usage.chat.succeeded_count == 2
    assert record.usage.chat.input_tokens == 13
    assert record.usage.chat.output_tokens == 24
    assert record.usage.chat.reasoning_output_tokens == 5
    assert record.usage.chat.cached_input_tokens == 2
    assert record.usage.chat.estimated_cost == Decimal("0.000120000000")
    assert record.usage.chat.currency == "CNY"
    assert record.usage.chat.cost_available is True
    assert record.usage.embedding.input_tokens == 6
    assert record.usage.embedding.estimated_cost is None
    assert record.usage.embedding.currency is None
    assert record.usage.embedding.cost_available is False


async def test_invalid_persisted_result_fails_closed(runtime: _Runtime) -> None:
    tenant = await _personal_tenant(runtime, "invalid-result-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Research")
    now = datetime.now(UTC)
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(Run)
            .where(Run.id == accepted.run_id)
            .values(
                status="completed",
                started_at=now,
                finished_at=now,
                result_json={"unexpected": "shape"},
            )
        )

    with pytest.raises(DomainInvariantError, match="persisted run result is invalid"):
        await runtime.run_service.get_run(tenant=tenant, run_id=accepted.run_id)


async def test_real_api_persists_fake_actor_intent_without_graph_execution(
    migrated_database_url: str,
) -> None:
    bootstrap_engine = create_database_engine(SecretStr(migrated_database_url))
    bootstrap_sessions = create_session_factory(bootstrap_engine)
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(bootstrap_sessions))
    identity = await provisioning.provision_personal_workspace(FAKE_ACTOR_SUBJECT)
    await bootstrap_engine.dispose()
    application = create_app(
        Settings(
            database_url=SecretStr(migrated_database_url),
            log_level="ERROR",
        )
    )

    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                f"/api/v1/workspaces/{identity.workspace_id}/runs",
                json={"mode": "research", "query": "API accepted intent"},
            )
            run_id = UUID(response.json()["run_id"])
            detail_response = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}"
            )
            cancel_response = await client.post(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}/cancel"
            )
            event_response = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}/events"
            )
            cancelled_detail_response = await client.get(
                f"/api/v1/workspaces/{identity.workspace_id}/runs/{run_id}"
            )

    assert response.status_code == 202
    assert detail_response.status_code == 200
    assert detail_response.json()["status"] == "queued"
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"
    assert event_response.status_code == 200
    assert "id: 1\nevent: run.created\n" in event_response.text
    assert "id: 2\nevent: run.cancelled\n" in event_response.text
    assert cancelled_detail_response.status_code == 200
    assert cancelled_detail_response.json()["status"] == "cancelled"
    with connect_database(migrated_database_url) as connection:
        facts = connection.execute(
            """
            SELECT r.status, j.status, count(e.id), max(e.seq), m.content
            FROM runs AS r
            JOIN run_jobs AS j ON j.run_id = r.id
            JOIN run_events AS e ON e.run_id = r.id
            JOIN messages AS m ON m.id = r.request_message_id
            WHERE r.id = %s
            GROUP BY r.status, j.status, m.content
            """,
            (run_id,),
        ).fetchone()
    assert facts == ("cancelled", "done", 2, 2, "API accepted intent")


async def test_event_reader_replays_old_and_current_cursor_and_rejects_future(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "event-replay-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Replay events")
    await runtime.run_service.cancel_run(tenant=tenant, run_id=accepted.run_id)
    reader = SqlAlchemyRunEventReader(runtime.session_factory)

    replay = await reader.read_after(
        tenant=tenant,
        run_id=accepted.run_id,
        after_seq=0,
        limit=100,
    )
    reconnect = await reader.read_after(
        tenant=tenant,
        run_id=accepted.run_id,
        after_seq=1,
        limit=100,
    )
    current = await reader.read_after(
        tenant=tenant,
        run_id=accepted.run_id,
        after_seq=2,
        limit=100,
    )

    assert [(event.seq, event.type) for event in replay.events] == [
        (1, RunEventType.RUN_CREATED),
        (2, RunEventType.RUN_CANCELLED),
    ]
    assert [event.seq for event in reconnect.events] == [2]
    assert current.events == ()
    assert current.high_watermark == 2
    assert current.run_is_terminal is True
    with pytest.raises(InvalidEventCursorError):
        await reader.read_after(
            tenant=tenant,
            run_id=accepted.run_id,
            after_seq=3,
            limit=100,
        )


async def test_event_reader_hides_cross_workspace_unknown_and_stale_tenant(
    runtime: _Runtime,
) -> None:
    owner = await _personal_tenant(runtime, "event-owner")
    outsider = await _personal_tenant(runtime, "event-outsider")
    accepted = await runtime.run_service.create_research_run(tenant=owner, query="Tenant boundary")
    reader = SqlAlchemyRunEventReader(runtime.session_factory)

    with pytest.raises(DomainNotFoundError):
        await reader.read_after(
            tenant=outsider,
            run_id=accepted.run_id,
            after_seq=0,
            limit=100,
        )
    with pytest.raises(DomainNotFoundError):
        await reader.read_after(
            tenant=owner,
            run_id=uuid4(),
            after_seq=0,
            limit=100,
        )

    other_workspace_id = uuid4()
    async with transaction(runtime.session_factory) as session:
        session.add(
            Workspace(
                id=other_workspace_id,
                kind=WorkspaceKind.TEAM.value,
                name="Same actor, different workspace",
                created_by_user_id=owner.actor_user_id,
            )
        )
        session.add(
            WorkspaceMembership(
                workspace_id=other_workspace_id,
                user_id=owner.actor_user_id,
                role=WorkspaceRole.MEMBER.value,
            )
        )
    other_tenant = TenantContext(
        workspace_id=other_workspace_id,
        actor_user_id=owner.actor_user_id,
        role=WorkspaceRole.MEMBER,
    )
    with pytest.raises(DomainNotFoundError):
        await reader.read_after(
            tenant=other_tenant,
            run_id=accepted.run_id,
            after_seq=0,
            limit=100,
        )

    first_page = await reader.read_after(
        tenant=owner,
        run_id=accepted.run_id,
        after_seq=0,
        limit=1,
    )
    assert [event.seq for event in first_page.events] == [1]
    async with transaction(runtime.session_factory) as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == owner.workspace_id,
                WorkspaceMembership.user_id == owner.actor_user_id,
            )
            .values(revoked_at=datetime.now(UTC))
        )
    with pytest.raises(DomainNotFoundError):
        await reader.read_after(
            tenant=owner,
            run_id=accepted.run_id,
            after_seq=first_page.events[-1].seq,
            limit=100,
        )


async def test_sse_follow_uses_short_reads_and_terminal_event_closes(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "event-follow-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Follow events")
    reader = SqlAlchemyRunEventReader(runtime.session_factory)
    initial = await reader.read_after(
        tenant=tenant,
        run_id=accepted.run_id,
        after_seq=0,
        limit=100,
    )

    async def connected() -> bool:
        return False

    stream = stream_run_events(
        reader=reader,
        tenant=tenant,
        run_id=accepted.run_id,
        cursor=0,
        initial_batch=initial,
        is_disconnected=connected,
        poll_interval_seconds=0,
        heartbeat_interval_seconds=60,
    )
    assert (await anext(stream)).startswith("id: 1\nevent: run.created\n")
    engine = runtime.session_factory.kw["bind"]
    assert engine.sync_engine.pool.checkedout() == 0

    # The client can pause here while another short transaction acquires a connection and writes.
    await runtime.run_service.cancel_run(tenant=tenant, run_id=accepted.run_id)
    assert (await anext(stream)).startswith("id: 2\nevent: run.cancelled\n")
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_sse_heartbeat_is_transport_only_and_releases_database_resources(
    runtime: _Runtime,
) -> None:
    tenant = await _personal_tenant(runtime, "event-heartbeat-user")
    accepted = await runtime.run_service.create_research_run(tenant=tenant, query="Stay open")
    reader = SqlAlchemyRunEventReader(runtime.session_factory)
    initial = await reader.read_after(
        tenant=tenant,
        run_id=accepted.run_id,
        after_seq=1,
        limit=100,
    )

    async def connected() -> bool:
        return False

    stream = stream_run_events(
        reader=reader,
        tenant=tenant,
        run_id=accepted.run_id,
        cursor=1,
        initial_batch=initial,
        is_disconnected=connected,
        poll_interval_seconds=0,
        heartbeat_interval_seconds=0,
    )
    assert await anext(stream) == ": keepalive\n\n"
    engine = runtime.session_factory.kw["bind"]
    assert engine.sync_engine.pool.checkedout() == 0

    # A slow consumer is suspended after yield; the reader's session is already closed.
    async with runtime.session_factory() as session:
        event_count = await session.scalar(
            select(func.count()).select_from(RunEvent).where(RunEvent.run_id == accepted.run_id)
        )
    assert event_count == 1
    await stream.aclose()
