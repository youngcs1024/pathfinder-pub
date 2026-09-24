"""R1.2 command acceptance through a test-only business writer and PostgreSQL."""

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import httpx
import pytest
import sqlalchemy as sa
from alembic import command
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import func, select, text

from app.api.errors import install_exception_handlers
from app.api.run_request_identity import require_idempotency_key
from app.db.models import Conversation, Message, ResumeCommand, Run, RunEvent, RunJob
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_commands import SqlAlchemyResumeCommandStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainConflictError, DomainInvariantError, DomainNotFoundError
from app.domain.provisioning import ProvisioningService
from app.domain.resume_commands import (
    CommandReceiptV1,
    ResumeCommandRequest,
    SessionWritePrecondition,
    SessionWriteState,
)
from app.domain.tenancy import TenantContext, TenantService
from tests.integration.support import alembic_config

pytestmark = pytest.mark.integration


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str


@dataclass
class _Runtime:
    engine: object
    session_factory: object
    provisioning: ProvisioningService
    tenancy: TenantService
    commands: SqlAlchemyResumeCommandStore


@pytest.fixture
async def runtime(migrated_database_url: str):
    engine = create_database_engine(SecretStr(migrated_database_url))
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE command_test_sessions ("
                "id uuid PRIMARY KEY, workspace_id uuid NOT NULL, actor_user_id uuid NOT NULL, "
                "revision integer NOT NULL, current_version_id uuid, active boolean NOT NULL)"
            )
        )
        await connection.execute(
            text(
                "CREATE TABLE command_test_intents ("
                "command_id uuid PRIMARY KEY, workspace_id uuid NOT NULL, payload text NOT NULL)"
            )
        )
    session_factory = create_session_factory(engine)
    try:
        yield _Runtime(
            engine=engine,
            session_factory=session_factory,
            provisioning=ProvisioningService(SqlAlchemyProvisioningStore(session_factory)),
            tenancy=TenantService(SqlAlchemyTenantResolver(session_factory)),
            commands=SqlAlchemyResumeCommandStore(session_factory),
        )
    finally:
        await engine.dispose()


async def _tenant(runtime: _Runtime, subject: str) -> TenantContext:
    identity = await runtime.provisioning.provision_personal_workspace(subject)
    return await runtime.tenancy.resolve_tenant(
        workspace_id=identity.workspace_id, actor_user_id=identity.user_id
    )


async def _session(runtime: _Runtime, tenant: TenantContext) -> tuple[UUID, UUID]:
    session_id, version_id = uuid4(), uuid4()
    async with runtime.engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO command_test_sessions "
                "(id, workspace_id, actor_user_id, revision, current_version_id, active) "
                "VALUES (:id, :workspace, :actor, 4, :version, false)"
            ),
            {
                "id": session_id,
                "workspace": tenant.workspace_id,
                "actor": tenant.actor_user_id,
                "version": version_id,
            },
        )
    return session_id, version_id


def _request(
    *,
    key: UUID | None = None,
    text_value: str = "Synthetic revision",
    session_id: UUID | None = None,
    version_id: UUID | None = None,
    revision: int = 4,
) -> ResumeCommandRequest:
    return ResumeCommandRequest(
        client_request_id=key or uuid4(),
        kind="resume_revision" if session_id is not None else "material_import",
        target_id=session_id,
        payload_version=1,
        payload=_Payload(text=text_value),
        session_write=(
            SessionWritePrecondition(revision, version_id) if session_id is not None else None
        ),
    )


class _Writer:
    supported_kinds = frozenset({"material_import", "resume_revision"})

    def __init__(self, *, fail_after_intent: bool = False, skip_job: bool = False) -> None:
        self.fail_after_intent = fail_after_intent
        self.skip_job = skip_job
        self.apply_count = 0

    async def authorize_and_lock(self, session, tenant, request):
        if request.target_id is None:
            return None
        row = (
            await session.execute(
                text(
                    "SELECT revision, current_version_id, active FROM command_test_sessions "
                    "WHERE id = :id AND workspace_id = :workspace AND actor_user_id = :actor "
                    "FOR UPDATE"
                ),
                {
                    "id": request.target_id,
                    "workspace": tenant.workspace_id,
                    "actor": tenant.actor_user_id,
                },
            )
        ).one_or_none()
        if row is None:
            raise DomainNotFoundError
        return SessionWriteState(row.revision, row.current_version_id, row.active)

    async def apply(self, session, tenant, request, command_id):
        self.apply_count += 1
        await session.execute(
            text(
                "INSERT INTO command_test_intents (command_id, workspace_id, payload) "
                "VALUES (:command_id, :workspace_id, :payload)"
            ),
            {
                "command_id": command_id,
                "workspace_id": tenant.workspace_id,
                "payload": request.payload.text,
            },
        )
        conversation_id, message_id, run_id = uuid4(), uuid4(), uuid4()
        session.add(
            Conversation(
                id=conversation_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                title="Synthetic command acceptance",
            )
        )
        await session.flush()
        session.add(
            Message(
                id=message_id,
                workspace_id=tenant.workspace_id,
                conversation_id=conversation_id,
                actor_user_id=tenant.actor_user_id,
                role="user",
                content=request.payload.text,
            )
        )
        await session.flush()
        session.add(
            Run(
                id=run_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                conversation_id=conversation_id,
                request_message_id=message_id,
                mode="research",
                input_json={"schema_version": 1, "query": request.payload.text},
                limits_json={"schema_version": 1},
                status="queued",
                graph_version="pathfinder-research-v6",
                next_event_seq=2,
            )
        )
        await session.flush()
        if not self.skip_job:
            session.add(
                RunJob(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    originating_actor_user_id=tenant.actor_user_id,
                    run_id=run_id,
                    status="queued",
                )
            )
        session.add(
            RunEvent(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                run_id=run_id,
                actor_user_id=tenant.actor_user_id,
                seq=1,
                type="run.created",
                version=1,
                payload={"mode": "research", "status": "queued"},
            )
        )
        if request.target_id is not None:
            await session.execute(
                text(
                    "UPDATE command_test_sessions SET revision = revision + 1, active = true "
                    "WHERE id = :id"
                ),
                {"id": request.target_id},
            )
        if self.fail_after_intent:
            raise RuntimeError("synthetic failure before commit")
        return CommandReceiptV1(command_id=command_id, run_id=run_id, status="queued")


class _SynchronousWriter(_Writer):
    async def apply(self, session, tenant, request, command_id):
        self.apply_count += 1
        await session.execute(
            text(
                "INSERT INTO command_test_intents (command_id, workspace_id, payload) "
                "VALUES (:command_id, :workspace_id, :payload)"
            ),
            {
                "command_id": command_id,
                "workspace_id": tenant.workspace_id,
                "payload": request.payload.text,
            },
        )
        return CommandReceiptV1(command_id=command_id, status="completed")


async def _counts(runtime: _Runtime) -> tuple[int, ...]:
    async with runtime.session_factory() as session:
        counts = [(await session.scalar(text("SELECT count(*) FROM command_test_intents"))) or 0]
        for model in (ResumeCommand, Conversation, Message, Run, RunJob, RunEvent):
            counts.append((await session.scalar(select(func.count()).select_from(model))) or 0)
        return tuple(counts)


async def test_replay_keeps_one_intent_run_job_event_and_original_receipt(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-replay")
    session_id, version_id = await _session(runtime, tenant)
    request = _request(session_id=session_id, version_id=version_id)
    writer = _Writer()
    first = await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    replay = await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    assert not first.replayed and replay.replayed
    assert replay.receipt == first.receipt
    assert writer.apply_count == 1
    assert await _counts(runtime) == (1, 1, 1, 1, 1, 1, 1)
    async with runtime.engine.connect() as connection:
        row = (
            await connection.execute(
                text("SELECT revision, active FROM command_test_sessions WHERE id=:id"),
                {"id": session_id},
            )
        ).one()
    assert row == (5, True)


async def test_key_collision_stale_write_and_active_task_leave_no_extra_intent(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-conflicts")
    session_id, version_id = await _session(runtime, tenant)
    key = uuid4()
    writer = _Writer()
    await runtime.commands.accept(
        tenant=tenant,
        request=_request(key=key, session_id=session_id, version_id=version_id),
        writer=writer,
    )
    for request in (
        _request(key=key, session_id=session_id, version_id=version_id, text_value="Different"),
        _request(session_id=session_id, version_id=version_id),
        _request(session_id=session_id, version_id=version_id, revision=5),
    ):
        with pytest.raises(DomainConflictError):
            await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    assert await _counts(runtime) == (1, 1, 1, 1, 1, 1, 1)


async def test_base_mismatch_and_rollback_do_not_accept_command(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-rollback")
    session_id, version_id = await _session(runtime, tenant)
    with pytest.raises(DomainConflictError):
        await runtime.commands.accept(
            tenant=tenant,
            request=_request(session_id=session_id, version_id=uuid4()),
            writer=_Writer(),
        )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        await runtime.commands.accept(
            tenant=tenant,
            request=_request(session_id=session_id, version_id=version_id),
            writer=_Writer(fail_after_intent=True),
        )
    with pytest.raises(DomainInvariantError, match="did not enqueue"):
        await runtime.commands.accept(
            tenant=tenant,
            request=_request(session_id=session_id, version_id=version_id),
            writer=_Writer(skip_job=True),
        )
    assert await _counts(runtime) == (0, 0, 0, 0, 0, 0, 0)
    async with runtime.engine.connect() as connection:
        assert (
            await connection.scalar(
                text("SELECT revision FROM command_test_sessions WHERE id=:id"), {"id": session_id}
            )
            == 4
        )


async def test_synchronous_command_accepts_without_job(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-synchronous")
    request = _request()
    writer = _SynchronousWriter()
    first = await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    replay = await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    assert first.receipt.status == "completed"
    assert first.receipt.run_id is None
    assert replay.receipt == first.receipt and replay.replayed
    assert writer.apply_count == 1
    assert await _counts(runtime) == (1, 1, 0, 0, 0, 0, 0)


async def test_revoked_replay_and_foreign_target_fail_before_receipt(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-revoked")
    foreign = await _tenant(runtime, "r12-foreign")
    session_id, version_id = await _session(runtime, tenant)
    request = _request(session_id=session_id, version_id=version_id)
    writer = _Writer()
    await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    with pytest.raises(DomainNotFoundError):
        await runtime.commands.accept(tenant=foreign, request=request, writer=writer)
    async with runtime.engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE workspace_memberships SET revoked_at = now() "
                "WHERE workspace_id=:workspace AND user_id=:actor"
            ),
            {"workspace": tenant.workspace_id, "actor": tenant.actor_user_id},
        )
    with pytest.raises(DomainNotFoundError):
        await runtime.commands.accept(tenant=tenant, request=request, writer=writer)
    assert await _counts(runtime) == (1, 1, 1, 1, 1, 1, 1)


async def test_parallel_same_key_converges_and_different_payload_conflicts(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-race")
    key = uuid4()
    writer = _Writer()
    same = _request(key=key)
    start = asyncio.Event()

    async def accept(request):
        await start.wait()
        return await runtime.commands.accept(tenant=tenant, request=request, writer=writer)

    tasks = [asyncio.create_task(accept(same)) for _ in range(6)]
    start.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    accepted = [item for item in results if not isinstance(item, Exception)]
    assert len(accepted) == 6
    assert not any(isinstance(item, Exception) for item in results)
    assert len({item.receipt.command_id for item in accepted}) == 1
    assert sum(not item.replayed for item in accepted) == 1
    with pytest.raises(DomainConflictError):
        await runtime.commands.accept(
            tenant=tenant,
            request=_request(key=key, text_value="Different"),
            writer=writer,
        )
    assert await _counts(runtime) == (1, 1, 1, 1, 1, 1, 1)


class _DropFirstResponse(httpx.AsyncBaseTransport):
    def __init__(self, app):
        self.inner = httpx.ASGITransport(app=app)
        self.dropped = False

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        if not self.dropped and response.status_code == 202:
            await response.aread()
            await response.aclose()
            self.dropped = True
            raise httpx.ReadError("synthetic response loss", request=request)
        return response

    async def aclose(self):
        await self.inner.aclose()


async def test_test_only_http_missing_key_and_lost_response(runtime: _Runtime):
    tenant = await _tenant(runtime, "r12-http")
    writer = _Writer()
    app = FastAPI()
    install_exception_handlers(app)

    @app.post("/test-only/commands")
    async def accept_http(request: Request):
        key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
        body = _Payload.model_validate(await request.json())
        accepted = await runtime.commands.accept(
            tenant=tenant,
            request=_request(key=key, text_value=body.text),
            writer=writer,
        )
        return JSONResponse(
            status_code=202,
            content=accepted.receipt.model_dump(mode="json"),
            headers={"Idempotency-Replayed": str(accepted.replayed).lower()},
        )

    transport = _DropFirstResponse(app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        missing = await client.post("/test-only/commands", json={"text": "Synthetic"})
        assert missing.status_code == 400
        assert missing.json()["type"] == "urn:pathfinder:problem:invalid-idempotency-key"
        key = str(uuid4())
        with pytest.raises(httpx.ReadError, match="synthetic response loss"):
            await client.post(
                "/test-only/commands",
                headers={"Idempotency-Key": key},
                json={"text": "Synthetic"},
            )
        replay = await client.post(
            "/test-only/commands",
            headers={"Idempotency-Key": key},
            json={"text": "Synthetic"},
        )
    assert replay.status_code == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert writer.apply_count == 1
    assert await _counts(runtime) == (1, 1, 1, 1, 1, 1, 1)


def test_0017_empty_downgrade_and_nonempty_guard(database_url: str):
    config = alembic_config(database_url)
    command.upgrade(config, "0016_r1_execution_contracts")
    command.upgrade(config, "head")
    engine = sa.create_engine(database_url)
    try:
        command.downgrade(config, "0016_r1_execution_contracts")
        assert "resume_commands" not in sa.inspect(engine).get_table_names()
        command.upgrade(config, "head")
        with engine.begin() as connection:
            actor = uuid4()
            workspace = uuid4()
            connection.execute(
                text("INSERT INTO users (id, auth_subject) VALUES (:id, 'r12-migration')"),
                {"id": actor},
            )
            connection.execute(
                text(
                    "INSERT INTO workspaces (id, kind, name, created_by_user_id) "
                    "VALUES (:id, 'personal', 'r12', :actor)"
                ),
                {"id": workspace, "actor": actor},
            )
            connection.execute(
                text(
                    "INSERT INTO workspace_memberships (workspace_id, user_id, role) "
                    "VALUES (:workspace, :actor, 'admin')"
                ),
                {"workspace": workspace, "actor": actor},
            )
            command_id = uuid4()
            connection.execute(
                text(
                    "INSERT INTO resume_commands "
                    "(id, workspace_id, actor_user_id, client_request_id, kind, "
                    "digest_version, request_digest, receipt_version, receipt_json) "
                    "VALUES (:id, :workspace, :actor, :key, 'material_import', 1, :digest, "
                    "1, CAST(:receipt AS jsonb))"
                ),
                {
                    "id": command_id,
                    "workspace": workspace,
                    "actor": actor,
                    "key": uuid4(),
                    "digest": "a" * 64,
                    "receipt": (
                        '{"command_id":"' + str(command_id) + '","run_id":null,'
                        '"resource_id":null,"status":"completed"}'
                    ),
                },
            )
        with pytest.raises(RuntimeError, match="cannot be safely downgraded"):
            command.downgrade(config, "0016_r1_execution_contracts")
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                "0024_r51_resume_revision"
            )
            assert connection.scalar(text("SELECT count(*) FROM resume_commands")) == 1
    finally:
        engine.dispose()
