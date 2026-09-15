"""Owned E7-A generation database. No arbitrary DSN, production root or checkpoint.

The handle is an in-process capability for accidental-misuse prevention, not an OS
sandbox. Only the creator may dispose its container. Evidence files are never removed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from alembic import command
from alembic.config import Config
from pydantic import SecretStr
from sqlalchemy import select, text

from app.db.models import Conversation, Document, Message, Run, RunEvent, RunJob
from app.db.readiness import REQUIRED_DATABASE_REVISION
from app.db.runs import _require_current_tenant
from app.db.session import create_database_engine, create_session_factory, transaction
from app.domain.errors import DomainNotFoundError
from app.domain.research import ResearchOutputV2, ResearchRequestV1
from app.domain.runs import CURRENT_GRAPH_VERSION, DEFAULT_RUN_LIMITS
from tests.evals.quality_e7a_contracts import EvidenceContext, parse_output
from tests.evals.quality_e7a_graph import GRAPH_VERSION

POSTGRES_IMAGE = "pgvector/pgvector:0.8.5-pg16"
FIXTURE_VERSION = "e7a-generation-database-v1"
BASE_REVISION = "0015_e3_run_request_identity"
GRAPH_CHECK = "ck_runs_graph_version"
BASE_GRAPHS = tuple(f"pathfinder-research-v{x}" for x in range(1, 7))
LABEL = "pathfinder.e7a.database-owner"
ROOT = Path(__file__).resolve().parents[2]
ERRORS = frozenset(
    {
        "invalid_handle",
        "ownership_mismatch",
        "unsafe_binding",
        "schema_drift",
        "startup_failed",
        "cleanup_failed",
        "resource_busy",
        "invalid_run",
        "invalid_result",
        "access_denied",
        "persistence_failed",
        "invalid_configuration",
    }
)


class ExperimentDatabaseError(Exception):
    def __init__(self, category):
        self.category = category if category in ERRORS else "invalid_configuration"
        super().__init__(self.category)


def _fail(category):
    raise ExperimentDatabaseError(category) from None


def _digest(value):
    return (
        "sha256:"
        + sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


async def schema_snapshot(session):
    """Stable definitions only: no OIDs, credentials, rows or business content."""
    revision = tuple(await session.scalars(text("SELECT version_num FROM alembic_version")))
    constraints = (
        await session.execute(
            text("""
        SELECT c.relname, k.conname, pg_get_constraintdef(k.oid), k.convalidated
        FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public' ORDER BY c.relname,k.conname
    """)
        )
    ).all()
    columns = (
        await session.execute(
            text("""
        SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default
        FROM information_schema.columns WHERE table_schema='public'
        ORDER BY table_name,ordinal_position
    """)
        )
    ).all()
    indexes = (
        await session.execute(
            text("""
        SELECT tablename,indexname,indexdef FROM pg_indexes
        WHERE schemaname='public' ORDER BY tablename,indexname
    """)
        )
    ).all()
    return dict(
        revision=revision,
        constraints=[tuple(r) for r in constraints],
        columns=[tuple(r) for r in columns],
        indexes=[tuple(r) for r in indexes],
    )


def _without_graph(snapshot):
    return {
        **snapshot,
        "constraints": [r for r in snapshot["constraints"] if r[:2] != ("runs", GRAPH_CHECK)],
    }


def validate_graph_check(snapshot, versions):
    rows = [r for r in snapshot["constraints"] if r[:2] == ("runs", GRAPH_CHECK)]
    if len(rows) != 1 or not rows[0][3] or snapshot["revision"] != (BASE_REVISION,):
        _fail("schema_drift")
    # Compare the complete PostgreSQL-normalized expression, not just extracted literals.
    expected = (
        "CHECK ((graph_version = ANY (ARRAY["
        + ", ".join(f"'{version}'::text" for version in versions)
        + "])))"
    )
    if rows[0][2] != expected:
        _fail("schema_drift")


class OwnedDatabaseHandle:
    __slots__ = ("_owner", "_pid")

    def __init__(self):
        _fail("invalid_handle")

    def __repr__(self):
        return "<OwnedDatabaseHandle>"

    def __reduce__(self):
        _fail("invalid_handle")


@dataclass(frozen=True, slots=True, repr=False)
class ExperimentRun:
    run_id: UUID
    workspace_id: UUID
    actor_user_id: UUID
    conversation_id: UUID
    resume_document_id: UUID | None
    graph_version: str
    mode: str
    request: ResearchRequestV1


_OWNERS: dict[int, object] = {}


def require_owned_database(handle):
    if type(handle) is not OwnedDatabaseHandle:
        _fail("invalid_handle")
    owner = _OWNERS.get(id(handle))
    if (
        owner is None
        or getattr(handle, "_owner", None) is not owner
        or getattr(handle, "_pid", None) != os.getpid()
        or owner._handle is not handle
        or owner._closed
        or owner._engine is None
    ):
        _fail("invalid_handle")
    return owner


class OwnedExperimentDatabase:
    """Explicit async context creates a fresh container; construction/import does no I/O."""

    def __init__(self):
        self._owner = uuid4().hex
        self._name = f"pf-e7a-{self._owner}"
        self._database = f"pathfinder_test_{self._owner}"
        self._container = self._engine = self._sessions = self._handle = None
        self._container_id = None
        self._closed = False
        self._schema = None
        self._runs = {}
        self._active_slot = None
        self.cleanup_failed = False

    def __repr__(self):
        return "<OwnedExperimentDatabase>"

    def _verify_container(self, *, bindings=True):
        if self._container is None:
            _fail("ownership_mismatch")
        wrapped = self._container.get_wrapped_container()
        wrapped.reload()
        attrs = wrapped.attrs
        if (
            attrs.get("Config", {}).get("Labels", {}).get(LABEL) != self._owner
            or wrapped.name != self._name
            or (self._container_id is not None and wrapped.id != self._container_id)
        ):
            _fail("ownership_mismatch")
        if bindings:
            host = attrs.get("HostConfig", {})
            ports = attrs.get("NetworkSettings", {}).get("Ports", {}).get("5432/tcp")
            if (
                not attrs.get("State", {}).get("Running")
                or not isinstance(ports, list)
                or len(ports) != 1
                or ports[0].get("HostIp") != "127.0.0.1"
                or host.get("NetworkMode") != "bridge"
                or host.get("Privileged")
                or host.get("Binds")
                or host.get("NanoCpus") != 2_000_000_000
                or host.get("Memory") != 2_147_483_648
            ):
                _fail("unsafe_binding")
            if not re.fullmatch(r"[0-9]{1,5}", ports[0].get("HostPort", "")):
                _fail("unsafe_binding")
            port = int(ports[0]["HostPort"])
            if not 0 < port <= 65535:
                _fail("unsafe_binding")
            return port
        return None

    async def __aenter__(self):
        if self._container is not None or self._closed:
            _fail("invalid_handle")
        try:
            from testcontainers.community.postgres import PostgresContainer
            from testcontainers.core.container import DockerContainer

            password = uuid4().hex
            self._container = PostgresContainer(
                POSTGRES_IMAGE,
                username="pf_e7a",
                password=password,
                dbname=self._database,
                driver="psycopg",
                name=self._name,
                docker_client_kw={"timeout": 5},
                network_mode="bridge",
                labels={LABEL: self._owner},
                nano_cpus=2_000_000_000,
                mem_limit=2_147_483_648,
                pids_limit=128,
            )
            self._container.ports = {"5432/tcp": ("127.0.0.1", 0)}
            DockerContainer.start(self._container)
            port = self._verify_container()
            self._container_id = self._container.get_wrapped_container().id
            self._container._connect()
            url = f"postgresql+psycopg://pf_e7a:{password}@127.0.0.1:{port}/{self._database}"
            config = Config(str(ROOT / "alembic.ini"))
            config.attributes["database_url"] = url
            if REQUIRED_DATABASE_REVISION != BASE_REVISION:
                _fail("schema_drift")
            command.upgrade(config, "head")
            self._engine = create_database_engine(SecretStr(url))
            self._sessions = create_session_factory(self._engine)
            async with transaction(self._sessions) as session:
                before = await schema_snapshot(session)
                validate_graph_check(before, BASE_GRAPHS)
                await session.execute(text(f"ALTER TABLE runs DROP CONSTRAINT {GRAPH_CHECK}"))
                values = ", ".join(f"'{v}'" for v in (*BASE_GRAPHS, GRAPH_VERSION))
                await session.execute(
                    text(
                        f"ALTER TABLE runs ADD CONSTRAINT {GRAPH_CHECK} "
                        f"CHECK (graph_version IN ({values}))"
                    )
                )
                after = await schema_snapshot(session)
                validate_graph_check(after, (*BASE_GRAPHS, GRAPH_VERSION))
                if _without_graph(before) != _without_graph(after):
                    _fail("schema_drift")
                self._schema = _digest(after)
            handle = object.__new__(OwnedDatabaseHandle)
            handle._owner, handle._pid = self, os.getpid()
            self._handle = handle
            _OWNERS[id(handle)] = self
            return handle
        except asyncio.CancelledError:
            await self.aclose()
            raise
        except Exception:
            await self.aclose()
            _fail("startup_failed")

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self):
        if self._handle is not None:
            _OWNERS.pop(id(self._handle), None)
        self._closed = True
        errors = False
        if self._engine is not None:
            try:
                await self._engine.dispose()
            except Exception:
                errors = True
        if self._container is not None:
            try:
                # Attribute failure before Docker creation means there is nothing to remove.
                wrapped = self._container.get_wrapped_container()
                if wrapped is not None:
                    self._verify_container(bindings=False)
                    self._container.stop()
                    self._container = None
            except Exception:
                errors = True
        self.cleanup_failed = errors
        if errors:
            _fail("cleanup_failed")

    @property
    def identity(self):
        return dict(
            fixture_version=FIXTURE_VERSION,
            schema_digest=self._schema,
            base_revision=BASE_REVISION,
            graph_versions=(*BASE_GRAPHS, GRAPH_VERSION),
            production_schema=False,
        )

    async def verify(self):
        require_owned_database(self._handle)
        try:
            self._verify_container()
            async with self._sessions() as session:
                name = await session.scalar(text("SELECT current_database()"))
                snapshot = await schema_snapshot(session)
            if name != self._database or _digest(snapshot) != self._schema:
                _fail("schema_drift")
        except ExperimentDatabaseError:
            raise
        except Exception:
            _fail("schema_drift")

    async def authorize(self, tenant):
        try:
            async with self._sessions() as session:
                await _require_current_tenant(session, tenant, lock=False)
        except DomainNotFoundError:
            _fail("access_denied")
        except Exception:
            _fail("persistence_failed")

    def claim_slot(self, token):
        require_owned_database(self._handle)
        if self._active_slot is not None:
            _fail("resource_busy")
        self._active_slot = token

    def release_slot(self, token):
        if self._active_slot is not token:
            _fail("resource_busy")
        self._active_slot = None

    async def create_run(self, tenant, payload, document_id, timeout, *, arm):
        await self.verify()
        if arm not in {"baseline", "candidate"} or not 0 < timeout <= 600:
            _fail("invalid_configuration")
        request = ResearchRequestV1(
            query=payload.query, include_application_draft=payload.mode == "application"
        )
        fixture = ExperimentRun(
            uuid4(),
            tenant.workspace_id,
            tenant.actor_user_id,
            uuid4(),
            document_id,
            GRAPH_VERSION if arm == "candidate" else CURRENT_GRAPH_VERSION,
            payload.mode,
            request,
        )
        now, message_id = datetime.now(UTC), uuid4()
        try:
            async with transaction(self._sessions) as session:
                await _require_current_tenant(session, tenant, lock=True)
                if (
                    document_id is not None
                    and await session.scalar(
                        select(Document.id).where(
                            Document.id == document_id, Document.workspace_id == tenant.workspace_id
                        )
                    )
                    is None
                ):
                    _fail("access_denied")
                session.add(
                    Conversation(
                        id=fixture.conversation_id,
                        workspace_id=tenant.workspace_id,
                        created_by_user_id=tenant.actor_user_id,
                        title="Research request",
                    )
                )
                session.add(
                    Message(
                        id=message_id,
                        workspace_id=tenant.workspace_id,
                        conversation_id=fixture.conversation_id,
                        actor_user_id=tenant.actor_user_id,
                        role="user",
                        content=request.query,
                    )
                )
                session.add(
                    Run(
                        id=fixture.run_id,
                        workspace_id=tenant.workspace_id,
                        created_by_user_id=tenant.actor_user_id,
                        conversation_id=fixture.conversation_id,
                        request_message_id=message_id,
                        mode=payload.mode,
                        resume_document_id=document_id,
                        input_json=request.model_dump(mode="json"),
                        limits_json=dict(DEFAULT_RUN_LIMITS),
                        graph_version=fixture.graph_version,
                        status="running",
                        started_at=now,
                        next_event_seq=2,
                    )
                )
                session.add(
                    RunJob(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        originating_actor_user_id=tenant.actor_user_id,
                        run_id=fixture.run_id,
                        status="leased",
                        attempt=1,
                        leased_by="quality-generation-harness",
                        owner_token=uuid4(),
                        lease_expires_at=now + timedelta(seconds=timeout + 60),
                    )
                )
                await session.flush()
                session.add(
                    RunEvent(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        run_id=fixture.run_id,
                        actor_user_id=tenant.actor_user_id,
                        seq=1,
                        type="run.created",
                        version=1,
                        payload={
                            "mode": payload.mode,
                            "status": "queued",
                            "graph_version": fixture.graph_version,
                        },
                    )
                )
            self._runs[fixture.run_id] = fixture
            return fixture
        except DomainNotFoundError:
            _fail("access_denied")
        except ExperimentDatabaseError:
            raise
        except Exception:
            _fail("persistence_failed")

    async def _row(self, session, tenant, fixture, *, lock=False, check_membership=True):
        if (
            type(fixture) is not ExperimentRun
            or self._runs.get(fixture.run_id) is not fixture
            or tenant.workspace_id != fixture.workspace_id
            or tenant.actor_user_id != fixture.actor_user_id
        ):
            _fail("access_denied")
        if check_membership:
            await _require_current_tenant(session, tenant, lock=lock)
        query = select(Run).where(Run.id == fixture.run_id, Run.workspace_id == tenant.workspace_id)
        row = await session.scalar(query.with_for_update() if lock else query)
        if (
            row is None
            or row.graph_version != fixture.graph_version
            or row.input_json != fixture.request.model_dump(mode="json")
            or row.resume_document_id != fixture.resume_document_id
            or row.created_by_user_id != fixture.actor_user_id
            or row.conversation_id != fixture.conversation_id
            or row.mode != fixture.mode
            or row.limits_json != dict(DEFAULT_RUN_LIMITS)
        ):
            _fail("invalid_run")
        if (
            fixture.resume_document_id is not None
            and await session.scalar(
                select(Document.id).where(
                    Document.id == fixture.resume_document_id,
                    Document.workspace_id == tenant.workspace_id,
                )
            )
            is None
        ):
            _fail("access_denied")
        return row

    def _parse(self, fixture, raw, context):
        if fixture.graph_version == GRAPH_VERSION:
            if (
                type(context) is not EvidenceContext
                or context.request != fixture.request
                or context.resume_document_id != fixture.resume_document_id
            ):
                _fail("invalid_result")
            return parse_output(raw, context)
        if fixture.graph_version != CURRENT_GRAPH_VERSION:
            _fail("invalid_result")
        return ResearchOutputV2.model_validate_json(raw, strict=True)

    async def finish_run(self, tenant, fixture, output, failure, *, context=None):
        """Single terminal transition. Re-entry verifies equality instead of overwriting facts."""
        await self.verify()
        try:
            data = None if output is None else json.loads(output.model_dump_json())
            if data is not None:
                self._parse(fixture, json.dumps(data, allow_nan=False), context)
            status = "cancelled" if failure == "cancelled" else "failed" if failure else "completed"
            if status == "completed" and data is None:
                _fail("invalid_result")
            async with transaction(self._sessions) as session:
                row = await self._row(
                    session,
                    tenant,
                    fixture,
                    lock=True,
                    check_membership=not (failure and data is None),
                )
                job = await session.scalar(
                    select(RunJob)
                    .where(
                        RunJob.run_id == fixture.run_id, RunJob.workspace_id == tenant.workspace_id
                    )
                    .with_for_update()
                )
                if job is None:
                    _fail("invalid_run")
                if row.status in {"completed", "failed", "cancelled"}:
                    if row.status != status or row.result_json != data or job.status != "done":
                        _fail("invalid_run")
                    return
                if row.status != "running" or job.status != "leased":
                    _fail("invalid_run")
                row.status, row.result_json, row.finished_at = status, data, datetime.now(UTC)
                row.error_category = "graph_execution_failed" if status == "failed" else None
                job.status, job.leased_by, job.owner_token, job.lease_expires_at = (
                    "done",
                    None,
                    None,
                    None,
                )
        except DomainNotFoundError:
            _fail("access_denied")
        except ExperimentDatabaseError:
            raise
        except Exception:
            _fail("persistence_failed")

    async def read_result(self, tenant, fixture, *, context=None):
        await self.verify()
        try:
            async with self._sessions() as session:
                row = await self._row(session, tenant, fixture)
                if (
                    row.status not in {"completed", "failed", "cancelled"}
                    or row.result_json is None
                ):
                    _fail("invalid_result")
                raw = json.dumps(row.result_json, allow_nan=False)
            return self._parse(fixture, raw, context)
        except DomainNotFoundError:
            _fail("access_denied")
        except ExperimentDatabaseError:
            raise
        except Exception:
            _fail("invalid_result")
