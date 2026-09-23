"""R2.1 synthetic material imports through PostgreSQL and the single worker."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select, update

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import Document, MaterialSnapshotFile, Run, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainConflictError, DomainNotFoundError
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantContext, TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.material.aliases import MaterialAlias, MaterialAliasRegistry
from app.material.reader import read_alias
from app.retrieval.documents import DocumentIngestionService
from app.worker.backoff import ExponentialBackoff
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.material_executor import MaterialRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings

pytestmark = pytest.mark.integration


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture_aliases(tmp_path: Path, workspace_id):
    root = tmp_path / "code"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "main.py").write_text("print('synthetic')\n", encoding="utf-8")
    (root / "shared.md").write_text("# Shared\nSame evidence.\n", encoding="utf-8")
    _git(root, "add", "main.py", "shared.md")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "sample",
    )
    commit = _git(root, "rev-parse", "HEAD")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# Shared\nSame evidence.\n", encoding="utf-8")
    return MaterialAliasRegistry(
        (
            MaterialAlias("code", "git", root, ("main.py", "shared.md"), (workspace_id,), commit),
            MaterialAlias("notes", "file", docs, ("notes.md",), (workspace_id,)),
        )
    ), docs


def _runner(sessions, store):
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )
    document_repository = SqlAlchemyDocumentRepository(sessions)
    reader = SqlAlchemyRunExecutionReader(sessions)
    executor = MaterialRunExecutor(
        reader=reader,
        materials=store,
        ingestion_factory=lambda tenant, run_id: DocumentIngestionService(
            repository=document_repository,
            embedding=factory.create_embedding_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            ),
        ),
    )
    return WorkerRunner(
        worker_id="material-test",
        store=SqlAlchemyWorkerJobStore(
            sessions, ExponentialBackoff(WorkerRuntimeSettings(), Random(1))
        ),
        tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
        executor=RunExecutorDispatcher({"pathfinder-resume-v1": executor}),
        settings=WorkerRuntimeSettings(),
        unsupported_work_guard=reader.has_unsupported_pending_work,
    )


async def test_code_and_independent_document_keep_provenance_and_refresh(
    migrated_database_url: str, tmp_path: Path
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("material-r21")
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            identity.workspace_id, identity.user_id
        )
        aliases, docs = _fixture_aliases(tmp_path, tenant.workspace_id)
        store = SqlAlchemyMaterialStore(sessions, aliases)
        key = uuid4()
        project = await store.create_project(tenant, "Synthetic project", key)
        assert await store.create_project(tenant, "Synthetic project", key) == project
        with pytest.raises(DomainConflictError):
            await store.create_project(tenant, "Different", key)
        code = await store.create_source(tenant, project["id"], "code", uuid4())
        notes = await store.create_source(tenant, project["id"], "notes", uuid4())
        source_ids = (code["id"], notes["id"])
        import_key = uuid4()
        accepted = await store.submit_import(tenant, project["id"], source_ids, import_key)
        replay = await store.submit_import(tenant, project["id"], source_ids, import_key)
        assert replay.replayed and replay.receipt == accepted.receipt
        with pytest.raises(DomainConflictError):
            await store.submit_import(tenant, project["id"], source_ids[:1], import_key)
        runner = _runner(sessions, store)
        assert await runner.run_once(asyncio.Event())
        progress = await store.get_import(tenant, project["id"], accepted.receipt.resource_id)
        assert progress["status"] == "completed"
        assert progress["cache_digest"] is not None
        assert (await store.list_imports(tenant, project["id"]))[0][
            "id"
        ] == accepted.receipt.resource_id
        assert len(progress["snapshots"]) == 2
        assert sum(item["file_count"] for item in progress["snapshots"]) == 3
        for snapshot in progress["snapshots"]:
            assert snapshot["inventory"]["coverage"] == "authorized_paths_only"
            assert snapshot["inventory"]["outside_authorized_paths"] == "not_inventoried"
            assert snapshot["inventory"]["authorized_paths"]
        async with sessions() as session:
            files = (
                await session.scalars(
                    select(MaterialSnapshotFile).where(
                        MaterialSnapshotFile.workspace_id == tenant.workspace_id
                    )
                )
            ).all()
            assert files[0].document_id is not None
            assert len({item.document_id for item in files}) == 2
            first_doc = await session.scalar(
                select(Document).where(Document.id == files[0].document_id)
            )
            assert first_doc is not None
            old_content = {item.path: item.content for item in files}
        (docs / "notes.md").write_text(
            "# Shared\nChanged independent evidence.\n", encoding="utf-8"
        )
        second = await store.submit_import(tenant, project["id"], source_ids, uuid4())
        assert await runner.run_once(asyncio.Event())
        second_progress = await store.get_import(tenant, project["id"], second.receipt.resource_id)
        assert second_progress["status"] == "completed"
        assert second_progress["cache_digest"] != progress["cache_digest"]
        old_by_source = {
            item["source_id"]: item["manifest_digest"] for item in progress["snapshots"]
        }
        new_by_source = {
            item["source_id"]: item["manifest_digest"] for item in second_progress["snapshots"]
        }
        assert old_by_source[code["id"]] == new_by_source[code["id"]]
        assert old_by_source[notes["id"]] != new_by_source[notes["id"]]
        async with sessions() as session:
            historical = (
                await session.scalars(
                    select(MaterialSnapshotFile).where(
                        MaterialSnapshotFile.snapshot_id.in_(
                            [item["id"] for item in progress["snapshots"]]
                        )
                    )
                )
            ).all()
            assert {item.path: item.content for item in historical} == old_content
        other = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("material-other")
        foreign = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            other.workspace_id, other.user_id
        )
        with pytest.raises(DomainNotFoundError):
            await store.get_import(foreign, project["id"], accepted.receipt.resource_id)
    finally:
        await engine.dispose()


async def test_revoked_actor_cannot_read_or_execute_material(
    migrated_database_url: str, tmp_path: Path
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("material-revoked")
        tenant = TenantContext(identity.workspace_id, identity.user_id, identity.role)
        aliases, _ = _fixture_aliases(tmp_path, tenant.workspace_id)
        store = SqlAlchemyMaterialStore(sessions, aliases)
        project = await store.create_project(tenant, "Synthetic project", uuid4())
        source = await store.create_source(tenant, project["id"], "code", uuid4())
        accepted = await store.submit_import(tenant, project["id"], (source["id"],), uuid4())
        async with sessions.begin() as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=datetime.now(UTC))
            )
        with pytest.raises(DomainNotFoundError):
            await store.submit_import(tenant, project["id"], (source["id"],), uuid4())
        assert await _runner(sessions, store).run_once(asyncio.Event())
        async with sessions() as session:
            run = await session.scalar(select(Run).where(Run.id == accepted.receipt.run_id))
            assert run.status == "cancelled"
    finally:
        await engine.dispose()


async def test_retry_uses_fixed_snapshot_after_source_changes(
    migrated_database_url: str, tmp_path: Path
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("material-resume")
        tenant = TenantContext(identity.workspace_id, identity.user_id, identity.role)
        aliases, docs = _fixture_aliases(tmp_path, tenant.workspace_id)
        store = SqlAlchemyMaterialStore(sessions, aliases)
        project = await store.create_project(tenant, "Resume snapshot", uuid4())
        source = await store.create_source(tenant, project["id"], "notes", uuid4())
        accepted = await store.submit_import(tenant, project["id"], (source["id"],), uuid4())
        original = read_alias(aliases.get("notes", tenant.workspace_id))
        fixed_id = await store.persist_snapshot(
            tenant, accepted.receipt.resource_id, source["id"], original
        )
        (docs / "notes.md").write_text("Changed before retry.\n", encoding="utf-8")
        assert await _runner(sessions, store).run_once(asyncio.Event())
        progress = await store.get_import(tenant, project["id"], accepted.receipt.resource_id)
        assert progress["status"] == "completed"
        assert progress["snapshots"][0]["id"] == fixed_id
        files = await store.snapshot_files(tenant, fixed_id)
        assert files[0].content == original.files[0].content
    finally:
        await engine.dispose()
