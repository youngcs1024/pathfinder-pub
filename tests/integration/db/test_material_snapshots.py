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

from app.agents.material_facts import PROMPT_VERSION, MaterialFactExtractor
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import Document, MaterialSnapshotFile, Run, WorkspaceMembership
from app.db.project_facts import SqlAlchemyProjectFactStore, extractor_identity
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainConflictError, DomainNotFoundError
from app.domain.project_facts import CandidateFactV1
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
    facts = SqlAlchemyProjectFactStore(sessions)
    executor = MaterialRunExecutor(
        reader=reader,
        materials=store,
        aliases=store.aliases,
        facts=facts,
        extractor_digest=extractor_identity(PROMPT_VERSION, "fake:qwen3.6-flash-2026-04-16"),
        extractor_factory=lambda tenant, run_id, _scope: MaterialFactExtractor(
            model=factory.create_chat_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            )
        ),
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
        executor=RunExecutorDispatcher(
            {
                "pathfinder-resume-v2": executor,
                "pathfinder-resume-v3": executor,
                "pathfinder-resume-v4": executor,
            }
        ),
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
            workspace_id=identity.workspace_id, actor_user_id=identity.user_id
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
        facts = SqlAlchemyProjectFactStore(sessions)
        catalog = await facts.current_facts(tenant, project["id"])
        assert catalog["fact_set_id"] is not None
        assert catalog["complete"] is False
        assert catalog["issues"] == [{"code": "offline_fake_no_semantic_extraction"}]
        fact_key = uuid4()
        candidate = CandidateFactV1(
            claim="I prepared synthetic documentation", kind="personal_statement"
        )
        added = await facts.command(
            tenant,
            kind="material_fact_add",
            project_id=project["id"],
            import_id=catalog["import_id"],
            request_id=fact_key,
            candidate=candidate,
        )
        assert (
            await facts.command(
                tenant,
                kind="material_fact_add",
                project_id=project["id"],
                import_id=catalog["import_id"],
                request_id=fact_key,
                candidate=candidate,
            )
        ).replayed
        fact_id = added.receipt.resource_id
        with pytest.raises(DomainConflictError):
            await facts.command(
                tenant,
                kind="material_fact_review",
                project_id=project["id"],
                import_id=catalog["import_id"],
                fact_id=fact_id,
                expected_version=1,
                decision="confirm",
                request_id=uuid4(),
            )
        await facts.command(
            tenant,
            kind="material_fact_review",
            project_id=project["id"],
            import_id=catalog["import_id"],
            fact_id=fact_id,
            expected_version=1,
            decision="confirm",
            attested=True,
            request_id=uuid4(),
        )
        assert len(await facts.search_confirmed(tenant, (project["id"],), "synthetic")) == 1
        with pytest.raises(DomainConflictError):
            await facts.command(
                tenant,
                kind="material_fact_review",
                project_id=project["id"],
                import_id=catalog["import_id"],
                fact_id=fact_id,
                expected_version=1,
                decision="reject",
                request_id=uuid4(),
            )
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
            code_file = next(item for item in files if item.path == "main.py")
            assert code_file.line_ranges_json == [{"ordinal": 0, "start_line": 1, "end_line": 1}]
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
        assert (await facts.current_facts(tenant, project["id"]))["facts"] == []
        assert (await facts.search_confirmed(tenant, (project["id"],), "synthetic")) == []
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
            workspace_id=other.workspace_id, actor_user_id=other.user_id
        )
        with pytest.raises(DomainNotFoundError):
            await store.get_import(foreign, project["id"], accepted.receipt.resource_id)
        with pytest.raises(DomainNotFoundError):
            await facts.current_facts(foreign, project["id"])
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


async def test_multi_project_scope_requires_each_exact_project_import_pair(
    migrated_database_url: str, tmp_path: Path
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("material-multi-project")
        tenant = TenantContext(identity.workspace_id, identity.user_id, identity.role)
        aliases, _ = _fixture_aliases(tmp_path, tenant.workspace_id)
        materials = SqlAlchemyMaterialStore(sessions, aliases)
        facts = SqlAlchemyProjectFactStore(sessions)
        pairs = []
        runner = _runner(sessions, materials)
        for name in ("Project A", "Project B"):
            project = await materials.create_project(tenant, name, uuid4())
            source = await materials.create_source(tenant, project["id"], "notes", uuid4())
            accepted = await materials.submit_import(
                tenant, project["id"], (source["id"],), uuid4()
            )
            assert await runner.run_once(asyncio.Event())
            pairs.append((project["id"], accepted.receipt.resource_id))
        scope = await facts.retrieval_scope(tenant, tuple(pairs))
        assert len(scope.files) == 2
        assert len({item.id for item in scope.files}) == 2
        with pytest.raises(DomainNotFoundError):
            await facts.retrieval_scope(tenant, ((pairs[0][0], pairs[1][1]),))
    finally:
        await engine.dispose()
