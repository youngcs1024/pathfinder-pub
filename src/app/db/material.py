"""Workspace-scoped material commands and immutable snapshot persistence."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    MaterialImport,
    MaterialProject,
    MaterialSnapshot,
    MaterialSnapshotFile,
    MaterialSource,
    Message,
    Run,
    RunEvent,
    RunJob,
    WorkspaceMembership,
)
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.session import AsyncSessionFactory, database_session, transaction
from app.domain.errors import DomainInvariantError, DomainNotFoundError, DomainValidationError
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandAccepted, CommandReceiptV1, ResumeCommandRequest
from app.domain.run_payloads import (
    MaterialPreparationInputV1,
    MaterialPreparationRunInputV1,
    RunMode,
)
from app.domain.tenancy import TenantContext
from app.material.aliases import MaterialAliasRegistry
from app.material.reader import MaterialRead


class MaterialImportCommandV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    project_id: UUID
    source_ids: tuple[UUID, ...]


class MaterialProjectCommandV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: str


class MaterialSourceCommandV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    project_id: UUID
    alias: str


async def _active_role(session: AsyncSession, tenant: TenantContext) -> str:
    role = await session.scalar(
        select(WorkspaceMembership.role)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
            WorkspaceMembership.revoked_at.is_(None),
        )
        .with_for_update(read=True)
    )
    if role is None:
        raise DomainNotFoundError
    return role


async def _editable_project(
    session: AsyncSession, tenant: TenantContext, project_id: UUID
) -> MaterialProject:
    role = await _active_role(session, tenant)
    if role == WorkspaceRole.REVIEWER.value:
        raise DomainNotFoundError
    project = await session.scalar(
        select(MaterialProject)
        .where(
            MaterialProject.workspace_id == tenant.workspace_id,
            MaterialProject.id == project_id,
        )
        .with_for_update(read=True)
    )
    if project is None or (
        project.created_by_user_id != tenant.actor_user_id and role != WorkspaceRole.ADMIN.value
    ):
        raise DomainNotFoundError
    return project


class _ProjectWriter(ResumeCommandWriter):
    supported_kinds = frozenset({"material_project_create"})

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        if await _active_role(session, tenant) == WorkspaceRole.REVIEWER.value:
            raise DomainNotFoundError
        if not isinstance(request.payload, MaterialProjectCommandV1):
            raise DomainValidationError("project request is invalid")
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        payload = request.payload
        assert isinstance(payload, MaterialProjectCommandV1)
        project_id = uuid4()
        session.add(
            MaterialProject(
                id=project_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                name=payload.name,
            )
        )
        return CommandReceiptV1(command_id=command_id, resource_id=project_id, status="completed")


class _SourceWriter(ResumeCommandWriter):
    supported_kinds = frozenset({"material_source_create"})

    def __init__(self, aliases: MaterialAliasRegistry) -> None:
        self.aliases = aliases

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        payload = request.payload
        if (
            not isinstance(payload, MaterialSourceCommandV1)
            or request.target_id != payload.project_id
        ):
            raise DomainValidationError("source request is invalid")
        await _editable_project(session, tenant, payload.project_id)
        self.aliases.get(payload.alias, tenant.workspace_id)
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        payload = request.payload
        assert isinstance(payload, MaterialSourceCommandV1)
        alias = self.aliases.get(payload.alias, tenant.workspace_id)
        source_id = uuid4()
        session.add(
            MaterialSource(
                id=source_id,
                workspace_id=tenant.workspace_id,
                project_id=payload.project_id,
                alias_name=alias.name,
                alias_digest=alias.digest,
                kind=alias.kind,
            )
        )
        return CommandReceiptV1(command_id=command_id, resource_id=source_id, status="completed")


class _ImportWriter(ResumeCommandWriter):
    supported_kinds = frozenset({"material_import"})

    def __init__(self, aliases: MaterialAliasRegistry) -> None:
        self.aliases = aliases

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        payload = request.payload
        if (
            not isinstance(payload, MaterialImportCommandV1)
            or request.target_id != payload.project_id
        ):
            raise DomainValidationError("material import request is invalid")
        await _editable_project(session, tenant, payload.project_id)
        if (
            not payload.source_ids
            or len(payload.source_ids) > 20
            or len(set(payload.source_ids)) != len(payload.source_ids)
        ):
            raise DomainValidationError("material import sources are invalid")
        rows = (
            await session.scalars(
                select(MaterialSource).where(
                    MaterialSource.workspace_id == tenant.workspace_id,
                    MaterialSource.project_id == payload.project_id,
                    MaterialSource.id.in_(payload.source_ids),
                )
            )
        ).all()
        if len(rows) != len(payload.source_ids):
            raise DomainNotFoundError
        for row in rows:
            if self.aliases.get(row.alias_name, tenant.workspace_id).digest != row.alias_digest:
                raise DomainNotFoundError
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        payload = request.payload
        assert isinstance(payload, MaterialImportCommandV1)
        import_id, run_id, conversation_id, message_id = uuid4(), uuid4(), uuid4(), uuid4()
        session.add(
            Conversation(
                id=conversation_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                title="Material preparation",
            )
        )
        session.add(
            Message(
                id=message_id,
                workspace_id=tenant.workspace_id,
                conversation_id=conversation_id,
                actor_user_id=tenant.actor_user_id,
                role="user",
                content="Material preparation request",
            )
        )
        run_input = MaterialPreparationRunInputV1(
            payload=MaterialPreparationInputV1(
                import_id=import_id, project_id=payload.project_id, source_ids=payload.source_ids
            )
        )
        session.add(
            Run(
                id=run_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                conversation_id=conversation_id,
                request_message_id=message_id,
                mode=RunMode.MATERIAL_PREPARATION.value,
                resume_document_id=None,
                input_json=run_input.model_dump(mode="json", round_trip=True),
                limits_json={
                    "schema_version": 1,
                    "max_model_calls": 100,
                    "max_tool_calls": 0,
                    "max_tool_results": 0,
                    "max_iterations": 200,
                },
                status="queued",
                graph_version="pathfinder-resume-v1",
            )
        )
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
            MaterialImport(
                id=import_id,
                workspace_id=tenant.workspace_id,
                project_id=payload.project_id,
                run_id=run_id,
                source_ids=[str(value) for value in payload.source_ids],
            )
        )
        await session.flush()
        seq = await session.scalar(
            update(Run)
            .where(Run.id == run_id, Run.workspace_id == tenant.workspace_id)
            .values(next_event_seq=Run.next_event_seq + 1)
            .returning(Run.next_event_seq - 1)
        )
        if seq != 1:
            raise DomainInvariantError("new material run event sequence is invalid")
        session.add(
            RunEvent(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                run_id=run_id,
                actor_user_id=tenant.actor_user_id,
                seq=1,
                type="run.created",
                version=1,
                payload={
                    "mode": "material_preparation",
                    "status": "queued",
                    "graph_version": "pathfinder-resume-v1",
                },
            )
        )
        return CommandReceiptV1(
            command_id=command_id, run_id=run_id, resource_id=import_id, status="queued"
        )


class SqlAlchemyMaterialStore:
    def __init__(self, sessions: AsyncSessionFactory, aliases: MaterialAliasRegistry) -> None:
        self.sessions = sessions
        self.aliases = aliases
        self.commands = SqlAlchemyResumeCommandStore(sessions)

    async def list_projects(self, tenant: TenantContext) -> list[dict[str, object]]:
        async with database_session(self.sessions) as session:
            await _active_role(session, tenant)
            rows = (
                await session.scalars(
                    select(MaterialProject)
                    .where(MaterialProject.workspace_id == tenant.workspace_id)
                    .order_by(MaterialProject.created_at, MaterialProject.id)
                )
            ).all()
            return [
                {"id": row.id, "name": row.name, "created_by_user_id": row.created_by_user_id}
                for row in rows
            ]

    async def create_project(
        self, tenant: TenantContext, name: str, client_request_id: UUID
    ) -> dict[str, object]:
        if not isinstance(name, str) or name != name.strip() or not 1 <= len(name) <= 120:
            raise DomainValidationError("project name is invalid")
        accepted = await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=client_request_id,
                kind="material_project_create",
                target_id=None,
                payload_version=1,
                payload=MaterialProjectCommandV1(name=name),
            ),
            writer=_ProjectWriter(),
        )
        async with database_session(self.sessions) as session:
            row = await session.scalar(
                select(MaterialProject).where(
                    MaterialProject.workspace_id == tenant.workspace_id,
                    MaterialProject.id == accepted.receipt.resource_id,
                )
            )
            if row is None:
                raise DomainInvariantError("accepted project is missing")
            return {"id": row.id, "name": row.name, "created_by_user_id": row.created_by_user_id}

    async def list_sources(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]:
        async with database_session(self.sessions) as session:
            await _active_role(session, tenant)
            project = await session.scalar(
                select(MaterialProject.id).where(
                    MaterialProject.workspace_id == tenant.workspace_id,
                    MaterialProject.id == project_id,
                )
            )
            if project is None:
                raise DomainNotFoundError
            rows = (
                await session.scalars(
                    select(MaterialSource)
                    .where(
                        MaterialSource.workspace_id == tenant.workspace_id,
                        MaterialSource.project_id == project_id,
                    )
                    .order_by(MaterialSource.created_at, MaterialSource.id)
                )
            ).all()
            return [
                {
                    "id": row.id,
                    "project_id": row.project_id,
                    "alias": row.alias_name,
                    "kind": row.kind,
                }
                for row in rows
            ]

    async def create_source(
        self,
        tenant: TenantContext,
        project_id: UUID,
        alias_name: str,
        client_request_id: UUID,
    ) -> dict[str, object]:
        accepted = await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=client_request_id,
                kind="material_source_create",
                target_id=project_id,
                payload_version=1,
                payload=MaterialSourceCommandV1(project_id=project_id, alias=alias_name),
            ),
            writer=_SourceWriter(self.aliases),
        )
        async with database_session(self.sessions) as session:
            row = await session.scalar(
                select(MaterialSource).where(
                    MaterialSource.workspace_id == tenant.workspace_id,
                    MaterialSource.project_id == project_id,
                    MaterialSource.id == accepted.receipt.resource_id,
                )
            )
            if row is None:
                raise DomainInvariantError("accepted source is missing")
            return {
                "id": row.id,
                "project_id": row.project_id,
                "alias": row.alias_name,
                "kind": row.kind,
            }

    async def submit_import(
        self,
        tenant: TenantContext,
        project_id: UUID,
        source_ids: tuple[UUID, ...],
        client_request_id: UUID,
    ) -> CommandAccepted:
        payload = MaterialImportCommandV1(project_id=project_id, source_ids=source_ids)
        return await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=client_request_id,
                kind="material_import",
                target_id=project_id,
                payload_version=1,
                payload=payload,
            ),
            writer=_ImportWriter(self.aliases),
        )

    async def get_import(
        self, tenant: TenantContext, project_id: UUID, import_id: UUID
    ) -> dict[str, object]:
        async with database_session(self.sessions) as session:
            role = await _active_role(session, tenant)
            project = await session.scalar(
                select(MaterialProject).where(
                    MaterialProject.workspace_id == tenant.workspace_id,
                    MaterialProject.id == project_id,
                )
            )
            if project is None:
                raise DomainNotFoundError
            if (
                project.created_by_user_id != tenant.actor_user_id
                and role != WorkspaceRole.ADMIN.value
            ):
                raise DomainNotFoundError
            row = await session.scalar(
                select(MaterialImport).where(
                    MaterialImport.workspace_id == tenant.workspace_id,
                    MaterialImport.project_id == project_id,
                    MaterialImport.id == import_id,
                )
            )
            if row is None:
                raise DomainNotFoundError
            run = await session.scalar(
                select(Run).where(Run.workspace_id == tenant.workspace_id, Run.id == row.run_id)
            )
            snapshots = (
                await session.scalars(
                    select(MaterialSnapshot)
                    .where(
                        MaterialSnapshot.workspace_id == tenant.workspace_id,
                        MaterialSnapshot.import_id == import_id,
                    )
                    .order_by(MaterialSnapshot.created_at, MaterialSnapshot.id)
                )
            ).all()
            snapshot_items = []
            for snapshot in snapshots:
                files = (
                    await session.scalars(
                        select(MaterialSnapshotFile)
                        .where(
                            MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                            MaterialSnapshotFile.snapshot_id == snapshot.id,
                        )
                        .order_by(MaterialSnapshotFile.path)
                    )
                ).all()
                snapshot_items.append(
                    {
                        "id": snapshot.id,
                        "source_id": snapshot.source_id,
                        "source_revision": snapshot.source_revision,
                        "manifest_digest": snapshot.manifest_digest,
                        "file_count": len(files),
                        "indexed_count": sum(item.document_id is not None for item in files),
                        "unindexed_paths": [
                            item.path for item in files if item.document_id is None
                        ],
                        "inventory": snapshot.inventory_json,
                    }
                )
            return {
                "id": row.id,
                "project_id": row.project_id,
                "run_id": row.run_id,
                "status": run.status,
                "error_category": run.error_category,
                "cache_digest": row.cache_digest,
                "snapshots": snapshot_items,
            }

    async def list_imports(
        self, tenant: TenantContext, project_id: UUID
    ) -> list[dict[str, object]]:
        async with database_session(self.sessions) as session:
            role = await _active_role(session, tenant)
            project = await session.scalar(
                select(MaterialProject).where(
                    MaterialProject.workspace_id == tenant.workspace_id,
                    MaterialProject.id == project_id,
                )
            )
            if project is None or (
                project.created_by_user_id != tenant.actor_user_id
                and role != WorkspaceRole.ADMIN.value
            ):
                raise DomainNotFoundError
            rows = (
                await session.execute(
                    select(MaterialImport.id, Run.status, MaterialImport.created_at)
                    .join(
                        Run,
                        (Run.workspace_id == MaterialImport.workspace_id)
                        & (Run.id == MaterialImport.run_id),
                    )
                    .where(
                        MaterialImport.workspace_id == tenant.workspace_id,
                        MaterialImport.project_id == project_id,
                    )
                    .order_by(MaterialImport.created_at.desc(), MaterialImport.id.desc())
                    .limit(50)
                )
            ).all()
            return [
                {"id": row.id, "status": row.status, "created_at": row.created_at} for row in rows
            ]

    async def source_for_execution(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID
    ) -> MaterialSource:
        async with database_session(self.sessions) as session:
            await _active_role(session, tenant)
            row = await session.scalar(
                select(MaterialImport).where(
                    MaterialImport.workspace_id == tenant.workspace_id,
                    MaterialImport.id == import_id,
                )
            )
            if row is None or str(source_id) not in row.source_ids:
                raise DomainNotFoundError
            source = await session.scalar(
                select(MaterialSource).where(
                    MaterialSource.workspace_id == tenant.workspace_id,
                    MaterialSource.id == source_id,
                    MaterialSource.project_id == row.project_id,
                )
            )
            if source is None:
                raise DomainNotFoundError
            return source

    async def persist_snapshot(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID, read: MaterialRead
    ) -> UUID:
        async with transaction(self.sessions) as session:
            await _active_role(session, tenant)
            existing = await session.scalar(
                select(MaterialSnapshot).where(
                    MaterialSnapshot.workspace_id == tenant.workspace_id,
                    MaterialSnapshot.import_id == import_id,
                    MaterialSnapshot.source_id == source_id,
                )
            )
            if existing is not None:
                return existing.id
            snapshot_id = uuid4()
            cache_digest = hashlib.sha256(
                (
                    f"{read.digest}\n"
                    "reader=git-object-or-nofollow-utf8-v1\n"
                    "normalization=nfc-lf-v1\n"
                    "chunking=material-lines-utf8-800-v1\n"
                    "embedding=qwen-beijing-text-embedding-v4-1536-v1\n"
                ).encode()
            ).hexdigest()
            session.add(
                MaterialSnapshot(
                    id=snapshot_id,
                    workspace_id=tenant.workspace_id,
                    import_id=import_id,
                    source_id=source_id,
                    source_revision=read.source_revision,
                    manifest_digest=read.digest,
                    cache_digest=cache_digest,
                    inventory_json={
                        "dependency_files": list(read.dependency_files),
                        "entrypoint_files": list(read.entrypoint_files),
                        "omitted_files": list(read.omitted_files),
                        "coverage": "authorized_paths_only",
                        "authorized_paths": list(read.authorized_paths),
                        "outside_authorized_paths": "not_inventoried",
                    },
                )
            )
            await session.flush()
            session.add_all(
                MaterialSnapshotFile(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    snapshot_id=snapshot_id,
                    path=item.path,
                    content=item.content,
                    content_digest=item.digest,
                )
                for item in read.files
            )
            return snapshot_id

    async def existing_snapshot(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID
    ) -> UUID | None:
        async with database_session(self.sessions) as session:
            await _active_role(session, tenant)
            return await session.scalar(
                select(MaterialSnapshot.id).where(
                    MaterialSnapshot.workspace_id == tenant.workspace_id,
                    MaterialSnapshot.import_id == import_id,
                    MaterialSnapshot.source_id == source_id,
                )
            )

    async def record_import_cache_digest(
        self, tenant: TenantContext, import_id: UUID, source_ids: tuple[UUID, ...]
    ) -> str:
        async with transaction(self.sessions) as session:
            await _active_role(session, tenant)
            row = await session.scalar(
                select(MaterialImport)
                .where(
                    MaterialImport.workspace_id == tenant.workspace_id,
                    MaterialImport.id == import_id,
                )
                .with_for_update()
            )
            if row is None or row.source_ids != [str(value) for value in source_ids]:
                raise DomainNotFoundError
            snapshots = (
                await session.scalars(
                    select(MaterialSnapshot).where(
                        MaterialSnapshot.workspace_id == tenant.workspace_id,
                        MaterialSnapshot.import_id == import_id,
                    )
                )
            ).all()
            by_source = {item.source_id: item.cache_digest for item in snapshots}
            if len(by_source) != len(source_ids) or any(
                source_id not in by_source for source_id in source_ids
            ):
                raise DomainInvariantError("import snapshots are incomplete")
            digest = hashlib.sha256(
                "".join(
                    f"{source_id}:{by_source[source_id]}\n" for source_id in source_ids
                ).encode()
            ).hexdigest()
            if row.cache_digest is not None and row.cache_digest != digest:
                raise DomainInvariantError("import cache identity changed")
            row.cache_digest = digest
            return digest

    async def snapshot_files(
        self, tenant: TenantContext, snapshot_id: UUID
    ) -> list[MaterialSnapshotFile]:
        async with database_session(self.sessions) as session:
            await _active_role(session, tenant)
            return list(
                (
                    await session.scalars(
                        select(MaterialSnapshotFile)
                        .where(
                            MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                            MaterialSnapshotFile.snapshot_id == snapshot_id,
                        )
                        .order_by(MaterialSnapshotFile.path)
                    )
                ).all()
            )

    async def attach_document(
        self, tenant: TenantContext, file_id: UUID, document_id: UUID
    ) -> None:
        async with transaction(self.sessions) as session:
            await _active_role(session, tenant)
            row = await session.scalar(
                select(MaterialSnapshotFile)
                .where(
                    MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                    MaterialSnapshotFile.id == file_id,
                )
                .with_for_update()
            )
            if row is None:
                raise DomainNotFoundError
            if row.document_id is not None and row.document_id != document_id:
                raise DomainInvariantError("snapshot file document identity changed")
            row.document_id = document_id
