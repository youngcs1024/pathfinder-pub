"""Deterministic material preparation executed by the existing single worker."""

from __future__ import annotations

import asyncio
from typing import Protocol
from uuid import UUID

from app.domain.errors import DomainNotFoundError, DomainUnavailableError, DomainValidationError
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInvalidError,
    RunExecutionReader,
)
from app.domain.run_payloads import (
    MaterialPreparationResultV1,
    MaterialPreparationRunInputV1,
    MaterialPreparationRunOutputV1,
)
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMAccountingError, LLMProviderError
from app.material.aliases import MaterialAliasRegistry
from app.material.chunking import prepare_material_file
from app.material.reader import MaterialFile, MaterialRead, MaterialReadError, read_alias
from app.retrieval.chunking import PreparedIngestionBatch
from app.retrieval.documents import (
    DocumentIngestionAuthorizationError,
    DocumentIngestionInvariantError,
    DocumentIngestionService,
)
from app.worker.contracts import RunExecutionResult

MAX_INDEX_CHUNKS = 100


class MaterialSourceView(Protocol):
    alias_name: str
    alias_digest: str


class MaterialFileView(Protocol):
    id: UUID
    path: str
    content: bytes
    content_digest: str
    document_id: UUID | None


class MaterialExecutionPort(Protocol):
    async def source_for_execution(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID
    ) -> MaterialSourceView: ...
    async def existing_snapshot(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID
    ) -> UUID | None: ...
    async def persist_snapshot(
        self, tenant: TenantContext, import_id: UUID, source_id: UUID, read: MaterialRead
    ) -> UUID: ...
    async def snapshot_files(
        self, tenant: TenantContext, snapshot_id: UUID
    ) -> list[MaterialFileView]: ...
    async def attach_document(
        self,
        tenant: TenantContext,
        file_id: UUID,
        document_id: UUID,
        line_ranges: tuple[tuple[int, int], ...],
    ) -> None: ...
    async def record_import_cache_digest(
        self, tenant: TenantContext, import_id: UUID, source_ids: tuple[UUID, ...]
    ) -> str: ...


class MaterialRunExecutor:
    def __init__(
        self,
        *,
        reader: RunExecutionReader,
        materials: MaterialExecutionPort,
        aliases: MaterialAliasRegistry,
        ingestion_factory,
    ) -> None:
        self.reader = reader
        self.materials = materials
        self.aliases = aliases
        self.ingestion_factory = ingestion_factory

    async def execute(
        self, run_id: UUID, tenant: TenantContext, graph_version: str
    ) -> RunExecutionResult:
        if graph_version != "pathfinder-resume-v1":
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="unknown_graph_version"
            )
        try:
            execution = await self.reader.read_for_execution(
                run_id=run_id,
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                graph_version=graph_version,
            )
            request = execution.request
            if not isinstance(request, MaterialPreparationRunInputV1):
                return RunExecutionResult(
                    status=RunStatus.FAILED, error_category="invalid_run_input"
                )
            payload = request.payload
            snapshot_ids: list[UUID] = []
            file_count = document_count = indexed_chunks = 0
            for source_id in payload.source_ids:
                await self.reader.assert_execution_allowed(
                    run_id=run_id,
                    workspace_id=tenant.workspace_id,
                    actor_user_id=tenant.actor_user_id,
                    graph_version=graph_version,
                )
                source = await self.materials.source_for_execution(
                    tenant, payload.import_id, source_id
                )
                try:
                    alias = self.aliases.get(source.alias_name, tenant.workspace_id)
                except DomainValidationError:
                    raise MaterialReadError("alias_unavailable") from None
                if alias.digest != source.alias_digest:
                    raise MaterialReadError("alias_changed")
                snapshot_id = await self.materials.existing_snapshot(
                    tenant, payload.import_id, source_id
                )
                if snapshot_id is None:
                    material = await asyncio.to_thread(read_alias, alias)
                    snapshot_id = await self.materials.persist_snapshot(
                        tenant, payload.import_id, source_id, material
                    )
                snapshot_ids.append(snapshot_id)
                files = await self.materials.snapshot_files(tenant, snapshot_id)
                file_count += len(files)
                for item in files:
                    if item.document_id is not None:
                        document_count += 1
                        indexed_chunks += len(
                            prepare_material_file(
                                MaterialFile(
                                    path=item.path,
                                    content=item.content,
                                    digest=item.content_digest,
                                    line_count=0,
                                )
                            ).chunks
                        )
                        continue
                    prepared = prepare_material_file(
                        MaterialFile(
                            path=item.path,
                            content=item.content,
                            digest=item.content_digest,
                            line_count=0,
                        )
                    )
                    if indexed_chunks + len(prepared.chunks) > MAX_INDEX_CHUNKS:
                        continue
                    await self.reader.assert_execution_allowed(
                        run_id=run_id,
                        workspace_id=tenant.workspace_id,
                        actor_user_id=tenant.actor_user_id,
                        graph_version=graph_version,
                    )
                    service: DocumentIngestionService = self.ingestion_factory(tenant, run_id)
                    (document_id,) = await service.ingest(
                        tenant=tenant,
                        batch=PreparedIngestionBatch((prepared,)),
                    )
                    await self.materials.attach_document(
                        tenant,
                        item.id,
                        document_id,
                        tuple((chunk.start_line, chunk.end_line) for chunk in prepared.chunks),
                    )
                    document_count += 1
                    indexed_chunks += len(prepared.chunks)
            if document_count == 0:
                return RunExecutionResult(
                    status=RunStatus.FAILED, error_category="index_budget_exhausted"
                )
            await self.materials.record_import_cache_digest(
                tenant, payload.import_id, payload.source_ids
            )
            result = MaterialPreparationRunOutputV1(
                payload=MaterialPreparationResultV1(
                    import_id=payload.import_id,
                    snapshot_ids=tuple(snapshot_ids),
                    file_count=file_count,
                    document_count=document_count,
                )
            )
            return RunExecutionResult(status=RunStatus.COMPLETED, result=result)
        except asyncio.CancelledError:
            raise
        except RunExecutionCancelledError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except (DomainNotFoundError, DocumentIngestionAuthorizationError):
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except RunExecutionInvalidError as error:
            return RunExecutionResult(status=RunStatus.FAILED, error_category=error.category)
        except MaterialReadError as error:
            return RunExecutionResult(status=RunStatus.FAILED, error_category=error.code)
        except DomainUnavailableError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="database_unavailable", retryable=True
            )
        except LLMAccountingError as error:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="llm_accounting_failed",
                retryable=error.retryable,
            )
        except LLMProviderError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="provider_unavailable", retryable=True
            )
        except DocumentIngestionInvariantError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="document_ingestion_invalid"
            )
