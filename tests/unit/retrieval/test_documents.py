from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.ports import EmbeddingResult
from app.retrieval.chunking import (
    PreparedIngestionBatch,
    PreparedIngestionChunk,
    PreparedIngestionSource,
)
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentIdentity,
    DocumentIngestionInvariantError,
    DocumentIngestionService,
    ImmutableDocumentRepresentation,
    build_document_identity,
    partition_embedding_texts,
)


def _source(*, content_hash: str = "a" * 64, chunk_count: int = 2) -> PreparedIngestionSource:
    chunks = tuple(
        PreparedIngestionChunk(
            ordinal=index,
            section=None,
            text=f"chunk {index}",
            content_hash=f"{index + 1:064x}",
            token_count=len(f"chunk {index}".encode()),
        )
        for index in range(chunk_count)
    )
    return PreparedIngestionSource(
        source_name="resume.md",
        source_type="markdown",
        title="resume",
        content="\n".join(chunk.text for chunk in chunks),
        content_hash=content_hash,
        normalization_version="nfc-lf-v1",
        chunking_version="heading-paragraph-utf8-budget-800-v1",
        chunks=chunks,
    )


class _Embedding:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts, metadata):  # type: ignore[no-untyped-def]
        assert metadata == {"graph_node": "document_ingestion"}
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple((float(index),) * 1536 for index, _ in enumerate(texts))
        )


class _Repository:
    def __init__(self, existing: UUID | None = None) -> None:
        self.existing = existing
        self.persisted: list[ImmutableDocumentRepresentation] = []

    async def find_complete(self, *, tenant, identity, expected):  # type: ignore[no-untyped-def]
        return self.existing

    async def persist(self, representation):  # type: ignore[no-untyped-def]
        self.persisted.append(representation)
        self.existing = uuid4()
        return self.existing


def test_profile_identity_and_batch_partition_are_stable() -> None:
    workspace_id = uuid4()
    identity = build_document_identity(workspace_id=workspace_id, source=_source())

    assert identity == DocumentIdentity(
        workspace_id=workspace_id,
        content_hash="a" * 64,
        normalization_version="nfc-lf-v1",
        chunking_version="heading-paragraph-utf8-budget-800-v1",
        embedding_model=EMBEDDING_PROFILE,
    )
    groups = partition_embedding_texts(tuple(str(index) for index in range(21)))
    assert tuple(map(len, groups)) == (10, 10, 1)


async def test_embedding_order_maps_to_ordinals_and_duplicate_batch_identity_is_deduped() -> None:
    source = _source(chunk_count=12)
    embedding = _Embedding()
    repository = _Repository()
    service = DocumentIngestionService(repository, embedding)

    result = await service.ingest(
        tenant=TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN),
        batch=PreparedIngestionBatch((source, source)),
    )

    assert result[0] == result[1]
    assert tuple(map(len, embedding.calls)) == (10, 2)
    assert len(repository.persisted) == 1
    representation = repository.persisted[0]
    assert [chunk.ordinal for chunk in representation.chunks] == list(range(12))
    assert [chunk.embedding[0] for chunk in representation.chunks] == [
        *map(float, range(10)),
        0.0,
        1.0,
    ]


async def test_existing_document_skips_embedding() -> None:
    existing = uuid4()
    embedding = _Embedding()
    repository = _Repository(existing)

    result = await DocumentIngestionService(repository, embedding).ingest(
        tenant=TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN),
        batch=PreparedIngestionBatch((_source(),)),
    )

    assert result == (existing,)
    assert embedding.calls == []
    assert repository.persisted == []


async def test_provider_count_mismatch_fails_closed() -> None:
    class _MalformedEmbedding:
        async def embed(self, texts, metadata):  # type: ignore[no-untyped-def]
            return type("Malformed", (), {"vectors": ((0.0,) * 1536,)})()

    repository = _Repository()
    service = DocumentIngestionService(repository, _MalformedEmbedding())

    with pytest.raises(DocumentIngestionInvariantError):
        await service.ingest(
            tenant=TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN),
            batch=PreparedIngestionBatch((_source(),)),
        )
    assert repository.persisted == []


async def test_cancellation_propagates_without_persistence() -> None:
    class _CancelledEmbedding:
        async def embed(self, texts, metadata):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    repository = _Repository()
    with pytest.raises(asyncio.CancelledError):
        await DocumentIngestionService(repository, _CancelledEmbedding()).ingest(
            tenant=TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN),
            batch=PreparedIngestionBatch((_source(),)),
        )
    assert repository.persisted == []


def test_sensitive_dtos_have_safe_repr() -> None:
    source = _source()
    identity = build_document_identity(workspace_id=uuid4(), source=source)
    assert source.content not in repr(identity)
