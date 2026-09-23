from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from time import monotonic
from typing import Protocol, runtime_checkable
from uuid import UUID

from app.domain.tenancy import TenantContext
from app.domain.tracing import (
    SpanStatus,
    bind_trace_scope,
    current_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.llm.ports import LOCKED_EMBEDDING_DIMENSION, EmbeddingPort
from app.retrieval.chunking import PreparedIngestionBatch, PreparedIngestionSource

EMBEDDING_PROFILE = "qwen-beijing-text-embedding-v4-1536-v1"
EMBEDDING_DIMENSION = LOCKED_EMBEDDING_DIMENSION
MAX_EMBEDDING_BATCH_SIZE = 10
INGESTION_GRAPH_NODE = "document_ingestion"
RETRIEVAL_GRAPH_NODE = "document_retrieval"
RETRIEVAL_TOP_K = 5
RETRIEVAL_CONTEXT_BYTE_BUDGET = 4_000
RETRIEVED_CHUNK_TEXT_BYTE_LIMIT = 800


class DocumentIngestionInvariantError(Exception):
    """A safe failure for a malformed provider result or persisted representation."""

    def __init__(self) -> None:
        super().__init__("document ingestion invariant failed")


class DocumentIngestionAuthorizationError(Exception):
    """The actor lost active workspace membership before persistence."""

    def __init__(self) -> None:
        super().__init__("document ingestion authorization is unavailable")


class DocumentRetrievalAuthorizationError(Exception):
    """The actor has no active membership at the retrieval boundary."""

    def __init__(self) -> None:
        super().__init__("document retrieval authorization is unavailable")


class DocumentRetrievalInvariantError(Exception):
    """A safe failure for malformed retrieval input or evidence."""

    def __init__(self) -> None:
        super().__init__("document retrieval invariant failed")


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    workspace_id: UUID
    content_hash: str
    normalization_version: str
    chunking_version: str
    embedding_model: str = EMBEDDING_PROFILE


@dataclass(frozen=True, slots=True, repr=False)
class ImmutableDocumentChunk:
    ordinal: int
    section: str | None
    text: str = field(repr=False)
    content_hash: str
    token_count: int
    embedding: tuple[float, ...] = field(repr=False)
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ExpectedDocumentChunk:
    ordinal: int
    section: str | None
    text: str = field(repr=False)
    content_hash: str
    token_count: int
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ExpectedDocumentRepresentation:
    content: str = field(repr=False)
    chunks: tuple[ExpectedDocumentChunk, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class ImmutableDocumentRepresentation:
    identity: DocumentIdentity
    created_by_user_id: UUID
    title: str
    source_type: str
    source_name: str
    content: str = field(repr=False)
    chunks: tuple[ImmutableDocumentChunk, ...] = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class RetrievedDocumentChunk:
    document_id: UUID
    chunk_id: UUID
    source_name: str
    section: str | None
    ordinal: int
    cosine_distance: float
    text: str = field(repr=False)


@runtime_checkable
class DocumentRepositoryPort(Protocol):
    async def find_complete(
        self,
        *,
        tenant: TenantContext,
        identity: DocumentIdentity,
        expected: ExpectedDocumentRepresentation,
    ) -> UUID | None: ...

    async def persist(
        self,
        representation: ImmutableDocumentRepresentation,
    ) -> UUID: ...


@runtime_checkable
class DocumentRetrievalRepositoryPort(Protocol):
    async def search(
        self,
        *,
        tenant: TenantContext,
        allowed_document_ids: tuple[UUID, ...],
        embedding_model: str,
        query_embedding: tuple[float, ...],
        limit: int,
    ) -> tuple[RetrievedDocumentChunk, ...]: ...


AfterEmbeddingHook = Callable[[DocumentIdentity], Awaitable[None]]


async def _no_hook(_identity: DocumentIdentity) -> None:
    return None


def build_document_identity(
    *, workspace_id: UUID, source: PreparedIngestionSource
) -> DocumentIdentity:
    return DocumentIdentity(
        workspace_id=workspace_id,
        content_hash=source.content_hash,
        normalization_version=source.normalization_version,
        chunking_version=source.chunking_version,
    )


def build_document_expectation(source: PreparedIngestionSource) -> ExpectedDocumentRepresentation:
    return ExpectedDocumentRepresentation(
        content=source.content,
        chunks=tuple(
            ExpectedDocumentChunk(
                ordinal=chunk.ordinal,
                section=chunk.section,
                text=chunk.text,
                content_hash=chunk.content_hash,
                token_count=chunk.token_count,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            for chunk in source.chunks
        ),
    )


def partition_embedding_texts(texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise DocumentIngestionInvariantError
    return tuple(
        tuple(texts[index : index + MAX_EMBEDDING_BATCH_SIZE])
        for index in range(0, len(texts), MAX_EMBEDDING_BATCH_SIZE)
    )


@dataclass(frozen=True, slots=True, repr=False)
class DocumentIngestionService:
    repository: DocumentRepositoryPort
    embedding: EmbeddingPort
    after_embedding: AfterEmbeddingHook = _no_hook

    def __post_init__(self) -> None:
        if not isinstance(self.repository, DocumentRepositoryPort) or not isinstance(
            self.embedding, EmbeddingPort
        ):
            raise TypeError("document ingestion dependencies use the wrong contract")

    async def ingest(
        self,
        *,
        tenant: TenantContext,
        batch: PreparedIngestionBatch,
    ) -> tuple[UUID, ...]:
        results: list[UUID | None] = [None] * len(batch.sources)
        positions: dict[DocumentIdentity, list[int]] = {}
        unique_sources: dict[DocumentIdentity, PreparedIngestionSource] = {}
        for index, source in enumerate(batch.sources):
            identity = build_document_identity(workspace_id=tenant.workspace_id, source=source)
            positions.setdefault(identity, []).append(index)
            unique_sources.setdefault(identity, source)

        for identity, source in unique_sources.items():
            existing = await self.repository.find_complete(
                tenant=tenant,
                identity=identity,
                expected=build_document_expectation(source),
            )
            if existing is None:
                vectors: list[tuple[float, ...]] = []
                chunk_texts = tuple(chunk.text for chunk in source.chunks)
                for texts in partition_embedding_texts(chunk_texts):
                    result = await self.embedding.embed(
                        texts,
                        {"graph_node": INGESTION_GRAPH_NODE},
                    )
                    if len(result.vectors) != len(texts) or any(
                        len(vector) != LOCKED_EMBEDDING_DIMENSION for vector in result.vectors
                    ):
                        raise DocumentIngestionInvariantError
                    vectors.extend(result.vectors)
                if len(vectors) != len(source.chunks):
                    raise DocumentIngestionInvariantError
                await self.after_embedding(identity)
                representation = ImmutableDocumentRepresentation(
                    identity=identity,
                    created_by_user_id=tenant.actor_user_id,
                    title=source.title,
                    source_type=source.source_type,
                    source_name=source.source_name,
                    content=source.content,
                    chunks=tuple(
                        ImmutableDocumentChunk(
                            ordinal=chunk.ordinal,
                            section=chunk.section,
                            text=chunk.text,
                            content_hash=chunk.content_hash,
                            token_count=chunk.token_count,
                            embedding=vector,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                        )
                        for chunk, vector in zip(source.chunks, vectors, strict=True)
                    ),
                )
                existing = await self.repository.persist(representation)
            for index in positions[identity]:
                results[index] = existing

        if any(result is None for result in results):
            raise DocumentIngestionInvariantError
        return tuple(result for result in results if result is not None)


def _truncate_utf8(text: str, byte_limit: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


@dataclass(frozen=True, slots=True, repr=False)
class DocumentRetrievalService:
    repository: DocumentRetrievalRepositoryPort
    embedding: EmbeddingPort

    def __post_init__(self) -> None:
        if not isinstance(self.repository, DocumentRetrievalRepositoryPort) or not isinstance(
            self.embedding, EmbeddingPort
        ):
            raise TypeError("document retrieval dependencies use the wrong contract")

    async def retrieve(
        self,
        *,
        tenant: TenantContext,
        query: str,
        allowed_document_ids: Sequence[UUID],
    ) -> tuple[RetrievedDocumentChunk, ...]:
        if (
            not isinstance(tenant, TenantContext)
            or not isinstance(query, str)
            or not query.strip()
            or isinstance(allowed_document_ids, str | bytes)
            or not isinstance(allowed_document_ids, Sequence)
            or any(not isinstance(document_id, UUID) for document_id in allowed_document_ids)
        ):
            raise DocumentRetrievalInvariantError

        allowlist = tuple(dict.fromkeys(allowed_document_ids))
        scope = current_trace_scope()
        started_at = monotonic()
        context = (
            None
            if scope is None
            else start_trace_span(
                scope,
                span_kind="retrieval",
                metadata={
                    "embedding_profile": EMBEDDING_PROFILE,
                    "top_k": RETRIEVAL_TOP_K,
                    "allowed_document_count": len(allowlist),
                },
            )
        )
        status: SpanStatus = "failed"
        error_category = "retrieval_failed"
        result: tuple[RetrievedDocumentChunk, ...] = ()
        try:
            with bind_trace_scope(None if scope is None else replace(scope, parent=context)):
                result = await self._retrieve_bounded(
                    tenant=tenant, query=query, allowlist=allowlist
                )
            status = "succeeded" if result else "no_result"
            return result
        except DocumentRetrievalInvariantError:
            error_category = "retrieval_invariant_failed"
            raise
        except asyncio.CancelledError:
            status, error_category = "cancelled", "retrieval_cancelled"
            raise
        finally:
            if scope is not None:
                finish_trace_span(
                    scope,
                    context,
                    started_at=started_at,
                    status=status,
                    error_category=error_category if status in {"failed", "cancelled"} else None,
                    metadata={
                        "returned_chunk_count": len(result),
                        "context_bytes": sum(len(hit.text.encode("utf-8")) for hit in result),
                    }
                    if status in {"succeeded", "no_result"}
                    else {},
                )

    async def _retrieve_bounded(
        self, *, tenant: TenantContext, query: str, allowlist: tuple[UUID, ...]
    ) -> tuple[RetrievedDocumentChunk, ...]:
        if not allowlist:
            return ()

        embedded = await self.embedding.embed(
            (query,),
            {"graph_node": RETRIEVAL_GRAPH_NODE},
        )
        if len(embedded.vectors) != 1:
            raise DocumentRetrievalInvariantError
        vector = embedded.vectors[0]
        if len(vector) != LOCKED_EMBEDDING_DIMENSION or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
            for value in vector
        ):
            raise DocumentRetrievalInvariantError

        hits = await self.repository.search(
            tenant=tenant,
            allowed_document_ids=allowlist,
            embedding_model=EMBEDDING_PROFILE,
            query_embedding=tuple(float(value) for value in vector),
            limit=RETRIEVAL_TOP_K,
        )
        if not isinstance(hits, tuple) or len(hits) > RETRIEVAL_TOP_K:
            raise DocumentRetrievalInvariantError

        bounded: list[RetrievedDocumentChunk] = []
        remaining = RETRIEVAL_CONTEXT_BYTE_BUDGET
        for hit in hits:
            if (
                not isinstance(hit, RetrievedDocumentChunk)
                or not isinstance(hit.document_id, UUID)
                or not isinstance(hit.chunk_id, UUID)
                or not isinstance(hit.source_name, str)
                or not hit.source_name
                or (hit.section is not None and not isinstance(hit.section, str))
                or isinstance(hit.ordinal, bool)
                or not isinstance(hit.ordinal, int)
                or hit.ordinal < 0
                or isinstance(hit.cosine_distance, bool)
                or not isinstance(hit.cosine_distance, int | float)
                or not isfinite(float(hit.cosine_distance))
                or not isinstance(hit.text, str)
            ):
                raise DocumentRetrievalInvariantError
            text = _truncate_utf8(
                hit.text,
                min(RETRIEVED_CHUNK_TEXT_BYTE_LIMIT, remaining),
            )
            if not text:
                break
            bounded.append(
                RetrievedDocumentChunk(
                    document_id=hit.document_id,
                    chunk_id=hit.chunk_id,
                    source_name=hit.source_name,
                    section=hit.section,
                    ordinal=hit.ordinal,
                    cosine_distance=float(hit.cosine_distance),
                    text=text,
                )
            )
            remaining -= len(text.encode("utf-8"))
            if remaining == 0:
                break
        return tuple(bounded)
