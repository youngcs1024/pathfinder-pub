from __future__ import annotations

from math import isfinite
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Document, DocumentChunk, WorkspaceMembership
from app.db.session import AsyncSessionFactory, transaction
from app.domain.tenancy import TenantContext
from app.retrieval.documents import (
    EMBEDDING_DIMENSION,
    EMBEDDING_PROFILE,
    RETRIEVAL_TOP_K,
    DocumentIdentity,
    DocumentIngestionAuthorizationError,
    DocumentIngestionInvariantError,
    DocumentRetrievalAuthorizationError,
    DocumentRetrievalInvariantError,
    ExpectedDocumentChunk,
    ExpectedDocumentRepresentation,
    ImmutableDocumentRepresentation,
    RetrievedDocumentChunk,
)


class SqlAlchemyDocumentRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def find_complete(
        self,
        *,
        tenant: TenantContext,
        identity: DocumentIdentity,
        expected: ExpectedDocumentRepresentation,
    ) -> UUID | None:
        if identity.workspace_id != tenant.workspace_id:
            raise DocumentIngestionAuthorizationError

        async with transaction(self._session_factory) as session:
            membership = await session.scalar(
                select(WorkspaceMembership.id)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
                .with_for_update(read=True)
            )
            if membership is None:
                raise DocumentIngestionAuthorizationError

            document = await session.scalar(
                select(Document).where(
                    Document.workspace_id == tenant.workspace_id,
                    Document.content_hash == identity.content_hash,
                    Document.normalization_version == identity.normalization_version,
                    Document.chunking_version == identity.chunking_version,
                    Document.embedding_model == identity.embedding_model,
                )
            )
            if document is None:
                return None
            chunks = (
                await session.scalars(
                    select(DocumentChunk)
                    .where(
                        DocumentChunk.workspace_id == tenant.workspace_id,
                        DocumentChunk.document_id == document.id,
                        DocumentChunk.embedding_model == identity.embedding_model,
                    )
                    .order_by(DocumentChunk.ordinal)
                )
            ).all()
            self._validate_existing(document, chunks, identity, expected)
            return document.id

    async def persist(self, representation: ImmutableDocumentRepresentation) -> UUID:
        identity = representation.identity
        async with transaction(self._session_factory) as session:
            membership = await session.scalar(
                select(WorkspaceMembership.id)
                .where(
                    WorkspaceMembership.workspace_id == identity.workspace_id,
                    WorkspaceMembership.user_id == representation.created_by_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
                .with_for_update()
            )
            if membership is None:
                raise DocumentIngestionAuthorizationError

            document_id = uuid4()
            inserted_id = await session.scalar(
                insert(Document)
                .values(
                    id=document_id,
                    workspace_id=identity.workspace_id,
                    created_by_user_id=representation.created_by_user_id,
                    title=representation.title,
                    source_type=representation.source_type,
                    source_name=representation.source_name,
                    content=representation.content,
                    content_hash=identity.content_hash,
                    normalization_version=identity.normalization_version,
                    chunking_version=identity.chunking_version,
                    embedding_model=identity.embedding_model,
                )
                .on_conflict_do_nothing(
                    index_elements=(
                        Document.workspace_id,
                        Document.content_hash,
                        Document.normalization_version,
                        Document.chunking_version,
                        Document.embedding_model,
                    )
                )
                .returning(Document.id)
            )
            if inserted_id is not None:
                session.add_all(
                    DocumentChunk(
                        id=uuid4(),
                        workspace_id=identity.workspace_id,
                        document_id=inserted_id,
                        ordinal=chunk.ordinal,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        section=chunk.section,
                        text=chunk.text,
                        content_hash=chunk.content_hash,
                        token_count=chunk.token_count,
                        embedding_model=identity.embedding_model,
                        embedding=list(chunk.embedding),
                    )
                    for chunk in representation.chunks
                )
                return inserted_id

            winner = await session.scalar(
                select(Document).where(
                    Document.workspace_id == identity.workspace_id,
                    Document.content_hash == identity.content_hash,
                    Document.normalization_version == identity.normalization_version,
                    Document.chunking_version == identity.chunking_version,
                    Document.embedding_model == identity.embedding_model,
                )
            )
            if winner is None:
                raise DocumentIngestionInvariantError
            winner_chunks = (
                await session.scalars(
                    select(DocumentChunk)
                    .where(
                        DocumentChunk.workspace_id == identity.workspace_id,
                        DocumentChunk.document_id == winner.id,
                        DocumentChunk.embedding_model == identity.embedding_model,
                    )
                    .order_by(DocumentChunk.ordinal)
                )
            ).all()
            expected = ExpectedDocumentRepresentation(
                content=representation.content,
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
                    for chunk in representation.chunks
                ),
            )
            self._validate_existing(winner, winner_chunks, identity, expected)
            return winner.id

    async def search(
        self,
        *,
        tenant: TenantContext,
        allowed_document_ids: tuple[UUID, ...],
        embedding_model: str,
        query_embedding: tuple[float, ...],
        limit: int,
    ) -> tuple[RetrievedDocumentChunk, ...]:
        if (
            not isinstance(tenant, TenantContext)
            or not isinstance(allowed_document_ids, tuple)
            or not allowed_document_ids
            or any(not isinstance(document_id, UUID) for document_id in allowed_document_ids)
            or embedding_model != EMBEDDING_PROFILE
            or not isinstance(query_embedding, tuple)
            or len(query_embedding) != EMBEDDING_DIMENSION
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(float(value))
                for value in query_embedding
            )
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= RETRIEVAL_TOP_K
        ):
            raise DocumentRetrievalInvariantError

        async with transaction(self._session_factory) as session:
            membership = await session.scalar(
                select(WorkspaceMembership.id)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
                .with_for_update(read=True)
            )
            if membership is None:
                raise DocumentRetrievalAuthorizationError

            distance = DocumentChunk.embedding.cosine_distance(list(query_embedding)).label(
                "cosine_distance"
            )
            rows = (
                await session.execute(
                    select(
                        DocumentChunk.document_id,
                        DocumentChunk.id,
                        Document.source_name,
                        DocumentChunk.section,
                        DocumentChunk.ordinal,
                        distance,
                        DocumentChunk.text,
                    )
                    .join(
                        Document,
                        (Document.workspace_id == DocumentChunk.workspace_id)
                        & (Document.id == DocumentChunk.document_id)
                        & (Document.embedding_model == DocumentChunk.embedding_model),
                    )
                    .where(
                        DocumentChunk.workspace_id == tenant.workspace_id,
                        Document.workspace_id == tenant.workspace_id,
                        DocumentChunk.document_id.in_(allowed_document_ids),
                        DocumentChunk.embedding_model == embedding_model,
                        Document.embedding_model == embedding_model,
                    )
                    .order_by(
                        distance.asc(),
                        DocumentChunk.document_id.asc(),
                        DocumentChunk.ordinal.asc(),
                        DocumentChunk.id.asc(),
                    )
                    .limit(limit)
                )
            ).all()

        hits: list[RetrievedDocumentChunk] = []
        for row in rows:
            cosine_distance = float(row.cosine_distance)
            if not isfinite(cosine_distance):
                raise DocumentRetrievalInvariantError
            hits.append(
                RetrievedDocumentChunk(
                    document_id=row.document_id,
                    chunk_id=row.id,
                    source_name=row.source_name,
                    section=row.section,
                    ordinal=row.ordinal,
                    cosine_distance=cosine_distance,
                    text=row.text,
                )
            )
        return tuple(hits)

    @staticmethod
    def _validate_existing(
        document: Document,
        chunks: list[DocumentChunk],
        identity: DocumentIdentity,
        expected: ExpectedDocumentRepresentation,
    ) -> None:
        expected_chunks = expected.chunks
        if (
            document.workspace_id != identity.workspace_id
            or document.content != expected.content
            or document.content_hash != identity.content_hash
            or document.normalization_version != identity.normalization_version
            or document.chunking_version != identity.chunking_version
            or document.embedding_model != identity.embedding_model
            or len(chunks) != len(expected_chunks)
            or any(
                (
                    actual.ordinal,
                    actual.section,
                    actual.text,
                    actual.content_hash,
                    actual.token_count,
                    actual.start_line,
                    actual.end_line,
                    actual.embedding_model,
                )
                != (
                    wanted.ordinal,
                    wanted.section,
                    wanted.text,
                    wanted.content_hash,
                    wanted.token_count,
                    wanted.start_line,
                    wanted.end_line,
                    identity.embedding_model,
                )
                for actual, wanted in zip(chunks, expected_chunks, strict=True)
            )
        ):
            raise DocumentIngestionInvariantError
