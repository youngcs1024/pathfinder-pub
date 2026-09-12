from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import (
    Document,
    DocumentChunk,
    LLMInvocation,
    RunEvent,
    ToolInvocation,
    WorkspaceMembership,
)
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory, transaction
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import EmbeddingResult, ProviderAdapterError
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentIdentity,
    DocumentRetrievalAuthorizationError,
    DocumentRetrievalService,
    ImmutableDocumentChunk,
    ImmutableDocumentRepresentation,
)

pytestmark = pytest.mark.integration


async def _runtime(database_url: str, subject: str):  # type: ignore[no-untyped-def]
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    actor = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(subject)
    tenant = TenantContext(actor.workspace_id, actor.user_id, actor.role)
    return engine, sessions, actor, tenant


async def _persist_vectors(
    repository: SqlAlchemyDocumentRepository,
    *,
    workspace_id: UUID,
    actor_user_id: UUID,
    vectors: tuple[tuple[float, ...], ...],
    source_name: str,
    profile: str = EMBEDDING_PROFILE,
) -> UUID:
    marker = source_name.encode("utf-8").hex()[:32].ljust(32, "0")
    content_hash = (marker * 4)[:64]
    chunks = tuple(
        ImmutableDocumentChunk(
            ordinal=index,
            section=None,
            text=f"{source_name} evidence {index}",
            content_hash=f"{index + 1:064x}",
            token_count=len(f"{source_name} evidence {index}".encode()),
            embedding=vector,
        )
        for index, vector in enumerate(vectors)
    )
    return await repository.persist(
        ImmutableDocumentRepresentation(
            identity=DocumentIdentity(
                workspace_id=workspace_id,
                content_hash=content_hash,
                normalization_version="nfc-lf-v1",
                chunking_version="heading-paragraph-utf8-budget-800-v1",
                embedding_model=profile,
            ),
            created_by_user_id=actor_user_id,
            title=source_name,
            source_type="markdown",
            source_name=source_name,
            content="\n".join(chunk.text for chunk in chunks),
            chunks=chunks,
        )
    )


class _VectorEmbedding:
    provider = "fake"
    model = "text-embedding-v4"

    def __init__(self, vector: tuple[float, ...]) -> None:
        self.vector = vector
        self.calls = 0

    async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
        self.calls += 1
        assert tuple(texts) == ("query",)
        assert metadata == {"graph_node": "document_retrieval"}
        return EmbeddingResult(vectors=(self.vector,))


def _service(
    sessions,
    tenant: TenantContext,
    adapter,
    *,
    repository=None,
):  # type: ignore[no-untyped-def]
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=adapter,
        provider="fake",
    )
    return DocumentRetrievalService(
        repository or SqlAlchemyDocumentRepository(sessions),
        factory.create_embedding_model(
            LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
        ),
    )


async def _fact_counts(sessions):  # type: ignore[no-untyped-def]
    async with sessions() as session:
        counts = []
        for model in (Document, DocumentChunk, LLMInvocation, ToolInvocation, RunEvent):
            counts.append(await session.scalar(select(func.count()).select_from(model)))
        return tuple(counts)


class _SearchSpyRepository:
    def __init__(self, delegate: SqlAlchemyDocumentRepository) -> None:
        self.delegate = delegate
        self.calls = 0

    async def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return await self.delegate.search(**kwargs)


async def test_factory_success_accounting_has_exact_tenant_context(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor, tenant = await _runtime(migrated_database_url, "retrieval-accounting")
    e1 = (1.0,) + (0.0,) * 1535
    repository = SqlAlchemyDocumentRepository(sessions)
    document_id = await _persist_vectors(
        repository,
        workspace_id=actor.workspace_id,
        actor_user_id=actor.user_id,
        vectors=(e1,),
        source_name="accounting.md",
    )
    try:
        result = await _service(sessions, tenant, _VectorEmbedding(e1)).retrieve(
            tenant=tenant,
            query="query",
            allowed_document_ids=(document_id,),
        )
        assert len(result) == 1
        assert await _fact_counts(sessions) == (1, 1, 1, 0, 0)
        async with sessions() as session:
            invocation = await session.scalar(select(LLMInvocation))
        assert invocation is not None
        assert (
            invocation.workspace_id,
            invocation.actor_user_id,
            invocation.run_id,
            invocation.invocation_kind,
            invocation.status,
            invocation.graph_node,
            invocation.error_category,
        ) == (
            tenant.workspace_id,
            tenant.actor_user_id,
            None,
            "embedding",
            "succeeded",
            "document_retrieval",
            None,
        )
    finally:
        await engine.dispose()


async def test_exact_cosine_ranking_ties_allowlist_profile_and_duplicate_query(
    migrated_database_url: str,
) -> None:
    engine, sessions, _actor, tenant = await _runtime(migrated_database_url, "retrieval-ranking")
    repository = SqlAlchemyDocumentRepository(sessions)
    e1 = (1.0,) + (0.0,) * 1535
    orthogonal = (0.0, 1.0) + (0.0,) * 1534
    negative = (-1.0,) + (0.0,) * 1535
    try:
        ranked_document = await _persist_vectors(
            repository,
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            vectors=(e1, orthogonal, negative),
            source_name="ranked.md",
        )
        closer_but_disallowed = await _persist_vectors(
            repository,
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            vectors=(e1,),
            source_name="disallowed.md",
        )
        legacy_document = await _persist_vectors(
            repository,
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            vectors=(e1,),
            source_name="legacy.md",
            profile="legacy-embedding-profile-v0",
        )

        direct = await repository.search(
            tenant=tenant,
            allowed_document_ids=(ranked_document, closer_but_disallowed, legacy_document),
            embedding_model=EMBEDDING_PROFILE,
            query_embedding=e1,
            limit=5,
        )
        assert [round(hit.cosine_distance, 8) for hit in direct] == [0.0, 0.0, 1.0, 2.0]
        assert {hit.document_id for hit in direct[:2]} == {
            ranked_document,
            closer_but_disallowed,
        }
        assert [hit.document_id for hit in direct[2:]] == [ranked_document, ranked_document]
        assert all(hit.document_id != legacy_document for hit in direct)

        repeated = [
            await repository.search(
                tenant=tenant,
                allowed_document_ids=(ranked_document, closer_but_disallowed),
                embedding_model=EMBEDDING_PROFILE,
                query_embedding=e1,
                limit=5,
            )
            for _ in range(3)
        ]
        assert repeated[0] == repeated[1] == repeated[2]
        tied = repeated[0][:2]
        assert [(hit.document_id, hit.ordinal, hit.chunk_id) for hit in tied] == sorted(
            (hit.document_id, hit.ordinal, hit.chunk_id) for hit in tied
        )

        only_ranked = await repository.search(
            tenant=tenant,
            allowed_document_ids=(ranked_document,),
            embedding_model=EMBEDDING_PROFILE,
            query_embedding=e1,
            limit=5,
        )
        assert all(hit.document_id == ranked_document for hit in only_ranked)
        assert [round(hit.cosine_distance, 8) for hit in only_ranked] == [0.0, 1.0, 2.0]
        assert (
            await repository.search(
                tenant=tenant,
                allowed_document_ids=(uuid4(),),
                embedding_model=EMBEDDING_PROFILE,
                query_embedding=e1,
                limit=5,
            )
            == ()
        )

        service = _service(sessions, tenant, _VectorEmbedding(e1))
        first = await service.retrieve(
            tenant=tenant, query="query", allowed_document_ids=(ranked_document,)
        )
        second = await service.retrieve(
            tenant=tenant, query="query", allowed_document_ids=(ranked_document,)
        )
        assert first == second
        assert (await _fact_counts(sessions))[2:] == (2, 0, 0)
    finally:
        await engine.dispose()


async def test_cross_workspace_filter_and_same_workspace_noncreator_share(
    migrated_database_url: str,
) -> None:
    engine, sessions, owner, owner_tenant = await _runtime(migrated_database_url, "retrieval-owner")
    second = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace("retrieval-second")
    repository = SqlAlchemyDocumentRepository(sessions)
    e1 = (1.0,) + (0.0,) * 1535
    try:
        owner_document = await _persist_vectors(
            repository,
            workspace_id=owner.workspace_id,
            actor_user_id=owner.user_id,
            vectors=((0.0, 1.0) + (0.0,) * 1534,),
            source_name="owner.md",
        )
        foreign_document = await _persist_vectors(
            repository,
            workspace_id=second.workspace_id,
            actor_user_id=second.user_id,
            vectors=(e1,),
            source_name="FOREIGN-CANARY.md",
        )
        async with transaction(sessions) as session:
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=owner.workspace_id,
                    user_id=second.user_id,
                    role="member",
                )
            )
        noncreator = TenantContext(owner.workspace_id, second.user_id, WorkspaceRole.MEMBER)

        hits = await repository.search(
            tenant=noncreator,
            allowed_document_ids=(foreign_document, owner_document),
            embedding_model=EMBEDDING_PROFILE,
            query_embedding=e1,
            limit=5,
        )
        assert [hit.document_id for hit in hits] == [owner_document]
        assert all("FOREIGN-CANARY" not in hit.source_name + hit.text for hit in hits)

        foreign_only = await repository.search(
            tenant=owner_tenant,
            allowed_document_ids=(foreign_document,),
            embedding_model=EMBEDDING_PROFILE,
            query_embedding=e1,
            limit=5,
        )
        assert foreign_only == ()
    finally:
        await engine.dispose()


async def test_stale_tenant_and_revocation_after_embedding_fail_closed(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor, tenant = await _runtime(migrated_database_url, "retrieval-revocation")
    repository = SqlAlchemyDocumentRepository(sessions)
    e1 = (1.0,) + (0.0,) * 1535
    document_id = await _persist_vectors(
        repository,
        workspace_id=actor.workspace_id,
        actor_user_id=actor.user_id,
        vectors=(e1,),
        source_name="private.md",
    )

    class _RevokingEmbedding(_VectorEmbedding):
        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            result = await super().embed(texts, metadata, attempt=attempt)
            async with transaction(sessions) as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(WorkspaceMembership.id == actor.membership_id)
                    .values(revoked_at=datetime.now(UTC))
                )
            return result

    try:
        with pytest.raises(DocumentRetrievalAuthorizationError):
            await _service(sessions, tenant, _RevokingEmbedding(e1)).retrieve(
                tenant=tenant,
                query="query",
                allowed_document_ids=(document_id,),
            )
        assert await _fact_counts(sessions) == (1, 1, 1, 0, 0)
        async with sessions() as session:
            invocation = await session.scalar(select(LLMInvocation))
        assert invocation is not None
        assert (invocation.status, invocation.graph_node) == (
            "succeeded",
            "document_retrieval",
        )

        with pytest.raises(DocumentRetrievalAuthorizationError):
            await repository.search(
                tenant=tenant,
                allowed_document_ids=(document_id,),
                embedding_model=EMBEDDING_PROFILE,
                query_embedding=e1,
                limit=5,
            )
    finally:
        await engine.dispose()


async def test_factory_retry_is_accounted_before_one_search(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor, tenant = await _runtime(migrated_database_url, "retrieval-retry")
    e1 = (1.0,) + (0.0,) * 1535
    repository = SqlAlchemyDocumentRepository(sessions)
    document_id = await _persist_vectors(
        repository,
        workspace_id=actor.workspace_id,
        actor_user_id=actor.user_id,
        vectors=(e1,),
        source_name="retry.md",
    )

    class _RetryEmbedding(_VectorEmbedding):
        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise ProviderAdapterError(category="rate_limited", retryable=True)
            return EmbeddingResult(vectors=(self.vector,))

    class _SpyRepository:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return await repository.search(**kwargs)

    async def no_wait(_delay: float) -> None:
        return None

    adapter = _RetryEmbedding(e1)
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=adapter,
        provider="fake",
        sleeper=no_wait,
        random_source=lambda: 0.0,
    )
    spy = _SpyRepository()
    service = DocumentRetrievalService(
        spy,
        factory.create_embedding_model(LLMInvocationContext(actor.workspace_id, actor.user_id)),
    )
    try:
        result = await service.retrieve(
            tenant=tenant, query="query", allowed_document_ids=(document_id,)
        )
        assert len(result) == 1
        assert spy.calls == 1
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(LLMInvocation.status, LLMInvocation.error_category).order_by(
                        LLMInvocation.created_at
                    )
                )
            ).all()
        assert rows == [("failed", "rate_limited"), ("succeeded", None)]
    finally:
        await engine.dispose()


async def test_cancellation_and_post_embedding_db_failure_preserve_business_rows(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor, tenant = await _runtime(migrated_database_url, "retrieval-failures")
    e1 = (1.0,) + (0.0,) * 1535
    repository = SqlAlchemyDocumentRepository(sessions)
    document_id = await _persist_vectors(
        repository,
        workspace_id=actor.workspace_id,
        actor_user_id=actor.user_id,
        vectors=(e1,),
        source_name="failure.md",
    )

    class _CancelledEmbedding(_VectorEmbedding):
        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    class _FailingRepository:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            raise RuntimeError("synthetic database failure")

    try:
        before = await _fact_counts(sessions)
        cancelled_spy = _SearchSpyRepository(repository)
        with pytest.raises(asyncio.CancelledError):
            await _service(
                sessions,
                tenant,
                _CancelledEmbedding(e1),
                repository=cancelled_spy,
            ).retrieve(tenant=tenant, query="query", allowed_document_ids=(document_id,))
        assert cancelled_spy.calls == 0
        after_cancel = await _fact_counts(sessions)
        assert after_cancel == (before[0], before[1], before[2] + 1, 0, 0)

        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=FakeChatModel(),
            embedding_adapter=_VectorEmbedding(e1),
            provider="fake",
        )
        failing = _FailingRepository()
        with pytest.raises(RuntimeError, match="synthetic database failure"):
            await DocumentRetrievalService(
                failing,
                factory.create_embedding_model(
                    LLMInvocationContext(actor.workspace_id, actor.user_id)
                ),
            ).retrieve(tenant=tenant, query="query", allowed_document_ids=(document_id,))
        assert failing.calls == 1
        assert await _fact_counts(sessions) == (
            before[0],
            before[1],
            before[2] + 2,
            0,
            0,
        )
        async with sessions() as session:
            states = (
                await session.execute(
                    select(LLMInvocation.status, LLMInvocation.error_category).order_by(
                        LLMInvocation.created_at
                    )
                )
            ).all()
        assert states == [("failed", "cancelled"), ("succeeded", None)]
    finally:
        await engine.dispose()
