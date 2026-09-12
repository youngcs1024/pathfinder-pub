from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import Document, DocumentChunk, LLMInvocation, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory, transaction
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory, LLMProviderError
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationAuthorizationError, LLMInvocationContext
from app.llm.ports import EmbeddingResult, ProviderAdapterError
from app.retrieval.chunking import PreparedIngestionBatch, normalize_and_chunk_batch
from app.retrieval.documents import (
    DocumentIngestionAuthorizationError,
    DocumentIngestionService,
    ImmutableDocumentChunk,
    ImmutableDocumentRepresentation,
    build_document_identity,
)
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource

pytestmark = pytest.mark.integration


def _prepared(*, name: str = "resume.md", text: str = "# Resume\n\nPython backend engineer."):
    return normalize_and_chunk_batch(
        ValidatedIngestionBatch(
            (
                ValidatedIngestionSource(
                    source_name=name,
                    source_type="markdown" if name.endswith(".md") else "text",
                    title=name.rsplit(".", 1)[0],
                    raw_text=text,
                    character_count=len(text),
                ),
            )
        )
    )


async def _runtime(database_url: str, subject: str):  # type: ignore[no-untyped-def]
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    provisioned = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(subject)
    return engine, sessions, provisioned


def _service(sessions, workspace_id: UUID, actor_id: UUID, *, adapter=None, hook=None):  # type: ignore[no-untyped-def]
    embedding_adapter = adapter or FakeEmbeddingModel()
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=FakeChatModel(),
        embedding_adapter=embedding_adapter,
        provider="fake",
    )
    kwargs = {} if hook is None else {"after_embedding": hook}
    return DocumentIngestionService(
        SqlAlchemyDocumentRepository(sessions),
        factory.create_embedding_model(LLMInvocationContext(workspace_id, actor_id)),
        **kwargs,
    )


async def _counts(sessions):  # type: ignore[no-untyped-def]
    async with sessions() as session:
        counts = []
        for model in (Document, DocumentChunk, LLMInvocation):
            counts.append(await session.scalar(select(func.count()).select_from(model)))
        return tuple(counts)


async def test_happy_path_duplicate_and_first_seen_metadata(migrated_database_url: str) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-happy")
    try:
        first_batch = _prepared(name="first.md")
        first = await _service(sessions, actor.workspace_id, actor.user_id).ingest(
            tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
            batch=first_batch,
        )
        second = await _service(sessions, actor.workspace_id, actor.user_id).ingest(
            tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
            batch=_prepared(name="renamed.md"),
        )
        assert first == second
        assert await _counts(sessions) == (1, len(first_batch.sources[0].chunks), 1)
        async with sessions() as session:
            document = await session.scalar(select(Document))
            chunks = (
                await session.scalars(select(DocumentChunk).order_by(DocumentChunk.ordinal))
            ).all()
        assert document is not None
        assert (document.title, document.source_name, document.created_by_user_id) == (
            "first",
            "first.md",
            actor.user_id,
        )
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        assert all(len(chunk.embedding) == 1536 for chunk in chunks)
    finally:
        await engine.dispose()


async def test_provider_failure_and_cancellation_leave_no_business_rows(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-provider-failures")

    class _Failure:
        provider = "fake"
        model = "text-embedding-v4"

        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            raise ProviderAdapterError(category="provider_rejected", retryable=False)

    class _Cancellation:
        provider = "fake"
        model = "text-embedding-v4"

        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    try:
        with pytest.raises(LLMProviderError):
            await _service(sessions, actor.workspace_id, actor.user_id, adapter=_Failure()).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(),
            )
        assert await _counts(sessions) == (0, 0, 1)
        with pytest.raises(asyncio.CancelledError):
            await _service(
                sessions, actor.workspace_id, actor.user_id, adapter=_Cancellation()
            ).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(),
            )
        assert await _counts(sessions) == (0, 0, 2)
        async with sessions() as session:
            states = (
                await session.scalars(
                    select(LLMInvocation.error_category).order_by(LLMInvocation.created_at)
                )
            ).all()
        assert states == ["provider_rejected", "cancelled"]
    finally:
        await engine.dispose()


async def test_provider_retry_records_each_attempt_before_single_commit(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-retry")

    class _QwenChat:
        provider = "qwen"
        model = "qwen3.6-flash-2026-04-16"

        async def invoke(self, messages, tools, metadata, *, attempt):  # type: ignore[no-untyped-def]
            raise AssertionError("chat is not used by ingestion")

    class _RetryEmbedding:
        provider = "qwen"
        model = "text-embedding-v4"

        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts, metadata, *, attempt):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise ProviderAdapterError(category="rate_limited", retryable=True)
            return EmbeddingResult(
                vectors=tuple((0.25,) * 1536 for _ in texts),
                provider="qwen",
            )

    async def no_wait(_delay: float) -> None:
        return None

    adapter = _RetryEmbedding()
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(sessions),
        chat_adapter=_QwenChat(),
        embedding_adapter=adapter,
        provider="qwen",
        sleeper=no_wait,
        random_source=lambda: 0.0,
    )
    service = DocumentIngestionService(
        SqlAlchemyDocumentRepository(sessions),
        factory.create_embedding_model(LLMInvocationContext(actor.workspace_id, actor.user_id)),
    )
    try:
        await service.ingest(
            tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
            batch=_prepared(),
        )
        assert adapter.calls == 2
        assert await _counts(sessions) == (1, 1, 2)
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


async def test_revocation_between_embedding_batches_stops_next_attempt(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-batch-revocation")
    text = "\n\n".join(f"paragraph {index} " + ("x" * 780) for index in range(11))
    batch = _prepared(text=text)
    assert len(batch.sources[0].chunks) == 11

    class _RevokingEmbedding(FakeEmbeddingModel):
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts, metadata, *, attempt=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            result = await super().embed(texts, metadata, attempt=attempt)
            if self.calls == 1:
                async with transaction(sessions) as session:
                    await session.execute(
                        update(WorkspaceMembership)
                        .where(WorkspaceMembership.id == actor.membership_id)
                        .values(revoked_at=datetime.now(UTC))
                    )
            return result

    adapter = _RevokingEmbedding()
    try:
        with pytest.raises(LLMInvocationAuthorizationError):
            await _service(sessions, actor.workspace_id, actor.user_id, adapter=adapter).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=batch,
            )
        assert adapter.calls == 1
        assert await _counts(sessions) == (0, 0, 1)
    finally:
        await engine.dispose()


async def test_revocation_before_provider_and_before_persistence(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-revocation")

    class _Counting(FakeEmbeddingModel):
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts, metadata, *, attempt=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            return await super().embed(texts, metadata, attempt=attempt)

    try:
        async with transaction(sessions) as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.id == actor.membership_id)
                .values(revoked_at=datetime.now(UTC))
            )
        adapter = _Counting()
        with pytest.raises(DocumentIngestionAuthorizationError):
            await _service(sessions, actor.workspace_id, actor.user_id, adapter=adapter).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(),
            )
        assert adapter.calls == 0
        assert await _counts(sessions) == (0, 0, 0)

        async with transaction(sessions) as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.id == actor.membership_id)
                .values(revoked_at=None)
            )

        async def revoke(_identity):  # type: ignore[no-untyped-def]
            async with transaction(sessions) as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(WorkspaceMembership.id == actor.membership_id)
                    .values(revoked_at=datetime.now(UTC))
                )

        with pytest.raises(DocumentIngestionAuthorizationError):
            await _service(sessions, actor.workspace_id, actor.user_id, hook=revoke).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(),
            )
        assert await _counts(sessions) == (0, 0, 1)
    finally:
        await engine.dispose()


async def test_revoked_membership_cannot_use_complete_document_dedupe_fast_path(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-dedupe-revocation")
    tenant = TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN)

    class _Counting(FakeEmbeddingModel):
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts, metadata, *, attempt=None):  # type: ignore[no-untyped-def]
            self.calls += 1
            return await super().embed(texts, metadata, attempt=attempt)

    try:
        document_ids = await _service(sessions, actor.workspace_id, actor.user_id).ingest(
            tenant=tenant,
            batch=_prepared(),
        )
        before = await _counts(sessions)
        assert before == (1, 1, 1)

        async with transaction(sessions) as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.id == actor.membership_id)
                .values(revoked_at=datetime.now(UTC))
            )

        adapter = _Counting()
        with pytest.raises(DocumentIngestionAuthorizationError):
            await _service(sessions, actor.workspace_id, actor.user_id, adapter=adapter).ingest(
                tenant=tenant,
                batch=_prepared(name="renamed.md"),
            )

        assert document_ids
        assert adapter.calls == 0
        assert await _counts(sessions) == before
    finally:
        await engine.dispose()


async def test_crash_after_embedding_then_rerun_converges(migrated_database_url: str) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-crash")

    async def crash(_identity):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic crash boundary")

    try:
        with pytest.raises(RuntimeError, match="synthetic crash boundary"):
            await _service(sessions, actor.workspace_id, actor.user_id, hook=crash).ingest(
                tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(),
            )
        assert await _counts(sessions) == (0, 0, 1)
        ids = await _service(sessions, actor.workspace_id, actor.user_id).ingest(
            tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
            batch=_prepared(),
        )
        assert len(ids) == 1
        assert await _counts(sessions) == (1, 1, 2)
        rerun = await _service(sessions, actor.workspace_id, actor.user_id).ingest(
            tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
            batch=_prepared(),
        )
        assert rerun == ids
        assert await _counts(sessions) == (1, 1, 2)
    finally:
        await engine.dispose()


async def test_concurrent_duplicate_converges_after_both_embeddings(
    migrated_database_url: str,
) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-concurrent")
    arrived = 0
    release = asyncio.Event()
    lock = asyncio.Lock()

    async def barrier(_identity):  # type: ignore[no-untyped-def]
        nonlocal arrived
        async with lock:
            arrived += 1
            if arrived == 2:
                release.set()
        await release.wait()

    try:
        services = [
            _service(sessions, actor.workspace_id, actor.user_id, hook=barrier) for _ in range(2)
        ]
        results = await asyncio.gather(
            *(
                service.ingest(
                    tenant=TenantContext(actor.workspace_id, actor.user_id, WorkspaceRole.ADMIN),
                    batch=_prepared(),
                )
                for service in services
            )
        )
        assert results[0] == results[1]
        assert await _counts(sessions) == (1, 1, 2)
    finally:
        await engine.dispose()


async def test_transaction_chunk_failure_rolls_back_parent(migrated_database_url: str) -> None:
    engine, sessions, actor = await _runtime(migrated_database_url, "rag-rollback")
    source = _prepared().sources[0]
    identity = build_document_identity(workspace_id=actor.workspace_id, source=source)
    invalid = ImmutableDocumentRepresentation(
        identity=identity,
        created_by_user_id=actor.user_id,
        title=source.title,
        source_type=source.source_type,
        source_name=source.source_name,
        content=source.content,
        chunks=(
            ImmutableDocumentChunk(
                ordinal=0,
                section=None,
                text="chunk",
                content_hash="a" * 64,
                token_count=999,
                embedding=(0.0,) * 1536,
            ),
        ),
    )
    try:
        with pytest.raises(IntegrityError):
            await SqlAlchemyDocumentRepository(sessions).persist(invalid)
        assert await _counts(sessions) == (0, 0, 0)
    finally:
        await engine.dispose()


async def test_workspace_and_profile_version_are_identity_components(
    migrated_database_url: str,
) -> None:
    engine, sessions, first = await _runtime(migrated_database_url, "rag-identity-a")
    other_engine, other_sessions, second = await _runtime(migrated_database_url, "rag-identity-b")
    try:
        source = _prepared().sources[0]
        first_id = (
            await _service(sessions, first.workspace_id, first.user_id).ingest(
                tenant=TenantContext(first.workspace_id, first.user_id, WorkspaceRole.ADMIN),
                batch=PreparedIngestionBatch((source,)),
            )
        )[0]
        second_id = (
            await _service(other_sessions, second.workspace_id, second.user_id).ingest(
                tenant=TenantContext(second.workspace_id, second.user_id, WorkspaceRole.ADMIN),
                batch=PreparedIngestionBatch((source,)),
            )
        )[0]
        assert first_id != second_id

        repository = SqlAlchemyDocumentRepository(sessions)
        base_identity = build_document_identity(workspace_id=first.workspace_id, source=source)
        alternate_ids = []
        for alternate_identity in (
            replace(base_identity, normalization_version="alternate-normalization-v2"),
            replace(base_identity, chunking_version="alternate-chunking-v2"),
            replace(base_identity, embedding_model="alternate-profile-v2"),
        ):
            alternate_ids.append(
                await repository.persist(
                    ImmutableDocumentRepresentation(
                        identity=alternate_identity,
                        created_by_user_id=first.user_id,
                        title=source.title,
                        source_type=source.source_type,
                        source_name=source.source_name,
                        content=source.content,
                        chunks=tuple(
                            ImmutableDocumentChunk(
                                chunk.ordinal,
                                chunk.section,
                                chunk.text,
                                chunk.content_hash,
                                chunk.token_count,
                                (0.0,) * 1536,
                            )
                            for chunk in source.chunks
                        ),
                    )
                )
            )
        assert len(set(alternate_ids) | {first_id, second_id}) == 5
        async with sessions() as session:
            first_workspace_documents = await session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.workspace_id == first.workspace_id)
            )
        assert first_workspace_documents == 4
    finally:
        await engine.dispose()
        await other_engine.dispose()


async def test_different_actor_reuses_first_creator(migrated_database_url: str) -> None:
    engine, sessions, first = await _runtime(migrated_database_url, "rag-first-actor")
    other_engine, _other_sessions, second = await _runtime(
        migrated_database_url, "rag-second-actor"
    )
    try:
        async with transaction(sessions) as session:
            session.add(
                WorkspaceMembership(
                    id=uuid4(),
                    workspace_id=first.workspace_id,
                    user_id=second.user_id,
                    role="member",
                    revoked_at=None,
                )
            )
        first_id = (
            await _service(sessions, first.workspace_id, first.user_id).ingest(
                tenant=TenantContext(first.workspace_id, first.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(name="first.md"),
            )
        )[0]
        second_id = (
            await _service(sessions, first.workspace_id, second.user_id).ingest(
                tenant=TenantContext(first.workspace_id, second.user_id, WorkspaceRole.ADMIN),
                batch=_prepared(name="second.md"),
            )
        )[0]
        assert first_id == second_id
        async with sessions() as session:
            document = await session.scalar(select(Document))
        assert document is not None
        assert document.created_by_user_id == first.user_id
        assert document.source_name == "first.md"
        assert await _counts(sessions) == (1, 1, 1)
    finally:
        await engine.dispose()
        await other_engine.dispose()
