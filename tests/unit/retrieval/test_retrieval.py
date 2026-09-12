from __future__ import annotations

import asyncio
from math import inf
from uuid import UUID, uuid4

import pytest

from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.ports import EmbeddingResult
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    RETRIEVAL_CONTEXT_BYTE_BUDGET,
    RETRIEVAL_TOP_K,
    DocumentRetrievalInvariantError,
    DocumentRetrievalService,
    RetrievedDocumentChunk,
)


def _tenant() -> TenantContext:
    return TenantContext(uuid4(), uuid4(), WorkspaceRole.MEMBER)


def _hit(
    *,
    text: str = "evidence",
    distance: float = 0.25,
    document_id: UUID | None = None,
    ordinal: int = 0,
) -> RetrievedDocumentChunk:
    return RetrievedDocumentChunk(
        document_id=document_id or uuid4(),
        chunk_id=uuid4(),
        source_name="resume.md",
        section="Experience",
        ordinal=ordinal,
        cosine_distance=distance,
        text=text,
    )


class _Embedding:
    def __init__(self, result: object | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.result = result or EmbeddingResult(vectors=((1.0,) + (0.0,) * 1535,))

    async def embed(self, texts, metadata):  # type: ignore[no-untyped-def]
        self.calls.append((tuple(texts), dict(metadata)))
        return self.result


class _Repository:
    def __init__(self, hits: tuple[RetrievedDocumentChunk, ...] = ()) -> None:
        self.hits = hits
        self.calls: list[dict[str, object]] = []

    async def search(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return self.hits


async def test_valid_request_dedupes_allowlist_and_preserves_repository_order() -> None:
    tenant = _tenant()
    first_document = uuid4()
    second_document = uuid4()
    hits = (
        _hit(document_id=first_document, distance=0.1, ordinal=2),
        _hit(document_id=second_document, distance=0.2, ordinal=0),
    )
    embedding = _Embedding()
    repository = _Repository(hits)

    result = await DocumentRetrievalService(repository, embedding).retrieve(
        tenant=tenant,
        query="backend experience",
        allowed_document_ids=(first_document, second_document, first_document),
    )

    assert result == hits
    assert embedding.calls == [(("backend experience",), {"graph_node": "document_retrieval"})]
    assert len(repository.calls) == 1
    assert repository.calls[0] == {
        "tenant": tenant,
        "allowed_document_ids": (first_document, second_document),
        "embedding_model": EMBEDDING_PROFILE,
        "query_embedding": (1.0,) + (0.0,) * 1535,
        "limit": RETRIEVAL_TOP_K,
    }


@pytest.mark.parametrize("query", ["", "  \n"])
async def test_empty_query_rejects_without_embedding(query: str) -> None:
    embedding = _Embedding()
    repository = _Repository()
    with pytest.raises(DocumentRetrievalInvariantError):
        await DocumentRetrievalService(repository, embedding).retrieve(
            tenant=_tenant(),
            query=query,
            allowed_document_ids=(uuid4(),),
        )
    assert embedding.calls == []
    assert repository.calls == []


async def test_empty_allowlist_returns_empty_without_embedding() -> None:
    embedding = _Embedding()
    repository = _Repository((_hit(),))

    result = await DocumentRetrievalService(repository, embedding).retrieve(
        tenant=_tenant(),
        query="valid query",
        allowed_document_ids=(),
    )

    assert result == ()
    assert embedding.calls == []
    assert repository.calls == []


@pytest.mark.parametrize(
    "vectors",
    [
        (),
        ((1.0,) * 1536, (1.0,) * 1536),
        ((1.0,) * 1535,),
        ((inf,) + (1.0,) * 1535,),
    ],
)
async def test_malformed_embedding_fails_closed_before_repository(vectors: object) -> None:
    malformed = type("MalformedEmbeddingResult", (), {"vectors": vectors})()
    repository = _Repository()
    with pytest.raises(DocumentRetrievalInvariantError):
        await DocumentRetrievalService(repository, _Embedding(malformed)).retrieve(
            tenant=_tenant(),
            query="valid query",
            allowed_document_ids=(uuid4(),),
        )
    assert repository.calls == []


async def test_result_count_and_utf8_context_are_bounded() -> None:
    chinese_emoji = "履历🧭" * 400
    repository = _Repository(tuple(_hit(text=chinese_emoji, ordinal=index) for index in range(5)))

    result = await DocumentRetrievalService(repository, _Embedding()).retrieve(
        tenant=_tenant(),
        query="valid query",
        allowed_document_ids=(uuid4(),),
    )

    assert len(result) <= RETRIEVAL_TOP_K
    assert sum(len(hit.text.encode("utf-8")) for hit in result) <= (RETRIEVAL_CONTEXT_BYTE_BUDGET)
    assert all(len(hit.text.encode("utf-8")) <= 800 for hit in result)
    assert all(hit.text.encode("utf-8").decode("utf-8") == hit.text for hit in result)
    assert all(not hit.text.endswith("�") for hit in result)


async def test_repository_over_return_fails_closed() -> None:
    repository = _Repository(tuple(_hit(ordinal=index) for index in range(6)))
    with pytest.raises(DocumentRetrievalInvariantError):
        await DocumentRetrievalService(repository, _Embedding()).retrieve(
            tenant=_tenant(),
            query="valid query",
            allowed_document_ids=(uuid4(),),
        )


def test_sensitive_text_is_absent_from_repr_and_errors() -> None:
    canary = "PRIVATE-RESUME-CANARY"
    hit = _hit(text=canary)

    assert canary not in repr(hit)
    assert canary not in str(DocumentRetrievalInvariantError())


async def test_embedding_cancellation_propagates_without_search() -> None:
    class _CancelledEmbedding:
        async def embed(self, texts, metadata):  # type: ignore[no-untyped-def]
            raise asyncio.CancelledError

    repository = _Repository()
    with pytest.raises(asyncio.CancelledError):
        await DocumentRetrievalService(repository, _CancelledEmbedding()).retrieve(
            tenant=_tenant(),
            query="valid query",
            allowed_document_ids=(uuid4(),),
        )
    assert repository.calls == []


async def test_repository_failure_after_one_embedding_propagates() -> None:
    class _FailingRepository(_Repository):
        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            raise RuntimeError("synthetic database failure")

    embedding = _Embedding()
    repository = _FailingRepository()
    with pytest.raises(RuntimeError, match="synthetic database failure"):
        await DocumentRetrievalService(repository, embedding).retrieve(
            tenant=_tenant(),
            query="valid query",
            allowed_document_ids=(uuid4(),),
        )
    assert len(embedding.calls) == 1
    assert len(repository.calls) == 1


@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "empty",
        "zero",
        "utf8",
        "embedding",
        "over_return",
        "malformed",
        "failure",
        "cancel",
    ],
)
async def test_retrieval_trace_boundary_counts_failures_and_privacy(case) -> None:
    from app.domain.tracing import current_trace_scope
    from tests.tracing import CollectingTraceSink, collecting_node

    first, second = uuid4(), uuid4()
    hits = (_hit(document_id=first, text="PRIVATE-CHUNK-CANARY"), _hit(document_id=second))
    embedding = _Embedding()
    repository = _Repository(hits)
    allowed = (first, second, first)
    if case == "empty":
        allowed = ()
    elif case == "zero":
        repository.hits = ()
    elif case == "utf8":
        repository.hits = tuple(_hit(text="履历🧭" * 400) for _ in range(5))
    elif case == "embedding":
        embedding.result = type("MalformedEmbeddingResult", (), {"vectors": ((1.0,),)})()
    elif case == "over_return":
        repository.hits = hits * 3
    elif case == "malformed":
        repository.hits = (_hit(distance=inf),)
    elif case == "failure":

        class FailingRepository(_Repository):
            async def search(self, **kwargs):
                raise RuntimeError("REPOSITORY-EXCEPTION-CANARY")

        repository = FailingRepository()
    elif case == "cancel":

        class CancelledEmbedding(_Embedding):
            async def embed(self, texts, metadata):
                raise asyncio.CancelledError

        embedding = CancelledEmbedding()

    sink = CollectingTraceSink()
    with collecting_node(sink) as scope:
        if case in {"embedding", "over_return", "malformed", "failure", "cancel"}:
            error = (
                asyncio.CancelledError
                if case == "cancel"
                else RuntimeError
                if case == "failure"
                else DocumentRetrievalInvariantError
            )
            with pytest.raises(error):
                await DocumentRetrievalService(repository, embedding).retrieve(
                    tenant=_tenant(),
                    query="PRIVATE-QUERY-CANARY",
                    allowed_document_ids=allowed,
                )
        else:
            result = await DocumentRetrievalService(repository, embedding).retrieve(
                tenant=_tenant(),
                query="PRIVATE-QUERY-CANARY",
                allowed_document_ids=allowed,
            )
        assert current_trace_scope() == scope
    [(context, start)] = [v for v in sink.starts.values() if v[1].span_kind == "retrieval"]
    assert start.parent == scope.parent
    assert dict(start.metadata) == {
        "embedding_profile": EMBEDDING_PROFILE,
        "top_k": RETRIEVAL_TOP_K,
        "allowed_document_count": 0 if case == "empty" else 2,
    }
    finish = sink.finishes[context.context_id]
    if case in {"embedding", "over_return", "malformed", "failure", "cancel"}:
        assert finish.status == ("cancelled" if case == "cancel" else "failed")
        assert finish.error_category == (
            "retrieval_cancelled"
            if case == "cancel"
            else "retrieval_failed"
            if case == "failure"
            else "retrieval_invariant_failed"
        )
    else:
        assert finish.status == ("no_result" if case in {"empty", "zero"} else "succeeded")
        assert dict(finish.metadata) == {
            "returned_chunk_count": len(result),
            "context_bytes": sum(len(hit.text.encode("utf-8")) for hit in result),
        }
        assert finish.metadata["context_bytes"] <= RETRIEVAL_CONTEXT_BYTE_BUDGET
    if case in {"empty", "embedding", "cancel"}:
        assert repository.calls == []
    if case == "empty":
        assert embedding.calls == []
    for canary in ("PRIVATE-QUERY-CANARY", "PRIVATE-CHUNK-CANARY", "REPOSITORY-EXCEPTION-CANARY"):
        assert canary not in sink.safe_json() + repr(sink.starts) + repr(sink.finishes)
