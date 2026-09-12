from __future__ import annotations

from hashlib import sha256

from app.retrieval.chunking import (
    normalize_and_chunk_batch,
)
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.evals.retrieval_benchmark import load_retrieval_dataset

EXPECTED_CHUNK_VECTORS = {
    "backend_profile": (
        "e5a604c250fff30c25018ba7380627b0424fe887cebe2d0e75cc549ea0130b52",
        "c27a0717e623bc74b68ef335b1f1a51daf210c92b910dd05872ead7a930c558c",
    ),
    "rag_profile": (
        "50fe7313e06233882aae1d72dafc222446149ea5ea5999ba1138e61fc1e22a3d",
        "c6490a9bb5b8eaedc6ac41aae205131d158bc76b376402b37e1256ea6b35833e",
    ),
    "agent_reliability": (
        "6bd34621fc409877ba294153a38cd93999666bedf25e2f229d9430cb9af2a7f2",
        "cb319faa1ffe11c70f6a076bdf20ad70168c77d237c715d69c68e0b227d5e688",
    ),
    "same_workspace_decoy": ("ba2f45a298132ac8ef4324a5c57e6488679df7dcd1bd9ecdbfafde51786d9d15",),
    "foreign_workspace_canary": (
        "552719440a0600a087921777008d30f10bbdbe2170a8430f14b6a8b5c97b840a",
    ),
}


def test_retrieval_v1_document_chunk_text_vectors_are_stable() -> None:
    dataset = load_retrieval_dataset()

    actual: dict[str, tuple[str, ...]] = {}
    for document in dataset.documents:
        prepared = normalize_and_chunk_batch(
            ValidatedIngestionBatch(
                sources=(
                    ValidatedIngestionSource(
                        source_name=document.source_name,
                        source_type=document.source_type,
                        title=document.title,
                        raw_text=document.raw_text,
                        character_count=len(document.raw_text),
                    ),
                )
            )
        )
        chunks = prepared.sources[0].chunks
        actual[document.alias] = tuple(
            sha256(chunk.text.encode("utf-8")).hexdigest() for chunk in chunks
        )

    assert actual == EXPECTED_CHUNK_VECTORS
