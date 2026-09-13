"""Seeded capacity fixtures: valid immutable documents, no semantic quality claim."""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import insert

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.models import Conversation, Message, Run, RunEvent, RunJob
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.domain.provisioning import ProvisioningService
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantContext
from app.llm.ports import LOCKED_EMBEDDING_MODEL, EmbeddingResult
from app.retrieval.chunking import normalize_and_chunk_batch
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentIdentity,
    ImmutableDocumentChunk,
    ImmutableDocumentRepresentation,
)
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.performance.retrieval_contracts import Point

OTHER_PROFILE = "capacity-other-1536-v1"
QUERIES = tuple(f"E58_SYNTHETIC_QUERY_CANARY query {i}" for i in range(5))


def vector(identity: str) -> tuple[float, ...]:
    seed = hashlib.sha256(f"retrieval-scale-seed58-v1/{identity}".encode()).digest()
    source = random.Random(int.from_bytes(seed, "big"))
    # Explicit float32 conversion matches vector storage and fixes the dataset digest basis.
    return tuple(
        struct.unpack("!f", struct.pack("!f", source.uniform(-1, 1)))[0] for _ in range(1536)
    )


class QueryEmbedding:
    provider = "fake"
    model = LOCKED_EMBEDDING_MODEL

    def __init__(self, limit):
        self.calls = 0
        self.limit = limit

    async def embed(self, texts, metadata, *, attempt):
        if (
            len(texts) != 1
            or texts[0] not in QUERIES
            or attempt is None
            or self.calls >= self.limit
        ):
            raise ValueError("invalid_embedding_call")
        self.calls += 1
        return EmbeddingResult(vectors=(vector(f"query/{QUERIES.index(texts[0])}"),))


def allocations(point: Point):
    point = Point.model_validate_json(point.model_dump_json())
    unit = 5 if point.kind == "instant" else 10
    fixed = [point.candidates, unit, unit, unit]
    quotient, remainder = divmod(point.chunks - sum(fixed), point.documents - 4)
    counts = fixed + [quotient + (i < remainder) for i in range(point.documents - 4)]
    workspaces = [0, 0, 0, 1] + [1 + i % (point.workspaces - 1) for i in range(point.documents - 4)]
    if min(counts) <= 0 or sum(counts) != point.chunks:
        raise ValueError("invalid_allocation")
    return tuple(zip(workspaces, counts, strict=True))


def prepared_document(ordinal, chunks):
    body = "\n".join(
        f"# section-{i}\nE58_SYNTHETIC_BODY_CANARY document-{ordinal} chunk-{i}\n"
        for i in range(chunks)
    )
    (source,) = normalize_and_chunk_batch(
        ValidatedIngestionBatch(
            sources=(
                ValidatedIngestionSource(
                    source_name=f"capacity-{ordinal}.md",
                    source_type="markdown",
                    title=f"capacity-{ordinal}",
                    raw_text=body,
                    character_count=len(body),
                ),
            )
        )
    ).sources
    if len(source.chunks) != chunks:
        raise ValueError("chunk_count_mismatch")
    return source


@dataclass(repr=False)
class Dataset:
    tenants: tuple[TenantContext, ...]
    documents: list[UUID] = field(default_factory=list)
    history: list[UUID] = field(default_factory=list)
    dataset_digest: str = ""


async def create_dataset(sessions, point, check, on_created):
    tenants = []
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
    for i in range(point.workspaces):
        await check()
        actor = await provisioning.provision_personal_workspace(f"capacity-e58-{i}")
        tenants.append(TenantContext(actor.workspace_id, actor.user_id, actor.role))
    dataset = Dataset(tuple(tenants))
    on_created(dataset)
    repository = SqlAlchemyDocumentRepository(sessions)
    hasher = hashlib.sha256()
    for i, (workspace, count) in enumerate(allocations(point)):
        await check()
        source = prepared_document(i, count)
        tenant = tenants[workspace]
        profile = OTHER_PROFILE if i == 2 else EMBEDDING_PROFILE
        chunks = []
        hasher.update(f"{i}/{workspace}/{profile}/{source.content_hash}".encode())
        for chunk in source.chunks:
            values = vector(f"document/{i}/chunk/{chunk.ordinal}")
            hasher.update(struct.pack("!1536f", *values))
            chunks.append(
                ImmutableDocumentChunk(
                    ordinal=chunk.ordinal,
                    section=chunk.section,
                    text=chunk.text,
                    content_hash=chunk.content_hash,
                    token_count=chunk.token_count,
                    embedding=values,
                )
            )
        document = await repository.persist(
            ImmutableDocumentRepresentation(
                identity=DocumentIdentity(
                    workspace_id=tenant.workspace_id,
                    content_hash=source.content_hash,
                    normalization_version=source.normalization_version,
                    chunking_version=source.chunking_version,
                    embedding_model=profile,
                ),
                created_by_user_id=tenant.actor_user_id,
                title=source.title,
                source_type=source.source_type,
                source_name=source.source_name,
                content=source.content,
                chunks=tuple(chunks),
            )
        )
        dataset.documents.append(document)
        dataset.dataset_digest = hasher.hexdigest()
    # A fixed set of cancelled historical storage fixtures. They are not graph execution
    # evidence; no fake completed action/result or tool/model invocation is inserted.
    events_per_run = point.events // point.history_runs
    for i in range(point.history_runs):
        await check()
        tenant = tenants[i % len(tenants)]
        conversation, message, run = uuid4(), uuid4(), uuid4()
        now = datetime.now(UTC)
        async with sessions() as session, session.begin():
            await session.execute(
                insert(Conversation).values(
                    id=conversation,
                    workspace_id=tenant.workspace_id,
                    created_by_user_id=tenant.actor_user_id,
                    title="capacity history",
                )
            )
            await session.execute(
                insert(Message).values(
                    id=message,
                    workspace_id=tenant.workspace_id,
                    conversation_id=conversation,
                    actor_user_id=tenant.actor_user_id,
                    role="user",
                    content="capacity fixture",
                )
            )
            await session.execute(
                insert(Run).values(
                    id=run,
                    workspace_id=tenant.workspace_id,
                    created_by_user_id=tenant.actor_user_id,
                    conversation_id=conversation,
                    request_message_id=message,
                    mode="research",
                    input_json={"query": "capacity"},
                    limits_json={},
                    status="cancelled",
                    graph_version=CURRENT_GRAPH_VERSION,
                    next_event_seq=events_per_run + 1,
                    started_at=now,
                    finished_at=now,
                    cancel_requested_at=now,
                )
            )
            await session.execute(
                insert(RunJob).values(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    originating_actor_user_id=tenant.actor_user_id,
                    run_id=run,
                    status="done",
                )
            )
            await session.execute(
                insert(RunEvent),
                [
                    dict(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        run_id=run,
                        actor_user_id=tenant.actor_user_id,
                        seq=seq,
                        version=1,
                        type=(
                            "run.created"
                            if seq == 1
                            else "run.cancelled"
                            if seq == events_per_run
                            else "source.discovered"
                        ),
                        payload=(
                            {"source_id": f"capacity-{seq}"}
                            if 1 < seq < events_per_run
                            else {"status": "queued" if seq == 1 else "cancelled"}
                        ),
                    )
                    for seq in range(1, events_per_run + 1)
                ],
            )
        dataset.history.append(run)
    hasher.update(f"history/{point.history_runs}/{point.events}".encode())
    dataset.dataset_digest = hasher.hexdigest()
    return dataset
