from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select, text

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import Document, DocumentChunk
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import AsyncSessionFactory
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from app.retrieval.chunking import (
    CHUNKING_VERSION,
    NORMALIZATION_VERSION,
    normalize_and_chunk_batch,
)
from app.retrieval.documents import (
    EMBEDDING_DIMENSION,
    EMBEDDING_PROFILE,
    RETRIEVAL_TOP_K,
    DocumentIngestionService,
    DocumentRetrievalService,
)
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.evals.retrieval_contracts import (
    RetrievalBenchmarkAggregateV1,
    RetrievalBenchmarkCaseReportV1,
    RetrievalBenchmarkReportV1,
    RetrievalChunkRefV1,
    RetrievalDatasetV1,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RETRIEVAL_DATASET_PATH = PROJECT_ROOT / "evals" / "datasets" / "retrieval_v1.json"
DISPOSABLE_DATABASE_NAME = re.compile(r"^pathfinder_test_[0-9a-f]{32}$")


class RetrievalBenchmarkConfigurationError(Exception):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return f"sha256:{sha256(_canonical(value)).hexdigest()}"


def load_retrieval_dataset(
    path: Path = DEFAULT_RETRIEVAL_DATASET_PATH,
) -> RetrievalDatasetV1:
    try:
        return RetrievalDatasetV1.model_validate_json(path.read_bytes(), strict=True)
    except (OSError, ValidationError):
        raise RetrievalBenchmarkConfigurationError(
            "retrieval benchmark dataset is invalid"
        ) from None


def retrieval_dataset_digests(dataset: RetrievalDatasetV1) -> tuple[str, str]:
    dataset_payload = dataset.model_dump(mode="json")
    case_ids = sorted(case.case_id for case in dataset.cases)
    return _digest(dataset_payload), _digest(case_ids)


def recall_at(ranked: tuple[UUID, ...], relevant: frozenset[UUID], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: tuple[UUID, ...], relevant: frozenset[UUID]) -> float:
    return next(
        (1 / rank for rank, chunk_id in enumerate(ranked, start=1) if chunk_id in relevant), 0.0
    )


def normalized_retrieval_report(report: RetrievalBenchmarkReportV1) -> bytes:
    return _canonical(report.model_dump(mode="json")) + b"\n"


def _embedding(factory: LLMFactory, tenant: TenantContext):
    return factory.create_embedding_model(
        LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
    )


async def run_retrieval_benchmark(
    sessions: AsyncSessionFactory,
    *,
    dataset: RetrievalDatasetV1 | None = None,
    subject_namespace: str = "gate-9-4",
    confirm_disposable_database: bool = False,
) -> RetrievalBenchmarkReportV1:
    if not confirm_disposable_database:
        raise RetrievalBenchmarkConfigurationError(
            "retrieval benchmark requires explicit disposable database confirmation"
        )
    async with sessions() as session:
        database_name = await session.scalar(text("SELECT current_database()"))
    if (
        not isinstance(database_name, str)
        or DISPOSABLE_DATABASE_NAME.fullmatch(database_name) is None
    ):
        raise RetrievalBenchmarkConfigurationError(
            "retrieval benchmark requires an isolated pathfinder test database"
        )
    dataset = dataset or load_retrieval_dataset()
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
    primary_actor = await provisioning.provision_personal_workspace(f"{subject_namespace}-primary")
    foreign_actor = await provisioning.provision_personal_workspace(f"{subject_namespace}-foreign")
    primary = TenantContext(primary_actor.workspace_id, primary_actor.user_id, primary_actor.role)
    foreign = TenantContext(foreign_actor.workspace_id, foreign_actor.user_id, foreign_actor.role)
    recorder = SqlAlchemyInvocationRecorder(sessions)
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )
    repository = SqlAlchemyDocumentRepository(sessions)

    document_ids: dict[str, UUID] = {}
    document_workspaces: dict[UUID, str] = {}
    for document in dataset.documents:
        tenant = primary if document.workspace == "primary" else foreign
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
        service = DocumentIngestionService(repository, _embedding(factory, tenant))
        (document_id,) = await service.ingest(tenant=tenant, batch=prepared)
        document_ids[document.alias] = document_id
        document_workspaces[document_id] = document.workspace

    id_to_ref: dict[UUID, RetrievalChunkRefV1] = {}
    ref_to_id: dict[RetrievalChunkRefV1, UUID] = {}
    async with sessions() as session:
        rows = (
            await session.execute(
                select(
                    Document.id,
                    Document.normalization_version,
                    Document.chunking_version,
                    Document.embedding_model,
                    DocumentChunk.id,
                    DocumentChunk.ordinal,
                    DocumentChunk.embedding_model,
                )
                .join(
                    DocumentChunk,
                    (DocumentChunk.workspace_id == Document.workspace_id)
                    & (DocumentChunk.document_id == Document.id),
                )
                .where(Document.id.in_(tuple(document_ids.values())))
            )
        ).all()
    alias_by_document_id = {value: key for key, value in document_ids.items()}
    for row in rows:
        document_id, normalization, chunking, document_profile, chunk_id, ordinal, chunk_profile = (
            row
        )
        if (
            normalization != NORMALIZATION_VERSION
            or chunking != CHUNKING_VERSION
            or document_profile != EMBEDDING_PROFILE
            or chunk_profile != EMBEDDING_PROFILE
        ):
            raise RetrievalBenchmarkConfigurationError(
                "persisted retrieval profile is incompatible"
            )
        reference = RetrievalChunkRefV1(
            document_alias=alias_by_document_id[document_id], ordinal=ordinal
        )
        id_to_ref[chunk_id] = reference
        ref_to_id[reference] = chunk_id

    for case in dataset.cases:
        for reference in case.relevant_chunks:
            if reference not in ref_to_id:
                raise RetrievalBenchmarkConfigurationError("relevant chunk ordinal is unresolved")

    foreign_alias = next(
        document.alias for document in dataset.documents if document.workspace == "foreign"
    )
    case_reports: list[RetrievalBenchmarkCaseReportV1] = []
    retrieval = DocumentRetrievalService(repository, _embedding(factory, primary))
    for case in dataset.cases:
        allowed_ids = tuple(document_ids[alias] for alias in case.allowed_document_aliases)
        caller_allowlist = allowed_ids
        if case.include_foreign_canary_in_allowlist:
            caller_allowlist += (document_ids[foreign_alias],)
        hits = await retrieval.retrieve(
            tenant=primary,
            query=case.query,
            allowed_document_ids=caller_allowlist,
        )
        ranked_ids = tuple(hit.chunk_id for hit in hits)
        try:
            ranked_refs = tuple(id_to_ref[chunk_id] for chunk_id in ranked_ids)
        except KeyError:
            raise RetrievalBenchmarkConfigurationError(
                "retrieval returned an unknown chunk"
            ) from None
        relevant_ids = frozenset(ref_to_id[reference] for reference in case.relevant_chunks)
        workspace_leakage = sum(
            document_workspaces.get(hit.document_id) != "primary" for hit in hits
        )
        allowlist_leakage = sum(hit.document_id not in allowed_ids for hit in hits)
        irrelevant = sum(chunk_id not in relevant_ids for chunk_id in ranked_ids)
        hard_passed = (
            workspace_leakage == 0 and allowlist_leakage == 0 and len(hits) <= RETRIEVAL_TOP_K
        )
        case_reports.append(
            RetrievalBenchmarkCaseReportV1(
                case_id=case.case_id,
                relevant_chunks=case.relevant_chunks,
                ranked_retrieved_chunks=ranked_refs,
                recall_at_1=recall_at(ranked_ids, relevant_ids, 1),
                recall_at_3=recall_at(ranked_ids, relevant_ids, 3),
                recall_at_5=recall_at(ranked_ids, relevant_ids, 5),
                reciprocal_rank=reciprocal_rank(ranked_ids, relevant_ids),
                irrelevant_context_count=irrelevant,
                workspace_leakage_count=workspace_leakage,
                allowlist_leakage_count=allowlist_leakage,
                passed=hard_passed,
                error_category=None if hard_passed else "retrieval_invariant_failed",
            )
        )

    count = len(case_reports)
    aggregate = RetrievalBenchmarkAggregateV1(
        macro_recall_at_1=sum(case.recall_at_1 for case in case_reports) / count,
        macro_recall_at_3=sum(case.recall_at_3 for case in case_reports) / count,
        macro_recall_at_5=sum(case.recall_at_5 for case in case_reports) / count,
        mean_reciprocal_rank=sum(case.reciprocal_rank for case in case_reports) / count,
        total_irrelevant_context_count=sum(case.irrelevant_context_count for case in case_reports),
        workspace_leakage_count=sum(case.workspace_leakage_count for case in case_reports),
        allowlist_leakage_count=sum(case.allowlist_leakage_count for case in case_reports),
    )
    dataset_digest, case_set_digest = retrieval_dataset_digests(dataset)
    return RetrievalBenchmarkReportV1(
        evidence_scope="deterministic_fake_embedding_regression",
        dataset_version=dataset.dataset_version,
        dataset_digest=dataset_digest,
        case_set_digest=case_set_digest,
        normalization_version=NORMALIZATION_VERSION,
        chunking_version=CHUNKING_VERSION,
        embedding_profile=EMBEDDING_PROFILE,
        embedding_dimension=EMBEDDING_DIMENSION,
        retrieval_top_k=RETRIEVAL_TOP_K,
        cases=tuple(case_reports),
        aggregate=aggregate,
        passed=all(case.passed for case in case_reports),
    )
