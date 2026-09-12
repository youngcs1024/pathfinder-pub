"""Gate 11.3 thin runner over production ingestion, pgvector, and retrieval."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, text

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.models import Document, DocumentChunk, LLMInvocation
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import AsyncSessionFactory
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMFactory, LLMProviderError
from app.llm.invocations import LLMInvocationContext, LLMInvocationProvider
from app.retrieval.chunking import CHUNKING_VERSION, NORMALIZATION_VERSION
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentIngestionInvariantError,
    DocumentIngestionService,
    DocumentRepositoryPort,
    DocumentRetrievalInvariantError,
    DocumentRetrievalRepositoryPort,
    DocumentRetrievalService,
)
from tests.evals.live_contracts import live_latency_summary
from tests.evals.live_retrieval_contracts import (
    PRIMARY_ALIASES,
    LiveRetrievalAggregateV1,
    LiveRetrievalCaseReportV1,
    LiveRetrievalChunkRefV1,
    LiveRetrievalDatasetV1,
    LiveRetrievalPolicyV1,
    LiveRetrievalReportV1,
    LiveRetrievalReportV2,
    LiveRetrievalStopCategory,
    live_retrieval_passes,
    load_live_retrieval_dataset,
    load_live_retrieval_policy,
    prepared_live_documents,
)

DISPOSABLE_DATABASE_NAME = re.compile(r"^pathfinder_test_[0-9a-f]{32}$")


class LiveRetrievalConfigurationError(Exception):
    """A sanitized benchmark setup or production-chain invariant failure."""


def _embedding(factory: LLMFactory, tenant: TenantContext):
    return factory.create_embedding_model(
        LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
    )


def _aggregate(cases: tuple[LiveRetrievalCaseReportV1, ...]) -> LiveRetrievalAggregateV1:
    count = len(cases)
    return LiveRetrievalAggregateV1(
        macro_recall_at_1=sum(case.recall_at_1 for case in cases) / count if count else 0,
        macro_recall_at_3=sum(case.recall_at_3 for case in cases) / count if count else 0,
        macro_recall_at_5=sum(case.recall_at_5 for case in cases) / count if count else 0,
        mean_reciprocal_rank=sum(case.reciprocal_rank for case in cases) / count if count else 0,
        total_irrelevant_context_count=sum(case.irrelevant_context_count for case in cases),
        workspace_leakage_count=sum(case.workspace_leakage_count for case in cases),
        allowlist_leakage_count=sum(case.allowlist_leakage_count for case in cases),
    )


async def _invocation_diagnostics(
    sessions: AsyncSessionFactory,
) -> tuple[int, int, Decimal, int, tuple[int, ...]]:
    async with sessions() as session:
        rows = (
            await session.execute(
                select(
                    LLMInvocation.token_usage,
                    LLMInvocation.estimated_cost,
                    LLMInvocation.latency_ms,
                )
                .where(LLMInvocation.invocation_kind == "embedding")
                .order_by(LLMInvocation.created_at, LLMInvocation.id)
            )
        ).all()
    return (
        len(rows),
        sum(
            int(row.token_usage.get("input_tokens", 0))
            for row in rows
            if isinstance(row.token_usage, dict)
        ),
        sum(
            (row.estimated_cost for row in rows if row.estimated_cost is not None),
            Decimal(0),
        ),
        sum(row.estimated_cost is None for row in rows),
        tuple(int(row.latency_ms) for row in rows if row.latency_ms is not None),
    )


async def _report(
    sessions: AsyncSessionFactory,
    *,
    policy: LiveRetrievalPolicyV1,
    cases: tuple[LiveRetrievalCaseReportV1, ...],
    candidate_pool_size: int,
    complete: bool,
    error_category: LiveRetrievalStopCategory | None,
    provider: LLMInvocationProvider,
    exploratory: bool,
) -> LiveRetrievalReportV1 | LiveRetrievalReportV2:
    aggregate = _aggregate(cases)
    attempts, input_tokens, known_cost, unknown_cost, latencies = await _invocation_diagnostics(
        sessions
    )
    common = dict(
        provider=provider,
        semantic_quality_claim=provider == "qwen" and complete,
        dataset_version=policy.dataset_version,
        dataset_digest=policy.dataset_digest,
        case_set_digest=policy.case_set_digest,
        normalization_version=policy.normalization_version,
        chunking_version=policy.chunking_version,
        embedding_profile=policy.embedding_profile,
        embedding_dimension=policy.embedding_dimension,
        retrieval_top_k=policy.retrieval_top_k,
        candidate_pool_size=candidate_pool_size,
        query_count=policy.query_count,
        complete=complete,
        cases=cases,
        aggregate=aggregate,
        provider_attempt_count=attempts,
        input_tokens=input_tokens,
        known_cost_cny=known_cost,
        unknown_cost_attempt_count=unknown_cost,
        observed_embedding_attempt_latency_ms=latencies,
        passed=provider == "qwen" and live_retrieval_passes(aggregate, policy, complete=complete),
        error_category=error_category,
    )
    if exploratory:
        return LiveRetrievalReportV1(
            evidence_scope="exploratory_live_embedding_retrieval_benchmark",
            **common,
        )
    return LiveRetrievalReportV2(
        exploratory=False,
        priced_attempt_count=attempts - unknown_cost,
        embedding_provider_latency=live_latency_summary(tuple(float(value) for value in latencies)),
        **common,
    )


def normalized_live_retrieval_report(
    report: LiveRetrievalReportV1 | LiveRetrievalReportV2,
) -> bytes:
    return (
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


async def run_live_retrieval_benchmark(
    sessions: AsyncSessionFactory,
    *,
    factory: LLMFactory,
    dataset: LiveRetrievalDatasetV1 | None = None,
    policy: LiveRetrievalPolicyV1 | None = None,
    repository: DocumentRepositoryPort | DocumentRetrievalRepositoryPort | None = None,
    subject_namespace: str = "gate-11-3",
    confirm_disposable_database: bool = False,
    accepted: bool = False,
    git_probe: Callable[[], str] | None = None,
) -> LiveRetrievalReportV1 | LiveRetrievalReportV2:
    """Run production RAG with injected Factory; never discovers provider secrets."""

    dataset = dataset or load_live_retrieval_dataset()
    policy = policy or load_live_retrieval_policy(dataset=dataset)
    cases: list[LiveRetrievalCaseReportV1] = []
    candidate_pool_size = 0
    if accepted:
        try:
            if git_probe is None:
                from tests.evals.live_baseline import probe_clean_git_head

                probe_clean_git_head()
            else:
                git_probe()
        except Exception:
            return await _report(
                sessions,
                policy=policy,
                cases=(),
                candidate_pool_size=0,
                complete=False,
                error_category="configuration_invariant_failure",
                provider=factory.provider,
                exploratory=False,
            )
    if not confirm_disposable_database:
        return await _report(
            sessions,
            policy=policy,
            cases=(),
            candidate_pool_size=0,
            complete=False,
            error_category="configuration_invariant_failure",
            provider=factory.provider,
            exploratory=not accepted,
        )
    async with sessions() as session:
        database_name = await session.scalar(text("SELECT current_database()"))
    if (
        not isinstance(database_name, str)
        or DISPOSABLE_DATABASE_NAME.fullmatch(database_name) is None
    ):
        return await _report(
            sessions,
            policy=policy,
            cases=(),
            candidate_pool_size=0,
            complete=False,
            error_category="configuration_invariant_failure",
            provider=factory.provider,
            exploratory=not accepted,
        )

    try:
        provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
        primary_actor = await provisioning.provision_personal_workspace(
            f"{subject_namespace}-primary"
        )
        foreign_actor = await provisioning.provision_personal_workspace(
            f"{subject_namespace}-foreign"
        )
        primary = TenantContext(
            primary_actor.workspace_id, primary_actor.user_id, primary_actor.role
        )
        foreign = TenantContext(
            foreign_actor.workspace_id, foreign_actor.user_id, foreign_actor.role
        )
        repository = repository or SqlAlchemyDocumentRepository(sessions)
        document_ids: dict[str, UUID] = {}
        document_workspaces: dict[UUID, UUID] = {}

        for document in dataset.documents:
            tenant = foreign if document.role == "foreign_workspace_canary" else primary
            service = DocumentIngestionService(repository, _embedding(factory, tenant))
            (document_id,) = await service.ingest(
                tenant=tenant,
                batch=prepared_live_documents((document,)),
            )
            document_ids[document.alias] = document_id
            document_workspaces[document_id] = tenant.workspace_id

        id_to_ref: dict[UUID, LiveRetrievalChunkRefV1] = {}
        ref_to_id: dict[LiveRetrievalChunkRefV1, UUID] = {}
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
            (
                document_id,
                normalization,
                chunking,
                document_profile,
                chunk_id,
                ordinal,
                chunk_profile,
            ) = row
            if (
                normalization != NORMALIZATION_VERSION
                or chunking != CHUNKING_VERSION
                or document_profile != EMBEDDING_PROFILE
                or chunk_profile != EMBEDDING_PROFILE
            ):
                raise LiveRetrievalConfigurationError
            reference = LiveRetrievalChunkRefV1(
                document_alias=alias_by_document_id[document_id], ordinal=ordinal
            )
            id_to_ref[chunk_id] = reference
            ref_to_id[reference] = chunk_id

        candidate_pool_size = sum(ref.document_alias in PRIMARY_ALIASES for ref in ref_to_id)
        if candidate_pool_size != policy.expected_primary_candidate_pool_size:
            raise LiveRetrievalConfigurationError
        for query in dataset.queries:
            if any(reference not in ref_to_id for reference in query.relevant_chunks):
                raise LiveRetrievalConfigurationError

        foreign_document_id = document_ids["foreign_workspace_canary"]
        retrieval = DocumentRetrievalService(repository, _embedding(factory, primary))
        for query in dataset.queries:
            allowed_ids = tuple(document_ids[alias] for alias in query.allowed_document_aliases)
            caller_allowlist = allowed_ids + (
                (foreign_document_id,) if query.include_foreign_canary_in_allowlist else ()
            )
            hits = await retrieval.retrieve(
                tenant=primary,
                query=query.query,
                allowed_document_ids=caller_allowlist,
            )
            try:
                ranked_refs = tuple(id_to_ref[hit.chunk_id] for hit in hits)
            except KeyError:
                raise LiveRetrievalConfigurationError from None
            ranked_ids = tuple(hit.chunk_id for hit in hits)
            relevant_ids = frozenset(ref_to_id[ref] for ref in query.relevant_chunks)
            workspace_leakage = sum(
                document_workspaces.get(hit.document_id) != primary.workspace_id for hit in hits
            )
            allowlist_leakage = sum(hit.document_id not in allowed_ids for hit in hits)
            relevant_count = len(relevant_ids)
            cases.append(
                LiveRetrievalCaseReportV1(
                    query_id=query.query_id,
                    relevant_chunks=query.relevant_chunks,
                    ranked_retrieved_chunks=ranked_refs,
                    recall_at_1=len(set(ranked_ids[:1]) & relevant_ids) / relevant_count,
                    recall_at_3=len(set(ranked_ids[:3]) & relevant_ids) / relevant_count,
                    recall_at_5=len(set(ranked_ids[:5]) & relevant_ids) / relevant_count,
                    reciprocal_rank=next(
                        (
                            1 / rank
                            for rank, chunk_id in enumerate(ranked_ids, start=1)
                            if chunk_id in relevant_ids
                        ),
                        0.0,
                    ),
                    irrelevant_context_count=sum(
                        chunk_id not in relevant_ids for chunk_id in ranked_ids
                    ),
                    workspace_leakage_count=workspace_leakage,
                    allowlist_leakage_count=allowlist_leakage,
                    returned_context_bytes=sum(len(hit.text.encode()) for hit in hits),
                )
            )
    except asyncio.CancelledError:
        error_category: LiveRetrievalStopCategory = "cancelled"
    except LLMProviderError:
        error_category = "provider_failure"
    except (DocumentIngestionInvariantError, DocumentRetrievalInvariantError):
        error_category = "embedding_or_retrieval_invariant_failure"
    except LiveRetrievalConfigurationError:
        error_category = "configuration_invariant_failure"
    except Exception:
        error_category = "benchmark_execution_failure"
    else:
        return await _report(
            sessions,
            policy=policy,
            cases=tuple(cases),
            candidate_pool_size=candidate_pool_size,
            complete=True,
            error_category=None,
            provider=factory.provider,
            exploratory=not accepted,
        )
    return await _report(
        sessions,
        policy=policy,
        cases=tuple(cases),
        candidate_pool_size=candidate_pool_size,
        complete=False,
        error_category=error_category,
        provider=factory.provider,
        exploratory=not accepted,
    )
