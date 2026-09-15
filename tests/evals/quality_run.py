"""E4 manual measurement entries over production services and Factory accounting.

Retrieval public files contain no source text or database IDs. Generation adds a
separate, protected review bundle. Neither entry discovers providers or accepts baselines.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path
from time import monotonic
from uuid import UUID

from langsmith import tracing_context
from sqlalchemy import select, text

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import DocumentChunk, User
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import AsyncSessionFactory
from app.domain.provisioning import ProvisioningService
from app.domain.runs import CURRENT_GRAPH_VERSION
from app.domain.tenancy import TenantContext
from app.domain.tracing import bind_trace_scope
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext, NoOpTraceSink
from app.llm.ports import LOCKED_EMBEDDING_MODEL
from app.retrieval.chunking import PreparedIngestionBatch, PreparedIngestionSource
from app.retrieval.documents import (
    EMBEDDING_PROFILE,
    DocumentIngestionInvariantError,
    DocumentIngestionService,
    DocumentRepositoryPort,
    DocumentRetrievalRepositoryPort,
    DocumentRetrievalService,
    build_document_expectation,
    build_document_identity,
)
from tests.evals.live_baseline import PROJECT_ROOT, probe_clean_git_head
from tests.evals.live_chat import CURRENT_LOGICAL_CALL, LiveAttemptRecorder
from tests.evals.live_retrieval import DISPOSABLE_DATABASE_NAME
from tests.evals.quality_contracts import (
    Failure,
    QualityCaseV1,
    QualityCostV1,
    QualityMappingV1,
    QualityObservationV1,
    QualityRatioV1,
    QualityRetrievalAdmissionV1,
    QualityRetrievalCaseV1,
    QualityRetrievalMetricsV1,
    QualityRetrievalPolicyV1,
    QualityRetrievalRefV1,
    QualityRetrievalReportV1,
    QualityRetrievalRepresentationV1,
    QualityRetrievalStartV1,
    QualityRetrievalUsageV1,
    QualityRunManifestV1,
    QualitySafetyCountsV1,
    QualityStageTimeV1,
)
from tests.evals.quality_dataset import (
    QualityDataset,
    QualityDatasetError,
    load_quality_dataset,
    prepare_quality_mapping_sources,
    quality_context_judgments,
    quality_digest,
    quality_identity_digest,
    validate_quality_mapping,
    validate_quality_run,
)

SUITE_VERSION = "quality-retrieval-v1"
# The required generic manifest field explicitly identifies the absence of chat prompts.
NO_CHAT_PROMPT_DIGEST = quality_digest(b"quality-retrieval-v1:no-chat-prompt")


async def run_quality_generation(sessions: AsyncSessionFactory | None = None, **kwargs):
    """Manual generation entry; keyword contract is documented by run_generation.

    A local import keeps existing retrieval imports and historical suite behavior stable.
    """
    from tests.evals.quality_generation import run_generation

    return await run_generation(sessions, **kwargs)


async def prepare_quality_generation(sessions: AsyncSessionFactory | None = None, **kwargs):
    """Prepare once; an authorized caller can interleave whole slots across arms."""
    from tests.evals.quality_generation import prepare_generation_session

    return await prepare_generation_session(sessions, **kwargs)


async def run_quality_generation_slot(session, index: int):
    from tests.evals.quality_generation import execute_generation_slot

    return await execute_generation_slot(session, index)


async def finish_quality_generation(session):
    from tests.evals.quality_generation import finish_generation_session

    return await finish_generation_session(session)


class QualityRetrievalError(Exception):
    """Sanitized configuration/publication boundary; never include underlying inputs."""


class _ScopeLeak(Exception):
    pass


class _QualityAttemptRecorder(LiveAttemptRecorder):
    """Reuse admission/persistence unchanged, retaining the scope of a failed write."""

    failed_logical_call: str | None = None

    async def prepare(self, attempt):
        try:
            await super().prepare(attempt)
        except BaseException:
            self.failed_logical_call = CURRENT_LOGICAL_CALL.get()
            raise

    async def finalize(self, attempt, outcome):
        try:
            await super().finalize(attempt, outcome)
        except BaseException:
            self.failed_logical_call = CURRENT_LOGICAL_CALL.get()
            raise


def ratio(numerator: int, denominator: int, *, unassessed: int = 0) -> QualityRatioV1:
    return QualityRatioV1(
        numerator=numerator,
        denominator=denominator,
        value=numerator / denominator if denominator else None,
        unassessed_count=unassessed,
        not_applicable_reason=None if denominator else "zero_denominator",
    )


def score_quality_retrieval(
    dataset: QualityDataset,
    mapping: QualityMappingV1,
    case: QualityCaseV1,
    refs: tuple[QualityRetrievalRefV1, ...],
) -> QualityRetrievalMetricsV1:
    """Score already verified exposed prefixes; gold never enters the retrieval call."""
    allowed = case.resume_alias if case.scope_expectation == "allowed" else None
    if any(r.source_alias != allowed for r in refs):
        raise _ScopeLeak
    ranked = tuple(r.chunk_ordinal for r in refs)
    if len(set(ranked)) != len(ranked):
        raise QualityRetrievalError("duplicate_retrieval_reference")
    judgments = {
        j.chunk_ordinal: j.relevance
        for j in quality_context_judgments(dataset, mapping)
        if j.case_id == case.case_id
    }
    relevant = {ordinal for ordinal, label in judgments.items() if label == "relevant"}
    chunks = {
        c.ordinal: c
        for source in mapping.sources
        if source.source_alias == allowed
        for c in source.chunks
    }
    covered: set[int] = set()
    for ref in refs:
        chunk = chunks.get(ref.chunk_ordinal)
        if chunk is None or ref.exposed_codepoints > chunk.codepoint_count:
            raise QualityRetrievalError("invalid_exposed_reference")
        for span in chunk.spans:
            end = min(span.chunk_end, ref.exposed_codepoints)
            if end > span.chunk_start:
                covered.update(range(span.source_start, span.source_start + end - span.chunk_start))
    units = {u.unit_id: u for u in dataset.units}
    required = [units[u] for u in case.required_unit_ids if units[u].source_kind == "resume"]
    satisfied = []
    for unit in required:
        candidates = [unit, *(units[a] for a in unit.alternative_unit_ids)]
        if any(
            candidate.source_kind == "resume"
            and candidate.source_alias == allowed
            and all(i in covered for i in range(candidate.start, candidate.end))
            for candidate in candidates
        ):
            satisfied.append(unit.unit_id)
    labels = [judgments.get(ordinal, "unjudged") for ordinal in ranked]
    rr = next((1 / rank for rank, ordinal in enumerate(ranked, 1) if ordinal in relevant), 0.0)
    return QualityRetrievalMetricsV1(
        recall_at_1=ratio(len(set(ranked[:1]) & relevant), len(relevant)),
        recall_at_3=ratio(len(set(ranked[:3]) & relevant), len(relevant)),
        recall_at_5=ratio(len(set(ranked[:5]) & relevant), len(relevant)),
        reciprocal_rank=rr if relevant else None,
        required_unit_coverage=ratio(len(satisfied), len(required)),
        covered_unit_ids=tuple(satisfied),
        web_required_units=sum(units[u].source_kind == "web" for u in case.required_unit_ids),
        relevant_contexts=labels.count("relevant"),
        irrelevant_contexts=labels.count("irrelevant"),
        unjudged_contexts=labels.count("unjudged"),
        returned_context_bytes=sum(r.exposed_bytes for r in refs),
        no_relevant_evidence=not relevant,
    )


def retrieval_configuration_digest(policy: QualityRetrievalPolicyV1, factory: LLMFactory) -> str:
    return quality_identity_digest(
        {
            "suite": SUITE_VERSION,
            "policy": policy.model_dump(mode="json"),
            "provider": factory.provider,
            "embedding_model": factory.embedding_model,
            "retry": asdict(factory.retry_policy),
            "pricing_version": factory.price_book.version,
            "graph_version": CURRENT_GRAPH_VERSION,
            "prompt_digest": NO_CHAT_PROMPT_DIGEST,
            "trace": "off",
            "web": "none",
        }
    )


def _head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def write_quality_retrieval_artifact(
    path: Path,
    payload: (
        QualityRetrievalStartV1
        | QualityRetrievalRepresentationV1
        | QualityRetrievalCaseV1
        | QualityRetrievalReportV1
    ),
) -> None:
    """Revalidate an allowlisted public contract; no overwrite or automatic cleanup."""
    try:
        if type(payload) not in (
            QualityRetrievalStartV1,
            QualityRetrievalRepresentationV1,
            QualityRetrievalCaseV1,
            QualityRetrievalReportV1,
        ):
            raise ValueError("unsupported artifact contract")
        checked = type(payload).model_validate_json(payload.model_dump_json())
        data = json.dumps(
            checked.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode()
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, ValueError, TypeError):
        raise QualityRetrievalError("artifact_publication_failed") from None


def _usage(recorder: _QualityAttemptRecorder, prefix: str = "") -> QualityRetrievalUsageV1:
    attempts = [key for key in recorder.attempts if recorder.logical_ids[key].startswith(prefix)]
    outcomes = [recorder.outcomes[key] for key in attempts if key in recorder.outcomes]
    priced = [o.estimated_cost for o in outcomes if o.estimated_cost is not None]
    known = sum(priced, Decimal(0))
    unknown = len(attempts) - len(priced)
    return QualityRetrievalUsageV1(
        provider_attempts=len(attempts),
        input_tokens=sum(o.token_usage.input_tokens for o in outcomes if o.token_usage),
        output_tokens=sum(o.token_usage.output_tokens for o in outcomes if o.token_usage),
        unknown_usage_attempts=len(attempts) - sum(o.token_usage is not None for o in outcomes),
        cost=QualityCostV1(
            scope="all",
            known_cost_cny=known,
            priced_attempts=len(priced),
            unknown_cost_attempts=unknown,
            total_cost_cny=None if unknown else known,
        ),
        accounting_complete=len(attempts) == len(outcomes)
        and not (
            (recorder.stop_reason or "").startswith("accounting_")
            and (recorder.failed_logical_call or "").startswith(prefix)
        ),
    )


def _stop_category(recorder: LiveAttemptRecorder) -> Failure | None:
    reason = recorder.stop_reason
    if reason is None:
        return None
    if reason in {"provider_cap_exceeded", "token_cap_exceeded", "budget_exhausted"}:
        return "budget"
    return "cancelled" if reason == "external_cancelled" else "integrity"


def _case_result(
    manifest: QualityRunManifestV1,
    index: int,
    recorder: _QualityAttemptRecorder,
    *,
    executed: bool = False,
    failure: Failure | None = None,
    seconds: float = 0.0,
    refs: tuple[QualityRetrievalRefV1, ...] = (),
    metrics: QualityRetrievalMetricsV1 | None = None,
) -> QualityRetrievalCaseV1:
    slot = manifest.execution_order[index]
    usage = _usage(recorder, f"query.{index}.")
    payload = {
        "refs": [r.model_dump(mode="json") for r in refs],
        "metrics": (metrics.model_dump(mode="json") if metrics is not None else None),
    }
    return QualityRetrievalCaseV1(
        observation=QualityObservationV1(
            experiment_id=manifest.experiment_id,
            case_id=slot.case_id,
            repeat_index=slot.repeat_index,
            measurement_scope=manifest.measurement_scope,
            status="not_run" if not executed else "failed" if failure else "succeeded",
            failure_type=failure,
            safety=QualitySafetyCountsV1(unauthorized_access=int(failure == "safety")),
            tool_calls=0,
            model_calls=int(usage.provider_attempts > 0),
            provider_attempts=usage.provider_attempts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=usage.cost,
            stage_times=(QualityStageTimeV1(stage="retrieval", seconds=seconds),)
            if executed
            else (),
            output_digest=quality_identity_digest(payload) if metrics is not None else None,
            assessment_required=False,
        ),
        refs=refs,
        metrics=metrics,
    )


async def _representation(
    sessions: AsyncSessionFactory,
    tenant: TenantContext,
    alias: str,
    source: PreparedIngestionSource,
    document_id: UUID,
) -> tuple[QualityRetrievalRepresentationV1, dict[UUID, int]]:
    # Re-read authoritative rows, independent of any injected fault repository.
    repository = SqlAlchemyDocumentRepository(sessions)
    if (
        await repository.find_complete(
            tenant=tenant,
            identity=build_document_identity(workspace_id=tenant.workspace_id, source=source),
            expected=build_document_expectation(source),
        )
        != document_id
    ):
        raise DocumentIngestionInvariantError
    async with sessions() as session:
        rows = (
            await session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.ordinal,
                    DocumentChunk.text,
                    DocumentChunk.content_hash,
                    DocumentChunk.embedding_model,
                )
                .where(
                    DocumentChunk.workspace_id == tenant.workspace_id,
                    DocumentChunk.document_id == document_id,
                )
                .order_by(DocumentChunk.ordinal)
            )
        ).all()
    if len(rows) != len(source.chunks) or any(
        row.ordinal != chunk.ordinal
        or row.text != chunk.text
        or row.content_hash != chunk.content_hash
        or row.embedding_model != EMBEDDING_PROFILE
        for row, chunk in zip(rows, source.chunks, strict=True)
    ):
        raise DocumentIngestionInvariantError
    return (
        QualityRetrievalRepresentationV1(
            source_alias=alias,
            normalized_text_digest=quality_digest(source.content.encode()),
            chunk_digests=tuple(quality_digest(row.text.encode()) for row in rows),
            chunk_count=len(rows),
        ),
        {row.id: row.ordinal for row in rows},
    )


async def run_quality_retrieval(
    sessions: AsyncSessionFactory,
    *,
    dataset_root: Path,
    dataset: QualityDataset,
    mapping: QualityMappingV1,
    mapping_rules_digest: str,
    manifest: QualityRunManifestV1,
    policy: QualityRetrievalPolicyV1,
    output_dir: Path,
    provider_mode: str = "fake",
    confirm_disposable_database: bool = False,
    confirm_live: bool = False,
    factory: LLMFactory | None = None,
    repository: DocumentRepositoryPort | DocumentRetrievalRepositoryPort | None = None,
    git_probe: Callable[[], str] | None = None,
) -> QualityRetrievalReportV1:
    """Manual/test entry. Injected Factory keeps adapters; its recorder is always DB-backed.

    Configuration failures precede mutation. Once reserved, the output directory retains
    manifest, completed slots, ingestion status and final/partial report, including cancellation.
    """
    try:
        manifest = QualityRunManifestV1.model_validate_json(manifest.model_dump_json())
        policy = QualityRetrievalPolicyV1.model_validate_json(policy.model_dump_json())
        mapping = QualityMappingV1.model_validate_json(mapping.model_dump_json())
        loaded = load_quality_dataset(dataset_root)
        if loaded != dataset or dataset.manifest.license_category != "synthetic":
            raise QualityDatasetError("dataset_not_verified_synthetic")
        validate_quality_run(dataset, manifest)
        prepared = prepare_quality_mapping_sources(dataset_root)
        mapping_report = validate_quality_mapping(
            dataset, mapping, prepared, rules_digest=mapping_rules_digest
        )
        if not mapping_report.mapping_complete or mapping_report.pending_review_chunks:
            raise QualityDatasetError("mapping_not_reviewed")
        delegate = SqlAlchemyInvocationRecorder(sessions)
        if factory is None:
            factory = LLMFactory(delegate, FakeChatModel(), FakeEmbeddingModel())
        if (
            provider_mode not in {"fake", "qwen"}
            or provider_mode != manifest.llm_mode
            or provider_mode != factory.provider
            or manifest.document_mode
            != ("fake_embedding_db" if provider_mode == "fake" else "real_embedding_db")
            or manifest.measurement_scope
            != ("contract" if provider_mode == "fake" else "retrieval")
            or manifest.web_mode != "none"
            or manifest.suite_version != SUITE_VERSION
            or manifest.model != LOCKED_EMBEDDING_MODEL
            or manifest.embedding_profile != policy.embedding_profile
            or manifest.prompt_digest != NO_CHAT_PROMPT_DIGEST
            or manifest.graph_version != CURRENT_GRAPH_VERSION
            or manifest.retrieval_policy_digest
            != quality_identity_digest(policy.model_dump(mode="json"))
            or manifest.configuration_digest != retrieval_configuration_digest(policy, factory)
            or policy.unknown_attempt_reserve_cny > manifest.cost_admission_budget_cny
            or not confirm_disposable_database
            or (provider_mode == "qwen" and not confirm_live)
        ):
            raise QualityRetrievalError("invalid_retrieval_configuration")
        probe = git_probe or (probe_clean_git_head if provider_mode == "qwen" else _head)
        if probe() != manifest.execution_source_sha:
            raise QualityRetrievalError("execution_source_mismatch")
        async with sessions() as session:
            name = await session.scalar(text("SELECT current_database()"))
        if not isinstance(name, str) or not DISPOSABLE_DATABASE_NAME.fullmatch(name):
            raise QualityRetrievalError("database_not_disposable")
        async with sessions() as session:
            reused = await session.scalar(
                select(User.id).where(User.auth_subject == f"quality-{manifest.experiment_id}")
            )
        if reused is not None:
            raise QualityRetrievalError("experiment_namespace_reused")
        admission = QualityRetrievalAdmissionV1.model_validate_json(
            json.dumps(
                {
                    **manifest.model_dump(mode="json"),
                    "unknown_attempt_reserve_cny": str(policy.unknown_attempt_reserve_cny),
                }
            )
        )
        recorder = _QualityAttemptRecorder(delegate, admission)
        factory = replace(factory, recorder=recorder, trace_sink=NoOpTraceSink())
        output_dir.mkdir(mode=0o700, exist_ok=False)
    except Exception:
        raise QualityRetrievalError("retrieval_preflight_failed") from None

    write_quality_retrieval_artifact(
        output_dir / "manifest.json",
        QualityRetrievalStartV1(
            manifest=manifest,
            policy=policy,
            mapping_digest=mapping_report.mapping_digest,
        ),
    )
    results: list[QualityRetrievalCaseV1] = []
    representations: list[QualityRetrievalRepresentationV1] = []
    representation_complete = False
    stop: Failure | None = None
    ingestion_started = monotonic()
    ingestion_seconds = 0.0
    cancellation: asyncio.CancelledError | None = None
    active_index: int | None = None
    query_started = 0.0
    ingestion_usage = _usage(recorder)
    try:
        with tracing_context(enabled=False), bind_trace_scope(None):
            provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
            actor = await provisioning.provision_personal_workspace(
                f"quality-{manifest.experiment_id}"
            )
            tenant = TenantContext(actor.workspace_id, actor.user_id, actor.role)
            actual_repository = repository or SqlAlchemyDocumentRepository(sessions)
            embedding = factory.create_embedding_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
            )
            documents: dict[str, UUID] = {}
            chunks: dict[UUID, tuple[str, int]] = {}
            for alias, source in prepared.items():
                token = CURRENT_LOGICAL_CALL.set(f"ingestion.{alias}.")
                try:
                    (document_id,) = await DocumentIngestionService(
                        actual_repository, embedding
                    ).ingest(tenant=tenant, batch=PreparedIngestionBatch((source,)))
                    representation, ids = await _representation(
                        sessions, tenant, alias, source, document_id
                    )
                    representations.append(representation)
                    write_quality_retrieval_artifact(
                        output_dir / f"representation-{len(representations):04d}.json",
                        representation,
                    )
                    documents[alias] = document_id
                    chunks.update({key: (alias, ordinal) for key, ordinal in ids.items()})
                finally:
                    CURRENT_LOGICAL_CALL.reset(token)
                if recorder.stop_reason:
                    break
            ingestion_seconds = monotonic() - ingestion_started
            ingestion_usage = _usage(recorder, "ingestion.")
            stop = _stop_category(recorder)
            representation_complete = len(representations) == len(prepared) and stop is None
            if representation_complete:
                cases = {case.case_id: case for case in dataset.cases}
                for index, slot in enumerate(manifest.execution_order):
                    active_index = index
                    query_started = monotonic()
                    token = CURRENT_LOGICAL_CALL.set(f"query.{index}.")
                    case = cases[slot.case_id]
                    allowlist = (
                        (documents[case.resume_alias],)
                        if case.resume_alias is not None and case.scope_expectation == "allowed"
                        else ()
                    )
                    failure: Failure | None = None
                    refs: tuple[QualityRetrievalRefV1, ...] = ()
                    metrics = None
                    try:
                        hits = await DocumentRetrievalService(
                            actual_repository, embedding
                        ).retrieve(tenant=tenant, query=case.query, allowed_document_ids=allowlist)
                        converted = []
                        for hit in hits:
                            if hit.document_id not in allowlist or hit.chunk_id not in chunks:
                                raise _ScopeLeak
                            alias, ordinal = chunks[hit.chunk_id]
                            if documents[alias] != hit.document_id or ordinal != hit.ordinal:
                                raise _ScopeLeak
                            original = prepared[alias].chunks[ordinal].text
                            if not original.startswith(hit.text):
                                raise QualityRetrievalError("exposed_text_mismatch")
                            converted.append(
                                QualityRetrievalRefV1(
                                    source_alias=alias,
                                    chunk_ordinal=ordinal,
                                    exposed_digest=quality_digest(hit.text.encode()),
                                    exposed_codepoints=len(hit.text),
                                    exposed_bytes=len(hit.text.encode()),
                                )
                            )
                        refs = tuple(converted)
                        metrics = score_quality_retrieval(dataset, mapping, case, refs)
                        failure = _stop_category(recorder)
                    except LLMProviderError:
                        failure = "provider"
                    except _ScopeLeak:
                        failure = "safety"
                    except LLMAccountingError:
                        failure = _stop_category(recorder) or "integrity"
                    finally:
                        CURRENT_LOGICAL_CALL.reset(token)
                    stop = _stop_category(recorder)
                    if failure == "safety":
                        stop = "safety"
                    elif failure == "integrity":
                        stop = "integrity"
                    results.append(
                        _case_result(
                            manifest,
                            index,
                            recorder,
                            executed=True,
                            failure=failure,
                            seconds=monotonic() - query_started,
                            refs=refs if failure is None else (),
                            metrics=metrics if failure is None else None,
                        )
                    )
                    active_index = None
                    write_quality_retrieval_artifact(
                        output_dir / f"case-{index:04d}.json", results[-1]
                    )
                    if stop:
                        break
    except asyncio.CancelledError as error:
        cancellation, stop = error, "cancelled"
    except LLMProviderError:
        stop = "provider"
    except Exception:
        # Boundary catches configuration/DB/invariant errors without logging exception bodies.
        stop = _stop_category(recorder) or "integrity"
    finally:
        if not ingestion_seconds:
            ingestion_seconds = monotonic() - ingestion_started
        ingestion_usage = _usage(recorder, "ingestion.")
        if active_index is not None:
            results.append(
                _case_result(
                    manifest,
                    active_index,
                    recorder,
                    executed=True,
                    failure=stop or "integrity",
                    seconds=monotonic() - query_started,
                )
            )
        while len(results) < len(manifest.execution_order):
            results.append(_case_result(manifest, len(results), recorder))
        full = [
            r.metrics.required_unit_coverage
            for r in results
            if r.metrics is not None and r.metrics.required_unit_coverage.denominator
        ]
        failed = sum(r.observation.status == "failed" for r in results)
        not_run = sum(r.observation.status == "not_run" for r in results)
        usage = _usage(recorder)
        valid = stop not in {"configuration", "integrity", "safety"} and usage.accounting_complete
        complete = representation_complete and not not_run and stop is None
        report = QualityRetrievalReportV1(
            manifest=manifest,
            mapping_digest=mapping_report.mapping_digest,
            policy=policy,
            semantic_quality_claim=provider_mode == "qwen"
            and valid
            and complete
            and any(r.metrics is not None and r.observation.provider_attempts for r in results),
            representations=tuple(representations),
            representation_complete=representation_complete,
            ingestion_usage=ingestion_usage,
            total_usage=usage,
            ingestion_seconds=ingestion_seconds,
            cases=tuple(results),
            full_coverage_cases=ratio(
                sum(r.numerator == r.denominator for r in full),
                len(full),
                unassessed=failed + not_run,
            ),
            executed=len(results) - not_run,
            failed=failed,
            not_run=not_run,
            scope_leakage_count=sum(r.observation.safety.unauthorized_access for r in results),
            evidence_valid=valid,
            measurement_complete=complete,
            stop_reason=stop,
        )
        try:
            write_quality_retrieval_artifact(output_dir / "report.json", report)
        except QualityRetrievalError:
            if cancellation is not None:
                raise cancellation from None
            raise
    if cancellation is not None:
        raise cancellation from None
    return report
