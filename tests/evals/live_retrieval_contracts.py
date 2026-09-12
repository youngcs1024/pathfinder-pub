"""Independent synthetic real-embedding benchmark input policy; no live execution."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from app.retrieval.chunking import (
    CHUNKING_VERSION,
    NORMALIZATION_VERSION,
    normalize_and_chunk_batch,
)
from app.retrieval.documents import EMBEDDING_DIMENSION, EMBEDDING_PROFILE, RETRIEVAL_TOP_K
from app.retrieval.ingestion import ValidatedIngestionBatch, ValidatedIngestionSource
from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier, EvalQuery
from tests.evals.live_contracts import (
    LiveLatencySummaryV1,
    live_latency_summary,
)
from tests.evals.live_suite import canonical_live_digest

PRIMARY_ALIASES = (
    "backend_foundations",
    "agent_orchestration",
    "rag_pipeline",
    "tool_governance",
    "reliability",
    "eval_observability",
)
DATASET_DIR = Path(__file__).resolve().parents[2] / "evals/datasets"
LIVE_RETRIEVAL_DATASET_DIGEST = (
    "sha256:8d2850aac4f9061fa8cbd9a90e8781c58c076e45b966bada0a1126c5c785d357"
)
LIVE_RETRIEVAL_CASE_SET_DIGEST = (
    "sha256:716da36138a7ff1cd2221462e25e885e11a0142fb328e3c39dd5adde085a7ede"
)
LIVE_RETRIEVAL_QUERY_IDS = (
    "python_api",
    "async_postgres",
    "offline_testing",
    "graph_state",
    "checkpoint_resume",
    "human_approval",
    "text_chunking",
    "vector_embedding",
    "workspace_filter",
    "rag_injection",
    "tool_schema",
    "trusted_context",
    "timeout_retry",
    "submission_idempotency",
    "at_least_once",
    "cancel_run",
    "unknown_outcome",
    "regression_baseline",
    "trace_tree",
    "attempt_metrics",
)


class LiveRetrievalChunkRefV1(EvalContractModel):
    document_alias: EvalIdentifier
    ordinal: int = Field(ge=0, le=3)


class LiveRetrievalDocumentV1(EvalContractModel):
    alias: EvalIdentifier
    role: Literal["primary", "same_workspace_decoy", "foreign_workspace_canary"]
    content: str = Field(min_length=1, max_length=4000)


class LiveRetrievalQueryV1(EvalContractModel):
    query_id: EvalIdentifier
    query: EvalQuery
    allowed_document_aliases: tuple[EvalIdentifier, ...]
    include_foreign_canary_in_allowlist: bool = False
    relevant_chunks: tuple[LiveRetrievalChunkRefV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_references_and_allowlist(self) -> LiveRetrievalQueryV1:
        if len(set(self.relevant_chunks)) != len(self.relevant_chunks):
            raise ValueError("duplicate relevant chunk")
        if self.allowed_document_aliases != PRIMARY_ALIASES:
            raise ValueError("query allowlist must contain exactly the six primary documents")
        return self


def prepared_live_documents(documents: tuple[LiveRetrievalDocumentV1, ...]):
    return normalize_and_chunk_batch(
        ValidatedIngestionBatch(
            sources=tuple(
                ValidatedIngestionSource(
                    source_name=f"{d.alias}.md",
                    source_type="markdown",
                    title=d.alias,
                    raw_text=d.content,
                    character_count=len(d.content),
                )
                for d in documents
            )
        )
    )


class LiveRetrievalDatasetV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_version: Literal["live-retrieval-v1"] = "live-retrieval-v1"
    evidence_scope: Literal["synthetic_live_embedding_benchmark_inputs"]
    label_method: Literal["manual_topic_to_chunk_review_v1"]
    documents: tuple[LiveRetrievalDocumentV1, ...] = Field(min_length=8, max_length=8)
    queries: tuple[LiveRetrievalQueryV1, ...] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def corpus_and_labels_resolve(self) -> LiveRetrievalDatasetV1:
        aliases = tuple(d.alias for d in self.documents)
        if (
            len(set(aliases)) != 8
            or tuple(d.alias for d in self.documents if d.role == "primary") != PRIMARY_ALIASES
        ):
            raise ValueError("expected six unique primary documents")
        if tuple(d.alias for d in self.documents if d.role == "same_workspace_decoy") != (
            "same_workspace_decoy",
        ) or tuple(d.alias for d in self.documents if d.role == "foreign_workspace_canary") != (
            "foreign_workspace_canary",
        ):
            raise ValueError("expected exactly one excluded decoy and foreign canary")
        prepared = prepared_live_documents(self.documents)
        counts = {
            d.alias: len(p.chunks) for d, p in zip(self.documents, prepared.sources, strict=True)
        }
        if (
            any(counts[a] != 4 for a in PRIMARY_ALIASES)
            or sum(counts[a] for a in PRIMARY_ALIASES) != 24
        ):
            raise ValueError("primary candidate pool must be exactly 24 chunks")
        if len({q.query_id for q in self.queries}) != 20:
            raise ValueError("expected 20 unique query IDs")
        if not any(q.include_foreign_canary_in_allowlist for q in self.queries):
            raise ValueError("workspace filtering needs a foreign allowlist canary query")
        for query in self.queries:
            for ref in query.relevant_chunks:
                if (
                    ref.document_alias not in PRIMARY_ALIASES
                    or ref.ordinal >= counts[ref.document_alias]
                ):
                    raise ValueError("relevant chunk must resolve inside the primary allowlist")
        return self


class LiveRetrievalThresholdsV1(EvalContractModel):
    minimum_macro_recall_at_1: Literal[0.50] = 0.50
    minimum_macro_recall_at_3: Literal[0.75] = 0.75
    minimum_macro_recall_at_5: Literal[0.85] = 0.85
    minimum_mean_reciprocal_rank: Literal[0.60] = 0.60


class LiveRetrievalHardInvariantsV1(EvalContractModel):
    workspace_leakage_count: Literal[0] = 0
    allowlist_leakage_count: Literal[0] = 0


class LiveRetrievalPolicyV1(EvalContractModel):
    schema_version: Literal[1] = 1
    dataset_version: Literal["live-retrieval-v1"]
    dataset_digest: EvalDigest
    case_set_digest: EvalDigest
    embedding_profile: str
    embedding_dimension: Literal[1536]
    normalization_version: str
    chunking_version: str
    retrieval_top_k: Literal[5]
    expected_primary_candidate_pool_size: Literal[24]
    query_count: Literal[20]
    thresholds: LiveRetrievalThresholdsV1
    hard_invariants: LiveRetrievalHardInvariantsV1
    irrelevant_context_policy: Literal["record_only_no_acceptance_threshold"]

    @model_validator(mode="after")
    def current_production_identity(self) -> LiveRetrievalPolicyV1:
        if (
            self.embedding_profile,
            self.embedding_dimension,
            self.normalization_version,
            self.chunking_version,
            self.retrieval_top_k,
        ) != (
            EMBEDDING_PROFILE,
            EMBEDDING_DIMENSION,
            NORMALIZATION_VERSION,
            CHUNKING_VERSION,
            RETRIEVAL_TOP_K,
        ):
            raise ValueError("live retrieval policy differs from current production profile")
        return self


class LiveRetrievalCaseReportV1(EvalContractModel):
    query_id: EvalIdentifier
    relevant_chunks: tuple[LiveRetrievalChunkRefV1, ...] = Field(min_length=1)
    ranked_retrieved_chunks: tuple[LiveRetrievalChunkRefV1, ...] = Field(max_length=5)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    irrelevant_context_count: int = Field(ge=0)
    workspace_leakage_count: int = Field(ge=0)
    allowlist_leakage_count: int = Field(ge=0)
    returned_context_bytes: int = Field(ge=0, le=4000)

    @model_validator(mode="after")
    def metrics_match_logical_rankings(self) -> LiveRetrievalCaseReportV1:
        if len(set(self.relevant_chunks)) != len(self.relevant_chunks) or len(
            set(self.ranked_retrieved_chunks)
        ) != len(self.ranked_retrieved_chunks):
            raise ValueError("live retrieval logical chunk references must be unique")
        relevant = frozenset(self.relevant_chunks)
        ranked = self.ranked_retrieved_chunks
        expected = (
            len(set(ranked[:1]) & relevant) / len(relevant),
            len(set(ranked[:3]) & relevant) / len(relevant),
            len(set(ranked[:5]) & relevant) / len(relevant),
            next(
                (1 / rank for rank, ref in enumerate(ranked, start=1) if ref in relevant),
                0.0,
            ),
            sum(ref not in relevant for ref in ranked),
            sum(ref.document_alias == "foreign_workspace_canary" for ref in ranked),
            sum(ref.document_alias not in PRIMARY_ALIASES for ref in ranked),
        )
        actual = (
            self.recall_at_1,
            self.recall_at_3,
            self.recall_at_5,
            self.reciprocal_rank,
            self.irrelevant_context_count,
            self.workspace_leakage_count,
            self.allowlist_leakage_count,
        )
        if actual != expected:
            raise ValueError("live retrieval case metrics do not match logical rankings")
        return self


class LiveRetrievalAggregateV1(EvalContractModel):
    macro_recall_at_1: float = Field(ge=0, le=1)
    macro_recall_at_3: float = Field(ge=0, le=1)
    macro_recall_at_5: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    total_irrelevant_context_count: int = Field(ge=0)
    workspace_leakage_count: int = Field(ge=0)
    allowlist_leakage_count: int = Field(ge=0)


LiveRetrievalStopCategory = Literal[
    "configuration_invariant_failure",
    "provider_failure",
    "embedding_or_retrieval_invariant_failure",
    "cancelled",
    "benchmark_execution_failure",
]


def live_retrieval_passes(
    aggregate: LiveRetrievalAggregateV1,
    policy: LiveRetrievalPolicyV1,
    *,
    complete: bool,
) -> bool:
    return bool(
        complete
        and aggregate.macro_recall_at_1 >= policy.thresholds.minimum_macro_recall_at_1
        and aggregate.macro_recall_at_3 >= policy.thresholds.minimum_macro_recall_at_3
        and aggregate.macro_recall_at_5 >= policy.thresholds.minimum_macro_recall_at_5
        and aggregate.mean_reciprocal_rank >= policy.thresholds.minimum_mean_reciprocal_rank
        and aggregate.workspace_leakage_count == policy.hard_invariants.workspace_leakage_count
        and aggregate.allowlist_leakage_count == policy.hard_invariants.allowlist_leakage_count
    )


class LiveRetrievalReportV1(EvalContractModel):
    schema_version: Literal[1] = 1
    evidence_scope: Literal["exploratory_live_embedding_retrieval_benchmark"]
    exploratory: Literal[True] = True
    provider: Literal["fake", "qwen"]
    semantic_quality_claim: bool
    dataset_version: Literal["live-retrieval-v1"]
    dataset_digest: Literal[LIVE_RETRIEVAL_DATASET_DIGEST]
    case_set_digest: Literal[LIVE_RETRIEVAL_CASE_SET_DIGEST]
    normalization_version: Literal["nfc-lf-v1"]
    chunking_version: Literal["heading-paragraph-utf8-budget-800-v1"]
    embedding_profile: Literal["qwen-beijing-text-embedding-v4-1536-v1"]
    embedding_dimension: Literal[1536]
    retrieval_top_k: Literal[5]
    candidate_pool_size: int = Field(ge=0)
    query_count: Literal[20]
    complete: bool
    cases: tuple[LiveRetrievalCaseReportV1, ...] = Field(max_length=20)
    aggregate: LiveRetrievalAggregateV1
    provider_attempt_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    known_cost_cny: Decimal = Field(ge=0, max_digits=20, decimal_places=12)
    unknown_cost_attempt_count: int = Field(ge=0)
    observed_embedding_attempt_latency_ms: tuple[int, ...] = Field(max_length=84)
    passed: bool
    error_category: LiveRetrievalStopCategory | None = None

    @model_validator(mode="after")
    def report_is_self_consistent(self) -> LiveRetrievalReportV1:
        if self.semantic_quality_claim != (self.provider == "qwen" and self.complete):
            raise ValueError("only a complete real Qwen report can make a semantic quality claim")
        query_ids = tuple(case.query_id for case in self.cases)
        if query_ids != LIVE_RETRIEVAL_QUERY_IDS[: len(query_ids)]:
            raise ValueError("live retrieval report queries must be one ordered dataset prefix")
        if self.complete and query_ids != LIVE_RETRIEVAL_QUERY_IDS:
            raise ValueError("complete live retrieval report must contain every query exactly once")
        if self.complete != (self.error_category is None):
            raise ValueError("live retrieval completion must match sanitized error category")
        dataset = load_live_retrieval_dataset()
        expected_relevant = {query.query_id: query.relevant_chunks for query in dataset.queries}
        known_aliases = {document.alias for document in dataset.documents}
        if any(
            case.relevant_chunks != expected_relevant[case.query_id]
            or any(ref.document_alias not in known_aliases for ref in case.ranked_retrieved_chunks)
            for case in self.cases
        ):
            raise ValueError("live retrieval report labels or ranked aliases differ from dataset")
        count = len(self.cases)
        expected_aggregate = LiveRetrievalAggregateV1(
            macro_recall_at_1=sum(case.recall_at_1 for case in self.cases) / count if count else 0,
            macro_recall_at_3=sum(case.recall_at_3 for case in self.cases) / count if count else 0,
            macro_recall_at_5=sum(case.recall_at_5 for case in self.cases) / count if count else 0,
            mean_reciprocal_rank=sum(case.reciprocal_rank for case in self.cases) / count
            if count
            else 0,
            total_irrelevant_context_count=sum(
                case.irrelevant_context_count for case in self.cases
            ),
            workspace_leakage_count=sum(case.workspace_leakage_count for case in self.cases),
            allowlist_leakage_count=sum(case.allowlist_leakage_count for case in self.cases),
        )
        if self.aggregate != expected_aggregate:
            raise ValueError("live retrieval aggregate does not match case observations")
        policy = load_live_retrieval_policy()
        expected_passed = bool(
            self.provider == "qwen"
            and live_retrieval_passes(
                self.aggregate,
                policy,
                complete=self.complete,
            )
        )
        if self.passed != expected_passed:
            raise ValueError("live retrieval pass status does not match fixed policy")
        if self.candidate_pool_size != policy.expected_primary_candidate_pool_size and (
            self.complete or self.passed
        ):
            raise ValueError("candidate pool mismatch cannot produce a complete passing report")
        if (
            self.unknown_cost_attempt_count > self.provider_attempt_count
            or len(self.observed_embedding_attempt_latency_ms) > self.provider_attempt_count
        ):
            raise ValueError("live retrieval invocation diagnostics are inconsistent")
        if self.complete and (
            self.provider_attempt_count < 28
            or len(self.observed_embedding_attempt_latency_ms) != self.provider_attempt_count
        ):
            raise ValueError("complete live retrieval report lacks terminal attempt evidence")
        return self


class LiveRetrievalReportV2(LiveRetrievalReportV1):
    """Accepted-capable report; V1 remains frozen exploratory evidence."""

    schema_version: Literal[2] = 2
    evidence_scope: Literal["live_embedding_retrieval_benchmark_v2"] = (
        "live_embedding_retrieval_benchmark_v2"
    )
    exploratory: bool
    priced_attempt_count: int = Field(ge=0)
    embedding_provider_latency: LiveLatencySummaryV1

    @model_validator(mode="after")
    def v2_invocation_summary_is_derived(self) -> LiveRetrievalReportV2:
        if self.priced_attempt_count != (
            self.provider_attempt_count - self.unknown_cost_attempt_count
        ):
            raise ValueError("priced retrieval attempts must be derived from all attempts")
        if self.embedding_provider_latency != live_latency_summary(
            tuple(float(value) for value in self.observed_embedding_attempt_latency_ms)
        ):
            raise ValueError("embedding latency summary must be derived from terminal attempts")
        return self


class AcceptedLiveRetrievalBaselineV2(EvalContractModel):
    schema_version: Literal[2] = 2
    report: LiveRetrievalReportV2

    @model_validator(mode="after")
    def acceptance_requires_current_complete_qwen_evidence(
        self,
    ) -> AcceptedLiveRetrievalBaselineV2:
        report = LiveRetrievalReportV2.model_validate_json(
            self.report.model_dump_json(), strict=True
        )
        policy = load_live_retrieval_policy()
        if (
            report.exploratory
            or report.provider != "qwen"
            or not report.complete
            or not report.semantic_quality_claim
            or not report.passed
            or report.error_category is not None
        ):
            raise ValueError("accepted retrieval evidence must be complete non-exploratory Qwen")
        if (
            report.dataset_version != policy.dataset_version
            or report.dataset_digest != policy.dataset_digest
            or report.case_set_digest != policy.case_set_digest
            or report.normalization_version != policy.normalization_version
            or report.chunking_version != policy.chunking_version
            or report.embedding_profile != policy.embedding_profile
            or report.embedding_dimension != policy.embedding_dimension
            or report.retrieval_top_k != policy.retrieval_top_k
            or report.candidate_pool_size != policy.expected_primary_candidate_pool_size
            or report.query_count != policy.query_count
        ):
            raise ValueError("accepted retrieval evidence must match current locked identity")
        if report.aggregate.workspace_leakage_count or report.aggregate.allowlist_leakage_count:
            raise ValueError("retrieval tenant and allowlist invariants must be clean")
        return self


def live_retrieval_digests(dataset: LiveRetrievalDatasetV1) -> tuple[str, str]:
    return (
        canonical_live_digest(
            "pathfinder-live-retrieval-dataset-v1", dataset.model_dump(mode="json")
        ),
        canonical_live_digest(
            "pathfinder-live-retrieval-case-set-v1", sorted(q.query_id for q in dataset.queries)
        ),
    )


def load_live_retrieval_dataset(
    path: Path = DATASET_DIR / "live_retrieval_v1.json",
) -> LiveRetrievalDatasetV1:
    return LiveRetrievalDatasetV1.model_validate_json(path.read_bytes(), strict=True)


def load_live_retrieval_policy(
    path: Path = DATASET_DIR / "live_retrieval_v1_policy.json",
    *,
    dataset: LiveRetrievalDatasetV1 | None = None,
) -> LiveRetrievalPolicyV1:
    policy = LiveRetrievalPolicyV1.model_validate_json(path.read_bytes(), strict=True)
    if (policy.dataset_digest, policy.case_set_digest) != live_retrieval_digests(
        dataset or load_live_retrieval_dataset()
    ):
        raise ValueError("live retrieval policy dataset identity mismatch")
    return policy
