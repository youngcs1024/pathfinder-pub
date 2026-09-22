"""E4.6 manual generation slice over the production graph, Registry and DB ports.

This dev-only harness does not start an API/worker, approve actions or accept baselines.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import uuid4

from langsmith import tracing_context
from sqlalchemy import select, text, update

from app.agents.contracts import AgentLoopControl, AgentLoopLimitsV1
from app.agents.research_contracts import (
    ResearchGraphInputV1,
    ResearchGraphOutputStateV1,
    ResearchRequestV1,
)
from app.agents.research_graph import (
    ResearchGraphNodes,
    ResearchGraphProtocolError,
    ResearchGraphRuntimeContext,
    build_research_state_graph,
)
from app.agents.research_nodes import (
    CreateAgentResearchNode,
    DeterministicEvidenceValidationNode,
    StructuredResearchPlanNode,
    StructuredResearchWriterNode,
)
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import Run, RunJob, ToolInvocation, User
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.retrieval_events import SqlAlchemyRetrievalEventRecorder
from app.db.session import AsyncSessionFactory, transaction
from app.db.tool_invocations import SqlAlchemyToolInvocationRecorder
from app.domain.provisioning import ProvisioningService
from app.domain.runs import CURRENT_GRAPH_VERSION, DEFAULT_RUN_LIMITS, RunMode
from app.domain.tenancy import TenantContext
from app.domain.tracing import bind_trace_scope
from app.llm.factory import LLMAccountingError, LLMFactory, LLMProviderError
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext, NoOpTraceSink
from app.llm.ports import LOCKED_CHAT_MODEL
from app.llm.qwen_adapters import QWEN_MAX_OUTPUT_TOKENS, QWEN_REASONING_EFFORT
from app.retrieval.chunking import PreparedIngestionBatch, normalize_document_content
from app.retrieval.documents import DocumentIngestionService, DocumentRetrievalService
from app.tools.contracts import ToolRunContext
from app.tools.document_retrieval import create_research_tool_registry
from app.tools.search import SearchResult, search_source_id, validate_search_call
from app.tools.web_search import RESEARCH_TOOL_POLICY_NAME
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from tests.evals.harness import _CollectingObserver, runtime_version_metadata
from tests.evals.live_baseline import PROJECT_ROOT, probe_clean_git_head
from tests.evals.live_chat import CURRENT_LOGICAL_CALL, _Cancellation
from tests.evals.live_retrieval import DISPOSABLE_DATABASE_NAME
from tests.evals.quality_contracts import (
    QualityGenerationCaseV1,
    QualityGenerationPolicyV1,
    QualityGenerationReportV1,
    QualityGenerationStartV1,
    QualityMappingV1,
    QualityModelPayloadV1,
    QualityObservationV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityPrivateSourceV1,
    QualityPrivateToolV1,
    QualityRetrievalAdmissionV1,
    QualityRunManifestV1,
    QualityStageTimeV1,
)
from tests.evals.quality_dataset import (
    QualityDataset,
    _read_file,
    load_quality_dataset,
    prepare_quality_mapping_sources,
    project_model_payload,
    quality_digest,
    quality_identity_digest,
    validate_quality_mapping,
    validate_quality_run,
)
from tests.evals.quality_e7a_assessment import load_assessment_prompt
from tests.evals.quality_e7a_contracts import ASSESSMENT_POLICY_VERSION, OUTPUT_CONTRACT
from tests.evals.quality_e7a_graph import (
    GRAPH_VERSION,
    E7AGenerationResultV1,
    E7AGraphError,
    E7AGraphRuntime,
    E7ASourceScope,
    E7AUsageOwner,
    build_e7a_generation_graph,
    evidence_context,
)
from tests.evals.quality_e7a_writer import load_writer_prompt as load_e7a_writer_prompt
from tests.evals.quality_experiment_database import (
    FIXTURE_VERSION,
    ExperimentDatabaseError,
    require_owned_database,
)
from tests.evals.quality_generation_support import (
    ExperimentGenerationReportV1,
    ExperimentGenerationStartV1,
    GenerationArtifacts,
    GenerationGuard,
    GenerationSafetyError,
    QualityGenerationError,
    artifact_bytes,
    secret_markers,
)
from tests.evals.quality_run import _QualityAttemptRecorder, _representation, _stop_category, _usage
from tests.legacy_runtime import SqlAlchemyRunStore

GENERATION_SUITE_VERSION = "quality-generation-v1"


def generation_prompt_digest(*, arm="baseline") -> str:
    versions = runtime_version_metadata()
    if arm == "candidate":
        return quality_identity_digest(
            [
                versions.plan_prompt_version,
                versions.research_prompt_version,
                load_assessment_prompt().version,
                load_e7a_writer_prompt().version,
            ]
        )
    if arm != "baseline":
        raise QualityGenerationError("invalid_generation_arm")
    return quality_identity_digest(
        [
            versions.plan_prompt_version,
            versions.research_prompt_version,
            versions.writer_prompt_version,
        ]
    )


def generation_configuration_digest(
    policy: QualityGenerationPolicyV1, factory: LLMFactory, *, arm="baseline"
) -> str:
    if arm == "candidate":
        return quality_identity_digest(
            {
                "baseline_configuration": generation_configuration_digest(policy, factory),
                "graph_version": GRAPH_VERSION,
                "output_contract": OUTPUT_CONTRACT,
                "assessment_policy_version": ASSESSMENT_POLICY_VERSION,
                "prompt_digest": generation_prompt_digest(arm=arm),
            }
        )
    if arm != "baseline":
        raise QualityGenerationError("invalid_generation_arm")
    return quality_identity_digest(
        {
            "suite": GENERATION_SUITE_VERSION,
            "policy": policy.model_dump(mode="json"),
            "versions": runtime_version_metadata().model_dump(mode="json"),
            "provider": factory.provider,
            "chat_model": factory.chat_model,
            "embedding_model": factory.embedding_model,
            "retry": asdict(factory.retry_policy),
            "pricing_version": factory.price_book.version,
            "graph_version": CURRENT_GRAPH_VERSION,
            "prompt_digest": generation_prompt_digest(),
            "reasoning_effort": QWEN_REASONING_EFFORT,
            "max_output_tokens": QWEN_MAX_OUTPUT_TOKENS,
            "seed": None,
            "temperature": None,
            "trace": "off",
            "web": "frozen_fixture",
        }
    )


class FrozenQualityWeb:
    """One verified source per case, independent of query spelling; never consults gold."""

    def __init__(self, source: QualityPrivateSourceV1 | None):
        self.results = ()
        if source is not None:
            content = source.text.strip()
            url = f"https://quality.invalid/{source.source_alias}"
            # Oversized fixtures fail preflight instead of silently dropping evidence.
            self.results = (
                SearchResult.model_validate_json(
                    json.dumps(
                        dict(
                            source_id=search_source_id(url),
                            title=source.source_alias,
                            url=url,
                            snippet=content,
                        )
                    )
                ),
            )

    async def search(self, query, max_results, deadline):
        validate_search_call(query, max_results, deadline)
        return self.results[:max_results]


class _CheckedRetrieval:
    def __init__(self, delegate, guard, documents, chunks, prepared):
        self.delegate, self.guard = delegate, guard
        self.documents, self.chunks, self.prepared = documents, chunks, prepared

    async def retrieve(self, *, tenant, query, allowed_document_ids):
        try:
            hits = await self.delegate.retrieve(
                tenant=tenant, query=query, allowed_document_ids=allowed_document_ids
            )
            for hit in hits:
                if hit.document_id not in allowed_document_ids or hit.chunk_id not in self.chunks:
                    self.guard.scope_failure()
                alias, ordinal = self.chunks[hit.chunk_id]
                if self.documents[alias] != hit.document_id or ordinal != hit.ordinal:
                    self.guard.scope_failure()
                if not self.prepared[alias].chunks[ordinal].text.startswith(hit.text):
                    self.guard.integrity_failed = True
                    raise QualityGenerationError("retrieved_content_mismatch")
                self.guard.scan(hit.text)
            return hits
        except (
            LLMProviderError,
            LLMAccountingError,
            GenerationSafetyError,
            asyncio.CancelledError,
        ):
            raise
        except Exception:
            self.guard.integrity_failed = True
            raise QualityGenerationError("retrieval_integrity_failure") from None


class _ObservedTools:
    def __init__(self, delegate, guard, prefix, recorder):
        self.delegate, self.guard, self.prefix = delegate, guard, prefix
        self.recorder = recorder
        self.authorize = None
        self.records: list[QualityPrivateToolV1] = []
        self.calls = 0

    def model_tools(self):
        return self.delegate.model_tools()

    def validate_call(self, call):
        self.delegate.validate_call(call)

    async def execute(self, call):
        if self.authorize is not None:
            await self.authorize()
        self.guard.check()
        if self.recorder.stop_reason:
            raise QualityGenerationError("generation_stopped")
        self.validate_call(call)
        self.guard.scan(call.model_dump_json())
        self.calls += 1
        record = QualityPrivateToolV1(
            tool_name=call.name, arguments=call.arguments, result=None, delivered=False
        )
        record_index = len(self.records)
        self.records.append(record)
        token = CURRENT_LOGICAL_CALL.set(f"{self.prefix}tool.{self.calls}")
        try:
            result = await self.delegate.execute(call)
            if self.recorder.stop_reason:
                raise QualityGenerationError("generation_stopped")
            self.guard.scan(result)
            self.records[record_index] = record.model_copy(
                update={"result": result, "delivered": True}
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            self.guard.tool_failed = True
            raise
        finally:
            CURRENT_LOGICAL_CALL.reset(token)


class _ObservedChat:
    def __init__(self, delegate, guard, prefix, recorder, *, authorize=None):
        self.delegate, self.guard, self.prefix = delegate, guard, prefix
        self.recorder = recorder
        self.authorize = authorize
        self.count = 0

    async def invoke(self, messages, tools, metadata):
        if self.authorize is not None:
            await self.authorize()
        self.guard.check()
        if self.recorder.stop_reason:
            raise QualityGenerationError("generation_stopped")
        self.count += 1
        for message in messages:
            self.guard.scan(message.model_dump_json())
        token = CURRENT_LOGICAL_CALL.set(f"{self.prefix}chat.{self.count}")
        try:
            result = await self.delegate.invoke(messages, tools, metadata)
            if self.recorder.stop_reason:
                raise QualityGenerationError("generation_stopped")
            self.guard.scan(
                result.model_dump_json() + (result.content or ""),
                system_prompts=tuple(
                    m.content for m in messages if m.role == "system" and m.content
                ),
            )
            return result
        except LLMAccountingError:
            if self.recorder.stop_reason is None:
                self.recorder.stop("accounting_integrity_failure")
            raise
        finally:
            CURRENT_LOGICAL_CALL.reset(token)


class _ObservedToolRecorder:
    """Delegates every write unchanged; remembers failures masked by Registry typed errors."""

    def __init__(self, delegate, guard):
        self.delegate, self.guard = delegate, guard

    def __getattr__(self, name):
        if name not in {"consumed_call_count", "reserve", "start_attempt", "succeed", "fail"}:
            raise AttributeError(name)

        async def call(**kwargs):
            try:
                return await getattr(self.delegate, name)(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.guard.tool_accounting_complete = False
                raise QualityGenerationError("tool_accounting_failure") from None

        return call


class _ObservedRetrievalEvents:
    def __init__(self, delegate, guard):
        self.delegate, self.guard = delegate, guard

    async def record_retrieved(self, **kwargs):
        try:
            await self.delegate.record_retrieved(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.guard.integrity_failed = True
            raise QualityGenerationError("retrieval_event_integrity_failure") from None


async def _prepare_test_run(sessions, tenant, case, document_id, timeout):
    """Only in the confirmed disposable DB: seed a leased fixture, no JobClaimer or worker."""
    request = ResearchRequestV1(
        query=case.query, include_application_draft=case.mode == "application"
    )
    accepted = await SqlAlchemyRunStore(sessions).create_run(
        tenant=tenant,
        mode=RunMode(case.mode),
        resume_document_id=document_id,
        request=request,
        limits=dict(DEFAULT_RUN_LIMITS),
        graph_version=CURRENT_GRAPH_VERSION,
    )
    now = datetime.now(UTC)
    async with transaction(sessions) as session:
        conversation_id = await session.scalar(
            update(Run)
            .where(Run.workspace_id == tenant.workspace_id, Run.id == accepted.run_id)
            .values(status="running", started_at=now)
            .returning(Run.conversation_id)
        )
        await session.execute(
            update(RunJob)
            .where(RunJob.workspace_id == tenant.workspace_id, RunJob.run_id == accepted.run_id)
            .values(
                status="leased",
                attempt=1,
                leased_by="quality-generation-harness",
                owner_token=uuid4(),
                lease_expires_at=now + timedelta(seconds=timeout + 60),
            )
        )
    return ResearchGraphInputV1(
        schema_version=2,
        run_id=accepted.run_id,
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
        conversation_id=conversation_id,
        graph_version=CURRENT_GRAPH_VERSION,
        mode=case.mode,
        resume_document_id=document_id,
        request=request,
    )


async def _finish_test_run(sessions, tenant, run_id, output, failure):
    async with transaction(sessions) as session:
        await session.execute(
            update(Run)
            .where(Run.workspace_id == tenant.workspace_id, Run.id == run_id)
            .values(
                status="cancelled"
                if failure == "cancelled"
                else "failed"
                if failure
                else "completed",
                result_json=output.model_dump(mode="json") if output is not None else None,
                error_category="graph_execution_failed"
                if failure and failure != "cancelled"
                else None,
                finished_at=datetime.now(UTC),
            )
        )
        await session.execute(
            update(RunJob)
            .where(RunJob.workspace_id == tenant.workspace_id, RunJob.run_id == run_id)
            .values(status="done", leased_by=None, owner_token=None, lease_expires_at=None)
        )


def _case_observation(
    manifest,
    index,
    recorder,
    *,
    executed,
    failure,
    guard,
    tools,
    elapsed,
    digest=None,
    private_digest=None,
    rejected=False,
):
    prefix = f"case.{index}."
    usage = _usage(recorder, prefix)
    # Count logical chat invocations once, including provider retries.
    chats = {
        recorder.logical_ids[key]
        for key, attempt in recorder.attempts.items()
        if recorder.logical_ids[key].startswith(prefix) and attempt.invocation_kind == "chat"
    }
    return QualityGenerationCaseV1(
        observation=QualityObservationV1(
            experiment_id=manifest.experiment_id,
            case_id=manifest.execution_order[index].case_id,
            repeat_index=manifest.execution_order[index].repeat_index,
            measurement_scope=manifest.measurement_scope,
            status="not_run" if not executed else "failed" if failure else "succeeded",
            failure_type=failure,
            safety=guard.safety(),
            tool_calls=tools.calls if tools else 0,
            model_calls=len(chats),
            provider_attempts=usage.provider_attempts,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=usage.cost,
            stage_times=(QualityStageTimeV1(stage="generation", seconds=elapsed),)
            if executed
            else (),
            output_digest=digest,
            assessment_required=digest is not None,
        ),
        scope_rejected=rejected,
        private_case_digest=private_digest,
        tool_accounting_complete=guard.tool_accounting_complete,
    )


def _global_failure(guard, recorder):
    if guard.secret_leak or guard.unauthorized_access:
        return "safety"
    if guard.configuration_failed:
        return "configuration"
    if guard.integrity_failed or not guard.tool_accounting_complete:
        return "integrity"
    return _stop_category(recorder)


@dataclass(repr=False)
class GenerationSession:
    sessions: AsyncSessionFactory
    dataset: QualityDataset
    prepared: dict
    sources: tuple
    webs: dict
    manifest: QualityRunManifestV1
    policy: QualityGenerationPolicyV1
    start: QualityGenerationStartV1
    recorder: object
    factory: LLMFactory
    artifacts: GenerationArtifacts
    guard: GenerationGuard
    repository: object
    owner: object = None
    arm: str = "baseline"
    tenant: TenantContext | None = None
    actual_repository: object = None
    documents: dict = field(default_factory=dict)
    chunks: dict = field(default_factory=dict)
    cases: dict = field(default_factory=dict)
    representations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    runs: dict = field(default_factory=dict)
    usage_owners: dict = field(default_factory=dict)
    complete_representation: bool = False
    ingestion_seconds: float = 0.0
    stop: str | None = None
    cancellation: BaseException | None = None
    active: bool = False
    closed: bool = False
    active_tools: object = None
    active_run: object = None
    active_guard: GenerationGuard | None = None
    active_started: float = 0.0
    experiment_hooks: object = None


@dataclass(frozen=True, slots=True, repr=False)
class _GenerationFactory(LLMFactory):
    observed_chat: object = None

    def create_chat_model(self, context):
        return self.observed_chat


async def prepare_generation_session(
    sessions: AsyncSessionFactory | None = None,
    *,
    dataset_root: Path,
    dataset: QualityDataset,
    mapping: QualityMappingV1,
    mapping_rules_digest: str,
    manifest: QualityRunManifestV1,
    policy: QualityGenerationPolicyV1,
    output_dir: Path,
    private_root: Path,
    provider_mode: str = "fake",
    confirm_disposable_database: bool = False,
    confirm_live: bool = False,
    factory: LLMFactory | None = None,
    repository=None,
    git_probe: Callable[[], str] | None = None,
    sensitive_markers: tuple[str, ...] = (),
    owned_database=None,
    arm: str = "baseline",
    experiment_hooks=None,
) -> GenerationSession:
    """Manual caller must own the disposable database exclusively; no concurrent worker.

    Live requires explicit mode, confirmation and an injected governed Factory. Test seams
    are trusted Python arguments, never tool/model input. All output paths are caller-owned.
    """
    owner = None
    if owned_database is not None:
        owner = require_owned_database(owned_database)
        await owner.verify()
        if sessions is not None and sessions is not owner._sessions:
            raise QualityGenerationError("foreign_generation_sessions")
        sessions = owner._sessions
    if arm not in {"baseline", "candidate"} or (arm == "candidate" and owner is None):
        raise QualityGenerationError("invalid_generation_arm")
    if experiment_hooks is not None and owner is None:
        raise QualityGenerationError("experiment_hooks_require_owned_database")
    graph_version = GRAPH_VERSION if arm == "candidate" else CURRENT_GRAPH_VERSION
    guard = GenerationGuard(secret_markers(sensitive_markers))
    try:
        manifest = QualityRunManifestV1.model_validate_json(manifest.model_dump_json())
        policy = QualityGenerationPolicyV1.model_validate_json(policy.model_dump_json())
        mapping = QualityMappingV1.model_validate_json(mapping.model_dump_json())
        if (
            load_quality_dataset(dataset_root) != dataset
            or dataset.manifest.license_category != "synthetic"
        ):
            raise QualityGenerationError("invalid_dataset")
        validate_quality_run(dataset, manifest)
        prepared = prepare_quality_mapping_sources(dataset_root)
        mapping_report = validate_quality_mapping(
            dataset, mapping, prepared, rules_digest=mapping_rules_digest
        )
        delegate = SqlAlchemyInvocationRecorder(sessions)
        if factory is None and arm == "candidate":
            from tests.evals.quality_generation_fixtures import E7AFakeChatAdapter

            factory = LLMFactory(delegate, E7AFakeChatAdapter(), FakeEmbeddingModel())
        if factory is None:
            factory = LLMFactory(
                delegate, DeterministicResearchFakeChatAdapter(), FakeEmbeddingModel()
            )
        if (
            not mapping_report.mapping_complete
            or mapping_report.pending_review_chunks
            or provider_mode not in {"fake", "qwen"}
            or provider_mode != manifest.llm_mode
            or provider_mode != factory.provider
            or manifest.suite_version != GENERATION_SUITE_VERSION
            or manifest.measurement_scope
            != ("contract" if provider_mode == "fake" else "generation")
            or manifest.web_mode != "frozen_fixture"
            or manifest.document_mode
            != ("fake_embedding_db" if provider_mode == "fake" else "real_embedding_db")
            or manifest.model != LOCKED_CHAT_MODEL
            or manifest.graph_version != graph_version
            or manifest.prompt_digest != generation_prompt_digest(arm=arm)
            or manifest.embedding_profile != policy.retrieval.embedding_profile
            or manifest.retrieval_policy_digest
            != quality_identity_digest(policy.retrieval.model_dump(mode="json"))
            or manifest.configuration_digest
            != generation_configuration_digest(policy, factory, arm=arm)
            or policy.retrieval.unknown_attempt_reserve_cny > manifest.cost_admission_budget_cny
            or (owner is None and not confirm_disposable_database)
            or (provider_mode == "qwen" and not confirm_live)
            or (git_probe or probe_clean_git_head)() != manifest.execution_source_sha
        ):
            raise QualityGenerationError("invalid_generation_configuration")
        source_files = {f.path: f.digest for f in dataset.manifest.files}
        loaded_sources = []
        for source in dataset.manifest.sources:
            raw_source = _read_file(dataset_root, source.path)
            if quality_digest(raw_source) != source_files[source.path]:
                raise QualityGenerationError("source_identity_changed")
            content = normalize_document_content(raw_source.decode("utf-8"))
            if source.kind == "resume" and content != prepared[source.alias].content:
                raise QualityGenerationError("source_representation_changed")
            loaded_sources.append(
                QualityPrivateSourceV1(
                    source_alias=source.alias,
                    kind=source.kind,
                    text=content,
                    digest=quality_digest(content.encode("utf-8")),
                )
            )
        sources = tuple(loaded_sources)
        webs = {s.source_alias: FrozenQualityWeb(s) for s in sources if s.kind == "web"}
        guard.scan(QualityPrivateSourcesV1(sources=sources).model_dump_json())
        for case in dataset.cases:
            if case.case_id in manifest.selected_case_ids:
                guard.scan(case.query)
        async with sessions() as session:
            name = await session.scalar(text("SELECT current_database()"))
            reused = await session.scalar(
                select(User.id).where(User.auth_subject == f"quality-{manifest.experiment_id}")
            )
        if not isinstance(name, str) or not DISPOSABLE_DATABASE_NAME.fullmatch(name) or reused:
            raise QualityGenerationError("invalid_experiment_database")
        versions = runtime_version_metadata()
        start = QualityGenerationStartV1(
            manifest=manifest,
            policy=policy,
            mapping_digest=mapping_report.mapping_digest,
            plan_prompt_digest=versions.plan_prompt_version,
            research_prompt_digest=versions.research_prompt_version,
            writer_prompt_digest=versions.writer_prompt_version,
        )
        if owner is not None:
            start = ExperimentGenerationStartV1(
                **{
                    **start.model_dump(),
                    "writer_prompt_digest": load_e7a_writer_prompt().version
                    if arm == "candidate"
                    else versions.writer_prompt_version,
                },
                arm=arm,
                fixture_version=FIXTURE_VERSION,
                schema_digest=owner.identity["schema_digest"],
                output_contract=OUTPUT_CONTRACT if arm == "candidate" else "research_output_v2",
                assessment_policy_version=ASSESSMENT_POLICY_VERSION if arm == "candidate" else None,
                assessment_prompt_digest=load_assessment_prompt().version
                if arm == "candidate"
                else None,
            )
        guard.scan(artifact_bytes(start, private=False).decode())
        admission = QualityRetrievalAdmissionV1.model_validate_json(
            json.dumps(
                {
                    **manifest.model_dump(mode="json"),
                    "unknown_attempt_reserve_cny": str(
                        policy.retrieval.unknown_attempt_reserve_cny
                    ),
                }
            )
        )
        if experiment_hooks is not None:
            experiment_hooks.bind(owned_database, manifest, policy)
        recorder = _QualityAttemptRecorder(delegate, admission, experiment_hooks=experiment_hooks)
        factory = replace(factory, recorder=recorder, trace_sink=NoOpTraceSink())
        artifacts = GenerationArtifacts.reserve(
            output_dir, private_root, manifest.experiment_id, guard, PROJECT_ROOT
        )
    except Exception:
        raise QualityGenerationError("generation_preflight_failed") from None

    state = GenerationSession(
        sessions=sessions,
        dataset=dataset,
        prepared=prepared,
        sources=sources,
        webs=webs,
        manifest=manifest,
        policy=policy,
        start=start,
        recorder=recorder,
        factory=factory,
        artifacts=artifacts,
        guard=guard,
        repository=repository,
        owner=owner,
        arm=arm,
        experiment_hooks=experiment_hooks,
    )
    token = object()
    if owner is not None:
        owner.claim_slot(token)
    try:
        await _ingest_generation(state)
    finally:
        if owner is not None:
            owner.release_slot(token)
    return state


async def _ingest_generation(state):
    sessions = state.sessions
    dataset = state.dataset
    prepared = state.prepared
    sources = state.sources
    manifest = state.manifest
    recorder = state.recorder
    factory = state.factory
    artifacts = state.artifacts
    guard = state.guard
    repository = state.repository
    representations = state.representations
    ingestion_started = monotonic()
    try:
        artifacts.write("manifest.json", state.start)
        artifacts.write("manifest.json", state.start, private=True)
        artifacts.write("sources.json", QualityPrivateSourcesV1(sources=sources), private=True)
        with tracing_context(enabled=False), bind_trace_scope(None):
            actor = await ProvisioningService(
                SqlAlchemyProvisioningStore(sessions)
            ).provision_personal_workspace(f"quality-{manifest.experiment_id}")
            tenant = TenantContext(actor.workspace_id, actor.user_id, actor.role)
            actual_repository = repository or SqlAlchemyDocumentRepository(sessions)
            embedding = factory.create_embedding_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
            )
            documents, chunks = {}, {}
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
                    documents[alias] = document_id
                    chunks.update({key: (alias, ordinal) for key, ordinal in ids.items()})
                    artifacts.write(
                        f"representation-{len(representations):04d}.json", representation
                    )
                finally:
                    CURRENT_LOGICAL_CALL.reset(token)
                if _stop_category(recorder):
                    break
            stop = _global_failure(guard, recorder)
            complete_representation = len(representations) == len(prepared) and stop is None
            cases = {case.case_id: case for case in dataset.cases}
        state.tenant, state.actual_repository = tenant, actual_repository
        state.documents, state.chunks, state.cases = documents, chunks, cases
        state.complete_representation, state.stop = complete_representation, stop
    except asyncio.CancelledError as error:
        state.cancellation, state.stop = error, "cancelled"
    except GenerationSafetyError:
        state.stop = "safety"
    except LLMProviderError:
        state.stop = "provider"
    except Exception:
        state.stop = _global_failure(guard, recorder) or "integrity"
    finally:
        state.ingestion_seconds = monotonic() - ingestion_started


async def _execute_generation_slot(state, index):
    sessions = state.sessions
    tenant = state.tenant
    documents = state.documents
    chunks = state.chunks
    prepared = state.prepared
    actual_repository = state.actual_repository
    webs = state.webs
    manifest = state.manifest
    policy = state.policy
    recorder = state.recorder
    factory = state.factory
    artifacts = state.artifacts
    guard = state.guard
    cancellation = None
    slot = manifest.execution_order[index]
    case = state.cases[slot.case_id]
    case_guard = GenerationGuard(guard.markers)
    state.active_guard = case_guard
    tools = None
    graph_input = None
    output = None
    evidence = None
    output_digest = private_digest = None
    graph_statistics = None
    failure = None
    timed_out = False
    rejected = case.scope_expectation == "reject"
    started = monotonic()
    state.active_started = started
    prefix = f"case.{index}."
    try:
        if rejected:
            failure = "business"
        else:
            payload = project_model_payload(case)
            document_id = documents.get(case.resume_alias)
            if state.owner is None:
                graph_input = await _prepare_test_run(
                    sessions, tenant, payload, document_id, policy.case_timeout_seconds
                )
            else:
                graph_input = await state.owner.create_run(
                    tenant, payload, document_id, policy.case_timeout_seconds, arm=state.arm
                )
            state.active_run = graph_input
            context = LLMInvocationContext(
                tenant.workspace_id, tenant.actor_user_id, run_id=graph_input.run_id
            )
            retrieval = _CheckedRetrieval(
                DocumentRetrievalService(
                    actual_repository, factory.create_embedding_model(context)
                ),
                case_guard,
                documents,
                chunks,
                prepared,
            )
            registry = create_research_tool_registry(
                search_port=webs[case.web_scenario_alias]
                if case.web_scenario_alias
                else FrozenQualityWeb(None),
                retrieval_service=retrieval,
                tenant=tenant,
                allowed_document_ids=(document_id,) if document_id else (),
                recorder=_ObservedToolRecorder(
                    SqlAlchemyToolInvocationRecorder(sessions), case_guard
                ),
                event_recorder=_ObservedRetrievalEvents(
                    SqlAlchemyRetrievalEventRecorder(sessions), case_guard
                ),
            )
            control = AgentLoopControl(
                limits=AgentLoopLimitsV1(**dict(DEFAULT_RUN_LIMITS)),
                deadline=started + policy.case_timeout_seconds,
                cancellation=_Cancellation(asyncio.Event()),
            )
            tools = _ObservedTools(
                registry.bind(
                    policy_name=RESEARCH_TOOL_POLICY_NAME,
                    context=ToolRunContext(
                        workspace_id=tenant.workspace_id,
                        actor_user_id=tenant.actor_user_id,
                        run_id=graph_input.run_id,
                        action_intent_id=None,
                        approval_request_id=None,
                        trusted_target=None,
                        deadline=control.deadline,
                        cancellation=control.cancellation,
                    ),
                ),
                case_guard,
                prefix,
                recorder,
            )
            state.active_tools = tools
            authorize = None
            if state.owner is not None:

                async def authorize():
                    try:
                        await state.owner.authorize(tenant)
                    except ExperimentDatabaseError as error:
                        if error.category == "access_denied":
                            recorder.stop("external_cancelled")
                            raise asyncio.CancelledError from None
                        case_guard.integrity_failed = True
                        raise

                tools.authorize = authorize
            model = _ObservedChat(
                factory.create_chat_model(context),
                case_guard,
                prefix,
                recorder,
                authorize=authorize,
            )
            async with asyncio.timeout(policy.case_timeout_seconds):
                if state.arm == "candidate":
                    observed_factory = _GenerationFactory(
                        **{f.name: getattr(factory, f.name) for f in fields(LLMFactory)},
                        observed_chat=model,
                    )
                    scope = E7ASourceScope(
                        allowed_document_ids=(document_id,) if document_id else (),
                        resume_document_id=document_id,
                        web_available=case.web_scenario_alias is not None,
                        job_tools=("search_web",) if case.web_scenario_alias else (),
                        task_tools=tuple(
                            name
                            for name, available in (
                                ("search_web", bool(case.web_scenario_alias)),
                                ("retrieve_documents", document_id is not None),
                            )
                            if available
                        ),
                    )
                    owner = E7AUsageOwner(on_admitted=lambda usage: None)
                    state.usage_owners[index] = owner
                    runtime = E7AGraphRuntime(
                        observed_factory,
                        context,
                        tools,
                        scope,
                        control,
                        _CollectingObserver(),
                        owner,
                    )
                    raw = await build_e7a_generation_graph().ainvoke(
                        {"payload": {"request": graph_input.request.model_dump(mode="json")}},
                        context=runtime,
                    )
                    generated = E7AGenerationResultV1.model_validate_json(
                        json.dumps(raw["payload"], allow_nan=False), strict=True
                    )
                    output = generated.output
                    graph_statistics = _experiment_graph_statistics(
                        generated.research_state.research.search_calls,
                        generated.research_state.research.document_retrieval_calls,
                        summaries=generated.research_state.summaries,
                    )
                    evidence = evidence_context(generated.research_state, scope)
                else:
                    graph = build_research_state_graph(
                        ResearchGraphNodes(
                            plan=StructuredResearchPlanNode(model),
                            research_agent=CreateAgentResearchNode(model, _CollectingObserver()),
                            validate_evidence=DeterministicEvidenceValidationNode(),
                            write_report=StructuredResearchWriterNode(model),
                        )
                    )
                    input_json = (
                        graph_input.model_dump(mode="json", round_trip=True)
                        if state.owner is None
                        else ResearchGraphInputV1(
                            schema_version=2,
                            run_id=graph_input.run_id,
                            workspace_id=tenant.workspace_id,
                            actor_user_id=tenant.actor_user_id,
                            conversation_id=graph_input.conversation_id,
                            graph_version=CURRENT_GRAPH_VERSION,
                            mode=case.mode,
                            resume_document_id=document_id,
                            request=graph_input.request,
                        ).model_dump(mode="json")
                    )
                    raw = await graph.ainvoke(
                        input_json,
                        context=ResearchGraphRuntimeContext(
                            tool_runtime=tools, agent_loop_control=control
                        ),
                    )
                    parsed = ResearchGraphOutputStateV1.model_validate_json(json.dumps(raw))
                    output = parsed.output
                    graph_statistics = _experiment_graph_statistics(
                        parsed.search_calls, parsed.document_retrieval_calls
                    )
            if _stop_category(recorder) == "cancelled":
                raise asyncio.CancelledError
            case_guard.scan(output.model_dump_json())
    except asyncio.CancelledError as error:
        cancellation, failure = error, "cancelled"
        recorder.stop("external_cancelled")
    except GenerationSafetyError:
        failure = "safety"
    except LLMProviderError:
        failure = "provider"
    except LLMAccountingError:
        failure = _stop_category(recorder) or "integrity"
    except ResearchGraphProtocolError as error:
        failure = "business"
        if error.cause_category == "configuration_error":
            case_guard.configuration_failed = True
        elif error.cause_category in {"cancelled", "external_cancelled"}:
            failure = "cancelled"
            recorder.stop("external_cancelled")
        elif error.cause_category in {
            "provider_timeout",
            "provider_unavailable",
            "model_invocation_failed",
        }:
            failure = "provider"
    except E7AGraphError as error:
        failure = "business"
        if error.category in {"configuration_error", "invalid_source_scope"}:
            failure = "configuration"
            case_guard.configuration_failed = True
        elif error.category in {"budget_exhausted", "deadline_exceeded"}:
            failure = "budget"
        elif error.category in {
            "provider_timeout",
            "provider_unavailable",
            "model_invocation_failed",
        }:
            failure = "provider"
        elif error.category == "cancelled":
            failure = "cancelled"
            recorder.stop("external_cancelled")
    except ExperimentDatabaseError as error:
        failure = (
            "configuration" if error.category in {"schema_drift", "invalid_handle"} else "integrity"
        )
        case_guard.integrity_failed = failure == "integrity"
        case_guard.configuration_failed = failure == "configuration"
    except TimeoutError:
        timed_out = True
        failure = "budget"
    except Exception:
        # LangGraph can wrap a cancelled child in NodeCancelledError.
        # Use Factory's recorded fact, never exception text, to recognize it.
        if _stop_category(recorder) == "cancelled":
            cancellation, failure = asyncio.CancelledError(), "cancelled"
        else:
            failure = "integrity"
            case_guard.integrity_failed = True
    failure = _global_failure(case_guard, recorder) or failure
    if timed_out and failure == "cancelled":
        failure = "budget"
    if case_guard.tool_failed and failure is None:
        failure = "tool"
    if graph_input is not None:
        try:
            async with sessions() as session:
                states = (
                    await session.scalars(
                        select(ToolInvocation.status).where(
                            ToolInvocation.workspace_id == tenant.workspace_id,
                            ToolInvocation.run_id == graph_input.run_id,
                        )
                    )
                ).all()
            if any(state not in {"succeeded", "failed"} for state in states):
                case_guard.tool_accounting_complete = False
            if not case_guard.tool_accounting_complete:
                failure = "integrity" if failure != "safety" else failure
        except Exception:
            case_guard.tool_accounting_complete = False
            failure = "integrity" if failure != "safety" else failure
    if graph_input is not None and state.owner is not None:
        try:
            await state.owner.finish_run(
                tenant,
                graph_input,
                output if failure != "safety" else None,
                failure,
                context=evidence,
            )
            if output is not None and failure != "safety":
                output = await state.owner.read_result(tenant, graph_input, context=evidence)
                case_guard.scan(output.model_dump_json())
        except GenerationSafetyError:
            failure = "safety"
            output = None
        except Exception:
            case_guard.integrity_failed = True
            failure = "integrity" if failure != "safety" else failure
            output = None
    try:
        # Never persist a secret-bearing result, including to the disposable DB.
        if output is not None and failure != "safety":
            output_digest = artifacts.write(
                f"output-{index:04d}.json",
                QualityPrivateOutputV1(
                    case_id=slot.case_id,
                    repeat_index=slot.repeat_index,
                    output=output.model_dump(mode="json"),
                ),
                private=True,
            )
        private_case = QualityPrivateCaseV1(
            case_id=slot.case_id,
            repeat_index=slot.repeat_index,
            input=QualityModelPayloadV1(mode=case.mode, query=case.query),
            resume_alias=case.resume_alias,
            web_scenario_alias=case.web_scenario_alias,
            tools=tuple(tools.records) if tools else (),
            output_digest=output_digest,
            failure_type=failure,
        )
        private_digest = artifacts.write(f"case-{index:04d}.json", private_case, private=True)
    except GenerationSafetyError:
        case_guard.secret_leak += 1
        failure = "safety"
    except QualityGenerationError:
        failure = "integrity"
        case_guard.integrity_failed = True
    if graph_input is not None and state.owner is None:
        try:
            await _finish_test_run(
                sessions,
                tenant,
                graph_input.run_id,
                output if failure != "safety" else None,
                failure,
            )
        except Exception:
            case_guard.integrity_failed = True
            failure = "integrity" if failure != "safety" else failure
    result = _case_observation(
        manifest,
        index,
        recorder,
        executed=True,
        failure=failure,
        guard=case_guard,
        tools=tools,
        elapsed=monotonic() - started,
        digest=output_digest,
        private_digest=private_digest,
        rejected=rejected,
    )
    state.results.append(result)
    artifacts.write(f"case-{index:04d}.json", result)
    if state.experiment_hooks is not None:
        state.experiment_hooks.record_slot(index, result, graph_statistics)
    stop = _global_failure(case_guard, recorder)
    if failure in {"configuration", "integrity", "safety", "cancelled", "budget"}:
        stop = failure
    state.stop = stop
    state.cancellation = cancellation
    if graph_input is not None:
        state.runs[index] = graph_input
    return result


async def finish_generation_session(state):
    if state.active or state.closed:
        raise QualityGenerationError("generation_session_unavailable")
    state.closed = True
    manifest = state.manifest
    recorder = state.recorder
    results = state.results
    artifacts = state.artifacts
    start = state.start
    representations = state.representations
    ingestion_seconds = state.ingestion_seconds
    stop = state.stop
    cancellation = state.cancellation
    complete_representation = state.complete_representation
    while len(results) < len(manifest.execution_order):
        results.append(
            _case_observation(
                manifest,
                len(results),
                recorder,
                executed=False,
                failure=None,
                guard=GenerationGuard(),
                tools=None,
                elapsed=0.0,
            )
        )
    usage = _usage(recorder)
    not_run = sum(c.observation.status == "not_run" for c in results)
    report_type = (
        ExperimentGenerationReportV1 if state.owner is not None else QualityGenerationReportV1
    )
    report = report_type(
        start=start,
        representations=tuple(representations),
        representation_complete=complete_representation,
        ingestion_usage=_usage(recorder, "ingestion."),
        total_usage=usage,
        ingestion_seconds=ingestion_seconds,
        cases=tuple(results),
        private_files=tuple(artifacts.files),
        executed=len(results) - not_run,
        failed=sum(c.observation.status == "failed" for c in results),
        not_run=not_run,
        evidence_valid=stop not in {"configuration", "integrity", "safety"}
        and usage.accounting_complete
        and all(c.tool_accounting_complete for c in results),
        measurement_complete=complete_representation and not not_run and stop is None,
        stop_reason=stop,
    )
    try:
        artifacts.write("report.json", report)
        artifacts.write("report.json", report, private=True)
    except Exception:
        if cancellation is not None:
            raise cancellation from None
        raise QualityGenerationError("generation_report_publication_failed") from None
    if cancellation is not None or stop == "cancelled":
        raise (cancellation or asyncio.CancelledError()) from None
    return report


async def execute_generation_slot(state, index):
    if (
        state.closed
        or state.active
        or not state.complete_representation
        or state.stop
        or type(index) is not int
        or index != len(state.results)
        or index >= len(state.manifest.execution_order)
    ):
        raise QualityGenerationError("generation_slot_unavailable")
    token = object()
    if state.owner is not None:
        state.owner.claim_slot(token)
    state.active = True
    state.active_run = state.active_tools = None
    state.active_guard = GenerationGuard(state.guard.markers)
    state.active_started = monotonic()
    try:
        if state.owner is not None:
            await state.owner.verify()
        with tracing_context(enabled=False), bind_trace_scope(None):
            return await _execute_generation_slot(state, index)
    except asyncio.CancelledError as error:
        state.cancellation, state.stop = error, "cancelled"
        await _preserve_interrupted_slot(state, index)
        raise
    except Exception:
        state.stop = "integrity"
        await _preserve_interrupted_slot(state, index)
        raise
    finally:
        state.active = False
        if state.owner is not None:
            state.owner.release_slot(token)


async def _preserve_interrupted_slot(state, index):
    """Unexpected cancellation during post-graph I/O must retain usage and its slot."""
    if state.active_run is not None:
        state.runs[index] = state.active_run
        try:
            if state.owner is not None:
                await state.owner.finish_run(state.tenant, state.active_run, None, state.stop)
            else:
                await _finish_test_run(
                    state.sessions, state.tenant, state.active_run.run_id, None, state.stop
                )
        except (Exception, asyncio.CancelledError):
            # Commit acknowledgement may be unknown; do not overwrite terminal facts.
            state.active_guard.integrity_failed = True
            state.stop = "integrity"
    if len(state.results) == index:
        files = {f.name: f.digest for f in state.artifacts.files}
        state.results.append(
            _case_observation(
                state.manifest,
                index,
                state.recorder,
                executed=True,
                failure=state.stop,
                guard=state.active_guard,
                tools=state.active_tools,
                elapsed=monotonic() - state.active_started,
                digest=files.get(f"output-{index:04d}.json"),
                private_digest=files.get(f"case-{index:04d}.json"),
            )
        )


async def run_generation(sessions: AsyncSessionFactory | None = None, **kwargs):
    """Legacy batch default and explicit owned experiment arms use the same slot seam."""
    state = await prepare_generation_session(sessions, **kwargs)
    try:
        if state.complete_representation:
            for index in range(len(state.manifest.execution_order)):
                if state.stop:
                    break
                await execute_generation_slot(state, index)
    except asyncio.CancelledError as error:
        state.cancellation, state.stop = error, "cancelled"
    except Exception:
        state.stop = state.stop or "integrity"
    return await finish_generation_session(state)


def _experiment_graph_statistics(search_calls, document_calls, *, summaries=()):
    """Identical returned-reference diagnostic for both arms; never claim entailment.

    Count pass-two tools with newly returned source/chunk identities. Candidate graph
    content-dedup summaries are additional diagnostics, not substituted into that ratio.
    No raw identity, query, evidence or reasoning is exported.
    """
    first, second = set(), set()
    pass_two = 0
    for kind, calls in (("web", search_calls), ("document", document_calls)):
        for call in calls:
            ids = tuple(x.source_id for x in call.results) if kind == "web" else call.chunk_ids
            target = first if call.research_pass_number == 1 else second
            target.update((kind, str(x)) for x in ids)
            pass_two += call.research_pass_number == 2
    return {
        "followup_tool_calls": pass_two,
        "new_returned_references": len(second - first),
        "candidate_pass_summaries": [x.model_dump(mode="json") for x in summaries],
    }
