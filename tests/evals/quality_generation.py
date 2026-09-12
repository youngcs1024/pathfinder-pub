"""E4.6 manual generation slice over the production graph, Registry and DB ports.

This dev-only harness does not start an API/worker, approve actions or accept baselines.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, replace
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
from app.db.runs import SqlAlchemyRunStore
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
from tests.evals.quality_generation_support import (
    GenerationArtifacts,
    GenerationGuard,
    GenerationSafetyError,
    QualityGenerationError,
    artifact_bytes,
    secret_markers,
)
from tests.evals.quality_run import _QualityAttemptRecorder, _representation, _stop_category, _usage

GENERATION_SUITE_VERSION = "quality-generation-v1"


def generation_prompt_digest() -> str:
    versions = runtime_version_metadata()
    return quality_identity_digest(
        [
            versions.plan_prompt_version,
            versions.research_prompt_version,
            versions.writer_prompt_version,
        ]
    )


def generation_configuration_digest(policy: QualityGenerationPolicyV1, factory: LLMFactory) -> str:
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
        self.records: list[QualityPrivateToolV1] = []
        self.calls = 0

    def model_tools(self):
        return self.delegate.model_tools()

    def validate_call(self, call):
        self.delegate.validate_call(call)

    async def execute(self, call):
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
    def __init__(self, delegate, guard, prefix, recorder):
        self.delegate, self.guard, self.prefix = delegate, guard, prefix
        self.recorder = recorder
        self.count = 0

    async def invoke(self, messages, tools, metadata):
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


async def run_generation(
    sessions: AsyncSessionFactory,
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
) -> QualityGenerationReportV1:
    """Manual caller must own the disposable database exclusively; no concurrent worker.

    Live requires explicit mode, confirmation and an injected governed Factory. Test seams
    are trusted Python arguments, never tool/model input. All output paths are caller-owned.
    """
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
            or manifest.graph_version != CURRENT_GRAPH_VERSION
            or manifest.prompt_digest != generation_prompt_digest()
            or manifest.embedding_profile != policy.retrieval.embedding_profile
            or manifest.retrieval_policy_digest
            != quality_identity_digest(policy.retrieval.model_dump(mode="json"))
            or manifest.configuration_digest != generation_configuration_digest(policy, factory)
            or policy.retrieval.unknown_attempt_reserve_cny > manifest.cost_admission_budget_cny
            or not confirm_disposable_database
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
        recorder = _QualityAttemptRecorder(delegate, admission)
        factory = replace(factory, recorder=recorder, trace_sink=NoOpTraceSink())
        artifacts = GenerationArtifacts.reserve(
            output_dir, private_root, manifest.experiment_id, guard, PROJECT_ROOT
        )
    except Exception:
        raise QualityGenerationError("generation_preflight_failed") from None

    results = []
    representations = []
    complete_representation = False
    stop = None
    cancellation = None
    ingestion_started = monotonic()
    ingestion_seconds = 0.0
    tenant = None
    active_index = None
    try:
        artifacts.write("manifest.json", start)
        artifacts.write("manifest.json", start, private=True)
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
            ingestion_seconds = monotonic() - ingestion_started
            stop = _global_failure(guard, recorder)
            complete_representation = len(representations) == len(prepared) and stop is None
            cases = {case.case_id: case for case in dataset.cases}
            if complete_representation:
                for index, slot in enumerate(manifest.execution_order):
                    active_index = index
                    case = cases[slot.case_id]
                    case_guard = GenerationGuard(guard.markers)
                    tools = None
                    graph_input = None
                    output = None
                    output_digest = private_digest = None
                    failure = None
                    timed_out = False
                    rejected = case.scope_expectation == "reject"
                    started = monotonic()
                    prefix = f"case.{index}."
                    try:
                        if rejected:
                            failure = "business"
                        else:
                            payload = project_model_payload(case)
                            document_id = documents.get(case.resume_alias)
                            graph_input = await _prepare_test_run(
                                sessions, tenant, payload, document_id, policy.case_timeout_seconds
                            )
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
                            model = _ObservedChat(
                                factory.create_chat_model(context), case_guard, prefix, recorder
                            )
                            graph = build_research_state_graph(
                                ResearchGraphNodes(
                                    plan=StructuredResearchPlanNode(model),
                                    research_agent=CreateAgentResearchNode(
                                        model, _CollectingObserver()
                                    ),
                                    validate_evidence=DeterministicEvidenceValidationNode(),
                                    write_report=StructuredResearchWriterNode(model),
                                )
                            )
                            async with asyncio.timeout(policy.case_timeout_seconds):
                                raw = await graph.ainvoke(
                                    graph_input.model_dump(mode="json", round_trip=True),
                                    context=ResearchGraphRuntimeContext(
                                        tool_runtime=tools, agent_loop_control=control
                                    ),
                                )
                            # Check the persisted stop before interpreting returned state.
                            if _stop_category(recorder) == "cancelled":
                                raise asyncio.CancelledError
                            output = ResearchGraphOutputStateV1.model_validate_json(
                                json.dumps(raw)
                            ).output
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
                        private_digest = artifacts.write(
                            f"case-{index:04d}.json", private_case, private=True
                        )
                    except GenerationSafetyError:
                        case_guard.secret_leak += 1
                        failure = "safety"
                    except QualityGenerationError:
                        failure = "integrity"
                        case_guard.integrity_failed = True
                    if graph_input is not None:
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
                    results.append(result)
                    active_index = None
                    artifacts.write(f"case-{index:04d}.json", result)
                    stop = _global_failure(case_guard, recorder)
                    if failure in {"configuration", "integrity", "safety", "cancelled", "budget"}:
                        stop = failure
                    if stop:
                        break
    except asyncio.CancelledError as error:
        cancellation, stop = error, "cancelled"
    except GenerationSafetyError:
        stop = "safety"
    except LLMProviderError:
        stop = "provider"
    except Exception:
        stop = _global_failure(guard, recorder) or "integrity"
    finally:
        if not ingestion_seconds:
            ingestion_seconds = monotonic() - ingestion_started
        if active_index is not None:
            failure = _global_failure(case_guard, recorder) or stop or "integrity"
            results.append(
                _case_observation(
                    manifest,
                    active_index,
                    recorder,
                    executed=True,
                    failure=failure,
                    guard=case_guard,
                    tools=tools,
                    elapsed=monotonic() - started,
                    digest=output_digest,
                    private_digest=private_digest,
                    rejected=False,
                )
            )
            if graph_input is not None:
                try:
                    await _finish_test_run(sessions, tenant, graph_input.run_id, None, failure)
                except Exception:
                    stop = "integrity"
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
        report = QualityGenerationReportV1(
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
