"""R7.1 synthetic acceptance through business commands and one real worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from random import Random
from time import monotonic
from types import SimpleNamespace
from uuid import UUID, uuid4

from pydantic import SecretStr
from sqlalchemy import select

from app.agents.material_facts import PROMPT_VERSION, MaterialFactExtractor
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import (
    MaterialSnapshotFile,
)
from app.db.project_facts import SqlAlchemyProjectFactStore, extractor_identity
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_confirmation import SqlAlchemyResumeConfirmationStore
from app.db.resume_generation import ResumeGenerationPublisher, SqlAlchemyResumeGenerationStore
from app.db.resume_profiles import ClaimReviewV1, ItemReviewV1, SqlAlchemyResumeProfileStore
from app.db.resume_revision import ResumeRevisionPublisher, SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.project_facts import CandidateFactV1, FactEvidenceV1
from app.domain.provisioning import ProvisioningService
from app.domain.resume_confirmation import ConfirmVersionV1
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.resume_revision import (
    ContentFeedbackV1,
    LockChangeV1,
    PatchV1,
)
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatModelResult
from app.material.aliases import MaterialAlias, MaterialAliasRegistry
from app.material.reader import read_alias
from app.resume.template_render import TemplateIdentity
from app.retrieval.documents import DocumentIngestionService
from app.worker.backoff import ExponentialBackoff
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.generation_executor import GenerationRunExecutor
from app.worker.material_executor import MaterialRunExecutor
from app.worker.revision_executor import RevisionRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.evals.product_acceptance_contracts import publish, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_quality_baseline import common_input, one_shot
from tests.evals.resume_quality_budget import ComparisonRecorder
from tests.evals.resume_quality_contracts import input_change, review_template

SOURCE = Path(__file__).resolve().parents[1] / "fixtures/resume/synthetic_main.tex"


def _material_runner(sessions, store, recorder):
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=FakeChatModel(),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )
    document_repository = SqlAlchemyDocumentRepository(sessions)
    reader = SqlAlchemyRunExecutionReader(sessions)
    facts = SqlAlchemyProjectFactStore(sessions)
    executor = MaterialRunExecutor(
        reader=reader,
        source_reader=read_alias,
        materials=store,
        aliases=store.aliases,
        facts=facts,
        extractor_digest=extractor_identity(PROMPT_VERSION, "fake:qwen3.6-flash-2026-04-16"),
        extractor_factory=lambda tenant, run_id, _scope: MaterialFactExtractor(
            model=factory.create_chat_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            )
        ),
        ingestion_factory=lambda tenant, run_id: DocumentIngestionService(
            repository=document_repository,
            embedding=factory.create_embedding_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            ),
        ),
    )
    return WorkerRunner(
        worker_id="material-test",
        store=SqlAlchemyWorkerJobStore(
            sessions, ExponentialBackoff(WorkerRuntimeSettings(), Random(1))
        ),
        tenant_service=TenantService(SqlAlchemyTenantResolver(sessions)),
        executor=RunExecutorDispatcher(
            {
                "pathfinder-resume-v2": executor,
                "pathfinder-resume-v3": executor,
                "pathfinder-resume-v4": executor,
            }
        ),
        settings=WorkerRuntimeSettings(),
        unsupported_work_guard=reader.has_unsupported_pending_work,
    )


async def seed(sessions, tenant, root, dataset, source_bytes, recorder):
    material_root = root / "material"
    material_root.mkdir(mode=0o700)
    (material_root / "facts.txt").write_text("\n".join(dataset.facts) + "\n")
    aliases = MaterialAliasRegistry(
        (MaterialAlias("code", "file", material_root, ("facts.txt",), (tenant.workspace_id,)),)
    )
    materials = SqlAlchemyMaterialStore(sessions, aliases)
    project = await materials.create_project(tenant, "Synthetic", uuid4())
    source = await materials.create_source(tenant, project["id"], "code", uuid4())
    imported = await materials.submit_import(tenant, project["id"], (source["id"],), uuid4())
    require(
        await _material_runner(sessions, materials, recorder).run_once(asyncio.Event()),
        "material_worker_idle",
    )
    facts = SqlAlchemyProjectFactStore(sessions)
    catalog = await facts.current_facts(tenant, project["id"])
    async with sessions() as db:
        snapshot_file_id = await db.scalar(
            select(MaterialSnapshotFile.id).where(
                MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                MaterialSnapshotFile.path == "facts.txt",
            )
        )
    require(snapshot_file_id is not None, "material_snapshot_missing")
    fact_version_ids = []
    for index, claim_text in enumerate(dataset.facts):
        added = await facts.command(
            tenant,
            kind="material_fact_add",
            project_id=project["id"],
            import_id=imported.receipt.resource_id,
            request_id=uuid4(),
            candidate=CandidateFactV1(
                claim=claim_text,
                kind="personal_statement",
                evidence=(
                    FactEvidenceV1(
                        snapshot_file_id=snapshot_file_id,
                        start_line=index + 1,
                        end_line=index + 1,
                        quote=claim_text,
                    ),
                ),
            ),
        )
        await facts.command(
            tenant,
            kind="material_fact_review",
            project_id=project["id"],
            import_id=catalog["import_id"],
            fact_id=added.receipt.resource_id,
            expected_version=1,
            decision="confirm",
            attested=True,
            request_id=uuid4(),
        )
        confirmed = await facts.current_facts(tenant, project["id"])
        fact_version_ids.append(
            next(
                item["version_id"]
                for item in confirmed["facts"]
                if item["id"] == added.receipt.resource_id
            )
        )
    source_text = source_bytes.decode()
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()
    preamble_hash = hashlib.sha256(
        source_text.split("\\begin{document}", 1)[0].encode()
    ).hexdigest()
    profiles = SqlAlchemyResumeProfileStore(
        sessions,
        expected_source_sha256=source_hash,
        expected_preamble_sha256=preamble_hash,
    )
    accepted = await profiles.import_source(tenant, source_text, uuid4())
    profile_id = accepted.receipt.resource_id
    profile = await profiles.get_profile(tenant, profile_id)
    project_item_id = UUID(profile["content"]["projects"][0]["id"])
    await profiles.command(
        tenant,
        kind="resume_profile_item_review",
        profile_id=profile_id,
        request_id=uuid4(),
        payload=ItemReviewV1(expected_version=1, item_id=project_item_id),
    )
    profile = await profiles.get_profile(tenant, profile_id)
    project_claim = next(
        claim for claim in profile["claims"] if claim["project_item_id"] == project_item_id
    )
    await profiles.command(
        tenant,
        kind="resume_claim_review",
        profile_id=profile_id,
        request_id=uuid4(),
        payload=ClaimReviewV1(
            claim_id=project_claim["id"],
            expected_review_version=0,
            decision="linked",
            project_id=project["id"],
            fact_version_ids=tuple(fact_version_ids),
        ),
    )
    return SimpleNamespace(
        sessions=sessions,
        tenant=tenant,
        request=SessionCreateV1(
            profile_version_id=profile["version_id"],
            preference_version=1,
            project_ids=(project["id"],),
            job=JobInputV1(source="paste", text=dataset.cases[0].jd),
            budget=GenerationBudgetV1(
                max_model_calls=12, max_tool_calls=0, max_cost_cny=Decimal("20")
            ),
        ),
        project_item_id=project_item_id,
        fact_ids=fact_version_ids,
        store=SqlAlchemyResumeGenerationStore(sessions),
        revisions=SqlAlchemyResumeRevisionStore(sessions),
        reader=SqlAlchemyRunExecutionReader(sessions),
        artifacts=SqlAlchemyResumeArtifactStore(
            sessions, expected_source_sha256=source_hash, expected_preamble_sha256=preamble_hash
        ),
        identity=TemplateIdentity(source_sha256=source_hash, preamble_sha256=preamble_hash),
        source_bytes=source_bytes,
        recorder=recorder,
    )


def factory(rig, script):
    return LLMFactory(
        recorder=rig.recorder,
        chat_adapter=ScriptedFakeChatModel(script),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )


def worker(rig, model_factory):
    def make_model(actor, run_id):
        return model_factory.create_chat_model(
            LLMInvocationContext(actor.workspace_id, actor.actor_user_id, run_id=run_id)
        )

    generation = GenerationRunExecutor(
        reader=rig.reader,
        sessions=rig.store,
        model_factory=make_model,
        tools_factory=lambda *_: None,
    )
    revision = RevisionRunExecutor(
        reader=rig.reader, revisions=rig.revisions, model_factory=make_model
    )
    jobs = SqlAlchemyWorkerJobStore(
        rig.sessions,
        ExponentialBackoff(WorkerRuntimeSettings(), Random(1)),
        generation_publisher=ResumeGenerationPublisher(rig.artifacts),
        revision_publisher=ResumeRevisionPublisher(rig.artifacts),
    )
    return WorkerRunner(
        worker_id="r71-comparison",
        store=jobs,
        tenant_service=TenantService(SqlAlchemyTenantResolver(rig.sessions)),
        executor=RunExecutorDispatcher(
            {
                "pathfinder-resume-v2": generation,
                "pathfinder-resume-v3": generation,
                "pathfinder-resume-v4": revision,
            }
        ),
        settings=WorkerRuntimeSettings(),
        unsupported_work_guard=rig.reader.has_unsupported_pending_work,
    )


def response(value):
    return ChatModelResult(content=json.dumps(value))


async def snapshot(rig, session_id):
    detail = await rig.store.get_session(rig.tenant, session_id)
    require(detail["run_status"] == "completed", "system_run_failed")
    version = await rig.store.get_version(rig.tenant, session_id, detail["current_version_id"])
    payload, digest = await rig.artifacts.get_bytes(rig.tenant, version["artifact_id"])
    return {
        "version_id": str(version["version_id"]),
        "artifact_id": str(version["artifact_id"]),
        "tex": payload.decode(),
        "tex_sha256": digest,
        "content": version["content"],
        "diff": version.get("diff", []),
        "run_id": str(detail["run_id"]),
    }


async def run_case(rig, case, root, record):
    started = monotonic()
    request = rig.request.model_copy(update={"job": JobInputV1(source="paste", text=case.jd)})
    key = uuid4()
    created = await rig.store.create(rig.tenant, request, key)
    require((await rig.store.create(rig.tenant, request, key)).replayed, "create_replay_failed")
    session_id = created.receipt.resource_id
    inputs = await rig.store.execution_inputs(rig.tenant, session_id)
    shared = common_input(inputs, rig.identity.source_sha256)
    digest = quality_identity_digest(shared)
    publish(root / f"{case.case_id}-input.json", shared)
    record.update(
        {
            "common_input_digest": digest,
            "session_id": str(session_id),
            "requirements": [r.model_dump(mode="json") for r in case.supported_requirements],
            "arms": {
                name: {"status": "NOT_RUN", "review": review_template(digest)}
                for name in ("b0", "b1", "system")
            },
        }
    )
    # B0 keeps the exact original source; it is not silently rewritten or pre-validated.
    record["arms"]["b0"].update(
        {
            "status": "AVAILABLE",
            "tex_sha256": rig.identity.source_sha256,
            "common_input_digest": digest,
            "logical_generations": 0,
        }
    )
    publish(
        root / f"{case.case_id}-b0.json",
        {
            "tex": rig.source_bytes.decode(),
            "source_sha256": rig.identity.source_sha256,
            "review": record["arms"]["b0"]["review"],
        },
    )
    before = await rig.recorder.measurement()
    baseline_content = inputs.profile_content.model_dump(mode="json")
    baseline_model = factory(
        rig, [response({k: baseline_content[k] for k in ("education", "projects", "skills")})]
    )
    try:
        b1 = await one_shot(
            baseline_model.create_chat_model(
                LLMInvocationContext(rig.tenant.workspace_id, rig.tenant.actor_user_id)
            ),
            inputs,
            rig.source_bytes,
            rig.identity,
        )
        publish(root / f"{case.case_id}-b1.json", b1)
        record["arms"]["b1"].update(
            {
                "status": "GENERATED",
                "tex_sha256": b1["tex_sha256"],
                "common_input_digest": b1["common_input_digest"],
                "logical_generations": 1,
                "automatic_repairs": 0,
            }
        )
    except Exception:
        record["arms"]["b1"]["status"] = "FAILED"
        record["arms"]["b1"]["failure_category"] = "b1_generation_failed"
    after = await rig.recorder.measurement()
    record["arms"]["b1"]["usage"] = usage_delta(before, after)
    require(not rig.recorder.stopped, "budget_stopped")
    chosen = case.supported_requirements[0]
    fact_id = rig.fact_ids[chosen.fact_indexes[0]]
    analysis = {
        "requirements": [
            {
                "kind": chosen.kind,
                "start": case.jd.index(chosen.quote),
                "end": case.jd.index(chosen.quote) + len(chosen.quote),
                "quote": chosen.quote,
            }
        ],
        "questions": [],
    }
    selection = {
        "bullets": [
            {
                "project_item_id": str(rig.project_item_id),
                "fact_version_id": str(fact_id),
                "requirement_ordinals": [0],
            }
        ],
        "questions": [],
        "omitted_fact_version_ids": [str(f) for f in rig.fact_ids if f != fact_id],
    }
    model_factory = factory(rig, [response(analysis), response(selection)])
    system = record["arms"]["system"]
    system.update({"status": "RUNNING", "common_input_digest": digest, "versions": []})
    before = await rig.recorder.measurement()
    require(await worker(rig, model_factory).run_once(asyncio.Event()), "worker_idle")
    initial = await snapshot(rig, session_id)
    system["initial_usage"] = usage_delta(before, await rig.recorder.measurement())
    system["versions"].append(initial)
    publish(root / f"{case.case_id}-system-0.json", initial)
    for ordinal, instruction in enumerate(case.revisions, 1):
        current = system["versions"][-1]
        detail = await rig.store.get_session(rig.tenant, session_id)
        project = current["content"]["projects"][0]
        bullet_id = UUID(project["bullet_ids"][0])
        if ordinal == 2:
            await rig.revisions.command(
                rig.tenant,
                session_id,
                LockChangeV1(
                    expected_session_revision=detail["revision"],
                    base_version_id=detail["current_version_id"],
                    item_id=bullet_id,
                    locked=True,
                ),
                uuid4(),
            )
            detail = await rig.store.get_session(rig.tenant, session_id)
        target = bullet_id if ordinal == 1 else UUID(project["id"])
        text = (
            (project["bullets"][0]["text"] + ".")
            if ordinal == 1
            else project["summary"]["text"] + "."
        )
        patch = PatchV1(
            operation="replace_text",
            item_id=target,
            field="bullet" if ordinal == 1 else "summary",
            text=text,
            fact_version_ids=(fact_id,),
        )
        feedback = ContentFeedbackV1(
            expected_session_revision=detail["revision"],
            base_version_id=detail["current_version_id"],
            target_item_ids=(target,),
            instruction=instruction,
        )
        feedback_key = uuid4()
        await rig.revisions.command(rig.tenant, session_id, feedback, feedback_key)
        require(
            (await rig.revisions.command(rig.tenant, session_id, feedback, feedback_key)).replayed,
            "feedback_replay_failed",
        )
        revision_factory = factory(rig, [response({"patches": [patch.model_dump(mode="json")]})])
        require(await worker(rig, revision_factory).run_once(asyncio.Event()), "worker_idle")
        revised = await snapshot(rig, session_id)
        if ordinal == 2:
            require(
                revised["content"]["projects"][0]["bullets"] == project["bullets"], "lock_changed"
            )
        require(revised["version_id"] != current["version_id"], "revision_not_published")
        system["versions"].append(revised)
        publish(root / f"{case.case_id}-system-{ordinal}.json", revised)
    final = system["versions"][-1]
    detail = await rig.store.get_session(rig.tenant, session_id)
    confirmations = SqlAlchemyResumeConfirmationStore(rig.sessions, rig.artifacts)
    confirmation = ConfirmVersionV1(
        version_id=UUID(final["version_id"]),
        expected_session_revision=detail["revision"],
        expected_current_version_id=detail["current_version_id"],
        artifact_id=UUID(final["artifact_id"]),
        tex_sha256=final["tex_sha256"],
        attested=True,
    )
    confirm_key = uuid4()
    await confirmations.confirm(
        rig.tenant, session_id, confirmation.version_id, confirmation, confirm_key
    )
    require(
        (
            await confirmations.confirm(
                rig.tenant, session_id, confirmation.version_id, confirmation, confirm_key
            )
        )["replayed"],
        "confirmation_replay_failed",
    )
    for version in system["versions"]:
        downloaded = await confirmations.download(
            rig.tenant, session_id, UUID(version["version_id"])
        )
        require(
            downloaded[0] == version["tex"].encode() and downloaded[1] == version["tex_sha256"],
            "history_changed",
        )
    system.update(
        {
            "status": "CONFIRMED_SYNTHETIC",
            "usage": usage_delta(before, await rig.recorder.measurement()),
            "revision_rounds": 2,
            "input_change": input_change(
                shared,
                common_input(
                    await rig.store.execution_inputs(rig.tenant, session_id),
                    rig.identity.source_sha256,
                ),
            ),
            "confirmation_is_human_quality_evidence": False,
        }
    )
    record.update(
        {
            "status": "COMPLETE" if record["arms"]["b1"]["status"] == "GENERATED" else "PARTIAL",
            "wall_seconds": monotonic() - started,
        }
    )


def usage_delta(before, after):
    return {
        "attempts": after["attempts"] - before["attempts"],
        "invocation_ids": [v for v in after["invocation_ids"] if v not in before["invocation_ids"]],
        "known_cost_cny": str(Decimal(after["known_cost_cny"]) - Decimal(before["known_cost_cny"])),
        "unknown_cost": after["unknown_cost"] - before["unknown_cost"],
        "latency_ms": after["latency_ms"] - before["latency_ms"],
        "cost_status": after["cost_status"],
    }


async def execute_synthetic(database_url, root, dataset, budget, *, source_check=lambda: None):
    """Caller owns the isolated, migrated database. No credentials or DSN in reports."""
    require(dataset.material_kind == "synthetic", "synthetic_inputs_required")
    engine = create_database_engine(SecretStr(database_url))
    sessions = create_session_factory(engine)
    report = {
        "artifact_kind": "resume_quality_report_v1",
        "mode": "fake",
        "status": "PARTIAL",
        "live_evidence": "NOT_RUN",
        "manual_compile_evidence": "NOT_RUN",
        "r71_status": "IN_PROGRESS",
        "cases": [{"case_id": case.case_id, "status": "NOT_RUN"} for case in dataset.cases],
    }
    recorder = None
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("r71-" + uuid4().hex)
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=identity.workspace_id, actor_user_id=identity.user_id
        )
        recorder = ComparisonRecorder(
            sessions, tenant, provider="fake", budget=budget, source_check=source_check
        )
        rig = await seed(sessions, tenant, root, dataset, SOURCE.read_bytes(), recorder)
        report["preparation_usage"] = await recorder.measurement()
        for case, record in zip(dataset.cases, report["cases"], strict=True):
            record["status"] = "RUNNING"
            try:
                await run_case(rig, case, root, record)
            except Exception:
                record["status"] = "FAILED"
                record["failure_category"] = (
                    "budget_stopped" if recorder.stopped else "execution_failed"
                )
                if recorder.stopped:
                    break
            finally:
                publish(root / f"{case.case_id}-result.json", record)
        report["status"] = (
            "SYNTHETIC_PASS"
            if all(c["status"] == "COMPLETE" for c in report["cases"])
            else "PARTIAL"
        )
    except Exception:
        report["failure_category"] = "preparation_failed"
    finally:
        if recorder is not None:
            report["usage"] = await recorder.measurement()
        publish(root / "report.json", report)
        await engine.dispose()
    return report
