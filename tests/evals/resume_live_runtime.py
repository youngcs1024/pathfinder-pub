"""Live-only composition of existing business commands, Factory and one worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from random import Random
from types import SimpleNamespace
from uuid import UUID, uuid5

from pydantic import SecretStr

from app.agents.material_facts import PROMPT_VERSION, MaterialFactExtractor
from app.db.documents import SqlAlchemyDocumentRepository
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.material import SqlAlchemyMaterialStore
from app.db.project_facts import SqlAlchemyProjectFactStore, extractor_identity
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_confirmation import SqlAlchemyResumeConfirmationStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_profiles import ClaimReviewV1, ItemReviewV1, SqlAlchemyResumeProfileStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService
from app.domain.resume_confirmation import ConfirmVersionV1
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.resume_revision import ContentFeedbackV1, LockChangeV1
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.invocations import LLMInvocationContext
from app.llm.qwen_adapters import create_qwen_adapters
from app.material.aliases import MaterialAlias, MaterialAliasRegistry
from app.material.reader import read_alias
from app.resume.template_render import TemplateIdentity
from app.retrieval.documents import DocumentIngestionService
from app.worker.backoff import ExponentialBackoff
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.material_executor import MaterialRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.evals.product_acceptance_budget import admit
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_pilot import credentials
from tests.evals.resume_live_baseline import one_shot_live
from tests.evals.resume_live_contracts import assessed_coverage, require_review, validate_rubric
from tests.evals.resume_quality_baseline import BaselineOutputError, common_input
from tests.evals.resume_quality_budget import ComparisonRecorder
from tests.evals.resume_quality_runtime import snapshot, usage_delta, worker


def save(path, value):
    publish(path, json.loads(json.dumps(value, default=str)))


def key(rig, name):
    return uuid5(rig.inputs.allocation_id, name)


def review(rig, name, kind):
    value = read_private_json(rig.root / name)
    require_review(value, binding=rig.inputs.digest, kind=kind)
    return value


async def materials(rig):
    aliases = MaterialAliasRegistry(
        tuple(
            MaterialAlias(p.alias, "file", rig.root / "inputs", p.files, (rig.tenant.workspace_id,))
            for p in rig.inputs.projects
        )
    )
    store = SqlAlchemyMaterialStore(rig.sessions, aliases)
    facts = SqlAlchemyProjectFactStore(rig.sessions)
    executor = MaterialRunExecutor(
        reader=rig.reader,
        source_reader=read_alias,
        materials=store,
        aliases=aliases,
        facts=facts,
        extractor_digest=extractor_identity(PROMPT_VERSION, f"qwen:{rig.inputs.chat_model}"),
        extractor_factory=lambda tenant, run_id, _scope: MaterialFactExtractor(
            model=rig.factory.create_chat_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            )
        ),
        ingestion_factory=lambda tenant, run_id: DocumentIngestionService(
            repository=SqlAlchemyDocumentRepository(rig.sessions),
            embedding=rig.factory.create_embedding_model(
                LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id, run_id=run_id)
            ),
        ),
    )
    runner = WorkerRunner(
        worker_id="r71-live-material",
        store=SqlAlchemyWorkerJobStore(
            rig.sessions, ExponentialBackoff(WorkerRuntimeSettings(), Random(1))
        ),
        tenant_service=TenantService(SqlAlchemyTenantResolver(rig.sessions)),
        executor=RunExecutorDispatcher(
            {
                v: executor
                for v in ("pathfinder-resume-v2", "pathfinder-resume-v3", "pathfinder-resume-v4")
            }
        ),
        settings=WorkerRuntimeSettings(),
        unsupported_work_guard=rig.reader.has_unsupported_pending_work,
    )
    projects = {}
    for p in rig.inputs.projects:
        created = await store.create_project(rig.tenant, p.title, key(rig, f"project-{p.alias}"))
        source = await store.create_source(
            rig.tenant, created["id"], p.alias, key(rig, f"source-{p.alias}")
        )
        imported = await store.submit_import(
            rig.tenant, created["id"], (source["id"],), key(rig, f"import-{p.alias}")
        )
        save(
            rig.root / f"{p.alias}-import.json",
            {"project_id": created["id"], "import_id": imported.receipt.resource_id},
        )
        require(await runner.run_once(asyncio.Event()), "material_worker_idle")
        catalog = await facts.current_facts(rig.tenant, created["id"])
        save(rig.root / f"{p.alias}-candidates.json", catalog)
        projects[p.alias] = {
            "project_id": str(created["id"]),
            "catalog_digest": quality_identity_digest(json.loads(json.dumps(catalog, default=str))),
        }
        require(catalog["complete"], "material_extraction_incomplete")
        require(not rig.recorder.stopped, "budget_stopped")
    return {"projects": projects}


async def facts_stage(rig):
    decisions = review(rig, "fact-review.json", "facts")
    imported = read_private_json(rig.root / "materials-done.json")
    store = SqlAlchemyProjectFactStore(rig.sessions)
    accepted = {}
    for alias, state in imported["projects"].items():
        frozen = read_private_json(rig.root / f"{alias}-candidates.json")
        chosen = decisions["projects"][alias]
        require(
            chosen["catalog_digest"] == state["catalog_digest"] == quality_identity_digest(frozen),
            "fact_review_stale",
        )
        require(
            set(chosen["decisions"]) == {f["id"] for f in frozen["facts"]}, "fact_review_incomplete"
        )
        for fact in frozen["facts"]:
            decision = chosen["decisions"][fact["id"]]
            require(
                decision["decision"] in {"confirm", "reject"} and bool(decision["rationale"]),
                "fact_review_invalid",
            )
            await store.command(
                rig.tenant,
                kind="material_fact_review",
                project_id=UUID(state["project_id"]),
                import_id=UUID(frozen["import_id"]),
                request_id=key(rig, f"review-{fact['id']}"),
                fact_id=UUID(fact["id"]),
                expected_version=fact["version"],
                decision=decision["decision"],
                attested=False,
            )
        current = await store.current_facts(rig.tenant, UUID(state["project_id"]))
        confirmed = [f["version_id"] for f in current["facts"] if f["review_status"] == "confirmed"]
        require(bool(confirmed), "no_confirmed_facts")
        accepted[alias] = {"project_id": state["project_id"], "fact_version_ids": confirmed}
        save(rig.root / f"{alias}-confirmed.json", current)
    return {"projects": accepted, "review_digest": quality_identity_digest(decisions)}


async def profile_stage(rig):
    approved = review(rig, "profile-review.json", "profile")
    source_hash = rig.identity.source_sha256
    require(approved["source_sha256"] == source_hash, "profile_review_stale")
    facts = read_private_json(rig.root / "facts-done.json")
    store = SqlAlchemyResumeProfileStore(
        rig.sessions,
        expected_source_sha256=source_hash,
        expected_preamble_sha256=rig.identity.preamble_sha256,
    )
    result = await store.import_source(
        rig.tenant, rig.source_bytes.decode(), key(rig, "profile-import")
    )
    profile_id = result.receipt.resource_id
    for item_id in approved["reviewed_item_ids"]:
        detail = await store.get_profile(rig.tenant, profile_id)
        await store.command(
            rig.tenant,
            kind="resume_profile_item_review",
            profile_id=profile_id,
            request_id=key(rig, f"profile-review-{item_id}"),
            payload=ItemReviewV1(expected_version=detail["version"], item_id=UUID(item_id)),
        )
    detail = await store.get_profile(rig.tenant, profile_id)
    # Mapping a project is explicit; original unsupported claims remain excluded.
    for claim in detail["claims"]:
        decision = approved["claims"].get(
            f"{claim['item_id']}:{claim['field']}", {"decision": "excluded"}
        )
        alias = decision.get("project_alias")
        linked = facts["projects"][alias] if alias is not None else None
        await store.command(
            rig.tenant,
            kind="resume_claim_review",
            profile_id=profile_id,
            request_id=key(rig, f"claim-{claim['id']}"),
            payload=ClaimReviewV1(
                claim_id=claim["id"],
                expected_review_version=claim["review_version"],
                decision=decision["decision"],
                project_id=UUID(linked["project_id"]) if linked else None,
                fact_version_ids=tuple(UUID(v) for v in linked["fact_version_ids"])
                if linked
                else (),
            ),
        )
    detail = await store.get_profile(rig.tenant, profile_id)
    save(rig.root / "profile.json", detail)
    return {
        "profile_id": profile_id,
        "version_id": detail["version_id"],
        "preference_version": detail["preference_version"],
    }


async def draft(rig, case):
    profile = read_private_json(rig.root / "profile-done.json")
    facts = read_private_json(rig.root / "facts-done.json")
    rubric = review(rig, f"{case.case_id}-rubric.json", "rubric")
    validate_rubric(
        rubric,
        case=case,
        confirmed_ids={f for p in facts["projects"].values() for f in p["fact_version_ids"]},
    )
    save(rig.root / f"{case.case_id}-frozen-rubric.json", rubric)
    request = SessionCreateV1(
        profile_version_id=UUID(profile["version_id"]),
        preference_version=profile["preference_version"],
        project_ids=tuple(UUID(p["project_id"]) for p in facts["projects"].values()),
        job=JobInputV1(source="paste", text=case.jd),
        budget=GenerationBudgetV1(max_model_calls=12, max_tool_calls=0, max_cost_cny=Decimal("20")),
    )
    created = await rig.store.create(rig.tenant, request, key(rig, f"{case.case_id}-session"))
    sid = created.receipt.resource_id
    save(rig.root / f"{case.case_id}-session.json", {"session_id": sid})
    inputs = await rig.store.execution_inputs(rig.tenant, sid)
    shared = common_input(inputs, rig.identity.source_sha256)
    save(rig.root / f"{case.case_id}-input.json", shared)
    save(
        rig.root / f"{case.case_id}-b0.json",
        {
            "tex": rig.source_bytes.decode(),
            "tex_sha256": rig.identity.source_sha256,
            "common_input_digest": quality_identity_digest(shared),
            "finalization_cost": "NOT_MEASURED",
        },
    )
    before = await rig.recorder.measurement()
    # Create-only barrier precedes any provider call. B1 failure never triggers repair.
    save(
        rig.root / f"{case.case_id}-b1-started.json",
        {"common_input_digest": quality_identity_digest(shared)},
    )
    try:
        baseline = await one_shot_live(
            rig.factory.create_chat_model(
                LLMInvocationContext(rig.tenant.workspace_id, rig.tenant.actor_user_id)
            ),
            inputs,
            rig.source_bytes,
            rig.identity,
        )
        save(rig.root / f"{case.case_id}-b1.json", baseline)
        b1_status = "GENERATED"
    except BaselineOutputError as error:
        save(rig.root / f"{case.case_id}-b1-failed.json", error.private_output)
        b1_status = "PARSE_FAILED"
    b1_usage = usage_delta(before, await rig.recorder.measurement())
    admit(await rig.recorder.usage(), rig.inputs.budget)
    before = await rig.recorder.measurement()
    require(await worker(rig, rig.factory).run_once(asyncio.Event()), "worker_idle")
    # Persist status even when the business run cannot publish a version.
    save(
        rig.root / f"{case.case_id}-draft-session.json",
        await rig.store.get_session(rig.tenant, sid),
    )
    initial = await snapshot(rig, sid)
    save(rig.root / f"{case.case_id}-system-0.json", initial)
    return {
        "session_id": sid,
        "b1_status": b1_status,
        "b1_usage": b1_usage,
        "b1_finalization_cost": "NOT_MEASURED",
        "initial_usage": usage_delta(before, await rig.recorder.measurement()),
        "common_input_digest": quality_identity_digest(shared),
    }


async def revise(rig, case, ordinal):
    name = f"{case.case_id}-round{ordinal}"
    approved = review(rig, f"{name}-review.json", "revision")
    sid = UUID(read_private_json(rig.root / f"{case.case_id}-draft-done.json")["session_id"])
    previous = read_private_json(rig.root / f"{case.case_id}-system-{ordinal - 1}.json")
    require(
        approved["base_version_id"] == previous["version_id"]
        and approved["tex_sha256"] == previous["tex_sha256"],
        "revision_review_stale",
    )
    require(
        bool(approved["initial_quality"]) if ordinal == 1 else bool(approved["compile_review"]),
        "quality_review_missing",
    )
    if ordinal == 1:
        rubric = read_private_json(rig.root / f"{case.case_id}-frozen-rubric.json")
        for arm in ("b0", "b1", "system"):
            assessed_coverage(rubric["requirements"], approved["initial_quality"][arm])
    detail = await rig.store.get_session(rig.tenant, sid)
    require(str(detail["current_version_id"]) == previous["version_id"], "revision_base_changed")
    if ordinal == 2:
        lock_id = UUID(approved["lock_item_id"])
        await rig.revisions.command(
            rig.tenant,
            sid,
            LockChangeV1(
                expected_session_revision=detail["revision"],
                base_version_id=detail["current_version_id"],
                item_id=lock_id,
                locked=True,
            ),
            key(rig, name + "-lock"),
        )
        detail = await rig.store.get_session(rig.tenant, sid)
    before = await rig.recorder.measurement()
    feedback = ContentFeedbackV1(
        expected_session_revision=detail["revision"],
        base_version_id=detail["current_version_id"],
        target_item_ids=tuple(UUID(v) for v in approved["target_item_ids"]),
        instruction=approved["instruction"],
    )
    await rig.revisions.command(rig.tenant, sid, feedback, key(rig, name))
    require(await worker(rig, rig.factory).run_once(asyncio.Event()), "worker_idle")
    save(rig.root / f"{name}-session.json", await rig.store.get_session(rig.tenant, sid))
    revised = await snapshot(rig, sid)
    require(revised["version_id"] != previous["version_id"], "revision_not_published")
    save(rig.root / f"{case.case_id}-system-{ordinal}.json", revised)
    return {
        "version_id": revised["version_id"],
        "revision_usage": usage_delta(before, await rig.recorder.measurement()),
        "review_digest": quality_identity_digest(approved),
        "input_changes": approved.get("input_changes", []),
    }


async def confirm(rig, case):
    approved = review(rig, f"{case.case_id}-final-review.json", "final")
    final = read_private_json(rig.root / f"{case.case_id}-system-2.json")
    rubric = read_private_json(rig.root / f"{case.case_id}-frozen-rubric.json")
    final_coverage = assessed_coverage(rubric["requirements"], approved["final_quality"])
    require(
        approved["version_id"] == final["version_id"]
        and approved["tex_sha256"] == final["tex_sha256"],
        "final_review_stale",
    )
    require(
        all(
            approved.get(k) is True
            for k in ("facts_pass", "layout_pass", "chinese_pass", "no_overlap", "no_truncation")
        ),
        "final_review_failed",
    )
    require(
        approved["pages"] == 1
        and approved["shell_escape"] is False
        and approved["compiler"] == "XeLaTeX 2023",
        "compile_review_failed",
    )
    for name in ("pdf", "compile_log", "page_image"):
        artifact = rig.root / approved[name]["file"]
        require(
            artifact.resolve().is_relative_to(rig.root.resolve()) and not artifact.is_symlink(),
            "external_artifact_path",
        )
        require(
            hashlib.sha256(artifact.read_bytes()).hexdigest() == approved[name]["sha256"],
            "external_artifact_changed",
        )
    sid = UUID(read_private_json(rig.root / f"{case.case_id}-draft-done.json")["session_id"])
    detail = await rig.store.get_session(rig.tenant, sid)
    store = SqlAlchemyResumeConfirmationStore(rig.sessions, rig.artifacts)
    request = ConfirmVersionV1(
        version_id=UUID(final["version_id"]),
        expected_session_revision=detail["revision"],
        expected_current_version_id=detail["current_version_id"],
        artifact_id=UUID(final["artifact_id"]),
        tex_sha256=final["tex_sha256"],
        attested=True,
    )
    await store.confirm(
        rig.tenant, sid, request.version_id, request, key(rig, f"{case.case_id}-confirm")
    )
    for ordinal in range(3):
        prior = read_private_json(rig.root / f"{case.case_id}-system-{ordinal}.json")
        data = await store.download(rig.tenant, sid, UUID(prior["version_id"]))
        require(
            data[0] == prior["tex"].encode() and data[1] == prior["tex_sha256"], "history_changed"
        )
    first = await store.download(rig.tenant, sid, request.version_id)
    second = await store.download(rig.tenant, sid, request.version_id)
    require(first == second and first[3], "unstable_download")
    return {
        "status": "AGENT_ACCEPTED",
        "version_id": final["version_id"],
        "tex_sha256": final["tex_sha256"],
        "review_digest": quality_identity_digest(approved),
        "human_review": "NOT_RUN",
        "stable_download": True,
        "final_coverage": final_coverage,
    }


async def run_stage(url, root, inputs, stage, credentials_path, *, source_check):
    engine = create_database_engine(SecretStr(url))
    sessions = create_session_factory(engine)
    bundle = recorder = None
    try:
        identity = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace(f"r71-live-{inputs.allocation_id}")
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=identity.workspace_id, actor_user_id=identity.user_id
        )
        recorder = ComparisonRecorder(
            sessions, tenant, provider="qwen", budget=inputs.budget, source_check=source_check
        )
        # Read-only/fact-review stages can still preserve reports after budget exhaustion.
        values = credentials(credentials_path)
        bundle = create_qwen_adapters(
            api_key=values["DASHSCOPE_API_KEY"], workspace_id=values["PF_QWEN_WORKSPACE_ID"]
        )
        source = (root / "inputs" / inputs.resume_file).read_bytes()
        template = TemplateIdentity(
            source_sha256=hashlib.sha256(source).hexdigest(),
            preamble_sha256=hashlib.sha256(source.split(b"\\begin{document}", 1)[0]).hexdigest(),
        )
        rig = SimpleNamespace(
            root=root,
            inputs=inputs,
            sessions=sessions,
            tenant=tenant,
            recorder=recorder,
            factory=LLMFactory(
                recorder=recorder,
                chat_adapter=bundle.chat,
                embedding_adapter=bundle.embedding,
                provider="qwen",
            ),
            reader=SqlAlchemyRunExecutionReader(sessions),
            store=SqlAlchemyResumeGenerationStore(sessions),
            revisions=SqlAlchemyResumeRevisionStore(sessions),
            source_bytes=source,
            identity=template,
            artifacts=SqlAlchemyResumeArtifactStore(
                sessions,
                expected_source_sha256=template.source_sha256,
                expected_preamble_sha256=template.preamble_sha256,
            ),
        )
        if stage == "materials":
            result = await materials(rig)
        elif stage == "facts":
            result = await facts_stage(rig)
        elif stage == "profile":
            result = await profile_stage(rig)
        elif stage == "report":
            cases = {
                c.case_id: read_private_json(root / f"{c.case_id}-confirm-done.json")
                for c in inputs.cases
            }
            comparisons = {}
            for case in inputs.cases:
                name = case.case_id
                rubric = read_private_json(root / f"{name}-frozen-rubric.json")
                initial = read_private_json(root / f"{name}-round1-review.json")["initial_quality"]
                comparisons[name] = {
                    "initial_quality": initial,
                    "initial_coverage": {
                        arm: assessed_coverage(rubric["requirements"], value)
                        for arm, value in initial.items()
                    },
                    "draft": read_private_json(root / f"{name}-draft-done.json"),
                    "revision_rounds": [
                        read_private_json(root / f"{name}-round{i}-done.json") for i in (1, 2)
                    ],
                    "final_quality": read_private_json(root / f"{name}-final-review.json"),
                    "b0_b1_finalization_cost": "NOT_MEASURED",
                    "human_minutes": None,
                }
            result = {
                "status": "AGENT_ACCEPTED",
                "r71_status": "ACCEPTED",
                "human_review": "NOT_RUN",
                "cases": cases,
                "comparisons": comparisons,
                "usage": await recorder.measurement(),
            }
            save(root / "report.json", result)
        else:
            case_id, operation = stage.rsplit("-", 1)
            case = next(c for c in inputs.cases if c.case_id == case_id)
            if operation == "draft":
                result = await draft(rig, case)
            elif operation in {"round1", "round2"}:
                result = await revise(rig, case, int(operation[-1]))
            else:
                result = await confirm(rig, case)
        return json.loads(
            json.dumps({**result, "usage": await recorder.measurement()}, default=str)
        )
    finally:
        if recorder is not None:
            save(root / f"{stage}-ledger.json", await recorder.measurement())
        if bundle is not None:
            await bundle.aclose()
        await engine.dispose()
