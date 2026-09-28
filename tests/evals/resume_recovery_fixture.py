"""D synthetic tenants and rehydratable business identities; no new database schema."""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

from sqlalchemy import select

from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import ResumeFeedback, ResumeTexArtifact, ResumeVersion, Run, RunEvent, RunJob
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.domain.resume_generation import SessionCreateV1
from app.domain.resume_revision import ContentFeedbackV1
from app.domain.tenancy import TenantContext, TenantService
from app.llm.ports import ChatModelResult, ModelUsage
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_quality_runtime import SOURCE, factory, seed, worker
from tests.unit.agents.test_resume_generation import _requirements, _selection


def script(state):
    if state["case"]["mode"] == "revision":
        values = [ChatModelResult(content=json.dumps({"patches": [state["patch"]]}))]
    else:
        values = [_requirements(), _selection(state["project_item_id"], state["fact_id"])]
    return [
        v.model_copy(update={"usage": ModelUsage(input_tokens=10, output_tokens=10)})
        for v in values
    ]


def restore(sessions, state):
    raw = SOURCE.read_bytes()
    template = dict(
        expected_source_sha256=hashlib.sha256(raw).hexdigest(),
        expected_preamble_sha256=hashlib.sha256(raw.split(b"\\begin{document}")[0]).hexdigest(),
    )
    return SimpleNamespace(
        sessions=sessions,
        tenant=TenantContext(
            UUID(state["workspace_id"]), UUID(state["actor_id"]), WorkspaceRole.OWNER
        ),
        store=SqlAlchemyResumeGenerationStore(sessions),
        revisions=SqlAlchemyResumeRevisionStore(sessions),
        reader=SqlAlchemyRunExecutionReader(sessions),
        artifacts=SqlAlchemyResumeArtifactStore(sessions, **template),
        recorder=SqlAlchemyInvocationRecorder(sessions),
    )


async def snapshot(sessions, state):
    workspace = UUID(state["workspace_id"])
    run_id = UUID(state["run_id"])
    async with sessions() as db:
        versions = list(
            await db.scalars(select(ResumeVersion).where(ResumeVersion.workspace_id == workspace))
        )
        artifacts = list(
            await db.scalars(
                select(ResumeTexArtifact).where(ResumeTexArtifact.workspace_id == workspace)
            )
        )
        events = list(
            await db.scalars(
                select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.type == "run.completed")
            )
        )
        feedback = list(
            await db.scalars(select(ResumeFeedback).where(ResumeFeedback.workspace_id == workspace))
        )
        run = await db.get(Run, run_id)
        job = await db.scalar(select(RunJob).where(RunJob.run_id == run_id))
        return {
            "versions": {str(v.id): quality_identity_digest(v.content_json) for v in versions},
            "artifacts": {str(a.id): hashlib.sha256(a.tex_bytes).hexdigest() for a in artifacts},
            "completed_events": len(events),
            "feedback": {
                str(f.id): str(f.target_version_id) if f.target_version_id else None
                for f in feedback
            },
            "run_status": run.status,
            "error_category": run.error_category,
            "job_status": job.status,
            "job_attempt": job.attempt,
        }


async def prepare(sessions, root, case):
    state_path = root / "state.json"
    if state_path.exists():
        state = read_private_json(state_path)
        require(state["case"] == case, "case_state_changed")
        return state
    require(not (root / "setup-started.json").exists(), "incomplete_setup_requires_review")
    identity = f"d-{root.parent.name}-{case['case_id']}-{uuid4().hex}"
    publish(root / "setup-started.json", {"identity": identity, "case": case})
    owner = await ProvisioningService(
        SqlAlchemyProvisioningStore(sessions)
    ).provision_personal_workspace(identity)
    tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
        workspace_id=owner.workspace_id, actor_user_id=owner.user_id
    )
    recorder = SqlAlchemyInvocationRecorder(sessions)
    dataset = SimpleNamespace(
        facts=(
            "Built a synthetic service"
            + ("", " with Python", " with a documented interface")[case["ordinal"] % 3],
        ),
        cases=(SimpleNamespace(jd=f"Build a synthetic service\nSynthetic case {case['ordinal']}"),),
    )
    rig = await seed(sessions, tenant, root, dataset, SOURCE.read_bytes(), recorder)
    key = uuid4()
    created = await rig.store.create(tenant, rig.request, key)
    state = {
        "case": case,
        "workspace_id": str(tenant.workspace_id),
        "actor_id": str(tenant.actor_user_id),
        "session_id": str(created.receipt.resource_id),
        "run_id": str(created.receipt.run_id),
        "request": rig.request.model_dump(mode="json"),
        "create_key": str(key),
        "project_item_id": str(rig.project_item_id),
        "fact_id": str(rig.fact_ids[0]),
    }
    # Initial revision draft is setup, not a measured fault attempt.
    if case["mode"] == "revision":
        generation_state = {**state, "case": {**case, "mode": "generation"}}
        require(
            await worker(rig, factory(rig, script(generation_state))).run_once(asyncio.Event()),
            "setup_worker_idle",
        )
        detail = await rig.store.get_session(tenant, UUID(state["session_id"]))
        version = await rig.store.get_version(
            tenant, UUID(state["session_id"]), detail["current_version_id"]
        )
        target = version["content"]["projects"][0]["bullet_ids"][0]
        state["patch"] = {
            "operation": "replace_text",
            "item_id": str(target),
            "field": "bullet",
            "text": dataset.facts[0] + ".",
            "fact_version_ids": [state["fact_id"]],
        }
        request = ContentFeedbackV1(
            expected_session_revision=detail["revision"],
            base_version_id=detail["current_version_id"],
            target_item_ids=(UUID(str(target)),),
            instruction="Clarify the existing synthetic service statement",
        )
        feedback_key = uuid4()
        accepted = await rig.revisions.command(
            tenant, UUID(state["session_id"]), request, feedback_key
        )
        state.update(
            run_id=str(accepted.receipt.run_id),
            feedback_id=str(accepted.receipt.resource_id),
            feedback_key=str(feedback_key),
            feedback_request=request.model_dump(mode="json"),
        )
    state["baseline"] = await snapshot(sessions, state)
    if case["ordinal"] % 3 == 0:
        if case["mode"] == "revision":
            require(
                await rig.revisions.reserve_repair(tenant, UUID(state["feedback_id"])),
                "repair_seed_failed",
            )
        else:
            require(
                await rig.store.reserve_repair(tenant, UUID(state["session_id"])),
                "repair_seed_failed",
            )
    publish(state_path, state)
    return state


async def replay(rig, state):
    if state["case"]["mode"] == "generation":
        result = await rig.store.create(
            rig.tenant,
            SessionCreateV1.model_validate_json(json.dumps(state["request"])),
            UUID(state["create_key"]),
        )
    else:
        result = await rig.revisions.command(
            rig.tenant,
            UUID(state["session_id"]),
            ContentFeedbackV1.model_validate_json(json.dumps(state["feedback_request"])),
            UUID(state["feedback_key"]),
        )
    require(
        result.replayed and str(result.receipt.run_id) == state["run_id"], "command_replay_changed"
    )
