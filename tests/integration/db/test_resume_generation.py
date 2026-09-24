"""R4.1 session acceptance, replay, access and cancellation on PostgreSQL."""

from __future__ import annotations

import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from random import Random
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update

from app.api.schemas.resume_generation import (
    SessionDetailResponse,
    SessionListItemResponse,
    VersionDetailResponse,
)
from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import (
    MaterialSnapshotFile,
    ResumeSession,
    ResumeTexArtifact,
    ResumeVersion,
    RunJob,
    WorkspaceMembership,
)
from app.db.project_facts import SqlAlchemyProjectFactStore
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_confirmation import SqlAlchemyResumeConfirmationStore
from app.db.resume_generation import ResumeGenerationPublisher, SqlAlchemyResumeGenerationStore
from app.db.resume_profiles import ClaimReviewV1, ItemReviewV1, SqlAlchemyResumeProfileStore
from app.db.resume_revision import ResumeRevisionPublisher, SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainConflictError, DomainInvariantError, DomainNotFoundError
from app.domain.job_inputs import ProvidedJobInputAdapter
from app.domain.project_facts import CandidateFactV1, FactEvidenceV1
from app.domain.provisioning import ProvisioningService
from app.domain.resume_confirmation import ConfirmVersionV1, VersionSummaryV1
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.resume_profile import JobPreferenceOverrideV1
from app.domain.resume_revision import (
    ContentFeedbackV1,
    FactFeedbackV1,
    FactReviewV1,
    LockChangeV1,
    PatchV1,
    UserFactInputV1,
)
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatModelResult
from app.material.aliases import MaterialAliasRegistry
from app.worker.backoff import ExponentialBackoff
from app.worker.dispatcher import RunExecutorDispatcher
from app.worker.generation_executor import GenerationRunExecutor
from app.worker.revision_executor import RevisionRunExecutor
from app.worker.runner import WorkerRunner
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.db.test_material_snapshots import _fixture_aliases, _runner
from tests.resume_extensions import alternate_kwargs

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


async def test_create_replay_scope_and_cancel(migrated_database_url: str) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
        tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
        owner = await provisioning.provision_personal_workspace("r41-session-owner")
        foreign = await provisioning.provision_personal_workspace("r41-session-foreign")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        outsider = await tenancy.resolve_tenant(
            workspace_id=foreign.workspace_id, actor_user_id=foreign.user_id
        )
        source = SOURCE.read_text()
        profiles = SqlAlchemyResumeProfileStore(
            sessions,
            expected_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            expected_preamble_sha256=hashlib.sha256(
                source.split("\\begin{document}", 1)[0].encode()
            ).hexdigest(),
        )
        accepted = await profiles.import_source(tenant, source, uuid4())
        profile = await profiles.get_profile(tenant, accepted.receipt.resource_id)
        project = await SqlAlchemyMaterialStore(sessions, MaterialAliasRegistry(())).create_project(
            tenant, "Synthetic", uuid4()
        )
        adapter_calls = []

        class RecordingInput(ProvidedJobInputAdapter):
            def snapshot(self, value):
                adapter_calls.append(value.digest)
                return super().snapshot(value)

        store = SqlAlchemyResumeGenerationStore(sessions, job_input_adapter=RecordingInput())
        request = SessionCreateV1(
            profile_version_id=profile["version_id"],
            preference_version=1,
            project_ids=(project["id"],),
            job=JobInputV1(source="paste", text="Build a synthetic service"),
            budget=GenerationBudgetV1(
                max_model_calls=3, max_tool_calls=0, max_cost_cny=Decimal("1")
            ),
        )
        key = uuid4()
        created = await store.create(tenant, request, key)
        replay = await store.create(tenant, request, key)
        assert replay.replayed and replay.receipt == created.receipt
        assert adapter_calls == [request.job.digest]
        assert created.receipt.resource_id is not None
        detail = await store.get_session(tenant, created.receipt.resource_id)
        SessionDetailResponse.model_validate(detail)
        assert detail["job"]["text"] == request.job.text
        assert detail["project_ids"] == [project["id"]]
        listing = await store.list_sessions(tenant)
        assert len(listing) == 1
        SessionListItemResponse.model_validate(listing[0])
        assert listing[0]["session_id"] == created.receipt.resource_id
        assert await store.list_sessions(outsider) == []
        assert detail["run_status"] == "queued"
        assert detail["result"] is None
        inputs = await store.execution_inputs(tenant, created.receipt.resource_id)
        assert inputs.job_text == request.job.text
        assert inputs.facts == ()
        async with sessions() as db:
            assert await db.scalar(select(func.count()).select_from(ResumeSession)) == 1
            assert await db.scalar(select(func.count()).select_from(RunJob)) == 1
        with pytest.raises(DomainConflictError):
            await store.create(
                tenant,
                request.model_copy(update={"job": JobInputV1(source="paste", text="Different")}),
                key,
            )
        with pytest.raises(DomainNotFoundError):
            await store.get_session(outsider, created.receipt.resource_id)
        cancelled = await store.cancel(tenant, created.receipt.resource_id)
        assert cancelled.status.value == "cancelled"
        assert (await store.get_session(tenant, created.receipt.resource_id))[
            "run_status"
        ] == "cancelled"
        async with sessions.begin() as db:
            await db.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=func.now())
            )
        with pytest.raises(DomainNotFoundError):
            await store.list_sessions(tenant)
        with pytest.raises(DomainNotFoundError):
            await store.get_session(tenant, created.receipt.resource_id)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("alternate_template", [False, True], ids=["fixed", "alternate"])
async def test_scripted_fake_publishes_downloadable_tex(
    migrated_database_url: str, tmp_path: Path, alternate_template: bool
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
        tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
        owner = await provisioning.provision_personal_workspace("r41-draft-owner")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        aliases, _ = _fixture_aliases(tmp_path, tenant.workspace_id)
        materials = SqlAlchemyMaterialStore(sessions, aliases)
        project = await materials.create_project(tenant, "Synthetic", uuid4())
        source = await materials.create_source(tenant, project["id"], "code", uuid4())
        imported = await materials.submit_import(tenant, project["id"], (source["id"],), uuid4())
        assert await _runner(sessions, materials).run_once(asyncio.Event())
        facts = SqlAlchemyProjectFactStore(sessions)
        catalog = await facts.current_facts(tenant, project["id"])
        async with sessions() as db:
            snapshot_file_id = await db.scalar(
                select(MaterialSnapshotFile.id).where(
                    MaterialSnapshotFile.workspace_id == tenant.workspace_id,
                    MaterialSnapshotFile.path == "main.py",
                )
            )
        assert snapshot_file_id is not None
        added = await facts.command(
            tenant,
            kind="material_fact_add",
            project_id=project["id"],
            import_id=imported.receipt.resource_id,
            request_id=uuid4(),
            candidate=CandidateFactV1(
                claim="Built a synthetic service",
                kind="personal_statement",
                evidence=(
                    FactEvidenceV1(
                        snapshot_file_id=snapshot_file_id,
                        start_line=1,
                        end_line=1,
                        quote="print('synthetic')",
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
        reviewed_fact = next(
            item for item in confirmed["facts"] if item["id"] == added.receipt.resource_id
        )
        fact_version_id = reviewed_fact["version_id"]
        source_text = SOURCE.read_text()
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
                fact_version_ids=(fact_version_id,),
            ),
        )
        store = SqlAlchemyResumeGenerationStore(sessions)
        created = await store.create(
            tenant,
            SessionCreateV1(
                profile_version_id=profile["version_id"],
                preference_version=1,
                project_ids=(project["id"],),
                job=JobInputV1(source="paste", text="Build a synthetic service"),
                budget=GenerationBudgetV1(
                    max_model_calls=3, max_tool_calls=0, max_cost_cny=Decimal("1")
                ),
            ),
            uuid4(),
        )
        assert (await store.execution_inputs(tenant, created.receipt.resource_id)).facts
        # A test-only batch is just independent handles over the single-job service.
        child_requests = {}
        children = {}
        for name, calls in (("sibling", 3), ("budget", 1), ("cancelled", 3)):
            child_requests[name] = SessionCreateV1(
                profile_version_id=profile["version_id"],
                preference_version=1,
                project_ids=(project["id"],),
                job=JobInputV1(source="paste", text="Build a synthetic service\n" + name),
                override=JobPreferenceOverrideV1(page_target=2, writing_advice=name),
                budget=GenerationBudgetV1(
                    max_model_calls=calls, max_tool_calls=0, max_cost_cny=Decimal("0.5")
                ),
            )
            child_key = uuid4()
            child = await store.create(tenant, child_requests[name], child_key)
            assert (await store.create(tenant, child_requests[name], child_key)).replayed
            children[name] = child.receipt.resource_id
        await store.cancel(tenant, children["cancelled"])
        script = ScriptedFakeChatModel(
            [
                ChatModelResult(
                    content=json.dumps(
                        {
                            "requirements": [
                                {
                                    "kind": "explicit",
                                    "start": 0,
                                    "end": 25,
                                    "quote": "Build a synthetic service",
                                }
                            ],
                            "questions": [],
                        }
                    )
                ),
                ChatModelResult(
                    content=json.dumps(
                        {
                            "bullets": [
                                {
                                    "project_item_id": str(project_item_id),
                                    "fact_version_id": str(fact_version_id),
                                    "requirement_ordinals": [0],
                                }
                            ],
                            "omitted_fact_version_ids": [],
                            "questions": [],
                        }
                    )
                ),
            ]
            * 2
            + [
                ChatModelResult(
                    content=json.dumps(
                        {
                            "requirements": [
                                {
                                    "kind": "explicit",
                                    "start": 0,
                                    "end": 25,
                                    "quote": "Build a synthetic service",
                                }
                            ],
                            "questions": [],
                        }
                    )
                )
            ]
        )
        factory = LLMFactory(
            recorder=SqlAlchemyInvocationRecorder(sessions),
            chat_adapter=script,
            embedding_adapter=FakeEmbeddingModel(),
            provider="fake",
        )
        reader = SqlAlchemyRunExecutionReader(sessions)
        executor = GenerationRunExecutor(
            reader=reader,
            sessions=store,
            model_factory=lambda actor, run_id: factory.create_chat_model(
                LLMInvocationContext(actor.workspace_id, actor.actor_user_id, run_id=run_id)
            ),
            tools_factory=lambda _actor, _run_id, _scope: None,
        )
        artifacts = SqlAlchemyResumeArtifactStore(
            sessions,
            expected_source_sha256=source_hash,
            expected_preamble_sha256=preamble_hash,
            **(alternate_kwargs() if alternate_template else {}),
        )
        runner = WorkerRunner(
            worker_id="generation-test",
            store=SqlAlchemyWorkerJobStore(
                sessions,
                ExponentialBackoff(WorkerRuntimeSettings(), Random(1)),
                generation_publisher=ResumeGenerationPublisher(artifacts),
            ),
            tenant_service=tenancy,
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
        assert await runner.run_once(asyncio.Event())
        detail = await store.get_session(tenant, created.receipt.resource_id)
        assert detail["run_status"] == "completed", detail["error_category"]
        assert detail["result"].outcome == "draft"
        assert await runner.run_once(asyncio.Event())
        sibling = await store.get_session(tenant, children["sibling"])
        assert sibling["run_status"] == "completed"
        assert sibling["current_version_id"] != detail["current_version_id"]
        assert await runner.run_once(asyncio.Event())
        states = {
            name: await store.get_session(tenant, handle) for name, handle in children.items()
        }
        assert {name: state["run_status"] for name, state in states.items()} == {
            "sibling": "completed",
            "budget": "failed",
            "cancelled": "cancelled",
        }
        assert states["budget"]["current_version_id"] is None
        assert states["budget"]["error_category"] == "generation_budget_exhausted"
        assert not await store.spend_allowed(tenant, children["budget"])
        assert await store.spend_allowed(tenant, children["sibling"])
        assert (await store.get_session(tenant, created.receipt.resource_id))[
            "current_version_id"
        ] == detail["current_version_id"]
        for name in children:
            assert states[name]["job"]["text"] == child_requests[name].job.text
            assert states[name]["budget"] == child_requests[name].budget
        with pytest.raises(DomainNotFoundError):
            await store.get_version(tenant, children["sibling"], detail["current_version_id"])
        version = await store.get_version(
            tenant, created.receipt.resource_id, detail["current_version_id"]
        )
        VersionDetailResponse.model_validate(version)
        assert version["coverage"][0]["fact_version_ids"] == [fact_version_id]
        fixed_fact = next(
            item for item in version["facts"] if item["version_id"] == fact_version_id
        )
        assert fixed_fact["evidence"][0]["path"] == "main.py"
        await facts.command(
            tenant,
            kind="material_fact_revise",
            project_id=project["id"],
            fact_id=added.receipt.resource_id,
            import_id=confirmed["import_id"],
            expected_version=reviewed_fact["version"],
            candidate=CandidateFactV1(claim="New claim after draft", kind="personal_statement"),
            request_id=uuid4(),
        )
        historic = await store.get_version(
            tenant, created.receipt.resource_id, detail["current_version_id"]
        )
        assert (
            next(item for item in historic["facts"] if item["version_id"] == fact_version_id)[
                "claim"
            ]
            == "Built a synthetic service"
        )
        assert version["validation"]["retrieval_config_version"].startswith("sha256:")
        payload, digest = await artifacts.get_bytes(tenant, version["artifact_id"])
        assert b"Built a synthetic service" in payload
        assert b"canary@example.test" in payload
        confirmations = SqlAlchemyResumeConfirmationStore(sessions, artifacts)
        session_id = created.receipt.resource_id
        history = await confirmations.list_versions(tenant, session_id)
        assert len(history) == 1
        VersionSummaryV1.model_validate(history[0])
        assert history[0]["confirmation"] is None
        first_request = ConfirmVersionV1(
            version_id=version["version_id"],
            expected_session_revision=detail["revision"],
            expected_current_version_id=detail["current_version_id"],
            artifact_id=version["artifact_id"],
            tex_sha256=digest,
            attested=True,
        )
        with pytest.raises(DomainConflictError):
            await confirmations.confirm(
                tenant,
                session_id,
                version["version_id"],
                first_request.model_copy(update={"attested": False}),
                uuid4(),
            )
        with pytest.raises(DomainConflictError):
            await confirmations.confirm(
                tenant,
                session_id,
                version["version_id"],
                first_request.model_copy(update={"tex_sha256": "0" * 64}),
                uuid4(),
            )
        first_key = uuid4()
        first_confirmation = await confirmations.confirm(
            tenant, session_id, version["version_id"], first_request, first_key
        )
        assert (
            await confirmations.confirm(
                tenant, session_id, version["version_id"], first_request, first_key
            )
        )["replayed"]
        assert first_confirmation["tex_sha256"] == digest
        assert (await confirmations.download(tenant, session_id, version["version_id"])) == (
            payload,
            digest,
            1,
            True,
        )
        detail = await store.get_session(tenant, session_id)
        revisions = SqlAlchemyResumeRevisionStore(sessions)
        bullet_id = UUID(version["content"]["projects"][0]["bullet_ids"][0])
        base_id = detail["current_version_id"]
        patch = PatchV1(
            operation="replace_text",
            item_id=bullet_id,
            field="bullet",
            text="Built a synthetic service.",
            fact_version_ids=(fact_version_id,),
        )
        feedback = ContentFeedbackV1(
            expected_session_revision=detail["revision"],
            base_version_id=base_id,
            target_item_ids=(bullet_id,),
            patches=(patch,),
        )
        key = uuid4()
        accepted = await revisions.command(tenant, created.receipt.resource_id, feedback, key)
        assert (
            await revisions.command(tenant, created.receipt.resource_id, feedback, key)
        ).replayed
        with pytest.raises(DomainConflictError):
            await revisions.command(tenant, created.receipt.resource_id, feedback, uuid4())
        fixed_revision_inputs = await revisions.revision_inputs(
            tenant, accepted.receipt.resource_id
        )
        assert fact_version_id in fixed_revision_inputs.permitted_fact_ids
        revision_executor = RevisionRunExecutor(
            reader=reader,
            revisions=revisions,
            model_factory=lambda actor, run_id: factory.create_chat_model(
                LLMInvocationContext(actor.workspace_id, actor.actor_user_id, run_id=run_id)
            ),
        )
        revision_runner = WorkerRunner(
            worker_id="revision-test",
            store=SqlAlchemyWorkerJobStore(
                sessions,
                ExponentialBackoff(WorkerRuntimeSettings(), Random(2)),
                revision_publisher=ResumeRevisionPublisher(artifacts),
            ),
            tenant_service=tenancy,
            executor=RunExecutorDispatcher(
                {
                    "pathfinder-resume-v2": executor,
                    "pathfinder-resume-v3": executor,
                    "pathfinder-resume-v4": revision_executor,
                }
            ),
            settings=WorkerRuntimeSettings(),
            unsupported_work_guard=reader.has_unsupported_pending_work,
        )
        assert await revision_runner.run_once(asyncio.Event())
        revised_detail = await store.get_session(tenant, created.receipt.resource_id)
        assert revised_detail["run_id"] == accepted.receipt.run_id
        assert revised_detail["run_status"] == "completed", revised_detail["error_category"]
        assert revised_detail["current_version_id"] != base_id
        revised = await store.get_version(
            tenant, created.receipt.resource_id, revised_detail["current_version_id"]
        )
        assert revised["parent_version_id"] == base_id
        assert revised["content"]["projects"][0]["bullets"][0]["text"] == patch.text
        assert revised["diff"][0]["item_id"] == str(bullet_id)
        history = await confirmations.list_versions(tenant, session_id)
        assert [item["version"] for item in history] == [2, 1]
        assert history[0]["confirmation"] is None
        assert history[1]["confirmation"].confirmation_id == first_confirmation["confirmation_id"]
        assert (await confirmations.download(tenant, session_id, base_id)) == (
            payload,
            digest,
            1,
            True,
        )
        with pytest.raises(DomainConflictError):
            await confirmations.confirm(
                tenant, session_id, revised["version_id"], first_request, uuid4()
            )
        assert (await store.get_version(tenant, created.receipt.resource_id, base_id))[
            "content"
        ] == version["content"]
        lock = await revisions.command(
            tenant,
            created.receipt.resource_id,
            LockChangeV1(
                expected_session_revision=revised_detail["revision"],
                base_version_id=revised_detail["current_version_id"],
                item_id=bullet_id,
                locked=True,
            ),
            uuid4(),
        )
        assert lock.receipt.status == "completed"
        locked_detail = await store.get_session(tenant, created.receipt.resource_id)
        assert str(bullet_id) in locked_detail["locked_item_ids"]
        fact_submission = await revisions.command(
            tenant,
            created.receipt.resource_id,
            FactFeedbackV1(
                expected_session_revision=locked_detail["revision"],
                base_version_id=locked_detail["current_version_id"],
                fact=UserFactInputV1(
                    project_id=project["id"],
                    scope="session",
                    claim="Reduced synthetic latency by 12%",
                    kind="implementation",
                    environment="synthetic benchmark",
                    fact_scope="API requests",
                    metric_basis="p95 before and after",
                ),
            ),
            uuid4(),
        )
        assert fact_submission.receipt.status == "completed"
        pending = (await revisions.list_user_facts(tenant, created.receipt.resource_id))[-1]
        assert pending["review_status"] == "pending"
        fact_detail = await store.get_session(tenant, created.receipt.resource_id)
        correction = await revisions.command(
            tenant,
            session_id,
            FactFeedbackV1(
                expected_session_revision=fact_detail["revision"],
                base_version_id=fact_detail["current_version_id"],
                fact=UserFactInputV1(
                    project_id=project["id"],
                    scope="session",
                    claim="Corrected synthetic service claim",
                    kind="personal_statement",
                    supersedes_material_version_id=fact_version_id,
                ),
            ),
            uuid4(),
        )
        assert correction.receipt.status == "completed"
        fact_detail = await store.get_session(tenant, session_id)
        second_request = ConfirmVersionV1(
            version_id=revised["version_id"],
            expected_session_revision=fact_detail["revision"],
            expected_current_version_id=fact_detail["current_version_id"],
            artifact_id=revised["artifact_id"],
            tex_sha256=(await artifacts.get_info(tenant, revised["artifact_id"])).tex_sha256,
            attested=True,
        )
        with pytest.raises(DomainConflictError, match="under correction"):
            await confirmations.confirm(
                tenant, session_id, revised["version_id"], second_request, uuid4()
            )
        correction_version_id = correction.receipt.resource_id
        # The feedback receipt identifies the feedback command; its normalized result
        # identifies the pending user fact version that must be reviewed.
        correction_feedback = next(
            item
            for item in await revisions.list_feedback(tenant, session_id)
            if item["feedback_id"] == correction_version_id
        )
        corrected_fact_version_id = UUID(correction_feedback["normalized"]["fact_version_id"])
        rejected = await revisions.command(
            tenant,
            session_id,
            FactReviewV1(
                expected_session_revision=fact_detail["revision"],
                base_version_id=fact_detail["current_version_id"],
                fact_version_id=corrected_fact_version_id,
                decision="reject",
            ),
            uuid4(),
        )
        assert rejected.receipt.status == "completed"
        fact_detail = await store.get_session(tenant, session_id)
        second_request = second_request.model_copy(
            update={"expected_session_revision": fact_detail["revision"]}
        )
        async with sessions.begin() as db:
            validation = dict(revised["validation"])
            await db.execute(
                update(ResumeVersion)
                .where(ResumeVersion.id == revised["version_id"])
                .values(validation_json={**validation, "questions": ["unresolved fact question"]})
            )
        with pytest.raises(DomainConflictError):
            await confirmations.confirm(
                tenant, session_id, revised["version_id"], second_request, uuid4()
            )
        async with sessions.begin() as db:
            await db.execute(
                update(ResumeVersion)
                .where(ResumeVersion.id == revised["version_id"])
                .values(validation_json=validation)
            )
        second_confirmation = await confirmations.confirm(
            tenant, session_id, revised["version_id"], second_request, uuid4()
        )
        assert second_confirmation["version_id"] == revised["version_id"]
        assert (await confirmations.download(tenant, session_id, revised["version_id"]))[3]
        fact_detail = await store.get_session(tenant, created.receipt.resource_id)
        reviewed = await revisions.command(
            tenant,
            created.receipt.resource_id,
            FactReviewV1(
                expected_session_revision=fact_detail["revision"],
                base_version_id=fact_detail["current_version_id"],
                fact_version_id=pending["fact_version_id"],
                decision="confirm",
                attested=True,
            ),
            uuid4(),
        )
        assert reviewed.receipt.status == "completed"
        assert (await revisions.list_user_facts(tenant, created.receipt.resource_id))[-1][
            "review_status"
        ] == "confirmed"
        sibling_history = await confirmations.list_versions(tenant, children["sibling"])
        assert len(sibling_history) == 1 and sibling_history[0]["confirmation"] is None
        assert await revisions.list_feedback(tenant, children["sibling"]) == []
        assert await revisions.list_user_facts(tenant, children["sibling"]) == []
        sibling_after = await store.get_session(tenant, children["sibling"])
        assert sibling_after["current_version_id"] == sibling["current_version_id"]
        assert sibling_after["locked_item_ids"] == []
        sibling_inputs = await store.execution_inputs(tenant, children["sibling"])
        assert sibling_inputs.preferences.page_target == 2
        assert sibling_inputs.preferences.writing_advice == "sibling"
        with pytest.raises(DomainNotFoundError):
            await confirmations.download(tenant, children["sibling"], base_id)
        with pytest.raises(DomainNotFoundError):
            await confirmations.confirm(
                tenant, children["sibling"], base_id, first_request, uuid4()
            )
        foreign = await provisioning.provision_personal_workspace("r52-foreign")
        outsider = await tenancy.resolve_tenant(
            workspace_id=foreign.workspace_id, actor_user_id=foreign.user_id
        )
        with pytest.raises(DomainNotFoundError):
            await confirmations.list_versions(outsider, session_id)
        with pytest.raises(DomainNotFoundError):
            await confirmations.download(outsider, session_id, base_id)
        with pytest.raises(DomainNotFoundError):
            await confirmations.confirm(outsider, session_id, base_id, first_request, first_key)
        async with sessions.begin() as db:
            await db.execute(
                update(ResumeTexArtifact)
                .where(ResumeTexArtifact.id == revised["artifact_id"])
                .values(tex_bytes=b"corrupt")
            )
        with pytest.raises(DomainInvariantError):
            await confirmations.download(tenant, session_id, revised["version_id"])
        async with sessions.begin() as db:
            await db.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=func.now())
            )
        with pytest.raises(DomainNotFoundError):
            await store.create(tenant, child_requests["sibling"], uuid4())
        with pytest.raises(DomainNotFoundError):
            await confirmations.list_versions(tenant, session_id)
        with pytest.raises(DomainNotFoundError):
            await confirmations.download(tenant, session_id, base_id)
        with pytest.raises(DomainNotFoundError):
            await confirmations.confirm(tenant, session_id, base_id, first_request, first_key)
    finally:
        await engine.dispose()
