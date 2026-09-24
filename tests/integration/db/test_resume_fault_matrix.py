"""R6.2 real PostgreSQL publication windows and persisted recovery guards."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from random import Random
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update

from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import (
    LLMInvocation,
    MaterialSnapshotFile,
    ResumeFeedback,
    ResumeSession,
    ResumeTexArtifact,
    ResumeVersion,
    Run,
    RunEvent,
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
from app.domain.errors import DomainConflictError, DomainNotFoundError
from app.domain.project_facts import CandidateFactV1, FactEvidenceV1
from app.domain.provisioning import ProvisioningService
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.resume_revision import (
    ContentFeedbackV1,
    FactFeedbackV1,
    FactReviewV1,
    LockChangeV1,
    PatchV1,
    UserFactInputV1,
)
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatModelResult
from app.worker.backoff import ExponentialBackoff
from app.worker.generation_executor import GenerationRunExecutor
from app.worker.revision_executor import RevisionRunExecutor
from app.worker.settings import WorkerRuntimeSettings
from tests.integration.db.test_material_snapshots import _fixture_aliases, _runner
from tests.unit.agents.test_resume_generation import _requirements, _selection

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


@pytest.fixture
async def rig(migrated_database_url, tmp_path):
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
        request = SessionCreateV1(
            profile_version_id=profile["version_id"],
            preference_version=1,
            project_ids=(project["id"],),
            job=JobInputV1(source="paste", text="Build a synthetic service"),
            budget=GenerationBudgetV1(
                max_model_calls=6, max_tool_calls=0, max_cost_cny=Decimal("1")
            ),
        )
        key = uuid4()
        created = await store.create(tenant, request, key)
        artifacts = SqlAlchemyResumeArtifactStore(
            sessions,
            expected_source_sha256=source_hash,
            expected_preamble_sha256=preamble_hash,
        )
        revisions = SqlAlchemyResumeRevisionStore(sessions)
        jobs = SqlAlchemyWorkerJobStore(
            sessions,
            ExponentialBackoff(WorkerRuntimeSettings(), Random(1)),
            generation_publisher=ResumeGenerationPublisher(artifacts),
            revision_publisher=ResumeRevisionPublisher(artifacts),
        )
        yield SimpleNamespace(
            sessions=sessions,
            tenant=tenant,
            store=store,
            request=request,
            key=key,
            created=created,
            artifacts=artifacts,
            jobs=jobs,
            revisions=revisions,
            project_id=project["id"],
            project_item_id=project_item_id,
            fact_version_id=fact_version_id,
            reader=SqlAlchemyRunExecutionReader(sessions),
            session_id=created.receipt.resource_id,
            confirmations=SqlAlchemyResumeConfirmationStore(sessions, artifacts),
        )
    finally:
        await engine.dispose()


async def claim(rig, now=None):
    now = now or datetime.now(UTC)
    job = await rig.jobs.claim_due_job(
        worker_id="r62",
        now=now,
        lease_duration=timedelta(minutes=5),
    )
    assert job is not None
    prepared = await rig.jobs.prepare_claimed_job(job=job, resolved_tenant=rig.tenant, now=now)
    assert prepared.disposition == "execute"
    return job


def executor(rig, *, script=None, revision=False):
    adapter = ScriptedFakeChatModel(
        script
        or [
            _requirements(),
            _selection(rig.project_item_id, rig.fact_version_id),
        ]
    )
    factory = LLMFactory(
        recorder=SqlAlchemyInvocationRecorder(rig.sessions),
        chat_adapter=adapter,
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )
    kwargs = dict(
        reader=rig.reader,
        model_factory=lambda actor, run_id: factory.create_chat_model(
            LLMInvocationContext(actor.workspace_id, actor.actor_user_id, run_id=run_id)
        ),
    )
    runner = (
        RevisionRunExecutor(revisions=rig.revisions, **kwargs)
        if revision
        else GenerationRunExecutor(sessions=rig.store, tools_factory=lambda *_: None, **kwargs)
    )
    return runner, adapter


async def candidate(rig, job, **kwargs):
    runner, adapter = executor(rig, **kwargs)
    result = await runner.execute(job.run_id, rig.tenant, job.graph_version)
    assert result.status == RunStatus.COMPLETED, result.error_category
    assert result.result is not None
    return result.result, adapter


async def counts(rig):
    async with rig.sessions() as db:
        return tuple(
            [
                await db.scalar(
                    select(func.count())
                    .select_from(model)
                    .where(model.workspace_id == rig.tenant.workspace_id)
                )
                for model in (ResumeVersion, ResumeTexArtifact)
            ]
        )


async def draft(rig):
    job = await claim(rig)
    output, _ = await candidate(rig, job)
    assert await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
    detail = await rig.store.get_session(rig.tenant, rig.session_id)
    version = await rig.store.get_version(rig.tenant, rig.session_id, detail["current_version_id"])
    return detail, version


async def revision(rig, *, instruction=False):
    detail, version = await draft(rig)
    target = UUID(version["content"]["projects"][0]["bullet_ids"][0])
    patch = PatchV1(
        operation="replace_text",
        item_id=target,
        field="bullet",
        text="Built a synthetic service.",
        fact_version_ids=(rig.fact_version_id,),
    )
    request = ContentFeedbackV1(
        expected_session_revision=detail["revision"],
        base_version_id=detail["current_version_id"],
        target_item_ids=(target,),
        patches=() if instruction else (patch,),
        instruction="Clarify the service work" if instruction else None,
    )
    key = uuid4()
    accepted = await rig.revisions.command(rig.tenant, rig.session_id, request, key)
    return accepted, request, key, patch, version


@pytest.mark.parametrize("mode", ["generation", "revision"])
@pytest.mark.parametrize("window", ["before_publish", "inside_transaction", "after_commit"])
async def test_publication_crash_windows_and_late_owner(rig, monkeypatch, mode, window):
    revision_mode = mode == "revision"
    accepted = request = key = None
    if revision_mode:
        accepted, request, key, _, _ = await revision(rig)
    baseline = await counts(rig)
    job = await claim(rig)
    output, _ = await candidate(rig, job, revision=revision_mode)
    before = await rig.store.get_session(rig.tenant, rig.session_id)
    if window == "inside_transaction":
        publisher = (
            rig.jobs._revision_publisher if revision_mode else rig.jobs._generation_publisher
        )
        publish = publisher.publish

        async def crash(*args, **kwargs):
            await publish(*args, **kwargs)
            await args[0].flush()
            raise RuntimeError("synthetic publication crash")

        with monkeypatch.context() as scoped:
            scoped.setattr(publisher, "publish", crash)
            with pytest.raises(RuntimeError, match="synthetic publication crash"):
                await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
    elif window == "after_commit":
        assert await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
        committed = await rig.store.get_session(rig.tenant, rig.session_id)
        stored = await rig.confirmations.download(
            rig.tenant, rig.session_id, committed["current_version_id"]
        )
        # The caller lost the return value, then repeats finalize with the old lease.
        assert not await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
        assert (
            await rig.jobs.claim_due_job(
                worker_id="restart", now=datetime.now(UTC), lease_duration=timedelta(minutes=5)
            )
            is None
        )
        assert (
            await rig.confirmations.download(
                rig.tenant, rig.session_id, committed["current_version_id"]
            )
            == stored
        )
    if window != "after_commit":
        assert await counts(rig) == baseline
        assert (await rig.store.get_session(rig.tenant, rig.session_id))[
            "current_version_id"
        ] == before["current_version_id"]
        later = job.lease_expires_at + timedelta(seconds=1)
        assert (await rig.jobs.reclaim_stale_leases(now=later, limit=10)).requeued == 1
        recovered = await claim(rig, later + timedelta(minutes=1))
        assert recovered.owner_token != job.owner_token
        assert not await rig.jobs.complete(job=job, result=output, now=later)
        rerun, _ = await candidate(rig, recovered, revision=revision_mode)
        assert await rig.jobs.complete(
            job=recovered, result=rerun, now=later + timedelta(minutes=1)
        )
    assert await counts(rig) == tuple(value + 1 for value in baseline)
    if revision_mode:
        assert (await rig.revisions.command(rig.tenant, rig.session_id, request, key)).replayed
        async with rig.sessions() as db:
            feedback = await db.get(ResumeFeedback, accepted.receipt.resource_id)
            assert feedback.target_version_id is not None
    else:
        assert (await rig.store.create(rig.tenant, rig.request, rig.key)).replayed
    async with rig.sessions() as db:
        assert (await db.get(Run, job.run_id)).status == "completed"
        assert (await db.get(RunJob, job.job_id)).status == "done"
        events = (
            await db.scalars(
                select(RunEvent).where(
                    RunEvent.run_id == job.run_id, RunEvent.type == "run.completed"
                )
            )
        ).all()
        assert len(events) == 1
        assert "canary@example.test" not in repr([event.payload for event in events])
        assert "Built a synthetic service" not in repr([event.payload for event in events])


@pytest.mark.parametrize("boundary", ["before_execute", "before_publish"])
@pytest.mark.parametrize("stop", ["cancel", "membership"])
async def test_cancellation_and_revocation_preserve_existing_draft(rig, boundary, stop):
    _, _, _, _, previous = await revision(rig)
    original = await rig.confirmations.download(rig.tenant, rig.session_id, previous["version_id"])
    job = await claim(rig)
    if boundary == "before_publish":
        output, _ = await candidate(rig, job, revision=True)
    if stop == "cancel":
        await rig.store.cancel(rig.tenant, rig.session_id)
    else:
        async with rig.sessions.begin() as db:
            await db.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == rig.tenant.workspace_id,
                    WorkspaceMembership.user_id == rig.tenant.actor_user_id,
                )
                .values(revoked_at=func.now())
            )
        with pytest.raises(DomainNotFoundError):
            await rig.store.create(rig.tenant, rig.request, rig.key)
    if boundary == "before_execute":
        runner, adapter = executor(rig, revision=True)
        result = await runner.execute(job.run_id, rig.tenant, job.graph_version)
        assert result.status == RunStatus.CANCELLED
        assert adapter.invoke_count == 0
    else:
        assert await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
        async with rig.sessions() as db:
            assert (await db.get(Run, job.run_id)).status == "cancelled"
    assert await counts(rig) == (1, 1)
    async with rig.sessions() as db:
        row = await db.get(ResumeSession, rig.session_id)
        assert row.current_version_id == previous["version_id"]
        artifact = await db.get(ResumeTexArtifact, previous["artifact_id"])
        assert artifact.tex_bytes == original[0]


async def test_recovery_does_not_reset_generation_budget_or_repair(rig):
    job = await claim(rig)
    assert await rig.store.reserve_repair(rig.tenant, rig.session_id)
    for _ in range(3):
        await candidate(rig, job)
    runner, adapter = executor(rig)
    result = await runner.execute(job.run_id, rig.tenant, job.graph_version)
    assert result.status == RunStatus.FAILED
    assert result.error_category == "generation_budget_exhausted"
    assert adapter.invoke_count == 0
    assert not await rig.store.reserve_repair(rig.tenant, rig.session_id)
    async with rig.sessions() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(LLMInvocation)
                .where(LLMInvocation.run_id == job.run_id)
            )
            == 6
        )
    assert await counts(rig) == (0, 0)


async def test_revision_recovery_exhausts_repair_without_publishing_invalid_patch(rig):
    accepted, _, _, patch, _ = await revision(rig, instruction=True)
    job = await claim(rig)
    invalid = patch.model_copy(update={"text": "Invented 999 deployments"})
    response = ChatModelResult(content=json.dumps({"patches": [invalid.model_dump(mode="json")]}))
    first, adapter = await candidate(rig, job, revision=True, script=[response, response])
    assert first.payload.content is None
    assert adapter.invoke_count == 2
    assert first.payload.correction_count == 1
    recovered, adapter = await candidate(rig, job, revision=True, script=[response])
    assert recovered.payload.content is None
    assert adapter.invoke_count == 1
    assert not await rig.revisions.reserve_repair(rig.tenant, accepted.receipt.resource_id)
    assert await rig.jobs.complete(job=job, result=recovered, now=datetime.now(UTC))
    assert await counts(rig) == (1, 1)
    async with rig.sessions() as db:
        feedback = await db.get(ResumeFeedback, accepted.receipt.resource_id)
        assert feedback.repair_count == 1 and feedback.target_version_id is None


@pytest.mark.parametrize("change", ["lock", "fact_correction"])
async def test_failed_revision_then_changed_facts_or_locks_cannot_reuse_old_candidate(rig, change):
    _, old_request, _, patch, previous = await revision(rig)
    job = await claim(rig)
    output, _ = await candidate(rig, job, revision=True)
    detail = await rig.store.get_session(rig.tenant, rig.session_id)
    fields = dict(
        expected_session_revision=detail["revision"], base_version_id=detail["current_version_id"]
    )
    command = (
        LockChangeV1(**fields, item_id=patch.item_id, locked=True)
        if change == "lock"
        else FactFeedbackV1(
            **fields,
            fact=UserFactInputV1(
                project_id=rig.project_id,
                scope="session",
                claim="Corrected service claim",
                kind="personal_statement",
                supersedes_material_version_id=rig.fact_version_id,
            ),
        )
    )
    # Active modifications serialize all feedback; another page cannot change the inputs.
    with pytest.raises(DomainConflictError):
        await rig.revisions.command(rig.tenant, rig.session_id, command, uuid4())
    assert await rig.jobs.fail(
        job=job, error_category="invalid_revision_input", now=datetime.now(UTC)
    )
    submitted = await rig.revisions.command(rig.tenant, rig.session_id, command, uuid4())
    if change == "fact_correction":
        item = next(
            item
            for item in await rig.revisions.list_feedback(rig.tenant, rig.session_id)
            if item["feedback_id"] == submitted.receipt.resource_id
        )
        detail = await rig.store.get_session(rig.tenant, rig.session_id)
        await rig.revisions.command(
            rig.tenant,
            rig.session_id,
            FactReviewV1(
                expected_session_revision=detail["revision"],
                base_version_id=detail["current_version_id"],
                fact_version_id=UUID(item["normalized"]["fact_version_id"]),
                decision="confirm",
                attested=True,
            ),
            uuid4(),
        )
    detail = await rig.store.get_session(rig.tenant, rig.session_id)
    retry = old_request.model_copy(update={"expected_session_revision": detail["revision"]})
    await rig.revisions.command(rig.tenant, rig.session_id, retry, uuid4())
    assert not await rig.jobs.complete(job=job, result=output, now=datetime.now(UTC))
    recovered_job = await claim(rig)
    recovered, _ = await candidate(rig, recovered_job, revision=True)
    assert recovered.payload.content is None
    assert await rig.jobs.complete(job=recovered_job, result=recovered, now=datetime.now(UTC))
    assert await counts(rig) == (1, 1)
    assert (await rig.store.get_session(rig.tenant, rig.session_id))[
        "current_version_id"
    ] == previous["version_id"]


@pytest.mark.parametrize("mode", ["generation", "revision"])
@pytest.mark.parametrize("failure", ["database", "provider", "accounting"])
async def test_execution_failure_is_classified_without_publishing(rig, monkeypatch, mode, failure):
    from app.domain.errors import DomainUnavailableError
    from app.llm.factory import LLMAccountingError, LLMProviderError

    revision_mode = mode == "revision"
    if revision_mode:
        await revision(rig, instruction=True)
    baseline = await counts(rig)
    job = await claim(rig)
    runner, adapter = executor(rig, revision=revision_mode)
    errors = {
        "database": DomainUnavailableError(),
        "provider": LLMProviderError(category="provider_unavailable"),
        "accounting": LLMAccountingError(phase="prepare", retryable=True),
    }

    async def unavailable(*args, **kwargs):
        raise errors[failure]

    if failure == "database":
        target = rig.revisions if revision_mode else rig.store
        monkeypatch.setattr(
            target, "revision_inputs" if revision_mode else "execution_inputs", unavailable
        )
    else:

        class Model:
            model = adapter.model
            invoke = staticmethod(unavailable)

        runner.model_factory = lambda *_: Model()
    result = await runner.execute(job.run_id, rig.tenant, job.graph_version)
    assert result.status == RunStatus.FAILED and result.retryable
    assert (
        result.error_category
        == {
            "database": "database_unavailable",
            "provider": "provider_unavailable",
            "accounting": "llm_accounting_failed",
        }[failure]
    )
    assert await counts(rig) == baseline
