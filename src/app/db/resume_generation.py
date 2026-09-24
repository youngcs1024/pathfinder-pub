"""Authorized, fixed-input resume sessions and atomic generation publication."""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation,
    JobRequirement,
    JobSnapshot,
    LLMInvocation,
    MaterialFact,
    MaterialFactEvidence,
    MaterialFactVersion,
    MaterialProject,
    MaterialSnapshot,
    MaterialSnapshotFile,
    Message,
    RequirementCoverage,
    RequirementCoverageFact,
    RequirementCoverageUserFact,
    ResumeClaimReview,
    ResumePreferenceVersion,
    ResumeProfile,
    ResumeProfileVersion,
    ResumeSession,
    ResumeSessionFact,
    ResumeSessionPreferenceVersion,
    ResumeSessionProject,
    ResumeSessionUserFact,
    ResumeSourceClaim,
    ResumeUserFact,
    ResumeUserFactVersion,
    ResumeVersion,
    ResumeVersionFact,
    Run,
    RunEvent,
    RunJob,
    WorkspaceMembership,
)
from app.db.project_facts import _current_set
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.runs import SqlAlchemyRunStore
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import DomainInvariantError, DomainNotFoundError, DomainValidationError
from app.domain.job_inputs import JobInputAdapter, ProvidedJobInputAdapter
from app.domain.project_facts import MaterialRetrievalScope, ScopedMaterialFile
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandReceiptV1, ResumeCommandRequest
from app.domain.resume_generation import (
    FixedFact,
    GenerationBudgetV1,
    GenerationCandidateV1,
    GenerationInputs,
    SessionCreateV1,
    validate_requirement_positions,
)
from app.domain.resume_profile import (
    JobPreferenceOverrideV1,
    ResumeContentV1,
    ResumePreferencesV1,
    effective_preferences,
    hard_constraint_issues,
    require_locked_items_unchanged,
    validate_preference_targets,
)
from app.domain.run_payloads import (
    READ_CONTRACTS,
    ResumeGenerationCandidateOutputV1,
    ResumeGenerationInputV1,
    ResumeGenerationResultV1,
    ResumeGenerationRunInputV1,
    ResumeGenerationRunOutputV1,
    RunMode,
    find_run_contract,
)
from app.domain.tenancy import TenantContext


async def _role(session: AsyncSession, tenant: TenantContext) -> str:
    value = await session.scalar(
        select(WorkspaceMembership.role)
        .where(
            WorkspaceMembership.workspace_id == tenant.workspace_id,
            WorkspaceMembership.user_id == tenant.actor_user_id,
            WorkspaceMembership.revoked_at.is_(None),
        )
        .with_for_update(read=True)
    )
    if value is None or value != tenant.role.value:
        raise DomainNotFoundError
    return value


async def _project_access(
    session: AsyncSession, tenant: TenantContext, project_id: UUID, role: str
) -> None:
    row = await session.scalar(
        select(MaterialProject).where(
            MaterialProject.workspace_id == tenant.workspace_id,
            MaterialProject.id == project_id,
        )
    )
    if row is None or (
        row.created_by_user_id != tenant.actor_user_id and role != WorkspaceRole.ADMIN.value
    ):
        raise DomainNotFoundError


async def _profile_inputs(
    session: AsyncSession, tenant: TenantContext, profile_version_id: UUID, preference_version: int
) -> tuple[ResumeProfileVersion, ResumePreferenceVersion, ResumeContentV1, ResumePreferencesV1]:
    version = await session.scalar(
        select(ResumeProfileVersion).where(
            ResumeProfileVersion.workspace_id == tenant.workspace_id,
            ResumeProfileVersion.id == profile_version_id,
        )
    )
    if version is None:
        raise DomainNotFoundError
    profile = await session.scalar(
        select(ResumeProfile).where(
            ResumeProfile.workspace_id == tenant.workspace_id,
            ResumeProfile.id == version.profile_id,
        )
    )
    if profile is None or (
        profile.owner_user_id != tenant.actor_user_id and tenant.role != WorkspaceRole.ADMIN
    ):
        raise DomainNotFoundError
    preference = await session.scalar(
        select(ResumePreferenceVersion).where(
            ResumePreferenceVersion.workspace_id == tenant.workspace_id,
            ResumePreferenceVersion.profile_id == profile.id,
            ResumePreferenceVersion.version == preference_version,
        )
    )
    if preference is None:
        raise DomainNotFoundError
    return (
        version,
        preference,
        ResumeContentV1.model_validate(version.content_json),
        ResumePreferencesV1.model_validate(preference.preferences_json),
    )


async def _session(
    session: AsyncSession, tenant: TenantContext, session_id: UUID, *, write: bool = False
) -> ResumeSession:
    role = await _role(session, tenant)
    query = select(ResumeSession).where(
        ResumeSession.workspace_id == tenant.workspace_id, ResumeSession.id == session_id
    )
    if write:
        query = query.with_for_update()
    row = await session.scalar(query)
    if row is None or (
        row.owner_user_id != tenant.actor_user_id and role != WorkspaceRole.ADMIN.value
    ):
        raise DomainNotFoundError
    profile = await session.scalar(
        select(ResumeProfileVersion).where(
            ResumeProfileVersion.workspace_id == tenant.workspace_id,
            ResumeProfileVersion.id == row.profile_version_id,
        )
    )
    if profile is None:
        raise DomainInvariantError("session profile version is missing")
    await _profile_inputs(session, tenant, profile.id, await _preference_number(session, row))
    projects = (
        await session.scalars(
            select(ResumeSessionProject.project_id)
            .where(
                ResumeSessionProject.workspace_id == tenant.workspace_id,
                ResumeSessionProject.session_id == session_id,
            )
            .distinct()
        )
    ).all()
    for project_id in projects:
        await _project_access(session, tenant, project_id, role)
    return row


async def _preference_number(session: AsyncSession, row: ResumeSession) -> int:
    number = await session.scalar(
        select(ResumePreferenceVersion.version).where(
            ResumePreferenceVersion.workspace_id == row.workspace_id,
            ResumePreferenceVersion.id == row.preference_version_id,
        )
    )
    if number is None:
        raise DomainInvariantError("session preference version is missing")
    return number


class _CreateWriter(ResumeCommandWriter):
    supported_kinds = frozenset({"resume_session_create"})

    def __init__(self, job_input_adapter: JobInputAdapter) -> None:
        self.job_input_adapter = job_input_adapter

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        payload = request.payload
        if not isinstance(payload, SessionCreateV1):
            raise DomainValidationError("session request is invalid")
        role = await _role(session, tenant)
        if role == WorkspaceRole.REVIEWER.value:
            raise DomainNotFoundError
        _, _, content, preferences = await _profile_inputs(
            session, tenant, payload.profile_version_id, payload.preference_version
        )
        effective = effective_preferences(preferences, payload.override)
        validate_preference_targets(content, effective)
        for project_id in payload.project_ids:
            await _project_access(session, tenant, project_id, role)
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        payload = request.payload
        assert isinstance(payload, SessionCreateV1)
        _, preference, _, _ = await _profile_inputs(
            session, tenant, payload.profile_version_id, payload.preference_version
        )
        job_snapshot = self.job_input_adapter.snapshot(payload.job).require_complete()
        if (
            job_snapshot.source != payload.job.source
            or job_snapshot.text != payload.job.text
            or job_snapshot.sha256 != payload.job.digest
        ):
            raise DomainValidationError("provided job adapter changed the fixed input")
        snapshot_id, session_id, run_id = uuid4(), uuid4(), uuid4()
        conversation_id, message_id = uuid4(), uuid4()
        session.add(
            JobSnapshot(
                id=snapshot_id,
                workspace_id=tenant.workspace_id,
                source=job_snapshot.source,
                filename=payload.job.filename,
                jd_text=job_snapshot.text,
                jd_sha256=job_snapshot.sha256,
            )
        )
        session.add(
            Conversation(
                id=conversation_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                title="Resume generation",
            )
        )
        session.add(
            Message(
                id=message_id,
                workspace_id=tenant.workspace_id,
                conversation_id=conversation_id,
                actor_user_id=tenant.actor_user_id,
                role="user",
                content="Resume generation request",
            )
        )
        run_input = ResumeGenerationRunInputV1(
            payload=ResumeGenerationInputV1(session_id=session_id)
        )
        session.add(
            Run(
                id=run_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                conversation_id=conversation_id,
                request_message_id=message_id,
                mode=RunMode.RESUME_GENERATION.value,
                resume_document_id=None,
                input_json=run_input.model_dump(mode="json", round_trip=True),
                limits_json={
                    "schema_version": 1,
                    "max_model_calls": payload.budget.max_model_calls,
                    "max_tool_calls": payload.budget.max_tool_calls,
                    "max_tool_results": payload.budget.max_tool_calls,
                    "max_iterations": 24,
                },
                status="queued",
                graph_version="pathfinder-resume-v3",
            )
        )
        await session.flush()
        session.add(
            RunJob(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                originating_actor_user_id=tenant.actor_user_id,
                run_id=run_id,
                status="queued",
            )
        )
        session.add(
            ResumeSession(
                id=session_id,
                workspace_id=tenant.workspace_id,
                owner_user_id=tenant.actor_user_id,
                profile_version_id=payload.profile_version_id,
                preference_version_id=preference.id,
                job_snapshot_id=snapshot_id,
                run_id=run_id,
                latest_run_id=run_id,
                current_version_id=None,
                revision=0,
                repair_count=0,
                override_json=payload.override.model_dump(mode="json", exclude_unset=True),
                budget_json=payload.budget.model_dump(mode="json"),
            )
        )
        await session.flush()
        for project_id in payload.project_ids:
            session.add(
                ResumeSessionProject(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    session_id=session_id,
                    project_id=project_id,
                )
            )
            fact_set = await _current_set(session, tenant, project_id)
            if fact_set is None:
                continue
            rows = (
                (
                    await session.execute(
                        select(MaterialFactVersion.id)
                        .join(
                            MaterialFact,
                            (MaterialFact.workspace_id == MaterialFactVersion.workspace_id)
                            & (MaterialFact.id == MaterialFactVersion.fact_id),
                        )
                        .where(
                            MaterialFact.workspace_id == tenant.workspace_id,
                            MaterialFact.fact_set_id == fact_set.id,
                            MaterialFactVersion.version == MaterialFact.current_version,
                            MaterialFactVersion.review_status == "confirmed",
                        )
                    )
                )
                .scalars()
                .all()
            )
            session.add_all(
                ResumeSessionFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    session_id=session_id,
                    project_id=project_id,
                    fact_version_id=value,
                )
                for value in rows
            )
        global_user_facts = (
            (
                await session.execute(
                    select(ResumeUserFactVersion.id)
                    .join(
                        ResumeUserFact,
                        (ResumeUserFact.workspace_id == ResumeUserFactVersion.workspace_id)
                        & (ResumeUserFact.id == ResumeUserFactVersion.fact_id),
                    )
                    .where(
                        ResumeUserFact.workspace_id == tenant.workspace_id,
                        ResumeUserFact.project_id.in_(payload.project_ids),
                        ResumeUserFact.scope_session_id.is_(None),
                        ResumeUserFactVersion.version == ResumeUserFact.current_version,
                        ResumeUserFactVersion.review_status == "confirmed",
                    )
                )
            )
            .scalars()
            .all()
        )
        session.add_all(
            ResumeSessionUserFact(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                session_id=session_id,
                fact_version_id=value,
            )
            for value in global_user_facts
        )
        session.add(
            ResumeSessionPreferenceVersion(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                session_id=session_id,
                version=1,
                preferences_json={
                    "override": payload.override.model_dump(mode="json", exclude_unset=True),
                    "locked_item_ids": [],
                },
            )
        )
        sequence = await session.scalar(
            update(Run)
            .where(Run.workspace_id == tenant.workspace_id, Run.id == run_id)
            .values(next_event_seq=Run.next_event_seq + 1)
            .returning(Run.next_event_seq - 1)
        )
        if sequence != 1:
            raise DomainInvariantError("new generation run event sequence is invalid")
        session.add(
            RunEvent(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                run_id=run_id,
                actor_user_id=tenant.actor_user_id,
                seq=1,
                type="run.created",
                version=1,
                payload={
                    "mode": "resume_generation",
                    "status": "queued",
                    "graph_version": "pathfinder-resume-v3",
                },
            )
        )
        return CommandReceiptV1(
            command_id=command_id, run_id=run_id, resource_id=session_id, status="queued"
        )


class SqlAlchemyResumeGenerationStore:
    def __init__(
        self, sessions: AsyncSessionFactory, *, job_input_adapter: JobInputAdapter | None = None
    ) -> None:
        self.job_input_adapter = job_input_adapter or ProvidedJobInputAdapter()
        self.sessions = sessions
        self.commands = SqlAlchemyResumeCommandStore(sessions)
        self.runs = SqlAlchemyRunStore(sessions)

    async def create(self, tenant: TenantContext, request: SessionCreateV1, request_id: UUID):
        return await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=request_id,
                kind="resume_session_create",
                target_id=None,
                payload_version=1,
                payload=request,
            ),
            writer=_CreateWriter(self.job_input_adapter),
        )

    async def list_sessions(self, tenant: TenantContext):
        async with database_session(self.sessions) as db:
            role = await _role(db, tenant)
            query = (
                select(
                    ResumeSession,
                    Run.status,
                    JobSnapshot.source,
                    JobSnapshot.filename,
                    JobSnapshot.jd_text,
                )
                .join(
                    Run,
                    (Run.workspace_id == ResumeSession.workspace_id)
                    & (Run.id == ResumeSession.latest_run_id),
                )
                .join(
                    JobSnapshot,
                    (JobSnapshot.workspace_id == ResumeSession.workspace_id)
                    & (JobSnapshot.id == ResumeSession.job_snapshot_id),
                )
                .where(ResumeSession.workspace_id == tenant.workspace_id)
                .order_by(ResumeSession.created_at.desc(), ResumeSession.id.desc())
            )
            if role != WorkspaceRole.ADMIN.value:
                query = query.where(ResumeSession.owner_user_id == tenant.actor_user_id)
            result = []
            offset = 0
            while len(result) < 50:
                page = (await db.execute(query.limit(100).offset(offset))).all()
                if not page:
                    break
                offset += len(page)
                for row, run_status, source, filename, jd_text in page:
                    try:
                        await _session(db, tenant, row.id)
                    except DomainNotFoundError:
                        continue
                    label = filename if source == "upload" else jd_text.strip().splitlines()[0]
                    result.append(
                        {
                            "session_id": row.id,
                            "run_id": row.latest_run_id,
                            "run_status": run_status,
                            "current_version_id": row.current_version_id,
                            "created_at": row.created_at,
                            "job_label": label[:120],
                        }
                    )
                    if len(result) == 50:
                        break
            return result

    async def get_session(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            snapshot = await db.scalar(
                select(JobSnapshot).where(
                    JobSnapshot.workspace_id == tenant.workspace_id,
                    JobSnapshot.id == row.job_snapshot_id,
                )
            )
            run = await db.scalar(
                select(Run).where(
                    Run.workspace_id == tenant.workspace_id,
                    Run.id == row.latest_run_id,
                )
            )
            requirements = (
                await db.scalars(
                    select(JobRequirement)
                    .where(
                        JobRequirement.workspace_id == tenant.workspace_id,
                        JobRequirement.session_id == session_id,
                    )
                    .order_by(JobRequirement.ordinal)
                )
            ).all()
            if snapshot is None or run is None:
                raise DomainInvariantError("session input or run is missing")
            project_ids = (
                await db.scalars(
                    select(ResumeSessionProject.project_id)
                    .where(
                        ResumeSessionProject.workspace_id == tenant.workspace_id,
                        ResumeSessionProject.session_id == session_id,
                    )
                    .order_by(ResumeSessionProject.project_id)
                )
            ).all()
            detail = {
                "session_id": row.id,
                "run_id": row.latest_run_id,
                "initial_run_id": row.run_id,
                "run_status": run.status,
                "error_category": run.error_category,
                "result": (
                    find_run_contract(READ_CONTRACTS, run.graph_version, RunMode(run.mode))
                    .decode_output(run.result_json)
                    .payload
                    if run.result_json is not None
                    else None
                ),
                "revision": row.revision,
                "current_version_id": row.current_version_id,
                "profile_version_id": row.profile_version_id,
                "preference_version": await _preference_number(db, row),
                "project_ids": project_ids,
                "session_preferences": (
                    await db.scalar(
                        select(ResumeSessionPreferenceVersion.preferences_json)
                        .where(
                            ResumeSessionPreferenceVersion.workspace_id == tenant.workspace_id,
                            ResumeSessionPreferenceVersion.session_id == row.id,
                        )
                        .order_by(ResumeSessionPreferenceVersion.version.desc())
                        .limit(1)
                    )
                    or {"override": row.override_json, "locked_item_ids": []}
                ),
                "budget": GenerationBudgetV1.model_validate_json(json.dumps(row.budget_json)),
                "job": {
                    "source": snapshot.source,
                    "filename": snapshot.filename,
                    "text": snapshot.jd_text,
                    "sha256": snapshot.jd_sha256,
                },
                "requirements": [
                    {
                        "id": value.id,
                        "ordinal": value.ordinal,
                        "kind": value.kind,
                        "start": value.start_offset,
                        "end": value.end_offset,
                        "quote": value.quote,
                        "inference_basis": value.inference_basis,
                    }
                    for value in requirements
                ],
            }
            session_preferences = detail.pop("session_preferences")
            detail["override"] = session_preferences["override"]
            detail["locked_item_ids"] = session_preferences.get("locked_item_ids", [])
            return detail

    async def get_version(self, tenant: TenantContext, session_id: UUID, version_id: UUID):
        async with database_session(self.sessions) as db:
            await _session(db, tenant, session_id)
            row = await db.scalar(
                select(ResumeVersion).where(
                    ResumeVersion.workspace_id == tenant.workspace_id,
                    ResumeVersion.session_id == session_id,
                    ResumeVersion.id == version_id,
                )
            )
            if row is None:
                raise DomainNotFoundError
            coverage = (
                await db.scalars(
                    select(RequirementCoverage).where(
                        RequirementCoverage.workspace_id == tenant.workspace_id,
                        RequirementCoverage.version_id == version_id,
                    )
                )
            ).all()
            result = []
            for item in coverage:
                fact_ids = (
                    await db.scalars(
                        select(RequirementCoverageFact.fact_version_id).where(
                            RequirementCoverageFact.workspace_id == tenant.workspace_id,
                            RequirementCoverageFact.coverage_id == item.id,
                        )
                    )
                ).all()
                user_fact_ids = (
                    await db.scalars(
                        select(RequirementCoverageUserFact.fact_version_id).where(
                            RequirementCoverageUserFact.workspace_id == tenant.workspace_id,
                            RequirementCoverageUserFact.coverage_id == item.id,
                        )
                    )
                ).all()
                result.append(
                    {
                        "requirement_id": item.requirement_id,
                        "support": item.support,
                        "verification": item.verification,
                        "reason": item.reason,
                        "fact_version_ids": fact_ids + user_fact_ids,
                        "item_ids": item.item_ids_json,
                    }
                )
            facts = []
            version_links = (
                await db.scalars(
                    select(ResumeVersionFact).where(
                        ResumeVersionFact.workspace_id == tenant.workspace_id,
                        ResumeVersionFact.version_id == version_id,
                    )
                )
            ).all()
            material_ids = [
                item.material_fact_version_id
                for item in version_links
                if item.material_fact_version_id is not None
            ]
            user_ids = [
                item.user_fact_version_id
                for item in version_links
                if item.user_fact_version_id is not None
            ]
            bound = (
                await db.execute(
                    select(ResumeSessionFact, MaterialFactVersion)
                    .join(
                        MaterialFactVersion,
                        (MaterialFactVersion.workspace_id == ResumeSessionFact.workspace_id)
                        & (MaterialFactVersion.id == ResumeSessionFact.fact_version_id),
                    )
                    .where(
                        ResumeSessionFact.workspace_id == tenant.workspace_id,
                        ResumeSessionFact.session_id == session_id,
                        ResumeSessionFact.fact_version_id.in_(material_ids),
                    )
                    .order_by(ResumeSessionFact.project_id, ResumeSessionFact.fact_version_id)
                )
            ).all()
            for binding, fact in bound:
                evidence_rows = (
                    await db.execute(
                        select(
                            MaterialFactEvidence,
                            MaterialSnapshotFile.path,
                            MaterialSnapshot.source_revision,
                        )
                        .join(
                            MaterialSnapshotFile,
                            (MaterialSnapshotFile.workspace_id == MaterialFactEvidence.workspace_id)
                            & (MaterialSnapshotFile.id == MaterialFactEvidence.snapshot_file_id),
                        )
                        .join(
                            MaterialSnapshot,
                            (MaterialSnapshot.workspace_id == MaterialSnapshotFile.workspace_id)
                            & (MaterialSnapshot.id == MaterialSnapshotFile.snapshot_id),
                        )
                        .where(
                            MaterialFactEvidence.workspace_id == tenant.workspace_id,
                            MaterialFactEvidence.fact_version_id == fact.id,
                        )
                        .order_by(MaterialSnapshotFile.path, MaterialFactEvidence.start_line)
                    )
                ).all()
                facts.append(
                    {
                        "version_id": fact.id,
                        "project_id": binding.project_id,
                        "claim": fact.claim,
                        "kind": fact.kind,
                        "conditions": fact.conditions_json,
                        "source": "material_snapshot",
                        "attested_at": None,
                        "evidence": [
                            {
                                "snapshot_file_id": evidence.snapshot_file_id,
                                "path": path,
                                "source_revision": revision,
                                "start_line": evidence.start_line,
                                "end_line": evidence.end_line,
                                "quote": evidence.quote,
                            }
                            for evidence, path, revision in evidence_rows
                        ],
                    }
                )
            user_bound = (
                await db.execute(
                    select(ResumeUserFact, ResumeUserFactVersion)
                    .join(
                        ResumeUserFactVersion,
                        (ResumeUserFactVersion.workspace_id == ResumeUserFact.workspace_id)
                        & (ResumeUserFactVersion.fact_id == ResumeUserFact.id),
                    )
                    .where(
                        ResumeUserFact.workspace_id == tenant.workspace_id,
                        ResumeUserFactVersion.id.in_(user_ids),
                    )
                )
            ).all()
            facts.extend(
                {
                    "version_id": version.id,
                    "project_id": user_fact.project_id,
                    "claim": version.claim,
                    "kind": version.kind,
                    "conditions": version.conditions_json,
                    "source": "user_attestation",
                    "attested_at": version.attested_at,
                    "evidence": [],
                }
                for user_fact, version in user_bound
            )
            return {
                "version_id": row.id,
                "session_id": session_id,
                "version": row.version,
                "artifact_id": row.artifact_id,
                "parent_version_id": row.parent_version_id,
                "feedback_id": row.feedback_id,
                "diff": row.diff_json,
                "impact": row.impact_json,
                "content": row.content_json,
                "validation": row.validation_json,
                "coverage": result,
                "facts": facts,
            }

    async def cancel(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            run_id = row.latest_run_id
        return await self.runs.cancel_run(
            tenant=tenant, run_id=run_id, allow_other_creator=tenant.role is WorkspaceRole.ADMIN
        )

    async def execution_inputs(self, tenant: TenantContext, session_id: UUID) -> GenerationInputs:
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            snapshot = await db.scalar(
                select(JobSnapshot).where(
                    JobSnapshot.workspace_id == tenant.workspace_id,
                    JobSnapshot.id == row.job_snapshot_id,
                )
            )
            if snapshot is None:
                raise DomainInvariantError("session job snapshot is missing")
            number = await _preference_number(db, row)
            _, _, content, global_preferences = await _profile_inputs(
                db, tenant, row.profile_version_id, number
            )
            override = JobPreferenceOverrideV1.model_validate(row.override_json)
            preferences = effective_preferences(global_preferences, override)
            linked = (
                await db.execute(
                    select(ResumeSessionFact, MaterialFactVersion)
                    .join(
                        MaterialFactVersion,
                        (MaterialFactVersion.workspace_id == ResumeSessionFact.workspace_id)
                        & (MaterialFactVersion.id == ResumeSessionFact.fact_version_id),
                    )
                    .where(
                        ResumeSessionFact.workspace_id == tenant.workspace_id,
                        ResumeSessionFact.session_id == session_id,
                    )
                )
            ).all()
            user_linked = (
                await db.execute(
                    select(ResumeUserFact, ResumeUserFactVersion)
                    .join(
                        ResumeUserFactVersion,
                        (ResumeUserFactVersion.workspace_id == ResumeUserFact.workspace_id)
                        & (ResumeUserFactVersion.fact_id == ResumeUserFact.id),
                    )
                    .join(
                        ResumeSessionUserFact,
                        (ResumeSessionUserFact.workspace_id == ResumeUserFactVersion.workspace_id)
                        & (ResumeSessionUserFact.fact_version_id == ResumeUserFactVersion.id),
                    )
                    .where(
                        ResumeSessionUserFact.workspace_id == tenant.workspace_id,
                        ResumeSessionUserFact.session_id == session_id,
                        ResumeUserFactVersion.review_status == "confirmed",
                    )
                )
            ).all()
            superseded_material = {
                fact.supersedes_material_version_id
                for fact, _ in user_linked
                if fact.supersedes_material_version_id is not None
            }
            superseded_user = {
                fact.supersedes_user_version_id
                for fact, _ in user_linked
                if fact.supersedes_user_version_id is not None
            }
            facts = tuple(
                FixedFact(
                    version_id=version.id,
                    project_id=link.project_id,
                    claim=version.claim,
                    kind=version.kind,
                    conditions=version.conditions_json,
                )
                for link, version in linked
                if version.id not in superseded_material
            ) + tuple(
                FixedFact(
                    version_id=version.id,
                    project_id=fact.project_id,
                    claim=version.claim,
                    kind=version.kind,
                    conditions=version.conditions_json,
                )
                for fact, version in user_linked
                if version.id not in superseded_user
            )
            source_import_id = await db.scalar(
                select(ResumeProfileVersion.source_import_id).where(
                    ResumeProfileVersion.workspace_id == tenant.workspace_id,
                    ResumeProfileVersion.id == row.profile_version_id,
                )
            )
            claims = (
                await db.scalars(
                    select(ResumeSourceClaim).where(
                        ResumeSourceClaim.workspace_id == tenant.workspace_id,
                        ResumeSourceClaim.source_import_id == source_import_id,
                    )
                )
            ).all()
            reviews = (
                (
                    await db.scalars(
                        select(ResumeClaimReview).where(
                            ResumeClaimReview.workspace_id == tenant.workspace_id,
                            ResumeClaimReview.claim_id.in_([item.id for item in claims]),
                        )
                    )
                ).all()
                if claims
                else []
            )
            latest: dict[UUID, ResumeClaimReview] = {}
            for review in reviews:
                if (
                    review.claim_id not in latest
                    or review.version > latest[review.claim_id].version
                ):
                    latest[review.claim_id] = review
            project_map: dict[UUID, UUID] = {}
            for claim in claims:
                review = latest.get(claim.id)
                if (
                    review is not None
                    and review.decision == "linked"
                    and review.project_id is not None
                ):
                    previous = project_map.setdefault(claim.project_item_id, review.project_id)
                    if previous != review.project_id:
                        raise DomainInvariantError("profile project has conflicting fact links")
            evidence = (
                (
                    await db.execute(
                        select(MaterialSnapshotFile, MaterialSnapshot.source_revision)
                        .join(
                            MaterialFactEvidence,
                            (MaterialFactEvidence.workspace_id == MaterialSnapshotFile.workspace_id)
                            & (MaterialFactEvidence.snapshot_file_id == MaterialSnapshotFile.id),
                        )
                        .join(
                            MaterialSnapshot,
                            (MaterialSnapshot.workspace_id == MaterialSnapshotFile.workspace_id)
                            & (MaterialSnapshot.id == MaterialSnapshotFile.snapshot_id),
                        )
                        .where(
                            MaterialFactEvidence.workspace_id == tenant.workspace_id,
                            MaterialFactEvidence.fact_version_id.in_(
                                [fact.version_id for fact in facts]
                            ),
                        )
                    )
                ).all()
                if facts
                else []
            )
            scope_files = {
                file.id: ScopedMaterialFile(
                    id=file.id,
                    path=file.path,
                    content=file.content,
                    document_id=file.document_id,
                    source_revision=revision,
                )
                for file, revision in evidence
            }
            return GenerationInputs(
                session_id=session_id,
                run_id=row.run_id,
                job_text=snapshot.jd_text,
                profile_content=content,
                preferences=preferences,
                budget=GenerationBudgetV1.model_validate_json(json.dumps(row.budget_json)),
                facts=facts,
                project_map=project_map,
                retrieval_scope=MaterialRetrievalScope(files=tuple(scope_files.values())),
            )

    async def spend_allowed(self, tenant: TenantContext, session_id: UUID) -> bool:
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            budget = GenerationBudgetV1.model_validate_json(json.dumps(row.budget_json))
            invocations = (
                await db.execute(
                    select(LLMInvocation.provider, LLMInvocation.estimated_cost).where(
                        LLMInvocation.workspace_id == tenant.workspace_id,
                        LLMInvocation.run_id == row.run_id,
                    )
                )
            ).all()
            if any(cost is None and provider != "fake" for provider, cost in invocations):
                return False
            return (
                len(invocations) < budget.max_model_calls
                and sum(
                    (Decimal(cost) for _, cost in invocations if cost is not None),
                    Decimal(0),
                )
                < budget.max_cost_cny
            )

    async def reserve_repair(self, tenant: TenantContext, session_id: UUID) -> bool:
        async with self.sessions.begin() as db:
            await _session(db, tenant, session_id, write=True)
            changed = await db.scalar(
                update(ResumeSession)
                .where(
                    ResumeSession.workspace_id == tenant.workspace_id,
                    ResumeSession.id == session_id,
                    ResumeSession.repair_count == 0,
                    ResumeSession.current_version_id.is_(None),
                )
                .values(repair_count=1)
                .returning(ResumeSession.id)
            )
            return changed is not None


class ResumeGenerationPublisher:
    """Called by the existing job completion transaction after lease validation."""

    def __init__(self, artifacts: SqlAlchemyResumeArtifactStore) -> None:
        self.artifacts = artifacts

    async def publish(
        self,
        db: AsyncSession,
        tenant: TenantContext,
        run_id: UUID,
        output: ResumeGenerationCandidateOutputV1,
    ) -> ResumeGenerationRunOutputV1:
        row = await db.scalar(
            select(ResumeSession)
            .where(
                ResumeSession.workspace_id == tenant.workspace_id,
                ResumeSession.run_id == run_id,
            )
            .with_for_update()
        )
        if row is None:
            raise DomainInvariantError("generation session is missing")
        await _session(db, tenant, row.id)
        candidate: GenerationCandidateV1 = output.payload
        snapshot = await db.scalar(
            select(JobSnapshot).where(
                JobSnapshot.workspace_id == tenant.workspace_id,
                JobSnapshot.id == row.job_snapshot_id,
            )
        )
        if snapshot is None:
            raise DomainInvariantError("generation job snapshot is missing")
        validate_requirement_positions(snapshot.jd_text, candidate.requirements)
        facts = (
            await db.scalars(
                select(ResumeSessionFact).where(
                    ResumeSessionFact.workspace_id == tenant.workspace_id,
                    ResumeSessionFact.session_id == row.id,
                )
            )
        ).all()
        material_ids = {item.fact_version_id for item in facts}
        user_ids = set(
            (
                await db.scalars(
                    select(ResumeSessionUserFact.fact_version_id).where(
                        ResumeSessionUserFact.workspace_id == tenant.workspace_id,
                        ResumeSessionUserFact.session_id == row.id,
                    )
                )
            ).all()
        )
        from app.db.resume_revision import _bound_facts

        active_material, active_user, _ = await _bound_facts(db, tenant, row.id)
        material_ids &= active_material
        user_ids &= active_user
        permitted_facts = material_ids | user_ids
        if len(candidate.coverage) != len(candidate.requirements):
            raise DomainValidationError("generation coverage is incomplete")
        if {item.requirement_ordinal for item in candidate.coverage} != set(
            range(len(candidate.requirements))
        ):
            raise DomainValidationError("generation coverage has invalid references")
        if any(not set(item.fact_version_ids) <= permitted_facts for item in candidate.coverage):
            raise DomainValidationError("generation coverage uses unbound facts")
        if not set(candidate.omitted_fact_version_ids) <= permitted_facts:
            raise DomainValidationError("generation omission uses unbound facts")
        if set(candidate.omission_reasons) != set(candidate.omitted_fact_version_ids):
            raise DomainValidationError("generation omission reasons are incomplete")
        if row.current_version_id is not None:
            return ResumeGenerationRunOutputV1(
                payload=ResumeGenerationResultV1(
                    session_id=row.id,
                    version_id=row.current_version_id,
                    artifact_id=await db.scalar(
                        select(ResumeVersion.artifact_id).where(
                            ResumeVersion.workspace_id == tenant.workspace_id,
                            ResumeVersion.id == row.current_version_id,
                        )
                    ),
                    outcome="draft",
                    questions=candidate.questions,
                )
            )
        requirements = []
        for ordinal, item in enumerate(candidate.requirements):
            requirement_id = uuid4()
            db.add(
                JobRequirement(
                    id=requirement_id,
                    workspace_id=tenant.workspace_id,
                    session_id=row.id,
                    ordinal=ordinal,
                    kind=item.kind,
                    start_offset=item.start,
                    end_offset=item.end,
                    quote=item.quote,
                    inference_basis=item.inference_basis,
                )
            )
            requirements.append(requirement_id)
        if candidate.content is None:
            return ResumeGenerationRunOutputV1(
                payload=ResumeGenerationResultV1(
                    session_id=row.id,
                    version_id=None,
                    artifact_id=None,
                    outcome="needs_input",
                    questions=candidate.questions,
                )
            )
        number = await _preference_number(db, row)
        _, _, profile_content, global_preferences = await _profile_inputs(
            db, tenant, row.profile_version_id, number
        )
        preferences = effective_preferences(
            global_preferences, JobPreferenceOverrideV1.model_validate(row.override_json)
        )
        require_locked_items_unchanged(profile_content, candidate.content, preferences)
        if hard_constraint_issues(candidate.content, preferences):
            raise DomainValidationError("generation violates hard constraints")
        known_items = candidate.content.item_ids()
        if any(not set(item.item_ids) <= known_items for item in candidate.coverage):
            raise DomainValidationError("generation coverage uses unknown items")
        artifact_id = await self.artifacts.stage(
            db,
            tenant,
            profile_version_id=row.profile_version_id,
            content=candidate.content,
            preferences=preferences,
        )
        version_id = uuid4()
        db.add(
            ResumeVersion(
                id=version_id,
                workspace_id=tenant.workspace_id,
                session_id=row.id,
                version=1,
                artifact_id=artifact_id,
                content_json=candidate.content.model_dump(mode="json"),
                validation_json={
                    "questions": list(candidate.questions),
                    "correction_count": candidate.correction_count,
                    "prompt_version": candidate.prompt_version,
                    "model_id": candidate.model_id,
                    "retrieval_config_version": candidate.retrieval_config_version,
                    "semantic_support": "needs_human_review",
                    "omitted_fact_version_ids": [
                        str(value) for value in candidate.omitted_fact_version_ids
                    ],
                    "omission_reasons": {
                        str(key): value for key, value in candidate.omission_reasons.items()
                    },
                },
            )
        )
        await db.flush()
        db.add_all(
            ResumeVersionFact(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                version_id=version_id,
                material_fact_version_id=value,
                user_fact_version_id=None,
            )
            for value in material_ids
        )
        db.add_all(
            ResumeVersionFact(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                version_id=version_id,
                material_fact_version_id=None,
                user_fact_version_id=value,
            )
            for value in user_ids
        )
        for item in candidate.coverage:
            coverage_id = uuid4()
            db.add(
                RequirementCoverage(
                    id=coverage_id,
                    workspace_id=tenant.workspace_id,
                    version_id=version_id,
                    requirement_id=requirements[item.requirement_ordinal],
                    support=item.support,
                    verification=item.verification,
                    reason=item.reason,
                    item_ids_json=[str(value) for value in item.item_ids],
                )
            )
            db.add_all(
                RequirementCoverageFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    coverage_id=coverage_id,
                    fact_version_id=value,
                )
                for value in item.fact_version_ids
                if value in material_ids
            )
            db.add_all(
                RequirementCoverageUserFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    coverage_id=coverage_id,
                    fact_version_id=value,
                )
                for value in item.fact_version_ids
                if value in user_ids
            )
        row.current_version_id = version_id
        row.revision += 1
        return ResumeGenerationRunOutputV1(
            payload=ResumeGenerationResultV1(
                session_id=row.id,
                version_id=version_id,
                artifact_id=artifact_id,
                outcome="draft",
                questions=candidate.questions,
            )
        )
