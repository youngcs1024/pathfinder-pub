"""Authorized R5.1 feedback, user attestations, and atomic revision publication."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    LLMInvocation,
    MaterialFact,
    MaterialFactSet,
    MaterialFactVersion,
    Message,
    RequirementCoverage,
    RequirementCoverageFact,
    RequirementCoverageUserFact,
    ResumeFeedback,
    ResumePreferenceVersion,
    ResumeProfile,
    ResumeSession,
    ResumeSessionFact,
    ResumeSessionPreferenceVersion,
    ResumeSessionProject,
    ResumeSessionUserFact,
    ResumeUserFact,
    ResumeUserFactVersion,
    ResumeVersion,
    ResumeVersionFact,
    Run,
    RunEvent,
    RunJob,
)
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.resume_generation import _preference_number, _profile_inputs, _session
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import (
    DomainConflictError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
)
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import (
    CommandReceiptV1,
    ResumeCommandRequest,
    SessionWritePrecondition,
    SessionWriteState,
)
from app.domain.resume_profile import (
    JobPreferenceOverrideV1,
    ResumeContentV1,
    ResumePreferencesV1,
    effective_preferences,
    hard_constraint_issues,
    require_locked_items_unchanged,
)
from app.domain.resume_revision import (
    AnswerFeedbackV1,
    ContentFeedbackV1,
    FactFeedbackV1,
    FactReviewV1,
    LockChangeV1,
    PreferenceFeedbackV1,
    RevisionCandidateV1,
    RevisionInputs,
    apply_scoped_patches,
    user_fact_issues,
)
from app.domain.run_payloads import (
    ResumeRevisionCandidateOutputV1,
    ResumeRevisionInputV1,
    ResumeRevisionResultV1,
    ResumeRevisionRunInputV1,
    ResumeRevisionRunOutputV1,
    RunMode,
)
from app.domain.tenancy import TenantContext

type FeedbackPayload = (
    ContentFeedbackV1
    | AnswerFeedbackV1
    | PreferenceFeedbackV1
    | FactFeedbackV1
    | FactReviewV1
    | LockChangeV1
)


def _question_id(session_id: UUID, run_id: UUID, index: int, value: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"resume-question:{session_id}:{run_id}:{index}:{value}")


async def _current_preferences(
    db: AsyncSession, row: ResumeSession
) -> tuple[int, JobPreferenceOverrideV1, tuple[UUID, ...]]:
    latest = await db.scalar(
        select(ResumeSessionPreferenceVersion)
        .where(
            ResumeSessionPreferenceVersion.workspace_id == row.workspace_id,
            ResumeSessionPreferenceVersion.session_id == row.id,
        )
        .order_by(ResumeSessionPreferenceVersion.version.desc())
        .limit(1)
    )
    if latest is None:
        return 0, JobPreferenceOverrideV1.model_validate(row.override_json), ()
    value = latest.preferences_json
    return (
        latest.version,
        JobPreferenceOverrideV1.model_validate(value["override"]),
        tuple(UUID(item) for item in value["locked_item_ids"]),
    )


async def _effective_preferences(
    db: AsyncSession,
    tenant: TenantContext,
    row: ResumeSession,
    round_override: JobPreferenceOverrideV1 | None = None,
) -> ResumePreferencesV1:
    _, _, _, global_preferences = await _profile_inputs(
        db, tenant, row.profile_version_id, await _preference_number(db, row)
    )
    _, session_override, locks = await _current_preferences(db, row)
    override = session_override
    if round_override is not None:
        override = JobPreferenceOverrideV1.model_validate(
            {
                **session_override.model_dump(exclude_unset=True),
                **round_override.model_dump(exclude_unset=True),
            }
        )
    effective = effective_preferences(global_preferences, override)
    return effective.model_copy(
        update={"locked_item_ids": tuple(dict.fromkeys((*effective.locked_item_ids, *locks)))}
    )


async def _questions(
    db: AsyncSession, tenant: TenantContext, row: ResumeSession
) -> list[dict[str, object]]:
    run = await db.scalar(
        select(Run).where(Run.workspace_id == tenant.workspace_id, Run.id == row.latest_run_id)
    )
    if run is None:
        raise DomainInvariantError("session run is missing")
    latest_questions = (run.result_json or {}).get("payload", {}).get("questions", [])
    if run.id != row.run_id and latest_questions:
        values = latest_questions
        source_run_id = run.id
    elif row.current_version_id is not None:
        version = await db.scalar(
            select(ResumeVersion).where(
                ResumeVersion.workspace_id == tenant.workspace_id,
                ResumeVersion.id == row.current_version_id,
                ResumeVersion.session_id == row.id,
            )
        )
        if version is None:
            raise DomainInvariantError("current resume version is missing")
        values = version.validation_json.get("questions", [])
        if version.feedback_id is None:
            source_run_id = row.run_id
        else:
            source_run_id = await db.scalar(
                select(ResumeFeedback.run_id).where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == version.feedback_id,
                )
            )
            if source_run_id is None:
                raise DomainInvariantError("version feedback Run is missing")
    else:
        values = (run.result_json or {}).get("payload", {}).get("questions", [])
        source_run_id = run.id
    if not isinstance(values, list):
        raise DomainInvariantError("persisted resume questions are invalid")
    return [
        {
            "id": _question_id(row.id, source_run_id, index, value),
            "text": value,
            "source_run_id": source_run_id,
            "version_id": row.current_version_id,
        }
        for index, value in enumerate(values)
        if isinstance(value, str)
    ]


async def _bound_facts(
    db: AsyncSession, tenant: TenantContext, session_id: UUID
) -> tuple[set[UUID], set[UUID], tuple[tuple[UUID, str, str, dict[str, object]], ...]]:
    material_rows = (
        await db.execute(
            select(ResumeSessionFact.fact_version_id, MaterialFactVersion)
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
    user_rows = (
        await db.execute(
            select(ResumeSessionUserFact.fact_version_id, ResumeUserFactVersion)
            .join(
                ResumeUserFactVersion,
                (ResumeUserFactVersion.workspace_id == ResumeSessionUserFact.workspace_id)
                & (ResumeUserFactVersion.id == ResumeSessionUserFact.fact_version_id),
            )
            .where(
                ResumeSessionUserFact.workspace_id == tenant.workspace_id,
                ResumeSessionUserFact.session_id == session_id,
                ResumeUserFactVersion.review_status == "confirmed",
            )
        )
    ).all()
    active_users = [item for _, item in user_rows]
    superseded_material = set()
    superseded_users = set()
    if active_users:
        parents = (
            await db.scalars(
                select(ResumeUserFact).where(
                    ResumeUserFact.workspace_id == tenant.workspace_id,
                    ResumeUserFact.id.in_([item.fact_id for item in active_users]),
                )
            )
        ).all()
        superseded_material = {
            item.supersedes_material_version_id
            for item in parents
            if item.supersedes_material_version_id is not None
        }
        superseded_users = {
            item.supersedes_user_version_id
            for item in parents
            if item.supersedes_user_version_id is not None
        }
    material = {
        version_id for version_id, _ in material_rows if version_id not in superseded_material
    }
    user = {version_id for version_id, _ in user_rows if version_id not in superseded_users}
    claims = tuple(
        [
            (version_id, fact.claim, fact.kind, fact.conditions_json)
            for version_id, fact in material_rows
            if version_id in material
        ]
        + [
            (version_id, fact.claim, fact.kind, fact.conditions_json)
            for version_id, fact in user_rows
            if version_id in user
        ]
    )
    return material, user, claims


class _FeedbackWriter(ResumeCommandWriter):
    supported_kinds = frozenset(
        {
            "resume_feedback_content",
            "resume_feedback_answer",
            "resume_feedback_preference",
            "resume_feedback_fact",
            "resume_feedback_fact_review",
            "resume_feedback_lock",
        }
    )

    async def authorize_and_lock(
        self, db: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ) -> SessionWriteState:
        if request.target_id is None:
            raise DomainValidationError("feedback session is required")
        row = await _session(db, tenant, request.target_id, write=True)
        if tenant.role is WorkspaceRole.REVIEWER:
            raise DomainNotFoundError
        latest = await db.scalar(
            select(Run.status).where(
                Run.workspace_id == tenant.workspace_id, Run.id == row.latest_run_id
            )
        )
        if latest is None:
            raise DomainInvariantError("session latest run is missing")
        return SessionWriteState(
            revision=row.revision,
            current_version_id=row.current_version_id,
            has_active_modification=latest in {"queued", "running"},
        )

    async def apply(
        self,
        db: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        assert request.target_id is not None
        row = await _session(db, tenant, request.target_id, write=True)
        payload: FeedbackPayload = request.payload  # type: ignore[assignment]
        feedback_id = uuid4()
        kind = (
            "fact_review"
            if isinstance(payload, FactReviewV1)
            else "lock"
            if isinstance(payload, LockChangeV1)
            else payload.kind
        )
        scope = (
            payload.scope
            if isinstance(payload, PreferenceFeedbackV1)
            else payload.fact.scope
            if isinstance(payload, FactFeedbackV1) and payload.fact is not None
            else "session"
        )
        normalized: dict[str, object] = {}
        run_id: UUID | None = None
        if isinstance(payload, ContentFeedbackV1):
            if row.current_version_id is None:
                raise DomainConflictError("a draft version is required")
            content = await db.scalar(
                select(ResumeVersion.content_json).where(
                    ResumeVersion.workspace_id == tenant.workspace_id,
                    ResumeVersion.id == row.current_version_id,
                    ResumeVersion.session_id == row.id,
                )
            )
            if content is None:
                raise DomainNotFoundError
            if (
                not set(payload.target_item_ids)
                <= ResumeContentV1.model_validate(content).item_ids()
            ):
                raise DomainValidationError("feedback targets are not in the current version")
            run_id = await self._enqueue(db, tenant, row, feedback_id, payload)
        elif isinstance(payload, AnswerFeedbackV1):
            if payload.question_id not in {
                item["id"] for item in await _questions(db, tenant, row)
            }:
                raise DomainConflictError("question is not current")
            run_id = await self._enqueue(db, tenant, row, feedback_id, payload)
        elif isinstance(payload, PreferenceFeedbackV1):
            if payload.scope == "global":
                assert isinstance(payload.preferences, ResumePreferencesV1)
                profile_version, _, _, _ = await _profile_inputs(
                    db, tenant, row.profile_version_id, await _preference_number(db, row)
                )
                profile = await db.scalar(
                    select(ResumeProfile)
                    .where(
                        ResumeProfile.workspace_id == tenant.workspace_id,
                        ResumeProfile.id == profile_version.profile_id,
                    )
                    .with_for_update()
                )
                if (
                    profile is None
                    or profile.current_preference_version
                    != payload.expected_global_preference_version
                ):
                    raise DomainConflictError("global preference version changed")
                profile.current_preference_version += 1
                db.add(
                    ResumePreferenceVersion(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        profile_id=profile.id,
                        version=profile.current_preference_version,
                        preferences_json=payload.preferences.model_dump(mode="json"),
                        created_by_user_id=tenant.actor_user_id,
                    )
                )
                normalized = {"global_preference_version": profile.current_preference_version}
            else:
                assert isinstance(payload.preferences, JobPreferenceOverrideV1)
                if payload.scope == "session":
                    _, _, locks = await _current_preferences(db, row)
                    await self._save_preferences(
                        db,
                        tenant,
                        row,
                        payload.preferences,
                        locks,
                    )
                if row.current_version_id is not None:
                    run_id = await self._enqueue(db, tenant, row, feedback_id, payload)
                else:
                    normalized = {"needs_draft": True}
        elif isinstance(payload, FactFeedbackV1):
            normalized = await self._fact(db, tenant, row, payload)
        elif isinstance(payload, FactReviewV1):
            normalized = await self._fact_review(db, tenant, row, payload)
        else:
            assert isinstance(payload, LockChangeV1)
            normalized = await self._lock(db, tenant, row, payload)
        db.add(
            ResumeFeedback(
                id=feedback_id,
                workspace_id=tenant.workspace_id,
                session_id=row.id,
                command_id=command_id,
                run_id=run_id,
                base_version_id=payload.base_version_id,
                target_version_id=None,
                kind=kind,
                scope=scope,
                request_json=payload.model_dump(mode="json", round_trip=True),
                normalized_json=normalized,
                questions_json=[],
                repair_count=0,
            )
        )
        row.revision += 1
        return CommandReceiptV1(
            command_id=command_id,
            run_id=run_id,
            resource_id=feedback_id,
            status="queued" if run_id is not None else "completed",
        )

    @staticmethod
    async def _save_preferences(
        db: AsyncSession,
        tenant: TenantContext,
        row: ResumeSession,
        override: JobPreferenceOverrideV1,
        locks: tuple[UUID, ...],
    ) -> None:
        current, _, _ = await _current_preferences(db, row)
        db.add(
            ResumeSessionPreferenceVersion(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                session_id=row.id,
                version=current + 1,
                preferences_json={
                    "override": override.model_dump(mode="json", exclude_unset=True),
                    "locked_item_ids": [str(value) for value in locks],
                },
            )
        )

    async def _lock(
        self, db: AsyncSession, tenant: TenantContext, row: ResumeSession, payload: LockChangeV1
    ) -> dict[str, object]:
        version = await db.scalar(
            select(ResumeVersion).where(
                ResumeVersion.workspace_id == tenant.workspace_id,
                ResumeVersion.id == row.current_version_id,
                ResumeVersion.session_id == row.id,
            )
        )
        if version is None:
            raise DomainNotFoundError
        content = ResumeContentV1.model_validate(version.content_json)
        if payload.item_id not in content.item_ids():
            raise DomainValidationError("lock target is not in this version")
        if payload.item_id == content.display_name_id or payload.item_id in {
            value.id for value in content.contact
        }:
            raise DomainValidationError("personal fields remain protected")
        _, override, locks = await _current_preferences(db, row)
        updated = set(locks)
        if payload.locked:
            updated.add(payload.item_id)
        else:
            updated.discard(payload.item_id)
        await self._save_preferences(db, tenant, row, override, tuple(sorted(updated)))
        return {"item_id": str(payload.item_id), "locked": payload.locked}

    async def _fact(
        self, db: AsyncSession, tenant: TenantContext, row: ResumeSession, payload: FactFeedbackV1
    ) -> dict[str, object]:
        if payload.adopt_material_version_id is not None:
            version = await db.scalar(
                select(MaterialFactVersion).where(
                    MaterialFactVersion.workspace_id == tenant.workspace_id,
                    MaterialFactVersion.id == payload.adopt_material_version_id,
                    MaterialFactVersion.review_status == "confirmed",
                )
            )
            if version is None:
                raise DomainNotFoundError
            fact = await db.scalar(
                select(MaterialFact).where(
                    MaterialFact.workspace_id == tenant.workspace_id,
                    MaterialFact.id == version.fact_id,
                    MaterialFact.current_version == version.version,
                )
            )
            if fact is None:
                raise DomainConflictError("fact is no longer current")
            existing = await db.scalar(
                select(ResumeSessionFact.id).where(
                    ResumeSessionFact.workspace_id == tenant.workspace_id,
                    ResumeSessionFact.session_id == row.id,
                    ResumeSessionFact.fact_version_id == version.id,
                )
            )
            if existing is None:
                project_id = await db.scalar(
                    select(ResumeSessionProject.project_id)
                    .join(
                        MaterialFactSet,
                        (MaterialFactSet.workspace_id == ResumeSessionProject.workspace_id)
                        & (MaterialFactSet.project_id == ResumeSessionProject.project_id),
                    )
                    .where(
                        ResumeSessionProject.workspace_id == tenant.workspace_id,
                        ResumeSessionProject.session_id == row.id,
                        MaterialFactSet.id == fact.fact_set_id,
                    )
                )
                if project_id is None:
                    raise DomainNotFoundError
                db.add(
                    ResumeSessionFact(
                        id=uuid4(),
                        workspace_id=tenant.workspace_id,
                        session_id=row.id,
                        project_id=project_id,
                        fact_version_id=version.id,
                    )
                )
            return {"adopted_material_version_id": str(version.id)}
        fact_input = payload.fact
        assert fact_input is not None
        allowed_project = await db.scalar(
            select(ResumeSessionProject.id).where(
                ResumeSessionProject.workspace_id == tenant.workspace_id,
                ResumeSessionProject.session_id == row.id,
                ResumeSessionProject.project_id == fact_input.project_id,
            )
        )
        if allowed_project is None:
            raise DomainNotFoundError
        if fact_input.supersedes_material_version_id is not None:
            bound = await db.scalar(
                select(ResumeSessionFact.id).where(
                    ResumeSessionFact.workspace_id == tenant.workspace_id,
                    ResumeSessionFact.session_id == row.id,
                    ResumeSessionFact.fact_version_id == fact_input.supersedes_material_version_id,
                )
            )
            if bound is None:
                raise DomainNotFoundError
        if fact_input.supersedes_user_version_id is not None:
            bound = await db.scalar(
                select(ResumeSessionUserFact.id).where(
                    ResumeSessionUserFact.workspace_id == tenant.workspace_id,
                    ResumeSessionUserFact.session_id == row.id,
                    ResumeSessionUserFact.fact_version_id == fact_input.supersedes_user_version_id,
                )
            )
            if bound is None:
                raise DomainNotFoundError
        fact_id, version_id = uuid4(), uuid4()
        db.add(
            ResumeUserFact(
                id=fact_id,
                workspace_id=tenant.workspace_id,
                project_id=fact_input.project_id,
                scope_session_id=row.id if fact_input.scope == "session" else None,
                current_version=1,
                supersedes_material_version_id=fact_input.supersedes_material_version_id,
                supersedes_user_version_id=fact_input.supersedes_user_version_id,
            )
        )
        await db.flush()
        db.add(
            ResumeUserFactVersion(
                id=version_id,
                workspace_id=tenant.workspace_id,
                fact_id=fact_id,
                version=1,
                claim=fact_input.claim,
                kind=fact_input.kind,
                conditions_json={
                    "environment": fact_input.environment,
                    "scope": fact_input.fact_scope,
                    "metric_basis": fact_input.metric_basis,
                    "source": "user_attestation",
                },
                review_status="pending",
                created_by_user_id=tenant.actor_user_id,
                attested_at=None,
            )
        )
        return {
            "fact_id": str(fact_id),
            "fact_version_id": str(version_id),
            "review_status": "pending",
            "issues": list(user_fact_issues(fact_input)),
        }

    async def _fact_review(
        self, db: AsyncSession, tenant: TenantContext, row: ResumeSession, payload: FactReviewV1
    ) -> dict[str, object]:
        previous = await db.scalar(
            select(ResumeUserFactVersion).where(
                ResumeUserFactVersion.workspace_id == tenant.workspace_id,
                ResumeUserFactVersion.id == payload.fact_version_id,
            )
        )
        if previous is None:
            raise DomainNotFoundError
        fact = await db.scalar(
            select(ResumeUserFact)
            .where(
                ResumeUserFact.workspace_id == tenant.workspace_id,
                ResumeUserFact.id == previous.fact_id,
            )
            .with_for_update()
        )
        if (
            fact is None
            or (fact.scope_session_id not in (None, row.id))
            or fact.current_version != previous.version
            or previous.review_status != "pending"
        ):
            raise DomainConflictError("fact review is stale")
        project = await db.scalar(
            select(ResumeSessionProject.id).where(
                ResumeSessionProject.workspace_id == tenant.workspace_id,
                ResumeSessionProject.session_id == row.id,
                ResumeSessionProject.project_id == fact.project_id,
            )
        )
        if project is None:
            raise DomainNotFoundError
        conditions = previous.conditions_json
        if payload.decision == "confirm":
            if not payload.attested:
                raise DomainConflictError("explicit user verification is required")
            original = FactFeedbackV1(
                expected_session_revision=payload.expected_session_revision,
                base_version_id=payload.base_version_id,
                fact={
                    "project_id": fact.project_id,
                    "scope": "session" if fact.scope_session_id else "project",
                    "claim": previous.claim,
                    "kind": previous.kind,
                    "environment": conditions.get("environment"),
                    "fact_scope": conditions.get("scope"),
                    "metric_basis": conditions.get("metric_basis"),
                },
            ).fact
            assert original is not None
            if user_fact_issues(original):
                raise DomainConflictError("fact still has verification questions")
        fact.current_version += 1
        version_id = uuid4()
        db.add(
            ResumeUserFactVersion(
                id=version_id,
                workspace_id=tenant.workspace_id,
                fact_id=fact.id,
                version=fact.current_version,
                claim=previous.claim,
                kind=previous.kind,
                conditions_json=conditions,
                review_status="confirmed" if payload.decision == "confirm" else "rejected",
                created_by_user_id=tenant.actor_user_id,
                attested_at=datetime.now(UTC) if payload.decision == "confirm" else None,
            )
        )
        if payload.decision == "confirm":
            await db.flush()
            db.add(
                ResumeSessionUserFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    session_id=row.id,
                    fact_version_id=version_id,
                )
            )
        return {
            "fact_id": str(fact.id),
            "fact_version_id": str(version_id),
            "review_status": "confirmed" if payload.decision == "confirm" else "rejected",
        }

    async def _enqueue(
        self,
        db: AsyncSession,
        tenant: TenantContext,
        row: ResumeSession,
        feedback_id: UUID,
        payload: ContentFeedbackV1 | AnswerFeedbackV1 | PreferenceFeedbackV1,
    ) -> UUID:
        parent = await db.scalar(
            select(Run).where(Run.workspace_id == tenant.workspace_id, Run.id == row.run_id)
        )
        if parent is None:
            raise DomainInvariantError("initial Run is missing")
        run_id, message_id = uuid4(), uuid4()
        budget = {
            "max_model_calls": payload.max_model_calls,
            "max_tool_calls": payload.max_tool_calls,
            "max_tool_results": payload.max_tool_calls,
            "max_iterations": 16,
        }
        db.add(
            Message(
                id=message_id,
                workspace_id=tenant.workspace_id,
                conversation_id=parent.conversation_id,
                actor_user_id=tenant.actor_user_id,
                role="user",
                content="Resume revision feedback",
            )
        )
        db.add(
            Run(
                id=run_id,
                workspace_id=tenant.workspace_id,
                created_by_user_id=tenant.actor_user_id,
                conversation_id=parent.conversation_id,
                request_message_id=message_id,
                mode=RunMode.RESUME_REVISION.value,
                resume_document_id=None,
                input_json=ResumeRevisionRunInputV1(
                    payload=ResumeRevisionInputV1(
                        session_id=row.id,
                        feedback_id=feedback_id,
                        base_version_id=payload.base_version_id,
                    )
                ).model_dump(mode="json", round_trip=True),
                limits_json={"schema_version": 1, **budget},
                status="queued",
                graph_version="pathfinder-resume-v4",
            )
        )
        await db.flush()
        db.add(
            RunJob(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                originating_actor_user_id=tenant.actor_user_id,
                run_id=run_id,
                status="queued",
            )
        )
        sequence = await db.scalar(
            update(Run)
            .where(Run.workspace_id == tenant.workspace_id, Run.id == run_id)
            .values(next_event_seq=Run.next_event_seq + 1)
            .returning(Run.next_event_seq - 1)
        )
        if sequence != 1:
            raise DomainInvariantError("revision run event sequence is invalid")
        db.add(
            RunEvent(
                id=uuid4(),
                workspace_id=tenant.workspace_id,
                run_id=run_id,
                actor_user_id=tenant.actor_user_id,
                seq=1,
                type="run.created",
                version=1,
                payload={
                    "mode": "resume_revision",
                    "status": "queued",
                    "graph_version": "pathfinder-resume-v4",
                },
            )
        )
        row.latest_run_id = run_id
        return run_id


class SqlAlchemyResumeRevisionStore:
    def __init__(self, sessions: AsyncSessionFactory) -> None:
        self.sessions = sessions
        self.commands = SqlAlchemyResumeCommandStore(sessions)

    async def command(
        self, tenant: TenantContext, session_id: UUID, payload: FeedbackPayload, key: UUID
    ):
        kind = (
            "fact_review"
            if isinstance(payload, FactReviewV1)
            else "lock"
            if isinstance(payload, LockChangeV1)
            else payload.kind
        )
        return await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=key,
                kind=f"resume_feedback_{kind}",
                target_id=session_id,
                payload_version=1,
                payload=payload,
                session_write=SessionWritePrecondition(
                    expected_session_revision=payload.expected_session_revision,
                    base_version_id=payload.base_version_id,
                ),
            ),
            writer=_FeedbackWriter(),
        )

    async def list_feedback(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            await _session(db, tenant, session_id)
            rows = (
                await db.scalars(
                    select(ResumeFeedback)
                    .where(
                        ResumeFeedback.workspace_id == tenant.workspace_id,
                        ResumeFeedback.session_id == session_id,
                    )
                    .order_by(ResumeFeedback.created_at, ResumeFeedback.id)
                )
            ).all()
            result = []
            for item in rows:
                run_status = (
                    await db.scalar(
                        select(Run.status).where(
                            Run.workspace_id == tenant.workspace_id,
                            Run.id == item.run_id,
                        )
                    )
                    if item.run_id
                    else None
                )
                result.append(
                    {
                        "feedback_id": item.id,
                        "kind": item.kind,
                        "scope": item.scope,
                        "run_id": item.run_id,
                        "run_status": run_status,
                        "base_version_id": item.base_version_id,
                        "target_version_id": item.target_version_id,
                        "normalized": item.normalized_json,
                        "questions": item.questions_json,
                        "created_at": item.created_at,
                    }
                )
            return result

    async def list_user_facts(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            project_ids = (
                await db.scalars(
                    select(ResumeSessionProject.project_id).where(
                        ResumeSessionProject.workspace_id == tenant.workspace_id,
                        ResumeSessionProject.session_id == row.id,
                    )
                )
            ).all()
            if not project_ids:
                return []
            values = (
                await db.execute(
                    select(ResumeUserFact, ResumeUserFactVersion)
                    .join(
                        ResumeUserFactVersion,
                        (ResumeUserFactVersion.workspace_id == ResumeUserFact.workspace_id)
                        & (ResumeUserFactVersion.fact_id == ResumeUserFact.id)
                        & (ResumeUserFactVersion.version == ResumeUserFact.current_version),
                    )
                    .where(
                        ResumeUserFact.workspace_id == tenant.workspace_id,
                        ResumeUserFact.project_id.in_(project_ids),
                        (ResumeUserFact.scope_session_id == row.id)
                        | ResumeUserFact.scope_session_id.is_(None),
                    )
                    .order_by(ResumeUserFactVersion.created_at, ResumeUserFact.id)
                )
            ).all()
            return [
                {
                    "fact_id": fact.id,
                    "fact_version_id": version.id,
                    "project_id": fact.project_id,
                    "scope": "session" if fact.scope_session_id else "project",
                    "claim": version.claim,
                    "kind": version.kind,
                    "conditions": version.conditions_json,
                    "review_status": version.review_status,
                    "source": "user_attestation",
                    "attested_at": version.attested_at,
                }
                for fact, version in values
            ]

    async def questions(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            row = await _session(db, tenant, session_id)
            return await _questions(db, tenant, row)

    async def revision_inputs(self, tenant: TenantContext, feedback_id: UUID) -> RevisionInputs:
        async with database_session(self.sessions) as db:
            feedback = await db.scalar(
                select(ResumeFeedback).where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == feedback_id,
                )
            )
            if feedback is None or feedback.run_id is None:
                raise DomainNotFoundError
            row = await _session(db, tenant, feedback.session_id)
            if (
                row.latest_run_id != feedback.run_id
                or row.current_version_id != feedback.base_version_id
            ):
                raise DomainConflictError("revision base changed")
            schema = {
                "content": ContentFeedbackV1,
                "answer": AnswerFeedbackV1,
                "preference": PreferenceFeedbackV1,
            }.get(feedback.kind)
            if schema is None:
                raise DomainInvariantError("feedback kind has no revision Run")
            request = schema.model_validate_json(json.dumps(feedback.request_json))
            base = (
                await db.scalar(
                    select(ResumeVersion).where(
                        ResumeVersion.workspace_id == tenant.workspace_id,
                        ResumeVersion.id == feedback.base_version_id,
                        ResumeVersion.session_id == row.id,
                    )
                )
                if feedback.base_version_id
                else None
            )
            _, _, profile_content, _ = await _profile_inputs(
                db, tenant, row.profile_version_id, await _preference_number(db, row)
            )
            round_override = (
                request.preferences
                if isinstance(request, PreferenceFeedbackV1) and request.scope == "round"
                else None
            )
            preferences = await _effective_preferences(db, tenant, row, round_override)
            material, user, claims = await _bound_facts(db, tenant, row.id)
            return RevisionInputs(
                session_id=row.id,
                run_id=feedback.run_id,
                feedback_id=feedback.id,
                base_version_id=feedback.base_version_id,
                base_content=ResumeContentV1.model_validate(base.content_json) if base else None,
                profile_content=profile_content,
                preferences=preferences,
                target_item_ids=(
                    request.target_item_ids if isinstance(request, ContentFeedbackV1) else ()
                ),
                request=request,
                permitted_fact_ids=frozenset(material | user),
                fact_claims=claims,
                instruction=request.instruction if isinstance(request, ContentFeedbackV1) else None,
            )

    async def spend_allowed(self, tenant: TenantContext, feedback_id: UUID) -> bool:
        async with database_session(self.sessions) as db:
            feedback = await db.scalar(
                select(ResumeFeedback).where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == feedback_id,
                )
            )
            if feedback is None or feedback.run_id is None:
                raise DomainNotFoundError
            await _session(db, tenant, feedback.session_id)
            request = feedback.request_json
            calls = int(request["max_model_calls"])
            max_cost = Decimal(str(request["max_cost_cny"]))
            attempts = (
                await db.execute(
                    select(LLMInvocation.provider, LLMInvocation.estimated_cost).where(
                        LLMInvocation.workspace_id == tenant.workspace_id,
                        LLMInvocation.run_id == feedback.run_id,
                    )
                )
            ).all()
            if any(cost is None and provider != "fake" for provider, cost in attempts):
                return False
            return (
                len(attempts) < calls
                and sum((Decimal(cost) for _, cost in attempts if cost is not None), Decimal(0))
                < max_cost
            )

    async def reserve_repair(self, tenant: TenantContext, feedback_id: UUID) -> bool:
        async with self.sessions.begin() as db:
            feedback = await db.scalar(
                select(ResumeFeedback).where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == feedback_id,
                )
            )
            if feedback is None:
                raise DomainNotFoundError
            await _session(db, tenant, feedback.session_id, write=True)
            changed = await db.scalar(
                update(ResumeFeedback)
                .where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == feedback_id,
                    ResumeFeedback.repair_count == 0,
                    ResumeFeedback.target_version_id.is_(None),
                )
                .values(repair_count=1)
                .returning(ResumeFeedback.id)
            )
            return changed is not None


class ResumeRevisionPublisher:
    def __init__(self, artifacts: SqlAlchemyResumeArtifactStore) -> None:
        self.artifacts = artifacts

    async def publish(
        self,
        db: AsyncSession,
        tenant: TenantContext,
        run_id: UUID,
        output: ResumeRevisionCandidateOutputV1,
    ) -> ResumeRevisionRunOutputV1:
        feedback = await db.scalar(
            select(ResumeFeedback)
            .where(
                ResumeFeedback.workspace_id == tenant.workspace_id,
                ResumeFeedback.run_id == run_id,
            )
            .with_for_update()
        )
        if feedback is None:
            raise DomainInvariantError("revision feedback is missing")
        row = await _session(db, tenant, feedback.session_id, write=True)
        if feedback.target_version_id is not None:
            version = await db.scalar(
                select(ResumeVersion).where(
                    ResumeVersion.workspace_id == tenant.workspace_id,
                    ResumeVersion.id == feedback.target_version_id,
                )
            )
            if version is None:
                raise DomainInvariantError("published version is missing")
            return ResumeRevisionRunOutputV1(
                payload=ResumeRevisionResultV1(
                    session_id=row.id,
                    feedback_id=feedback.id,
                    version_id=version.id,
                    artifact_id=version.artifact_id,
                    outcome="draft",
                    questions=tuple(feedback.questions_json),
                )
            )
        if row.current_version_id != feedback.base_version_id or row.latest_run_id != run_id:
            raise DomainConflictError("revision base changed before publication")
        candidate: RevisionCandidateV1 = output.payload
        feedback.normalized_json = {
            "patches": [item.model_dump(mode="json") for item in candidate.patches],
            "impact": list(candidate.impact),
        }
        feedback.questions_json = list(candidate.questions)
        if candidate.content is None:
            return ResumeRevisionRunOutputV1(
                payload=ResumeRevisionResultV1(
                    session_id=row.id,
                    feedback_id=feedback.id,
                    version_id=None,
                    artifact_id=None,
                    outcome="needs_input",
                    questions=candidate.questions,
                )
            )
        base = await db.scalar(
            select(ResumeVersion).where(
                ResumeVersion.workspace_id == tenant.workspace_id,
                ResumeVersion.id == feedback.base_version_id,
                ResumeVersion.session_id == row.id,
            )
        )
        if base is None:
            raise DomainInvariantError("revision base version is missing")
        _, _, profile_content, _ = await _profile_inputs(
            db, tenant, row.profile_version_id, await _preference_number(db, row)
        )
        request = (
            ContentFeedbackV1.model_validate_json(json.dumps(feedback.request_json))
            if feedback.kind == "content"
            else PreferenceFeedbackV1.model_validate_json(json.dumps(feedback.request_json))
        )
        round_override = (
            request.preferences
            if isinstance(request, PreferenceFeedbackV1) and request.scope == "round"
            else None
        )
        preferences = await _effective_preferences(db, tenant, row, round_override)
        material, user, claims = await _bound_facts(db, tenant, row.id)
        base_content = ResumeContentV1.model_validate(base.content_json)
        if isinstance(request, ContentFeedbackV1):
            expected, diff = apply_scoped_patches(
                base_content,
                candidate.patches,
                request.target_item_ids,
                preferences,
                frozenset(material | user),
                {key: (claim, kind) for key, claim, kind, _ in claims},
            )
            if (
                expected != candidate.content
                or tuple(candidate.diff) != diff.changes
                or tuple(candidate.impact) != diff.impact
            ):
                raise DomainValidationError("revision candidate differs from scoped patch")
        elif candidate.content != base_content or candidate.patches:
            raise DomainValidationError("preference revision changed content without a patch")
        require_locked_items_unchanged(profile_content, candidate.content, preferences)
        if hard_constraint_issues(candidate.content, preferences):
            raise DomainValidationError("revision violates hard constraints")
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
                version=base.version + 1,
                artifact_id=artifact_id,
                parent_version_id=base.id,
                feedback_id=feedback.id,
                content_json=candidate.content.model_dump(mode="json"),
                validation_json={
                    "questions": list(candidate.questions),
                    "correction_count": candidate.correction_count,
                    "prompt_version": candidate.prompt_version,
                    "model_id": candidate.model_id,
                    "retrieval_config_version": candidate.retrieval_config_version,
                    "semantic_support": "needs_human_review",
                },
                diff_json=list(candidate.diff),
                impact_json=list(candidate.impact),
            )
        )
        await db.flush()
        db.add_all(
            [
                ResumeVersionFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    version_id=version_id,
                    material_fact_version_id=value,
                    user_fact_version_id=None,
                )
                for value in material
            ]
            + [
                ResumeVersionFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    version_id=version_id,
                    material_fact_version_id=None,
                    user_fact_version_id=value,
                )
                for value in user
            ]
        )
        old_coverage = (
            await db.scalars(
                select(RequirementCoverage).where(
                    RequirementCoverage.workspace_id == tenant.workspace_id,
                    RequirementCoverage.version_id == base.id,
                )
            )
        ).all()
        changed_fact_refs = {
            patch.item_id: set(patch.fact_version_ids)
            for patch in candidate.patches
            if patch.operation in {"replace_text", "replace_items"}
        }
        for previous in old_coverage:
            kept_items = [
                value
                for value in previous.item_ids_json
                if UUID(value) in candidate.content.item_ids()
            ]
            changed_refs = set().union(
                *(changed_fact_refs.get(UUID(value), set()) for value in kept_items)
            )
            changed_coverage = any(UUID(value) in changed_fact_refs for value in kept_items)
            coverage_id = uuid4()
            db.add(
                RequirementCoverage(
                    id=coverage_id,
                    workspace_id=tenant.workspace_id,
                    version_id=version_id,
                    requirement_id=previous.requirement_id,
                    support=(
                        "partial"
                        if changed_coverage and changed_refs
                        else "no_support_found"
                        if changed_coverage or not kept_items
                        else previous.support
                    ),
                    verification="needs_human_review" if kept_items else "unchecked",
                    reason=(
                        "Revised wording; cited facts and JD coverage need human review."
                        if changed_coverage
                        else "Revised content; fact support needs human review."
                        if kept_items
                        else "Referenced content was removed in this revision."
                    ),
                    item_ids_json=kept_items,
                )
            )
            await db.flush()
            prior_material = (
                await db.scalars(
                    select(RequirementCoverageFact.fact_version_id).where(
                        RequirementCoverageFact.workspace_id == tenant.workspace_id,
                        RequirementCoverageFact.coverage_id == previous.id,
                    )
                )
            ).all()
            prior_user = (
                await db.scalars(
                    select(RequirementCoverageUserFact.fact_version_id).where(
                        RequirementCoverageUserFact.workspace_id == tenant.workspace_id,
                        RequirementCoverageUserFact.coverage_id == previous.id,
                    )
                )
            ).all()
            material_refs = (
                changed_refs & material if changed_coverage else set(prior_material) & material
            )
            user_refs = changed_refs & user if changed_coverage else set(prior_user) & user
            db.add_all(
                RequirementCoverageFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    coverage_id=coverage_id,
                    fact_version_id=value,
                )
                for value in material_refs
            )
            db.add_all(
                RequirementCoverageUserFact(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    coverage_id=coverage_id,
                    fact_version_id=value,
                )
                for value in user_refs
            )
        feedback.target_version_id = version_id
        row.current_version_id = version_id
        row.revision += 1
        return ResumeRevisionRunOutputV1(
            payload=ResumeRevisionResultV1(
                session_id=row.id,
                feedback_id=feedback.id,
                version_id=version_id,
                artifact_id=artifact_id,
                outcome="draft",
                questions=candidate.questions,
            )
        )
