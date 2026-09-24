"""Authorized version confirmation, history and byte delivery."""

from __future__ import annotations

from uuid import UUID, uuid4, uuid5

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    MaterialFactVersion,
    ResumeConfirmation,
    ResumeFeedback,
    ResumeTexArtifact,
    ResumeUserFact,
    ResumeUserFactVersion,
    ResumeVersion,
    ResumeVersionFact,
    Run,
)
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.resume_generation import _session
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import DomainConflictError, DomainInvariantError, DomainNotFoundError
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import (
    CommandReceiptV1,
    ResumeCommandRequest,
    SessionWritePrecondition,
    SessionWriteState,
)
from app.domain.resume_confirmation import ConfirmationV1, ConfirmVersionV1
from app.domain.resume_profile import ResumeContentV1
from app.domain.tenancy import TenantContext


def _confirmation(row: ResumeConfirmation) -> ConfirmationV1:
    return ConfirmationV1(
        confirmation_id=row.id,
        session_id=row.session_id,
        version_id=row.version_id,
        artifact_id=row.artifact_id,
        tex_sha256=row.tex_sha256,
        confirmed_by_user_id=row.confirmed_by_user_id,
        confirmed_at=row.confirmed_at,
    )


async def _version(
    db: AsyncSession, tenant: TenantContext, session_id: UUID, version_id: UUID
) -> ResumeVersion:
    row = await db.scalar(
        select(ResumeVersion).where(
            ResumeVersion.workspace_id == tenant.workspace_id,
            ResumeVersion.session_id == session_id,
            ResumeVersion.id == version_id,
        )
    )
    if row is None:
        raise DomainNotFoundError
    return row


async def _actual_fact_uses(
    db: AsyncSession, tenant: TenantContext, row: ResumeVersion
) -> set[UUID]:
    """Follow immutable content and patch lineage, not the broad fact snapshot."""
    lineage = [row]
    while lineage[-1].parent_version_id is not None:
        parent = await _version(db, tenant, row.session_id, lineage[-1].parent_version_id)
        if parent.version != lineage[-1].version - 1:
            raise DomainConflictError("version fact lineage is incomplete")
        lineage.append(parent)
    lineage.reverse()
    uses: dict[tuple[UUID, str], set[UUID]] = {}
    for index, version in enumerate(lineage):
        content = ResumeContentV1.model_validate(version.content_json)
        item_ids = content.item_ids()
        snapshot = {
            value
            for material, user in (
                await db.execute(
                    select(
                        ResumeVersionFact.material_fact_version_id,
                        ResumeVersionFact.user_fact_version_id,
                    ).where(
                        ResumeVersionFact.workspace_id == tenant.workspace_id,
                        ResumeVersionFact.version_id == version.id,
                    )
                )
            ).all()
            if (value := material or user) is not None
        }
        if index == 0:
            if version.version != 1:
                raise DomainConflictError("version fact lineage is incomplete")
            for project in content.projects:
                for bullet_id in project.bullet_ids:
                    matches = {
                        fact_id
                        for fact_id in snapshot
                        if uuid5(row.session_id, str(fact_id)) == bullet_id
                    }
                    if len(matches) != 1:
                        raise DomainConflictError("version fact references cannot be resolved")
                    uses[(bullet_id, "bullet")] = matches
                if project.bullet_ids:
                    uses[(project.id, "summary")] = uses[(project.bullet_ids[0], "bullet")]
        else:
            uses = {key: facts for key, facts in uses.items() if key[0] in item_ids}
            if version.feedback_id is None:
                raise DomainConflictError("version fact lineage is incomplete")
            feedback = await db.scalar(
                select(ResumeFeedback).where(
                    ResumeFeedback.workspace_id == tenant.workspace_id,
                    ResumeFeedback.id == version.feedback_id,
                    ResumeFeedback.session_id == row.session_id,
                )
            )
            if feedback is None:
                raise DomainConflictError("version feedback is missing")
            patches = feedback.normalized_json.get("patches")
            if not isinstance(patches, list):
                raise DomainConflictError("version patch references cannot be resolved")
            for patch in patches:
                if not isinstance(patch, dict):
                    raise DomainConflictError("version patch references cannot be resolved")
                operation = patch.get("operation")
                try:
                    item_id = UUID(str(patch["item_id"]))
                except (KeyError, TypeError, ValueError):
                    raise DomainConflictError(
                        "version patch references cannot be resolved"
                    ) from None
                if operation == "remove":
                    uses = {key: value for key, value in uses.items() if key[0] != item_id}
                elif operation in {"replace_text", "replace_items"}:
                    field = patch.get("field")
                    values = patch.get("fact_version_ids")
                    if not isinstance(field, str) or not isinstance(values, list) or not values:
                        raise DomainConflictError("version patch references cannot be resolved")
                    try:
                        cited = {UUID(str(value)) for value in values}
                    except (TypeError, ValueError):
                        raise DomainConflictError(
                            "version patch references cannot be resolved"
                        ) from None
                    uses[(item_id, field)] = cited
        if uses and not set().union(*uses.values()) <= snapshot:
            raise DomainConflictError("version fact snapshot omits a used fact")
    return set().union(*uses.values()) if uses else set()


async def _require_confirmable(
    db: AsyncSession, tenant: TenantContext, version: ResumeVersion
) -> None:
    questions = version.validation_json.get("questions")
    if not isinstance(questions, list) or questions:
        raise DomainConflictError("selected version has unresolved validation questions")
    used = await _actual_fact_uses(db, tenant, version)
    if not used:
        raise DomainConflictError("selected version has no resolvable fact support")
    material = {
        value.id: value.review_status
        for value in (
            await db.scalars(
                select(MaterialFactVersion).where(
                    MaterialFactVersion.workspace_id == tenant.workspace_id,
                    MaterialFactVersion.id.in_(used),
                )
            )
        ).all()
    }
    user = {
        value.id: value.review_status
        for value in (
            await db.scalars(
                select(ResumeUserFactVersion).where(
                    ResumeUserFactVersion.workspace_id == tenant.workspace_id,
                    ResumeUserFactVersion.id.in_(used),
                )
            )
        ).all()
    }
    if set(material) | set(user) != used or any(
        status != "confirmed" for status in (*material.values(), *user.values())
    ):
        raise DomainConflictError("selected version uses an unconfirmed fact")
    corrections = (
        await db.scalars(
            select(ResumeUserFact).where(
                ResumeUserFact.workspace_id == tenant.workspace_id,
                (
                    ResumeUserFact.supersedes_material_version_id.in_(used)
                    | ResumeUserFact.supersedes_user_version_id.in_(used)
                ),
            )
        )
    ).all()
    for correction in corrections:
        status = await db.scalar(
            select(ResumeUserFactVersion.review_status).where(
                ResumeUserFactVersion.workspace_id == tenant.workspace_id,
                ResumeUserFactVersion.fact_id == correction.id,
                ResumeUserFactVersion.version == correction.current_version,
            )
        )
        if status in {"pending", "confirmed"}:
            raise DomainConflictError("selected version cites a fact under correction")


class _ConfirmationWriter(ResumeCommandWriter):
    supported_kinds = frozenset({"resume_confirm_version"})

    def __init__(self, artifacts: SqlAlchemyResumeArtifactStore) -> None:
        self.artifacts = artifacts

    async def authorize_and_lock(
        self, db: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ) -> SessionWriteState:
        if request.target_id is None:
            raise DomainNotFoundError
        row = await _session(db, tenant, request.target_id, write=True)
        if tenant.role is WorkspaceRole.REVIEWER:
            raise DomainNotFoundError
        status = await db.scalar(
            select(Run.status).where(
                Run.workspace_id == tenant.workspace_id,
                Run.id == row.latest_run_id,
            )
        )
        if status is None:
            raise DomainInvariantError("session latest run is missing")
        return SessionWriteState(
            revision=row.revision,
            current_version_id=row.current_version_id,
            has_active_modification=status in {"queued", "running"},
        )

    async def apply(
        self,
        db: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        assert request.target_id is not None
        payload = ConfirmVersionV1.model_validate(request.payload)
        if not payload.attested:
            raise DomainConflictError("explicit review attestation is required")
        session = await _session(db, tenant, request.target_id, write=True)
        version = await _version(db, tenant, session.id, payload.version_id)
        if version.artifact_id != payload.artifact_id:
            raise DomainConflictError("selected version artifact changed")
        try:
            artifact = await self.artifacts._read(db, tenant, version.artifact_id)
        except DomainNotFoundError:
            raise DomainConflictError("selected version artifact is missing") from None
        if artifact.tex_sha256 != payload.tex_sha256:
            raise DomainConflictError("selected version digest changed")
        existing = await db.scalar(
            select(ResumeConfirmation).where(
                ResumeConfirmation.workspace_id == tenant.workspace_id,
                ResumeConfirmation.session_id == session.id,
                ResumeConfirmation.version_id == version.id,
            )
        )
        if existing is not None:
            if existing.artifact_id != artifact.id or existing.tex_sha256 != artifact.tex_sha256:
                raise DomainInvariantError("immutable confirmation differs from artifact")
            return CommandReceiptV1(
                command_id=command_id, resource_id=existing.id, status="completed"
            )
        await _require_confirmable(db, tenant, version)
        confirmation_id = uuid4()
        db.add(
            ResumeConfirmation(
                id=confirmation_id,
                workspace_id=tenant.workspace_id,
                session_id=session.id,
                version_id=version.id,
                artifact_id=artifact.id,
                tex_sha256=artifact.tex_sha256,
                confirmed_by_user_id=tenant.actor_user_id,
            )
        )
        session.revision += 1
        return CommandReceiptV1(
            command_id=command_id, resource_id=confirmation_id, status="completed"
        )


class SqlAlchemyResumeConfirmationStore:
    def __init__(
        self,
        sessions: AsyncSessionFactory,
        artifacts: SqlAlchemyResumeArtifactStore | None = None,
    ) -> None:
        self.sessions = sessions
        self.artifacts = artifacts or SqlAlchemyResumeArtifactStore(sessions)
        self.commands = SqlAlchemyResumeCommandStore(sessions)

    async def list_versions(self, tenant: TenantContext, session_id: UUID):
        async with database_session(self.sessions) as db:
            await _session(db, tenant, session_id)
            rows = (
                await db.execute(
                    select(ResumeVersion, ResumeTexArtifact, ResumeConfirmation)
                    .outerjoin(
                        ResumeTexArtifact,
                        (ResumeTexArtifact.workspace_id == ResumeVersion.workspace_id)
                        & (ResumeTexArtifact.id == ResumeVersion.artifact_id),
                    )
                    .outerjoin(
                        ResumeConfirmation,
                        (ResumeConfirmation.workspace_id == ResumeVersion.workspace_id)
                        & (ResumeConfirmation.session_id == ResumeVersion.session_id)
                        & (ResumeConfirmation.version_id == ResumeVersion.id),
                    )
                    .where(
                        ResumeVersion.workspace_id == tenant.workspace_id,
                        ResumeVersion.session_id == session_id,
                    )
                    .order_by(ResumeVersion.version.desc())
                )
            ).all()
            if any(artifact is None for _, artifact, _ in rows):
                raise DomainInvariantError("resume version artifact is missing")
            if any(
                confirmation is not None
                and (
                    confirmation.artifact_id != version.artifact_id
                    or confirmation.tex_sha256 != artifact.tex_sha256
                )
                for version, artifact, confirmation in rows
            ):
                raise DomainInvariantError("immutable confirmation differs from version artifact")
            return [
                {
                    "version_id": version.id,
                    "session_id": session_id,
                    "version": version.version,
                    "parent_version_id": version.parent_version_id,
                    "artifact_id": version.artifact_id,
                    "tex_sha256": artifact.tex_sha256,
                    "created_at": version.created_at,
                    "confirmation": _confirmation(confirmation) if confirmation else None,
                }
                for version, artifact, confirmation in rows
            ]

    async def confirm(
        self,
        tenant: TenantContext,
        session_id: UUID,
        version_id: UUID,
        request: ConfirmVersionV1,
        key: UUID,
    ):
        if request.version_id != version_id:
            raise DomainConflictError("selected version identity differs from request")
        accepted = await self.commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=key,
                kind="resume_confirm_version",
                target_id=session_id,
                payload_version=1,
                payload=request,
                session_write=SessionWritePrecondition(
                    expected_session_revision=request.expected_session_revision,
                    base_version_id=request.expected_current_version_id,
                ),
            ),
            writer=_ConfirmationWriter(self.artifacts),
        )
        assert accepted.receipt.resource_id is not None
        async with database_session(self.sessions) as db:
            await _session(db, tenant, session_id)
            row = await db.scalar(
                select(ResumeConfirmation).where(
                    ResumeConfirmation.workspace_id == tenant.workspace_id,
                    ResumeConfirmation.session_id == session_id,
                    ResumeConfirmation.version_id == version_id,
                    ResumeConfirmation.id == accepted.receipt.resource_id,
                )
            )
            if row is None:
                raise DomainInvariantError("confirmation receipt has no record")
            return {
                **_confirmation(row).model_dump(),
                "command_id": accepted.receipt.command_id,
                "replayed": accepted.replayed,
            }

    async def download(
        self, tenant: TenantContext, session_id: UUID, version_id: UUID
    ) -> tuple[bytes, str, int, bool]:
        async with database_session(self.sessions) as db:
            await _session(db, tenant, session_id)
            version = await _version(db, tenant, session_id, version_id)
            artifact = await self.artifacts._read(db, tenant, version.artifact_id)
            confirmation = await db.scalar(
                select(ResumeConfirmation).where(
                    ResumeConfirmation.workspace_id == tenant.workspace_id,
                    ResumeConfirmation.session_id == session_id,
                    ResumeConfirmation.version_id == version_id,
                )
            )
            if confirmation is not None and (
                confirmation.artifact_id != artifact.id
                or confirmation.tex_sha256 != artifact.tex_sha256
            ):
                raise DomainInvariantError("confirmed version artifact is inconsistent")
            return (
                artifact.tex_bytes,
                artifact.tex_sha256,
                version.version,
                confirmation is not None,
            )
