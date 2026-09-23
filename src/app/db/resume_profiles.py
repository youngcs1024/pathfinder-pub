"""Authorized immutable profile versions and source-claim review commands."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    MaterialFact,
    MaterialFactSet,
    MaterialFactVersion,
    MaterialProject,
    ResumeClaimFactLink,
    ResumeClaimReview,
    ResumePreferenceVersion,
    ResumeProfile,
    ResumeProfileImport,
    ResumeProfileVersion,
    ResumeSourceClaim,
    WorkspaceMembership,
)
from app.db.resume_commands import ResumeCommandWriter, SqlAlchemyResumeCommandStore
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import DomainConflictError, DomainNotFoundError, DomainValidationError
from app.domain.provisioning import WorkspaceRole
from app.domain.resume_commands import CommandReceiptV1, ResumeCommandRequest
from app.domain.resume_profile import (
    ResumeContentV1,
    ResumePreferencesV1,
    validate_preference_targets,
)
from app.domain.tenancy import TenantContext
from app.resume.template_import import (
    FIXED_PREAMBLE_SHA256,
    FIXED_SOURCE_SHA256,
    TEMPLATE_COMMIT,
    parse_resume_source,
)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class ImportInputV1(_Input):
    source_tex: str = Field(min_length=1, max_length=131072)


class ContactEditV1(_Input):
    expected_version: int = Field(ge=1)
    item_id: UUID
    value: str = Field(min_length=1, max_length=300)


class ItemReviewV1(_Input):
    expected_version: int = Field(ge=1)
    item_id: UUID


class PreferenceUpdateV1(_Input):
    expected_version: int = Field(ge=1)
    preferences: ResumePreferencesV1


class ClaimReviewV1(_Input):
    claim_id: UUID
    expected_review_version: int = Field(ge=0)
    decision: Literal["linked", "needs_evidence", "excluded"]
    project_id: UUID | None = None
    fact_version_ids: tuple[UUID, ...] = ()


async def _member(session: AsyncSession, tenant: TenantContext, *, lock: bool = False) -> str:
    statement = select(WorkspaceMembership.role).where(
        WorkspaceMembership.workspace_id == tenant.workspace_id,
        WorkspaceMembership.user_id == tenant.actor_user_id,
        WorkspaceMembership.revoked_at.is_(None),
        WorkspaceMembership.role == tenant.role.value,
    )
    if lock:
        statement = statement.with_for_update()
    role = await session.scalar(statement)
    if role is None:
        raise DomainNotFoundError
    return role


async def _profile(
    session: AsyncSession,
    tenant: TenantContext,
    profile_id: UUID,
    *,
    edit: bool,
    lock: bool = False,
) -> ResumeProfile:
    role = await _member(session, tenant)
    statement = select(ResumeProfile).where(
        ResumeProfile.workspace_id == tenant.workspace_id,
        ResumeProfile.id == profile_id,
    )
    if lock:
        statement = statement.with_for_update()
    profile = await session.scalar(statement)
    if (
        profile is None
        or (profile.owner_user_id != tenant.actor_user_id and role != WorkspaceRole.ADMIN.value)
        or (edit and role == WorkspaceRole.REVIEWER.value)
    ):
        raise DomainNotFoundError
    return profile


async def _version(
    session: AsyncSession, tenant: TenantContext, profile: ResumeProfile
) -> ResumeProfileVersion:
    row = await session.scalar(
        select(ResumeProfileVersion).where(
            ResumeProfileVersion.workspace_id == tenant.workspace_id,
            ResumeProfileVersion.profile_id == profile.id,
            ResumeProfileVersion.version == profile.current_version,
        )
    )
    if row is None:
        raise DomainNotFoundError
    return row


async def _preference(
    session: AsyncSession, tenant: TenantContext, profile: ResumeProfile
) -> ResumePreferenceVersion:
    row = await session.scalar(
        select(ResumePreferenceVersion).where(
            ResumePreferenceVersion.workspace_id == tenant.workspace_id,
            ResumePreferenceVersion.profile_id == profile.id,
            ResumePreferenceVersion.version == profile.current_preference_version,
        )
    )
    if row is None:
        raise DomainNotFoundError
    return row


def _new_version(
    session: AsyncSession,
    tenant: TenantContext,
    profile: ResumeProfile,
    old: ResumeProfileVersion,
    content: ResumeContentV1,
) -> None:
    profile.current_version += 1
    session.add(
        ResumeProfileVersion(
            id=uuid4(),
            workspace_id=tenant.workspace_id,
            profile_id=profile.id,
            source_import_id=old.source_import_id,
            version=profile.current_version,
            content_json=content.model_dump(mode="json"),
            created_by_user_id=tenant.actor_user_id,
        )
    )


class _ProfileWriter(ResumeCommandWriter):
    supported_kinds = frozenset(
        {
            "resume_profile_import",
            "resume_profile_contact_edit",
            "resume_profile_item_review",
            "resume_preference_update",
            "resume_claim_review",
        }
    )

    def __init__(self, expected_source_sha256: str, expected_preamble_sha256: str) -> None:
        self._source_sha = expected_source_sha256
        self._preamble_sha = expected_preamble_sha256

    async def authorize_and_lock(
        self, session: AsyncSession, tenant: TenantContext, request: ResumeCommandRequest
    ):
        if request.kind == "resume_profile_import":
            role = await _member(session, tenant)
            if role == WorkspaceRole.REVIEWER.value:
                raise DomainNotFoundError
        else:
            assert request.target_id is not None
            await _profile(session, tenant, request.target_id, edit=True, lock=True)
        return None

    async def apply(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        request: ResumeCommandRequest,
        command_id: UUID,
    ) -> CommandReceiptV1:
        if request.kind == "resume_profile_import":
            payload = ImportInputV1.model_validate(request.payload.model_dump())
            preview = parse_resume_source(
                payload.source_tex,
                expected_source_sha256=self._source_sha,
                expected_preamble_sha256=self._preamble_sha,
            )
            if not preview.complete or preview.content is None:
                raise DomainValidationError("resume source cannot be imported")
            existing = await session.scalar(
                select(ResumeProfile.id).where(
                    ResumeProfile.workspace_id == tenant.workspace_id,
                    ResumeProfile.owner_user_id == tenant.actor_user_id,
                )
            )
            if existing is not None:
                raise DomainConflictError("resume profile already exists")
            profile_id, import_id = uuid4(), uuid4()
            session.add(
                ResumeProfile(
                    id=profile_id,
                    workspace_id=tenant.workspace_id,
                    owner_user_id=tenant.actor_user_id,
                    current_version=1,
                    current_preference_version=1,
                )
            )
            await session.flush()
            session.add(
                ResumeProfileImport(
                    id=import_id,
                    workspace_id=tenant.workspace_id,
                    profile_id=profile_id,
                    template_commit=TEMPLATE_COMMIT,
                    source_sha256=preview.source_sha256,
                    source_bytes=payload.source_tex.encode("utf-8"),
                )
            )
            await session.flush()
            session.add(
                ResumeProfileVersion(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    profile_id=profile_id,
                    source_import_id=import_id,
                    version=1,
                    content_json=preview.content.model_dump(mode="json"),
                    created_by_user_id=tenant.actor_user_id,
                )
            )
            session.add(
                ResumePreferenceVersion(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    profile_id=profile_id,
                    version=1,
                    preferences_json=ResumePreferencesV1().model_dump(mode="json"),
                    created_by_user_id=tenant.actor_user_id,
                )
            )
            session.add_all(
                ResumeSourceClaim(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    source_import_id=import_id,
                    project_item_id=claim.project_item_id,
                    item_id=claim.item_id,
                    field=claim.field,
                    claim_text=claim.text,
                    source_json=claim.source.model_dump(mode="json"),
                    review_version=0,
                )
                for claim in preview.claims
            )
            return CommandReceiptV1(
                command_id=command_id, resource_id=profile_id, status="completed"
            )
        assert request.target_id is not None
        profile = await _profile(session, tenant, request.target_id, edit=True, lock=True)
        if request.kind == "resume_claim_review":
            payload = ClaimReviewV1.model_validate(request.payload.model_dump())
            current = await _version(session, tenant, profile)
            claim = await session.scalar(
                select(ResumeSourceClaim)
                .where(
                    ResumeSourceClaim.workspace_id == tenant.workspace_id,
                    ResumeSourceClaim.source_import_id == current.source_import_id,
                    ResumeSourceClaim.id == payload.claim_id,
                )
                .with_for_update()
            )
            if claim is None:
                raise DomainNotFoundError
            if claim.review_version != payload.expected_review_version:
                raise DomainConflictError("claim review is stale")
            if payload.decision == "linked":
                if (
                    payload.project_id is None
                    or not payload.fact_version_ids
                    or len(set(payload.fact_version_ids)) != len(payload.fact_version_ids)
                ):
                    raise DomainValidationError("linked claim requires distinct confirmed facts")
                project = await session.scalar(
                    select(MaterialProject).where(
                        MaterialProject.workspace_id == tenant.workspace_id,
                        MaterialProject.id == payload.project_id,
                    )
                )
                if project is None or (
                    project.created_by_user_id != tenant.actor_user_id
                    and tenant.role != WorkspaceRole.ADMIN
                ):
                    raise DomainNotFoundError
                versions = (
                    await session.scalars(
                        select(MaterialFactVersion)
                        .join(
                            MaterialFact,
                            (MaterialFact.workspace_id == MaterialFactVersion.workspace_id)
                            & (MaterialFact.id == MaterialFactVersion.fact_id),
                        )
                        .join(
                            MaterialFactSet,
                            (MaterialFactSet.workspace_id == MaterialFact.workspace_id)
                            & (MaterialFactSet.id == MaterialFact.fact_set_id),
                        )
                        .where(
                            MaterialFactVersion.workspace_id == tenant.workspace_id,
                            MaterialFactVersion.id.in_(payload.fact_version_ids),
                            MaterialFactVersion.review_status == "confirmed",
                            MaterialFactSet.project_id == payload.project_id,
                        )
                    )
                ).all()
                if len(versions) != len(payload.fact_version_ids):
                    raise DomainValidationError("linked fact is not confirmed in this project")
            elif payload.project_id is not None or payload.fact_version_ids:
                raise DomainValidationError("unlinked claim cannot carry fact references")
            review_id = uuid4()
            claim.review_version += 1
            session.add(
                ResumeClaimReview(
                    id=review_id,
                    workspace_id=tenant.workspace_id,
                    claim_id=claim.id,
                    version=claim.review_version,
                    decision=payload.decision,
                    project_id=payload.project_id,
                    actor_user_id=tenant.actor_user_id,
                )
            )
            session.add_all(
                ResumeClaimFactLink(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    review_id=review_id,
                    fact_version_id=fact_id,
                )
                for fact_id in payload.fact_version_ids
            )
            return CommandReceiptV1(
                command_id=command_id, resource_id=review_id, status="completed"
            )
        old = await _version(session, tenant, profile)
        content = ResumeContentV1.model_validate(old.content_json)
        if request.kind == "resume_profile_contact_edit":
            payload = ContactEditV1.model_validate(request.payload.model_dump())
            if payload.expected_version != profile.current_version:
                raise DomainConflictError("profile version is stale")
            if payload.item_id == content.display_name_id:
                if len(payload.value) > 200:
                    raise DomainValidationError("display name is too long")
                content = content.model_copy(
                    update={"display_name": payload.value, "display_name_review_status": "pending"}
                )
            else:
                if payload.item_id not in {item.id for item in content.contact}:
                    raise DomainNotFoundError
                fields = tuple(
                    item.model_copy(update={"value": payload.value, "review_status": "pending"})
                    if item.id == payload.item_id
                    else item
                    for item in content.contact
                )
                content = content.model_copy(update={"contact": fields})
            _new_version(session, tenant, profile, old, content)
        elif request.kind == "resume_profile_item_review":
            payload = ItemReviewV1.model_validate(request.payload.model_dump())
            if payload.expected_version != profile.current_version:
                raise DomainConflictError("profile version is stale")
            if payload.item_id == content.display_name_id:
                content = content.model_copy(update={"display_name_review_status": "reviewed"})
            else:
                replacements = {}
                for section in ("contact", "education", "projects", "skills"):
                    items = getattr(content, section)
                    if any(item.id == payload.item_id for item in items):
                        replacements[section] = tuple(
                            item.model_copy(update={"review_status": "reviewed"})
                            if item.id == payload.item_id
                            else item
                            for item in items
                        )
                if not replacements:
                    raise DomainNotFoundError
                content = content.model_copy(update=replacements)
            _new_version(session, tenant, profile, old, content)
        elif request.kind == "resume_preference_update":
            payload = PreferenceUpdateV1.model_validate(request.payload.model_dump())
            if payload.expected_version != profile.current_preference_version:
                raise DomainConflictError("preference version is stale")
            validate_preference_targets(content, payload.preferences)
            profile.current_preference_version += 1
            session.add(
                ResumePreferenceVersion(
                    id=uuid4(),
                    workspace_id=tenant.workspace_id,
                    profile_id=profile.id,
                    version=profile.current_preference_version,
                    preferences_json=payload.preferences.model_dump(mode="json"),
                    created_by_user_id=tenant.actor_user_id,
                )
            )
        else:
            raise DomainValidationError("profile command is invalid")
        return CommandReceiptV1(command_id=command_id, resource_id=profile.id, status="completed")


class SqlAlchemyResumeProfileStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        expected_source_sha256: str = FIXED_SOURCE_SHA256,
        expected_preamble_sha256: str = FIXED_PREAMBLE_SHA256,
    ) -> None:
        self._sessions = session_factory
        self._commands = SqlAlchemyResumeCommandStore(session_factory)
        self._writer = _ProfileWriter(expected_source_sha256, expected_preamble_sha256)
        self._source_sha = expected_source_sha256
        self._preamble_sha = expected_preamble_sha256

    async def preview(self, tenant: TenantContext, source_tex: str):
        async with database_session(self._sessions) as session:
            await _member(session, tenant)
        return parse_resume_source(
            source_tex,
            expected_source_sha256=self._source_sha,
            expected_preamble_sha256=self._preamble_sha,
        )

    async def import_source(self, tenant: TenantContext, source_tex: str, request_id: UUID):
        payload = ImportInputV1(source_tex=source_tex)
        try:
            return await self._commands.accept(
                tenant=tenant,
                request=ResumeCommandRequest(
                    client_request_id=request_id,
                    kind="resume_profile_import",
                    target_id=None,
                    payload_version=1,
                    payload=payload,
                ),
                writer=self._writer,
            )
        except IntegrityError as error:
            if (
                getattr(error.orig, "sqlstate", None) == "23505"
                and getattr(getattr(error.orig, "diag", None), "constraint_name", None)
                == "uq_resume_profiles_workspace_id_owner_user_id"
            ):
                raise DomainConflictError("profile already exists") from None
            raise

    async def command(
        self,
        tenant: TenantContext,
        *,
        kind: str,
        profile_id: UUID,
        request_id: UUID,
        payload: BaseModel,
    ):
        return await self._commands.accept(
            tenant=tenant,
            request=ResumeCommandRequest(
                client_request_id=request_id,
                kind=kind,
                target_id=profile_id,
                payload_version=1,
                payload=payload,
            ),
            writer=self._writer,
        )

    async def get_me(self, tenant: TenantContext):
        async with database_session(self._sessions) as session:
            await _member(session, tenant)
            profile_id = await session.scalar(
                select(ResumeProfile.id).where(
                    ResumeProfile.workspace_id == tenant.workspace_id,
                    ResumeProfile.owner_user_id == tenant.actor_user_id,
                )
            )
            if profile_id is None:
                raise DomainNotFoundError
            return await self._detail(session, tenant, profile_id)

    async def get_profile(self, tenant: TenantContext, profile_id: UUID):
        async with database_session(self._sessions) as session:
            return await self._detail(session, tenant, profile_id)

    async def _detail(self, session: AsyncSession, tenant: TenantContext, profile_id: UUID):
        profile = await _profile(session, tenant, profile_id, edit=False)
        version = await _version(session, tenant, profile)
        preference = await _preference(session, tenant, profile)
        claims = (
            await session.scalars(
                select(ResumeSourceClaim)
                .where(
                    ResumeSourceClaim.workspace_id == tenant.workspace_id,
                    ResumeSourceClaim.source_import_id == version.source_import_id,
                )
                .order_by(ResumeSourceClaim.field, ResumeSourceClaim.id)
            )
        ).all()
        reviews = (
            (
                await session.scalars(
                    select(ResumeClaimReview).where(
                        ResumeClaimReview.workspace_id == tenant.workspace_id,
                        ResumeClaimReview.claim_id.in_([claim.id for claim in claims]),
                    )
                )
            ).all()
            if claims
            else []
        )
        current_reviews = {}
        for review in reviews:
            if (
                review.claim_id not in current_reviews
                or review.version > current_reviews[review.claim_id].version
            ):
                current_reviews[review.claim_id] = review
        linked_review_ids = [
            review.id for review in current_reviews.values() if review.decision == "linked"
        ]
        links = (
            (
                await session.scalars(
                    select(ResumeClaimFactLink).where(
                        ResumeClaimFactLink.workspace_id == tenant.workspace_id,
                        ResumeClaimFactLink.review_id.in_(linked_review_ids),
                    )
                )
            ).all()
            if linked_review_ids
            else []
        )
        link_map: dict[UUID, list[str]] = {}
        for link in links:
            link_map.setdefault(link.review_id, []).append(str(link.fact_version_id))
        return {
            "profile_id": profile.id,
            "owner_user_id": profile.owner_user_id,
            "version_id": version.id,
            "version": profile.current_version,
            "content": version.content_json,
            "preference_version": profile.current_preference_version,
            "preferences": preference.preferences_json,
            "claims": [
                {
                    "id": claim.id,
                    "project_item_id": claim.project_item_id,
                    "item_id": claim.item_id,
                    "field": claim.field,
                    "text": claim.claim_text,
                    "source": claim.source_json,
                    "review_version": claim.review_version,
                    "decision": current_reviews[claim.id].decision
                    if claim.id in current_reviews
                    else "pending",
                    "project_id": current_reviews[claim.id].project_id
                    if claim.id in current_reviews
                    else None,
                    "fact_version_ids": link_map.get(current_reviews[claim.id].id, [])
                    if claim.id in current_reviews
                    else [],
                }
                for claim in claims
            ],
        }
