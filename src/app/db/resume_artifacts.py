"""Immutable TeX publication and currently authorized artifact reads."""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ResumeProfileImport, ResumeProfileVersion, ResumeTexArtifact
from app.db.resume_profiles import _profile
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import DomainInvariantError, DomainNotFoundError, DomainValidationError
from app.domain.resume_artifacts import TexArtifactInfoV1
from app.domain.resume_profile import ResumeContentV1, ResumePreferencesV1
from app.domain.resume_templates import TemplateManifest, TemplateRenderer
from app.domain.tenancy import TenantContext
from app.resume.template_import import FIXED_PREAMBLE_SHA256, FIXED_SOURCE_SHA256, TEMPLATE_COMMIT
from app.resume.template_render import (
    RENDERER_VERSION,
    FixedTemplateRenderer,
    TemplateIdentity,
    render_with_template,
)


class SqlAlchemyResumeArtifactStore:
    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        *,
        expected_source_sha256: str = FIXED_SOURCE_SHA256,
        expected_preamble_sha256: str = FIXED_PREAMBLE_SHA256,
        renderer: TemplateRenderer | None = None,
        template_manifest: TemplateManifest | None = None,
        template_source: bytes | None = None,
    ) -> None:
        if (template_manifest is None) != (template_source is None):
            raise ValueError("template manifest and source must be supplied together")
        self._renderer = renderer if renderer is not None else FixedTemplateRenderer()
        self._manifest = template_manifest or TemplateManifest(
            TEMPLATE_COMMIT,
            expected_source_sha256,
            expected_preamble_sha256,
            RENDERER_VERSION,
        )
        self._template_source = template_source
        self._sessions = session_factory
        self._identity = TemplateIdentity(
            source_sha256=expected_source_sha256,
            preamble_sha256=expected_preamble_sha256,
        )

    async def stage(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        *,
        profile_version_id: UUID,
        content: ResumeContentV1,
        preferences: ResumePreferencesV1,
    ) -> UUID:
        """Publish inside the caller's transaction; never commit independently."""
        version = await session.scalar(
            select(ResumeProfileVersion).where(
                ResumeProfileVersion.workspace_id == tenant.workspace_id,
                ResumeProfileVersion.id == profile_version_id,
            )
        )
        if version is None:
            raise DomainNotFoundError
        await _profile(session, tenant, version.profile_id, edit=True)
        source = await session.scalar(
            select(ResumeProfileImport).where(
                ResumeProfileImport.workspace_id == tenant.workspace_id,
                ResumeProfileImport.profile_id == version.profile_id,
                ResumeProfileImport.id == version.source_import_id,
            )
        )
        if source is None:
            raise DomainInvariantError("resume profile source is missing")
        if (
            source.template_commit != TEMPLATE_COMMIT
            or source.source_sha256 != self._identity.source_sha256
        ):
            raise DomainValidationError("resume template identity is unsupported")
        profile_content = ResumeContentV1.model_validate(version.content_json)
        rendered = render_with_template(
            self._renderer,
            source_bytes=source.source_bytes
            if self._template_source is None
            else self._template_source,
            manifest=self._manifest,
            profile_content=profile_content,
            content=content,
            preferences=preferences,
        )
        identity = {
            "workspace_id": tenant.workspace_id,
            "profile_version_id": version.id,
            "template_source_sha256": rendered.manifest.source_sha256,
            "content_sha256": rendered.content_sha256,
            "config_sha256": rendered.config_sha256,
            "renderer_version": self._manifest.renderer_version,
        }
        artifact_id = uuid4()
        statement = (
            pg_insert(ResumeTexArtifact)
            .values(
                id=artifact_id,
                created_by_user_id=tenant.actor_user_id,
                template_commit=rendered.manifest.commit,
                preamble_sha256=rendered.manifest.preamble_sha256,
                tex_sha256=rendered.tex_sha256,
                tex_bytes=rendered.tex_bytes,
                **identity,
            )
            .on_conflict_do_nothing(constraint="uq_resume_tex_artifacts_identity")
            .returning(ResumeTexArtifact.id)
        )
        inserted = await session.scalar(statement)
        if inserted is not None:
            return inserted
        existing = await session.scalar(
            select(ResumeTexArtifact).where(
                ResumeTexArtifact.workspace_id == tenant.workspace_id,
                ResumeTexArtifact.profile_version_id == version.id,
                ResumeTexArtifact.template_source_sha256 == rendered.manifest.source_sha256,
                ResumeTexArtifact.content_sha256 == rendered.content_sha256,
                ResumeTexArtifact.config_sha256 == rendered.config_sha256,
                ResumeTexArtifact.renderer_version == self._manifest.renderer_version,
            )
        )
        if (
            existing is None
            or existing.tex_sha256 != rendered.tex_sha256
            or existing.tex_bytes != rendered.tex_bytes
            or existing.preamble_sha256 != rendered.manifest.preamble_sha256
            or existing.template_commit != rendered.manifest.commit
        ):
            raise DomainInvariantError("resume artifact identity is inconsistent")
        return existing.id

    async def _read(
        self, session: AsyncSession, tenant: TenantContext, artifact_id: UUID
    ) -> ResumeTexArtifact:
        artifact = await session.scalar(
            select(ResumeTexArtifact).where(
                ResumeTexArtifact.workspace_id == tenant.workspace_id,
                ResumeTexArtifact.id == artifact_id,
            )
        )
        if artifact is None:
            raise DomainNotFoundError
        version = await session.scalar(
            select(ResumeProfileVersion).where(
                ResumeProfileVersion.workspace_id == tenant.workspace_id,
                ResumeProfileVersion.id == artifact.profile_version_id,
            )
        )
        if version is None:
            raise DomainInvariantError("resume artifact profile version is missing")
        await _profile(session, tenant, version.profile_id, edit=False)
        source = await session.scalar(
            select(ResumeProfileImport).where(
                ResumeProfileImport.workspace_id == tenant.workspace_id,
                ResumeProfileImport.profile_id == version.profile_id,
                ResumeProfileImport.id == version.source_import_id,
            )
        )
        if (
            source is None
            or source.template_commit != TEMPLATE_COMMIT
            or source.source_sha256 != self._identity.source_sha256
            or artifact.template_commit != self._manifest.commit
            or artifact.template_source_sha256 != self._manifest.source_sha256
            or artifact.preamble_sha256 != self._manifest.preamble_sha256
            or artifact.renderer_version != self._manifest.renderer_version
            or not artifact.tex_bytes
            or len(artifact.tex_bytes) > 512 * 1024
            or hashlib.sha256(artifact.tex_bytes).hexdigest() != artifact.tex_sha256
        ):
            raise DomainInvariantError("resume artifact bytes are inconsistent")
        return artifact

    async def get_info(self, tenant: TenantContext, artifact_id: UUID) -> TexArtifactInfoV1:
        async with database_session(self._sessions) as session:
            artifact = await self._read(session, tenant, artifact_id)
            return TexArtifactInfoV1(
                artifact_id=artifact.id,
                profile_version_id=artifact.profile_version_id,
                template_commit=artifact.template_commit,
                template_source_sha256=artifact.template_source_sha256,
                preamble_sha256=artifact.preamble_sha256,
                renderer_version=artifact.renderer_version,
                tex_sha256=artifact.tex_sha256,
                byte_count=len(artifact.tex_bytes),
                compile=self._manifest.compile,
            )

    async def get_bytes(self, tenant: TenantContext, artifact_id: UUID) -> tuple[bytes, str]:
        async with database_session(self._sessions) as session:
            artifact = await self._read(session, tenant, artifact_id)
            return artifact.tex_bytes, artifact.tex_sha256
