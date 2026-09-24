"""R3.2 artifact publication, authorization, and corruption checks in PostgreSQL."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import func, select, update

from app.db.models import ResumeTexArtifact, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_profiles import SqlAlchemyResumeProfileStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainInvariantError, DomainNotFoundError
from app.domain.provisioning import ProvisioningService
from app.domain.resume_profile import ResumeContentV1, ResumePreferencesV1
from app.domain.tenancy import TenantService
from tests.resume_extensions import ALTERNATE_MANIFEST, alternate_kwargs

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


async def test_artifact_transaction_replay_scope_revocation_and_digest(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    source = SOURCE.read_text()
    preamble = source.split("\\begin{document}", 1)[0]
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    preamble_sha = hashlib.sha256(preamble.encode()).hexdigest()
    profiles = SqlAlchemyResumeProfileStore(
        sessions,
        expected_source_sha256=source_sha,
        expected_preamble_sha256=preamble_sha,
    )
    artifacts = SqlAlchemyResumeArtifactStore(
        sessions,
        expected_source_sha256=source_sha,
        expected_preamble_sha256=preamble_sha,
    )
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
    tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
    try:
        owner = await provisioning.provision_personal_workspace("r32-artifact-owner")
        foreign = await provisioning.provision_personal_workspace("r32-artifact-foreign")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        other = await tenancy.resolve_tenant(
            workspace_id=foreign.workspace_id, actor_user_id=foreign.user_id
        )
        accepted = await profiles.import_source(tenant, source, uuid4())
        assert accepted.receipt.resource_id is not None
        detail = await profiles.get_profile(tenant, accepted.receipt.resource_id)
        content = ResumeContentV1.model_validate(detail["content"])
        async with sessions.begin() as session:
            artifact_id = await artifacts.stage(
                session,
                tenant,
                profile_version_id=detail["version_id"],
                content=content,
                preferences=ResumePreferencesV1(),
            )
        async with sessions.begin() as session:
            repeated = await artifacts.stage(
                session,
                tenant,
                profile_version_id=detail["version_id"],
                content=content,
                preferences=ResumePreferencesV1(),
            )
        assert repeated == artifact_id
        with pytest.raises(RuntimeError, match="rollback"):
            async with sessions.begin() as session:
                await artifacts.stage(
                    session,
                    tenant,
                    profile_version_id=detail["version_id"],
                    content=content,
                    preferences=ResumePreferencesV1(
                        section_order=("skills", "education", "projects")
                    ),
                )
                raise RuntimeError("rollback")
        async with sessions() as session:
            assert await session.scalar(select(func.count()).select_from(ResumeTexArtifact)) == 1
        info = await artifacts.get_info(tenant, artifact_id)
        payload, digest = await artifacts.get_bytes(tenant, artifact_id)
        assert info.tex_sha256 == digest == hashlib.sha256(payload).hexdigest()
        assert info.byte_count == len(payload)
        assert info.template_source_sha256 == source_sha
        assert payload.startswith(preamble.encode())
        alternate_options = alternate_kwargs()
        alternate = SqlAlchemyResumeArtifactStore(
            sessions,
            expected_source_sha256=source_sha,
            expected_preamble_sha256=preamble_sha,
            **alternate_options,
        )
        async with sessions.begin() as session:
            alternate_id = await alternate.stage(
                session,
                tenant,
                profile_version_id=detail["version_id"],
                content=content,
                preferences=ResumePreferencesV1(),
            )
        assert alternate_id != artifact_id
        alternate_info = await alternate.get_info(tenant, alternate_id)
        assert alternate_info.template_source_sha256 == ALTERNATE_MANIFEST.source_sha256
        assert alternate_info.compile.packages == ("fontspec",)
        calls_before = alternate_options["renderer"].calls
        alternate_bytes, alternate_digest = await alternate.get_bytes(tenant, alternate_id)
        assert alternate_options["renderer"].calls == calls_before
        assert alternate_digest == hashlib.sha256(alternate_bytes).hexdigest()
        assert alternate_bytes != payload
        with pytest.raises(DomainInvariantError):
            await artifacts.get_bytes(tenant, alternate_id)
        with pytest.raises(DomainInvariantError):
            await alternate.get_bytes(tenant, artifact_id)
        with pytest.raises(DomainNotFoundError):
            await alternate.get_bytes(other, alternate_id)
        assert await artifacts.get_bytes(tenant, artifact_id) == (payload, digest)
        with pytest.raises(DomainNotFoundError):
            await artifacts.get_info(other, artifact_id)
        with pytest.raises(DomainNotFoundError):
            await artifacts.get_bytes(tenant, uuid4())
        async with sessions.begin() as session:
            await session.execute(
                update(ResumeTexArtifact)
                .where(ResumeTexArtifact.id == artifact_id)
                .values(tex_bytes=b"corrupt")
            )
        with pytest.raises(DomainInvariantError):
            await artifacts.get_bytes(tenant, artifact_id)
        async with sessions.begin() as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=datetime.now(UTC))
            )
        with pytest.raises(DomainNotFoundError):
            await artifacts.get_info(tenant, artifact_id)
    finally:
        await engine.dispose()
