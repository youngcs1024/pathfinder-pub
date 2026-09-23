"""R3.1 synthetic profile import and review across real PostgreSQL transactions."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select, update

from app.db.models import ResumeProfileImport, WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_profiles import (
    ClaimReviewV1,
    ContactEditV1,
    ItemReviewV1,
    PreferenceUpdateV1,
    SqlAlchemyResumeProfileStore,
)
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainConflictError, DomainNotFoundError
from app.domain.provisioning import ProvisioningService
from app.domain.resume_profile import ResumePreferencesV1
from app.domain.tenancy import TenantService

pytestmark = pytest.mark.integration
SOURCE = Path(__file__).resolve().parents[2] / "fixtures/resume/synthetic_main.tex"


async def test_profile_import_versions_reviews_replay_scope_and_revocation(
    migrated_database_url: str,
) -> None:
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    source = SOURCE.read_text()
    preamble = source.split("\\begin{document}", 1)[0]
    store = SqlAlchemyResumeProfileStore(
        sessions,
        expected_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        expected_preamble_sha256=hashlib.sha256(preamble.encode()).hexdigest(),
    )
    provisioning = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
    tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
    try:
        owner = await provisioning.provision_personal_workspace("r31-owner")
        foreign = await provisioning.provision_personal_workspace("r31-foreign")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        other = await tenancy.resolve_tenant(
            workspace_id=foreign.workspace_id, actor_user_id=foreign.user_id
        )
        preview = await store.preview(tenant, source)
        assert preview.complete and preview.content is not None
        key = uuid4()
        accepted = await store.import_source(tenant, source, key)
        assert not accepted.replayed and accepted.receipt.resource_id is not None
        replay = await store.import_source(tenant, source, key)
        assert replay.replayed and replay.receipt == accepted.receipt
        with pytest.raises(DomainConflictError):
            await store.import_source(tenant, source.replace("SQL", "Rust"), key)
        with pytest.raises(DomainConflictError):
            await store.import_source(tenant, source, uuid4())
        profile_id = accepted.receipt.resource_id
        with pytest.raises(DomainNotFoundError):
            await store.get_profile(other, profile_id)
        profile = await store.get_me(tenant)
        assert profile["version"] == 1 and len(profile["claims"]) == 6
        assert profile["content"]["contact"][0]["locked"] is True
        assert all(claim["decision"] == "pending" for claim in profile["claims"])
        async with sessions() as session:
            raw = await session.scalar(
                select(ResumeProfileImport.source_bytes).where(
                    ResumeProfileImport.workspace_id == tenant.workspace_id,
                    ResumeProfileImport.profile_id == profile_id,
                )
            )
            assert raw == source.encode()
        contact_id = profile["content"]["contact"][0]["id"]
        edited = await store.command(
            tenant,
            kind="resume_profile_contact_edit",
            profile_id=profile_id,
            request_id=uuid4(),
            payload=ContactEditV1(expected_version=1, item_id=contact_id, value="555-0199"),
        )
        assert edited.receipt.status == "completed"
        current = await store.get_profile(tenant, profile_id)
        assert current["version"] == 2
        assert current["content"]["contact"][0]["value"] == "555-0199"
        with pytest.raises(DomainConflictError):
            await store.command(
                tenant,
                kind="resume_profile_item_review",
                profile_id=profile_id,
                request_id=uuid4(),
                payload=ItemReviewV1(expected_version=1, item_id=contact_id),
            )
        await store.command(
            tenant,
            kind="resume_profile_item_review",
            profile_id=profile_id,
            request_id=uuid4(),
            payload=ItemReviewV1(expected_version=2, item_id=contact_id),
        )
        assert (await store.get_profile(tenant, profile_id))["version"] == 3
        await store.command(
            tenant,
            kind="resume_preference_update",
            profile_id=profile_id,
            request_id=uuid4(),
            payload=PreferenceUpdateV1(
                expected_version=1,
                preferences=ResumePreferencesV1(
                    banned_terms=("explicit-only",), writing_advice="Stay concise"
                ),
            ),
        )
        assert (await store.get_profile(tenant, profile_id))["preference_version"] == 2
        claim_id = profile["claims"][0]["id"]
        review = await store.command(
            tenant,
            kind="resume_claim_review",
            profile_id=profile_id,
            request_id=uuid4(),
            payload=ClaimReviewV1(
                claim_id=claim_id, expected_review_version=0, decision="needs_evidence"
            ),
        )
        assert review.receipt.resource_id is not None
        current_claim = next(
            item
            for item in (await store.get_profile(tenant, profile_id))["claims"]
            if item["id"] == claim_id
        )
        assert current_claim["decision"] == "needs_evidence"
        assert current_claim["review_version"] == 1
        with pytest.raises(DomainConflictError):
            await store.command(
                tenant,
                kind="resume_claim_review",
                profile_id=profile_id,
                request_id=uuid4(),
                payload=ClaimReviewV1(
                    claim_id=claim_id, expected_review_version=0, decision="excluded"
                ),
            )
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
            await store.import_source(tenant, source, key)
        with pytest.raises(DomainNotFoundError):
            await store.get_profile(tenant, profile_id)
    finally:
        await engine.dispose()
