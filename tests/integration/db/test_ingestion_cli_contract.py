from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import update

from app.cli.ingest_documents import (
    IngestionCommandArguments,
    compose_authorized_ingestion_batch,
)
from app.config import Settings
from app.db.models import WorkspaceMembership
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory, transaction
from app.domain.provisioning import ProvisioningService, WorkspaceRole
from app.retrieval.ingestion import (
    IngestionErrorCode,
    IngestionInputError,
    ValidatedIngestionBatch,
)

pytestmark = pytest.mark.integration


async def _provision(migrated_database_url: str, subject: str):  # type: ignore[no-untyped-def]
    engine = create_database_engine(SecretStr(migrated_database_url))
    session_factory = create_session_factory(engine)
    try:
        provisioned = await ProvisioningService(
            SqlAlchemyProvisioningStore(session_factory)
        ).provision_personal_workspace(subject)
        return provisioned, session_factory, engine
    except BaseException:
        await engine.dispose()
        raise


def _settings(database_url: str) -> Settings:
    return Settings(database_url=SecretStr(database_url), log_level="ERROR")


@pytest.mark.parametrize(
    "role",
    [WorkspaceRole.MEMBER, WorkspaceRole.REVIEWER, WorkspaceRole.ADMIN],
)
async def test_active_workspace_roles_are_authorized(
    migrated_database_url: str,
    role: WorkspaceRole,
) -> None:
    provisioned, session_factory, engine = await _provision(
        migrated_database_url, f"ingestion-{role.value}"
    )
    try:
        if role is not WorkspaceRole.ADMIN:
            async with transaction(session_factory) as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(WorkspaceMembership.id == provisioned.membership_id)
                    .values(role=role.value)
                )
    finally:
        await engine.dispose()

    expected_batch = ValidatedIngestionBatch(sources=())
    result = await compose_authorized_ingestion_batch(
        IngestionCommandArguments(
            provisioned.workspace_id,
            provisioned.user_id,
            (Path("explicit.txt"),),
        ),
        settings=_settings(migrated_database_url),
        batch_loader=lambda _paths: expected_batch,
    )

    assert result.tenant.role is role
    assert result.batch is expected_batch


async def _assert_access_denied_before_load(
    *,
    database_url: str,
    workspace_id: UUID,
    actor_user_id: UUID,
) -> None:
    loader_called = False

    def forbidden_loader(_paths: object) -> ValidatedIngestionBatch:
        nonlocal loader_called
        loader_called = True
        raise AssertionError("document body loader ran before authorization")

    with pytest.raises(IngestionInputError) as raised:
        await compose_authorized_ingestion_batch(
            IngestionCommandArguments(
                workspace_id,
                actor_user_id,
                (Path("must-not-open.txt"),),
            ),
            settings=_settings(database_url),
            batch_loader=forbidden_loader,
        )

    assert raised.value.code is IngestionErrorCode.WORKSPACE_ACCESS_DENIED
    assert loader_called is False


async def test_revoked_membership_is_denied_before_file_loading(
    migrated_database_url: str,
) -> None:
    provisioned, session_factory, engine = await _provision(
        migrated_database_url, "ingestion-revoked"
    )
    try:
        async with transaction(session_factory) as session:
            await session.execute(
                update(WorkspaceMembership)
                .where(WorkspaceMembership.id == provisioned.membership_id)
                .values(revoked_at=datetime.now(UTC))
            )
    finally:
        await engine.dispose()

    await _assert_access_denied_before_load(
        database_url=migrated_database_url,
        workspace_id=provisioned.workspace_id,
        actor_user_id=provisioned.user_id,
    )


async def test_cross_workspace_actor_is_denied(migrated_database_url: str) -> None:
    first, _first_sessions, first_engine = await _provision(
        migrated_database_url, "ingestion-workspace-a"
    )
    second, _second_sessions, second_engine = await _provision(
        migrated_database_url, "ingestion-workspace-b"
    )
    await first_engine.dispose()
    await second_engine.dispose()

    await _assert_access_denied_before_load(
        database_url=migrated_database_url,
        workspace_id=second.workspace_id,
        actor_user_id=first.user_id,
    )


async def test_unknown_actor_is_denied(migrated_database_url: str) -> None:
    provisioned, _sessions, engine = await _provision(
        migrated_database_url, "ingestion-known-workspace"
    )
    await engine.dispose()

    await _assert_access_denied_before_load(
        database_url=migrated_database_url,
        workspace_id=provisioned.workspace_id,
        actor_user_id=uuid4(),
    )


async def test_unknown_workspace_is_denied(migrated_database_url: str) -> None:
    provisioned, _sessions, engine = await _provision(
        migrated_database_url, "ingestion-known-actor"
    )
    await engine.dispose()

    await _assert_access_denied_before_load(
        database_url=migrated_database_url,
        workspace_id=uuid4(),
        actor_user_id=provisioned.user_id,
    )
