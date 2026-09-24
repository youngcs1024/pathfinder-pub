"""Reader substitution executes ingestion and fact preparation in the real worker."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from app.db.material import SqlAlchemyMaterialStore
from app.db.models import MaterialSnapshotFile
from app.db.project_facts import SqlAlchemyProjectFactStore
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantService
from app.material.aliases import MaterialAlias, MaterialAliasRegistry
from app.material.reader import MaterialReadError, read_alias
from tests.integration.db.test_material_snapshots import _fixture_aliases, _runner

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "failure", [None, "partial_read", "file_unavailable", "snapshot_too_large"]
)
async def test_source_substitution_uses_same_fact_pipeline(
    migrated_database_url, tmp_path, failure
):
    engine = create_database_engine(SecretStr(migrated_database_url))
    sessions = create_session_factory(engine)
    try:
        owner = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace("r61-reader")
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=owner.workspace_id,
            actor_user_id=owner.user_id,
        )
        aliases, _ = _fixture_aliases(tmp_path, tenant.workspace_id)
        git = aliases.get("code", tenant.workspace_id)
        # Same committed content through a different authorized reader.
        aliases = MaterialAliasRegistry(
            (git, MaterialAlias("copy", "file", git.root, git.paths, (tenant.workspace_id,)))
        )
        store = SqlAlchemyMaterialStore(sessions, aliases)
        calls = []

        def reader(alias):
            calls.append(alias.kind)
            result = read_alias(alias)
            if failure == "partial_read":
                return replace(result, omitted_files=("main.py",))
            if failure:
                raise MaterialReadError(failure)
            return result

        runner = _runner(sessions, store, source_reader=reader)
        catalogs = []
        snapshots = []
        for alias in ("code", "copy"):
            project = await store.create_project(tenant, alias, uuid4())
            source = await store.create_source(tenant, project["id"], alias, uuid4())
            accepted = await store.submit_import(tenant, project["id"], (source["id"],), uuid4())
            assert await runner.run_once(asyncio.Event())
            progress = await store.get_import(tenant, project["id"], accepted.receipt.resource_id)
            assert progress["status"] == ("failed" if failure else "completed")
            if not failure:
                catalogs.append(
                    await SqlAlchemyProjectFactStore(sessions).current_facts(tenant, project["id"])
                )
                snapshots.extend(progress["snapshots"])
        assert calls == ["git", "file"]
        async with sessions() as db:
            files = (await db.scalars(select(MaterialSnapshotFile))).all()
            if failure:
                assert files == []
            else:
                assert len(files) == 4
                assert len({item.snapshot_id for item in files}) == 2
                assert len({item.document_id for item in files}) == 2
                assert catalogs[0]["issues"] == catalogs[1]["issues"]
                assert catalogs[0]["fact_set_id"] != catalogs[1]["fact_set_id"]
                assert snapshots[0]["manifest_digest"] != snapshots[1]["manifest_digest"]
    finally:
        await engine.dispose()
