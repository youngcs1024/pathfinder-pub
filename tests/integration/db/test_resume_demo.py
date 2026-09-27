"""Canonical R7.2 fake/off product demonstration."""

import pytest

from tests.integration.resume_product import check_sse, confirm, product, revise, seed

pytestmark = pytest.mark.integration


async def test_resume_demo_complete_flow(migrated_database_url, tmp_path):
    async with product(migrated_database_url, tmp_path / "demo") as rig:
        state = await seed(rig)
        await revise(rig, state)
        await revise(rig, state, lock=True)
        await confirm(rig, state)
        await check_sse(rig, state)
        assert len(state["versions"]) == 3
