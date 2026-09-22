from uuid import uuid4

import pytest

from app.config import Settings
from app.main import create_app
from tests.unit.api import test_action_intents as actions
from tests.unit.api import test_runs as runs


def production_app(legacy):
    application = create_app(legacy.state.settings)
    for name in ("actor_provider", "tenant_service", "run_service", "approval_service"):
        if hasattr(legacy.state, name):
            setattr(application.state, name, getattr(legacy.state, name))
    return application


@pytest.mark.parametrize("suffix", ["", "/cancel"])
async def test_retired_run_writes_are_safe_410_without_replay_or_business_calls(suffix):
    old, tenant, provider, store = runs._application()
    application = production_app(old)

    async def forbidden(**kwargs):
        pytest.fail("retired route attempted a business write")

    store.create_run = forbidden
    store.cancel_run = forbidden
    path = f"/api/v1/workspaces/{tenant.workspace_id}/runs" + (
        f"/{store.run_id}{suffix}" if suffix else ""
    )
    for _ in range(2):
        response = await runs._request(
            application,
            "POST",
            path,
            json={"unexpected": "private_body_canary"},
            headers={"Idempotency-Key": str(uuid4())},
        )
        assert response.status_code == 410
        assert response.headers["content-type"] == "application/problem+json"
        assert response.json()["status"] == 410
        assert "retired" in response.json()["detail"]
        assert "private_body_canary" not in response.text
        assert "Idempotency-Replayed" not in response.headers
    assert not store.identities and not store.created
    assert provider.calls == 2


async def test_retired_approval_keeps_reads_and_never_decides():
    old, tenant, store = actions._application()
    application = production_app(old)
    path = f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{store.action_id}"
    assert (await runs._request(application, "GET", path)).status_code == 200
    for _ in range(2):
        assert (
            await runs._request(
                application, "POST", path + "/decision", json={"decision": "approve"}
            )
        ).status_code == 410
    assert store.commands == []


@pytest.mark.parametrize("suffix", ["runs", "runs/{id}/cancel", "action-intents/{id}/decision"])
async def test_retired_routes_preserve_authentication_and_tenant_isolation(suffix):
    old, _tenant, _, _ = runs._application()
    app = production_app(old)
    path = f"/api/v1/workspaces/{uuid4()}/" + suffix.format(id=uuid4())
    assert (await runs._request(app, "POST", path)).status_code == 404
    app.state.settings = Settings(
        auth_mode="supabase",
        supabase_project_ref="abcdefghijklmnopqrst",
        supabase_publishable_key="sb_publishable_test-public-key",
    )
    assert (await runs._request(app, "POST", path)).status_code == 401


async def test_production_has_no_mock_routes_or_unimplemented_v2_routes():
    app = create_app(Settings(log_level="ERROR"))
    paths = tuple(app.openapi()["paths"])
    assert not any("mock" in path or path.startswith("/api/v2") for path in paths)
    assert (
        await runs._request(app, "POST", "/internal/mock/applications", json={})
    ).status_code == 404
