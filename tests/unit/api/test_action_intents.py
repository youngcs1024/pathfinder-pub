from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest

from app.auth.contracts import ActorContext
from app.config import Settings
from app.domain.approvals import ApprovalService, ApprovalStatus
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext, TenantService
from tests.legacy_app import create_app


class _ActorProvider:
    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id

    async def get_actor(self, access_token: str | None = None) -> ActorContext:
        del access_token
        return ActorContext(user_id=self.user_id, subject="action-reviewer")


class _TenantResolver:
    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant

    async def resolve_tenant(self, *, workspace_id: UUID, actor_user_id: UUID):
        if workspace_id == self.tenant.workspace_id and actor_user_id == self.tenant.actor_user_id:
            return self.tenant
        return None


class _ApprovalStore:
    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant
        self.action_id = uuid4()
        self.request_id = uuid4()
        self.decision_id = uuid4()
        self.commands = []
        now = datetime(2026, 8, 24, 12, tzinfo=UTC)
        digest = f"sha256:{'a' * 64}"
        self.review = SimpleNamespace(
            intent=SimpleNamespace(
                action_intent_id=self.action_id,
                action_key="submit_application",
                action_revision=1,
                tool_name="submit_mock_application",
                effect="irreversible",
                args_snapshot={"job_ref": "job"},
                canonicalization_version=1,
                args_digest=digest,
                target_snapshot={"provider": "mock_portal"},
                target_canonicalization_version=1,
                target_digest=digest,
                approval_binding_version=1,
                approval_binding_digest=digest,
                status="proposed",
                recovery_attempts=0,
                result=None,
                evidence=None,
            ),
            approval_request=SimpleNamespace(
                request_id=self.request_id,
                status=ApprovalStatus.PENDING,
                version=1,
                expires_at=now + timedelta(hours=1),
                args_digest=digest,
                target_digest=digest,
                approval_binding_version=1,
                approval_binding_digest=digest,
                policy_version=1,
                policy_snapshot={
                    "eligible_roles": ["reviewer", "admin"],
                    "required_approvals": 1,
                    "separation_of_duty": False,
                },
            ),
            decision=None,
        )
        self.now = now

    async def get_action_review(self, *, tenant, action_intent_id):
        assert tenant == self.tenant and action_intent_id == self.action_id
        return self.review

    async def decide(self, command):
        self.commands.append(command)
        return SimpleNamespace(
            approval_decision_id=self.decision_id,
            approval_request_id=self.request_id,
            action_intent_id=self.action_id,
            decision=command.decision,
            reason=command.reason,
            decided_at=self.now,
        )


def _application():
    user_id = uuid4()
    tenant = TenantContext(uuid4(), user_id, WorkspaceRole.REVIEWER)
    store = _ApprovalStore(tenant)
    application = create_app(Settings(log_level="ERROR"))
    application.state.actor_provider = _ActorProvider(user_id)
    application.state.tenant_service = TenantService(_TenantResolver(tenant))
    application.state.approval_service = ApprovalService(store)
    return application, tenant, store


async def _request(application, method: str, path: str, json=None):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://testserver"
    ) as client:
        return await client.request(method, path, json=json)


async def test_action_review_returns_exact_snapshots_binding_and_request() -> None:
    application, tenant, store = _application()
    response = await _request(
        application,
        "GET",
        f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{store.action_id}",
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["action_intent_id"] == str(store.action_id)
    assert payload["args_snapshot"] == {"job_ref": "job"}
    assert payload["target_snapshot"] == {"provider": "mock_portal"}
    assert payload["approval_request"]["version"] == 1
    assert payload["decision"] is None
    assert payload["recovery_attempts"] == 0
    assert payload["manual_review_required"] is False
    assert payload["manual_review_instruction"] is None


async def test_unknown_action_review_exposes_stable_manual_verification_instruction() -> None:
    application, tenant, store = _application()
    store.review.intent.status = "outcome_unknown"
    store.review.intent.recovery_attempts = 3
    store.review.intent.evidence = {"classification": "recovery_exhausted"}
    response = await _request(
        application,
        "GET",
        f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{store.action_id}",
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["action_intent_id"] == str(store.action_id)
    assert payload["status"] == "outcome_unknown"
    assert payload["recovery_attempts"] == 3
    assert payload["manual_review_required"] is True
    assert "do not resubmit or retry" in payload["manual_review_instruction"]


async def test_decision_body_forbids_trusted_fields_and_bounds_reason() -> None:
    application, tenant, store = _application()
    path = f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{store.action_id}/decision"
    valid = await _request(
        application,
        "POST",
        path,
        json={"decision": "reject", "expected_version": 1, "reason": "not now"},
    )
    assert valid.status_code == 200
    assert valid.json()["approval_decision_id"] == str(store.decision_id)
    assert len(store.commands) == 1
    for extra in ("workspace_id", "actor_user_id", "role", "target", "binding"):
        response = await _request(
            application,
            "POST",
            path,
            json={"decision": "approve", "expected_version": 1, extra: "forged"},
        )
        assert response.status_code == 422
    too_long = await _request(
        application,
        "POST",
        path,
        json={"decision": "approve", "expected_version": 1, "reason": "x" * 1001},
    )
    assert too_long.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        {"decision": "allow", "expected_version": 1},
        {"decision": "approve", "expected_version": 0},
    ],
)
async def test_decision_rejects_invalid_value_or_version(payload: object) -> None:
    application, tenant, store = _application()
    response = await _request(
        application,
        "POST",
        f"/api/v1/workspaces/{tenant.workspace_id}/action-intents/{store.action_id}/decision",
        json=payload,
    )
    assert response.status_code == 422
    assert store.commands == []
