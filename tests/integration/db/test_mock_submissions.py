from __future__ import annotations

from datetime import timedelta

import pytest

from app.db.action_execution import SqlAlchemyActionExecutionStore
from app.db.mock_submissions import SqlAlchemyMockSubmissionRepository
from app.mock_portal.contracts import MockSubmissionRequestV1
from app.mock_portal.service import MockPortalService, MockSubmissionConflictError
from tests.integration.db.test_action_execution import NOW, _approved

pytestmark = pytest.mark.integration


async def test_mock_submission_is_postgres_idempotent_and_queryable(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "mock-portal")
    try:
        prepared = await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
            identity, now=NOW + timedelta(minutes=2)
        )
        request = MockSubmissionRequestV1(
            workspace_id=prepared.workspace_id,
            originating_actor_user_id=prepared.originating_actor_user_id,
            run_id=prepared.run_id,
            action_intent_id=prepared.action_intent_id,
            payload=prepared.args,
        )
        service = MockPortalService(SqlAlchemyMockSubmissionRepository(runtime.sessions))
        first = await service.submit(request, idempotency_key=prepared.idempotency_key)
        second = await service.submit(request, idempotency_key=prepared.idempotency_key)
        found = await service.get(prepared.idempotency_key)
        assert first.created is True
        assert second.created is False
        assert first.external_ref == second.external_ref == found.external_ref

        changed = request.model_copy(
            update={"payload": request.payload.model_copy(update={"cover_letter": "tampered"})}
        )
        with pytest.raises(MockSubmissionConflictError):
            await service.submit(changed, idempotency_key=prepared.idempotency_key)
    finally:
        await engine.dispose()


async def test_mock_submission_commit_survives_injected_response_loss(
    migrated_database_url: str,
) -> None:
    engine, runtime, identity = await _approved(migrated_database_url, "response-loss")

    def lose_response(_point: str) -> None:
        raise RuntimeError("response lost")

    try:
        prepared = await SqlAlchemyActionExecutionStore(runtime.sessions).prepare_execution(
            identity, now=NOW + timedelta(minutes=2)
        )
        request = MockSubmissionRequestV1(
            workspace_id=prepared.workspace_id,
            originating_actor_user_id=prepared.originating_actor_user_id,
            run_id=prepared.run_id,
            action_intent_id=prepared.action_intent_id,
            payload=prepared.args,
        )
        service = MockPortalService(
            SqlAlchemyMockSubmissionRepository(runtime.sessions),
            fault_injector=lose_response,
        )
        with pytest.raises(RuntimeError, match="response lost"):
            await service.submit(request, idempotency_key=prepared.idempotency_key)
        found = await service.get(prepared.idempotency_key)
        assert found is not None
        assert found.action_intent_id == prepared.action_intent_id
    finally:
        await engine.dispose()
