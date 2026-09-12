from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Literal
from uuid import UUID, uuid4

from app.mock_portal.contracts import (
    MockSubmissionRepository,
    MockSubmissionRequestV1,
    MockSubmissionResponseV1,
    mock_submission_payload_digest,
)


class MockSubmissionConflictError(Exception):
    """An idempotency key was reused with different persisted payload facts."""


MockPortalFaultPoint = Literal["mock_portal_after_insert_before_response"]
type MockPortalFaultInjector = Callable[[MockPortalFaultPoint], Awaitable[None] | None]


class MockPortalService:
    def __init__(
        self,
        repository: MockSubmissionRepository,
        *,
        fault_injector: MockPortalFaultInjector | None = None,
    ) -> None:
        self._repository = repository
        self._fault_injector = fault_injector

    async def submit(
        self,
        request: MockSubmissionRequestV1,
        *,
        idempotency_key: str,
    ) -> MockSubmissionResponseV1:
        if not isinstance(request, MockSubmissionRequestV1):
            raise TypeError("mock submission request is invalid")
        try:
            parsed_key = UUID(idempotency_key)
        except (TypeError, ValueError):
            raise ValueError("mock submission idempotency key is invalid") from None
        if str(parsed_key) != idempotency_key or parsed_key != request.action_intent_id:
            raise ValueError("mock submission idempotency key does not match action intent")
        digest = mock_submission_payload_digest(request)
        stored = await self._repository.put_if_absent(
            request=request,
            idempotency_key=idempotency_key,
            payload_digest=digest,
            external_ref=f"mock-submission:{uuid4()}",
        )
        if stored.record.payload_digest != digest:
            raise MockSubmissionConflictError
        if stored.created and self._fault_injector is not None:
            injected = self._fault_injector("mock_portal_after_insert_before_response")
            if isawaitable(injected):
                await injected
        return MockSubmissionResponseV1(
            id=stored.record.id,
            workspace_id=stored.record.workspace_id,
            originating_actor_user_id=stored.record.originating_actor_user_id,
            run_id=stored.record.run_id,
            action_intent_id=stored.record.action_intent_id,
            idempotency_key=stored.record.idempotency_key,
            payload_digest=stored.record.payload_digest,
            external_ref=stored.record.external_ref,
            created_at=stored.record.created_at,
            created=stored.created,
        )

    async def get(self, idempotency_key: str) -> MockSubmissionResponseV1 | None:
        try:
            parsed_key = UUID(idempotency_key)
        except (TypeError, ValueError):
            raise ValueError("mock submission idempotency key is invalid") from None
        if str(parsed_key) != idempotency_key:
            raise ValueError("mock submission idempotency key is invalid")
        record = await self._repository.get_by_idempotency_key(idempotency_key)
        if record is None:
            return None
        return MockSubmissionResponseV1(
            id=record.id,
            workspace_id=record.workspace_id,
            originating_actor_user_id=record.originating_actor_user_id,
            run_id=record.run_id,
            action_intent_id=record.action_intent_id,
            idempotency_key=record.idempotency_key,
            payload_digest=record.payload_digest,
            external_ref=record.external_ref,
            created_at=record.created_at,
            created=False,
        )
