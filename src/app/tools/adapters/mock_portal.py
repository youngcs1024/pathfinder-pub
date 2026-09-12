from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.actions import SubmitApplicationArgsV1
from app.mock_portal.contracts import MockSubmissionRequestV1, mock_submission_payload_digest


class MockPortalTransportError(Exception):
    """The worker could not safely classify an HTTP submission outcome."""


class MockPortalConflictError(Exception):
    """The target rejected reuse of an idempotency key with different payload."""


class MockPortalRejectedError(Exception):
    """The target explicitly rejected the request before creating a business effect."""


@dataclass(frozen=True, slots=True)
class MockPortalEvidence:
    external_ref: str
    payload_digest: str


class _MockPortalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    id: UUID
    workspace_id: UUID
    originating_actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID
    idempotency_key: str
    payload_digest: str
    external_ref: str
    created_at: datetime
    created: bool


class MockPortalHTTPAdapter:
    def __init__(self, client: httpx.AsyncClient) -> None:
        if not isinstance(client, httpx.AsyncClient):
            raise TypeError("mock portal adapter requires an AsyncClient")
        self._client = client

    async def submit(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        action_intent_id: UUID,
        idempotency_key: str,
        payload: SubmitApplicationArgsV1,
    ) -> str:
        body = {
            "workspace_id": str(workspace_id),
            "originating_actor_user_id": str(actor_user_id),
            "run_id": str(run_id),
            "action_intent_id": str(action_intent_id),
            "payload": payload.model_dump(mode="json", round_trip=True),
        }
        try:
            response = await self._client.post(
                "/internal/mock-portal/submissions",
                headers={"Idempotency-Key": idempotency_key},
                json=body,
            )
        except httpx.HTTPError:
            raise MockPortalTransportError from None
        if response.status_code == 409:
            raise MockPortalConflictError
        if response.status_code in {400, 422}:
            raise MockPortalRejectedError
        if response.status_code not in {200, 201}:
            raise MockPortalTransportError
        try:
            result = _MockPortalResponse.model_validate_json(response.content, strict=True)
        except (TypeError, ValueError, ValidationError):
            raise MockPortalTransportError from None
        expected_request = MockSubmissionRequestV1(
            workspace_id=workspace_id,
            originating_actor_user_id=actor_user_id,
            run_id=run_id,
            action_intent_id=action_intent_id,
            payload=payload,
        )
        if (
            result.workspace_id != workspace_id
            or result.originating_actor_user_id != actor_user_id
            or result.run_id != run_id
            or result.action_intent_id != action_intent_id
            or result.idempotency_key != idempotency_key
            or result.payload_digest != mock_submission_payload_digest(expected_request)
        ):
            raise MockPortalTransportError
        return result.external_ref

    async def get_by_idempotency_key(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
        run_id: UUID,
        action_intent_id: UUID,
        idempotency_key: str,
        payload: SubmitApplicationArgsV1,
    ) -> MockPortalEvidence | None:
        try:
            response = await self._client.get(
                f"/internal/mock-portal/submissions/by-idempotency-key/{idempotency_key}"
            )
        except httpx.HTTPError:
            raise MockPortalTransportError from None
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise MockPortalTransportError
        try:
            result = _MockPortalResponse.model_validate_json(response.content, strict=True)
        except (TypeError, ValueError, ValidationError):
            raise MockPortalTransportError from None
        expected_request = MockSubmissionRequestV1(
            workspace_id=workspace_id,
            originating_actor_user_id=actor_user_id,
            run_id=run_id,
            action_intent_id=action_intent_id,
            payload=payload,
        )
        expected_digest = mock_submission_payload_digest(expected_request)
        if (
            result.workspace_id != workspace_id
            or result.originating_actor_user_id != actor_user_id
            or result.run_id != run_id
            or result.action_intent_id != action_intent_id
            or result.idempotency_key != idempotency_key
            or result.payload_digest != expected_digest
        ):
            raise MockPortalTransportError
        return MockPortalEvidence(
            external_ref=result.external_ref,
            payload_digest=result.payload_digest,
        )
