from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.domain.actions import SubmitApplicationArgsV1, canonical_json_bytes


class MockPortalContractModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
    )


class MockSubmissionRequestV1(MockPortalContractModel):
    workspace_id: UUID
    originating_actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID
    payload: SubmitApplicationArgsV1


class MockSubmissionResponseV1(MockPortalContractModel):
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


@dataclass(frozen=True, slots=True, repr=False)
class MockSubmissionRecord:
    id: UUID
    workspace_id: UUID
    originating_actor_user_id: UUID
    run_id: UUID
    action_intent_id: UUID
    idempotency_key: str
    payload_digest: str
    payload: dict[str, object]
    external_ref: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredMockSubmission:
    record: MockSubmissionRecord
    created: bool


def mock_submission_payload_digest(request: MockSubmissionRequestV1) -> str:
    payload = request.model_dump(mode="json", round_trip=True)
    canonical = canonical_json_bytes(payload)
    digest = hashlib.sha256(b"pathfinder-mock-submission-v1\0" + canonical).hexdigest()
    return f"sha256:{digest}"


class MockSubmissionRepository(Protocol):
    async def put_if_absent(
        self,
        *,
        request: MockSubmissionRequestV1,
        idempotency_key: str,
        payload_digest: str,
        external_ref: str,
    ) -> StoredMockSubmission: ...

    async def get_by_idempotency_key(self, idempotency_key: str) -> MockSubmissionRecord | None: ...
