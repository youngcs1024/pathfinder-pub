from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.domain.actions import SubmitApplicationArgsV1
from app.mock_portal.contracts import (
    MockSubmissionRecord,
    MockSubmissionRequestV1,
    StoredMockSubmission,
)
from app.mock_portal.service import MockPortalService, MockSubmissionConflictError


class _Repository:
    def __init__(self) -> None:
        self.rows = {}

    async def put_if_absent(self, *, request, idempotency_key, payload_digest, external_ref):
        existing = self.rows.get(idempotency_key)
        if existing is not None:
            return StoredMockSubmission(existing, False)
        row = MockSubmissionRecord(
            id=uuid4(),
            workspace_id=request.workspace_id,
            originating_actor_user_id=request.originating_actor_user_id,
            run_id=request.run_id,
            action_intent_id=request.action_intent_id,
            idempotency_key=idempotency_key,
            payload_digest=payload_digest,
            payload=request.model_dump(mode="json"),
            external_ref=external_ref,
            created_at=datetime.now(UTC),
        )
        self.rows[idempotency_key] = row
        return StoredMockSubmission(row, True)

    async def get_by_idempotency_key(self, idempotency_key):
        return self.rows.get(idempotency_key)


def _request(*, cover_letter: str = "Exact draft") -> MockSubmissionRequestV1:
    action_id = uuid4()
    return MockSubmissionRequestV1(
        workspace_id=uuid4(),
        originating_actor_user_id=uuid4(),
        run_id=uuid4(),
        action_intent_id=action_id,
        payload=SubmitApplicationArgsV1(
            job_ref="job",
            resume_document_id=uuid4(),
            answers={},
            cover_letter=cover_letter,
        ),
    )


async def test_same_key_and_payload_returns_stable_ref_without_second_row() -> None:
    repository = _Repository()
    service = MockPortalService(repository)
    request = _request()
    key = str(request.action_intent_id)

    first = await service.submit(request, idempotency_key=key)
    second = await service.submit(request, idempotency_key=key)
    found = await service.get(key)

    assert first.created is True
    assert second.created is False
    assert first.external_ref == second.external_ref == found.external_ref
    assert len(repository.rows) == 1


async def test_same_key_with_different_payload_conflicts() -> None:
    repository = _Repository()
    service = MockPortalService(repository)
    first = _request()
    await service.submit(first, idempotency_key=str(first.action_intent_id))
    conflict = first.model_copy(
        update={"payload": first.payload.model_copy(update={"cover_letter": "Changed draft"})}
    )

    with pytest.raises(MockSubmissionConflictError):
        await service.submit(conflict, idempotency_key=str(first.action_intent_id))
