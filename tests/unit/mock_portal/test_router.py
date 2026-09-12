from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import httpx
from fastapi import FastAPI

from app.domain.actions import SubmitApplicationArgsV1
from app.mock_portal.contracts import MockSubmissionRecord, StoredMockSubmission
from app.mock_portal.router import router
from app.mock_portal.service import MockPortalService


class _Repository:
    def __init__(self) -> None:
        self.rows: dict[str, MockSubmissionRecord] = {}

    async def put_if_absent(
        self,
        *,
        request,
        idempotency_key,
        payload_digest,
        external_ref,
    ) -> StoredMockSubmission:
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
            payload=request.model_dump(mode="json", round_trip=True),
            external_ref=external_ref,
            created_at=datetime.now(UTC),
        )
        self.rows[idempotency_key] = row
        return StoredMockSubmission(row, True)

    async def get_by_idempotency_key(self, idempotency_key: str) -> MockSubmissionRecord | None:
        return self.rows.get(idempotency_key)


def _application(repository: _Repository) -> FastAPI:
    application = FastAPI()
    application.state.mock_portal_service = MockPortalService(repository)
    application.include_router(router)
    return application


def _body() -> tuple[dict[str, object], str]:
    action_id = uuid4()
    body = {
        "workspace_id": str(uuid4()),
        "originating_actor_user_id": str(uuid4()),
        "run_id": str(uuid4()),
        "action_intent_id": str(action_id),
        "payload": SubmitApplicationArgsV1(
            job_ref="synthetic-job",
            resume_document_id=uuid4(),
            answers={},
            cover_letter="Synthetic exact draft",
        ).model_dump(mode="json", round_trip=True),
    }
    return body, str(action_id)


async def test_real_fastapi_json_boundary_accepts_canonical_uuid_strings() -> None:
    repository = _Repository()
    body, idempotency_key = _body()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_application(repository)),
        base_url="http://testserver",
    ) as client:
        first = await client.post(
            "/internal/mock-portal/submissions",
            headers={"Idempotency-Key": idempotency_key},
            json=body,
        )
        second = await client.post(
            "/internal/mock-portal/submissions",
            headers={"Idempotency-Key": idempotency_key},
            json=body,
        )

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert len(repository.rows) == 1


async def test_invalid_json_shape_fails_closed_without_repository_write() -> None:
    repository = _Repository()
    body, idempotency_key = _body()
    body["workspace_id"] = "not-a-uuid"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_application(repository)),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/internal/mock-portal/submissions",
            headers={"Idempotency-Key": idempotency_key},
            json=body,
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "invalid submission"}
    assert repository.rows == {}
