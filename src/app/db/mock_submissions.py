from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db.models import MockSubmission
from app.db.session import AsyncSessionFactory, database_session, transaction
from app.domain.errors import DomainInvariantError
from app.mock_portal.contracts import (
    MockSubmissionRecord,
    MockSubmissionRequestV1,
    StoredMockSubmission,
)


def _record(row: MockSubmission) -> MockSubmissionRecord:
    return MockSubmissionRecord(
        id=row.id,
        workspace_id=row.workspace_id,
        originating_actor_user_id=row.originating_actor_user_id,
        run_id=row.run_id,
        action_intent_id=row.action_intent_id,
        idempotency_key=row.idempotency_key,
        payload_digest=row.payload_digest,
        payload=dict(row.payload),
        external_ref=row.external_ref,
        created_at=row.created_at,
    )


class SqlAlchemyMockSubmissionRepository:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def put_if_absent(
        self,
        *,
        request: MockSubmissionRequestV1,
        idempotency_key: str,
        payload_digest: str,
        external_ref: str,
    ) -> StoredMockSubmission:
        payload = request.model_dump(mode="json", round_trip=True)
        async with transaction(self._session_factory) as session:
            inserted_id = await session.scalar(
                insert(MockSubmission)
                .values(
                    workspace_id=request.workspace_id,
                    originating_actor_user_id=request.originating_actor_user_id,
                    run_id=request.run_id,
                    action_intent_id=request.action_intent_id,
                    idempotency_key=idempotency_key,
                    payload_digest=payload_digest,
                    payload=payload,
                    external_ref=external_ref,
                )
                .on_conflict_do_nothing(index_elements=[MockSubmission.idempotency_key])
                .returning(MockSubmission.id)
            )
            row = await session.scalar(
                select(MockSubmission)
                .where(MockSubmission.idempotency_key == idempotency_key)
                .with_for_update()
            )
            if row is None:
                raise DomainInvariantError("mock submission insert did not converge")
            if (
                row.workspace_id != request.workspace_id
                or row.originating_actor_user_id != request.originating_actor_user_id
                or row.run_id != request.run_id
                or row.action_intent_id != request.action_intent_id
                or row.payload != payload
            ):
                # The service maps a differing digest to HTTP 409; identity conflicts are
                # equally unsafe and are represented by a deliberately nonmatching digest.
                row_record = _record(row)
                return StoredMockSubmission(record=row_record, created=False)
            return StoredMockSubmission(record=_record(row), created=inserted_id is not None)

    async def get_by_idempotency_key(self, idempotency_key: str) -> MockSubmissionRecord | None:
        async with database_session(self._session_factory) as session:
            row = await session.scalar(
                select(MockSubmission).where(MockSubmission.idempotency_key == idempotency_key)
            )
            return _record(row) if row is not None else None
