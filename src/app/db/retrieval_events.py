from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import update

from app.db.models import Run, RunEvent
from app.db.session import AsyncSessionFactory, transaction
from app.domain.errors import DomainInvariantError
from app.events.contracts import CURRENT_RUN_EVENT_VERSION, RunEventType


class SqlAlchemyRetrievalEventRecorder:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def record_retrieved(
        self,
        *,
        workspace_id: UUID,
        run_id: UUID,
        actor_user_id: UUID,
        tool_invocation_id: UUID,
        document_ids: tuple[UUID, ...],
        chunk_ids: tuple[UUID, ...],
        embedding_profile: str,
    ) -> None:
        async with transaction(self._session_factory) as session:
            seq = await session.scalar(
                update(Run)
                .where(Run.workspace_id == workspace_id, Run.id == run_id)
                .values(next_event_seq=Run.next_event_seq + 1)
                .returning(Run.next_event_seq - 1)
            )
            if seq is None:
                raise DomainInvariantError("retrieval event run is unavailable")
            session.add(
                RunEvent(
                    id=uuid4(),
                    workspace_id=workspace_id,
                    run_id=run_id,
                    actor_user_id=actor_user_id,
                    seq=seq,
                    type=RunEventType.RAG_RETRIEVED.value,
                    version=CURRENT_RUN_EVENT_VERSION,
                    payload={
                        "tool_invocation_id": str(tool_invocation_id),
                        "result_count": len(chunk_ids),
                        "document_ids": [str(value) for value in document_ids],
                        "chunk_ids": [str(value) for value in chunk_ids],
                        "embedding_profile": embedding_profile,
                    },
                )
            )
