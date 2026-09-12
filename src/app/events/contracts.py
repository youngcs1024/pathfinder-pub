from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from app.domain.tenancy import TenantContext


class RunEventType(StrEnum):
    RUN_CREATED = "run.created"
    RUN_STATUS_CHANGED = "run.status_changed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    JOB_LEASE_EXPIRED = "job.lease_expired"
    JOB_DEAD = "job.dead"
    AGENT_PLAN_CREATED = "agent.plan.created"
    AGENT_RESEARCH_STARTED = "agent.research.started"
    SOURCE_DISCOVERED = "source.discovered"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    REPORT_COMPLETED = "report.completed"
    RAG_RETRIEVED = "rag.retrieved"
    ACTION_PROPOSED = "action.proposed"
    APPROVAL_EXPIRED = "approval.expired"
    APPROVAL_DECIDED = "approval.decided"
    ACTION_CANCELLED = "action.cancelled"
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"
    ACTION_OUTCOME_UNKNOWN = "action.outcome_unknown"


CURRENT_RUN_EVENT_VERSION = 1

TERMINAL_RUN_EVENT_TYPES = frozenset(
    {
        RunEventType.RUN_COMPLETED,
        RunEventType.RUN_FAILED,
        RunEventType.RUN_CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class RunEventRecord:
    run_id: UUID
    seq: int
    type: RunEventType
    version: int
    payload: dict[str, object]
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class RunEventBatch:
    events: tuple[RunEventRecord, ...]
    high_watermark: int
    run_is_terminal: bool


@runtime_checkable
class RunEventReader(Protocol):
    async def read_after(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        after_seq: int,
        limit: int,
    ) -> RunEventBatch: ...
