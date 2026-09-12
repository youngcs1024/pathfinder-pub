from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.research import QueryText, ResearchOutputV1, ResearchOutputV2
from app.domain.runs import RunMode, RunStatus


class RunAPIModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        from_attributes=True,
        hide_input_in_errors=True,
        strict=True,
    )


class RunCreateRequest(RunAPIModel):
    mode: Literal["research", "application"]
    query: QueryText
    resume_document_id: Annotated[UUID, Field(strict=False)] | None = None


class RunAcceptedResponse(RunAPIModel):
    run_id: UUID
    status: Literal[RunStatus.QUEUED]
    events_url: str = Field(pattern=r"^/api/v1/workspaces/[0-9a-f-]+/runs/[0-9a-f-]+/events$")


class RunUsageBucketResponse(RunAPIModel):
    attempt_count: int = Field(ge=0)
    succeeded_count: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_write_input_tokens: int = Field(ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0, max_digits=20, decimal_places=12)
    currency: Literal["CNY"] | None = None
    cost_available: bool


class RunUsageResponse(RunAPIModel):
    chat: RunUsageBucketResponse
    embedding: RunUsageBucketResponse


class RunDetailResponse(RunAPIModel):
    run_id: UUID
    mode: RunMode
    status: RunStatus
    graph_version: str
    resume_document_id: UUID | None
    result: ResearchOutputV1 | ResearchOutputV2 | None
    error_category: str | None
    cancel_requested_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    usage: RunUsageResponse


class RunCancellationResponse(RunAPIModel):
    run_id: UUID
    status: RunStatus
    cancel_requested_at: datetime | None
