"""Typed v2 feedback and R5.1 review projections."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.domain.resume_revision import (
    AnswerFeedbackV1,
    ContentFeedbackV1,
    FactFeedbackV1,
    FactReviewV1,
    LockChangeV1,
    PreferenceFeedbackV1,
)


class RevisionApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


type FeedbackInput = ContentFeedbackV1 | AnswerFeedbackV1 | PreferenceFeedbackV1 | FactFeedbackV1


class FeedbackReceiptResponse(RevisionApiModel):
    command_id: UUID
    feedback_id: UUID
    run_id: UUID | None
    status: Literal["queued", "completed"]
    replayed: bool


class FeedbackResponse(RevisionApiModel):
    feedback_id: UUID
    kind: str
    scope: str
    run_id: UUID | None
    run_status: str | None
    base_version_id: UUID | None
    target_version_id: UUID | None
    normalized: dict[str, object]
    questions: list[str]
    created_at: datetime


class QuestionResponse(RevisionApiModel):
    id: UUID
    text: str
    source_run_id: UUID
    version_id: UUID | None


class UserFactResponse(RevisionApiModel):
    fact_id: UUID
    fact_version_id: UUID
    project_id: UUID
    scope: Literal["session", "project"]
    claim: str
    kind: Literal["implementation", "plan", "experiment", "personal_statement"]
    conditions: dict[str, object]
    review_status: Literal["pending", "confirmed", "rejected"]
    source: Literal["user_attestation"]
    attested_at: datetime | None


__all__ = (
    "AnswerFeedbackV1",
    "ContentFeedbackV1",
    "FactFeedbackV1",
    "FactReviewV1",
    "FeedbackInput",
    "FeedbackReceiptResponse",
    "FeedbackResponse",
    "LockChangeV1",
    "PreferenceFeedbackV1",
    "QuestionResponse",
    "UserFactResponse",
)
