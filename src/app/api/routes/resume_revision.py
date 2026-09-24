"""Authorized classified feedback and session-scoped review commands."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Request, status

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.api.schemas.resume_revision import (
    FactReviewV1,
    FeedbackInput,
    FeedbackReceiptResponse,
    FeedbackResponse,
    LockChangeV1,
    QuestionResponse,
    UserFactResponse,
)
from app.domain.resume_revision import ResumeRevisionService

router = APIRouter(
    prefix="/api/v2/workspaces/{workspace_id}/resume-sessions",
    tags=["resume-feedback"],
)


def _service(request: Request) -> ResumeRevisionService:
    service = getattr(request.app.state, "resume_revision_service", None)
    if not isinstance(service, ResumeRevisionService):
        raise RuntimeError("resume revision service is unavailable")
    return service


def _receipt(accepted) -> FeedbackReceiptResponse:
    receipt = accepted.receipt
    assert receipt.resource_id is not None
    return FeedbackReceiptResponse(
        command_id=receipt.command_id,
        feedback_id=receipt.resource_id,
        run_id=receipt.run_id,
        status=receipt.status,
        replayed=accepted.replayed,
    )


@router.post(
    "/{session_id}/feedback",
    response_model=FeedbackReceiptResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_feedback(
    request: Request,
    tenant: TenantDependency,
    session_id: UUID,
    body: Annotated[FeedbackInput, Body(discriminator="kind")],
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(await _service(request).command(tenant, session_id, body, key))


@router.get("/{session_id}/feedback", response_model=list[FeedbackResponse])
async def list_feedback(request: Request, tenant: TenantDependency, session_id: UUID):
    return await _service(request).list_feedback(tenant, session_id)


@router.get("/{session_id}/questions", response_model=list[QuestionResponse])
async def list_questions(request: Request, tenant: TenantDependency, session_id: UUID):
    return await _service(request).questions(tenant, session_id)


@router.get("/{session_id}/user-facts", response_model=list[UserFactResponse])
async def list_user_facts(request: Request, tenant: TenantDependency, session_id: UUID):
    return await _service(request).list_user_facts(tenant, session_id)


@router.post(
    "/{session_id}/user-facts/{fact_version_id}/reviews",
    response_model=FeedbackReceiptResponse,
)
async def review_user_fact(
    request: Request,
    tenant: TenantDependency,
    session_id: UUID,
    fact_version_id: UUID,
    body: FactReviewV1,
):
    if fact_version_id != body.fact_version_id:
        from app.domain.errors import DomainValidationError

        raise DomainValidationError("fact identity mismatch")
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(await _service(request).command(tenant, session_id, body, key))


@router.post("/{session_id}/locks", response_model=FeedbackReceiptResponse)
async def change_lock(
    request: Request,
    tenant: TenantDependency,
    session_id: UUID,
    body: LockChangeV1,
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(await _service(request).command(tenant, session_id, body, key))
