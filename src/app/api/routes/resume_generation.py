"""Authorized first-draft session commands and reads."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, status

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.api.schemas.resume_generation import (
    SessionCancelResponse,
    SessionCreateV1,
    SessionDetailResponse,
    SessionListItemResponse,
    SessionReceiptResponse,
    VersionDetailResponse,
)
from app.domain.resume_generation import ResumeGenerationService

router = APIRouter(
    prefix="/api/v2/workspaces/{workspace_id}/resume-sessions", tags=["resume-sessions"]
)


def _service(request: Request) -> ResumeGenerationService:
    service = getattr(request.app.state, "resume_generation_service", None)
    if not isinstance(service, ResumeGenerationService):
        raise RuntimeError("resume generation service is unavailable")
    return service


@router.post("", response_model=SessionReceiptResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_session(request: Request, tenant: TenantDependency, body: SessionCreateV1):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    accepted = await _service(request).create(tenant, body, key)
    receipt = accepted.receipt
    assert receipt.resource_id is not None and receipt.run_id is not None
    return SessionReceiptResponse(
        command_id=receipt.command_id,
        session_id=receipt.resource_id,
        run_id=receipt.run_id,
        replayed=accepted.replayed,
    )


@router.get("", response_model=list[SessionListItemResponse])
async def list_sessions(request: Request, tenant: TenantDependency):
    return await _service(request).list_sessions(tenant)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(request: Request, tenant: TenantDependency, session_id: UUID):
    return await _service(request).get_session(tenant, session_id)


@router.get("/{session_id}/versions/{version_id}", response_model=VersionDetailResponse)
async def get_version(
    request: Request, tenant: TenantDependency, session_id: UUID, version_id: UUID
):
    return await _service(request).get_version(tenant, session_id, version_id)


@router.post("/{session_id}/cancel", response_model=SessionCancelResponse)
async def cancel_session(request: Request, tenant: TenantDependency, session_id: UUID):
    cancelled = await _service(request).cancel(tenant, session_id)
    return SessionCancelResponse(run_id=cancelled.run_id, status=cancelled.status.value)
