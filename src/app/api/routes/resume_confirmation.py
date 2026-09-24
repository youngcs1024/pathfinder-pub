"""Version-scoped resume confirmation, history and TeX delivery."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.domain.resume_confirmation import (
    ConfirmationReceiptV1,
    ConfirmVersionV1,
    ResumeConfirmationService,
    VersionSummaryV1,
)

router = APIRouter(
    prefix="/api/v2/workspaces/{workspace_id}/resume-sessions",
    tags=["resume-confirmations"],
)


def _service(request: Request) -> ResumeConfirmationService:
    service = getattr(request.app.state, "resume_confirmation_service", None)
    if not isinstance(service, ResumeConfirmationService):
        raise RuntimeError("resume confirmation service is unavailable")
    return service


@router.get("/{session_id}/versions", response_model=list[VersionSummaryV1])
async def list_versions(request: Request, tenant: TenantDependency, session_id: UUID):
    return await _service(request).list_versions(tenant, session_id)


@router.post(
    "/{session_id}/versions/{version_id}/confirm",
    response_model=ConfirmationReceiptV1,
)
async def confirm_version(
    request: Request,
    tenant: TenantDependency,
    session_id: UUID,
    version_id: UUID,
    body: ConfirmVersionV1,
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return await _service(request).confirm(tenant, session_id, version_id, body, key)


@router.get("/{session_id}/versions/{version_id}/download", response_class=Response)
async def download_version(
    request: Request,
    tenant: TenantDependency,
    session_id: UUID,
    version_id: UUID,
) -> Response:
    payload, digest, number, confirmed = await _service(request).download(
        tenant, session_id, version_id
    )
    state = "confirmed" if confirmed else "draft"
    return Response(
        content=payload,
        media_type="application/x-tex; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="resume-v{number}-{state}.tex"',
            "Cache-Control": "private, no-store",
            "X-Content-SHA256": digest,
            "X-Resume-Version-ID": str(version_id),
            "X-Resume-Status": state,
        },
    )
