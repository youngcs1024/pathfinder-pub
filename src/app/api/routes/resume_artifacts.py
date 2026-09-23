"""Authorized metadata and byte delivery for existing TeX artifacts."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, Response

from app.api.dependencies import TenantDependency
from app.domain.resume_artifacts import ResumeArtifactService, TexArtifactInfoV1

router = APIRouter(prefix="/api/v2/workspaces/{workspace_id}/artifacts", tags=["resume-artifacts"])


def _service(request: Request) -> ResumeArtifactService:
    service = getattr(request.app.state, "resume_artifact_service", None)
    if not isinstance(service, ResumeArtifactService):
        raise RuntimeError("resume artifact service is unavailable")
    return service


@router.get("/{artifact_id}", response_model=TexArtifactInfoV1)
async def artifact_info(
    request: Request, tenant: TenantDependency, artifact_id: UUID
) -> TexArtifactInfoV1:
    return await _service(request).get_info(tenant, artifact_id)


@router.get("/{artifact_id}/download", response_class=Response)
async def download_artifact(
    request: Request, tenant: TenantDependency, artifact_id: UUID
) -> Response:
    payload, digest = await _service(request).get_bytes(tenant, artifact_id)
    return Response(
        content=payload,
        media_type="application/x-tex; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="resume-{artifact_id}.tex"',
            "Cache-Control": "private, no-store",
            "X-Content-SHA256": digest,
        },
    )
