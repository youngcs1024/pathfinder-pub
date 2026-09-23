from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, status

from app.api.dependencies import TenantDependency
from app.api.run_request_identity import require_idempotency_key
from app.api.schemas.resume_profiles import (
    ClaimReviewInput,
    ContactEditInput,
    ItemReviewInput,
    PreferenceUpdateInput,
    ProfileCommandResponse,
    ProfileDetailResponse,
    ProfileImportInput,
)
from app.domain.resume_profiles import ResumeProfileService
from app.resume.template_import import ResumeImportPreviewV1

router = APIRouter(prefix="/api/v2/workspaces/{workspace_id}/profiles", tags=["profiles"])


def _service(request: Request) -> ResumeProfileService:
    service = getattr(request.app.state, "resume_profile_service", None)
    if not isinstance(service, ResumeProfileService):
        raise RuntimeError("resume profile service is unavailable")
    return service


def _receipt(accepted) -> ProfileCommandResponse:
    receipt = accepted.receipt
    assert receipt.resource_id is not None
    return ProfileCommandResponse(
        command_id=receipt.command_id,
        resource_id=receipt.resource_id,
        status="completed",
        replayed=accepted.replayed,
    )


@router.post("/import-preview", response_model=ResumeImportPreviewV1)
async def preview_resume(request: Request, tenant: TenantDependency, body: ProfileImportInput):
    return await _service(request).preview(tenant, body.source_tex)


@router.post("/imports", response_model=ProfileCommandResponse, status_code=status.HTTP_201_CREATED)
async def import_resume(request: Request, tenant: TenantDependency, body: ProfileImportInput):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(await _service(request).import_source(tenant, body.source_tex, key))


@router.get("/me", response_model=ProfileDetailResponse)
async def current_profile(request: Request, tenant: TenantDependency):
    return await _service(request).get_me(tenant)


@router.get("/{profile_id}", response_model=ProfileDetailResponse)
async def get_profile(request: Request, tenant: TenantDependency, profile_id: UUID):
    return await _service(request).get_profile(tenant, profile_id)


@router.post(
    "/{profile_id}/contact-edits",
    response_model=ProfileCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def edit_contact(
    request: Request, tenant: TenantDependency, profile_id: UUID, body: ContactEditInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="resume_profile_contact_edit",
            profile_id=profile_id,
            request_id=key,
            payload=body,
        )
    )


@router.post(
    "/{profile_id}/item-reviews",
    response_model=ProfileCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def review_item(
    request: Request, tenant: TenantDependency, profile_id: UUID, body: ItemReviewInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="resume_profile_item_review",
            profile_id=profile_id,
            request_id=key,
            payload=body,
        )
    )


@router.post(
    "/{profile_id}/preference-versions",
    response_model=ProfileCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def update_preferences(
    request: Request, tenant: TenantDependency, profile_id: UUID, body: PreferenceUpdateInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="resume_preference_update",
            profile_id=profile_id,
            request_id=key,
            payload=body,
        )
    )


@router.post(
    "/{profile_id}/claim-reviews",
    response_model=ProfileCommandResponse,
    status_code=status.HTTP_201_CREATED,
)
async def review_claim(
    request: Request, tenant: TenantDependency, profile_id: UUID, body: ClaimReviewInput
):
    key = require_idempotency_key(request.headers.getlist("Idempotency-Key"))
    return _receipt(
        await _service(request).command(
            tenant,
            kind="resume_claim_review",
            profile_id=profile_id,
            request_id=key,
            payload=body,
        )
    )
