import hashlib

from fastapi import APIRouter, status

from app.api.dependencies import ActorDependency, TenantServiceDependency
from app.api.schemas.me import ActorWorkspaceResponse, MeResponse

router = APIRouter(prefix="/api/v1", tags=["identity"])


def _subject_summary(subject: str) -> str:
    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


@router.get("/me", response_model=MeResponse, status_code=status.HTTP_200_OK)
async def get_me(
    actor: ActorDependency,
    tenant_service: TenantServiceDependency,
) -> MeResponse:
    memberships = await tenant_service.list_active_memberships(actor_user_id=actor.user_id)
    return MeResponse(
        user_id=actor.user_id,
        subject_summary=_subject_summary(actor.subject),
        workspaces=[
            ActorWorkspaceResponse(
                workspace_id=membership.workspace_id,
                kind=membership.kind,
                name=membership.name,
                role=membership.role,
            )
            for membership in memberships
        ],
    )
