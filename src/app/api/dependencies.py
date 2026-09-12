from typing import Annotated, cast
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.contracts import ActorContext, ActorProvider
from app.auth.errors import AccessTokenVerificationError, JwksUnavailableError
from app.config import Settings
from app.domain.approvals import ApprovalService
from app.domain.runs import RunService
from app.domain.tenancy import TenantContext, TenantService
from app.events.contracts import RunEventReader

_bearer = HTTPBearer(auto_error=False)
_BEARER_HEADERS = {"WWW-Authenticate": "Bearer"}


def _application_service(request: Request, name: str) -> object:
    service = getattr(request.app.state, name, None)
    if service is None:
        raise RuntimeError(f"application service is unavailable: {name}")
    return service


async def actor_context(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> ActorContext:
    provider = cast(ActorProvider, _application_service(request, "actor_provider"))
    settings = _application_service(request, "settings")
    if not isinstance(settings, Settings):
        raise RuntimeError("application settings are invalid")
    if settings.auth_mode == "fake":
        return await provider.get_actor(None)
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers=_BEARER_HEADERS,
        )
    try:
        return await provider.get_actor(credentials.credentials)
    except JwksUnavailableError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE) from None
    except AccessTokenVerificationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers=_BEARER_HEADERS,
        ) from None


def run_service(request: Request) -> RunService:
    service = _application_service(request, "run_service")
    if not isinstance(service, RunService):
        raise RuntimeError("application run service is invalid")
    return service


def tenant_service(request: Request) -> TenantService:
    service = _application_service(request, "tenant_service")
    if not isinstance(service, TenantService):
        raise RuntimeError("application tenant service is invalid")
    return service


def run_event_reader(request: Request) -> RunEventReader:
    reader = _application_service(request, "run_event_reader")
    if not isinstance(reader, RunEventReader):
        raise RuntimeError("application run event reader is invalid")
    return reader


def approval_service(request: Request) -> ApprovalService:
    service = _application_service(request, "approval_service")
    if not isinstance(service, ApprovalService):
        raise RuntimeError("application approval service is invalid")
    return service


async def tenant_context(
    workspace_id: UUID,
    actor: Annotated[ActorContext, Depends(actor_context)],
    service: Annotated[TenantService, Depends(tenant_service)],
) -> TenantContext:
    return await service.resolve_tenant(
        workspace_id=workspace_id,
        actor_user_id=actor.user_id,
    )


ActorDependency = Annotated[ActorContext, Depends(actor_context)]
TenantDependency = Annotated[TenantContext, Depends(tenant_context)]
TenantServiceDependency = Annotated[TenantService, Depends(tenant_service)]
RunServiceDependency = Annotated[RunService, Depends(run_service)]
RunEventReaderDependency = Annotated[RunEventReader, Depends(run_event_reader)]
ApprovalServiceDependency = Annotated[ApprovalService, Depends(approval_service)]
