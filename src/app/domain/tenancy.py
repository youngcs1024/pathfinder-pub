from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.domain.errors import DomainNotFoundError
from app.domain.provisioning import WorkspaceKind, WorkspaceRole


@dataclass(frozen=True, slots=True)
class ActorWorkspaceMembership:
    workspace_id: UUID
    kind: WorkspaceKind
    name: str
    role: WorkspaceRole


@dataclass(frozen=True, slots=True, repr=False)
class TenantContext:
    workspace_id: UUID
    actor_user_id: UUID
    role: WorkspaceRole

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, UUID):
            raise TypeError("tenant workspace_id must be a UUID")
        if not isinstance(self.actor_user_id, UUID):
            raise TypeError("tenant actor_user_id must be a UUID")
        if not isinstance(self.role, WorkspaceRole):
            raise TypeError("tenant role must be a WorkspaceRole")


class TenantResolver(Protocol):
    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None: ...

    async def list_active_memberships(
        self,
        *,
        actor_user_id: UUID,
    ) -> tuple[ActorWorkspaceMembership, ...]: ...


class TenantService:
    def __init__(self, resolver: TenantResolver) -> None:
        self._resolver = resolver

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext:
        if not isinstance(workspace_id, UUID) or not isinstance(actor_user_id, UUID):
            raise DomainNotFoundError
        tenant = await self._resolver.resolve_tenant(
            workspace_id=workspace_id,
            actor_user_id=actor_user_id,
        )
        if tenant is None:
            raise DomainNotFoundError
        return tenant

    async def list_active_memberships(
        self,
        *,
        actor_user_id: UUID,
    ) -> tuple[ActorWorkspaceMembership, ...]:
        if not isinstance(actor_user_id, UUID):
            raise DomainNotFoundError
        return await self._resolver.list_active_memberships(actor_user_id=actor_user_id)
