from uuid import UUID

from sqlalchemy import case, select

from app.db.models import Workspace, WorkspaceMembership
from app.db.session import AsyncSessionFactory, database_session
from app.domain.errors import DomainInvariantError
from app.domain.provisioning import WorkspaceKind, WorkspaceRole
from app.domain.tenancy import ActorWorkspaceMembership, TenantContext


class SqlAlchemyTenantResolver:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        async with database_session(self._session_factory) as session:
            role_value = await session.scalar(
                select(WorkspaceMembership.role).where(
                    WorkspaceMembership.workspace_id == workspace_id,
                    WorkspaceMembership.user_id == actor_user_id,
                    WorkspaceMembership.revoked_at.is_(None),
                )
            )
        if role_value is None:
            return None
        try:
            role = WorkspaceRole(role_value)
        except ValueError:
            raise DomainInvariantError("persisted workspace role is invalid") from None
        return TenantContext(
            workspace_id=workspace_id,
            actor_user_id=actor_user_id,
            role=role,
        )

    async def list_active_memberships(
        self,
        *,
        actor_user_id: UUID,
    ) -> tuple[ActorWorkspaceMembership, ...]:
        async with database_session(self._session_factory) as session:
            rows = (
                await session.execute(
                    select(
                        Workspace.id,
                        Workspace.kind,
                        Workspace.name,
                        WorkspaceMembership.role,
                    )
                    .join(
                        WorkspaceMembership,
                        WorkspaceMembership.workspace_id == Workspace.id,
                    )
                    .where(
                        WorkspaceMembership.user_id == actor_user_id,
                        WorkspaceMembership.revoked_at.is_(None),
                    )
                    .order_by(
                        case((Workspace.kind == WorkspaceKind.PERSONAL.value, 0), else_=1),
                        Workspace.id,
                    )
                )
            ).all()

        memberships: list[ActorWorkspaceMembership] = []
        for workspace_id, kind_value, name, role_value in rows:
            try:
                kind = WorkspaceKind(kind_value)
                role = WorkspaceRole(role_value)
            except ValueError:
                raise DomainInvariantError(
                    "persisted workspace membership values are invalid"
                ) from None
            memberships.append(
                ActorWorkspaceMembership(
                    workspace_id=workspace_id,
                    kind=kind,
                    name=name,
                    role=role,
                )
            )
        return tuple(memberships)
