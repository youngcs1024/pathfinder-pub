from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Workspace, WorkspaceMembership
from app.db.session import AsyncSessionFactory, transaction
from app.domain.errors import DomainForbiddenError, DomainInvariantError
from app.domain.provisioning import (
    ProvisionedPersonalWorkspace,
    WorkspaceKind,
    WorkspaceRole,
)


class SqlAlchemyProvisioningStore:
    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def provision_personal_workspace(
        self,
        *,
        auth_subject: str,
        workspace_name: str,
    ) -> ProvisionedPersonalWorkspace:
        async with transaction(self._session_factory) as session:
            user_id = await self._get_or_create_user(session, auth_subject)
            workspace_id = await self._get_or_create_workspace(
                session,
                user_id=user_id,
                workspace_name=workspace_name,
            )
            membership_id = await self._get_or_create_admin_membership(
                session,
                workspace_id=workspace_id,
                user_id=user_id,
            )

        return ProvisionedPersonalWorkspace(
            user_id=user_id,
            workspace_id=workspace_id,
            membership_id=membership_id,
            kind=WorkspaceKind.PERSONAL,
            role=WorkspaceRole.ADMIN,
        )

    @staticmethod
    async def _get_or_create_user(session: AsyncSession, auth_subject: str) -> UUID:
        result = await session.execute(
            insert(User)
            .values(id=uuid4(), auth_subject=auth_subject)
            .on_conflict_do_nothing(index_elements=[User.auth_subject])
            .returning(User.id)
        )
        user_id = result.scalar_one_or_none()
        if user_id is not None:
            return user_id

        existing_id = await session.scalar(select(User.id).where(User.auth_subject == auth_subject))
        if existing_id is None:
            raise DomainInvariantError("provisioned user could not be resolved")
        return existing_id

    @staticmethod
    async def _get_or_create_workspace(
        session: AsyncSession,
        *,
        user_id: UUID,
        workspace_name: str,
    ) -> UUID:
        result = await session.execute(
            insert(Workspace)
            .values(
                id=uuid4(),
                kind=WorkspaceKind.PERSONAL.value,
                name=workspace_name,
                created_by_user_id=user_id,
            )
            .on_conflict_do_nothing(
                index_elements=[Workspace.created_by_user_id],
                index_where=text("kind = 'personal'"),
            )
            .returning(Workspace.id)
        )
        workspace_id = result.scalar_one_or_none()
        if workspace_id is not None:
            return workspace_id

        existing_id = await session.scalar(
            select(Workspace.id).where(
                Workspace.created_by_user_id == user_id,
                Workspace.kind == WorkspaceKind.PERSONAL.value,
            )
        )
        if existing_id is None:
            raise DomainInvariantError("personal workspace could not be resolved")
        return existing_id

    @staticmethod
    async def _get_or_create_admin_membership(
        session: AsyncSession,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> UUID:
        result = await session.execute(
            insert(WorkspaceMembership)
            .values(
                id=uuid4(),
                workspace_id=workspace_id,
                user_id=user_id,
                role=WorkspaceRole.ADMIN.value,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    WorkspaceMembership.workspace_id,
                    WorkspaceMembership.user_id,
                ]
            )
            .returning(
                WorkspaceMembership.id,
                WorkspaceMembership.role,
                WorkspaceMembership.revoked_at,
            )
        )
        membership = result.one_or_none()
        if membership is None:
            membership = (
                await session.execute(
                    select(
                        WorkspaceMembership.id,
                        WorkspaceMembership.role,
                        WorkspaceMembership.revoked_at,
                    ).where(
                        WorkspaceMembership.workspace_id == workspace_id,
                        WorkspaceMembership.user_id == user_id,
                    )
                )
            ).one_or_none()

        if membership is None:
            raise DomainInvariantError("admin membership could not be resolved")
        if membership.revoked_at is not None:
            raise DomainForbiddenError("creator membership has been revoked")
        if membership.role != WorkspaceRole.ADMIN.value:
            raise DomainInvariantError("creator membership is not an active admin membership")
        return membership.id
