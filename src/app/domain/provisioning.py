from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from app.domain.errors import DomainValidationError

PERSONAL_WORKSPACE_NAME = "Personal Workspace"


class WorkspaceKind(StrEnum):
    PERSONAL = "personal"
    TEAM = "team"


class WorkspaceRole(StrEnum):
    MEMBER = "member"
    REVIEWER = "reviewer"
    ADMIN = "admin"


@dataclass(frozen=True)
class ProvisionedPersonalWorkspace:
    user_id: UUID
    workspace_id: UUID
    membership_id: UUID
    kind: WorkspaceKind
    role: WorkspaceRole


class ProvisioningStore(Protocol):
    async def provision_personal_workspace(
        self,
        *,
        auth_subject: str,
        workspace_name: str,
    ) -> ProvisionedPersonalWorkspace: ...


class ProvisioningService:
    def __init__(self, store: ProvisioningStore) -> None:
        self._store = store

    async def provision_personal_workspace(
        self,
        auth_subject: str,
    ) -> ProvisionedPersonalWorkspace:
        if not isinstance(auth_subject, str) or not auth_subject.strip():
            raise DomainValidationError("auth subject must not be blank")

        return await self._store.provision_personal_workspace(
            auth_subject=auth_subject,
            workspace_name=PERSONAL_WORKSPACE_NAME,
        )
