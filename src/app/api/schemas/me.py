from uuid import UUID

from pydantic import BaseModel

from app.domain.provisioning import WorkspaceKind, WorkspaceRole


class ActorWorkspaceResponse(BaseModel):
    workspace_id: UUID
    kind: WorkspaceKind
    name: str
    role: WorkspaceRole


class MeResponse(BaseModel):
    user_id: UUID
    subject_summary: str
    workspaces: list[ActorWorkspaceResponse]
