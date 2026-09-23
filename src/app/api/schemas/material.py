from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MaterialApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AliasResponse(MaterialApiModel):
    name: str
    kind: Literal["git", "file"]


class ProjectInput(MaterialApiModel):
    name: str = Field(min_length=1, max_length=120)


class ProjectResponse(MaterialApiModel):
    id: UUID
    name: str
    created_by_user_id: UUID


class SourceInput(MaterialApiModel):
    alias: str = Field(min_length=1, max_length=64)


class SourceResponse(MaterialApiModel):
    id: UUID
    project_id: UUID
    alias: str
    kind: Literal["git", "file"]


class ImportInput(MaterialApiModel):
    source_ids: tuple[UUID, ...] = Field(min_length=1, max_length=20)


class ImportReceiptResponse(MaterialApiModel):
    command_id: UUID
    run_id: UUID
    import_id: UUID
    status: Literal["queued"]
    replayed: bool


class ImportListItemResponse(MaterialApiModel):
    id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: datetime


class SnapshotProgressResponse(MaterialApiModel):
    id: UUID
    source_id: UUID
    source_revision: str
    manifest_digest: str
    file_count: int
    indexed_count: int
    unindexed_paths: list[str]
    inventory: dict[str, object]


class ImportProgressResponse(MaterialApiModel):
    id: UUID
    project_id: UUID
    run_id: UUID
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    error_category: str | None
    cache_digest: str | None
    snapshots: list[SnapshotProgressResponse]
