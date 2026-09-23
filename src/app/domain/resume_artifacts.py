"""Read-only delivery contract for immutable resume TeX artifacts."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.tenancy import TenantContext

COMPILER = "XeLaTeX"
FONT = "Microsoft YaHei"
PACKAGES = (
    "fullpage",
    "titlesec",
    "xcolor",
    "tcolorbox",
    "enumitem",
    "fancyhdr",
    "ctex",
    "fontspec",
)


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class CompileInstructionsV1(ArtifactModel):
    engine: str = COMPILER
    font: str = FONT
    packages: tuple[str, ...] = PACKAGES
    instructions: tuple[str, ...] = (
        "Save the downloaded UTF-8 file as a .tex source.",
        "Compile it with XeLaTeX in your own editor with Microsoft YaHei installed.",
        "On Windows, compile from a local directory; a WSL UNC workdir can prevent PDF output.",
        "Inspect the output, including the actual page count and any overflow.",
    )


class TexArtifactInfoV1(ArtifactModel):
    artifact_id: UUID
    profile_version_id: UUID
    template_commit: str
    template_source_sha256: str
    preamble_sha256: str
    renderer_version: str
    tex_sha256: str
    byte_count: int = Field(ge=1, le=512 * 1024)
    compile: CompileInstructionsV1 = CompileInstructionsV1()


class ResumeArtifactPort(Protocol):
    async def get_info(self, tenant: TenantContext, artifact_id: UUID) -> TexArtifactInfoV1: ...
    async def get_bytes(self, tenant: TenantContext, artifact_id: UUID) -> tuple[bytes, str]: ...


class ResumeArtifactService:
    def __init__(self, port: ResumeArtifactPort) -> None:
        self.port = port

    async def get_info(self, tenant: TenantContext, artifact_id: UUID) -> TexArtifactInfoV1:
        return await self.port.get_info(tenant, artifact_id)

    async def get_bytes(self, tenant: TenantContext, artifact_id: UUID) -> tuple[bytes, str]:
        return await self.port.get_bytes(tenant, artifact_id)
