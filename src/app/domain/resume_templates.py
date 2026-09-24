"""Narrow rendering and manual compilation contracts, without runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.domain.errors import DomainValidationError
from app.domain.resume_artifacts import CompileInstructionsV1
from app.domain.resume_profile import ResumeContentV1, ResumePreferencesV1


@dataclass(frozen=True)
class TemplateManifest:
    commit: str
    source_sha256: str
    preamble_sha256: str
    renderer_version: str
    compile: CompileInstructionsV1 = field(default_factory=CompileInstructionsV1)


@dataclass(frozen=True)
class TemplateRenderResult:
    tex_bytes: bytes
    tex_sha256: str
    content_sha256: str
    config_sha256: str
    manifest: TemplateManifest


class TemplateRenderer(Protocol):
    def render(
        self,
        *,
        source_bytes: bytes,
        manifest: TemplateManifest,
        profile_content: ResumeContentV1,
        content: ResumeContentV1,
        preferences: ResumePreferencesV1,
    ) -> TemplateRenderResult: ...


@dataclass(frozen=True)
class CompilePreparation:
    tex_sha256: str
    manifest: TemplateManifest
    status: Literal["NOT_RUN"] = "NOT_RUN"


@dataclass(frozen=True)
class CompileResult:
    tex_sha256: str
    status: Literal["success", "failed"]
    simulated: bool

    def require_source(self, preparation: CompilePreparation) -> None:
        if self.tex_sha256 != preparation.tex_sha256:
            raise DomainValidationError("compile source identity mismatch")
