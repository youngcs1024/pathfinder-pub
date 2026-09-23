"""Read-only retrieval within a server-built collection of material snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from pydantic import Field

from app.domain.project_facts import MaterialRetrievalScope
from app.domain.tenancy import TenantContext
from app.domain.tool_effects import ToolEffect
from app.domain.tool_invocations import ToolInvocationRecorderPort
from app.retrieval.documents import DocumentRetrievalService
from app.retrieval.material_scope import allowed_document_ids
from app.tools.contracts import (
    CredentialSource,
    GraphToolPolicy,
    ToolExecutionContext,
    ToolInputModel,
    ToolOutputModel,
    ToolSpec,
)
from app.tools.registry import ToolRegistry

MATERIAL_POLICY = "material_fact_read_only"


class RetrieveMaterialInputV1(ToolInputModel):
    query: str = Field(min_length=1, max_length=2000)


class MaterialHitV1(ToolOutputModel):
    file_ref: UUID
    path: str
    source_revision: str
    untrusted_text: str


class RetrieveMaterialOutputV1(ToolOutputModel):
    results: tuple[MaterialHitV1, ...] = Field(default=(), max_length=5)


class ReadExcerptInputV1(ToolInputModel):
    file_ref: str = Field(pattern=r"^[0-9a-f-]{36}$")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class ReadExcerptOutputV1(ToolOutputModel):
    file_ref: UUID
    path: str
    start_line: int
    end_line: int
    untrusted_text: str


@dataclass(frozen=True, slots=True, repr=False)
class RetrieveMaterialHandler:
    scope: MaterialRetrievalScope
    service: DocumentRetrievalService
    tenant: TenantContext

    async def __call__(
        self, tool_input: ToolInputModel, _context: ToolExecutionContext
    ) -> RetrieveMaterialOutputV1:
        if not isinstance(tool_input, RetrieveMaterialInputV1):
            raise TypeError("material retrieval input is invalid")
        hits = await self.service.retrieve(
            tenant=self.tenant,
            query=tool_input.query,
            allowed_document_ids=allowed_document_ids(self.scope),
        )
        results: list[MaterialHitV1] = []
        for hit in hits:
            for file in self.scope.files:
                if file.document_id != hit.document_id:
                    continue
                results.append(
                    MaterialHitV1(
                        file_ref=file.id,
                        path=file.path,
                        source_revision=file.source_revision,
                        untrusted_text=hit.text,
                    )
                )
                if len(results) == 5:
                    return RetrieveMaterialOutputV1(results=tuple(results))
        return RetrieveMaterialOutputV1(results=tuple(results))


@dataclass(frozen=True, slots=True, repr=False)
class ReadExcerptHandler:
    scope: MaterialRetrievalScope

    async def __call__(
        self, tool_input: ToolInputModel, _context: ToolExecutionContext
    ) -> ReadExcerptOutputV1:
        if not isinstance(tool_input, ReadExcerptInputV1):
            raise TypeError("material excerpt input is invalid")
        file = next(
            (item for item in self.scope.files if str(item.id) == tool_input.file_ref), None
        )
        if (
            file is None
            or tool_input.end_line < tool_input.start_line
            or tool_input.end_line - tool_input.start_line >= 80
        ):
            raise ValueError("material excerpt is outside allowed scope")
        lines = file.content.decode("utf-8", errors="strict").splitlines()
        if tool_input.end_line > len(lines):
            raise ValueError("material excerpt lines are unavailable")
        value = "\n".join(lines[tool_input.start_line - 1 : tool_input.end_line])
        if len(value.encode("utf-8")) > 4000:
            raise ValueError("material excerpt exceeds the byte limit")
        return ReadExcerptOutputV1(
            file_ref=file.id,
            path=file.path,
            start_line=tool_input.start_line,
            end_line=tool_input.end_line,
            untrusted_text=value,
        )


def create_material_registry(
    *,
    scope: MaterialRetrievalScope,
    service: DocumentRetrievalService,
    tenant: TenantContext,
    recorder: ToolInvocationRecorderPort | None = None,
) -> ToolRegistry:
    return ToolRegistry(
        specs=(
            ToolSpec(
                name="retrieve_project_material",
                description="Find bounded text in authorized project snapshots.",
                input_model=RetrieveMaterialInputV1,
                output_model=RetrieveMaterialOutputV1,
                effect=ToolEffect.READ_ONLY,
                credential_source=CredentialSource.NONE,
                timeout_seconds=30,
                max_attempts=1,
                per_run_call_limit=8,
                max_output_bytes=8_000,
                handler=RetrieveMaterialHandler(scope, service, tenant),
            ),
            ToolSpec(
                name="read_material_excerpt",
                description="Read up to 80 lines from an authorized fixed snapshot.",
                input_model=ReadExcerptInputV1,
                output_model=ReadExcerptOutputV1,
                effect=ToolEffect.READ_ONLY,
                credential_source=CredentialSource.NONE,
                timeout_seconds=10,
                max_attempts=1,
                per_run_call_limit=8,
                max_output_bytes=5_000,
                handler=ReadExcerptHandler(scope),
            ),
        ),
        policies=(
            GraphToolPolicy(
                name=MATERIAL_POLICY,
                allowed_tool_names=frozenset(
                    {"retrieve_project_material", "read_material_excerpt"}
                ),
                allowed_effects=frozenset({ToolEffect.READ_ONLY}),
            ),
        ),
        recorder=recorder,
    )
