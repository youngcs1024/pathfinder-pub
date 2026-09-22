import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from app.domain.errors import DomainValidationError
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1, normalize_research_query
from app.domain.run_payloads import (
    EXECUTION_CONTRACTS,
    LEGACY_GRAPH_VERSION,
    LEGACY_RUN_MODES,
    READ_CONTRACTS,
    RunInput,
    RunMode,
    RunOutput,
)
from app.domain.tenancy import TenantContext


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class RunCreateIdentity:
    client_request_id: UUID
    create_request_digest: str
    create_request_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.client_request_id, UUID)
            or self.client_request_id.version != 4
            or type(self.create_request_version) is not int
            or self.create_request_version != 1
            or not isinstance(self.create_request_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.create_request_digest) is None
        ):
            raise DomainValidationError("run creation identity is invalid")


def create_request_digest_v1(
    *, mode: RunMode, query: str, resume_document_id: UUID | None = None
) -> str:
    """Hash explicit request content; preserve v1 normalization for future replay."""
    if (
        (not isinstance(mode, RunMode) or mode not in LEGACY_RUN_MODES)
        or not isinstance(query, str)
        or (resume_document_id is not None and not isinstance(resume_document_id, UUID))
        or (mode is RunMode.APPLICATION and resume_document_id is None)
    ):
        raise DomainValidationError("run creation input is invalid")
    try:
        request = ResearchRequestV1(
            query=normalize_research_query(query),
            include_application_draft=mode is RunMode.APPLICATION,
        )
        canonical = json.dumps(
            {
                "version": 1,
                "mode": mode.value,
                "query": request.query,
                "resume_document_id": (
                    str(resume_document_id) if resume_document_id is not None else None
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValidationError, ValueError):
        raise DomainValidationError("run creation input is invalid") from None
    return hashlib.sha256(canonical).hexdigest()


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Historical compatibility name for the preserved research graph and its fixtures.
# Production execution is determined exclusively by EXECUTION_CONTRACTS.
CURRENT_GRAPH_VERSION = LEGACY_GRAPH_VERSION
EXECUTABLE_GRAPH_VERSIONS = frozenset(c.graph_version for c in EXECUTION_CONTRACTS)
READABLE_GRAPH_VERSIONS = frozenset(c.graph_version for c in READ_CONTRACTS)
SUPPORTED_GRAPH_VERSIONS = EXECUTABLE_GRAPH_VERSIONS

_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.WAITING_APPROVAL,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.WAITING_APPROVAL: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


def is_valid_run_transition(current: RunStatus, target: RunStatus) -> bool:
    return target in _RUN_TRANSITIONS[current]


DEFAULT_RUN_LIMITS: Mapping[str, int] = MappingProxyType(
    {
        "schema_version": 1,
        "max_model_calls": 12,
        "max_tool_calls": 8,
        "max_tool_results": 8,
        "max_iterations": 24,
    }
)


@dataclass(frozen=True, slots=True)
class RunAccepted:
    run_id: UUID
    status: RunStatus
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class RunUsageBucket:
    attempt_count: int
    succeeded_count: int
    input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int
    estimated_cost: Decimal | None
    currency: str | None
    cost_available: bool


@dataclass(frozen=True, slots=True)
class RunUsageSummary:
    chat: RunUsageBucket
    embedding: RunUsageBucket


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: UUID
    mode: RunMode
    status: RunStatus
    graph_version: str
    result: RunOutput | None
    error_category: str | None
    cancel_requested_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    usage: RunUsageSummary
    resume_document_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class RunCancellation:
    run_id: UUID
    status: RunStatus
    cancel_requested_at: datetime | None


class RunStore(Protocol):
    async def create_run(
        self,
        *,
        tenant: TenantContext,
        mode: RunMode,
        resume_document_id: UUID | None,
        request: RunInput,
        limits: dict[str, int],
        graph_version: str,
        request_identity: RunCreateIdentity | None = None,
    ) -> RunAccepted: ...

    async def get_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
    ) -> RunRecord: ...

    async def cancel_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        allow_other_creator: bool,
    ) -> RunCancellation: ...


class RunService:
    def __init__(self, store: RunStore) -> None:
        self._store = store

    async def create_run(
        self,
        *,
        tenant: TenantContext,
        mode: RunMode,
        query: str,
        resume_document_id: UUID | None = None,
        client_request_id: UUID | None = None,
    ) -> RunAccepted:
        if not isinstance(tenant, TenantContext):
            raise DomainValidationError("tenant context is invalid")
        if (not isinstance(mode, RunMode) or mode not in LEGACY_RUN_MODES) or (
            resume_document_id is not None and not isinstance(resume_document_id, UUID)
        ):
            raise DomainValidationError("run creation input is invalid")
        if mode is RunMode.APPLICATION and resume_document_id is None:
            raise DomainValidationError("application requires a resume document")
        try:
            request = ResearchRequestV1(
                query=normalize_research_query(query),
                include_application_draft=mode is RunMode.APPLICATION,
            )
        except (TypeError, ValidationError, ValueError):
            raise DomainValidationError("research query is invalid") from None
        request_identity = (
            RunCreateIdentity(
                client_request_id=client_request_id,
                create_request_digest=create_request_digest_v1(
                    mode=mode, query=query, resume_document_id=resume_document_id
                ),
            )
            if client_request_id is not None
            else None
        )
        return await self._store.create_run(
            tenant=tenant,
            mode=mode,
            resume_document_id=resume_document_id,
            request=request,
            limits=dict(DEFAULT_RUN_LIMITS),
            graph_version=CURRENT_GRAPH_VERSION,
            request_identity=request_identity,
        )

    async def create_research_run(self, *, tenant: TenantContext, query: str) -> RunAccepted:
        return await self.create_run(tenant=tenant, mode=RunMode.RESEARCH, query=query)

    async def get_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
    ) -> RunRecord:
        if not isinstance(tenant, TenantContext) or not isinstance(run_id, UUID):
            raise DomainValidationError("run lookup is invalid")
        return await self._store.get_run(tenant=tenant, run_id=run_id)

    async def cancel_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
    ) -> RunCancellation:
        if not isinstance(tenant, TenantContext) or not isinstance(run_id, UUID):
            raise DomainValidationError("run cancellation is invalid")
        return await self._store.cancel_run(
            tenant=tenant,
            run_id=run_id,
            allow_other_creator=tenant.role is WorkspaceRole.ADMIN,
        )
