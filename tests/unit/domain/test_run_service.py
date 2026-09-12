from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.errors import DomainNotFoundError, DomainValidationError
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchRequestV1
from app.domain.runs import (
    CURRENT_GRAPH_VERSION,
    DEFAULT_RUN_LIMITS,
    RunAccepted,
    RunCancellation,
    RunCreateIdentity,
    RunMode,
    RunRecord,
    RunService,
    RunStatus,
    RunUsageBucket,
    RunUsageSummary,
    create_request_digest_v1,
)
from app.domain.tenancy import TenantContext, TenantService


def _empty_bucket() -> RunUsageBucket:
    return RunUsageBucket(
        attempt_count=0,
        succeeded_count=0,
        input_tokens=0,
        output_tokens=0,
        reasoning_output_tokens=0,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        estimated_cost=None,
        currency=None,
        cost_available=False,
    )


class _RunStore:
    def __init__(self) -> None:
        self.created: list[
            tuple[TenantContext, RunMode, UUID | None, ResearchRequestV1, dict[str, int], str]
        ] = []
        self.cancelled: list[tuple[TenantContext, UUID, bool]] = []
        self.identities: list[RunCreateIdentity | None] = []
        self.run_id = uuid4()

    async def create_run(
        self,
        *,
        tenant: TenantContext,
        mode: RunMode,
        resume_document_id: UUID | None,
        request: ResearchRequestV1,
        limits: dict[str, int],
        graph_version: str,
        request_identity: RunCreateIdentity | None = None,
    ) -> RunAccepted:
        self.identities.append(request_identity)
        self.created.append((tenant, mode, resume_document_id, request, limits, graph_version))
        return RunAccepted(run_id=self.run_id, status=RunStatus.QUEUED)

    async def get_run(self, *, tenant: TenantContext, run_id: UUID) -> RunRecord:
        now = datetime.now(UTC)
        return RunRecord(
            run_id=run_id,
            mode=RunMode.RESEARCH,
            status=RunStatus.QUEUED,
            graph_version=CURRENT_GRAPH_VERSION,
            result=None,
            error_category=None,
            cancel_requested_at=None,
            started_at=None,
            finished_at=None,
            created_at=now,
            updated_at=now,
            usage=RunUsageSummary(chat=_empty_bucket(), embedding=_empty_bucket()),
        )

    async def cancel_run(
        self,
        *,
        tenant: TenantContext,
        run_id: UUID,
        allow_other_creator: bool,
    ) -> RunCancellation:
        self.cancelled.append((tenant, run_id, allow_other_creator))
        return RunCancellation(
            run_id=run_id,
            status=RunStatus.CANCELLED,
            cancel_requested_at=None,
        )


def _tenant(role: WorkspaceRole = WorkspaceRole.MEMBER) -> TenantContext:
    return TenantContext(
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        role=role,
    )


async def test_create_uses_fixed_research_request_limits_and_graph_version() -> None:
    store = _RunStore()
    service = RunService(store)
    tenant = _tenant()

    accepted = await service.create_research_run(tenant=tenant, query=" Senior Python\tEngineer \n")

    assert accepted == RunAccepted(run_id=store.run_id, status=RunStatus.QUEUED)
    assert store.identities == [None]
    assert store.created == [
        (
            tenant,
            RunMode.RESEARCH,
            None,
            ResearchRequestV1(query="Senior Python Engineer", include_application_draft=False),
            DEFAULT_RUN_LIMITS,
            CURRENT_GRAPH_VERSION,
        )
    ]
    store.created[0][4]["max_model_calls"] = 999
    assert DEFAULT_RUN_LIMITS["max_model_calls"] == 12


@pytest.mark.parametrize("query", ["", "   ", " \t\n\u3000", "x" * 2_001, "ﬃ" * 667])
async def test_create_rejects_invalid_query_before_store(query: str) -> None:
    store = _RunStore()
    with pytest.raises(DomainValidationError, match="research query is invalid"):
        await RunService(store).create_research_run(tenant=_tenant(), query=query)
    assert store.created == []


@pytest.mark.parametrize(
    ("role", "allow_other_creator"),
    [
        (WorkspaceRole.MEMBER, False),
        (WorkspaceRole.REVIEWER, False),
        (WorkspaceRole.ADMIN, True),
    ],
)
async def test_cancel_derives_scope_only_from_trusted_role(
    role: WorkspaceRole,
    allow_other_creator: bool,
) -> None:
    store = _RunStore()
    tenant = _tenant(role)
    run_id = uuid4()

    await RunService(store).cancel_run(tenant=tenant, run_id=run_id)

    assert store.cancelled == [(tenant, run_id, allow_other_creator)]


class _TenantResolver:
    def __init__(self, tenant: TenantContext | None) -> None:
        self.tenant = tenant
        self.calls: list[tuple[UUID, UUID]] = []

    async def resolve_tenant(
        self,
        *,
        workspace_id: UUID,
        actor_user_id: UUID,
    ) -> TenantContext | None:
        self.calls.append((workspace_id, actor_user_id))
        return self.tenant


async def test_tenant_service_returns_only_resolver_supplied_context() -> None:
    tenant = _tenant(WorkspaceRole.REVIEWER)
    resolver = _TenantResolver(tenant)
    service = TenantService(resolver)

    resolved = await service.resolve_tenant(
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
    )

    assert resolved is tenant
    assert resolver.calls == [(tenant.workspace_id, tenant.actor_user_id)]


async def test_tenant_service_hides_missing_or_revoked_membership() -> None:
    with pytest.raises(DomainNotFoundError):
        await TenantService(_TenantResolver(None)).resolve_tenant(
            workspace_id=uuid4(),
            actor_user_id=uuid4(),
        )


def test_usage_bucket_keeps_exact_decimal_type() -> None:
    bucket = RunUsageBucket(
        attempt_count=1,
        succeeded_count=1,
        input_tokens=1,
        output_tokens=2,
        reasoning_output_tokens=1,
        cached_input_tokens=0,
        cache_write_input_tokens=0,
        estimated_cost=Decimal("0.000001000000"),
        currency="CNY",
        cost_available=True,
    )
    assert bucket.estimated_cost == Decimal("0.000001000000")


@pytest.mark.parametrize("mode", [RunMode.RESEARCH, RunMode.APPLICATION])
async def test_create_constructs_identity_from_normalized_content(mode: RunMode) -> None:
    store = _RunStore()
    key, resume = uuid4(), uuid4()
    await RunService(store).create_run(
        tenant=_tenant(),
        mode=mode,
        query="  \uff21\uff29\tEngineer  ",
        resume_document_id=resume,
        client_request_id=key,
    )
    assert store.identities == [
        RunCreateIdentity(
            client_request_id=key,
            create_request_digest=create_request_digest_v1(
                mode=mode, query="AI Engineer", resume_document_id=resume
            ),
        )
    ]
    assert store.created[0][3].query == "AI Engineer"


@pytest.mark.parametrize("key", ["CANARY", 1, UUID(int=0)])
async def test_invalid_request_identity_never_reaches_store(key: object) -> None:
    store = _RunStore()
    with pytest.raises(DomainValidationError, match="run creation identity is invalid"):
        await RunService(store).create_run(
            tenant=_tenant(), mode=RunMode.RESEARCH, query="Research", client_request_id=key
        )
    assert store.created == []
    assert store.identities == []
