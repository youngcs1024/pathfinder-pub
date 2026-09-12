from datetime import UTC, datetime, timedelta
from random import Random
from uuid import uuid4

import pytest

from app.domain.jobs import ClaimedJob, PrepareClaimResult, ReclaimSummary
from app.domain.provisioning import WorkspaceRole
from app.domain.research import ResearchLimitationV1, ResearchOutputV1
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.worker.backoff import ExponentialBackoff
from app.worker.contracts import RunExecutionResult
from app.worker.fake_executor import CONTRACT_FAKE_LIMITATION, ContractFakeRunExecutor
from app.worker.settings import WorkerRuntimeSettings


def _insufficient_output() -> ResearchOutputV1:
    return ResearchOutputV1(
        evidence_sufficient=False,
        limitations=(
            ResearchLimitationV1(
                code="insufficient_evidence",
                detail="No evidence was gathered.",
            ),
        ),
    )


def test_run_execution_result_uses_only_existing_run_statuses() -> None:
    completed = RunExecutionResult(status=RunStatus.COMPLETED, result=_insufficient_output())
    failed = RunExecutionResult(
        status=RunStatus.FAILED,
        error_category="provider_timeout",
        retryable=True,
    )
    request_id = uuid4()
    waiting = RunExecutionResult(
        status=RunStatus.WAITING_APPROVAL,
        approval_request_id=request_id,
    )
    cancelled = RunExecutionResult(status=RunStatus.CANCELLED)

    assert completed.status is RunStatus.COMPLETED
    assert failed.retryable is True
    assert waiting.status is RunStatus.WAITING_APPROVAL
    assert waiting.approval_request_id == request_id
    assert cancelled.status is RunStatus.CANCELLED

    with pytest.raises(ValueError, match="unsupported run status"):
        RunExecutionResult(status=RunStatus.RUNNING)
    with pytest.raises(TypeError, match="must be a RunStatus"):
        RunExecutionResult(status="completed")
    with pytest.raises(ValueError, match="completed executor result"):
        RunExecutionResult(status=RunStatus.COMPLETED)
    with pytest.raises(ValueError, match="error category"):
        RunExecutionResult(status=RunStatus.FAILED, error_category="BAD-CATEGORY")
    with pytest.raises(ValueError, match="waiting executor result"):
        RunExecutionResult(status=RunStatus.WAITING_APPROVAL)


async def test_contract_fake_returns_explicit_schema_valid_limitation() -> None:
    tenant = TenantContext(
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        role=WorkspaceRole.ADMIN,
    )

    result = await ContractFakeRunExecutor().execute(
        uuid4(),
        tenant,
        "pathfinder-research-v1",
    )

    assert result.status is RunStatus.COMPLETED
    assert result.result is not None
    assert result.result.evidence_sufficient is False
    assert result.result.limitations[0].detail == CONTRACT_FAKE_LIMITATION


def test_claimed_job_and_control_results_are_strict() -> None:
    now = datetime.now(UTC)
    claimed = ClaimedJob(
        job_id=uuid4(),
        workspace_id=uuid4(),
        run_id=uuid4(),
        originating_actor_user_id=uuid4(),
        graph_version="pathfinder-research-v1",
        attempt=1,
        max_attempts=3,
        owner_token=uuid4(),
        lease_expires_at=now + timedelta(seconds=30),
    )
    tenant = TenantContext(
        workspace_id=claimed.workspace_id,
        actor_user_id=claimed.originating_actor_user_id,
        role=WorkspaceRole.MEMBER,
    )

    assert PrepareClaimResult(disposition="execute", tenant=tenant).tenant is tenant
    assert ReclaimSummary(scanned=3, requeued=1, finished=1, dead=1).scanned == 3
    with pytest.raises(ValueError, match="disposition"):
        PrepareClaimResult(disposition="unknown")
    with pytest.raises(ValueError, match="tenant must exist"):
        PrepareClaimResult(disposition="execute")
    with pytest.raises(ValueError, match="account for every"):
        ReclaimSummary(scanned=2, requeued=1, finished=0, dead=0)


def test_worker_settings_and_backoff_are_bounded_and_deterministic() -> None:
    settings = WorkerRuntimeSettings(retry_jitter_ratio=0.25)
    assert settings.action_recovery_max_attempts == 3
    first = ExponentialBackoff(settings, Random(7))
    second = ExponentialBackoff(settings, Random(7))

    delays = [first(attempt).total_seconds() for attempt in (1, 2, 3, 10)]

    assert delays == [second(attempt).total_seconds() for attempt in (1, 2, 3, 10)]
    assert 1 <= delays[0] <= 1.25
    assert 2 <= delays[1] <= 2.5
    assert 4 <= delays[2] <= 5
    assert 30 <= delays[3] <= 37.5
    with pytest.raises(ValueError, match="two heartbeat"):
        WorkerRuntimeSettings(lease_seconds=20, heartbeat_seconds=10)
    with pytest.raises(ValueError, match="action recovery"):
        WorkerRuntimeSettings(action_recovery_max_attempts=0)
    with pytest.raises(ValueError, match="retry attempt"):
        first(0)
