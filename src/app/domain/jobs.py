from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol
from uuid import UUID

from app.domain.action_execution import ActionExecutionIdentity
from app.domain.run_payloads import RunOutput
from app.domain.tenancy import TenantContext


class JobStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    DONE = "done"
    DEAD = "dead"


_JOB_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.LEASED, JobStatus.DONE}),
    JobStatus.LEASED: frozenset({JobStatus.DONE, JobStatus.QUEUED, JobStatus.DEAD}),
    JobStatus.DONE: frozenset({JobStatus.QUEUED}),
    JobStatus.DEAD: frozenset(),
}


def is_valid_job_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in _JOB_TRANSITIONS[current]


@dataclass(frozen=True, slots=True, repr=False)
class ClaimedJob:
    job_id: UUID
    workspace_id: UUID
    run_id: UUID
    originating_actor_user_id: UUID
    graph_version: str
    attempt: int
    max_attempts: int
    owner_token: UUID
    lease_expires_at: datetime
    resume_approval_request_id: UUID | None = None

    def __post_init__(self) -> None:
        uuid_fields = (
            self.job_id,
            self.workspace_id,
            self.run_id,
            self.originating_actor_user_id,
            self.owner_token,
        )
        if not all(isinstance(value, UUID) for value in uuid_fields):
            raise TypeError("claimed job identifiers must be UUIDs")
        if (
            not isinstance(self.graph_version, str)
            or not self.graph_version.strip()
            or len(self.graph_version) > 100
        ):
            raise TypeError("claimed job graph_version must be non-blank")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.attempt < 1
            or self.max_attempts < self.attempt
        ):
            raise ValueError("claimed job attempt values are invalid")
        if not isinstance(self.lease_expires_at, datetime) or self.lease_expires_at.tzinfo is None:
            raise TypeError("claimed job lease expiry must be timezone-aware")
        if self.resume_approval_request_id is not None and not isinstance(
            self.resume_approval_request_id, UUID
        ):
            raise TypeError("claimed job resume approval identity must be a UUID")


@dataclass(frozen=True, slots=True)
class ReclaimSummary:
    scanned: int
    requeued: int
    finished: int
    dead: int

    def __post_init__(self) -> None:
        values = (self.scanned, self.requeued, self.finished, self.dead)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("reclaim summary counts must be non-negative integers")
        if self.requeued + self.finished + self.dead != self.scanned:
            raise ValueError("reclaim summary counts must account for every scanned job")


@dataclass(frozen=True, slots=True)
class PrepareClaimResult:
    disposition: Literal["execute", "recover_action", "finished", "lease_lost"]
    tenant: TenantContext | None = None
    action_identity: ActionExecutionIdentity | None = None

    def __post_init__(self) -> None:
        if self.disposition not in {"execute", "recover_action", "finished", "lease_lost"}:
            raise ValueError("prepared claim disposition is invalid")
        if (self.disposition == "execute") != (self.tenant is not None):
            raise ValueError("prepared claim tenant must exist only for execution")
        if (self.disposition == "recover_action") != (self.action_identity is not None):
            raise ValueError("recovery action identity must exist only for recovery")
        if self.tenant is not None and not isinstance(self.tenant, TenantContext):
            raise TypeError("prepared claim tenant must be a TenantContext")


class RetryDelayPolicy(Protocol):
    def __call__(self, attempt: int) -> timedelta: ...


class JobClaimer(Protocol):
    async def claim_due_job(
        self,
        *,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> ClaimedJob | None: ...


class StaleLeaseReclaimer(Protocol):
    async def reclaim_stale_leases(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> ReclaimSummary: ...


class WorkerJobStore(JobClaimer, StaleLeaseReclaimer, Protocol):
    async def prepare_claimed_job(
        self,
        *,
        job: ClaimedJob,
        resolved_tenant: TenantContext | None,
        now: datetime,
    ) -> PrepareClaimResult: ...

    async def heartbeat(
        self,
        *,
        job: ClaimedJob,
        now: datetime,
        lease_duration: timedelta,
    ) -> bool: ...

    async def complete(
        self,
        *,
        job: ClaimedJob,
        result: RunOutput,
        now: datetime,
    ) -> bool: ...

    async def wait_for_approval(
        self,
        *,
        job: ClaimedJob,
        approval_request_id: UUID,
        now: datetime,
    ) -> bool: ...

    async def requeue(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool: ...

    async def fail(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool: ...

    async def cancel(
        self,
        *,
        job: ClaimedJob,
        reason: str,
        now: datetime,
    ) -> bool: ...

    async def release(
        self,
        *,
        job: ClaimedJob,
        error_category: str,
        now: datetime,
    ) -> bool: ...
