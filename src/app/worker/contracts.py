from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.domain.research import ResearchOutputV1, ResearchOutputV2
from app.domain.run_payloads import ResumeRunOutputV1, RunOutput
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext

_EXECUTOR_RESULT_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.WAITING_APPROVAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class RunExecutionResult:
    status: RunStatus
    result: RunOutput | None = None
    error_category: str | None = None
    retryable: bool = False
    approval_request_id: UUID | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, RunStatus):
            raise TypeError("executor result status must be a RunStatus")
        if self.status not in _EXECUTOR_RESULT_STATUSES:
            raise ValueError("executor result uses an unsupported run status")
        if not isinstance(self.retryable, bool):
            raise TypeError("executor result retryable must be a boolean")
        if self.status is RunStatus.COMPLETED:
            if (
                not isinstance(self.result, ResearchOutputV1 | ResearchOutputV2 | ResumeRunOutputV1)
                or self.error_category is not None
                or self.retryable
                or self.approval_request_id is not None
            ):
                raise ValueError("completed executor result fields are invalid")
            if isinstance(self.result, ResumeRunOutputV1):
                try:
                    type(self.result).model_validate_json(
                        self.result.model_dump_json(warnings="error"), strict=True
                    )
                except (TypeError, ValueError, AttributeError):
                    raise ValueError("completed executor result schema is invalid") from None
            return
        if self.status is RunStatus.FAILED:
            if self.result is not None or self.approval_request_id is not None:
                raise ValueError("failed executor result cannot contain a result")
            if (
                not isinstance(self.error_category, str)
                or not self.error_category
                or len(self.error_category) > 100
                or not self.error_category.replace("_", "a").isalnum()
                or not self.error_category[0].islower()
            ):
                raise ValueError("failed executor result error category is invalid")
            return
        if self.status is RunStatus.WAITING_APPROVAL:
            if (
                not isinstance(self.approval_request_id, UUID)
                or self.result is not None
                or self.error_category is not None
                or self.retryable
            ):
                raise ValueError("waiting executor result fields are invalid")
            return
        if (
            self.result is not None
            or self.error_category is not None
            or self.retryable
            or self.approval_request_id is not None
        ):
            raise ValueError("non-terminal-data executor result fields are invalid")


class RunExecutor(Protocol):
    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult: ...
