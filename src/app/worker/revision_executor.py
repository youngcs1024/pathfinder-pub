"""Execute a single fixed-input resume revision on the existing worker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from app.agents.resume_revision import ResumeRevisionGraph
from app.domain.errors import (
    DomainConflictError,
    DomainNotFoundError,
    DomainUnavailableError,
    DomainValidationError,
)
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInvalidError,
    RunExecutionReader,
)
from app.domain.run_payloads import ResumeRevisionCandidateOutputV1, ResumeRevisionRunInputV1
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMAccountingError, LLMProviderError
from app.llm.ports import ChatModelPort
from app.worker.contracts import RunExecutionResult


class RevisionRunExecutor:
    def __init__(
        self,
        *,
        reader: RunExecutionReader,
        revisions,
        model_factory: Callable[[TenantContext, UUID], ChatModelPort],
    ) -> None:
        self.reader = reader
        self.revisions = revisions
        self.model_factory = model_factory

    async def execute(
        self, run_id: UUID, tenant: TenantContext, graph_version: str
    ) -> RunExecutionResult:
        if graph_version != "pathfinder-resume-v4":
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="unknown_graph_version"
            )
        try:
            execution = await self.reader.read_for_execution(
                run_id=run_id,
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                graph_version=graph_version,
            )
            if not isinstance(execution.request, ResumeRevisionRunInputV1):
                return RunExecutionResult(
                    status=RunStatus.FAILED, error_category="invalid_run_input"
                )
            payload = execution.request.payload
            inputs = await self.revisions.revision_inputs(tenant, payload.feedback_id)
            if (
                inputs.run_id != run_id
                or inputs.session_id != payload.session_id
                or inputs.base_version_id != payload.base_version_id
            ):
                return RunExecutionResult(
                    status=RunStatus.FAILED, error_category="invalid_run_input"
                )

            async def allowed() -> bool:
                await self.reader.assert_execution_allowed(
                    run_id=run_id,
                    workspace_id=tenant.workspace_id,
                    actor_user_id=tenant.actor_user_id,
                    graph_version=graph_version,
                )
                return await self.revisions.spend_allowed(tenant, payload.feedback_id)

            async def repair() -> bool:
                await self.reader.assert_execution_allowed(
                    run_id=run_id,
                    workspace_id=tenant.workspace_id,
                    actor_user_id=tenant.actor_user_id,
                    graph_version=graph_version,
                )
                return await self.revisions.reserve_repair(tenant, payload.feedback_id)

            candidate = await ResumeRevisionGraph(
                model=self.model_factory(tenant, run_id),
                spend_allowed=allowed,
                reserve_repair=repair,
            ).generate(inputs)
            await self.reader.assert_execution_allowed(
                run_id=run_id,
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                graph_version=graph_version,
            )
            return RunExecutionResult(
                status=RunStatus.COMPLETED,
                result=ResumeRevisionCandidateOutputV1(payload=candidate),
            )
        except asyncio.CancelledError:
            raise
        except RunExecutionCancelledError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except DomainNotFoundError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except RunExecutionInvalidError as error:
            return RunExecutionResult(status=RunStatus.FAILED, error_category=error.category)
        except (DomainConflictError, DomainValidationError):
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="invalid_revision_input"
            )
        except DomainUnavailableError:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="database_unavailable",
                retryable=True,
            )
        except LLMAccountingError as error:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="llm_accounting_failed",
                retryable=error.retryable,
            )
        except LLMProviderError:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="provider_unavailable",
                retryable=True,
            )
