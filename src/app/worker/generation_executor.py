"""Run a fixed-input draft graph inside the existing single worker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

from app.agents.resume_generation import GenerationError, ResumeGenerationGraph
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.domain.errors import DomainNotFoundError, DomainUnavailableError, DomainValidationError
from app.domain.run_execution import (
    RunExecutionCancelledError,
    RunExecutionInvalidError,
    RunExecutionReader,
)
from app.domain.run_payloads import (
    ResumeGenerationCandidateOutputV1,
    ResumeGenerationRunInputV1,
)
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.llm.factory import LLMAccountingError, LLMProviderError
from app.llm.ports import ChatModelPort
from app.tools.contracts import ToolRuntime
from app.worker.contracts import RunExecutionResult


class GenerationRunExecutor:
    def __init__(
        self,
        *,
        reader: RunExecutionReader,
        sessions: SqlAlchemyResumeGenerationStore,
        model_factory: Callable[[TenantContext, UUID], ChatModelPort],
        tools_factory: Callable[[TenantContext, UUID, object], ToolRuntime],
    ) -> None:
        self.reader = reader
        self.sessions = sessions
        self.model_factory = model_factory
        self.tools_factory = tools_factory

    async def execute(
        self, run_id: UUID, tenant: TenantContext, graph_version: str
    ) -> RunExecutionResult:
        if graph_version != "pathfinder-resume-v3":
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
            if not isinstance(execution.request, ResumeGenerationRunInputV1):
                return RunExecutionResult(
                    status=RunStatus.FAILED, error_category="invalid_run_input"
                )
            session_id = execution.request.payload.session_id
            inputs = await self.sessions.execution_inputs(tenant, session_id)
            if inputs.run_id != run_id:
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
                return await self.sessions.spend_allowed(tenant, session_id)

            async def repair() -> bool:
                await self.reader.assert_execution_allowed(
                    run_id=run_id,
                    workspace_id=tenant.workspace_id,
                    actor_user_id=tenant.actor_user_id,
                    graph_version=graph_version,
                )
                return await self.sessions.reserve_repair(tenant, session_id)

            graph = ResumeGenerationGraph(
                model=self.model_factory(tenant, run_id),
                spend_allowed=allowed,
                reserve_repair=repair,
                tools=self.tools_factory(tenant, run_id, inputs.retrieval_scope),
            )
            candidate = await graph.generate(inputs)
            await self.reader.assert_execution_allowed(
                run_id=run_id,
                workspace_id=tenant.workspace_id,
                actor_user_id=tenant.actor_user_id,
                graph_version=graph_version,
            )
            return RunExecutionResult(
                status=RunStatus.COMPLETED,
                result=ResumeGenerationCandidateOutputV1(payload=candidate),
            )
        except asyncio.CancelledError:
            raise
        except RunExecutionCancelledError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except DomainNotFoundError:
            return RunExecutionResult(status=RunStatus.CANCELLED)
        except RunExecutionInvalidError as error:
            return RunExecutionResult(status=RunStatus.FAILED, error_category=error.category)
        except GenerationError as error:
            return RunExecutionResult(status=RunStatus.FAILED, error_category=error.code)
        except DomainValidationError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="invalid_generation_input"
            )
        except DomainUnavailableError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="database_unavailable", retryable=True
            )
        except LLMAccountingError as error:
            return RunExecutionResult(
                status=RunStatus.FAILED,
                error_category="llm_accounting_failed",
                retryable=error.retryable,
            )
        except LLMProviderError:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="provider_unavailable", retryable=True
            )
