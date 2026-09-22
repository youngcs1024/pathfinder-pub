"""Dispatch only explicitly composed executable versions; never fall back to an old graph."""

from collections.abc import Mapping
from types import MappingProxyType
from uuid import UUID

from app.domain.runs import EXECUTABLE_GRAPH_VERSIONS, RunStatus
from app.domain.tenancy import TenantContext
from app.worker.contracts import RunExecutionResult, RunExecutor


class RunExecutorDispatcher:
    def __init__(self, executors: Mapping[str, RunExecutor]) -> None:
        if frozenset(executors) != EXECUTABLE_GRAPH_VERSIONS:
            raise ValueError("executor bindings must match supported execution versions")
        self._executors = MappingProxyType(dict(executors))

    async def execute(
        self, run_id: UUID, tenant: TenantContext, graph_version: str
    ) -> RunExecutionResult:
        executor = self._executors.get(graph_version)
        if executor is None:
            return RunExecutionResult(
                status=RunStatus.FAILED, error_category="unknown_graph_version"
            )
        return await executor.execute(run_id, tenant, graph_version)
