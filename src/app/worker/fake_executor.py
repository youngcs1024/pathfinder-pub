from uuid import UUID

from app.domain.research import ResearchLimitationV1, ResearchOutputV1
from app.domain.runs import RunStatus
from app.domain.tenancy import TenantContext
from app.worker.contracts import RunExecutionResult

CONTRACT_FAKE_LIMITATION = (
    "Gate 4 Step 4.3 contract fake completed queue processing without executing the research graph."
)


class ContractFakeRunExecutor:
    async def execute(
        self,
        run_id: UUID,
        tenant: TenantContext,
        graph_version: str,
    ) -> RunExecutionResult:
        if not isinstance(run_id, UUID):
            raise TypeError("contract fake run_id must be a UUID")
        if not isinstance(tenant, TenantContext):
            raise TypeError("contract fake tenant must be a TenantContext")
        if not isinstance(graph_version, str) or not graph_version.strip():
            raise TypeError("contract fake graph_version must be non-blank")
        return RunExecutionResult(
            status=RunStatus.COMPLETED,
            result=ResearchOutputV1(
                evidence_sufficient=False,
                limitations=(
                    ResearchLimitationV1(
                        code="insufficient_evidence",
                        detail=CONTRACT_FAKE_LIMITATION,
                    ),
                ),
            ),
        )
