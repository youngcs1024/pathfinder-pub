"""One PostgreSQL attempt budget across material, B1 and system runs."""

from tests.evals.product_acceptance_budget import DatabaseBudgetRecorder


class ComparisonRecorder(DatabaseBudgetRecorder):
    def __init__(self, sessions, tenant, *, provider, budget, source_check=lambda: None):
        super().__init__(sessions, tenant, provider=provider, source_check=source_check)
        self.budget = budget

    async def measurement(self):
        value = await self.usage()
        rows = await self.rows()
        return {
            **value,
            "latency_ms": sum(row.latency_ms or 0 for row in rows),
            "invocation_ids": [str(row.id) for row in rows],
            "cost_status": "ESTIMATED" if self.provider == "qwen" else "NOT_APPLICABLE",
        }
