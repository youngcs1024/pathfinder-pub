"""Budget admission from the authoritative PostgreSQL attempt ledger."""

from decimal import Decimal

from sqlalchemy import select

from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.models import LLMInvocation
from tests.evals.live_suite import can_admit_attempt
from tests.evals.product_acceptance_contracts import Budget, require


def summarize(rows, *, provider):
    return {
        "attempts": len(rows),
        "unfinished": sum(r.status == "started" for r in rows),
        "unknown_cost": sum(r.estimated_cost is None for r in rows) if provider == "qwen" else 0,
        "unknown_usage": sum(r.token_usage is None for r in rows),
        "input_tokens": sum((r.token_usage or {}).get("input_tokens", 0) for r in rows),
        "output_tokens": sum((r.token_usage or {}).get("output_tokens", 0) for r in rows),
        "known_cost_cny": str(
            sum((r.estimated_cost for r in rows if r.estimated_cost is not None), Decimal(0))
        ),
        "pricing_applicable": provider == "qwen",
    }


def admit(usage, budget, *, after=False):
    require(not usage["unfinished"], "unfinished_accounting")
    require(
        not usage["pricing_applicable"]
        or (not usage["unknown_cost"] and not usage["unknown_usage"]),
        "unknown_usage_or_cost",
    )
    known = Decimal(usage["known_cost_cny"])
    if after:
        allowed = (
            known <= budget.cost_admission_budget_cny
            and usage["attempts"] <= budget.provider_attempt_cap
            and usage["input_tokens"] <= budget.input_token_cap
            and usage["output_tokens"] <= budget.output_token_cap
        )
    else:
        allowed = can_admit_attempt(
            budget,
            known_cost_cny=known,
            unknown_cost_attempt_count=usage["unknown_cost"],
            provider_attempts=usage["attempts"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
    require(allowed, "budget_exhausted")


class DatabaseBudgetRecorder:
    def __init__(self, sessions, tenant, *, provider, source_check=lambda: None):
        self.sessions, self.tenant, self.provider = sessions, tenant, provider
        self.delegate = SqlAlchemyInvocationRecorder(sessions)
        self.budget = Budget()
        self.source_check = source_check
        self.stopped = False

    async def rows(self):
        async with self.sessions() as session:
            return list(
                await session.scalars(
                    select(LLMInvocation)
                    .where(
                        LLMInvocation.workspace_id == self.tenant.workspace_id,
                        LLMInvocation.actor_user_id == self.tenant.actor_user_id,
                    )
                    .order_by(LLMInvocation.created_at, LLMInvocation.id)
                )
            )

    async def usage(self):
        return summarize(await self.rows(), provider=self.provider)

    async def prepare(self, attempt):
        try:
            require(not self.stopped, "execution_stopped")
            require(
                (attempt.workspace_id, attempt.actor_user_id, attempt.provider)
                == (self.tenant.workspace_id, self.tenant.actor_user_id, self.provider),
                "attempt_scope_mismatch",
            )
            self.source_check()
            admit(await self.usage(), self.budget)
            await self.delegate.prepare(attempt)
        except BaseException:
            self.stopped = True
            raise

    async def finalize(self, attempt, outcome):
        try:
            await self.delegate.finalize(attempt, outcome)
            admit(await self.usage(), self.budget, after=True)
        except BaseException:
            self.stopped = True
            raise
