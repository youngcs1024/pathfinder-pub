"""Fail-closed, append-only journal pinned to one existing PostgreSQL ledger."""

from decimal import Decimal
from uuid import uuid4

from tests.evals.product_acceptance_budget import DatabaseBudgetRecorder, summarize
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_live_budget import ledger_identity

# Frozen qwen3.6-flash model envelope includes reasoning, not just max_tokens.
# https://help.aliyun.com/zh/model-studio/qwen3-6-flash (verified 2026-09-27)
# Use the full 65536 response + 131072 reasoning limit, conservatively even though
# the adapter restricts response content to 4096. Unknown-cost continuation is forbidden.
CHAT_HEADROOM_CNY = Decimal("4.8") + Decimal(65536 + 131072) * Decimal("28.8") / Decimal(1000000)
EMBEDDING_HEADROOM_CNY = Decimal("5")  # 10 items, <= 1M tokens each at CNY .5/M.


def admit(usage, budget, *, after=False, kind="chat"):
    require(not usage["unfinished"], "unfinished_accounting")
    require(not usage["unknown_usage"] and not usage["unknown_cost"], "unknown_usage_or_cost")
    amount = Decimal(usage["known_cost_cny"])
    require(amount.is_finite() and amount >= 0, "invalid_cost")
    if after:
        require(
            amount <= budget.cost_cap_cny and usage["attempts"] <= budget.attempt_cap,
            "budget_exhausted",
        )
    else:
        headroom = CHAT_HEADROOM_CNY if kind == "chat" else EMBEDDING_HEADROOM_CNY
        require(
            amount + headroom <= budget.cost_cap_cny and usage["attempts"] < budget.attempt_cap,
            "budget_exhausted",
        )


class ExperimentRecorder(DatabaseBudgetRecorder):
    def __init__(
        self,
        sessions,
        tenant,
        *,
        root,
        inputs,
        database_identity,
        provider="qwen",
        source_check=lambda: None,
    ):
        super().__init__(sessions, tenant, provider=provider, source_check=source_check)
        self.root, self.inputs, self.budget = root, inputs, inputs.budget
        self.binding = {
            "allocation_id": str(inputs.allocation_id),
            "execution_root": str(root.resolve()),
            "input_digest": inputs.digest,
            "database_identity": database_identity,
            "workspace_id": str(tenant.workspace_id),
            "actor_user_id": str(tenant.actor_user_id),
            "provider": provider,
        }

    async def initialize(self):
        pin = self.root / "ledger-binding.json"
        if pin.exists():
            require(read_private_json(pin) == self.binding, "ledger_binding_changed")
        else:
            require(not await self.rows(), "unbound_ledger_not_empty")
            require(not list(self.root.glob("attempt-*.json")), "ledger_binding_missing")
            publish(pin, self.binding)
        await self.audit()

    async def audit(self):
        require(
            read_private_json(self.root / "ledger-binding.json") == self.binding,
            "ledger_binding_changed",
        )
        rows = await self.rows()
        actual = {str(r.id): ledger_identity(r) for r in rows}
        expected = {}
        for path in self.root.glob("attempt-*.json"):
            item = read_private_json(path)
            require(
                item["binding"] == quality_identity_digest(self.binding), "attempt_binding_changed"
            )
            expected[item["invocation_id"]] = item
        require(actual.keys() == expected.keys(), "ledger_attempt_mismatch")
        for key, row in actual.items():
            require(
                expected[key]["request_hash"] == row["request_hash"]
                and expected[key]["graph_node"] == row["graph_node"],
                "attempt_identity_changed",
            )
            done = self.root / f"accounted-{key}.json"
            if done.exists():
                require(read_private_json(done) == row, "ledger_history_changed")
            elif row["status"] != "started":
                # A finalize-to-file crash is recoverable only from authoritative terminal data.
                publish(done, row)
        return rows

    async def measurement(self):
        rows = await self.audit()
        usage = summarize(rows, provider=self.provider)
        return {
            **usage,
            "invocation_ids": [str(r.id) for r in rows],
            "latency_ms": sum(r.latency_ms or 0 for r in rows),
            "cost_status": "ESTIMATED" if self.provider == "qwen" else "NOT_APPLICABLE",
            "remaining_attempts": max(0, self.budget.attempt_cap - len(rows)),
            "remaining_known_cost_cny": str(
                self.budget.cost_cap_cny - Decimal(usage["known_cost_cny"])
            ),
            "ledger_digest": quality_identity_digest([ledger_identity(r) for r in rows]),
        }

    async def check_admission(self, *, after=False, kind="chat"):
        self.source_check()
        usage = await self.measurement()
        admit(usage, self.budget, after=after, kind=kind)
        return usage

    async def prepare(self, attempt):
        try:
            require(not self.stopped, "execution_stopped")
            require(
                (attempt.workspace_id, attempt.actor_user_id, attempt.provider)
                == (self.tenant.workspace_id, self.tenant.actor_user_id, self.provider),
                "attempt_scope_mismatch",
            )
            await self.check_admission(kind=attempt.invocation_kind)
            publish(
                self.root / f"attempt-{attempt.invocation_id}.json",
                {
                    "invocation_id": str(attempt.invocation_id),
                    "binding": quality_identity_digest(self.binding),
                    "request_hash": attempt.request_hash,
                    "graph_node": attempt.graph_node,
                },
            )
            await self.delegate.prepare(attempt)
        except BaseException:
            self.stopped = True
            raise

    async def finalize(self, attempt, outcome):
        try:
            await self.delegate.finalize(attempt, outcome)
            await self.check_admission(after=True)
        except BaseException:
            self.stopped = True
            raise

    async def snapshot(self):
        usage = await self.measurement()
        publish(self.root / f"ledger-snapshot-{uuid4().hex}.json", usage)
        return usage
