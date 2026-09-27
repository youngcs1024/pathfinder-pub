"""Explicit R7.1-only timeout reservations; never rewrite unknown provider accounting."""

from decimal import Decimal

from tests.evals.product_acceptance_budget import admit, summarize
from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_quality_budget import ComparisonRecorder

AUTH_FILE = "timeout-recovery-authorization.json"
PIN_FILE = "timeout-recovery-binding.json"
RESERVE = Decimal("1")


def ledger_identity(row):
    """Freeze authoritative attempt identity and outcome, without request/response bodies."""
    return {
        key: str(value) if isinstance(value, Decimal) else value
        for key, value in {
            "id": str(row.id),
            "workspace_id": str(row.workspace_id),
            "actor_user_id": str(row.actor_user_id),
            "run_id": str(row.run_id) if row.run_id else None,
            "provider": row.provider,
            "model": row.model,
            "graph_node": row.graph_node,
            "request_hash": row.request_hash,
            "status": row.status,
            "error_category": row.error_category,
            "token_usage": row.token_usage,
            "estimated_cost": row.estimated_cost,
            "pricing_version": row.pricing_version,
            "currency": row.currency,
            "latency_ms": row.latency_ms,
        }.items()
    }


def validate_authorization(value, *, inputs, tenant, rows):
    require(
        value.get("kind") == "r71_timeout_recovery_v1"
        and value.get("approved_by") == "user_explicit_r71_timeout_reservation"
        and value.get("authorization_digest") == inputs.digest
        and value.get("allocation_id") == str(inputs.allocation_id)
        and value.get("execution_root") == inputs.execution_root
        and value.get("workspace_id") == str(tenant.workspace_id)
        and value.get("actor_user_id") == str(tenant.actor_user_id)
        and value.get("budget") == inputs.budget.model_dump(mode="json")
        and value.get("reserve_per_timeout_cny") == "1"
        and value.get("future_timeouts_allowed") is True,
        "timeout_authorization_mismatch",
    )
    require(
        all(
            row.provider == "qwen"
            and row.workspace_id == tenant.workspace_id
            and row.actor_user_id == tenant.actor_user_id
            for row in rows
        ),
        "timeout_ledger_scope_mismatch",
    )
    anchor = value.get("ledger_anchor", [])
    require(bool(anchor), "timeout_ledger_anchor_missing")
    ids = [item["id"] for item in anchor]
    current = {str(row.id): ledger_identity(row) for row in rows}
    require(
        len(ids) == len(set(ids)) and all(current.get(item["id"]) == item for item in anchor),
        "timeout_ledger_anchor_changed",
    )


def reservation(rows):
    usage = summarize(rows, provider="qwen")
    unknown = [r for r in rows if r.estimated_cost is None or r.token_usage is None]
    eligible = [
        r for r in unknown if r.status == "failed" and r.error_category == "provider_timeout"
    ]
    # Set union: missing cost and missing usage in one attempt reserve only once.
    ids = sorted({str(r.id) for r in eligible})
    reserved = RESERVE * len(ids)
    return {
        "unknown_timeout_ids": ids,
        "reserve_per_timeout_cny": str(RESERVE),
        "reserved_timeout_cost_cny": str(reserved),
        "known_plus_reserved_cny": str(Decimal(usage["known_cost_cny"]) + reserved),
        "unreserved_unknown_ids": sorted({str(r.id) for r in unknown} - set(ids)),
        "actual_total_cost_cny": None,
        "estimated_total_cost_cny": None if unknown else usage["known_cost_cny"],
        "token_usage_complete": not usage["unknown_usage"],
        "reservation_is_actual_cost": False,
    }


def admit_reserved(rows, budget, *, after=False):
    usage, reserved = summarize(rows, provider="qwen"), reservation(rows)
    require(not usage["unfinished"], "unfinished_accounting")
    require(not reserved["unreserved_unknown_ids"], "unknown_usage_or_cost")
    require(
        Decimal(reserved["known_plus_reserved_cny"]) + (Decimal(0) if after else RESERVE)
        <= budget.cost_admission_budget_cny
        and usage["attempts"] <= budget.provider_attempt_cap - (0 if after else 1)
        and usage["input_tokens"] <= budget.input_token_cap - (0 if after else 1)
        and usage["output_tokens"] <= budget.output_token_cap - (0 if after else 1),
        "budget_exhausted",
    )
    return reserved


class LiveComparisonRecorder(ComparisonRecorder):
    def __init__(self, sessions, tenant, *, root, inputs, source_check=lambda: None):
        super().__init__(
            sessions, tenant, provider="qwen", budget=inputs.budget, source_check=source_check
        )
        self.root, self.inputs = root, inputs

    def authorization(self, rows):
        require(
            str(self.root.resolve()) == self.inputs.execution_root,
            "timeout_allocation_root_changed",
        )
        path, pin = self.root / AUTH_FILE, self.root / PIN_FILE
        if not path.exists() and not pin.exists():
            return None
        value, binding = read_private_json(path), read_private_json(pin)
        require(
            binding == {"authorization_file_digest": quality_identity_digest(value)},
            "timeout_authorization_changed",
        )
        validate_authorization(value, inputs=self.inputs, tenant=self.tenant, rows=rows)
        return quality_identity_digest(value)

    async def check_admission(self, *, after=False):
        rows = await self.rows()
        if self.authorization(rows) is None:
            admit(summarize(rows, provider="qwen"), self.budget, after=after)
        else:
            admit_reserved(rows, self.budget, after=after)

    async def measurement(self):
        value = await super().measurement()
        rows = await self.rows()
        digest = self.authorization(rows)
        if digest is not None:
            value["timeout_reservation"] = {
                "authorization_file_digest": digest,
                **reservation(rows),
            }
            if value["unknown_cost"] or value["unknown_usage"]:
                value["cost_status"] = "PARTIAL_ESTIMATE_TIMEOUT_RESERVED"
        return value

    async def prepare(self, attempt):
        try:
            require(not self.stopped, "execution_stopped")
            require(
                (attempt.workspace_id, attempt.actor_user_id, attempt.provider)
                == (self.tenant.workspace_id, self.tenant.actor_user_id, self.provider),
                "attempt_scope_mismatch",
            )
            self.source_check()
            await self.check_admission()
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
