"""Explicit A-only estimates; missing provider accounting is never rewritten."""

from decimal import Decimal

from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiments import preserve
from tests.evals.resume_live_budget import ledger_identity

POLICY_FILE = "a-unknown-estimate-authorization.json"
AMOUNT = Decimal("1")


def policy(root, binding):
    path = root / POLICY_FILE
    if not path.exists():
        require(
            not (root / "a-unknown-estimates").exists()
            and not (root / "a-unknown-estimate-pin.json").exists(),
            "estimate_authorization_removed",
        )
        return None
    value = read_private_json(path)
    require(
        value["version"] == "a_terminal_unknown_estimate_v1"
        and value["binding"] == binding
        and value["authorization"]
        and value["estimated_cost_cny"] == str(AMOUNT)
        and value["historical_invocation_ids"]
        and set(value["eligible_existing_ids"]) <= set(value["historical_invocation_ids"]),
        "estimate_authorization_changed",
    )
    preserve(root / "a-unknown-estimate-pin.json", {"digest": quality_identity_digest(value)})
    return value


def apply_estimates(root, binding, rows, usage):
    value = policy(root, binding)
    if value is None:
        return usage
    historical = set(value["historical_invocation_ids"])
    existing = set(value["eligible_existing_ids"])
    excluded = set(usage.get("a_score_timeouts", {}))
    if usage.get("generation_interruption"):
        excluded.add(usage["generation_interruption"]["invocation_id"])
    if usage.get("timeout_exception"):
        excluded.add(usage["timeout_exception"]["invocation_id"])
    receipts = root / "a-unknown-estimates"
    expected = {}
    for row in rows:
        key = str(row.id)
        if key in excluded or (key in historical and key not in existing):
            continue
        if row.status not in ("failed", "succeeded"):
            continue  # An active/unreconciled execution is never an estimated terminal outcome.
        if row.estimated_cost is not None and row.token_usage is not None:
            continue
        require(
            row.provider == "qwen"
            and row.invocation_kind == "chat"
            and str(row.workspace_id) == binding["workspace_id"]
            and str(row.actor_user_id) == binding["actor_user_id"],
            "estimate_call_identity_changed",
        )
        attempt = read_private_json(root / f"attempt-{key}.json")
        require(
            attempt["binding"] == quality_identity_digest(binding)
            and attempt["invocation_id"] == key
            and attempt["request_hash"] == row.request_hash
            and attempt["graph_node"] == row.graph_node,
            "estimate_journal_changed",
        )
        receipt = {
            "policy_digest": quality_identity_digest(value),
            "ledger": ledger_identity(row),
            "estimated_cost_cny": str(AMOUNT) if row.estimated_cost is None else "0",
            "cost_is_estimate": True,
            "usage_status": "UNKNOWN" if row.token_usage is None else "KNOWN",
            "provider_outcome": row.status,
        }
        receipts.mkdir(mode=0o700, exist_ok=True)
        preserve(receipts / f"{key}.json", receipt)
        expected[key] = receipt
    if receipts.exists():
        require(
            {p.stem for p in receipts.glob("*.json")} == expected.keys(), "estimate_receipt_changed"
        )
    amount = sum((Decimal(v["estimated_cost_cny"]) for v in expected.values()), Decimal(0))
    return {
        **usage,
        "a_unknown_estimates": expected,
        "estimated_unknown_cost_count": sum(
            v["ledger"]["estimated_cost"] is None for v in expected.values()
        ),
        "estimated_unknown_usage_count": sum(
            v["ledger"]["token_usage"] is None for v in expected.values()
        ),
        "estimated_unknown_cost_cny": str(amount),
        "budget_occupied_cny": str(Decimal(usage["budget_occupied_cny"]) + amount),
        "remaining_admission_cny": str(Decimal(usage["remaining_admission_cny"]) - amount),
    }


def estimated_failure(usage, invocation_id):
    receipt = usage.get("a_unknown_estimates", {}).get(str(invocation_id))
    return receipt is not None and receipt["ledger"]["status"] == "failed"
