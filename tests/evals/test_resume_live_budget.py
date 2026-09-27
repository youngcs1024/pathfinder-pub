"""Timeout exception is opt-in, cumulative, immutable, and scoped to R7.1 live."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tests.evals.product_acceptance_budget import admit, summarize
from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import ExperimentError
from tests.evals.resume_live_budget import (
    AUTH_FILE,
    PIN_FILE,
    LiveComparisonRecorder,
    admit_reserved,
    ledger_identity,
    reservation,
)
from tests.evals.resume_quality_contracts import ComparisonBudget


def row(tenant, **changes):
    fields = dict(
        id=uuid4(),
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
        run_id=uuid4(),
        provider="qwen",
        model="qwen3.6-flash-2026-04-16",
        graph_node="revise",
        request_hash="sha256:" + "a" * 64,
        status="failed",
        error_category="provider_timeout",
        token_usage=None,
        estimated_cost=None,
        pricing_version=None,
        currency=None,
        latency_ms=120000,
    )
    return SimpleNamespace(**(fields | changes))


def authorize(root, inputs, tenant, rows, **changes):
    value = (
        dict(
            kind="r71_timeout_recovery_v1",
            approved_by="user_explicit_r71_timeout_reservation",
            authorization_digest=inputs.digest,
            allocation_id=str(inputs.allocation_id),
            execution_root=inputs.execution_root,
            workspace_id=str(tenant.workspace_id),
            actor_user_id=str(tenant.actor_user_id),
            budget=inputs.budget.model_dump(mode="json"),
            reserve_per_timeout_cny="1",
            future_timeouts_allowed=True,
            ledger_anchor=[ledger_identity(r) for r in rows],
        )
        | changes
    )
    publish(root / AUTH_FILE, value)
    publish(root / PIN_FILE, {"authorization_file_digest": quality_identity_digest(value)})
    return value


def setup(root):
    tenant = SimpleNamespace(workspace_id=uuid4(), actor_user_id=uuid4())
    inputs = SimpleNamespace(
        digest="original-inputs",
        allocation_id=uuid4(),
        execution_root=str(root),
        budget=ComparisonBudget(),
    )
    rows = [row(tenant)]
    recorder = LiveComparisonRecorder(None, tenant, root=root, inputs=inputs)
    recorder.rows = AsyncMock(side_effect=lambda: list(rows))
    return tenant, inputs, rows, recorder


@pytest.mark.asyncio
async def test_default_stops_and_explicit_exception_preserves_unknown_fields(tmp_path):
    tenant, inputs, rows, recorder = setup(tmp_path)
    before = ledger_identity(rows[0])
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await recorder.check_admission()
    authorize(tmp_path, inputs, tenant, rows)
    await recorder.check_admission()
    first = await recorder.measurement()
    # Rebuilding a recorder cannot reset the anchor, unknown attempt, or reservation.
    resumed = LiveComparisonRecorder(None, tenant, root=tmp_path, inputs=inputs)
    resumed.rows = AsyncMock(return_value=rows)
    assert await resumed.measurement() == first
    assert first["unknown_cost"] == first["unknown_usage"] == 1
    assert first["timeout_reservation"]["reserved_timeout_cost_cny"] == "1"
    assert first["timeout_reservation"]["actual_total_cost_cny"] is None
    assert ledger_identity(rows[0]) == before
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        admit(summarize(rows, provider="qwen"), inputs.budget)


@pytest.mark.asyncio
async def test_future_timeouts_and_partial_usage_reserve_each_attempt_once(tmp_path):
    tenant, inputs, rows, recorder = setup(tmp_path)
    authorize(tmp_path, inputs, tenant, rows)
    rows.append(row(tenant, token_usage={"input_tokens": 100, "output_tokens": 20}))
    await recorder.check_admission()
    evidence = await recorder.measurement()
    assert evidence["attempts"] == 2
    assert evidence["timeout_reservation"]["reserved_timeout_cost_cny"] == "2"
    assert evidence["input_tokens"] == 100


@pytest.mark.parametrize(
    "cost,after,allowed",
    [("18", False, True), ("18.01", False, False), ("19", True, True), ("19.01", True, False)],
)
def test_budget_includes_past_timeout_and_next_attempt(cost, after, allowed):
    tenant = SimpleNamespace(workspace_id=uuid4(), actor_user_id=uuid4())
    rows = [
        row(tenant),
        row(
            tenant,
            status="succeeded",
            error_category=None,
            estimated_cost=Decimal(cost),
            token_usage={"input_tokens": 1},
        ),
    ]
    if allowed:
        admit_reserved(rows, ComparisonBudget(), after=after)
    else:
        with pytest.raises(AcceptanceError, match="budget_exhausted"):
            admit_reserved(rows, ComparisonBudget(), after=after)


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "started"},
        {"error_category": "provider_error"},
        {"status": "succeeded", "error_category": None},
    ],
)
def test_unfinished_and_non_timeout_unknown_still_stop(patch):
    tenant = SimpleNamespace(workspace_id=uuid4(), actor_user_id=uuid4())
    rows = [row(tenant, **patch)]
    with pytest.raises(AcceptanceError, match=r"unfinished_accounting|unknown_usage_or_cost"):
        admit_reserved(rows, ComparisonBudget())
    # Even rejected attempts can be reported without losing their failure evidence.
    assert reservation(rows)["actual_total_cost_cny"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("authorization_digest", "changed"),
        ("allocation_id", "different"),
        ("workspace_id", "different"),
        ("actor_user_id", "different"),
        ("execution_root", "/another/root"),
        ("reserve_per_timeout_cny", "0.1"),
        ("future_timeouts_allowed", False),
    ],
)
async def test_authorization_identity_mismatch_rejected(tmp_path, field, value):
    tenant, inputs, rows, recorder = setup(tmp_path)
    authorize(tmp_path, inputs, tenant, rows, **{field: value})
    with pytest.raises(AcceptanceError, match="timeout_authorization_mismatch"):
        await recorder.check_admission()


@pytest.mark.asyncio
async def test_missing_or_changed_anchor_cannot_reset_ledger(tmp_path):
    tenant, inputs, rows, recorder = setup(tmp_path)
    authorize(tmp_path, inputs, tenant, rows)
    rows[0].estimated_cost = Decimal("0")
    with pytest.raises(AcceptanceError, match="timeout_ledger_anchor_changed"):
        await recorder.check_admission()
    rows.clear()
    with pytest.raises(AcceptanceError, match="timeout_ledger_anchor_changed"):
        await recorder.check_admission()
    with pytest.raises(ExperimentError, match="artifact_publication_failed"):
        authorize(tmp_path, inputs, tenant, rows)


@pytest.mark.parametrize(
    "budget",
    [
        ComparisonBudget(provider_attempt_cap=1),
        ComparisonBudget(input_token_cap=1),
        ComparisonBudget(output_token_cap=1),
    ],
)
def test_attempt_and_known_token_caps_not_relaxed(budget):
    tenant = SimpleNamespace(workspace_id=uuid4(), actor_user_id=uuid4())
    rows = [row(tenant, token_usage={"input_tokens": 1, "output_tokens": 1})]
    with pytest.raises(AcceptanceError, match="budget_exhausted"):
        admit_reserved(rows, budget)


@pytest.mark.asyncio
async def test_changed_authorization_pin_blocks_before_provider(tmp_path):
    tenant, inputs, rows, recorder = setup(tmp_path)
    authorize(tmp_path, inputs, tenant, rows)
    (tmp_path / PIN_FILE).write_text("{}")
    recorder.delegate = SimpleNamespace(prepare=AsyncMock())
    attempt = SimpleNamespace(
        workspace_id=tenant.workspace_id, actor_user_id=tenant.actor_user_id, provider="qwen"
    )
    with pytest.raises(AcceptanceError, match="timeout_authorization_changed"):
        await recorder.prepare(attempt)
    recorder.delegate.prepare.assert_not_awaited()
    assert recorder.stopped


@pytest.mark.asyncio
async def test_non_timeout_failure_remains_in_measurement_after_rejection(tmp_path):
    tenant, inputs, rows, recorder = setup(tmp_path)
    authorize(tmp_path, inputs, tenant, rows)
    rows.append(row(tenant, error_category="provider_unavailable"))
    with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
        await recorder.check_admission(after=True)
    evidence = await recorder.measurement()
    assert evidence["unknown_cost"] == 2
    assert evidence["timeout_reservation"]["unreserved_unknown_ids"] == [str(rows[-1].id)]
