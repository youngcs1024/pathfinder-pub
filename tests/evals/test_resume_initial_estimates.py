"""Unknown accounting stays immutable and cannot authorize another tenant or replay."""

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_initial_estimates import POLICY_FILE, apply_estimates


def setup(tmp_path):
    tmp_path.chmod(0o700)
    workspace, actor, key = uuid4(), uuid4(), uuid4()
    binding = {"workspace_id": str(workspace), "actor_user_id": str(actor)}
    row = SimpleNamespace(
        id=key,
        workspace_id=workspace,
        actor_user_id=actor,
        run_id=None,
        provider="qwen",
        invocation_kind="chat",
        model="synthetic",
        graph_node="selection",
        request_hash="sha256:synthetic",
        status="failed",
        error_category="provider_unavailable",
        token_usage=None,
        estimated_cost=None,
        pricing_version=None,
        currency=None,
        latency_ms=10,
    )
    publish(
        tmp_path / POLICY_FILE,
        {
            "version": "a_terminal_unknown_estimate_v1",
            "binding": binding,
            "authorization": "Synthetic user approval",
            "estimated_cost_cny": "1",
            "historical_invocation_ids": [str(key)],
            "eligible_existing_ids": [str(key)],
        },
    )
    publish(
        tmp_path / f"attempt-{key}.json",
        {
            "binding": quality_identity_digest(binding),
            "invocation_id": str(key),
            "request_hash": row.request_hash,
            "graph_node": row.graph_node,
        },
    )
    usage = {
        "unknown_cost": 1,
        "unknown_usage": 1,
        "reserved_cost_cny": "0",
        "budget_occupied_cny": "5",
        "remaining_admission_cny": "95",
    }
    return binding, row, usage


def test_estimate_replay_preserves_unknowns_and_adds_once(tmp_path):
    binding, row, usage = setup(tmp_path)
    first = apply_estimates(tmp_path, binding, [row], usage)
    assert first == apply_estimates(tmp_path, binding, [row], usage)
    assert first["unknown_cost"] == first["unknown_usage"] == 1
    assert first["estimated_unknown_cost_cny"] == "1"
    assert first["budget_occupied_cny"] == "6"
    assert row.estimated_cost is None and row.token_usage is None


@pytest.mark.parametrize("mutation", ["tenant", "actor", "request", "node", "provider"])
def test_estimate_rejects_wrong_identity(tmp_path, mutation):
    binding, row, usage = setup(tmp_path)
    if mutation == "tenant":
        row.workspace_id = uuid4()
    if mutation == "actor":
        row.actor_user_id = uuid4()
    if mutation == "request":
        row.request_hash = "different"
    if mutation == "node":
        row.graph_node = "different"
    if mutation == "provider":
        row.provider = "fake"
    with pytest.raises(AcceptanceError):
        apply_estimates(tmp_path, binding, [row], usage)


def test_active_call_not_estimated_and_partial_accounting_kept(tmp_path):
    binding, row, usage = setup(tmp_path)
    row.status = "started"
    assert apply_estimates(tmp_path, binding, [row], usage)["a_unknown_estimates"] == {}
    row.status = "failed"
    row.estimated_cost = Decimal("0.02")
    got = apply_estimates(tmp_path, binding, [row], usage)
    assert got["estimated_unknown_cost_cny"] == "0"
    assert got["estimated_unknown_cost_count"] == 0
    assert got["estimated_unknown_usage_count"] == 1


async def test_generation_unknown_is_not_replayed(tmp_path):
    from app.llm.invocations import LOCKED_CHAT_MODEL, chat_request_hash
    from app.llm.ports import ChatMessage
    from tests.evals.resume_initial_recording import EstimatedGenerationFailure, JournalModel

    _binding, row, _usage = setup(tmp_path)
    messages = (ChatMessage(role="user", content="synthetic"),)
    metadata = {"graph_node": "selection"}
    row.request_hash = chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=())
    identity = quality_identity_digest(
        {
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": [],
            "metadata": metadata,
        }
    )
    publish(
        tmp_path / "call-00-started.json", {"identity": identity, "before": {"invocation_ids": []}}
    )

    async def check_admission(**kwargs):
        return {"a_unknown_estimates": {str(row.id): {"ledger": {"status": "failed"}}}}

    async def audit():
        return [row]

    async def invoke(*args):
        pytest.fail("Unknown generation must not be replayed")

    journal = JournalModel(
        SimpleNamespace(model=LOCKED_CHAT_MODEL, invoke=invoke),
        tmp_path,
        SimpleNamespace(check_admission=check_admission, audit=audit),
    )
    with pytest.raises(EstimatedGenerationFailure):
        await journal.invoke(messages, (), metadata)
    assert not (tmp_path / "call-00-response.json").exists()


async def test_estimates_still_enforce_shared_cap():
    from tests.evals.resume_initial_reconciliation import ARecorder

    usage = {
        "attempts": 404,
        "unfinished": 0,
        "unknown_cost": 1,
        "unknown_usage": 1,
        "reserved_unknown_attempts": 0,
        "reserved_cost_cny": "0",
        "known_cost_cny": "99.5",
        "estimated_unknown_cost_count": 1,
        "estimated_unknown_usage_count": 1,
        "estimated_unknown_cost_cny": "1",
    }

    async def measurement():
        return usage

    recorder = SimpleNamespace(
        measurement=measurement, budget=SimpleNamespace(cost_cap_cny=Decimal(100), attempt_cap=3000)
    )
    with pytest.raises(AcceptanceError, match="budget_exhausted"):
        await ARecorder.checked_usage(recorder, after=True)


def test_accounting_compatibility_is_exact_for_reviewed_files():
    import ast
    import hashlib
    from pathlib import Path

    from tests.evals.resume_initial_reuse import ACCOUNTING_COMPATIBILITY

    recording = Path("tests/evals/resume_initial_recording.py").read_bytes()
    mapping = ACCOUNTING_COMPATIBILITY["tests/evals/resume_initial_recording.py"]
    assert hashlib.sha256(recording).hexdigest() in mapping
    assert hashlib.sha256(recording + b"\n# unreviewed mutation\n").hexdigest() not in mapping
    tree = ast.parse(Path("tests/evals/resume_initial.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_sample"
    )
    digest = quality_identity_digest(ast.dump(function, include_attributes=False))
    assert digest in ACCOUNTING_COMPATIBILITY["generate_sample"]
    function.body.append(ast.Pass())
    changed = quality_identity_digest(ast.dump(function, include_attributes=False))
    assert changed not in ACCOUNTING_COMPATIBILITY["generate_sample"]
