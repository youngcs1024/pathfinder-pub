"""Exact quote recovery and bounded paid-stage reconciliation contracts."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.llm.ports import ChatModelResult
from tests.evals.product_acceptance_contracts import AcceptanceError
from tests.evals.resume_initial_scoring import packets_for
from tests.evals.resume_initial_unit_review import (
    QuoteClaim,
    exact_span,
    packet_for,
    recover,
    validate,
)
from tests.evals.test_resume_initial_scoring import fixture


def quote_value(packet):
    return {
        "units": [
            {
                "unit": packet["items"][0]["id"],
                "nonfactual": None,
                "claims": [
                    {
                        "whole": True,
                        "quote": None,
                        "occurrence": 0,
                        "support": "full",
                        "facts": ["F0"],
                        "profile": [],
                        "experimental": False,
                        "conditions_complete": True,
                        "rationale": "Synthetic evidence",
                    }
                ],
            }
        ]
    }


def test_exact_quotes_repeated_unicode_and_whole_conflict():
    claim = quote_value({"items": [{"id": "U0"}]})["units"][0]["claims"][0]
    c = QuoteClaim.model_validate_json(
        json.dumps({**claim, "whole": False, "quote": "甲😀", "occurrence": 1})
    )
    assert exact_span("甲😀/甲😀", c) == (3, 5)
    with pytest.raises(AcceptanceError, match="exact_quote_not_found"):
        exact_span("甲😀", c)
    with pytest.raises(AcceptanceError, match="whole_quote_conflict"):
        exact_span("甲😀", c.model_copy(update={"whole": True}))


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("fact", "unknown_fact"),
        ("support", "support_missing"),
        ("unit", "unit_denominator"),
        ("quote", "exact_quote_not_found"),
    ],
)
def test_unit_recovery_remains_strict(mutation, error):
    packets, mapping = packets_for(*fixture())
    packet = packet_for(packets[0], packets[0]["items"][0])
    value = quote_value(packet)
    c = value["units"][0]["claims"][0]
    if mutation == "fact":
        c["facts"] = ["F999"]
    elif mutation == "support":
        c["facts"] = []
    elif mutation == "unit":
        value["units"][0]["unit"] = "U999"
    else:
        c.update(whole=False, quote="absent quote")
    with pytest.raises(AcceptanceError, match=error):
        validate(value, packet, mapping)


async def test_unit_recovery_bounded_and_replay(tmp_path):
    packets, mapping = packets_for(*fixture())
    calls = []

    async def call(path, payload, stage):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        calls.append(stage)
        if stage == "unit":
            assert "previous_response" not in payload
            return ChatModelResult(content="invalid")
        return ChatModelResult(content=json.dumps(quote_value(payload)))

    packet = packets[0]
    result = await recover(packet, mapping, tmp_path / "recovery", call)
    assert result["status"] == "ASSESSED"
    assert len(calls) == 2 * len(packet["items"])
    assert await recover(packet, mapping, tmp_path / "recovery", call) == result
    assert len(calls) == 2 * len(packet["items"])


async def test_reconciliation_rejects_ambiguous_or_mismatched_request(tmp_path):
    from app.llm.invocations import LOCKED_CHAT_MODEL, chat_request_hash
    from app.llm.ports import ChatMessage
    from tests.evals.resume_experiments import preserve
    from tests.evals.resume_initial_reconciliation import call_identity, reconcile_started

    messages = (ChatMessage(role="user", content="synthetic"),)
    metadata = {"graph_node": "score_initial"}
    started = tmp_path / "started.json"
    preserve(
        started, {"identity": call_identity(messages, metadata), "before": {"invocation_ids": []}}
    )
    rows = [
        SimpleNamespace(
            id=uuid4(),
            request_hash=chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=()),
            graph_node="score_initial",
        )
        for _ in range(2)
    ]

    async def audit():
        return rows

    with pytest.raises(AcceptanceError, match="score_reconciliation_ambiguous"):
        await reconcile_started(SimpleNamespace(audit=audit), started, messages, metadata)
    with pytest.raises(AcceptanceError, match="score_identity_changed"):
        await reconcile_started(
            SimpleNamespace(audit=audit), started, messages, {"graph_node": "score_review"}
        )


@pytest.mark.parametrize("mutation", ["none", "run", "tenant", "ledger", "journal", "response"])
def test_generation_interruption_exact_binding(tmp_path, mutation):
    from tests.evals.product_acceptance_contracts import publish, read_private_json
    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_initial_interruption import AUTHORIZATION, validate_authorization
    from tests.evals.resume_live_budget import ledger_identity

    tmp_path.chmod(0o700)
    row = SimpleNamespace(
        id=uuid4(),
        workspace_id=uuid4(),
        actor_user_id=uuid4(),
        run_id=uuid4(),
        provider="qwen",
        invocation_kind="chat",
        model="synthetic",
        graph_node="analyze_job",
        request_hash="synthetic",
        status="started",
        error_category=None,
        token_usage=None,
        estimated_cost=None,
        pricing_version=None,
        currency=None,
        latency_ms=None,
    )
    binding = {"workspace_id": str(row.workspace_id), "actor_user_id": str(row.actor_user_id)}
    path = tmp_path / "a" / ("a" * 40) / "formal/test-r1-pathfinder/call-00-started.json"
    path.parent.mkdir(mode=0o700, parents=True)
    publish(path, {"identity": "exact", "before": {"invocation_ids": []}})
    value = {
        "policy": "single_interrupted_generation_no_replay_v1",
        "binding": binding,
        "authorization": "Synthetic approval",
        "reserved_cost_cny": "10.4623104",
        "invocation_id": str(row.id),
        "ledger": ledger_identity(row),
        "run_id": str(row.run_id),
        "source_sha": "a" * 40,
        "sample_id": "test-r1-pathfinder",
        "started_path": str(path.relative_to(tmp_path)),
        "started_digest": quality_identity_digest(read_private_json(path)),
    }
    if mutation == "run":
        value["run_id"] = str(uuid4())
    elif mutation == "tenant":
        value["binding"] = {**binding, "workspace_id": str(uuid4())}
    elif mutation == "ledger":
        row.request_hash = "changed"
    elif mutation == "journal":
        value["started_digest"] = "changed"
    elif mutation == "response":
        publish(path.with_name("call-00-response.json"), {"content": "known"})
    publish(tmp_path / AUTHORIZATION, value)
    if mutation == "none":
        assert validate_authorization(tmp_path, binding, [row]) == value
    else:
        with pytest.raises(AcceptanceError, match="interruption_"):
            validate_authorization(tmp_path, binding, [row])
