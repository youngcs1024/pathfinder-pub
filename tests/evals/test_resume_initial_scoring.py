"""Synthetic v2 scorer failures, isolation and bounded recovery contracts."""

import json

import pytest

from app.llm.ports import ChatModelResult
from tests.evals.product_acceptance_contracts import AcceptanceError
from tests.evals.resume_initial_scoring import assess, packets_for, validate_batch
from tests.unit.agents.test_resume_generation import _inputs


def synthetic_response(packet):
    if packet["kind"] == "coverage":
        return {
            "coverage": [
                {
                    "requirement": i["id"],
                    "status": "full" if i["support"] == "full" else "none",
                    "rationale": "Synthetic",
                }
                for i in packet["items"]
            ]
        }
    return {
        "units": [
            {
                "unit": u["id"],
                "claims": [
                    {
                        "start": 0,
                        "end": -1,
                        "support": "full",
                        "facts": ["F0"],
                        "profile": [],
                        "experimental": False,
                        "conditions_complete": True,
                        "rationale": "Synthetic",
                    }
                ],
                "nonfactual": None,
            }
            for u in packet["items"]
        ]
    }


def fixture():
    _, _, inputs, _, fact_id = _inputs()
    content = inputs.profile_content.model_dump(mode="json")
    facts = [
        {
            "version_id": str(fact_id),
            "claim": "Confirmed implementation",
            "kind": "implementation",
            "conditions": {},
        }
    ]
    annotation = {
        "requirements": [
            {
                "requirement_id": "requirement_long",
                "quote": "Python",
                "start": 0,
                "end": 6,
                "kind": "explicit",
                "applicable": True,
                "support": "full",
                "fact_version_ids": [str(fact_id)],
                "profile_item_ids": [],
                "necessary_conditions": [],
                "rationale": "Synthetic",
            }
        ]
    }
    return content, facts, {}, annotation, "Python"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("fact", "unknown_fact"),
        ("profile", "unknown_profile_item"),
        ("span", "claim_span_invalid"),
        ("missing", "unit_denominator"),
        ("support", "support_missing"),
    ],
)
def test_exact_reference_and_complete_units(mutation, error):
    packets, mapping = packets_for(*fixture())
    packet = packets[0]
    value = synthetic_response(packet)
    claim = value["units"][0]["claims"][0]
    if mutation == "fact":
        claim["facts"] = ["F99"]
    if mutation == "profile":
        claim["profile"] = ["P99"]
    if mutation == "span":
        claim["end"] = 99999
    if mutation == "missing":
        value["units"].pop()
    if mutation == "support":
        claim["facts"] = []
    with pytest.raises(AcceptanceError, match=error):
        validate_batch(value, packet, mapping)


def test_chunking_privacy_and_zero_support():
    values = fixture()
    values[3]["requirements"][0]["support"] = "insufficient"
    packets, mapping = packets_for(*values)
    assert all(len(p["items"]) <= 8 for p in packets)
    serialized = json.dumps(packets)
    assert "canary@example.test" not in serialized
    assert values[1][0]["version_id"] not in serialized
    coverage = packets[-1]
    v = synthetic_response(coverage)
    v["coverage"][0]["status"] = "full"
    with pytest.raises(AcceptanceError, match="unsupported_coverage"):
        validate_batch(v, coverage, mapping)


@pytest.mark.parametrize("valid_stage", ["initial", "correction", "review", None])
async def test_bounded_recovery_and_replay(tmp_path, valid_stage):
    calls = []

    async def call(path, payload, stage):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        calls.append((path, payload, stage))
        if stage == "review":
            assert "previous_response" not in payload and "validation_error" not in payload
        if stage == valid_stage:
            return ChatModelResult(content=json.dumps(synthetic_response(payload)))
        return ChatModelResult(content="```", finish_status="incomplete")

    values = fixture()
    assessment, outcomes = await assess(*values, tmp_path / "score", call)
    batches = len(packets_for(*values)[0])
    multiplier = {"initial": 1, "correction": 2, "review": 3, None: 3}[valid_stage]
    assert len(calls) == batches * multiplier
    assert (assessment is not None) == (valid_stage is not None)
    before = len(calls)
    repeated = await assess(*values, tmp_path / "score", call)
    assert repeated == (assessment, outcomes) and len(calls) == before


def test_reuse_requires_matching_dependencies_and_audited_artifacts(tmp_path, monkeypatch):
    import hashlib

    from tests.evals import resume_initial_reuse as reuse
    from tests.evals.product_acceptance_contracts import publish

    source = "a" * 40
    root = tmp_path / "root"
    phase = root / "a" / source / "pilot"
    phase.mkdir(mode=0o700, parents=True)
    phase.parent.chmod(0o700)
    manifest = {
        "source": {"source_sha": source},
        "ci": {},
        "input_digest": "input",
        "preparation_digest": "prep",
        "d_evidence": [],
        "seed": 42,
        "budget": {},
        "pilot": [{"sample_id": "sample"}],
        "prompts": {k: "same" for k in ("one_shot", "selection", "pathfinder", "annotate")},
    }
    publish(phase.parent / "manifest.json", manifest)
    (phase / "sample").mkdir(mode=0o700)
    publish(phase / "sample/generation.json", {"output": None})
    audit = {
        "artifact_sha256": {
            "sample/generation.json": hashlib.sha256(
                (phase / "sample/generation.json").read_bytes()
            ).hexdigest()
        }
    }
    publish(phase / "audit.json", audit)
    monkeypatch.setattr(reuse, "validate_ci", lambda *args: None)
    monkeypatch.setattr(reuse, "generation_fingerprint", lambda ref="HEAD": "same")
    assert reuse.verify_origin(root, source, manifest)[0] == phase.parent
    monkeypatch.setattr(reuse, "generation_fingerprint", lambda ref="HEAD": ref)
    with pytest.raises(AcceptanceError, match="reuse_generation_code_changed"):
        reuse.verify_origin(root, source, manifest)
    monkeypatch.setattr(reuse, "generation_fingerprint", lambda ref="HEAD": "same")
    (phase / "sample/generation.json").write_text("{}")
    with pytest.raises(AcceptanceError, match="reuse_artifact_changed"):
        reuse.verify_origin(root, source, manifest)


def test_coverage_ids_are_explicit_enum():
    from tests.evals.resume_initial_scoring import COVERAGE_PROMPT

    packets, mapping = packets_for(*fixture())
    packet = packets[-1]
    assert packet["output_schema"]["$defs"]["RequirementReview"]["properties"]["requirement"][
        "enum"
    ] == ["R0"]
    assert "NEVER its quote" in COVERAGE_PROMPT
    value = synthetic_response(packet)
    value["coverage"][0]["requirement"] = packet["items"][0]["quote"]
    with pytest.raises(AcceptanceError, match="coverage_denominator"):
        validate_batch(value, packet, mapping)


async def test_reuse_only_identical_successful_claim_batches(tmp_path):
    from tests.evals.product_acceptance_contracts import publish
    from tests.evals.resume_initial_scoring import PROMPT_DIGEST

    calls = []

    async def call(path, payload, stage):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        result = ChatModelResult(content=json.dumps(synthetic_response(payload)))
        publish(
            path / "call-00-response.json",
            {"response": result.model_dump(mode="json"), "invocation_ids": ["synthetic"]},
        )
        calls.append(payload["kind"])
        return result

    values = fixture()
    old = tmp_path / "old"
    first, _ = await assess(*values, old, call)
    before = len(calls)
    new, outcomes = await assess(*values, tmp_path / "new", call, reuse_path=old)
    assert first == new
    assert calls[before:] == []
    assert all(r["reused"] for r in outcomes)
    # A different prompt cannot reuse paid assessments even if the result parses.
    from unittest.mock import patch

    from tests.evals import resume_initial_scoring as scoring

    original = scoring.read_private_json

    def changed(path):
        value = original(path)
        if path == old / "protocol.json":
            value = {**value, "prompt": PROMPT_DIGEST + "changed"}
        return value

    with patch.object(scoring, "read_private_json", changed):
        with pytest.raises(AcceptanceError, match="reuse_score_prompt_changed"):
            await assess(*values, tmp_path / "changed", call, reuse_path=old)


def test_conflicting_duplicate_spans_fail_before_stage_acceptance():
    packets, mapping = packets_for(*fixture())
    value = synthetic_response(packets[0])
    claim = value["units"][0]["claims"][0]
    value["units"][0]["claims"].append({**claim, "support": "unsupported"})
    with pytest.raises(AcceptanceError, match="conflicting_claim_review"):
        validate_batch(value, packets[0], mapping)


async def test_cross_batch_conflict_gets_one_independent_review(tmp_path):
    from tests.evals.resume_initial_scoring import reconcile_duplicates

    values = fixture()
    packets, mapping = packets_for(*values)
    claims = validate_batch(synthetic_response(packets[0]), packets[0], mapping)["claims"]
    conflicting = [*claims, {**claims[0], "support": "unsupported", "rationale": "Different"}]
    calls = []

    async def call(path, packet, stage):
        calls.append(stage)
        assert stage == "review" and "previous_response" not in packet
        assert "Prior verdicts are withheld" in packet["review_instruction"]
        return ChatModelResult(content=json.dumps(synthetic_response(packet)))

    resolved, outcomes = await reconcile_duplicates(conflicting, packets, mapping, tmp_path, call)
    assert resolved is not None and calls == ["review"]
    assert outcomes[0]["status"] == "ASSESSED"
    assert await reconcile_duplicates(conflicting, packets, mapping, tmp_path, call) == (
        resolved,
        outcomes,
    )
    assert calls == ["review"]
