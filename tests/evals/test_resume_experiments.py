"""Public synthetic preparation contracts; no private files or live providers."""

import json
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.domain.errors import DomainValidationError
from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.resume_experiment_budget import CHAT_HEADROOM_CNY, admit
from tests.evals.resume_experiment_contracts import (
    ExperimentBudget,
    Inputs,
    checked_annotation,
    locate_annotation,
    normalize_json,
)
from tests.evals.resume_experiment_runtime import invoke_stage, smoke_inputs
from tests.evals.resume_experiment_scoring import (
    PROMPT_DIGESTS,
    aggregate,
    blind_packets,
    checked_assessment,
    generation_payload,
    render_structured,
    score_metrics,
)
from tests.evals.resume_experiments import main, preserve
from tests.unit.agents.test_resume_generation import _inputs


def input_dict(tmp_path):
    cases = []
    for i, category in enumerate(["backend"] * 8 + ["agent"] * 8 + ["retrieval_data"] * 4):
        cases.append(
            dict(
                case_id=f"job_{i}",
                category=category,
                official_job_id=f"new_{i}",
                source_url=f"https://official.example/jobs/{i}",
                retrieved_at="2026-09-27",
                title=f"Synthetic {i}",
                jd=f"Unique JD {i}",
                raw_file=f"raw_{i}.html",
                body_file=f"body_{i}.txt",
            )
        )
    names = {"facts.json", "profile.json", "resume.tex"}
    names.update(c["raw_file"] for c in cases)
    names.update(c["body_file"] for c in cases)
    return dict(
        allocation_id=str(uuid4()),
        execution_root=str(tmp_path),
        source_material_identity="sha256:" + "a" * 64,
        files=dict.fromkeys(names, "sha256:" + "b" * 64),
        cases=cases,
        pilot_job_ids=["old_1", "old_2", "old_3"],
        approved_by="user_explicit_experiment_preparation",
        materials_may_leave_machine=True,
    )


@pytest.mark.parametrize(
    "mutation", ["valid", "quota", "identity", "body", "pilot", "path", "budget"]
)
def test_formal_manifest_scope(tmp_path, mutation):
    data = input_dict(tmp_path)
    if mutation == "quota":
        data["cases"][0]["category"] = "agent"
    elif mutation == "identity":
        data["cases"][1]["official_job_id"] = data["cases"][0]["official_job_id"]
    elif mutation == "body":
        data["cases"][1]["jd"] = data["cases"][0]["jd"] + " \n"
    elif mutation == "pilot":
        data["cases"][0]["official_job_id"] = "old_1"
    elif mutation == "path":
        data["files"]["../escape"] = "sha256:" + "c" * 64
    elif mutation == "budget":
        data["budget"] = {"attempt_cap": 3001}
    if mutation == "valid":
        inputs = Inputs.model_validate_json(json.dumps(data))
        assert inputs.budget.attempt_cap == 3000 and len(inputs.cases) == 20
        assert inputs.digest == Inputs.model_validate_json(json.dumps(data)).digest
    else:
        with pytest.raises(ValueError):
            Inputs.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("raw", ['{"a":1}', ' \n```json\n{"a":1}\n```\n'])
def test_json_normalization_has_no_semantic_repair(raw):
    assert normalize_json(raw) == {"a": 1}


@pytest.mark.parametrize("raw", ['prefix {"a":1}', '{"a":1,"a":2}', '{"a":NaN}', "[]", '{"a":}'])
def test_json_normalization_rejects_invalid(raw):
    with pytest.raises((ValueError, AcceptanceError)):
        normalize_json(raw)


def test_annotation_positions_and_unknown_facts():
    facts, annotation, _, _ = smoke_inputs()
    case = SimpleNamespace(jd="要求:实现分页查询。")
    value = annotation.model_dump(mode="json")
    value["requirements"][0]["start"] = 0
    with pytest.raises(AcceptanceError, match="quote"):
        checked_annotation(value, case, facts)
    checked = checked_annotation(locate_annotation(value, case), case, facts)
    assert checked == annotation
    value["requirements"][0]["quote"] = "不存在"
    with pytest.raises(AcceptanceError):
        checked_annotation(locate_annotation(value, case), case, facts)
    value = annotation.model_dump(mode="json")
    value["requirements"][0]["fact_version_ids"] = ["foreign"]
    with pytest.raises(AcceptanceError, match="unknown_fact"):
        checked_annotation(value, case, facts)


def test_blinding_allowlists_and_determinism():
    facts, annotation, _, _ = smoke_inputs()
    candidates = [
        {
            "sample_id": str(i),
            "text": f"Body {i}",
            "arm": "HIDDEN_GROUP",
            "history": "HIDDEN_HISTORY",
            "cost": "HIDDEN_COST",
        }
        for i in range(8)
    ]
    kwargs = dict(case=SimpleNamespace(jd="JD"), facts=facts, annotation=annotation, seed=47)
    packets, mapping = blind_packets(candidates, **kwargs)
    assert (packets, mapping) == blind_packets(list(reversed(candidates)), **kwargs)
    assert "HIDDEN" not in json.dumps(packets)
    assert list(mapping.values()) != [str(i) for i in range(8)]
    _, _, inputs, _, _ = _inputs()
    value = generation_payload(
        SimpleNamespace(jd="JD"), facts, inputs.profile_content, inputs.preferences
    )
    assert set(value) == {"jd", "facts", "profile", "preferences"}
    assert "canary@example.test" not in json.dumps(value)


def assessment_dict():
    return {
        "claims": [
            {
                "quote": "实现分页查询",
                "support": "full",
                "fact_version_ids": ["synthetic_fact"],
                "experimental": False,
                "conditions_complete": True,
                "rationale": "explicit code",
            }
        ],
        "coverage": [{"requirement_id": "pagination", "status": "full", "rationale": "covered"}],
    }


def test_scoring_dedup_denominators_and_unusable():
    facts, annotation, _, _ = smoke_inputs()
    value = assessment_dict()
    value["claims"] *= 2
    checked = checked_assessment(
        value, text="实现分页查询;实现分页查询", facts=facts, annotation=annotation
    )
    metrics = score_metrics(checked, content_ok=True)
    assert metrics["fact_support"]["denominator"] == 1
    assert metrics["usable"] and metrics["condition_omission"]["value"] is None
    assert not score_metrics(checked, content_ok=False)["usable"]
    results = {"ok": {"status": "COMPLETE", "metrics": metrics}, "bad": {"status": "FAILED"}}
    summary = aggregate(["ok", "bad", "missing"], results)
    assert summary["usable"]["denominator"] == 3 and summary["usable"]["numerator"] == 1
    assert summary["failed_or_blocked"] == summary["missing"] == 1
    value["coverage"] = []
    with pytest.raises(AcceptanceError, match="denominator"):
        checked_assessment(value, text="实现分页查询", facts=facts, annotation=annotation)


def test_shared_renderer_rejects_personal_change():
    source, identity, inputs, _, _ = _inputs()
    raw = json.dumps(inputs.profile_content.model_dump(mode="json"))
    kwargs = dict(
        profile=inputs.profile_content,
        preferences=inputs.preferences,
        source_bytes=source,
        identity=identity,
    )
    _, first = render_structured(raw, **kwargs)
    _, second = render_structured(f"```json\n{raw}\n```", **kwargs)
    assert first.tex_bytes == second.tex_bytes
    value = json.loads(raw)
    value["display_name"] = "changed"
    with pytest.raises(DomainValidationError):
        render_structured(json.dumps(value), **kwargs)


def usage():
    return dict(unfinished=0, unknown_usage=0, unknown_cost=0, known_cost_cny="0", attempts=0)


@pytest.mark.parametrize(
    "problem", ["unfinished", "unknown_usage", "unknown_cost", "attempts", "cost"]
)
def test_strict_budget_stop(problem):
    value = usage()
    if problem == "attempts":
        value["attempts"] = 3000
    elif problem == "cost":
        value["known_cost_cny"] = str(Decimal("100") - CHAT_HEADROOM_CNY + Decimal("0.01"))
    else:
        value[problem] = 1
    with pytest.raises(AcceptanceError):
        admit(value, ExperimentBudget())
    if problem == "attempts":
        admit(value, ExperimentBudget(), after=True)


def test_budget_does_not_inherit_token_caps():
    value = {**usage(), "input_tokens": 99999999, "output_tokens": 99999999}
    admit(value, ExperimentBudget())


async def test_replay_and_interrupted_stage_never_call_provider(tmp_path):
    tmp_path.chmod(0o700)
    binding = "binding"
    payload = {"safe": True}
    from tests.evals.quality_dataset import quality_identity_digest

    class Model:
        async def invoke(self, *args):
            pytest.fail("completed or interrupted stage must not call provider")

    class Recorder:
        async def check_admission(self, **kwargs):
            return {"invocation_ids": []}

    publish(
        tmp_path / "done-response.json",
        dict(
            binding=binding,
            payload_digest=quality_identity_digest(payload),
            prompt_digest=PROMPT_DIGESTS["annotate"],
            content='{"ok":true}',
            finish_status="completed",
            tool_calls=False,
        ),
    )
    kwargs = dict(
        task="annotate", payload=payload, model=Model(), recorder=Recorder(), binding=binding
    )
    assert await invoke_stage(tmp_path, stage="done", **kwargs) == {"ok": True}
    publish(tmp_path / "interrupted-started.json", {"binding": binding})
    with pytest.raises(AcceptanceError, match="reconciliation"):
        await invoke_stage(tmp_path, stage="interrupted", **kwargs)
    preserve(tmp_path / "old-failure.json", {"status": "FAILED"})
    with pytest.raises(AcceptanceError, match="changed"):
        preserve(tmp_path / "old-failure.json", {"status": "COMPLETE"})


def test_future_experiment_entrypoints_are_absent(tmp_path):
    with pytest.raises(SystemExit):
        main(["run-d", "--root", str(tmp_path)])


def test_private_input_tampering_is_rejected_before_provider(tmp_path):
    from tests.evals.quality_experiment_binding import file_inventory
    from tests.evals.resume_experiments import load_inputs

    tmp_path.chmod(0o700)
    material = tmp_path / "inputs"
    material.mkdir(mode=0o700)
    source, _, original, _, _ = _inputs()
    data = input_dict(tmp_path)
    for case in data["cases"]:
        for name, body in (
            (case["raw_file"], "<html>synthetic</html>"),
            (case["body_file"], case["jd"]),
        ):
            path = material / name
            path.write_text(body)
            path.chmod(0o600)
    (material / "resume.tex").write_bytes(source)
    (material / "resume.tex").chmod(0o600)
    (material / "evidence.txt").write_text("verified fact\n")
    (material / "evidence.txt").chmod(0o600)
    publish(
        material / "facts.json",
        {
            "facts": [
                {
                    "version_id": "fact_version",
                    "review_status": "confirmed",
                    "claim": "verified fact",
                    "evidence": [
                        {
                            "path": "evidence.txt",
                            "start_line": 1,
                            "end_line": 1,
                            "quote": "verified fact",
                        }
                    ],
                }
            ]
        },
    )
    publish(
        material / "profile.json", {"content": original.profile_content.model_dump(mode="json")}
    )
    data["files"] = file_inventory(material, [*data["files"], "evidence.txt"])
    publish(tmp_path / "inputs.json", data)
    assert load_inputs(tmp_path).digest
    (material / "evidence.txt").write_text("changed\n")
    with pytest.raises(AcceptanceError, match="inputs_changed"):
        load_inputs(tmp_path)


def test_agent_review_binds_original_annotation_and_every_requirement(tmp_path):
    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_experiment_runtime import validate_reviews

    tmp_path.chmod(0o700)
    facts, annotation, _, _ = smoke_inputs()
    case = SimpleNamespace(case_id="synthetic", jd="要求:实现分页查询。")
    inputs = SimpleNamespace(cases=(case,), digest="binding")
    value = annotation.model_dump(mode="json")
    publish(tmp_path / "annotation-synthetic.json", value)
    publish(
        tmp_path / "review-synthetic.json",
        {
            "binding": "binding",
            "annotation_digest": quality_identity_digest(value),
            "review_kind": "AGENT_ASSESSED",
            "approved": True,
            "rationale": "reviewed source",
            "reviewed_requirement_ids": ["pagination"],
        },
    )
    assert "synthetic" in validate_reviews(tmp_path, inputs, facts)
    inputs.digest = "changed"
    with pytest.raises(AcceptanceError, match="binding"):
        validate_reviews(tmp_path, inputs, facts)


def test_reviewed_profile_is_self_report_not_new_project_facts():
    from tests.evals.resume_experiment_scoring import annotation_payload, profile_evidence

    _, _, original, _, _ = _inputs()
    snapshot = {
        "content": original.profile_content.model_dump(mode="json"),
        "version_id": str(uuid4()),
    }
    profile = profile_evidence(snapshot)
    assert profile["source_kind"] == "user_reported"
    assert "projects" not in profile and "contact" not in profile and "display_name" not in profile
    assert "canary@example.test" not in json.dumps(profile)
    facts, annotation, _, _ = smoke_inputs()
    case = SimpleNamespace(jd="要求:实现分页查询。")
    payload = annotation_payload(case, facts, profile)
    assert payload["facts"] == facts and payload["reviewed_profile"] == profile
    value = annotation.model_dump(mode="json")
    value["requirements"][0]["profile_item_ids"] = ["not_a_profile_item"]
    with pytest.raises(AcceptanceError, match="unknown_profile_item"):
        checked_annotation(value, case, facts, profile)
