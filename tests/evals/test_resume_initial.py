"""A synthetic contract coverage; no private inputs or paid calls."""

import json
from dataclasses import replace
from uuid import uuid4

import pytest

from app.agents.resume_generation import ResumeGenerationGraph
from app.domain.errors import DomainValidationError
from app.llm.ports import ChatMessage, ChatModelResult
from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.resume_initial_baselines import (
    assemble_selection,
    candidate_text,
    generate,
    shared_input,
)
from tests.evals.resume_initial_contracts import ARMS, samples, summarize
from tests.evals.resume_initial_recording import JournalModel
from tests.unit.agents.test_resume_generation import _inputs, _requirements, _selection


class Model:
    model = "synthetic"

    def __init__(self, responses):
        self.responses, self.calls = responses, []

    async def invoke(self, messages, tools, metadata):
        self.calls.append((messages, tools, metadata))
        return self.responses[len(self.calls) - 1]


def proposal(inputs):
    p = inputs.profile_content.projects[0]
    return {
        "projects": [
            {
                "id": str(p.id),
                "summary": inputs.facts[0].claim,
                "technologies": [],
                "bullets": [inputs.facts[0].claim],
            }
        ]
    }


def selection(project, fact):
    return {
        "analysis": json.loads(_requirements().content),
        "selection": json.loads(_selection(project, fact).content),
    }


def test_fixed_schedule_and_c_start():
    planned = samples([f"job{i}" for i in range(20)])
    assert len(planned) == len({s["sample_id"] for s in planned}) == 180
    assert sum(s["c_start"] for s in planned) == 40
    for arm in ARMS:
        assert sum(s["arm"] == arm for s in planned) == 60
        for job in range(20):
            positions = [
                i % 3
                for i, s in enumerate(planned)
                if s["arm"] == arm and s["case_id"] == f"job{job}"
            ]
            assert set(positions) == {0, 1, 2}
    assert len(samples(["a", "b", "c"], pilot=True)) == 9


@pytest.mark.parametrize("arm", ARMS[:2])
async def test_baselines_single_call_privacy_and_shared_facts(arm):
    raw, identity, inputs, project, fact = _inputs()
    value = proposal(inputs) if arm == "one_shot" else selection(project, fact)
    model = Model([ChatModelResult(content="```json\n" + json.dumps(value) + "\n```")])
    output = await generate(arm, model, inputs, raw, identity)
    assert len(model.calls) == 1 and output["automatic_repairs"] == 0
    assert b"canary@example.test" in output["tex"].encode()
    assert "canary@example.test" not in candidate_text(output["content"])
    payload = json.loads(model.calls[0][0][1].content)
    assert "canary@example.test" not in json.dumps(payload)
    assert "annotations" not in payload and "requirements" not in payload
    assert payload["facts"] == shared_input(inputs, identity.source_sha256)["facts"]


@pytest.mark.parametrize("arm", ARMS[:2])
async def test_malformed_baseline_is_not_repaired(arm):
    raw, identity, inputs, _, _ = _inputs()
    model = Model([ChatModelResult(content='{"bad": true}')])
    with pytest.raises(ValueError):
        await generate(arm, model, inputs, raw, identity)
    assert len(model.calls) == 1


@pytest.mark.parametrize("mutation", ["quote", "foreign_fact", "index", "plan", "duplicate"])
def test_selection_matches_system_validation(mutation):
    _, _, inputs, project, fact = _inputs()
    value = selection(project, fact)
    if mutation == "quote":
        value["analysis"]["requirements"][0]["quote"] = "invented"
        with pytest.raises(DomainValidationError):
            assemble_selection(value, inputs)
        return
    if mutation == "foreign_fact":
        value["selection"]["bullets"][0]["fact_version_id"] = str(uuid4())
    if mutation == "index":
        value["selection"]["bullets"][0]["requirement_ordinals"] = []
    if mutation == "plan":
        inputs = replace(inputs, facts=(replace(inputs.facts[0], kind="plan"),))
    if mutation == "duplicate":
        value["selection"]["bullets"] *= 2
    content, diagnostics = assemble_selection(value, inputs)
    assert bool(content) == (mutation == "duplicate")
    if content:
        assert len(content.projects[0].bullets) == 1 and diagnostics["removed_duplicates"] == 1


async def test_system_repair_cap_remains_shared():
    _, _, inputs, project, _fact = _inputs()
    invalid = _selection(project, uuid4())
    model = Model([ChatModelResult(content="bad"), _requirements(), invalid])
    repairs = []

    async def allowed():
        return True

    async def repair():
        repairs.append(True)
        return True

    result = await ResumeGenerationGraph(
        model=model, spend_allowed=allowed, reserve_repair=repair
    ).generate(inputs)
    assert len(model.calls) == 3 and len(repairs) == result.correction_count == 1
    assert result.content is None


async def test_locked_baseline_rejected():
    raw, identity, inputs, project, _ = _inputs()
    inputs = replace(
        inputs, preferences=inputs.preferences.model_copy(update={"locked_item_ids": (project,)})
    )
    with pytest.raises(DomainValidationError):
        await generate(
            "one_shot",
            Model([ChatModelResult(content=json.dumps(proposal(inputs)))]),
            inputs,
            raw,
            identity,
        )


def test_no_output_zero_support_and_unresolved_keep_denominators():
    planned = samples(["job"])
    annotations = {"job": {"requirements": [{"applicable": True, "support": "unsupported"}]}}
    results = {
        s["sample_id"]: {
            "generation_status": "FAILED",
            "metrics": None,
            "score_status": "NOT_APPLICABLE",
        }
        for s in planned
    }
    summary = summarize(planned, results, annotations)
    assert summary["status"] == "PASS" and summary["zero_fully_supported_jds"] == ["job"]
    for group in summary["groups"].values():
        assert group["usable"]["denominator"] == 3 and group["usable"]["numerator"] == 0
        assert group["cross_jd"]["coverage"]["mean"] == 0
        assert group["cross_jd"]["fact_support"]["mean"] is None
    results[planned[0]["sample_id"]] = {
        "generation_status": "COMPLETE",
        "score_status": "UNRESOLVED",
    }
    assert summarize(planned, results, annotations)["status"] == "PARTIAL"
    assert summarize(planned, {}, annotations)["missing"] == 9


async def test_journal_replays_without_new_call_and_refuses_uncertain(tmp_path):
    tmp_path.chmod(0o700)

    class Recorder:
        async def check_admission(self, **kwargs):
            return {"invocation_ids": []}

    recorder = Recorder()
    model = Model([ChatModelResult(content="saved")])
    args = ((ChatMessage(role="user", content="safe"),), (), {"graph_node": "test"})
    assert (await JournalModel(model, tmp_path, recorder).invoke(*args)).content == "saved"
    assert (await JournalModel(model, tmp_path, recorder).invoke(*args)).content == "saved"
    assert len(model.calls) == 1
    with pytest.raises(AcceptanceError, match="identity"):
        await JournalModel(model, tmp_path, recorder).invoke(
            (ChatMessage(role="user", content="changed"),), (), {}
        )
    uncertain = tmp_path / "uncertain"
    uncertain.mkdir(mode=0o700)
    publish(uncertain / "call-00-started.json", {})
    with pytest.raises(AcceptanceError, match="uncertain"):
        await JournalModel(model, uncertain, recorder).invoke(*args)
    assert len(model.calls) == 1
