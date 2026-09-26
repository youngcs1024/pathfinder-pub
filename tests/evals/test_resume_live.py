"""Opt-in and evidence binding regressions; no provider or Docker access."""

import json
from uuid import uuid4

import pytest

from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.quality_experiment_binding import file_inventory
from tests.evals.resume_live import begin_stage, main, verify_inputs
from tests.evals.resume_live_contracts import LiveInputs, require_review
from tests.evals.resume_live_environment import ResumableDatabase


def config(root):
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    (inputs / "resume.tex").write_text("synthetic private original")
    (inputs / "code.txt").write_text("synthetic allowed source")
    return {
        "allocation_id": str(uuid4()),
        "execution_root": str(root),
        "materials_may_leave_machine": True,
        "budget": {},
        "projects": [
            {
                "alias": "project",
                "title": "Project",
                "profile_project_title": "Project",
                "files": ["code.txt"],
                "provenance": [{"commit": "a" * 40, "path": "src/code.py", "start": 1, "end": 2}],
            }
        ],
        "cases": [
            {
                "case_id": f"case{i}",
                "relationship": "close" if i < 2 else "different",
                "jd": "Python",
                "source_url": "https://example.com/job",
                "retrieved_at": "2026-09-25",
            }
            for i in range(3)
        ],
        "files": file_inventory(inputs, ["resume.tex", "code.txt"]),
    }


def parse(value):
    return LiveInputs.model_validate_json(json.dumps(value))


@pytest.mark.parametrize(
    "change", ["permission", "root", "budget", "attempts", "scope", "path", "model"]
)
def test_live_config_rejects_unsafe_scope(tmp_path, change):
    value = config(tmp_path)
    if change == "permission":
        value["materials_may_leave_machine"] = False
    elif change == "root":
        value["execution_root"] = "relative"
    elif change == "budget":
        value["budget"] = {"cost_admission_budget_cny": "21"}
    elif change == "attempts":
        value["budget"] = {"provider_attempt_cap": 101}
    elif change == "scope":
        value["files"]["credentials.env"] = "sha256:" + "0" * 64
    elif change == "path":
        value["resume_file"] = "../resume.tex"
        value["files"]["../resume.tex"] = value["files"].pop("resume.tex")
    else:
        value["chat_model"] = "unapproved"
    with pytest.raises(ValueError):
        parse(value)


def test_inputs_and_allocation_root_cannot_change(tmp_path):
    value = parse(config(tmp_path))
    verify_inputs(tmp_path, value)
    with pytest.raises(AcceptanceError):
        verify_inputs(tmp_path / "new-budget", value)
    (tmp_path / "inputs/code.txt").write_text("changed")
    with pytest.raises(AcceptanceError):
        verify_inputs(tmp_path, value)


def test_review_is_delegated_and_bound_to_exact_authorization():
    value = {
        "reviewer": "codex_agent_delegated",
        "authorization_digest": "fixed",
        "kind": "facts",
        "rationale": "source checked",
    }
    require_review(value, binding="fixed", kind="facts")
    for patch in (
        {"reviewer": "human"},
        {"authorization_digest": "other"},
        {"kind": "final"},
        {"rationale": ""},
    ):
        with pytest.raises(AcceptanceError):
            require_review({**value, **patch}, binding="fixed", kind="facts")


def test_stage_replay_and_failure_preservation(tmp_path):
    tmp_path.chmod(0o700)
    assert begin_stage(tmp_path, "case0-draft", "fixed") is None
    before = (tmp_path / "case0-draft-started.json").read_bytes()
    with pytest.raises(AcceptanceError, match="interrupted_stage"):
        begin_stage(tmp_path, "case0-draft", "fixed")
    assert (tmp_path / "case0-draft-started.json").read_bytes() == before
    publish(
        tmp_path / "case0-draft-done.json",
        {"authorization_digest": "fixed", "b1_status": "PARSE_FAILED"},
    )
    assert begin_stage(tmp_path, "case0-draft", "fixed")["b1_status"] == "PARSE_FAILED"
    with pytest.raises(AcceptanceError):
        begin_stage(tmp_path, "case0-draft", "other")


async def test_missing_database_receipt_cannot_reset_allocation(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    publish(tmp_path / "database-started.json", {"image_id": "fixed"})
    monkeypatch.setattr("tests.evals.resume_live_environment.local_docker", lambda: None)
    with pytest.raises(AcceptanceError, match="database_receipt_missing"):
        await ResumableDatabase(tmp_path, "fixed").__aenter__()


def test_default_execute_never_calls_provider(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    monkeypatch.setattr(
        "tests.evals.resume_live.execute", lambda *args: pytest.fail("must not execute")
    )
    assert main(["execute", "--root", str(tmp_path), "--stage", "materials"]) == 1


def test_frozen_applicable_denominator_and_human_time():
    from types import SimpleNamespace

    from tests.evals.resume_live_contracts import assessed_coverage, validate_rubric

    case = SimpleNamespace(jd="Python backend")
    rows = [
        {
            "requirement_id": "python",
            "quote": "Python",
            "fact_version_ids": ["f1"],
            "rationale": "source implements Python",
        }
    ]
    assert validate_rubric({"requirements": rows}, case=case, confirmed_ids={"f1"}) == rows
    with pytest.raises(AcceptanceError):
        validate_rubric({"requirements": rows}, case=case, confirmed_ids={"other"})
    quality = {
        "human_minutes": None,
        "rationale": "partial implementation",
        "fully_covered_ids": [],
        "partially_covered_ids": ["python"],
    }
    result = assessed_coverage(rows, quality)
    assert result["ratio"] == 0 and result["partially_covered_ids"] == ["python"]
    empty = {**quality, "partially_covered_ids": []}
    assert assessed_coverage([], empty)["status"] == "NOT_APPLICABLE"
    with pytest.raises(AcceptanceError):
        assessed_coverage(rows, {**quality, "human_minutes": 0})


async def test_live_b1_has_one_call_redacts_identity_and_fills_source():
    from app.llm.ports import ChatModelResult
    from tests.evals.resume_live_baseline import one_shot_live
    from tests.unit.agents.test_resume_generation import _inputs

    source, identity, inputs, _, _ = _inputs()
    profile = inputs.profile_content
    calls = []

    class Model:
        async def invoke(self, messages, tools, metadata):
            calls.append(messages)
            payload = json.loads(messages[1].content)
            assert "display_name" not in payload["profile"]
            assert "contact" not in payload["profile"]
            assert "source" not in payload["profile"]["projects"][0]
            return ChatModelResult(
                content=json.dumps(
                    {
                        "projects": [
                            {
                                "id": str(profile.projects[0].id),
                                "summary": "Supported project",
                                "technologies": [],
                                "bullets": ["Supported implementation"],
                            }
                        ],
                        "education_ids": [str(i.id) for i in profile.education],
                        "skill_ids": [str(i.id) for i in profile.skills],
                    }
                )
            )

    result = await one_shot_live(Model(), inputs, source, identity)
    assert len(calls) == result["logical_generations"] == 1
    assert result["automatic_repairs"] == 0
    assert result["content"]["contact"] == profile.model_dump(mode="json")["contact"]
    assert result["content"]["projects"][0]["source"] == profile.projects[0].source.model_dump(
        mode="json"
    )


async def test_live_b1_parse_failure_retains_body_without_second_call():
    from app.llm.ports import ChatModelResult
    from tests.evals.resume_live_baseline import one_shot_live
    from tests.evals.resume_quality_baseline import BaselineOutputError
    from tests.unit.agents.test_resume_generation import _inputs

    source, identity, inputs, _, _ = _inputs()
    calls = []

    class Model:
        async def invoke(self, *args):
            calls.append(1)
            return ChatModelResult(content="invalid raw output")

    with pytest.raises(BaselineOutputError) as exc:
        await one_shot_live(Model(), inputs, source, identity)
    assert exc.value.private_output["content"] == "invalid raw output"
    assert calls == [1]


def test_business_keys_are_persisted_v4_and_not_reallocated(tmp_path):
    from types import SimpleNamespace

    from app.db.material import MaterialProjectCommandV1
    from app.domain.resume_commands import ResumeCommandRequest
    from tests.evals.resume_live_runtime import key

    tmp_path.chmod(0o700)
    rig = SimpleNamespace(root=tmp_path, inputs=SimpleNamespace(digest="fixed"))
    first = key(rig, "project-a")
    assert first.version == 4
    assert key(rig, "project-a") == first
    assert key(rig, "project-b") != first
    ResumeCommandRequest(
        client_request_id=first,
        kind="material_project_create",
        target_id=None,
        payload_version=1,
        payload=MaterialProjectCommandV1(name="Project"),
    )
    rig.inputs.digest = "changed"
    with pytest.raises(AcceptanceError):
        key(rig, "project-a")


def test_source_rebind_preserves_old_manifest_and_requires_chain(tmp_path):
    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_live import effective_source

    tmp_path.chmod(0o700)
    old = {"source_sha": "old", "files": {}}
    manifest = {**old, "authorization_digest": "fixed"}
    assert effective_source(tmp_path, manifest) == old
    review = {
        "reviewer": "codex_agent_delegated",
        "authorization_digest": "fixed",
        "kind": "source_rebind",
        "rationale": "bounded fix",
    }
    new = {"source_sha": "new", "files": {"code.py": "digest"}}
    publish(
        tmp_path / "source-rebind-001.json",
        {"previous_digest": quality_identity_digest(old), "source": new, "review": review},
    )
    assert effective_source(tmp_path, manifest) == new
    assert manifest["source_sha"] == "old"
    publish(
        tmp_path / "source-rebind-002.json",
        {"previous_digest": "wrong", "source": new, "review": review},
    )
    with pytest.raises(AcceptanceError):
        effective_source(tmp_path, manifest)


async def test_material_prompt_is_bounded_and_failed_response_is_retained(tmp_path):
    from app.agents.material_facts import SYSTEM_PROMPT
    from app.llm.ports import ChatMessage, ChatModelResult
    from tests.evals.resume_live_material import LIMITS, MATERIAL_PROMPT_VERSION, LiveMaterialModel

    tmp_path.chmod(0o700)
    calls = []

    class Model:
        async def invoke(self, messages, tools, metadata):
            calls.append((messages, metadata))
            return ChatModelResult(content='{"facts":', finish_status="incomplete")

    value = await LiveMaterialModel(Model(), tmp_path).invoke(
        (
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content="source"),
        ),
        (),
        {"prompt_version": "old"},
    )
    assert value.content == '{"facts":'
    assert len(calls) == 1
    assert calls[0][0][0].content.endswith(LIMITS)
    assert calls[0][1]["prompt_version"] == MATERIAL_PROMPT_VERSION
    paths = list(tmp_path.glob("material-response-*.json"))
    assert len(paths) == 1 and json.loads(paths[0].read_text())["finish_status"] == "incomplete"


async def test_delegated_fact_correction_uses_versioned_business_commands(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.evals.quality_dataset import quality_identity_digest
    from tests.evals.resume_live_runtime import facts_stage

    tmp_path.chmod(0o700)
    fact_id, project_id, import_id, file_id = (str(uuid4()) for _ in range(4))
    catalog = {"import_id": import_id, "facts": [{"id": fact_id, "version": 1}]}
    digest = quality_identity_digest(catalog)
    publish(tmp_path / "candidates.json", catalog)
    publish(
        tmp_path / "materials-done.json",
        {
            "projects": {
                "project": {
                    "project_id": project_id,
                    "catalog_digest": digest,
                    "catalog_file": "candidates.json",
                }
            }
        },
    )
    candidate = {
        "claim": "Conditionally enabled",
        "kind": "implementation",
        "evidence": [
            {"snapshot_file_id": file_id, "start_line": 1, "end_line": 1, "quote": "if enabled"}
        ],
    }
    publish(
        tmp_path / "fact-review.json",
        {
            "reviewer": "codex_agent_delegated",
            "authorization_digest": "fixed",
            "kind": "facts",
            "rationale": "checked original source",
            "projects": {
                "project": {
                    "catalog_digest": digest,
                    "decisions": {
                        fact_id: {
                            "decision": "confirm",
                            "rationale": "correct unconditional wording",
                            "candidate": candidate,
                        }
                    },
                }
            },
        },
    )
    calls = []

    class Store:
        async def command(self, tenant, **kwargs):
            calls.append(kwargs)

        async def current_facts(self, tenant, project_id):
            return {"facts": [{"review_status": "confirmed", "version_id": str(uuid4())}]}

    monkeypatch.setattr(
        "tests.evals.resume_live_runtime.SqlAlchemyProjectFactStore", lambda _: Store()
    )
    rig = SimpleNamespace(
        root=tmp_path, inputs=SimpleNamespace(digest="fixed"), sessions=None, tenant=None
    )
    result = await facts_stage(rig)
    assert result["projects"]["project"]["fact_version_ids"]
    assert [c["kind"] for c in calls] == ["material_fact_revise", "material_fact_review"]
    assert [c["expected_version"] for c in calls] == [1, 2]
    assert calls[-1]["attested"] is False
