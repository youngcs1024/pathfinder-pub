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
