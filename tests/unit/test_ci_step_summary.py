import json

import pytest

from scripts.ci_step_summary import BRANCHES, STEP_IDS, main, step_summary

PROFILES = [(job, branch) for job in STEP_IDS for branch in sorted(BRANCHES.get(job, {""}))]


def _steps(job, branch):
    inactive = set()
    if job == "contracts":
        inactive = {"eval", "regression"} if branch == "operational" else {"operational"}
    elif job == "integration" and branch == "1":
        inactive = {"retrieval", "demo"}
    return {
        step: dict.fromkeys(("outcome", "conclusion"), "skipped" if step in inactive else "success")
        for step in STEP_IDS[job]
    }


@pytest.mark.parametrize(("job", "branch"), PROFILES)
def test_every_matrix_profile_reports_required_and_inapplicable_steps(job, branch):
    text, passed = step_summary(job, branch, json.dumps(_steps(job, branch)))
    assert passed
    for step, record in _steps(job, branch).items():
        status = record["outcome"]
        label = "not applicable" if status == "skipped" else "required"
        assert f"| {step} | {label} | {status} | {status} |" in text


@pytest.mark.parametrize(("job", "branch"), PROFILES)
@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None, "BODY-CANARY", []])
def test_required_failures_and_missing_or_invalid_results_never_pass(job, branch, status):
    steps = _steps(job, branch)
    steps["checkout"] = {"outcome": status, "conclusion": status}
    text, passed = step_summary(job, branch, json.dumps(steps))
    assert not passed
    assert "BODY-CANARY" not in text


def test_continued_failure_cannot_be_disguised_by_success_conclusion():
    steps = _steps("python-validation", "")
    steps["rules"] = {"outcome": "failure", "conclusion": "success"}
    text, passed = step_summary("python-validation", "", json.dumps(steps))
    assert not passed
    assert "| rules | required | failure | success |" in text


@pytest.mark.parametrize("record", [None, {}, [], "CANARY"])
def test_missing_or_malformed_step_is_visible_and_fails(record):
    steps = _steps("python-validation", "")
    steps["collect"] = record
    text, passed = step_summary("python-validation", "", json.dumps(steps))
    assert not passed
    assert "| collect | required | missing | missing |" in text
    assert "CANARY" not in text


def test_unexpected_matrix_execution_is_not_silently_accepted():
    steps = _steps("integration", "1")
    steps["demo"] = {"outcome": "success", "conclusion": "success"}
    _, passed = step_summary("integration", "1", json.dumps(steps))
    assert not passed


def test_setup_failure_keeps_skipped_checks_visible():
    steps = _steps("python-validation", "")
    for step in steps:
        status = "success" if step == "checkout" else "failure" if step == "setup" else "skipped"
        steps[step] = {"outcome": status, "conclusion": status}
    text, passed = step_summary("python-validation", "", json.dumps(steps))
    assert not passed
    assert "| collect | required | skipped | skipped |" in text


def _environment(monkeypatch, tmp_path, payload):
    destination = tmp_path / "summary.md"
    destination.write_text("Existing summary\n", encoding="utf-8")
    monkeypatch.setenv("PF_CI_JOB", "python-validation")
    monkeypatch.setenv("PF_CI_BRANCH", "")
    monkeypatch.setenv("PF_CI_STEPS", payload)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    return destination


def test_main_appends_only_allowlisted_fields_to_stdout_and_summary(tmp_path, monkeypatch, capsys):
    steps = _steps("python-validation", "")
    steps["collect"]["outputs"] = {"private": "OUTPUT-CANARY"}
    steps["UNKNOWN-CANARY"] = {"outcome": "success", "conclusion": "success"}
    destination = _environment(monkeypatch, tmp_path, json.dumps(steps))
    assert main() == 0
    captured = capsys.readouterr()
    content = destination.read_text()
    assert content.startswith("Existing summary\n")
    assert "CANARY" not in content + captured.out + captured.err
    assert "| collect | required | success | success |" in content


@pytest.mark.parametrize("payload", ["CANARY", "[]", "null", "{}", "x" * 262_145])
def test_invalid_input_returns_safe_nonzero(tmp_path, monkeypatch, capsys, payload):
    destination = _environment(monkeypatch, tmp_path, payload)
    assert main() == 1
    captured = capsys.readouterr()
    assert "CANARY" not in captured.out + captured.err + destination.read_text()


@pytest.mark.parametrize("field", ["PF_CI_JOB", "PF_CI_BRANCH", "GITHUB_STEP_SUMMARY"])
def test_bad_identity_or_write_failure_never_echoes_input(tmp_path, monkeypatch, capsys, field):
    _environment(monkeypatch, tmp_path, json.dumps(_steps("python-validation", "")))
    monkeypatch.setenv(field, str(tmp_path / "missing" / "CANARY"))
    assert main() == 1
    captured = capsys.readouterr()
    assert "CANARY" not in captured.out + captured.err
    assert "could not be produced" in captured.err


def test_missing_summary_destination_is_an_error(tmp_path, monkeypatch):
    _environment(monkeypatch, tmp_path, json.dumps(_steps("python-validation", "")))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    assert main() == 1


REQUIRED_CHECKS = [
    (job, branch, step)
    for job, branch in PROFILES
    for step, record in _steps(job, branch).items()
    if record["outcome"] == "success"
]
INACTIVE_CHECKS = [
    (job, branch, step)
    for job, branch in PROFILES
    for step, record in _steps(job, branch).items()
    if record["outcome"] == "skipped"
]


@pytest.mark.parametrize(("job", "branch", "step"), REQUIRED_CHECKS)
@pytest.mark.parametrize(
    ("outcome", "conclusion", "diagnostic"),
    [
        ("failure", "failure", "failed"),
        ("cancelled", "cancelled", "cancelled"),
        ("skipped", "skipped", "required check skipped"),
        (None, None, "missing status"),
        ("STATUS-CANARY", [], "invalid status"),
        ("failure", "success", "outcome/conclusion mismatch"),
        ("success", "failure", "outcome/conclusion mismatch"),
    ],
)
def test_each_required_check_has_safe_diagnostic_before_table(
    job, branch, step, outcome, conclusion, diagnostic
):
    steps = _steps(job, branch)
    steps[step] = {"outcome": outcome, "conclusion": conclusion}
    text, passed = step_summary(job, branch, json.dumps(steps))
    assert not passed
    overview = text.split("| Check |", 1)[0]
    assert f"- {step}:" in overview
    assert diagnostic in overview
    assert "CANARY" not in text


@pytest.mark.parametrize(("job", "branch", "step"), INACTIVE_CHECKS)
@pytest.mark.parametrize("status", ["success", "failure", "cancelled", None, "CANARY"])
def test_each_inactive_check_must_be_explicitly_skipped(job, branch, step, status):
    steps = _steps(job, branch)
    text, passed = step_summary(job, branch, json.dumps(steps))
    assert passed
    assert f"- {step}:" not in text
    steps[step] = dict.fromkeys(("outcome", "conclusion"), status)
    text, passed = step_summary(job, branch, json.dumps(steps))
    assert not passed
    assert f"- {step}:" in text
    if status in {"success", "failure", "cancelled"}:
        assert "non-applicable check executed" in text
    assert "CANARY" not in text


@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None, "CANARY"])
def test_cli_reports_all_problems_and_nonzero(tmp_path, monkeypatch, capsys, status):
    steps = _steps("python-validation", "")
    steps["rules"] = dict.fromkeys(("outcome", "conclusion"), status)
    steps["routing"] = {"outcome": "failure", "conclusion": "success"}
    destination = _environment(monkeypatch, tmp_path, json.dumps(steps))
    assert main() == 1
    output = capsys.readouterr()
    for content in (output.out, destination.read_text()):
        overview = content.split("| Check |", 1)[0]
        assert "- rules:" in overview and "- routing:" in overview
        assert "outcome/conclusion mismatch" in overview
        assert "CANARY" not in content


@pytest.mark.parametrize("step", ["setup_node", "ui"])
@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None])
def test_ui_runtime_and_behavior_checks_are_required(step, status):
    steps = _steps("quality", "")
    steps[step] = {"outcome": status, "conclusion": status}
    _, passed = step_summary("quality", "", json.dumps(steps))
    assert not passed


@pytest.mark.parametrize(
    ("job", "branch", "prerequisite", "dependent"),
    [
        ("python-validation", "", "collect", "routing"),
        ("quality", "", "setup_node", "ui"),
        ("image-security", "", "build", "smoke"),
        ("contracts", "eval", "setup", "eval"),
        ("integration", "0", "setup", "core"),
        ("preflight", "", "checkout", "links"),
    ],
)
def test_blocked_skips_name_only_the_actual_prerequisite(job, branch, prerequisite, dependent):
    steps = _steps(job, branch)
    steps[prerequisite] = dict.fromkeys(("outcome", "conclusion"), "failure")
    steps[dependent] = dict.fromkeys(("outcome", "conclusion"), "skipped")
    text, passed = step_summary(job, branch, json.dumps(steps))
    assert not passed
    line = next(line for line in text.splitlines() if line.startswith(f"- {dependent}:"))
    assert f"blocked by prerequisite: {prerequisite}" in line
    assert "unexpected skip" not in line


def test_independent_failure_does_not_explain_an_unexpected_skip():
    steps = _steps("python-validation", "")
    steps["rules"] = dict.fromkeys(("outcome", "conclusion"), "failure")
    steps["routing"] = dict.fromkeys(("outcome", "conclusion"), "skipped")
    text, passed = step_summary("python-validation", "", json.dumps(steps))
    assert not passed
    assert "- routing: required check skipped, unexpected skip: prerequisites succeeded" in text
    assert "blocked by prerequisite: rules" not in text


def test_cancelled_check_is_not_mislabeled_as_an_unexplained_skip():
    steps = _steps("python-validation", "")
    steps["routing"] = {"outcome": "cancelled", "conclusion": "skipped"}
    text, passed = step_summary("python-validation", "", json.dumps(steps))
    assert not passed
    assert "cancelled" in text
    assert "unexpected skip" not in text
