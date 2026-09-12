import pytest

from scripts.ci_gate import HEAVY_JOBS, evaluate_ci_gate

BASELINE = {"baseline_sha": "1" * 40, "baseline_run_id": "164", "baseline_attempt": "1"}


def _results(heavy_result: str, *, preflight: str = "success") -> dict[str, str]:
    return {"preflight": preflight, **dict.fromkeys(HEAVY_JOBS, heavy_result)}


def test_full_ci_passes_only_when_every_required_job_succeeds() -> None:
    passed, reason = evaluate_ci_gate(run_full="true", results=_results("success"))

    assert passed is True
    assert reason == "full CI succeeded"


def test_docs_only_passes_only_when_every_heavy_job_is_skipped() -> None:
    passed, reason = evaluate_ci_gate(run_full="false", results=_results("skipped"), **BASELINE)

    assert passed is True
    assert "docs-only" in reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_sha", ""),
        ("baseline_sha", "0" * 40),
        ("baseline_sha", "SHA-CANARY\n"),
        ("baseline_run_id", ""),
        ("baseline_run_id", "0"),
        ("baseline_attempt", ""),
        ("baseline_attempt", "ATTEMPT-CANARY"),
    ],
)
def test_docs_gate_requires_sanitized_baseline_evidence(field, value):
    from scripts.ci_gate import job_summary

    evidence = {**BASELINE, field: value}
    passed, reason = evaluate_ci_gate(run_full="false", results=_results("skipped"), **evidence)
    summary = job_summary(_results("skipped"), run_full="false", **evidence)
    assert not passed
    assert "baseline" in reason and "- baseline:" in summary
    assert "CANARY" not in reason + summary


def test_docs_summary_identifies_inherited_attempt_without_claiming_new_business_execution():
    from scripts.ci_gate import job_summary

    summary = job_summary(_results("skipped"), run_full="false", **BASELINE)
    assert f"commit {BASELINE['baseline_sha']}, run 164, attempt 1" in summary
    assert "do not re-execute business tests" in summary


@pytest.mark.parametrize("run_full", ["", "TRUE", "invalid"])
def test_classifier_inconsistency_fails_closed(run_full: str) -> None:
    passed, _reason = evaluate_ci_gate(run_full=run_full, results=_results("skipped"))

    assert passed is False


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", "missing"])
@pytest.mark.parametrize("job", HEAVY_JOBS)
def test_full_ci_rejects_any_non_success_heavy_job(result: str, job: str) -> None:
    results = _results("success")
    if result == "missing":
        results.pop(job)
    else:
        results[job] = result

    passed, _reason = evaluate_ci_gate(run_full="true", results=results)

    assert passed is False


@pytest.mark.parametrize("result", ["success", "failure", "cancelled", "missing"])
def test_docs_only_rejects_any_unexpected_heavy_job_result(result: str) -> None:
    results = _results("skipped")
    results["integration"] = result

    passed, _reason = evaluate_ci_gate(run_full="false", results=results, **BASELINE)

    assert passed is False


@pytest.mark.parametrize("run_full", ["true", "false"])
@pytest.mark.parametrize("preflight", ["failure", "cancelled", "skipped", "missing"])
def test_preflight_must_succeed(run_full: str, preflight: str) -> None:
    heavy_result = "success" if run_full == "true" else "skipped"
    passed, _reason = evaluate_ci_gate(
        run_full=run_full,
        results=_results(heavy_result, preflight=preflight),
    )

    assert passed is False


@pytest.mark.parametrize("heavy_status", ["success", "failure", "cancelled", "skipped"])
def test_summary_records_status_without_changing_gate_result(tmp_path, monkeypatch, heavy_status):
    from scripts.ci_gate import main

    summary = tmp_path / "summary.md"
    summary.write_text("Existing summary\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["--run-full", "true", "--preflight", "success"]
    for job in HEAVY_JOBS:
        args.extend([f"--{job}", heavy_status])
    assert main(args) == (0 if heavy_status == "success" else 1)
    content = summary.read_text(encoding="utf-8")
    assert content.startswith("Existing summary\n")
    for job in HEAVY_JOBS:
        assert f"| {job} | {heavy_status} |" in content


def test_summary_drops_unknown_job_names_and_sanitizes_statuses():
    from scripts.ci_gate import job_summary

    content = job_summary({"preflight": "BODY-CANARY", "UNKNOWN-JOB-CANARY": "success"})
    assert "CANARY" not in content
    assert "| preflight | invalid |" in content
    for job in HEAVY_JOBS:
        assert f"| {job} | missing |" in content


def test_summary_write_failure_does_not_report_a_green_gate(tmp_path, monkeypatch, capsys):
    from scripts.ci_gate import main

    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing" / "CANARY"))
    args = ["--run-full", "true", "--preflight", "success"]
    for job in HEAVY_JOBS:
        args.extend([f"--{job}", "success"])
    assert main(args) == 1
    output = capsys.readouterr()
    assert "could not write CI job summary" in output.err
    assert "CANARY" not in output.err
    assert "PASS" not in output.out


def test_gate_sanitizes_unknown_statuses_in_all_outputs(tmp_path, monkeypatch, capsys):
    from scripts.ci_gate import main

    destination = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(destination))
    args = ["--run-full", "true", "--preflight", "success"]
    for job in HEAVY_JOBS:
        args.extend([f"--{job}", "STATUS-CANARY" if job == "quality" else "success"])
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "CANARY" not in captured.out + captured.err + destination.read_text()
    assert "invalid" in captured.out
    passed, reason = evaluate_ci_gate(
        run_full="true", results={"preflight": "success", "quality": "STATUS-CANARY"}
    )
    assert not passed and "CANARY" not in reason


@pytest.mark.parametrize("run_full", ["true", "false", "CANARY"])
def test_gate_overview_lists_all_inconsistencies_even_after_preflight_failure(run_full):
    from scripts.ci_gate import job_summary

    results = _results("failure", preflight="cancelled")
    results["integration"] = "STATUS-CANARY"
    content = job_summary(results, run_full=run_full)
    overview = content.split("| Job |", 1)[0]
    for job in ("preflight", *HEAVY_JOBS):
        assert f"- {job}:" in overview
    assert "CANARY" not in content
    if run_full == "CANARY":
        assert "CI path: invalid" in overview
        assert "- classifier:" in overview


@pytest.mark.parametrize(
    ("run_full", "status", "path"), [("true", "success", "full"), ("false", "skipped", "docs-only")]
)
def test_gate_cli_success_reports_selected_path(
    tmp_path, monkeypatch, capsys, run_full, status, path
):
    from scripts.ci_gate import main

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["--run-full", run_full, "--preflight", "success"]
    for name, value in BASELINE.items():
        args.extend(["--" + name.replace("_", "-"), value])
    for job in HEAVY_JOBS:
        args.extend([f"--{job}", status])
    assert main(args) == 0
    assert "ci-gate PASS" in capsys.readouterr().out
    content = summary.read_text()
    assert f"CI path: {path}" in content
    assert "All job statuses match" in content
    assert "- quality:" not in content


@pytest.mark.parametrize("run_full", ["true", "false", "CANARY"])
def test_gate_cli_failure_keeps_all_job_diagnostics(tmp_path, monkeypatch, capsys, run_full):
    from scripts.ci_gate import main

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["--run-full", run_full, "--preflight", "failure"]
    for job in HEAVY_JOBS:
        args.extend([f"--{job}", "STATUS-CANARY"])
    assert main(args) == 1
    output = capsys.readouterr()
    content = summary.read_text()
    assert "ci-gate FAIL" in output.out
    assert "CANARY" not in content + output.out + output.err
    for job in ("preflight", *HEAVY_JOBS):
        assert f"- {job}:" in content


@pytest.mark.parametrize("status", ["failure", "cancelled"])
def test_shared_precheck_failure_is_distinguished_from_downstream_blocking(status):
    from scripts.ci_gate import job_summary

    results = _results("success")
    results["python-validation"] = status
    for job in ("quality", "contracts", "integration"):
        results[job] = "skipped"
    summary = job_summary(results, run_full="true")
    assert f"- python-validation: {status}" in summary
    for job in ("quality", "contracts", "integration"):
        line = next(line for line in summary.splitlines() if line.startswith(f"- {job}:"))
        assert "blocked by prerequisite: python-validation" in line
    assert not evaluate_ci_gate(run_full="true", results=results)[0]


def test_unexpected_skip_is_not_explained_by_unrelated_job_failure():
    from scripts.ci_gate import job_summary

    results = _results("success")
    results.update(quality="failure", integration="skipped")
    line = next(
        line
        for line in job_summary(results, run_full="true").splitlines()
        if line.startswith("- integration:")
    )
    assert "expected success" in line and "blocked" not in line
