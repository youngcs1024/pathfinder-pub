"""Offline platform evidence fixtures, independent of runners and account billing."""

import copy
import json
import subprocess
import sys

import pytest

from scripts import ci_inspect_run as ci
from scripts.ci_contract import CHECKS, FULL_JOB_REQUIRED_STEPS, HEAVY_JOBS
from scripts.ci_evidence import full_attempt
from tests.unit.test_ci_baseline import _jobs, _run

REPO = "owner/pathfinder"
RUN_ID = 164
SHA = "1" * 40
CANARY = "private-ci-inspector-canary"


def run_fixture(**changes):
    return _run(**changes)


def jobs_fixture(run=None):
    run = run or run_fixture()
    payload = _jobs(run)
    for index, job in enumerate(payload["jobs"], 1):
        job.update(
            id=index, check_run_url=f"https://api.github.com/repos/{REPO}/check-runs/{index}"
        )
    return payload


def api_fixture(run=None, jobs=None, annotations=None, change=None):
    run = run or run_fixture()
    jobs = jobs if jobs is not None else jobs_fixture(run)
    calls = []

    def read(path, timeout):
        calls.append((path, timeout))
        if "/annotations?" in path:
            value = [] if annotations is None else annotations
        elif path.endswith("/jobs?per_page=100"):
            value = jobs
        else:
            value = run
        value = copy.deepcopy(value)
        return change(path, value, calls) if change else value

    return read, calls


def inspect(request, **changes):
    return ci.inspect_run(repo=REPO, run_id=RUN_ID, expected_sha=SHA, request=request, **changes)


def failed_start():
    run = run_fixture(conclusion="failure")
    jobs = jobs_fixture(run)
    jobs["jobs"] = [j for j in jobs["jobs"] if j["name"] in ("preflight", "ci-gate")]
    for job in jobs["jobs"]:
        job.update(conclusion="failure", steps=[])
    for name in HEAVY_JOBS:
        number = 20 + len(jobs["jobs"])
        jobs["jobs"].append(
            dict(
                name=name,
                id=number,
                run_id=RUN_ID,
                run_attempt=1,
                head_sha=SHA,
                status="completed",
                conclusion="skipped",
                steps=[],
            )
        )
    jobs["total_count"] = len(jobs["jobs"])
    return run, jobs


def test_complete_success_requires_nine_real_jobs_and_one_pinned_attempt():
    request, calls = api_fixture()
    result, code = inspect(request)
    assert code == 0 and result["category"] == "full_success"
    assert len(result["jobs"]) == 9
    assert result["attempt"] == 1 and result["sha"] == SHA
    assert [path for path, _ in calls] == [
        f"repos/{REPO}/actions/runs/164",
        f"repos/{REPO}/actions/runs/164/attempts/1",
        f"repos/{REPO}/actions/runs/164/attempts/1/jobs?per_page=100",
        f"repos/{REPO}/actions/runs/164",
    ]
    assert all(0 < timeout <= 5 for _, timeout in calls)


def test_real_platform_billing_refusal_is_distinct_and_body_free():
    run, jobs = failed_start()
    annotations = [
        {
            "annotation_level": "failure",
            "message": "The job was not started because recent account payments have failed "
            "or your spending limit needs to be increased. " + CANARY,
        }
    ]
    request, _ = api_fixture(run, jobs, annotations)
    result, code = inspect(request)
    assert code == 1 and result["category"] == "billing_blocked"
    assert {j["category"] for j in result["jobs"]} == {"billing_blocked", "prerequisite_blocked"}
    assert CANARY not in json.dumps(result)
    assert "payments" not in json.dumps(result)


@pytest.mark.parametrize(
    "annotations",
    [[], [{"annotation_level": "failure", "message": "Unrecognized platform reason " + CANARY}]],
)
def test_missing_or_unknown_annotation_does_not_guess_billing(annotations):
    run, jobs = failed_start()
    request, _ = api_fixture(run, jobs, annotations)
    result, code = inspect(request)
    assert code == 1 and result["category"] == "startup_unknown"
    assert CANARY not in json.dumps(result)


def test_failed_step_is_reported_by_allowlisted_name_only():
    run = run_fixture(conclusion="failure")
    jobs = jobs_fixture(run)
    job = next(j for j in jobs["jobs"] if j["name"] == "python-validation")
    job["conclusion"] = "failure"
    step = next(s for s in job["steps"] if s["name"] == "Collect all tests")
    step["conclusion"] = "failure"
    job["steps"].append(dict(name=CANARY, status="completed", conclusion="failure"))
    request, _ = api_fixture(run, jobs)
    result, code = inspect(request)
    assert code == 1 and result["category"] == "step_failed"
    assert next(j for j in result["jobs"] if j["name"] == job["name"])["failed_steps"] == [
        "Collect all tests"
    ]
    assert CANARY not in json.dumps(result)


@pytest.mark.parametrize(
    "status,conclusion,category,exit_code",
    [
        ("queued", None, "pending", 2),
        ("in_progress", None, "pending", 2),
        ("completed", "cancelled", "cancelled", 1),
    ],
)
def test_non_successful_run_state_is_honest(status, conclusion, category, exit_code):
    run = run_fixture(status=status, conclusion=conclusion)
    request, _ = api_fixture(run, {"total_count": 0, "jobs": []})
    result, code = inspect(request)
    assert (result["category"], code) == (category, exit_code)


def test_docs_only_is_recognized_but_never_full_success():
    run, jobs = failed_start()
    run["conclusion"] = "success"
    for job in jobs["jobs"]:
        if job["name"] in ("preflight", "ci-gate"):
            job.update(
                conclusion="success",
                steps=[
                    dict(name=name, status="completed", conclusion="success")
                    for name in FULL_JOB_REQUIRED_STEPS[job["name"]]
                ],
            )
    request, _ = api_fixture(run, jobs)
    result, code = inspect(request)
    assert code == 2 and result["category"] == "docs_only_not_full"
    assert set(j["name"] for j in result["jobs"]) == set(CHECKS)


@pytest.mark.parametrize(
    "changes",
    [
        {"head_sha": "2" * 40},
        {"repository": {"full_name": "other/repository"}},
        {"head_repository": {"full_name": "foreign/fork"}},
        {"path": ".github/workflows/other.yml"},
        {"id": 165},
        {"run_attempt": True},
    ],
)
def test_unrelated_run_identity_never_passes(changes):
    request, _ = api_fixture(run_fixture(**changes))
    result, code = inspect(request)
    assert code == 2 and result["category"] == "run_identity_mismatch"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_job",
        "duplicate_job",
        "other_attempt",
        "other_sha",
        "other_run",
        "missing_step",
        "duplicate_step",
        "skipped_step",
        "pagination",
    ],
)
def test_incomplete_or_mixed_success_evidence_never_passes(mutation):
    jobs = jobs_fixture()
    if mutation == "missing_job":
        jobs["jobs"].pop()
        jobs["total_count"] -= 1
    elif mutation == "duplicate_job":
        jobs["jobs"][-1] = copy.deepcopy(jobs["jobs"][0])
    elif mutation.startswith("other_"):
        key, value = {
            "other_attempt": ("run_attempt", 2),
            "other_sha": ("head_sha", "2" * 40),
            "other_run": ("run_id", 999),
        }[mutation]
        jobs["jobs"][0][key] = value
    elif mutation == "missing_step":
        jobs["jobs"][0]["steps"].pop()
    elif mutation == "duplicate_step":
        jobs["jobs"][0]["steps"].append(copy.deepcopy(jobs["jobs"][0]["steps"][0]))
    elif mutation == "skipped_step":
        jobs["jobs"][0]["steps"][0]["conclusion"] = "skipped"
    else:
        jobs["total_count"] += 1
    request, _ = api_fixture(jobs=jobs)
    result, code = inspect(request)
    assert code == 2 and result["category"] != "full_success"


def test_new_attempt_started_during_read_invalidates_default_latest_result():
    def change(path, value, calls):
        if len(calls) == 4:
            value["run_attempt"] = 2
        return value

    request, _ = api_fixture(change=change)
    result, code = inspect(request)
    assert code == 2 and result["category"] == "newer_attempt_available"


def test_explicit_attempt_is_pinned_and_does_not_read_newer_jobs():
    def change(path, value, calls):
        if "/attempts/1" not in path:
            value["run_attempt"] = 2
        return value

    request, calls = api_fixture(change=change)
    result, code = inspect(request, attempt=1)
    assert code == 0 and result["attempt"] == 1
    assert len(calls) == 3


@pytest.mark.parametrize(
    "category",
    [
        "query_timeout",
        "response_limit",
        "pagination_incomplete",
        "api_permission_or_limit",
        "api_unavailable",
        CANARY,
    ],
)
def test_transport_failure_never_passes_or_leaks_error_text(category):
    def request(*args):
        raise ci.EvidenceError(category)

    result, code = inspect(request)
    assert code == 2 and CANARY not in json.dumps(result)


def test_total_budget_exhaustion_stops_more_requests():
    current = [0]
    request, calls = api_fixture()

    def slow(path, timeout):
        current[0] += 31
        return request(path, timeout)

    result, code = inspect(slow, clock=lambda: current[0])
    assert code == 2 and result["category"] == "query_budget_exhausted"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body,code,category",
    [
        (
            b'HTTP/2.0 200 OK\r\nLink: <https://example.invalid>; rel="next"\r\n\r\n{}',
            0,
            "pagination_incomplete",
        ),
        (b"HTTP/2.0 403 Forbidden\r\n\r\n" + CANARY.encode(), 1, "api_permission_or_limit"),
        (b"HTTP/2.0 429 Limited\n\n{}", 1, "api_permission_or_limit"),
        (b"HTTP/2.0 200 OK\n\nnot-json", 0, "invalid_api_response"),
        (b"HTTP/2.0 500 Error\n\n{}", 1, "api_unavailable"),
        (b"x" * (ci.MAX_BYTES + 1), 0, "response_limit"),
    ],
    ids=["pagination", "forbidden", "rate-limited", "invalid-json", "server-error", "oversized"],
)
def test_bounded_http_response_parser(body, code, category):
    with pytest.raises(ci.EvidenceError) as error:
        ci.parse_response(body, code)
    assert error.value.category == category


def test_valid_http_array_and_object_responses():
    assert (
        ci.parse_response(b"HTTP/2.0 200 OK\r\nContent-Type: application/json\r\n\r\n[]", 0) == []
    )
    assert ci.parse_response(b'HTTP/2.0 200 OK\n\n{"jobs": []}', 0) == {"jobs": []}


def test_untrusted_check_url_is_never_followed():
    run, jobs = failed_start()
    jobs["jobs"][0]["check_run_url"] = "https://attacker.invalid/" + CANARY
    request, calls = api_fixture(run, jobs)
    result, code = inspect(request)
    assert code == 2 and result["category"] == "check_identity_mismatch"
    assert len(calls) == 3 and CANARY not in json.dumps(result)


def test_shared_validator_keeps_historical_baseline_contract():
    from scripts.ci_classify_changes import _full_attempt

    assert _full_attempt is full_attempt
    assert full_attempt(jobs_fixture(), run_fixture())
    assert not full_attempt(None, None)
    jobs = jobs_fixture()
    jobs["jobs"][0]["name"] = []
    assert not full_attempt(jobs, run_fixture())


def test_missing_gh_is_a_safe_incomplete_result(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError(CANARY)

    monkeypatch.setattr(subprocess, "Popen", missing)
    result, code = ci.inspect_run(repo=REPO, run_id=RUN_ID, expected_sha=SHA)
    assert code == 2 and result["category"] == "api_unavailable"
    assert CANARY not in json.dumps(result)


@pytest.mark.parametrize(
    "program,timeout,category",
    [
        ("import time; time.sleep(10)", 0.05, "query_timeout"),
        ("import sys; sys.stdout.write('x' * 3000000); sys.stdout.flush()", 5, "response_limit"),
    ],
)
def test_transport_bounds_live_child_process_without_network(
    monkeypatch, program, timeout, category
):
    original = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        assert command[:5] == ["gh", "api", "--hostname", "github.com", "--include"]
        process = original([sys.executable, "-c", program], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    with pytest.raises(ci.EvidenceError) as error:
        ci.github_json("repos/owner/pathfinder/actions/runs/164", timeout)
    assert error.value.category == category
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize(
    "mutation", ["array_name", "boolean_attempt", "foreign_check", "invalid_annotation"]
)
def test_invalid_nested_platform_evidence_stays_incomplete(mutation):
    run, jobs = failed_start()
    annotations = []
    if mutation == "array_name":
        jobs["jobs"][0]["name"] = [CANARY]
    elif mutation == "boolean_attempt":
        jobs["jobs"][0]["run_attempt"] = True
    elif mutation == "foreign_check":
        jobs["jobs"][0]["check_run_url"] = "https://api.github.com/repos/foreign/repo/check-runs/1"
    else:
        annotations = {"message": CANARY}
    request, _ = api_fixture(run, jobs, annotations)
    result, code = inspect(request)
    assert code == 2 and CANARY not in json.dumps(result)


def test_cli_emits_safe_json_and_preserves_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(ci, "inspect_run", lambda **kwargs: ({"category": "billing_blocked"}, 1))
    assert ci.main(["--repo", REPO, "--run-id", "164", "--expected-sha", SHA]) == 1
    assert json.loads(capsys.readouterr().out) == {"category": "billing_blocked"}


def test_failed_attempt_with_no_jobs_is_not_misdiagnosed_as_step_failure():
    run = run_fixture(conclusion="failure")
    request, _ = api_fixture(run=run, jobs={"total_count": 0, "jobs": []})
    result, code = inspect(request)
    assert code == 1 and result["category"] == "no_jobs_failure"
