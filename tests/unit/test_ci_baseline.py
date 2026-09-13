"""Deterministic Actions evidence and real Git ancestry; no external API calls."""

import copy
import http.client
import json
import subprocess

import pytest

from scripts import ci_classify_changes as ci

BASE = "1" * 40
CODE = "2" * 40
DOCS = "3" * 40
MERGE = "4" * 40
JOBS = tuple(ci.FULL_JOB_REQUIRED_STEPS)


@pytest.fixture(autouse=True)
def actions_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/pathfinder")
    monkeypatch.delenv("GH_TOKEN", raising=False)


def _run(**overrides):
    return {
        "id": 164,
        "run_attempt": 1,
        "head_sha": BASE,
        "head_branch": "main",
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": "owner/pathfinder"},
        "head_repository": {"full_name": "owner/pathfinder"},
        **overrides,
    }


def _jobs(run=None):
    run = run or _run()
    return {
        "total_count": len(JOBS),
        "jobs": [
            {
                "name": name,
                "run_id": run["id"],
                "run_attempt": run["run_attempt"],
                "head_sha": run["head_sha"],
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {
                        "name": step,
                        "status": "completed",
                        "conclusion": "success",
                    }
                    for step in ci.FULL_JOB_REQUIRED_STEPS[name]
                ]
                + [
                    {"name": step, "status": "completed", "conclusion": "skipped"}
                    for step in ci.FULL_JOB_SKIPPED_STEPS.get(name, ())
                ],
            }
            for name in JOBS
        ],
    }


def _api(runs=None, jobs=None):
    calls = []

    def request(path, timeout):
        calls.append((path, timeout))
        if "/workflows/ci.yml/runs?" in path:
            return {"workflow_runs": [_run()] if runs is None else runs}
        return _jobs() if jobs is None else jobs

    return request, calls


def _find(tmp_path, request, **kwargs):
    return ci.find_verified_baseline(
        DOCS, tmp_path, request_json=request, is_ancestor=lambda *args: True, **kwargs
    )


def test_completed_full_ancestor_uses_one_attempt_and_bounded_queries(tmp_path):
    run = _run(run_attempt=2)
    request, calls = _api([run], _jobs(run))
    assert _find(tmp_path, request) == ci.VerifiedBaseline(BASE, 164, 2)
    assert calls == [
        ("/repos/owner/pathfinder/actions/workflows/ci.yml/runs?branch=main&per_page=100", 5),
        ("/repos/owner/pathfinder/actions/runs/164/attempts/2/jobs?per_page=100", 5),
    ]


@pytest.mark.parametrize("job_name", JOBS)
@pytest.mark.parametrize("status", ["failure", "cancelled", "skipped", None])
def test_every_required_matrix_branch_must_succeed(tmp_path, job_name, status):
    jobs = _jobs()
    next(job for job in jobs["jobs"] if job["name"] == job_name)["conclusion"] = status
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "other_attempt", "other_sha", "other_run", "summary", "partial"],
)
def test_incomplete_or_mixed_attempt_evidence_is_rejected(tmp_path, mutation):
    jobs = _jobs()
    if mutation == "missing":
        jobs["jobs"].pop()
    elif mutation == "duplicate":
        jobs["jobs"][-1] = copy.deepcopy(jobs["jobs"][0])
    elif mutation == "summary":
        next(
            step for step in jobs["jobs"][0]["steps"] if step["name"] == "Summarize check outcomes"
        )["conclusion"] = "skipped"
    elif mutation == "partial":
        jobs["total_count"] += 1
    else:
        key, value = {
            "other_attempt": ("run_attempt", 2),
            "other_sha": ("head_sha", CODE),
            "other_run": ("run_id", 165),
        }[mutation]
        jobs["jobs"][0][key] = value
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


@pytest.mark.parametrize(
    "override",
    [
        {"status": "in_progress"},
        {"conclusion": "failure"},
        {"conclusion": "cancelled"},
        {"event": "pull_request"},
        {"head_branch": "feature"},
        {"path": ".github/workflows/other.yml"},
        {"head_sha": "0" * 40},
        {"id": True},
        {"run_attempt": 0},
        {"head_repository": {"full_name": "fork/pathfinder"}},
    ],
)
def test_candidate_identity_and_completion_are_required(tmp_path, override):
    request, calls = _api([_run(**override)])
    assert _find(tmp_path, request) is None
    assert len(calls) == 1


def test_non_ancestor_is_not_used(tmp_path):
    request, calls = _api()
    assert (
        ci.find_verified_baseline(
            DOCS, tmp_path, request_json=request, is_ancestor=lambda *args: False
        )
        is None
    )
    assert len(calls) == 1


def test_docs_green_is_not_a_full_baseline(tmp_path):
    jobs = _jobs()
    for job in jobs["jobs"]:
        if job["name"] not in {"preflight", "ci-gate"}:
            job["conclusion"] = "skipped"
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


@pytest.mark.parametrize("failure", [TimeoutError, OSError, ValueError, http.client.HTTPException])
def test_api_failure_is_conservative_and_never_leaks_response_body(tmp_path, failure):
    def request(*args):
        raise failure("PRIVATE-RESPONSE-CANARY")

    assert _find(tmp_path, request) is None


def test_search_checks_at_most_ten_job_candidates(tmp_path):
    request, calls = _api([_run(id=i) for i in range(1, 101)], {"jobs": [], "total_count": 0})
    assert _find(tmp_path, request) is None
    assert len(calls) == 11


def test_total_budget_cannot_be_extended_by_successful_responses(tmp_path):
    ticks = iter([0, 0, 29, 29, 29, 31])
    request, calls = _api()
    assert _find(tmp_path, request, clock=lambda: next(ticks)) is None
    assert calls[-1][1] == 1


def _classify(tmp_path, diff, lookup, **kwargs):
    args = dict(
        event_name="push",
        before=CODE,
        sha=DOCS,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=diff,
        baseline_lookup=lookup,
    )
    return ci.classify_changes(**(args | kwargs))


@pytest.mark.parametrize("code_state", ["in_progress", "failure", "cancelled"])
def test_docs_after_unverified_code_cannot_turn_ci_green(tmp_path, code_state):
    pending = _run(id=165, head_sha=CODE)
    pending["status" if code_state == "in_progress" else "conclusion"] = code_state
    request, _ = _api([pending, _run()])

    def lookup(head, repository):
        return ci.find_verified_baseline(
            head, repository, request_json=request, is_ancestor=lambda *args: True
        )

    def diff(base, head, repository):
        return b"docs/progress.md\0" if base == CODE else b"src/app/main.py\0docs/progress.md\0"

    assert _classify(tmp_path, diff, lookup).run_full


def test_pr_compares_verified_baseline_to_actual_merge_not_only_pr_head(tmp_path):
    calls = []

    def diff(base, head, repository):
        calls.append((base, head))
        return b"README.md\0" if head == DOCS else b"src/app/main.py\0README.md\0"

    def lookup(head, repository):
        assert head == MERGE
        return ci.VerifiedBaseline(BASE, 164, 1)

    result = _classify(
        tmp_path,
        diff,
        lookup,
        event_name="pull_request",
        pull_request_base=CODE,
        pull_request_head=DOCS,
        sha=MERGE,
    )
    assert result.run_full
    assert calls == [(CODE, DOCS), (BASE, MERGE)]


@pytest.mark.parametrize(
    ("event", "paths"), [("push", b"src/app/main.py\0"), ("workflow_dispatch", b"")]
)
def test_full_paths_do_not_query_actions(tmp_path, event, paths):
    def lookup(*args):
        pytest.fail("full path must not query Actions")

    assert _classify(tmp_path, lambda *args: paths, lookup, event_name=event).run_full


def test_missing_baseline_and_unreliable_baseline_diff_run_full(tmp_path):
    assert _classify(tmp_path, lambda *args: b"README.md\0", lambda *args: None).run_full
    assert _classify(
        tmp_path,
        lambda base, *args: b"README.md\0" if base == CODE else b"",
        lambda *args: ci.VerifiedBaseline(BASE, 164, 1),
    ).run_full


def test_github_outputs_include_only_verified_identity(tmp_path):
    destination = tmp_path / "outputs"
    result = _classify(
        tmp_path,
        lambda *args: b"README.md\0",
        lambda *args: ci.VerifiedBaseline(BASE, 164, 1),
    )
    ci._write_github_output(result, str(destination))
    assert destination.read_text().splitlines() == [
        "run_full=false",
        "classification=docs-only",
        f"baseline_sha={BASE}",
        "baseline_run_id=164",
        "baseline_attempt=1",
    ]


def test_http_reader_uses_read_token_fixed_endpoint_and_timeout(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "TOKEN-CANARY")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            assert size == 2_000_001
            return json.dumps({"workflow_runs": []}).encode()

    def open_request(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), timeout))
        return Response()

    monkeypatch.setattr(ci.urllib.request, "urlopen", open_request)
    assert ci._github_json("/repos/owner/pathfinder/actions/workflows/ci.yml/runs", 3) == {
        "workflow_runs": []
    }
    assert calls == [
        (
            "https://api.github.com/repos/owner/pathfinder/actions/workflows/ci.yml/runs",
            "Bearer TOKEN-CANARY",
            3,
        )
    ]


def test_real_git_ancestry_and_diff_include_unverified_code(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "user.name", "CI test")
    git("config", "user.email", "ci@example.invalid")
    (tmp_path / "README.md").write_text("baseline\n")
    git("add", "README.md")
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD")
    (tmp_path / "code.py").write_text("value = 1\n")
    git("add", "code.py")
    git("commit", "-m", "unverified code")
    code = git("rev-parse", "HEAD")
    (tmp_path / "README.md").write_text("documentation\n")
    git("add", "README.md")
    git("commit", "-m", "documentation")
    docs = git("rev-parse", "HEAD")
    assert ci._git_is_ancestor(baseline, docs, tmp_path)
    assert not ci._git_is_ancestor(docs, baseline, tmp_path)
    result = _classify(
        tmp_path,
        ci._git_changed_paths,
        lambda *args: ci.VerifiedBaseline(baseline, 164, 1),
        before=code,
        sha=docs,
    )
    assert result.run_full


@pytest.mark.parametrize("payload", [{}, {"workflow_runs": None}, {"workflow_runs": [None]}, []])
def test_malformed_api_listing_disables_shortcut(tmp_path, payload):
    assert _find(tmp_path, lambda *args: payload) is None


def test_missing_actions_token_disables_shortcut_without_network(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("missing token must not make network requests")

    monkeypatch.setattr(ci.urllib.request, "urlopen", forbidden)
    assert ci.find_verified_baseline(DOCS, tmp_path) is None


REQUIRED_STEPS = [
    (job, step) for job, steps in ci.FULL_JOB_REQUIRED_STEPS.items() for step in steps
]
SKIPPED_STEPS = [(job, step) for job, steps in ci.FULL_JOB_SKIPPED_STEPS.items() for step in steps]


@pytest.mark.parametrize(("job_name", "step_name"), REQUIRED_STEPS + SKIPPED_STEPS)
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incomplete", "wrong_status"])
def test_each_declared_step_is_required_once_with_its_expected_status(
    tmp_path, job_name, step_name, mutation
):
    jobs = _jobs()
    job = next(job for job in jobs["jobs"] if job["name"] == job_name)
    record = next(step for step in job["steps"] if step["name"] == step_name)
    if mutation == "missing":
        job["steps"].remove(record)
    elif mutation == "duplicate":
        job["steps"].append(copy.deepcopy(record))
    elif mutation == "incomplete":
        record["status"] = "in_progress"
    else:
        record["conclusion"] = "skipped" if record["conclusion"] == "success" else "success"
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


@pytest.mark.parametrize(("job_name", "step_name"), REQUIRED_STEPS + SKIPPED_STEPS)
@pytest.mark.parametrize("status", ["failure", "cancelled", None, "unknown"])
def test_green_job_cannot_hide_failed_or_invalid_check(tmp_path, job_name, step_name, status):
    jobs = _jobs()
    job = next(job for job in jobs["jobs"] if job["name"] == job_name)
    next(step for step in job["steps"] if step["name"] == step_name)["conclusion"] = status
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


def test_successful_summary_alone_does_not_prove_a_full_attempt(tmp_path):
    jobs = _jobs()
    for job in jobs["jobs"]:
        job["steps"] = [step for step in job["steps"] if step["name"] == "Summarize check outcomes"]
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) is None


def test_actions_automatic_steps_do_not_replace_or_invalidate_explicit_checks(tmp_path):
    jobs = _jobs()
    for job in jobs["jobs"]:
        job["steps"].insert(
            0, {"name": "Set up job", "status": "completed", "conclusion": "success"}
        )
        job["steps"].extend(
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in ("Post Check out repository", "Complete job")
        )
    request, _ = _api(jobs=jobs)
    assert _find(tmp_path, request) == ci.VerifiedBaseline(BASE, 164, 1)


@pytest.mark.parametrize("path", ["docs/helper.py", "docs/settings.yaml", "docs/data.json"])
def test_non_markdown_documentation_since_baseline_requires_full_ci(tmp_path, path):
    result = _classify(
        tmp_path,
        lambda base, *args: b"README.md\0" if base == CODE else path.encode() + b"\0",
        lambda *args: ci.VerifiedBaseline(BASE, 164, 1),
    )
    assert result.run_full
    assert result.reason == "code changed since full CI baseline"


def test_old_eight_job_baseline_cannot_satisfy_shared_precheck_contract():
    jobs = _jobs()
    jobs["jobs"] = [job for job in jobs["jobs"] if job["name"] != "python-validation"]
    jobs["total_count"] = len(jobs["jobs"])
    assert not ci._full_attempt(jobs, _run())


def test_old_attempt_without_diagnostic_publication_is_not_a_baseline():
    jobs = _jobs()
    for job in jobs["jobs"]:
        job["steps"] = [
            step
            for step in job["steps"]
            if step["name"] not in {"Validate safe diagnostics", "Upload safe diagnostics"}
        ]
    assert not ci._full_attempt(jobs, _run())
