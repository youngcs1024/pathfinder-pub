"""Inspect a pinned GitHub CI attempt without starting jobs or changing account settings."""

from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import time

if __package__:
    from .ci_contract import CHECKS, FULL_JOB_REQUIRED_STEPS, HEAVY_JOBS, JOB_PREREQUISITES
    from .ci_evidence import full_attempt
else:
    from ci_contract import CHECKS, FULL_JOB_REQUIRED_STEPS, HEAVY_JOBS, JOB_PREREQUISITES
    from ci_evidence import full_attempt

REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
SHA = re.compile(r"[0-9a-f]{40}")
STATES = {"queued", "in_progress", "completed", "waiting", "pending", "requested"}
CONCLUSIONS = {
    None,
    "success",
    "failure",
    "cancelled",
    "skipped",
    "timed_out",
    "neutral",
    "action_required",
    "stale",
    "startup_failure",
}
PROFILES = set(FULL_JOB_REQUIRED_STEPS) | set(CHECKS)
MAX_BYTES = 2_000_000
ERROR_CATEGORIES = {
    "evidence_incomplete",
    "response_limit",
    "api_permission_or_limit",
    "api_unavailable",
    "pagination_incomplete",
    "invalid_api_response",
    "query_timeout",
    "run_identity_mismatch",
    "job_identity_mismatch",
    "invalid_step_evidence",
    "invalid_annotation_evidence",
    "invalid_input",
    "query_budget_exhausted",
    "check_identity_mismatch",
    "newer_attempt_available",
}
BILLING_MESSAGE = (
    "the job was not started because recent account payments have failed "
    "or your spending limit needs to be increased"
)


class EvidenceError(Exception):
    """Only a fixed category is allowed across the CLI boundary."""

    def __init__(self, category="evidence_incomplete"):
        self.category = category


def _positive(value):
    return type(value) is int and 0 < value < 10**20


def parse_response(payload: bytes, code: int):
    if len(payload) > MAX_BYTES:
        raise EvidenceError("response_limit")
    try:
        headers, body = payload.decode("utf-8").replace("\r\n", "\n").split("\n\n", 1)
        status = int(headers.splitlines()[0].split()[1])
        if status in (401, 403, 429):
            raise EvidenceError("api_permission_or_limit")
        if code or status != 200:
            raise EvidenceError("api_unavailable")
        if any(
            line.lower().startswith("link:") and re.search(r'rel="?next\b', line.lower())
            for line in headers.splitlines()
        ):
            raise EvidenceError("pagination_incomplete")
        return json.loads(body)
    except (UnicodeError, ValueError, IndexError, RecursionError):
        raise EvidenceError("invalid_api_response") from None


def github_json(path: str, timeout: float):
    """Bound stdout while draining; discard stderr rather than publishing gh errors."""
    environment = dict(os.environ, GH_PROMPT_DISABLED="1")
    try:
        with subprocess.Popen(
            ["gh", "api", "--hostname", "github.com", "--include", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        ) as process:
            deadline = time.monotonic() + timeout
            chunks = []
            size = 0
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise EvidenceError("query_timeout")
                        for key, _ in selector.select(min(0.1, remaining)):
                            data = os.read(key.fileobj.fileno(), 16384)
                            if not data:
                                selector.unregister(key.fileobj)
                                continue
                            size += len(data)
                            if size > MAX_BYTES:
                                raise EvidenceError("response_limit")
                            chunks.append(data)
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except (EvidenceError, OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        return parse_response(b"".join(chunks), code)
    except subprocess.TimeoutExpired:
        raise EvidenceError("query_timeout") from None
    except OSError:
        raise EvidenceError("api_unavailable") from None


def validate_run(run, repo, run_id, sha, attempt=None):
    if (
        not isinstance(run, dict)
        or run.get("id") != run_id
        or not _positive(run.get("id"))
        or not _positive(run.get("run_attempt"))
        or run.get("head_sha") != sha
        or run.get("path") != ".github/workflows/ci.yml"
        or not isinstance(run.get("repository"), dict)
        or run["repository"].get("full_name") != repo
        or not isinstance(run.get("head_repository"), dict)
        or run["head_repository"].get("full_name") != repo
        or run.get("status") not in STATES
        or run.get("conclusion") not in CONCLUSIONS
        or (attempt is not None and run.get("run_attempt") != attempt)
    ):
        raise EvidenceError("run_identity_mismatch")


def validate_jobs(payload, run):
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
        raise EvidenceError()
    jobs = payload["jobs"]
    if (
        type(payload.get("total_count")) is not int
        or payload["total_count"] != len(jobs)
        or len(jobs) > 100
    ):
        raise EvidenceError("pagination_incomplete")
    names, ids = set(), set()
    for job in jobs:
        if (
            not isinstance(job, dict)
            or not isinstance(job.get("name"), str)
            or job["name"] not in PROFILES
            or job["name"] in names
            or not _positive(job.get("id"))
            or job["id"] in ids
            or job.get("run_id") != run["id"]
            or job.get("run_attempt") != run["run_attempt"]
            or not _positive(job.get("run_id"))
            or not _positive(job.get("run_attempt"))
            or job.get("head_sha") != run["head_sha"]
            or job.get("status") not in STATES
            or job.get("conclusion") not in CONCLUSIONS
            or not isinstance(job.get("steps"), list)
        ):
            raise EvidenceError("job_identity_mismatch")
        names.add(job["name"])
        ids.add(job["id"])
        steps = job["steps"]
        if any(
            not isinstance(s, dict)
            or not isinstance(s.get("name"), str)
            or s.get("status") not in STATES
            or s.get("conclusion") not in CONCLUSIONS
            for s in steps
        ):
            raise EvidenceError("invalid_step_evidence")
    return jobs


def _billing_annotation(annotations):
    if (
        not isinstance(annotations, list)
        or len(annotations) > 100
        or any(
            not isinstance(a, dict) or not isinstance(a.get("message"), str) for a in annotations
        )
    ):
        raise EvidenceError("invalid_annotation_evidence")
    return any(
        a.get("annotation_level") == "failure"
        and BILLING_MESSAGE in " ".join(a["message"].lower().split())
        for a in annotations
    )


def inspect_run(
    *, repo, run_id, expected_sha, attempt=None, request=github_json, clock=time.monotonic
):
    """Return (allowlisted JSON summary, exit code); never trust a top-level green alone."""
    result = {"category": "invalid_input"}
    try:
        if (
            not isinstance(repo, str)
            or not REPOSITORY.fullmatch(repo)
            or any(part in (".", "..") for part in repo.split("/"))
            or not isinstance(expected_sha, str)
            or not SHA.fullmatch(expected_sha)
            or expected_sha == "0" * 40
            or not _positive(run_id)
            or (attempt is not None and not _positive(attempt))
        ):
            raise EvidenceError("invalid_input")
        prefix = f"repos/{repo}"
        deadline = clock() + 30

        def read(path):
            remaining = deadline - clock()
            if remaining <= 0:
                raise EvidenceError("query_budget_exhausted")
            value = request(path, min(5, remaining))
            if clock() >= deadline:
                raise EvidenceError("query_budget_exhausted")
            return value

        latest = read(f"{prefix}/actions/runs/{run_id}")
        validate_run(latest, repo, run_id, expected_sha)
        selected = attempt or latest["run_attempt"]
        run = read(f"{prefix}/actions/runs/{run_id}/attempts/{selected}")
        validate_run(run, repo, run_id, expected_sha, selected)
        result = {
            "repo": repo,
            "run_id": run_id,
            "attempt": selected,
            "sha": expected_sha,
            "url": f"https://github.com/{repo}/actions/runs/{run_id}/attempts/{selected}",
            "category": "evidence_incomplete",
            "jobs": [],
        }
        payload = read(f"{prefix}/actions/runs/{run_id}/attempts/{selected}/jobs?per_page=100")
        jobs = validate_jobs(payload, run)
        categories = []
        for job in jobs:
            state = job["status"]
            conclusion = job["conclusion"]
            category = "success"
            if state != "completed":
                category = "running" if state == "in_progress" else "queued"
            elif conclusion == "cancelled":
                category = "cancelled"
            elif conclusion == "skipped":
                base = job["name"].split(" (")[0]
                blocked = []
                for prerequisite in JOB_PREREQUISITES[base]:
                    upstream = [j for j in jobs if j["name"].split(" (")[0] == prerequisite]
                    if not upstream or any(j["conclusion"] != "success" for j in upstream):
                        blocked.append(prerequisite)
                category = "prerequisite_blocked" if blocked else "skipped"
            elif conclusion != "success":
                if not job["steps"]:
                    # Derive check identity only from the validated GitHub API URL;
                    # never follow a URL supplied in an untrusted response.
                    url = job.get("check_run_url", "")
                    match = re.fullmatch(
                        rf"https://api\.github\.com/repos/{re.escape(repo)}/check-runs/([1-9][0-9]*)",
                        url,
                    )
                    if match is None:
                        raise EvidenceError("check_identity_mismatch")
                    annotations = read(f"{prefix}/check-runs/{match[1]}/annotations?per_page=100")
                    category = (
                        "billing_blocked" if _billing_annotation(annotations) else "startup_unknown"
                    )
                else:
                    category = "step_failed"
            known = set(FULL_JOB_REQUIRED_STEPS.get(job["name"], ()))
            failed_steps = [
                s["name"]
                for s in job["steps"]
                if s["name"] in known and s["conclusion"] in ("failure", "timed_out")
            ]
            result["jobs"].append(
                {
                    "name": job["name"],
                    "status": state,
                    "conclusion": conclusion,
                    "category": category,
                    "failed_steps": failed_steps,
                }
            )
            categories.append(category)
        if attempt is None:
            current = read(f"{prefix}/actions/runs/{run_id}")
            validate_run(current, repo, run_id, expected_sha)
            if current["run_attempt"] != selected:
                raise EvidenceError("newer_attempt_available")
        if "billing_blocked" in categories:
            result["category"] = "billing_blocked"
            return result, 1
        if any(c in categories for c in ("startup_unknown", "step_failed", "cancelled")):
            result["category"] = next(
                c for c in ("startup_unknown", "step_failed", "cancelled") if c in categories
            )
            return result, 1
        if run["status"] != "completed" or any(c in categories for c in ("queued", "running")):
            result["category"] = "pending"
            return result, 2
        if run["conclusion"] in ("cancelled", "failure", "timed_out", "startup_failure"):
            result["category"] = "cancelled" if run["conclusion"] == "cancelled" else "run_failed"
            return result, 1
        if run["conclusion"] == "success" and full_attempt(payload, run):
            result["category"] = "full_success"
            return result, 0
        if run["conclusion"] == "success" and {j["name"] for j in jobs} == set(CHECKS):
            active = [j for j in jobs if j["name"] in ("preflight", "ci-gate")]
            if all(j["conclusion"] == "skipped" for j in jobs if j["name"] in HEAVY_JOBS) and all(
                j["conclusion"] == "success"
                and all(
                    sum(
                        s["name"] == name
                        and s["status"] == "completed"
                        and s["conclusion"] == "success"
                        for s in j["steps"]
                    )
                    == 1
                    for name in FULL_JOB_REQUIRED_STEPS[j["name"]]
                )
                for j in active
            ):
                result["category"] = "docs_only_not_full"
                return result, 2
        return result, 2
    except EvidenceError as error:
        result["category"] = (
            error.category if error.category in ERROR_CATEGORIES else "evidence_incomplete"
        )
        return result, 2
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        result["category"] = "evidence_incomplete"
        return result, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args(argv)
    result, code = inspect_run(
        repo=args.repo, run_id=args.run_id, expected_sha=args.expected_sha, attempt=args.attempt
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
