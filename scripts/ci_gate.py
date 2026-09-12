#!/usr/bin/env python3
"""Fail-closed aggregation for the stable CI gate job."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from .ci_contract import HEAVY_JOBS, JOB_PREREQUISITES
else:
    from ci_contract import HEAVY_JOBS, JOB_PREREQUISITES


def valid_baseline(sha: str, run_id: str, attempt: str) -> bool:
    return (
        bool(re.fullmatch(r"[0-9a-fA-F]{40}", sha))
        and sha != "0" * 40
        and bool(re.fullmatch(r"[1-9][0-9]{0,19}", run_id))
        and bool(re.fullmatch(r"[1-9][0-9]{0,9}", attempt))
    )


def safe_results(results: Mapping[str, str]) -> dict[str, str]:
    statuses = {"success", "failure", "cancelled", "skipped", "missing"}
    return {
        job: value if isinstance(value, str) and value in statuses else "invalid"
        for job in ("preflight", *HEAVY_JOBS)
        for value in (results.get(job, "missing"),)
    }


def job_summary(
    results: Mapping[str, str],
    *,
    run_full: str = "",
    baseline_sha: str = "",
    baseline_run_id: str = "",
    baseline_attempt: str = "",
) -> str:
    """Only fixed job names and allowlisted statuses may reach the Actions summary."""
    path = {"true": "full", "false": "docs-only"}.get(run_full, "invalid")
    issues = []
    if path == "invalid":
        issues.append("- classifier: missing or invalid output; CI path cannot be established.")
    baseline_valid = valid_baseline(baseline_sha, baseline_run_id, baseline_attempt)
    if path == "docs-only" and not baseline_valid:
        issues.append("- baseline: missing or invalid verified full CI evidence.")
    for job, status in safe_results(results).items():
        expected = "success" if job == "preflight" or path == "full" else "skipped"
        if job != "preflight" and path == "invalid":
            issues.append(
                f"- {job}: {status}; expected status unavailable without a valid CI path."
            )
        elif status != expected:
            blocked = [name for name in JOB_PREREQUISITES[job] if results.get(name) != "success"]
            detail = (
                "; blocked by prerequisite: " + ", ".join(blocked)
                if status == "skipped" and blocked
                else ""
            )
            issues.append(f"- {job}: {status}; expected {expected}{detail}.")
    lines = [
        f"CI path: {path}",
        "",
        *(issues or ["All job statuses match the selected CI path."]),
        "",
        "Gate aggregates upstream checks and CI path consistency; inspect check logs for causes.",
        "",
        "| Job | Status |",
        "| --- | --- |",
    ]
    for job, status in safe_results(results).items():
        lines.append(f"| {job} | {status} |")
    if path == "docs-only" and baseline_valid:
        lines.extend(
            [
                "",
                f"Inherited full CI: commit {baseline_sha}, run {baseline_run_id}, "
                f"attempt {baseline_attempt}.",
                "Current docs checks do not re-execute business tests or close stage acceptance.",
            ]
        )
    return "\n".join(lines) + "\n"


def evaluate_ci_gate(
    *,
    run_full: str,
    results: Mapping[str, str],
    baseline_sha: str = "",
    baseline_run_id: str = "",
    baseline_attempt: str = "",
) -> tuple[bool, str]:
    results = safe_results(results)
    if results.get("preflight") != "success":
        return False, "preflight did not succeed"
    if run_full == "true":
        unexpected = {
            job: results.get(job, "missing") for job in HEAVY_JOBS if results.get(job) != "success"
        }
        if unexpected:
            return False, f"full CI requires every heavy job to succeed: {unexpected}"
        return True, "full CI succeeded"
    if run_full == "false":
        if not valid_baseline(baseline_sha, baseline_run_id, baseline_attempt):
            return False, "docs-only CI requires verified full CI baseline evidence"
        unexpected = {
            job: results.get(job, "missing") for job in HEAVY_JOBS if results.get(job) != "skipped"
        }
        if unexpected:
            return False, f"docs-only CI requires every heavy job to be skipped: {unexpected}"
        return True, "docs-only preflight succeeded and heavy jobs were skipped"
    return False, "classifier output is missing or invalid"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-full", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--baseline-sha", default="")
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument("--baseline-attempt", default="")
    for job in HEAVY_JOBS:
        parser.add_argument(f"--{job}", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = safe_results(
        {
            "preflight": args.preflight,
            **{job: getattr(args, job.replace("-", "_")) for job in HEAVY_JOBS},
        }
    )
    baseline = {
        "baseline_sha": args.baseline_sha,
        "baseline_run_id": args.baseline_run_id,
        "baseline_attempt": args.baseline_attempt,
    }
    passed, reason = evaluate_ci_gate(run_full=args.run_full, results=results, **baseline)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with Path(summary_path).open("a", encoding="utf-8") as summary:
                summary.write(job_summary(results, run_full=args.run_full, **baseline))
        except OSError:
            print("could not write CI job summary", file=sys.stderr)
            return 1
    print(f"ci-gate {'PASS' if passed else 'FAIL'}: {reason}")
    if not passed:
        print(f"job results: {results}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
