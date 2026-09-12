"""Shared, fail-closed complete Actions attempt evidence; standard library only."""

from __future__ import annotations

if __package__:
    from .ci_contract import FULL_JOB_REQUIRED_STEPS, FULL_JOB_SKIPPED_STEPS
else:
    from ci_contract import FULL_JOB_REQUIRED_STEPS, FULL_JOB_SKIPPED_STEPS

FULL_JOBS = frozenset(FULL_JOB_REQUIRED_STEPS)


def full_attempt(jobs: dict, run: dict) -> bool:
    if not isinstance(jobs, dict) or not isinstance(run, dict):
        return False
    records = jobs.get("jobs")
    if not isinstance(records, list) or jobs.get("total_count") != len(FULL_JOBS):
        return False
    if len(records) != len(FULL_JOBS) or any(
        not isinstance(job, dict) or not isinstance(job.get("name"), str) for job in records
    ):
        return False
    if {job.get("name") for job in records} != FULL_JOBS:
        return False
    for job in records:
        if (
            job.get("run_id") != run["id"]
            or job.get("run_attempt") != run["run_attempt"]
            or job.get("head_sha") != run["head_sha"]
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            return False
        steps = job.get("steps")
        if not isinstance(steps, list) or any(not isinstance(step, dict) for step in steps):
            return False
        for names, expected in (
            (FULL_JOB_REQUIRED_STEPS[job["name"]], "success"),
            (FULL_JOB_SKIPPED_STEPS.get(job["name"], ()), "skipped"),
        ):
            for name in names:
                matches = [step for step in steps if step.get("name") == name]
                if (
                    len(matches) != 1
                    or matches[0].get("status") != "completed"
                    or matches[0].get("conclusion") != expected
                ):
                    return False
    return True
