"""Report fixed CI steps without exposing Actions outputs or exception text."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if __package__:
    from .ci_contract import BRANCHES, CHECKS, STEP_IDS
else:
    from ci_contract import BRANCHES, CHECKS, STEP_IDS

STATUSES = {"success", "failure", "cancelled", "skipped"}


def step_summary(job: str, branch: str, payload: str) -> tuple[str, bool]:
    if job not in STEP_IDS or branch not in BRANCHES.get(job, {""}):
        raise ValueError("invalid CI summary identity")
    if len(payload) > 262_144:
        raise ValueError("invalid CI summary data")
    try:
        steps = json.loads(payload)
    except (ValueError, RecursionError):
        raise ValueError("invalid CI summary data") from None
    if not isinstance(steps, dict):
        raise ValueError("invalid CI summary data")
    inactive = {check.step_id for check in CHECKS[job] if not check.applies(branch)}
    dependencies = {check.step_id: check.prerequisites for check in CHECKS[job]}
    title = job + (f" ({branch})" if branch else "")
    lines = [
        f"### {title}",
        "",
        "| Check | Applicability | Outcome | Conclusion |",
        "| --- | --- | --- | --- |",
    ]
    issues: list[str] = []
    for step_id in STEP_IDS[job]:
        record = steps.get(step_id)
        values = []
        for field in ("outcome", "conclusion"):
            value = record.get(field) if isinstance(record, dict) else None
            values.append(
                value
                if isinstance(value, str) and value in STATUSES
                else "missing"
                if value is None
                else "invalid"
            )
        outcome, conclusion = values
        applicable = step_id not in inactive
        expected = "success" if applicable else "skipped"
        if outcome != expected or conclusion != expected:
            categories = []
            if "failure" in values:
                categories.append("failed")
            if "cancelled" in values:
                categories.append("cancelled")
            if applicable and "skipped" in values:
                categories.append("required check skipped")
                blocked = []
                for dependency in dependencies[step_id]:
                    prerequisite = steps.get(dependency)
                    if (
                        not isinstance(prerequisite, dict)
                        or prerequisite.get("outcome") != "success"
                    ):
                        blocked.append(dependency)
                if blocked:
                    categories.append("blocked by prerequisite: " + ", ".join(blocked))
                elif "cancelled" not in values:
                    categories.append("unexpected skip: prerequisites succeeded")
            if "missing" in values:
                categories.append("missing status")
            if "invalid" in values:
                categories.append("invalid status")
            if outcome != conclusion:
                categories.append("outcome/conclusion mismatch")
            if not applicable and any(
                value in {"success", "failure", "cancelled"} for value in values
            ):
                categories.append("non-applicable check executed")
            issues.append(f"- {step_id}: {', '.join(categories)} ({outcome}/{conclusion}).")
        label = "required" if applicable else "not applicable"
        lines.append(f"| {step_id} | {label} | {outcome} | {conclusion} |")
    lines.extend(["", "Skipped/missing required checks are not evidence of success.", ""])
    overview = issues or ["No check status problems detected."]
    lines[2:2] = [
        *overview,
        "",
        "Status diagnostics only; inspect the original check logs for causes.",
        "",
    ]
    return "\n".join(lines), not issues


def main() -> int:
    try:
        summary, passed = step_summary(
            os.environ.get("PF_CI_JOB", ""),
            os.environ.get("PF_CI_BRANCH", ""),
            os.environ.get("PF_CI_STEPS", ""),
        )
        destination = os.environ.get("GITHUB_STEP_SUMMARY")
        if not destination:
            raise ValueError("missing summary destination")
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(summary)
    except (OSError, ValueError):
        print("CI step summary could not be produced", file=sys.stderr)
        return 1
    print(summary, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
