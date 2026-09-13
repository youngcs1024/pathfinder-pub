"""Bounded, allowlisted CI diagnostics. Raw reports never enter the upload directory."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1_048_576
STATUSES = {"success", "failure", "cancelled", "skipped"}
PHASES = {"collect", "setup", "call", "teardown"}
TEST_STEPS = {
    "collect",
    "unit",
    "architecture",
    "operational",
    "eval",
    "core",
    "retrieval",
    "demo",
    "local",
}
OUTCOMES = {"passed", "failed", "skipped"}
ROLES = {"api", "driver", "worker", "supervisor"}
REASONS = {
    "missing_role",
    "missing_sample",
    "missing_segment",
    "missing_fact",
    "write_failed",
    "sample_limit",
    "unfinished",
    "clock_invalid",
}
CATEGORIES = {
    "http_failed",
    "request_limit",
    "deadline",
    "cancelled",
    "business_failed",
    "evidence_mismatch",
    "cleanup_failed",
    "report_failed",
    "environment_failed",
}


def check(condition):
    if not condition:
        raise ValueError("invalid_diagnostic")


def number(value, maximum=100_000):
    check(type(value) is int and 0 <= value <= maximum)
    return value


def test_path(value):
    check(isinstance(value, str) and len(value) <= 512)
    path = Path(value)
    check(not path.is_absolute() and ".." not in path.parts and value.startswith("tests/"))
    manifest = json.loads((ROOT / ".github/public-files.json").read_text())["files"]
    check(value in manifest and path.suffix == ".py")
    check((ROOT / path).resolve().is_relative_to(ROOT) and (ROOT / path).is_file())
    return value


def smoke_projection(value):
    if not isinstance(value, dict):
        check(callable(getattr(value, "model_dump", None)))
        value = value.model_dump(mode="json")
    result = {}
    for key, allowed in {
        "mode": {"research", "application"},
        "status": {"PASS", "IN_PROGRESS"},
        "category": CATEGORIES | {None},
        "failure_stage": {"environment", "drive", "cleanup", "metrics", "report", None},
        "metrics_status": {"PASS", "IN_PROGRESS", "NOT_RUN"},
    }.items():
        check(value.get(key) in allowed)
        result[key] = value[key]
    for key, allowed in {
        "missing_roles": ROLES,
        "metrics_reasons": REASONS,
        "diagnostic_errors": {"cleanup_failed", "report_failed", "metrics_incomplete"},
    }.items():
        items = value.get(key)
        check(isinstance(items, (list, tuple)) and len(items) <= len(allowed))
        check(all(isinstance(item, str) and item in allowed for item in items))
        result[key] = list(items)
    check(type(value.get("resources_released")) is bool)
    result["resources_released"] = value["resources_released"]
    for key in ("http_requests", "unfinished_calls"):
        result[key] = number(value.get(key))
    return result


def pytest_projection(value):
    check(isinstance(value, dict) and value.get("schema_version") == 1)
    check(value.get("step", "local") in TEST_STEPS)
    records = value.get("records")
    check(isinstance(records, list) and len(records) <= 256)
    safe = []
    for record in records:
        check(isinstance(record, dict))
        check(record.get("phase") in PHASES and record.get("outcome") in OUTCOMES)
        item = {
            "path": test_path(record.get("path")),
            "phase": record["phase"],
            "outcome": record["outcome"],
            "category": "test_" + record["outcome"],
        }
        if "smoke" in record:
            item["smoke"] = smoke_projection(record["smoke"])
        safe.append(item)
    counts = value.get("counts")
    check(
        isinstance(counts, dict) and set(counts) == {p + ":" + o for p in PHASES for o in OUTCOMES}
    )
    return {
        "schema_version": 1,
        "step": value.get("step", "local"),
        "exit_code": number(value.get("exit_code"), 5),
        "dropped": number(value.get("dropped")),
        "records": safe,
        "counts": {key: number(count) for key, count in counts.items()},
    }


def read_json(path, limit=LIMIT):
    check(not path.is_symlink())
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        check(stat.S_ISREG(os.fstat(stream.fileno()).st_mode))
        body = stream.read(limit + 1)
    check(len(body) <= limit)
    return json.loads(body)


def write_json(path, value):
    body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    check(len(body) <= LIMIT)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)


def scanner_projection(value, outcome, kind):
    check(isinstance(value, dict) and type(value.get("SchemaVersion")) is int)
    results = value.get("Results", [])
    check(isinstance(results, list) and len(results) <= 4096)
    findings = []
    count = 0
    for result in results:
        check(isinstance(result, dict))
        items = result.get("Secrets" if kind == "secret_scan" else "Vulnerabilities", []) or []
        check(isinstance(items, list))
        for item in items:
            check(isinstance(item, dict))
            if item.get("Severity") not in {"HIGH", "CRITICAL"}:
                continue
            count += 1
            if kind != "secret_scan" and len(findings) < 256:
                safe = {}
                for field in (
                    "VulnerabilityID",
                    "PkgName",
                    "InstalledVersion",
                    "FixedVersion",
                    "Severity",
                ):
                    text = item.get(field, "")
                    check(
                        isinstance(text, str)
                        and len(text) <= 200
                        and re.fullmatch(r"[A-Za-z0-9_.:+, /~@()-]*", text) is not None
                    )
                    safe[field] = text
                findings.append(safe)
    return {
        "category": "vulnerabilities_found"
        if count
        else "scanner_failed"
        if outcome == "failure"
        else "scan_clean",
        "scanner_version": "0.70.0",
        "count": count,
        "truncated": count > len(findings) if kind != "secret_scan" else False,
        "findings": findings,
    }


def main():
    from scripts.ci_contract import CHECKS

    job = os.environ["PF_CI_JOB"]
    check(job in CHECKS)
    check(len(os.environ["PF_CI_STEPS"]) <= LIMIT)
    steps = json.loads(os.environ["PF_CI_STEPS"])
    check(isinstance(steps, dict))
    output = {
        "schema_version": 1,
        "job": job,
        "steps": {},
        "pytest": [],
        "scans": {},
        "diagnostic_errors": [],
    }
    for item in CHECKS[job]:
        if item.step_id in {"diagnostics", "upload"}:
            continue
        record = steps.get(item.step_id, {})
        check(isinstance(record, dict))
        output["steps"][item.step_id] = {
            key: record.get(key) if record.get(key) in STATUSES else "missing"
            for key in ("outcome", "conclusion")
        }
    base = Path(os.environ["RUNNER_TEMP"])
    incoming = base / "pf-pytest"
    if incoming.exists():
        check(not incoming.is_symlink())
        for index, path in enumerate(incoming.iterdir()):
            if index >= 32:
                output["diagnostic_errors"].append("report_limit")
                break
            try:
                check(re.fullmatch(r"[0-9a-f]{32}\.json", path.name) is not None)
                output["pytest"].append(pytest_projection(read_json(path)))
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                output["diagnostic_errors"].append("invalid_pytest_report")
    received = {report["step"] for report in output["pytest"]}
    for step, record in output["steps"].items():
        if (
            step in TEST_STEPS
            and record["outcome"] in {"success", "failure"}
            and step not in received
        ):
            output["diagnostic_errors"].append("missing_pytest_report")
    if job == "image-security":
        for kind in ("filesystem", "library", "os", "secret_scan"):
            outcome = output["steps"][kind]["outcome"]
            if outcome in {"skipped", "cancelled"}:
                output["scans"][kind] = {"category": outcome}
                continue
            try:
                output["scans"][kind] = scanner_projection(
                    read_json(base / f"pf-trivy-{kind}.json", 32 * LIMIT), outcome, kind
                )
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                output["scans"][kind] = {
                    "category": "report_missing_or_invalid",
                    "tool_outcome": outcome,
                }
                output["diagnostic_errors"].append("invalid_scan_report")
        image_path = base / "pf-image-id.txt"
        if image_path.exists():
            check(not image_path.is_symlink() and image_path.stat().st_size <= 100)
            content = image_path.read_text().strip()
            check(re.fullmatch(r"sha256:[0-9a-f]{64}", content) is not None)
            output["image_id"] = content
    destination = base / "pf-diagnostics"
    destination.mkdir(mode=0o700, exist_ok=False)
    write_json(destination / "diagnostics.json", output)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as stream:
            stream.write(
                "\n### Safe diagnostics\n\n"
                + f"Pytest reports: {len(output['pytest'])}; "
                + f"diagnostic errors: {len(output['diagnostic_errors'])}.\n"
            )
            for kind, scan in output["scans"].items():
                stream.write(f"- {kind}: {scan['category']}\n")
    return int(bool(output["diagnostic_errors"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print("diagnostic_publication_failed")
        raise SystemExit(1) from None
