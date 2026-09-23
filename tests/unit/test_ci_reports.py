import json
from types import SimpleNamespace

import pytest

from scripts.ci_reports import (
    LIMIT,
    OUTCOMES,
    PHASES,
    RESTORE_CHECKS,
    RESTORE_TABLES,
    main,
    pytest_projection,
    read_json,
    restore_projection,
    scanner_projection,
    smoke_projection,
    write_json,
)
from scripts.ci_reports import (
    test_path as checked_path,
)
from tests.ci_reports import Reports


def restore_packet():
    return {
        "status": "PASS",
        "stage": "complete",
        "elapsed_ms": 1234,
        "old_revision": "0014_gate6_action_recovery",
        "revision": "0017_r12_resume_commands",
        "graph": "pathfinder-research-v6",
        "postgres_image": "pgvector/pgvector:0.8.5-pg16",
        "postgres_image_id": "sha256:" + "a" * 64,
        "checks": dict.fromkeys(RESTORE_CHECKS, True),
        "resources_stopped": True,
        "tables": {name: {"rows": 1, "sha256": "b" * 64} for name in RESTORE_TABLES},
        "dump_bytes": 1024,
        "dump_sha256": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "business_digest": "e" * 64,
    }


def test_successful_restore_evidence_survives_both_projections():
    value = restore_packet()
    value["raw_dump"] = "PRIVATE-CANARY"
    value["tables"]["public.runs"]["body"] = "PRIVATE-CANARY"
    reporter = Reports()
    reporter.record(
        SimpleNamespace(
            nodeid="tests/integration/db/test_release_restore.py::test_real_upgrade_dump_restore_and_resume",
            outcome="passed",
            passed=True,
            user_properties=[("restore", value)],
        ),
        "call",
    )
    result = pytest_projection(
        {**packet(), "records": reporter.records, "counts": dict(reporter.counts)}
    )
    assert result["records"][0]["restore"] == restore_packet()
    assert "PRIVATE-CANARY" not in json.dumps(result)


@pytest.mark.parametrize("change", ["stage", "digest", "table", "check", "missing", "stopped"])
def test_restore_projection_rejects_malformed_or_incomplete_pass(change):
    value = restore_packet()
    if change == "stage":
        value["stage"] = "PRIVATE-CANARY"
    elif change == "digest":
        value["dump_sha256"] = "PRIVATE-CANARY"
    elif change == "table":
        value["tables"]["PRIVATE-CANARY"] = {"rows": 1, "sha256": "a" * 64}
    elif change == "check":
        value["checks"]["exact_restore"] = False
    elif change == "missing":
        value["checks"].pop("exact_restore")
    else:
        value["resources_stopped"] = False
    with pytest.raises(ValueError, match="invalid_diagnostic"):
        restore_projection(value)


def test_partial_restore_keeps_completed_checks_without_fabricating_pass():
    value = restore_packet()
    value.update(status="IN_PROGRESS", stage="restore", checks={"legacy_upgrade": True})
    value.pop("business_digest")
    assert restore_projection(value) == value


@pytest.mark.parametrize("primary", [0, 1])
def test_invalid_restore_evidence_cannot_silently_turn_green(tmp_path, monkeypatch, primary):
    monkeypatch.setenv("PF_CI_PYTEST_REPORTS", str(tmp_path / "reports"))
    reporter = Reports()
    reporter.record(
        SimpleNamespace(
            nodeid="tests/integration/db/test_release_restore.py::test_real_upgrade_dump_restore_and_resume",
            outcome="passed",
            passed=True,
            user_properties=[("restore", {"status": "PASS"})],
        ),
        "call",
    )
    session = SimpleNamespace(exitstatus=primary)
    reporter.pytest_sessionfinish(session, primary)
    assert session.exitstatus == (primary or 3)


def packet():
    return {
        "schema_version": 1,
        "records": [],
        "counts": {p + ":" + o: 0 for p in PHASES for o in OUTCOMES},
        "exit_code": 1,
        "dropped": 0,
    }


def test_reports_exclude_parameters_longrepr_and_captured_secrets():
    reporter = Reports()
    for phase in ("setup", "call", "teardown"):
        reporter.record(
            SimpleNamespace(
                nodeid="tests/unit/test_ci_reports.py::test_name[PRIVATE_CANARY]",
                outcome="failed",
                passed=False,
                longrepr="PRIVATE_CANARY",
                capstdout="PRIVATE_CANARY",
                user_properties=[],
            ),
            phase,
        )
    value = {**packet(), "records": reporter.records, "counts": dict(reporter.counts)}
    result = pytest_projection(value)
    assert len(result["records"]) == 3
    assert "PRIVATE_CANARY" not in json.dumps(result)
    assert result["exit_code"] == 1


@pytest.mark.parametrize(
    "path",
    ["../secret.py", "/tmp/secret.py", "tests/unit/../../../secret.py", "tests/unit/unlisted.py"],
)
def test_rejects_unverified_paths(path):
    with pytest.raises(ValueError):
        checked_path(path)


def test_missing_corrupt_large_symlink_and_create_only(tmp_path):
    with pytest.raises(OSError):
        read_json(tmp_path / "missing")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError):
        read_json(bad)
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (LIMIT + 1))
    with pytest.raises(ValueError):
        read_json(large)
    link = tmp_path / "link.json"
    link.symlink_to(bad)
    with pytest.raises(ValueError):
        read_json(link)
    with pytest.raises(FileExistsError):
        write_json(bad, {})


def test_record_limit_and_cancelled_exit_preserved():
    reporter = Reports()
    report = SimpleNamespace(
        nodeid="tests/unit/test_ci_reports.py::test",
        passed=False,
        outcome="skipped",
        user_properties=[],
    )
    for _ in range(300):
        reporter.record(report, "call")
    value = pytest_projection(
        {
            **packet(),
            "records": reporter.records,
            "counts": dict(reporter.counts),
            "dropped": reporter.dropped,
            "exit_code": 2,
        }
    )
    assert len(value["records"]) == 256 and value["dropped"] == 44
    assert value["counts"]["call:skipped"] == 300 and value["exit_code"] == 2


def test_publication_failure_does_not_replace_primary_exit(tmp_path, monkeypatch):
    destination = tmp_path / "file"
    destination.write_text("occupied")
    monkeypatch.setenv("PF_CI_PYTEST_REPORTS", str(destination))
    for original, expected in ((1, 1), (2, 2), (0, 3)):
        session = SimpleNamespace(
            exitstatus=original,
            config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: None)),
        )
        Reports().pytest_sessionfinish(session, original)
        assert session.exitstatus == expected


def test_scan_projection_separates_findings_and_tool_failure():
    item = {
        "VulnerabilityID": "CVE-2026-86145",
        "PkgName": "libpcre2-8-0",
        "InstalledVersion": "10.42-1",
        "FixedVersion": "10.42-1+deb12u1",
        "Severity": "HIGH",
        "Description": "PRIVATE_CANARY",
    }
    value = scanner_projection(
        {"SchemaVersion": 2, "Results": [{"Vulnerabilities": [item]}]}, "failure", "os"
    )
    assert value["category"] == "vulnerabilities_found" and value["count"] == 1
    assert "PRIVATE_CANARY" not in json.dumps(value)
    assert scanner_projection({"SchemaVersion": 2}, "failure", "os")["category"] == "scanner_failed"
    secret = scanner_projection(
        {
            "SchemaVersion": 2,
            "Results": [{"Secrets": [{"Severity": "HIGH", "Match": "PRIVATE_CANARY"}]}],
        },
        "failure",
        "secret_scan",
    )
    assert secret["count"] == 1 and secret["findings"] == []
    assert "PRIVATE_CANARY" not in json.dumps(secret)


def test_missing_scan_reports_fail_diagnostics_and_leave_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("PF_CI_JOB", "image-security")
    monkeypatch.setenv("PF_CI_STEPS", json.dumps({"os": {"outcome": "failure"}}))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main() == 1
    value = read_json(tmp_path / "pf-diagnostics/diagnostics.json")
    assert value["steps"]["os"]["outcome"] == "failure"
    assert value["scans"]["os"]["category"] == "report_missing_or_invalid"


def test_smoke_projection_rejects_canary_and_omits_body():
    from tests.performance.smoke import SmokeRecord

    record = SmokeRecord(
        source_commit="a" * 40, mode="research", profile="instant-v1", approval_mode="none"
    )
    value = record.model_dump(mode="json") | {"body": "PRIVATE_CANARY"}
    assert "PRIVATE_CANARY" not in json.dumps(smoke_projection(value))
    value["missing_roles"] = ["PRIVATE_CANARY"]
    with pytest.raises(ValueError):
        smoke_projection(value)


def test_successful_step_requires_corresponding_report(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("PF_CI_JOB", "quality")
    monkeypatch.setenv("PF_CI_STEPS", json.dumps({"unit": {"outcome": "success"}}))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert main() == 1
    result = read_json(tmp_path / "pf-diagnostics/diagnostics.json")
    assert result["diagnostic_errors"] == ["missing_pytest_report"]


def test_optional_plugin_runs_in_real_pytest_process(tmp_path):
    import os
    import subprocess
    import sys

    from scripts.ci_reports import ROOT

    directory = tmp_path / "reports"
    child_env = {**os.environ, "PF_CI_PYTEST_REPORTS": str(directory), "PF_CI_TEST_STEP": "unit"}
    # Select one existing deterministic test; never recurse into this test itself.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests.ci_reports",
            "-q",
            "tests/unit/test_ci_reports.py::test_smoke_projection_rejects_canary_and_omits_body",
        ],
        cwd=ROOT,
        env=child_env,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0
    paths = list(directory.iterdir())
    assert len(paths) == 1
    report = pytest_projection(read_json(paths[0]))
    assert report["step"] == "unit" and report["exit_code"] == 0
    assert report["counts"]["call:passed"] == 1


@pytest.mark.parametrize("value", [None, [], "PRIVATE_CANARY", 1])
def test_malformed_smoke_payload_has_only_fixed_error(value):
    with pytest.raises(ValueError, match=r"^invalid_diagnostic$"):
        smoke_projection(value)
