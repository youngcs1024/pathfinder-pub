"""Opt-in pytest diagnostics; never serialize longrepr, captured output or parameters."""

import os
from collections import Counter
from pathlib import Path
from uuid import uuid4

from scripts.ci_reports import (
    OUTCOMES,
    PHASES,
    restore_projection,
    smoke_projection,
    test_path,
    write_json,
)


class Reports:
    def __init__(self):
        self.records = []
        self.counts = Counter({p + ":" + o: 0 for p in PHASES for o in OUTCOMES})
        self.dropped = 0
        self.invalid_restore = False

    def record(self, report, phase):
        self.counts[phase + ":" + report.outcome] += 1
        smoke = next(
            (value for key, value in getattr(report, "user_properties", ()) if key == "smoke"), None
        )
        restore = (
            next(
                (
                    value
                    for key, value in getattr(report, "user_properties", ())
                    if key == "restore"
                ),
                None,
            )
            if phase == "call"
            else None
        )
        if report.passed and smoke is None and restore is None:
            return
        if len(self.records) >= 256:
            self.dropped += 1
            self.invalid_restore |= restore is not None
            return
        try:
            path = test_path(report.nodeid.split("::", 1)[0])
            item = {"path": path, "phase": phase, "outcome": report.outcome}
            if smoke is not None:
                item["smoke"] = smoke_projection(smoke)
            if restore is not None:
                item["restore"] = restore_projection(restore)
            self.records.append(item)
        except (ValueError, TypeError, KeyError):
            self.dropped += 1
            self.invalid_restore |= restore is not None

    def pytest_collectreport(self, report):
        self.record(report, "collect")

    def pytest_runtest_logreport(self, report):
        self.record(report, report.when)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.invalid_restore and not session.exitstatus:
            session.exitstatus = 3
            exitstatus = 3
        try:
            directory = Path(os.environ["PF_CI_PYTEST_REPORTS"])
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink():
                raise ValueError("invalid_directory")
            write_json(
                directory / (uuid4().hex + ".json"),
                {
                    "schema_version": 1,
                    "step": os.environ.get("PF_CI_TEST_STEP", "local"),
                    "exit_code": int(exitstatus),
                    "dropped": self.dropped,
                    "counts": dict(self.counts),
                    "records": self.records,
                },
            )
        except (OSError, ValueError):
            # A primary pytest error retains its exit status.
            if not session.exitstatus:
                session.exitstatus = 3
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line("diagnostic_publication_failed")


def pytest_configure(config):
    if os.environ.get("PF_CI_PYTEST_REPORTS"):
        config.pluginmanager.register(Reports(), "safe_ci_reports")
