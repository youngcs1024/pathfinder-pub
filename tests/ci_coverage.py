"""Opt-in collection ownership check; does not execute test fixtures or bodies."""

import os
from pathlib import Path

import pytest

from scripts.ci_contract import collection_ownership
from scripts.ci_diagnostics import collection_diagnostic


def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.config.option.collectonly:
        raise pytest.UsageError("CI ownership plugin requires collect-only")
    if session.testsfailed:
        return  # Preserve pytest's original collection error and exit status.
    try:
        counts = collection_ownership(tuple(item.nodeid for item in session.items))
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "CI execution ownership (collection only)")
        for target, count in counts.items():
            reporter.write_line(f"{target}: {count}")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # At most eight validated file pointers; never copy exception or parameter bodies.
    messages = []
    for report in terminalreporter.stats.get("error", ())[:8]:
        diagnostic = collection_diagnostic(
            "ERROR collecting " + report.nodeid + "\n" + str(report.longrepr), root=config.rootpath
        )
        messages.append(
            f"CI collection FAIL: phase=collect exit_code={int(exitstatus)} {diagnostic}"
        )
    if not messages:
        return
    message = "\n".join(messages)
    terminalreporter.write_line(message)
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        try:
            with Path(destination).open("a") as stream:
                stream.write(message + "\n")
        except OSError:
            raise pytest.UsageError("could not write CI collection summary") from None
