"""No network or tool installation in unit tests; CI preflight executes the real validator."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import workflow_lint as lint


def test_missing_tool_and_wrong_digest_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="workflow_tool_missing"):
        lint.verify(tmp_path / "missing")
    path = tmp_path / "tool"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="workflow_tool_digest_mismatch"):
        lint.verify(path)


def test_wrong_version_rejected_even_after_digest_verification(tmp_path, monkeypatch):
    path = tmp_path / "tool"
    path.write_bytes(b"tool")
    monkeypatch.setattr(lint, "BINARY_SHA", lint.hashlib.sha256(b"tool").hexdigest())
    monkeypatch.setattr(
        lint.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=b"1.0.0\n")
    )
    with pytest.raises(ValueError, match="workflow_tool_version_mismatch"):
        lint.verify(path)


def test_archive_digest_checked_before_parsing():
    with pytest.raises(ValueError, match="workflow_archive_digest_mismatch"):
        lint.unpack(b"untrusted_archive_canary")


def test_all_inventory_workflows_and_failure_exit_are_preserved(tmp_path, monkeypatch):
    directory = tmp_path / ".github/workflows"
    directory.mkdir(parents=True)
    for name in ("one.yml", "two.yaml"):
        (directory / name).write_text("name: test")
    (tmp_path / ".github/public-files.json").write_text(
        json.dumps(
            {"files": [".github/workflows/one.yml", ".github/workflows/two.yaml", "README.md"]}
        )
    )
    monkeypatch.setattr(lint, "verify", lambda _: None)
    monkeypatch.setattr(lint, "self_check", lambda _: None)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(lint.subprocess, "run", run)
    assert lint.lint(Path("/tool"), tmp_path) == 1
    assert calls == [
        [
            "/tool",
            "-shellcheck=",
            "-pyflakes=",
            ".github/workflows/one.yml",
            ".github/workflows/two.yaml",
        ]
    ]


def test_preparation_is_not_implicit_in_check(monkeypatch):
    monkeypatch.setattr(lint, "prepare", lambda _: pytest.fail("unexpected_install"))
    monkeypatch.setattr(lint, "lint", lambda _: 7)
    assert lint.main(["check"]) == 7


def test_context_regression_checks_both_rejection_and_acceptance(monkeypatch):
    inputs = []

    def run(command, **kwargs):
        inputs.append(kwargs["input"])
        return SimpleNamespace(returncode=1 if len(inputs) == 1 else 0)

    monkeypatch.setattr(lint.subprocess, "run", run)
    lint.self_check(Path("/tool"))
    assert b"env:\n  BAD: ${{ runner.temp }}" in inputs[0]
    assert b"env:\n  BAD:" not in inputs[1]
    assert b"GOOD: ${{ runner.temp }}" in inputs[1]


def test_self_check_failure_is_not_ignored(monkeypatch):
    monkeypatch.setattr(lint.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="workflow_validator_regression"):
        lint.self_check(Path("/tool"))
