from pathlib import Path

import pytest

from scripts.ci_classify_changes import VerifiedBaseline, classify_changes

BASE = "1" * 40
HEAD = "2" * 40


def _diff_for(*paths: str):
    payload = b"\0".join(path.encode() for path in paths) + b"\0"

    def run(_base: str, _head: str, _repository: Path) -> bytes:
        return payload

    return run


@pytest.mark.parametrize(
    "paths",
    [
        ("README.md",),
        ("AGENTS.md",),
        ("pathfinder-project-overview.md",),
        ("pathfinder-development-phases.md",),
        ("pathfinder-enhancement-roadmap.md",),
        ("pathfinder-enhancement-task-index.md",),
        ("docs/learning-log.md",),
        ("README.md", "docs/architecture/overview.md"),
        ("docs/file with spaces.md",),
    ],
)
def test_documentation_allowlist_skips_full_ci(paths: tuple[str, ...], tmp_path: Path) -> None:
    result = classify_changes(
        event_name="push",
        before=BASE,
        sha=HEAD,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=_diff_for(*paths),
        baseline_lookup=lambda head, repo: VerifiedBaseline(BASE, 164, 1),
    )

    assert result.run_full is False
    assert result.changed_paths == paths


@pytest.mark.parametrize(
    "path",
    [
        "src/app/foo.py",
        "tests/foo.py",
        "tests/fixtures/prompt.md",
        "src/app/agents/prompts/research.md",
        "evals/datasets/research.json",
        "evals/baselines/research_v3.json",
        "evals/instructions.md",
        "pathfinder-unknown.md",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "Makefile",
        ".github/workflows/ci.yml",
        "scripts/foo.sh",
        "CHANGELOG.md",
        "docs",
        "docs\\unsafe.md",
        "../docs/unsafe.md",
        "docs/helper.py",
        "docs/script.sh",
        "docs/settings.yaml",
        "docs/data.json",
        "docs/diagram.svg",
        "docs/README.MD",
        "docs/README.md.py",
        "docs//note.md",
        "docs/./note.md",
        "docs/../note.md",
        "docs/note.md/",
        "docs/line\nbreak.md",
        "docs/control\x7f.md",
    ],
)
def test_non_documentation_or_unknown_path_runs_full_ci(path: str, tmp_path: Path) -> None:
    result = classify_changes(
        event_name="pull_request",
        before="",
        sha=HEAD,
        pull_request_base=BASE,
        pull_request_head=HEAD,
        repository=tmp_path,
        git_diff=_diff_for(path),
        baseline_lookup=lambda head, repo: VerifiedBaseline(BASE, 164, 1),
    )

    assert result.run_full is True


@pytest.mark.parametrize(
    "changed_path",
    [
        "src/app/foo.py",
        "tests/foo.py",
        "evals/dataset.json",
        "src/app/agents/prompts/run.md",
        "docs/helper.py",
    ],
)
def test_mixed_documentation_and_source_runs_full_ci(tmp_path: Path, changed_path: str) -> None:
    result = classify_changes(
        event_name="push",
        before=BASE,
        sha=HEAD,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=_diff_for("pathfinder-enhancement-roadmap.md", changed_path),
        baseline_lookup=lambda head, repo: VerifiedBaseline(BASE, 164, 1),
    )

    assert result.run_full is True


@pytest.mark.parametrize(
    ("event_name", "before", "sha", "pr_base", "pr_head"),
    [
        ("push", "", HEAD, "", ""),
        ("push", "0" * 40, HEAD, "", ""),
        ("pull_request", "", "", BASE, "bad-sha"),
        ("schedule", BASE, HEAD, "", ""),
    ],
)
def test_unreliable_event_or_revision_runs_full_ci(
    event_name: str,
    before: str,
    sha: str,
    pr_base: str,
    pr_head: str,
    tmp_path: Path,
) -> None:
    result = classify_changes(
        event_name=event_name,
        before=before,
        sha=sha,
        pull_request_base=pr_base,
        pull_request_head=pr_head,
        repository=tmp_path,
        git_diff=_diff_for("README.md"),
    )

    assert result.run_full is True


def test_workflow_dispatch_always_runs_full_ci(tmp_path: Path) -> None:
    result = classify_changes(
        event_name="workflow_dispatch",
        before="",
        sha="",
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=_diff_for("README.md"),
    )

    assert result.run_full is True


@pytest.mark.parametrize("payload", [b"", b"README.md", b"README.md\0\0"])
def test_unreliable_diff_output_runs_full_ci(payload: bytes, tmp_path: Path) -> None:
    result = classify_changes(
        event_name="push",
        before=BASE,
        sha=HEAD,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=lambda _base, _head, _repository: payload,
    )

    assert result.run_full is True


def test_git_diff_failure_runs_full_ci(tmp_path: Path) -> None:
    def fail(_base: str, _head: str, _repository: Path) -> bytes:
        raise RuntimeError("missing commit")

    result = classify_changes(
        event_name="push",
        before=BASE,
        sha=HEAD,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=fail,
    )

    assert result.run_full is True


@pytest.mark.parametrize("failure", [OSError, RuntimeError, ValueError])
def test_classifier_does_not_include_exception_body(tmp_path, failure):
    def fail(*args):
        raise failure("PRIVATE-GIT-CANARY")

    result = classify_changes(
        event_name="push",
        before=BASE,
        sha=HEAD,
        pull_request_base="",
        pull_request_head="",
        repository=tmp_path,
        git_diff=fail,
    )
    assert result.run_full and result.reason == "unreliable git diff"


def test_classifier_cli_does_not_echo_unknown_event(monkeypatch, capsys):
    from scripts.ci_classify_changes import main

    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert main(["--event-name", "EVENT-CANARY"]) == 0
    output = capsys.readouterr()
    assert "CANARY" not in output.out + output.err
    assert "run_full=true" in output.out


def test_raw_git_stderr_is_not_propagated(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import ci_classify_changes

    monkeypatch.setattr(
        ci_classify_changes.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=128, stderr=b"STDERR-CANARY"),
    )
    with pytest.raises(RuntimeError, match=r"^git diff failed$"):
        ci_classify_changes._git_changed_paths(BASE, HEAD, tmp_path)
