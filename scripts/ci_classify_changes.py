#!/usr/bin/env python3
"""Conservatively classify a GitHub Actions change set as docs-only or full CI."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

if __package__:
    from .ci_contract import (
        FULL_JOB_REQUIRED_STEPS as FULL_JOB_REQUIRED_STEPS,
    )
    from .ci_contract import (
        FULL_JOB_SKIPPED_STEPS as FULL_JOB_SKIPPED_STEPS,
    )
    from .ci_evidence import full_attempt as _full_attempt
else:
    from ci_contract import (
        FULL_JOB_REQUIRED_STEPS as FULL_JOB_REQUIRED_STEPS,
    )
    from ci_contract import (
        FULL_JOB_SKIPPED_STEPS as FULL_JOB_SKIPPED_STEPS,
    )
    from ci_evidence import full_attempt as _full_attempt

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
_ROOT_DOCUMENTATION = frozenset(
    {
        "README.md",
        "AGENTS.md",
        "pathfinder-project-overview.md",
        "pathfinder-development-phases.md",
        "pathfinder-enhancement-roadmap.md",
        "pathfinder-enhancement-task-index.md",
    }
)


@dataclass(frozen=True)
class VerifiedBaseline:
    sha: str
    run_id: int
    attempt: int


@dataclass(frozen=True)
class Classification:
    run_full: bool
    reason: str
    changed_paths: tuple[str, ...] = ()
    baseline: VerifiedBaseline | None = None


GitDiff = Callable[[str, str, Path], bytes]
BaselineLookup = Callable[[str, Path], VerifiedBaseline | None]


def _valid_commit_sha(value: str) -> bool:
    return isinstance(value, str) and bool(_COMMIT_SHA.fullmatch(value)) and value != "0" * 40


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _git_is_ancestor(base: str, head: str, repository: Path) -> bool:
    result = subprocess.run(
        ("git", "merge-base", "--is-ancestor", base, head),
        cwd=repository,
        capture_output=True,
        check=False,
        timeout=5,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeError("ancestry unavailable")
    return result.returncode == 0


def _github_json(path: str, timeout: float) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # Bound both memory and processing; diagnostics never include API response bodies.
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("oversized Actions response")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("invalid Actions response")
    return value


def find_verified_baseline(
    head: str,
    repository: Path,
    *,
    request_json: Callable[[str, float], dict] | None = None,
    is_ancestor: Callable[[str, str, Path], bool] = _git_is_ancestor,
    clock: Callable[[], float] = time.monotonic,
) -> VerifiedBaseline | None:
    """Inspect bounded, same-attempt evidence; uncertainty always disables the shortcut."""
    owner_repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", owner_repo):
        return None
    if not _valid_commit_sha(head) or (request_json is None and not os.environ.get("GH_TOKEN")):
        return None
    request_json = request_json or _github_json
    deadline = clock() + 30

    def read(path: str) -> dict:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("Actions evidence budget exhausted")
        value = request_json(path, min(5, remaining))
        if clock() >= deadline or not isinstance(value, dict):
            raise ValueError("unreliable Actions evidence")
        return value

    prefix = f"/repos/{owner_repo}/actions"
    try:
        payload = read(f"{prefix}/workflows/ci.yml/runs?branch=main&per_page=100")
        runs = payload.get("workflow_runs")
        if not isinstance(runs, list) or len(runs) > 100:
            return None
        checked = 0
        for run in runs:
            if not isinstance(run, dict):
                return None
            if (
                run.get("status") != "completed"
                or run.get("conclusion") != "success"
                or run.get("event") not in {"push", "workflow_dispatch"}
                or run.get("head_branch") != "main"
                or run.get("path") != ".github/workflows/ci.yml"
                or not isinstance(run.get("repository"), dict)
                or run["repository"].get("full_name") != owner_repo
                or not isinstance(run.get("head_repository"), dict)
                or run["head_repository"].get("full_name") != owner_repo
                or not _valid_commit_sha(run.get("head_sha"))
                or not _positive_integer(run.get("id"))
                or not _positive_integer(run.get("run_attempt"))
            ):
                continue
            if clock() >= deadline or checked >= 10:
                return None
            if not is_ancestor(run["head_sha"], head, repository):
                continue
            checked += 1
            jobs = read(
                f"{prefix}/runs/{run['id']}/attempts/{run['run_attempt']}/jobs?per_page=100"
            )
            if _full_attempt(jobs, run):
                return VerifiedBaseline(run["head_sha"], run["id"], run["run_attempt"])
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
        http.client.HTTPException,
    ):
        return None
    return None


def _is_docs_only_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    return value in _ROOT_DOCUMENTATION or (
        len(path.parts) > 1 and path.parts[0] == "docs" and path.suffix == ".md"
    )


def _git_changed_paths(base: str, head: str, repository: Path) -> bytes:
    result = subprocess.run(
        (
            "git",
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            base,
            head,
            "--",
        ),
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError("git diff failed")
    return result.stdout


def _parse_changed_paths(payload: bytes) -> tuple[str, ...]:
    if not payload or not payload.endswith(b"\0"):
        raise ValueError("git diff returned an empty or non-NUL-terminated path set")
    raw_paths = payload[:-1].split(b"\0")
    paths: list[str] = []
    for raw_path in raw_paths:
        if not raw_path:
            raise ValueError("git diff returned an empty path")
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("git diff returned a non-UTF-8 path") from exc
        paths.append(path)
    return tuple(paths)


def classify_changes(
    *,
    event_name: str,
    before: str,
    sha: str,
    pull_request_base: str,
    pull_request_head: str,
    repository: Path,
    git_diff: GitDiff = _git_changed_paths,
    baseline_lookup: BaselineLookup = find_verified_baseline,
) -> Classification:
    if event_name == "workflow_dispatch":
        return Classification(run_full=True, reason="manual workflow dispatch")
    if event_name == "push":
        base, head = before, sha
    elif event_name == "pull_request":
        base, head = pull_request_base, pull_request_head
    else:
        return Classification(run_full=True, reason="unknown or missing event")

    if not _valid_commit_sha(base) or not _valid_commit_sha(head):
        return Classification(run_full=True, reason="missing, zero, or invalid base/head SHA")

    try:
        paths = _parse_changed_paths(git_diff(base, head, repository))
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
        return Classification(run_full=True, reason="unreliable git diff")

    if all(_is_docs_only_path(path) for path in paths):
        if not _valid_commit_sha(sha):
            return Classification(run_full=True, reason="missing checked-out SHA")
        try:
            baseline = baseline_lookup(sha, repository)
            if baseline is None:
                return Classification(run_full=True, reason="no verified full CI ancestor")
            verified_paths = _parse_changed_paths(git_diff(baseline.sha, sha, repository))
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            return Classification(run_full=True, reason="unreliable baseline diff")
        if not all(_is_docs_only_path(path) for path in verified_paths):
            return Classification(run_full=True, reason="code changed since full CI baseline")
        return Classification(
            run_full=False,
            reason="allowlisted docs-only change since verified full CI",
            changed_paths=verified_paths,
            baseline=baseline,
        )
    return Classification(
        run_full=True, reason="non-documentation or unknown path", changed_paths=paths
    )


def _write_github_output(classification: Classification, output_path: str) -> None:
    path = Path(output_path)
    classification_name = "full" if classification.run_full else "docs-only"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"run_full={str(classification.run_full).lower()}\n")
        stream.write(f"classification={classification_name}\n")
        if classification.baseline:
            stream.write(f"baseline_sha={classification.baseline.sha}\n")
            stream.write(f"baseline_run_id={classification.baseline.run_id}\n")
            stream.write(f"baseline_attempt={classification.baseline.attempt}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--before", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--pull-request-base", default="")
    parser.add_argument("--pull-request-head", default="")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    classification = classify_changes(
        event_name=args.event_name,
        before=args.before,
        sha=args.sha,
        pull_request_base=args.pull_request_base,
        pull_request_head=args.pull_request_head,
        repository=args.repository,
    )
    print(f"CI classification: {classification.reason}", file=sys.stderr)
    print(f"run_full={str(classification.run_full).lower()}")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        _write_github_output(classification, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
