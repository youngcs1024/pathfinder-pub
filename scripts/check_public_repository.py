#!/usr/bin/env python3
"""Check only staged/committed Git objects against the reviewed public file inventory."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

MANIFEST = ".github/public-files.json"
ROOT_FILES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitignore",
        ".node-version",
        ".python-version",
        "Caddyfile",
        "Dockerfile",
        "Makefile",
        "alembic.ini",
        "compose.dev.yaml",
        "compose.yaml",
        "pyproject.toml",
        "uv.lock",
    }
)
FIXTURES = {
    "e47-rubric-v1": (
        "rubric.json",
        "examples.json",
        "appropriate_refusal.txt",
        "exaggerated_experience.txt",
        "normal_citation.txt",
        "partial_support.txt",
        "unnecessary_refusal.txt",
        "wrong_citation.txt",
    ),
    "e47-agent-delegated-v1": ("rubric.json", "examples.json"),
}
REQUIRED = ROOT_FILES | {
    MANIFEST,
    ".github/workflows/ci.yml",
    ".github/actions/setup-python/action.yml",
    "scripts/check_public_repository.py",
    "evals/baselines/research_v3.json",
    *(
        f"src/app/agents/prompts/{name}.md"
        for name in ("system", "tool", "synthesis", "research", "research_plan", "research_writer")
    ),
    *(
        f"tests/fixtures/quality_reviews/{group}/{name}"
        for group in FIXTURES
        for name in FIXTURES[group]
    ),
}
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".codex",
        ".agents",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "docs",
        "node_modules",
        "repository-backups",
        "pathfinder-private",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".dump",
        ".sqlite",
        ".sqlite3",
        ".db",
        ".log",
        ".zip",
        ".tar",
        ".gz",
    }
)


class BoundaryError(ValueError):
    """Only fixed categories may be printed; rejected paths/content may be private."""


def _git(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", *arguments), cwd=repository, capture_output=True, check=False, timeout=30
    )
    if result.returncode:
        raise BoundaryError("git_read_failed")
    return result.stdout


def _allowed_path(value: str) -> bool:
    if not value or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts:
        return False
    parts = tuple(part.casefold() for part in path.parts)
    if any(part in FORBIDDEN_PARTS for part in parts):
        return False
    if any(part == ".env" or part.startswith(".env.") for part in parts):
        return value == ".env.example"
    if parts[-1] in {
        "agents.md",
        "readme.md",
        "credentials.json",
        "secrets.json",
        "id_rsa",
        "id_ed25519",
    }:
        return False
    if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
        return False
    return value in ROOT_FILES or (
        len(parts) > 1 and parts[0] in {"src", "scripts", "tests", "evals", ".github"}
    )


def _entries(repository: Path, revision: str | None) -> dict[str, tuple[str, str]]:
    if revision is None:
        raw = _git(repository, "ls-files", "--stage", "-z")
    else:
        commit = (
            _git(repository, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
            .decode("ascii")
            .strip()
        )
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise BoundaryError("invalid_revision")
        raw = _git(repository, "ls-tree", "-r", "-z", commit)
    entries = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, second, third = metadata.decode("ascii").split()
        if revision is None:
            oid = second
            if third != "0":
                raise BoundaryError("unmerged_index")
        else:
            oid = third
            if second != "blob":
                raise BoundaryError("non_regular_entry")
        name = name.decode("utf-8")
        if not _allowed_path(name):
            raise BoundaryError("forbidden_path")
        if mode not in {"100644", "100755"}:
            raise BoundaryError("non_regular_entry")
        if name in entries or not re.fullmatch(r"[0-9a-f]{40}", oid):
            raise BoundaryError("invalid_entry")
        entries[name] = (mode, oid)
    return entries


def check_public_repository(repository: Path, revision: str | None = None) -> int:
    entries = _entries(repository, revision)
    if MANIFEST not in entries:
        raise BoundaryError("missing_manifest")
    manifest = json.loads(_git(repository, "cat-file", "blob", entries[MANIFEST][1]))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files"}
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or not isinstance(manifest["files"], list)
        or not all(isinstance(name, str) and _allowed_path(name) for name in manifest["files"])
    ):
        raise BoundaryError("invalid_manifest")
    files = manifest["files"]
    if files != sorted(set(files)):
        raise BoundaryError("invalid_manifest")
    if not REQUIRED <= set(files):
        raise BoundaryError("missing_required_resource")
    if set(files) != set(entries):
        raise BoundaryError("inventory_mismatch")
    # Read objects, not worktree paths: an unstaged replacement cannot hide staged bytes.
    for name in REQUIRED:
        if not _git(repository, "cat-file", "blob", entries[name][1]).strip():
            raise BoundaryError("empty_required_resource")
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", help="Check a commit instead of the current Git index.")
    args = parser.parse_args(argv)
    try:
        count = check_public_repository(args.repository, args.revision)
    except BoundaryError as exc:
        print(f"public boundary: rejected ({exc})")
        return 1
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError):
        print("public boundary: rejected (unreadable_inventory)")
        return 1
    print(f"public boundary: verified {count} files; private documentation is outside CI scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
