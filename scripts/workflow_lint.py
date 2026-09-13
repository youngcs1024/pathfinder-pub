"""Pinned workflow semantic validator; preparation is separate from read-only checking."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.7.12"
ARCHIVE_SHA = "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
BINARY_SHA = "c872d6db8c6bf83a8eaa704fc93999f027d55dffbc63b8a6abdccb47df5f4cd4"
URL = f"https://github.com/rhysd/actionlint/releases/download/v{VERSION}/actionlint_{VERSION}_linux_amd64.tar.gz"
LIMIT = 20_000_000


def require(condition, category):
    if not condition:
        raise ValueError(category)


def binary_path():
    cache = Path(
        os.environ.get(
            "ACTIONLINT_CACHE",
            str(Path(tempfile.gettempdir()) / f"pathfinder-actionlint-{os.getuid()}"),
        )
    )
    return cache / f"actionlint-{VERSION}"


def verify(binary):
    require(binary.is_file() and not binary.is_symlink(), "workflow_tool_missing")
    require(binary.stat().st_size <= LIMIT, "workflow_tool_digest_mismatch")
    require(
        hashlib.sha256(binary.read_bytes()).hexdigest() == BINARY_SHA,
        "workflow_tool_digest_mismatch",
    )
    result = subprocess.run([str(binary), "-version"], capture_output=True, timeout=5, check=False)
    require(
        result.returncode == 0 and result.stdout.splitlines()[0:1] == [VERSION.encode()],
        "workflow_tool_version_mismatch",
    )


def unpack(body):
    require(
        len(body) <= LIMIT and hashlib.sha256(body).hexdigest() == ARCHIVE_SHA,
        "workflow_archive_digest_mismatch",
    )
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        member = archive.getmember("actionlint")
        require(member.isfile() and member.size <= LIMIT, "workflow_archive_invalid")
        data = archive.extractfile(member).read(LIMIT + 1)
    require(hashlib.sha256(data).hexdigest() == BINARY_SHA, "workflow_tool_digest_mismatch")
    return data


def prepare(binary):
    require(
        platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"},
        "workflow_platform_unsupported",
    )
    binary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = binary.parent.lstat()
    require(
        stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o022,
        "workflow_cache_unsafe",
    )
    if binary.exists():
        verify(binary)
        return
    # No floating script execution, extraction paths, overwrites or implicit tool upgrades.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL, timeout=10) as response:
                body = response.read(LIMIT + 1)
            break
        except OSError:
            if attempt == 2:
                raise
    data = unpack(body)
    descriptor = os.open(binary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    verify(binary)


def workflows(root):
    manifest = json.loads((root / ".github/public-files.json").read_text())["files"]
    paths = sorted(
        p
        for p in manifest
        if p.startswith(".github/workflows/") and Path(p).suffix in {".yml", ".yaml"}
    )
    require(bool(paths), "workflow_inventory_empty")
    for path in paths:
        require(
            not (root / path).is_symlink()
            and (root / path).resolve().is_relative_to(root.resolve()),
            "workflow_path_invalid",
        )
    return paths


def self_check(binary):
    # Exercise the actual expression parser, not a regex approximation of Actions semantics.
    valid = b"""name: regression
on: push
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
        env:
          GOOD: ${{ runner.temp }}
"""
    invalid = b"env:\n  BAD: ${{ runner.temp }}\n" + valid
    for source, expected in ((invalid, 1), (valid, 0)):
        result = subprocess.run(
            [str(binary), "-shellcheck=", "-pyflakes=", "-"],
            input=source,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        require(result.returncode == expected, "workflow_validator_regression")


def lint(binary, root=ROOT):
    verify(binary)
    self_check(binary)
    return subprocess.run(
        [str(binary), "-shellcheck=", "-pyflakes=", *workflows(root)],
        cwd=root,
        timeout=30,
        check=False,
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    args = parser.parse_args(argv)
    try:
        binary = binary_path()
        if args.command == "prepare":
            prepare(binary)
            print(f"actionlint {VERSION} verified")
            return 0
        return lint(binary)
    except ValueError as error:
        print(str(error) if str(error).startswith("workflow_") else "workflow_validation_failed")
    except (OSError, subprocess.SubprocessError, tarfile.TarError, KeyError):
        print("workflow_tool_unavailable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
