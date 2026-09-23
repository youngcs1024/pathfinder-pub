from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.material.aliases import MaterialAlias, safe_relative_path

MAX_FILE_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 20 * 1024 * 1024
MAX_SNAPSHOT_FILES = 2000
_MAX_GIT_OUTPUT = 4 * 1024 * 1024
_SECRET = re.compile(
    r"(^|/)(\.env(?:\..*)?|\.git|id_(?:rsa|ed25519)|credentials(?:\..*)?|"
    r"secrets?(?:\..*)?|.*\.(?:pem|key|p12|pfx))$",
    re.IGNORECASE,
)
_TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".cjs",
        ".mjs",
        ".sql",
        ".sh",
        ".md",
        ".txt",
        ".toml",
        ".json",
        ".yaml",
        ".yml",
    }
)
_DEPENDENCY_NAMES = frozenset(
    {"pyproject.toml", "requirements.txt", "package.json", "Cargo.toml", "go.mod"}
)
_ENTRY_NAMES = frozenset(
    {"main.py", "app.py", "__main__.py", "index.js", "index.ts", "server.js", "server.ts"}
)


class MaterialReadError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class MaterialFile:
    path: str
    content: bytes
    digest: str
    line_count: int


@dataclass(frozen=True, slots=True, repr=False)
class MaterialRead:
    source_revision: str
    authorized_paths: tuple[str, ...]
    files: tuple[MaterialFile, ...]
    dependency_files: tuple[str, ...]
    entrypoint_files: tuple[str, ...]
    omitted_files: tuple[str, ...]
    digest: str


def _validate_name(path: str) -> None:
    try:
        safe_relative_path(path)
    except Exception:
        raise MaterialReadError("unsafe_path") from None
    if _SECRET.search(path) or Path(path).suffix.lower() not in _TEXT_SUFFIXES:
        raise MaterialReadError("forbidden_file")


def _bounded_content(path: str, content: bytes) -> MaterialFile:
    if len(content) > MAX_FILE_BYTES:
        raise MaterialReadError("file_too_large")
    if b"\x00" in content:
        raise MaterialReadError("binary_file")
    try:
        decoded = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise MaterialReadError("invalid_utf8") from None
    return MaterialFile(
        path=path,
        content=content,
        digest=hashlib.sha256(content).hexdigest(),
        line_count=len(decoded.splitlines()),
    )


def _git(alias: MaterialAlias, *args: str) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": "/tmp",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(alias.root), *args],
            check=False,
            capture_output=True,
            timeout=15,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise MaterialReadError("git_unavailable") from None
    if result.returncode != 0:
        raise MaterialReadError("git_object_missing")
    if len(result.stdout) > _MAX_GIT_OUTPUT:
        raise MaterialReadError("inventory_too_large")
    return result.stdout


def _read_git(alias: MaterialAlias) -> tuple[str, tuple[MaterialFile, ...]]:
    assert alias.commit is not None
    resolved = _git(alias, "rev-parse", "--verify", f"{alias.commit}^{{commit}}").strip()
    if resolved.decode("ascii", errors="ignore") != alias.commit:
        raise MaterialReadError("git_object_missing")
    tree = _git(alias, "ls-tree", "-r", "-z", alias.commit, "--", *alias.paths)
    files: list[MaterialFile] = []
    total_bytes = 0
    for item in tree.split(b"\x00"):
        if not item:
            continue
        try:
            meta, raw_path = item.split(b"\t", 1)
            mode, object_type, oid = meta.split(b" ", 2)
            path = raw_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError):
            raise MaterialReadError("git_tree_invalid") from None
        if not any(
            path == allowed or path.startswith(allowed.rstrip("/") + "/") for allowed in alias.paths
        ):
            raise MaterialReadError("git_tree_invalid")
        _validate_name(path)
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise MaterialReadError("symlink_or_submodule")
        if len(files) >= MAX_SNAPSHOT_FILES:
            raise MaterialReadError("too_many_files")
        content = _git(alias, "cat-file", "blob", oid.decode("ascii"))
        file = _bounded_content(path, content)
        total_bytes += len(file.content)
        if total_bytes > MAX_SNAPSHOT_BYTES:
            raise MaterialReadError("snapshot_too_large")
        files.append(file)
    if not files:
        raise MaterialReadError("no_files")
    return alias.commit, tuple(files)


def _read_regular(root: Path, relative: str) -> bytes:
    _validate_name(relative)
    # Walk each component with O_NOFOLLOW so a replacement symlink cannot escape.
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            parts = relative.split("/")
            for part in parts[:-1]:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                before = os.fstat(file_fd)
                if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
                    raise MaterialReadError("file_not_regular_or_too_large")
                with os.fdopen(file_fd, "rb", closefd=False) as stream:
                    content = stream.read(MAX_FILE_BYTES + 1)
                after = os.fstat(file_fd)
                if (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mtime_ns,
                    after.st_size,
                ):
                    raise MaterialReadError("file_changed_during_read")
                return content
            finally:
                os.close(file_fd)
        finally:
            os.close(fd)
    except OSError:
        raise MaterialReadError("file_unavailable") from None


def read_alias(alias: MaterialAlias) -> MaterialRead:
    if len(alias.paths) > MAX_SNAPSHOT_FILES:
        raise MaterialReadError("too_many_files")
    if alias.kind == "git":
        revision, files = _read_git(alias)
    else:
        selected: list[MaterialFile] = []
        total_bytes = 0
        for path in alias.paths:
            file = _bounded_content(path, _read_regular(alias.root, path))
            total_bytes += len(file.content)
            if total_bytes > MAX_SNAPSHOT_BYTES:
                raise MaterialReadError("snapshot_too_large")
            selected.append(file)
        files = tuple(selected)
        revision = hashlib.sha256(
            "".join(f"{file.path}:{file.digest}\n" for file in files).encode("utf-8")
        ).hexdigest()
    if sum(len(item.content) for item in files) > MAX_SNAPSHOT_BYTES:
        raise MaterialReadError("snapshot_too_large")
    manifest = "".join(f"{item.path}:{item.digest}\n" for item in files)
    return MaterialRead(
        source_revision=revision,
        authorized_paths=alias.paths,
        files=files,
        dependency_files=tuple(
            item.path for item in files if Path(item.path).name in _DEPENDENCY_NAMES
        ),
        entrypoint_files=tuple(item.path for item in files if Path(item.path).name in _ENTRY_NAMES),
        omitted_files=(),
        digest=hashlib.sha256((alias.digest + revision + manifest).encode("utf-8")).hexdigest(),
    )
