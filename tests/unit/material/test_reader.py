from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.errors import DomainValidationError
from app.material import reader
from app.material.aliases import MaterialAlias, MaterialAliasRegistry, safe_relative_path
from app.material.chunking import prepare_material_file
from app.material.reader import MaterialReadError, read_alias


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_git_object_and_independent_document_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "main.py").write_text("print('synthetic')\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = 'sample'\n", encoding="utf-8")
    (root / "outside.md").write_text("outside authorized scope\n", encoding="utf-8")
    _git(root, "add", "main.py", "pyproject.toml", "outside.md")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "sample",
    )
    commit = _git(root, "rev-parse", "HEAD")
    workspace = uuid4()
    git_alias = MaterialAlias(
        "code", "git", root, ("main.py", "pyproject.toml"), (workspace,), commit
    )
    first = read_alias(git_alias)
    assert first.source_revision == commit
    assert first.entrypoint_files == ("main.py",)
    assert first.dependency_files == ("pyproject.toml",)
    assert first.files[0].content == b"print('synthetic')\n"
    assert first.authorized_paths == ("main.py", "pyproject.toml")
    assert all(item.path != "outside.md" for item in first.files)
    prepared = prepare_material_file(first.files[0])
    assert prepared.source_type == "code"
    assert prepared.chunks[0].start_line == prepared.chunks[0].end_line == 1
    (root / "main.py").write_text("print('changed working tree')\n", encoding="utf-8")
    assert read_alias(git_alias).digest == first.digest
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("first\n", encoding="utf-8")
    doc_alias = MaterialAlias("notes", "file", docs, ("notes.md",), (workspace,))
    before = read_alias(doc_alias)
    (docs / "notes.md").write_text("second\n", encoding="utf-8")
    after = read_alias(doc_alias)
    assert before.digest != after.digest
    assert before.files[0].content == b"first\n"
    assert MaterialAliasRegistry((git_alias, doc_alias)).get("code", workspace) == git_alias
    with pytest.raises(DomainValidationError):
        MaterialAliasRegistry((git_alias, doc_alias)).get("code", uuid4())


@pytest.mark.parametrize(
    "relative", ["../escape.md", "/absolute.md", "a//b.md", "a/./b.md", "a\\b.md"]
)
def test_relative_paths_reject_escape(relative: str) -> None:
    with pytest.raises(DomainValidationError):
        safe_relative_path(relative)


def test_file_reader_rejects_symlink_binary_secret_and_limits(tmp_path: Path) -> None:
    workspace = uuid4()
    (tmp_path / "target.md").write_text("safe", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(tmp_path / "target.md")
    (tmp_path / "binary.txt").write_bytes(b"a\x00b")
    (tmp_path / ".env").write_text("SYNTHETIC=1", encoding="utf-8")
    (tmp_path / "large.md").write_bytes(b"x" * (1024 * 1024 + 1))
    for path, expected in (
        ("link.md", "file_unavailable"),
        ("binary.txt", "binary_file"),
        (".env", "forbidden_file"),
        ("large.md", "file_not_regular_or_too_large"),
    ):
        with pytest.raises(MaterialReadError) as caught:
            read_alias(MaterialAlias("files", "file", tmp_path, (path,), (workspace,)))
        assert caught.value.code == expected


def test_snapshot_limits_are_enforced_during_read(tmp_path: Path, monkeypatch) -> None:
    workspace = uuid4()
    (tmp_path / "one.md").write_text("one", encoding="utf-8")
    (tmp_path / "two.md").write_text("two", encoding="utf-8")
    alias = MaterialAlias("files", "file", tmp_path, ("one.md", "two.md"), (workspace,))
    monkeypatch.setattr(reader, "MAX_SNAPSHOT_FILES", 1)
    with pytest.raises(MaterialReadError, match="too_many_files"):
        read_alias(alias)
    monkeypatch.setattr(reader, "MAX_SNAPSHOT_FILES", 2000)
    monkeypatch.setattr(reader, "MAX_SNAPSHOT_BYTES", 4)
    with pytest.raises(MaterialReadError, match="snapshot_too_large"):
        read_alias(alias)
