"""The publication boundary checks Git objects even when private files remain locally."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts.check_public_repository import MANIFEST, REQUIRED, main


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-c", "core.hooksPath=/dev/null", *arguments), cwd=root, stderr=subprocess.PIPE
    ).decode()


def inventory(root: Path, files: set[str]) -> None:
    (root / MANIFEST).write_text(json.dumps({"schema_version": 1, "files": sorted(files)}))


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--initial-branch=main")
    for name in REQUIRED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (tmp_path / ".gitignore").write_text("/docs/\n/*.md\n.env\n")
    inventory(tmp_path, set(REQUIRED))
    git(tmp_path, "add", "--all")
    return tmp_path


def test_no_technical_documents_needed_and_private_worktree_content_is_ignored(repository, capsys):
    (repository / "docs").mkdir()
    (repository / "docs/progress.md").write_text("PRIVATE_BODY_CANARY")
    (repository / "AGENTS.md").write_text("PRIVATE_INSTRUCTIONS_CANARY")
    (repository / "README.md").write_text("PRIVATE_README_CANARY")
    (repository / ".env").write_text("PRIVATE_SECRET_CANARY")
    assert main(["--repository", str(repository)]) == 0
    assert "CANARY" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "name", ["docs/progress.md", "AGENTS.md", "README.md", ".env", "src/key.pem"]
)
def test_forbidden_paths_fail_even_if_added_to_manifest(repository, name, capsys):
    path = repository / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("PRIVATE_BODY_CANARY")
    inventory(repository, set(REQUIRED) | {name})
    git(repository, "add", "--force", "--", name, MANIFEST)
    assert main(["--repository", str(repository)]) == 1
    output = capsys.readouterr().out
    assert "forbidden_path" in output
    assert name not in output and "CANARY" not in output


def test_unregistered_source_cannot_be_published(repository, capsys):
    path = repository / "src/new.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text("pass\n")
    git(repository, "add", "--", "src/new.py")
    assert main(["--repository", str(repository)]) == 1
    assert "inventory_mismatch" in capsys.readouterr().out


@pytest.mark.parametrize(
    "name",
    [
        "src/app/agents/prompts/system.md",
        "evals/baselines/research_v3.json",
        "tests/fixtures/quality_reviews/e47-rubric-v1/examples.json",
    ],
)
@pytest.mark.parametrize("also_remove_from_manifest", [False, True])
def test_missing_resources_fail_even_when_local_file_still_exists(
    repository, name, also_remove_from_manifest, capsys
):
    git(repository, "update-index", "--force-remove", "--", name)
    if also_remove_from_manifest:
        inventory(repository, set(REQUIRED) - {name})
        git(repository, "add", "--", MANIFEST)
    assert main(["--repository", str(repository)]) == 1
    assert (repository / name).exists()
    assert "rejected" in capsys.readouterr().out


def test_symlink_cannot_read_outside_repository(repository, capsys):
    path = repository / "src/external.py"
    path.symlink_to("../../PRIVATE_TARGET_CANARY")
    inventory(repository, set(REQUIRED) | {"src/external.py"})
    git(repository, "add", "--", "src/external.py", MANIFEST)
    assert main(["--repository", str(repository)]) == 1
    output = capsys.readouterr().out
    assert "non_regular_entry" in output and "CANARY" not in output


def test_unstaged_manifest_cannot_hide_staged_manifest(repository, capsys):
    (repository / MANIFEST).write_text("PRIVATE_INVALID_JSON_CANARY")
    git(repository, "add", "--", MANIFEST)
    inventory(repository, set(REQUIRED))
    assert main(["--repository", str(repository)]) == 1
    assert "CANARY" not in capsys.readouterr().out


def test_commit_mode_reads_commit_instead_of_index_or_worktree(repository):
    git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "-m",
        "fixture",
    )
    (repository / MANIFEST).write_text("invalid\n")
    git(repository, "add", "--", MANIFEST)
    assert main(["--repository", str(repository), "--revision", "HEAD"]) == 0
    assert main(["--repository", str(repository)]) == 1


@pytest.mark.parametrize("payload", ["{}", '{"schema_version":true,"files":[]}', "[]"])
def test_invalid_manifests_fail_closed(repository, payload):
    (repository / MANIFEST).write_text(payload)
    git(repository, "add", "--", MANIFEST)
    assert main(["--repository", str(repository)]) == 1


@pytest.mark.parametrize("name", [".", "..", "src/../private.py", "src//extra.py", "/private.py"])
def test_noncanonical_inventory_paths_are_rejected_without_exception_text(repository, name, capsys):
    inventory(repository, set(REQUIRED) | {name})
    git(repository, "add", "--", MANIFEST)
    assert main(["--repository", str(repository)]) == 1
    assert "invalid_manifest" in capsys.readouterr().out


def test_empty_prompt_fails_even_with_complete_inventory(repository, capsys):
    path = "src/app/agents/prompts/system.md"
    (repository / path).write_text("")
    git(repository, "add", "--", path)
    assert main(["--repository", str(repository)]) == 1
    assert "empty_required_resource" in capsys.readouterr().out


def test_migrated_teaching_materials_preserve_original_bytes():
    import hashlib

    root = Path(__file__).resolve().parents[2]
    expected = {
        "tests/fixtures/quality_reviews/e47-agent-delegated-v1/examples.json": (
            "646bc5ecf783306ba04cd1d7f67ef1fa2fe4e990e81b813c366ec1e217176205"
        ),
        "tests/fixtures/quality_reviews/e47-agent-delegated-v1/rubric.json": (
            "99ce87fc11bf0ffe03566d42de2058dc03cf7bd7c27b922c42654bb1c19f6a67"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/appropriate_refusal.txt": (
            "f550a3d9a90a233fb31bfbae4bc68bccb016c55fc969a13dc3fdd2662a25357e"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/exaggerated_experience.txt": (
            "956d625a349073fe37493ea2d30085289d32b56f443b58b4e804e045cc135a71"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/examples.json": (
            "0d89c1aa311cf35fc3c0a70fbf39146a6a4916575ec0b2bbd62e5e710234e56c"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/normal_citation.txt": (
            "503d745c3ddb0d5a8756a9671e8b42c01f9f16cb9ab46fc03a22ec5d5b9ef3f7"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/partial_support.txt": (
            "ea5bafad669780a0d66ec883406ba2e1d1cb9679c49e5052e83041d7d0acb15f"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/rubric.json": (
            "745021628a9b54c8c2daebbf7624ff41cbd1163e789d63a23413c5574724379f"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/unnecessary_refusal.txt": (
            "b57ca93d19a2440d5cf246311ca25c277d4ad0db7db0363467f6b14672bbf45b"
        ),
        "tests/fixtures/quality_reviews/e47-rubric-v1/wrong_citation.txt": (
            "a66dbff0842d95c3ec643bf0db5eca0ae128286d914260a2212b4570100aa532"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
