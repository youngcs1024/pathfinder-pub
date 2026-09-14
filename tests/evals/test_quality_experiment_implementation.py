"""Assignment integrity, scope rejection and read-only behavior; no E7 runtime claims."""

import io
import json
import socket
import subprocess
from pathlib import Path

import pytest

from tests.evals import quality_experiment_implementation as implementation

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "evals/experiments/e65-evidence-sufficiency-implementation-v1.json"
CANARY = "private-implementation-error-canary"


@pytest.fixture
def raw():
    return json.loads(PACKAGE.read_bytes())


def write_package(tmp_path, raw):
    target = tmp_path / "package.json"
    target.write_text(json.dumps(raw))
    return target


def test_package_binds_sources_and_covers_each_handoff_without_claiming_execution():
    package = implementation.load_implementation_package(PACKAGE)
    assert implementation.validate_implementation_package(package) == implementation.PACKAGE_DIGEST
    assert len(package.steps) == 7
    assert package.next_step == "E7-A.1"
    assert package.execution_readiness == "NOT_VERIFIED"
    assert package.candidate_execution == package.adoption == package.deployment == "NOT_RUN"
    assert package.baseline_source_sha != package.source_commit
    assert len({name for step in package.steps for name in step.handoffs}) == 7
    assert all(not name.startswith("src/") for step in package.steps for name in step.modify_files)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_source_sha", "a" * 40),
        ("source_commit", "b" * 40),
        ("baseline_src_tree", "c" * 40),
        ("plan_digest", "sha256:" + "d" * 64),
        ("impact_digest", "sha256:" + "e" * 64),
        ("selected_branch", "B"),
        ("experiment_only", False),
        ("experiment_only", 1),
        ("candidate_graph_version", "pathfinder-research-v6"),
        ("candidate_output_contract", "ResearchOutputV2"),
        ("execution_readiness", "PASS"),
        ("candidate_execution", "PASS"),
        ("adoption", "PASS"),
        ("deployment", "PASS"),
        ("authorization", "execute_all_steps"),
        ("next_step", "E8"),
    ],
)
def test_identity_and_authorization_cannot_be_relabelled(raw, tmp_path, field, value):
    raw[field] = value
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.load_implementation_package(write_package(tmp_path, raw))


@pytest.mark.parametrize(
    "mutation",
    ["missing_step", "reorder", "duplicate", "dependency", "handoff_missing", "handoff_duplicate"],
)
def test_steps_and_handoffs_cannot_be_omitted_or_reordered(raw, tmp_path, mutation):
    if mutation == "missing_step":
        raw["steps"].pop()
    elif mutation == "reorder":
        raw["steps"][0], raw["steps"][1] = raw["steps"][1], raw["steps"][0]
    elif mutation == "duplicate":
        raw["steps"][1] = raw["steps"][0]
    elif mutation == "dependency":
        raw["steps"][3]["depends_on"] = ["E7-A.1"]
    elif mutation == "handoff_missing":
        raw["steps"][-1]["handoffs"].pop()
    else:
        raw["steps"][0]["handoffs"] = raw["steps"][-1]["handoffs"]
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.load_implementation_package(write_package(tmp_path, raw))


@pytest.mark.parametrize(
    "mutation",
    ["rule", "gate", "prohibition", "production", "deliverable", "failure", "test", "command"],
)
def test_valid_looking_policy_relaxation_is_rejected(raw, tmp_path, mutation):
    if mutation in {"rule", "gate", "prohibition", "production"}:
        key = {
            "rule": "inherited_rules",
            "gate": "manual_gates",
            "prohibition": "forbidden",
            "production": "production_followup",
        }[mutation]
        raw[key][0] = "relaxed"
    else:
        key = {
            "deliverable": "deliverables",
            "failure": "failure_paths",
            "test": "test_files",
            "command": "acceptance_commands",
        }[mutation]
        raw["steps"][-1][key].pop()
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.load_implementation_package(write_package(tmp_path, raw))


@pytest.mark.parametrize(
    "name",
    [
        "../outside.py",
        "/tmp/outside.py",
        "tests//evals/a.py",
        "tests/./evals/a.py",
        "tests/evals/../a.py",
        "tests/evals/*.py",
        "tests/evals/a?.py",
        "tests/evals/[ab].py",
        "C:\\outside.py",
        "docs/private.md",
        "src/app/main.py",
        "",
    ],
)
def test_write_scope_rejects_traversal_globs_private_and_production(raw, tmp_path, name):
    raw["steps"][0]["create_files"][0] = name
    with pytest.raises(
        implementation.ImplementationPackageError, match="invalid_implementation_path"
    ):
        implementation.load_implementation_package(write_package(tmp_path, raw))


def test_unregistered_but_safe_looking_file_does_not_expand_scope(raw, tmp_path):
    raw["steps"][0]["create_files"].append("tests/evals/unapproved.py")
    with pytest.raises(implementation.ImplementationPackageError, match="implementation_policy"):
        implementation.load_implementation_package(write_package(tmp_path, raw))


@pytest.mark.parametrize("existing", [False, True])
def test_future_paths_need_not_exist_but_are_never_created(tmp_path, existing):
    name = "tests/evals/future.py"
    target = tmp_path / name
    if existing:
        target.parent.mkdir(parents=True)
        target.write_text("future implementation")
    before = set(tmp_path.rglob("*"))
    assert implementation._reference(tmp_path, name, must_exist=False) == target
    assert set(tmp_path.rglob("*")) == before
    if not existing:
        with pytest.raises(implementation.ImplementationPackageError):
            implementation._reference(tmp_path, name)


@pytest.mark.parametrize("directory_link", [False, True])
def test_symlinked_future_reference_cannot_escape_root(tmp_path, directory_link):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "source.py").write_text(CANARY)
    root = tmp_path / "root"
    (root / "tests").mkdir(parents=True)
    if directory_link:
        (root / "tests/evals").symlink_to(outside, target_is_directory=True)
        name = "tests/evals/source.py"
    else:
        (root / "tests/source.py").symlink_to(outside / "source.py")
        name = "tests/source.py"
    with pytest.raises(
        implementation.ImplementationPackageError, match="invalid_implementation_path"
    ):
        implementation._reference(root, name, must_exist=False)


@pytest.mark.parametrize("directory_link", [False, True])
def test_package_symlink_is_rejected_before_reading(tmp_path, monkeypatch, directory_link):
    if directory_link:
        target = tmp_path / "linked"
        target.symlink_to(PACKAGE.parent, target_is_directory=True)
        target = target / PACKAGE.name
    else:
        target = tmp_path / "package.json"
        target.symlink_to(PACKAGE)

    def forbidden(*args, **kwargs):
        pytest.fail("symlink target must not be opened")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(
        implementation.ImplementationPackageError, match="invalid_implementation_path"
    ):
        implementation.load_implementation_package(target)


@pytest.mark.parametrize(
    "name",
    [
        "evals/experiments/e63-evidence-sufficiency-v1.json",
        "evals/experiments/e64-evidence-sufficiency-impact-v1.json",
        "evals/baselines/quality/e410-agent-v1.json",
        "uv.lock",
    ],
)
def test_actual_source_tampering_fails_closed(monkeypatch, name):
    original = Path.open
    target = ROOT / name
    injected = []

    def changed(path, *args, **kwargs):
        if path == target:
            injected.append(path)
            return io.BytesIO(b"{}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(implementation.ImplementationPackageError, match="implementation_source"):
        implementation.load_implementation_package(PACKAGE)
    assert injected == [target]


def test_missing_existing_source_is_not_treated_as_future(monkeypatch):
    target = ROOT / "src/app/agents/research_nodes.py"
    original = Path.is_file

    def missing(path):
        return False if path == target else original(path)

    monkeypatch.setattr(Path, "is_file", missing)
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.load_implementation_package(PACKAGE)


@pytest.mark.parametrize("mutation", ["root", "nested", "duplicate", "oversized", "malformed"])
def test_invalid_json_and_unknown_fields_fail_closed(raw, tmp_path, mutation):
    if mutation == "root":
        raw[CANARY] = CANARY
    elif mutation == "nested":
        raw["steps"][0][CANARY] = CANARY
    target = write_package(tmp_path, raw)
    if mutation == "duplicate":
        target.write_text('{"steps":[],"steps":[]}')
    elif mutation == "oversized":
        target.write_bytes(b" " * (implementation.MAX_PACKAGE_BYTES + 1))
    elif mutation == "malformed":
        target.write_text(CANARY)
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.load_implementation_package(target)


def test_model_copy_cannot_bypass_validation():
    package = implementation.load_implementation_package(PACKAGE)
    forged = package.model_copy(update={"experiment_only": False})
    with pytest.raises(implementation.ImplementationPackageError):
        implementation.validate_implementation_package(forged)


def test_cli_reads_only_public_sources_without_network_processes_or_writes(monkeypatch, capsys):
    original = Path.open
    reads = []

    def checked(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        assert path.is_relative_to(ROOT)
        assert path.relative_to(ROOT).parts[0] not in {"docs", ".git"}
        reads.append(path)
        return original(path, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("validator attempted a side effect")

    monkeypatch.setattr(Path, "open", checked)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert implementation.main(["validate", "--package", str(PACKAGE)]) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["category"] == "implementation_package_valid"
    assert result["execution_readiness"] == "NOT_VERIFIED"
    assert PACKAGE in reads
    assert not output.err


@pytest.mark.parametrize("kind", ["arguments", "path", "schema", "source", "exception"])
def test_cli_failure_never_echoes_private_content(raw, tmp_path, monkeypatch, capsys, kind):
    target = tmp_path / CANARY
    args = ["validate", "--package", str(target)]
    if kind == "arguments":
        args = ["execute", CANARY]
    elif kind == "schema":
        raw["authorization"] = CANARY
        target.write_text(json.dumps(raw))
    elif kind in {"source", "exception"}:
        args = ["validate", "--package", str(PACKAGE)]

        def fail(*args, **kwargs):
            if kind == "exception":
                raise implementation.ImplementationPackageError(CANARY)
            raise OSError(CANARY)

        monkeypatch.setattr(implementation, "load_implementation_package", fail)
    assert implementation.main(args) == 1
    result = capsys.readouterr()
    assert set(json.loads(result.out)) == {"category"}
    assert CANARY not in result.out + result.err
    assert not result.err
