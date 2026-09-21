"""Real local Git/import provenance and private create-only boundaries; no network."""

import subprocess
from types import SimpleNamespace

import pytest

from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.quality_experiment_binding import (
    ExperimentError,
    ci_proof,
    file_inventory,
    harness_inventory,
    read_json,
    source_identity,
    verify_imports,
    write_new,
)


def local_git(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "src/app").mkdir(parents=True)
    (root / "src/app/__init__.py").write_text("VALUE = 1\n")
    (root / ".gitignore").write_text("ignored.py\n")

    def run(*args):
        return (
            subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

    run("init")
    run("add", ".")
    run(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return root, run("rev-parse", "HEAD"), run


def test_real_git_objects_and_clean_source(tmp_path):
    root, sha, run = local_git(tmp_path)
    assert source_identity(root, sha) == run("rev-parse", "HEAD:src")
    assert file_inventory(root, ["src/app/__init__.py"])
    with pytest.raises(ExperimentError, match="source_identity_mismatch"):
        source_identity(root, "a" * 40)


@pytest.mark.parametrize("kind", ["tracked", "untracked", "ignored", "symlink"])
def test_dirty_ignored_and_symlink_imports_fail_closed(tmp_path, kind):
    root, sha, _ = local_git(tmp_path)
    if kind == "tracked":
        (root / "src/app/__init__.py").write_text("VALUE = 2\n")
    elif kind == "untracked":
        (root / "src/app/new.py").write_text("VALUE = 2\n")
    elif kind == "ignored":
        (root / "src/app/ignored.py").write_text("VALUE = 2\n")
    else:
        (root / "src/app/ignored.py").symlink_to(root / "src/app/__init__.py")
    with pytest.raises(ExperimentError):
        source_identity(root, sha)


def test_harness_hash_covers_content_and_rejects_shadow_import(tmp_path):
    root, _, run = local_git(tmp_path)
    (root / "tests").mkdir()
    (root / "evals").mkdir()
    (root / "tests/__init__.py").write_text("")
    run("add", "tests/__init__.py")
    before = harness_inventory(root)
    (root / "tests/__init__.py").write_text("VALUE=1\n")
    assert before != harness_inventory(root)
    (root / "tests/ignored.py").write_text("")
    with pytest.raises(ExperimentError, match="untracked_harness_source"):
        harness_inventory(root)


def test_actual_module_roots_are_checked_not_just_declared_sha(tmp_path):
    source = tmp_path / "source"
    harness = tmp_path / "harness"
    modules = {
        "app": SimpleNamespace(__file__=str(source / "src/app/__init__.py")),
        "tests": SimpleNamespace(__file__=str(harness / "tests/__init__.py")),
    }
    assert verify_imports(source, harness, modules=modules) == 2
    modules["app"].__file__ = str(harness / "src/app/__init__.py")
    with pytest.raises(ExperimentError, match="wrong_import_origin"):
        verify_imports(source, harness, modules=modules)


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "not-json"])
def test_ambiguous_or_nonfinite_artifacts_rejected(tmp_path, payload):
    path = tmp_path / "input.json"
    path.write_text(payload)
    with pytest.raises(ExperimentError):
        read_json(path)


def test_create_only_artifacts_keep_prior_bytes_and_reject_links(tmp_path):
    tmp_path.chmod(0o700)
    path = tmp_path / "result.json"
    write_new(path, {"count": 1})
    assert path.stat().st_mode & 0o777 == 0o600
    original = path.read_bytes()
    with pytest.raises(ExperimentError):
        write_new(path, {"count": 2})
    assert path.read_bytes() == original
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ExperimentError, match="unsafe_path"):
        read_json(link)


def ci_summary():
    names = [
        "preflight",
        "python-validation",
        "quality",
        "contracts (operational)",
        "contracts (eval)",
        "integration (0)",
        "integration (1)",
        "image-security",
        "ci-gate",
    ]
    return {
        "repo": "youngcs1024/pathfinder-pub",
        "sha": "a" * 40,
        "run_id": 1,
        "attempt": 1,
        "category": "full_success",
        "jobs": [
            {
                "name": n,
                "category": "success",
                "conclusion": "success",
                "status": "completed",
                "failed_steps": [],
            }
            for n in names
        ],
    }


def test_complete_ci_proof_pins_commit_attempt_and_summary():
    proof = ci_proof(ci_summary())
    assert proof.source_sha == "a" * 40 and proof.attempt == 1 and proof.summary_digest


@pytest.mark.parametrize("kind", ["old_eight", "skip", "failure", "duplicate", "foreign", "fast"])
def test_incomplete_ci_never_authorizes_binding(kind):
    summary = ci_summary()
    if kind == "old_eight":
        summary["jobs"].pop(1)
    elif kind == "skip":
        summary["jobs"][1]["conclusion"] = "skipped"
    elif kind == "failure":
        summary["jobs"][1]["failed_steps"] = ["test"]
    elif kind == "duplicate":
        summary["jobs"][1] = summary["jobs"][0]
    elif kind == "foreign":
        summary["repo"] = "other/repo"
    else:
        summary["category"] = "docs_success"
    with pytest.raises(ExperimentError, match="ci_evidence_incomplete"):
        ci_proof(summary)


def test_source_errors_do_not_disclose_stderr_or_payload(tmp_path):
    with pytest.raises(ExperimentError) as error:
        source_identity(tmp_path / "private-body-canary", "a" * 40)
    assert "private-body-canary" not in str(error.value)


def test_binding_rechecks_git_and_rejects_changed_digest(tmp_path, monkeypatch):
    from tests.evals import quality_experiment_binding as binding
    from tests.evals.quality_experiment_execution import configured_policy, live_identity_factory
    from tests.evals.quality_generation import (
        generation_configuration_digest,
        generation_prompt_digest,
    )
    from tests.evals.test_quality_experiment_execution import binding_fixture

    baseline, sha, git = local_git(tmp_path)
    for name in ("uv.lock", "pyproject.toml"):
        (baseline / name).write_text("# local provenance fixture\n")
    git("add", "uv.lock", "pyproject.toml")
    git(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "lock",
    )
    sha = git("rev-parse", "HEAD")
    candidate = tmp_path / "candidate"
    subprocess.run(["git", "clone", "--quiet", str(baseline), str(candidate)], check=True)
    original = binding.load_experiment_plan(binding.ROOT / binding.PLAN)
    policy = configured_policy()
    identity = original.identity.model_copy(
        update={
            "baseline_source_sha": sha,
            "baseline_src_tree": git("rev-parse", "HEAD:src"),
            "baseline_prompt_digest": generation_prompt_digest(),
            "baseline_configuration_digest": generation_configuration_digest(
                policy, live_identity_factory(_StrictMemoryInvocationRecorder())
            ),
        }
    )
    plan = original.model_copy(update={"identity": identity})
    environment = binding_fixture(tmp_path).environment
    monkeypatch.setattr(
        "tests.evals.quality_experiment_execution.configured_policy", lambda *a: policy
    )
    monkeypatch.setattr(binding, "load_experiment_plan", lambda *a, **kw: plan)
    monkeypatch.setattr(binding, "validate_experiment_plan", lambda *a, **kw: "sha256:" + "3" * 64)
    monkeypatch.setattr(binding, "allowed_diff", lambda *a: "sha256:" + "4" * 64)
    ci = ci_proof(ci_summary()).model_copy(update={"source_sha": sha})
    value = binding.build_binding(
        baseline_root=baseline,
        candidate_root=candidate,
        experiment_id="binding_contract",
        environment=environment,
        ci=ci,
    )
    assert binding.verify_binding(value) == value
    with pytest.raises(ExperimentError, match="binding_drift"):
        binding.verify_binding(value.model_copy(update={"harness_digest": "sha256:" + "5" * 64}))
    (candidate / "src/app/__init__.py").write_text("VALUE = 99\n")
    with pytest.raises(ExperimentError, match="dirty_source"):
        binding.verify_binding(value)


@pytest.mark.parametrize("version", [1, 2])
def test_binding_version_roundtrip_and_tampering(tmp_path, version):
    from tests.evals.quality_experiment_binding import SHARED_CORRECTION, BindingV2, read_binding
    from tests.evals.test_quality_experiment_execution import binding_fixture

    value = binding_fixture(tmp_path)
    payload = value.model_dump(mode="json")
    if version == 2:
        payload.update(
            artifact_kind="e7a7_binding_v2",
            original_baseline_source_sha=value.baseline_source_sha,
            baseline_source_sha=SHARED_CORRECTION,
            shared_correction_source_sha=SHARED_CORRECTION,
            shared_correction_diff_digest="sha256:" + "9" * 64,
        )
        value = BindingV2.model_validate_json(__import__("json").dumps(payload))
    write_new(tmp_path / "binding.json", value)
    assert read_binding(tmp_path / "binding.json") == value
    payload["artifact_kind"] = "e7a7_binding_v99"
    write_new(tmp_path / "forged.json", payload)
    with pytest.raises(ExperimentError, match="unsupported_binding_version"):
        read_binding(tmp_path / "forged.json")
    payload["artifact_kind"] = "e7a7_binding_v1"
    payload["shared_correction_source_sha"] = "a" * 40
    write_new(tmp_path / "mixed.json", payload)
    with pytest.raises(ExperimentError, match="invalid_artifact"):
        read_binding(tmp_path / "mixed.json")


def test_shared_correction_cannot_accept_an_original_source_checkout(tmp_path, monkeypatch):
    from tests.evals import quality_experiment_binding as binding
    from tests.evals.test_quality_experiment_execution import binding_fixture

    root, sha, _ = local_git(tmp_path)
    original = binding.load_experiment_plan(binding.ROOT / binding.PLAN)
    monkeypatch.setattr(binding, "load_experiment_plan", lambda *a, **kw: original)
    with pytest.raises(ExperimentError, match="source_identity_mismatch"):
        binding.build_binding(
            baseline_root=root,
            candidate_root=root,
            experiment_id="wrong_shared_source",
            environment=binding_fixture(tmp_path).environment,
            ci=ci_proof(ci_summary()).model_copy(update={"source_sha": sha}),
            shared_correction=True,
        )


def test_corrected_binding_rejects_unequal_real_production_trees(tmp_path, monkeypatch):
    from tests.evals import quality_experiment_binding as binding
    from tests.evals.test_quality_experiment_execution import binding_fixture

    baseline, sha, run = local_git(tmp_path)
    candidate = tmp_path / "candidate"
    subprocess.run(["git", "clone", "--quiet", str(baseline), str(candidate)], check=True)
    (baseline / "src/app/__init__.py").write_text("VALUE = 2\n")
    run("add", "src/app/__init__.py")
    run(
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "changed",
    )
    monkeypatch.setattr(binding, "SHARED_CORRECTION", run("rev-parse", "HEAD"))
    original = binding.load_experiment_plan(binding.ROOT / binding.PLAN)
    monkeypatch.setattr(binding, "load_experiment_plan", lambda *a, **kw: original)
    with pytest.raises(ExperimentError, match="production_source_changed"):
        binding.build_binding(
            baseline_root=baseline,
            candidate_root=candidate,
            experiment_id="unequal_shared_sources",
            environment=binding_fixture(tmp_path).environment,
            ci=ci_proof(ci_summary()).model_copy(update={"source_sha": sha}),
            shared_correction=True,
        )
