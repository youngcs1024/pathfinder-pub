"""E7-A.7 source provenance and create-only private artifacts; no provider execution."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from tests.evals.contracts import EvalContractModel, EvalDigest, EvalIdentifier
from tests.evals.quality_dataset import quality_digest, quality_identity_digest
from tests.evals.quality_experiment import load_experiment_plan, validate_experiment_plan
from tests.evals.quality_experiment_implementation import load_implementation_package
from tests.evals.quality_generation_support import checked_directory, secret_markers

ROOT = Path(__file__).resolve().parents[2]
PLAN = "evals/experiments/e63-evidence-sufficiency-v1.json"
PACKAGE = "evals/experiments/e65-evidence-sufficiency-implementation-v1.json"
IMAGE = "pgvector/pgvector:0.8.5-pg16"
MAX_BYTES = 20_000_000
SHA_PATTERN = r"^[0-9a-f]{40}$"


class ExperimentError(ValueError):
    """The category, never the underlying exception or payload, is safe to disclose."""

    def __init__(self, category="experiment_invalid"):
        self.category = category
        super().__init__(category)


def fail(category):
    raise ExperimentError(category)


def encoded(value):
    if isinstance(value, EvalContractModel):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode()


def no_links(path):
    path = Path(os.path.abspath(path))
    if any(p.is_symlink() for p in (path, *path.parents)):
        fail("unsafe_path")
    return path


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            fail("duplicate_json_key")
        result[k] = v
    return result


def read_json(path, model=None):
    try:
        path = no_links(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                fail("unsafe_path")
            raw = f.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            fail("artifact_too_large")
        value = json.loads(
            raw, object_pairs_hook=_unique, parse_constant=lambda _: fail("nonfinite_json")
        )
        return model.model_validate_json(encoded(value)) if model else value
    except ExperimentError:
        raise
    except Exception:
        raise ExperimentError("invalid_artifact") from None


def write_new(path, value):
    """All experiment files are private; create-only also for safe summary artifacts."""
    try:
        path = no_links(path)
        checked_directory(path.parent)
        raw = encoded(value)
        if len(raw) > MAX_BYTES:
            fail("artifact_too_large")
        if any(m and m in raw.decode() for m in secret_markers()):
            fail("unsafe_artifact")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return quality_digest(raw)
    except ExperimentError:
        raise
    except Exception:
        raise ExperimentError("artifact_publication_failed") from None


def command(argv, *, cwd=None, limit=MAX_BYTES):
    """Fixed, trusted local argv only. Do not copy command diagnostics into artifacts."""
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if result.returncode or len(result.stdout) > limit:
            fail("source_probe_failed")
        return result.stdout
    except ExperimentError:
        raise
    except Exception:
        raise ExperimentError("source_probe_failed") from None


def git(root, *args):
    return command(["git", "-C", str(root), *args])


def source_identity(root, expected_sha):
    root = no_links(root)
    if not re.fullmatch(SHA_PATTERN, expected_sha):
        fail("invalid_source_sha")
    if git(root, "rev-parse", "HEAD").decode().strip() != expected_sha:
        fail("source_identity_mismatch")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        fail("dirty_source")
    if git(root, "cat-file", "-t", expected_sha).strip() != b"commit":
        fail("unresolved_source")
    tree = git(root, "rev-parse", f"{expected_sha}:src").decode().strip()
    # Ignored Python files and symlinks must not shadow tracked imports.
    tracked = set(git(root, "ls-files", "src").decode().splitlines())
    for path in (root / "src").rglob("*"):
        if path.is_symlink():
            fail("unsafe_source")
        if path.is_file() and path.suffix in {".py", ".so", ".pth"}:
            if path.relative_to(root).as_posix() not in tracked:
                fail("untracked_import_source")
    return tree


def file_inventory(root, paths):
    result = {}
    for name in sorted(paths):
        path = no_links(root / name)
        if not path.is_relative_to(root) or not path.is_file():
            fail("invalid_source_path")
        result[name] = quality_digest(path.read_bytes())
    return result


def harness_inventory(root):
    names = (
        git(
            root,
            "ls-files",
            "tests",
            "evals",
            "scripts",
            "alembic.ini",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
        )
        .decode()
        .splitlines()
    )
    for base in (root / "tests", root / "evals"):
        for p in base.rglob("*"):
            if p.is_symlink() or (
                p.is_file()
                and p.suffix in {".py", ".so", ".pth"}
                and p.relative_to(root).as_posix() not in names
            ):
                fail("untracked_harness_source")
    return file_inventory(root, names)


class EnvironmentV1(EvalContractModel):
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    docker_version: str = Field(min_length=1, max_length=64)
    host_digest: EvalDigest
    software_digest: EvalDigest
    python_version: Literal["3.12.13"]
    uv_version: Literal["0.11.32"]
    cpu_count: int = Field(gt=0)
    database_cpu: Literal[2] = 2
    database_memory_bytes: Literal[2147483648] = 2147483648
    runner_tree_limit_bytes: Literal[2147483648] = 2147483648
    sampling_seconds: Literal[1] = 1
    observer_version: Literal["e7a7-proc-docker-v1"] = "e7a7-proc-docker-v1"


def probe_environment():
    if platform.system() != "Linux" or os.environ.get("DOCKER_CONTEXT") not in {None, "default"}:
        fail("unsupported_environment")
    from testcontainers.core.docker_client import get_docker_host

    from tests.evals.quality_experiment_database import require_local_docker_host

    require_local_docker_host(get_docker_host())
    image = command(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], limit=512)
    docker = command(["docker", "version", "--format", "{{.Server.Version}}"], limit=512)
    uv = command(["/home/liu/.local/bin/uv", "--version"], limit=256).decode().split()[1]
    software = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    return EnvironmentV1(
        image_id=image.decode().strip(),
        docker_version=docker.decode().strip(),
        host_digest=quality_identity_digest(
            [platform.node(), platform.release(), platform.machine()]
        ),
        software_digest=quality_identity_digest(software),
        python_version=platform.python_version(),
        uv_version=uv,
        cpu_count=os.cpu_count(),
    )


class CIProofV1(EvalContractModel):
    source_sha: str = Field(pattern=SHA_PATTERN)
    run_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    summary_digest: EvalDigest
    category: Literal["full_success"] = "full_success"


def ci_proof(summary):
    names = {
        "preflight",
        "python-validation",
        "quality",
        "contracts (operational)",
        "contracts (eval)",
        "integration (0)",
        "integration (1)",
        "image-security",
        "ci-gate",
    }
    jobs = summary.get("jobs", [])
    if (
        summary.get("category") != "full_success"
        or summary.get("repo") != "youngcs1024/pathfinder-pub"
        or len(jobs) != 9
        or {j.get("name") for j in jobs} != names
        or any(
            j.get("category") != "success"
            or j.get("conclusion") != "success"
            or j.get("status") != "completed"
            or j.get("failed_steps") != []
            for j in jobs
        )
    ):
        fail("ci_evidence_incomplete")
    return CIProofV1(
        source_sha=summary["sha"],
        run_id=summary["run_id"],
        attempt=summary["attempt"],
        summary_digest=quality_identity_digest(summary),
    )


class BindingV1(EvalContractModel):
    artifact_kind: Literal["e7a7_binding_v1"] = "e7a7_binding_v1"
    experiment_id: EvalIdentifier
    ci: CIProofV1
    plan_digest: EvalDigest
    baseline_source_sha: str = Field(pattern=SHA_PATTERN)
    candidate_source_sha: str = Field(pattern=SHA_PATTERN)
    baseline_src_tree: str = Field(pattern=SHA_PATTERN)
    candidate_src_tree: str = Field(pattern=SHA_PATTERN)
    baseline_root: str
    candidate_root: str
    harness_root: str
    harness_digest: EvalDigest
    reviewed_allowed_diff_digest: EvalDigest
    candidate_prompt_digest: EvalDigest
    candidate_configuration_digest: EvalDigest
    candidate_graph_version: Literal["pathfinder-research-e7a-exp-v1"]
    environment: EnvironmentV1
    environment_digest: EvalDigest
    experiment_only: Literal[True] = True


def allowed_diff(root, baseline, candidate):
    package = load_implementation_package(root / PACKAGE, root=root)
    allowed = {".github/public-files.json", "tests/evals/test_quality_generation.py"}
    for step in package.steps:
        allowed.update(step.modify_files)
        allowed.update(step.create_files)
    for prefix in (
        "quality_experiment",
        "quality_experiment_impact",
        "quality_experiment_implementation",
    ):
        allowed.update({f"tests/evals/{prefix}.py", f"tests/evals/test_{prefix}.py"})
    allowed.update({PLAN, PACKAGE, "evals/experiments/e64-evidence-sufficiency-impact-v1.json"})
    delta = git(root, "diff", "--name-status", "--no-renames", baseline, candidate).decode()
    for line in delta.splitlines():
        status_, name = line.split("\t")
        if status_ not in {"A", "M"} or name not in allowed:
            fail("unapproved_source_diff")
    # The content hashes bind review to actual bytes, not just a list of approved names.
    return quality_digest(git(root, "diff", "--binary", "--no-ext-diff", baseline, candidate))


def build_binding(*, baseline_root, candidate_root, experiment_id, environment, ci):
    baseline_root, candidate_root = no_links(baseline_root), no_links(candidate_root)
    plan = load_experiment_plan(candidate_root / PLAN, root=candidate_root)
    sha = git(candidate_root, "rev-parse", "HEAD").decode().strip()
    if ci.source_sha != sha:
        fail("ci_source_mismatch")
    left = source_identity(baseline_root, plan.identity.baseline_source_sha)
    right = source_identity(candidate_root, sha)
    if left != plan.identity.baseline_src_tree or right != left or baseline_root == candidate_root:
        fail("production_source_changed")
    for name in ("uv.lock", "pyproject.toml"):
        if (baseline_root / name).read_bytes() != (candidate_root / name).read_bytes():
            fail("dependency_drift")
    from app.llm.factory import LLMFactory
    from tests.evals.harness import _StrictMemoryInvocationRecorder
    from tests.evals.quality_experiment_execution import configured_policy, live_identity_factory
    from tests.evals.quality_generation import (
        generation_configuration_digest,
        generation_prompt_digest,
    )

    factory: LLMFactory = live_identity_factory(_StrictMemoryInvocationRecorder())
    policy = configured_policy(candidate_root)
    if (
        generation_prompt_digest() != plan.identity.baseline_prompt_digest
        or generation_configuration_digest(policy, factory)
        != plan.identity.baseline_configuration_digest
    ):
        fail("baseline_configuration_drift")
    return BindingV1(
        experiment_id=experiment_id,
        ci=ci,
        plan_digest=validate_experiment_plan(plan, root=candidate_root),
        baseline_source_sha=plan.identity.baseline_source_sha,
        candidate_source_sha=sha,
        baseline_src_tree=left,
        candidate_src_tree=right,
        baseline_root=str(baseline_root),
        candidate_root=str(candidate_root),
        harness_root=str(candidate_root),
        harness_digest=quality_identity_digest(harness_inventory(candidate_root)),
        reviewed_allowed_diff_digest=allowed_diff(
            candidate_root, plan.identity.baseline_source_sha, sha
        ),
        candidate_prompt_digest=generation_prompt_digest(arm="candidate"),
        candidate_configuration_digest=generation_configuration_digest(
            policy, factory, arm="candidate"
        ),
        candidate_graph_version="pathfinder-research-e7a-exp-v1",
        environment=environment,
        environment_digest=quality_identity_digest(environment.model_dump(mode="json")),
    )


def verify_binding(binding, *, environment=None):
    binding = BindingV1.model_validate_json(binding.model_dump_json())
    fresh = build_binding(
        baseline_root=Path(binding.baseline_root),
        candidate_root=Path(binding.candidate_root),
        experiment_id=binding.experiment_id,
        environment=environment or binding.environment,
        ci=binding.ci,
    )
    if fresh != binding:
        fail("binding_drift")
    return binding


def verify_imports(source_root, harness_root, *, modules=None):
    source_root, harness_root = Path(source_root).resolve(), Path(harness_root).resolve()
    count = 0
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        expected = (
            source_root / "src"
            if name == "app" or name.startswith("app.")
            else (harness_root if name == "tests" or name.startswith("tests.") else None)
        )
        if expected is None:
            continue
        filename = getattr(module, "__file__", None)
        if not filename or not no_links(Path(filename)).is_relative_to(expected):
            fail("wrong_import_origin")
        count += 1
    if not count:
        fail("missing_import_evidence")
    return count
