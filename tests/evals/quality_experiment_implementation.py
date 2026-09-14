"""E6.5 offline implementation package integrity, never runtime readiness.

The frozen package is a reviewed task assignment, not a configurable execution
policy. Future file names need not exist yet. Registration never creates them,
resolves Git, loads private evidence or authorizes a later step or live run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from tests.evals.contracts import EvalContractModel, EvalIdentifier
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_impact import load_impact_review, validate_impact_review

ROOT = Path(__file__).resolve().parents[2]
MAX_PACKAGE_BYTES = 131_072
PACKAGE_DIGEST = "sha256:64aa45e24e93c458fb5f03bba5045fd075c97ef327bca9c05e28a20e8747620d"
STEPS = tuple(f"E7-A.{n}" for n in range(1, 8))
CATEGORIES = {
    "invalid_implementation_arguments",
    "invalid_implementation_path",
    "invalid_implementation_package",
    "implementation_source_mismatch",
    "implementation_coverage_mismatch",
    "implementation_policy_mismatch",
}


class ImplementationPackageError(ValueError):
    """Only fixed categories may cross the CLI boundary."""


class ImplementationStepV1(EvalContractModel):
    step: Literal["E7-A.1", "E7-A.2", "E7-A.3", "E7-A.4", "E7-A.5", "E7-A.6", "E7-A.7"]
    depends_on: tuple[str, ...] = Field(min_length=1, max_length=1)
    read_files: tuple[str, ...] = Field(min_length=1, max_length=16)
    modify_files: tuple[str, ...] = Field(max_length=16)
    create_files: tuple[str, ...] = Field(min_length=1, max_length=16)
    deliverables: tuple[EvalIdentifier, ...] = Field(min_length=1, max_length=16)
    failure_paths: tuple[EvalIdentifier, ...] = Field(min_length=1, max_length=16)
    acceptance_commands: tuple[str, ...] = Field(min_length=1, max_length=2)
    test_files: tuple[str, ...] = Field(min_length=1, max_length=16)
    handoffs: tuple[EvalIdentifier, ...] = Field(max_length=7)


class QualityExperimentImplementationV1(EvalContractModel):
    artifact_kind: Literal["quality_experiment_implementation_v1"]
    package_id: Literal["e65_evidence_sufficiency_v1"]
    source_commit: Literal["662073d4965d2b85d25b74b4d1c6e06d647bcfcb"]
    baseline_source_sha: Literal["6ee09703fd19dd3f68813bb40f2b314fba512995"]
    baseline_src_tree: Literal["99d2c2195add7c2644fe4f8efd7f26d1337faa15"]
    plan_path: Literal["evals/experiments/e63-evidence-sufficiency-v1.json"]
    plan_digest: Literal["sha256:7e7ae136f40a1eb2f7d292fa0f6ddb191fbd2e0b9e95726e8f0a7755210284ff"]
    impact_path: Literal["evals/experiments/e64-evidence-sufficiency-impact-v1.json"]
    impact_digest: Literal[
        "sha256:44aac606cb913af1afdb032615b8e381684f1c4d6c401017cf1077856c24794d"
    ]
    selected_branch: Literal["A"]
    experiment_only: Literal[True]
    candidate_graph_version: Literal["pathfinder-research-e7a-exp-v1"]
    candidate_output_contract: Literal["e7a_research_output_v1"]
    inherited_rules: tuple[EvalIdentifier, ...] = Field(min_length=9, max_length=9)
    steps: tuple[ImplementationStepV1, ...] = Field(min_length=7, max_length=7)
    shared_modify_files: tuple[Literal[".github/public-files.json"]]
    file_policy: Literal["exact_paths_create_at_own_step_no_future_placeholders"]
    authorization: Literal["package_registration_only_each_step_separately_authorized"]
    manual_gates: tuple[EvalIdentifier, ...] = Field(min_length=4, max_length=4)
    forbidden: tuple[EvalIdentifier, ...] = Field(min_length=10, max_length=10)
    production_followup: tuple[EvalIdentifier, ...] = Field(min_length=5, max_length=5)
    verification_limit: Literal["implementation_registration_only"]
    execution_readiness: Literal["NOT_VERIFIED"]
    candidate_execution: Literal["NOT_RUN"]
    adoption: Literal["NOT_RUN"]
    deployment: Literal["NOT_RUN"]
    next_step: Literal["E7-A.1"]


def _no_links(path: Path) -> None:
    for part in (*path.absolute().parents, path.absolute()):
        if part.is_symlink():
            raise ImplementationPackageError("invalid_implementation_path")


def _reference(root: Path, name: str, *, must_exist: bool = True) -> Path:
    relative = PurePosixPath(name)
    if (
        not name
        or not relative.parts
        or relative.is_absolute()
        or name != relative.as_posix()
        or any(part in {".", ".."} for part in relative.parts)
        or any(char in name for char in "\\:*?[]\x00")
        or relative.parts[0] not in {"src", "tests", "evals", ".github"}
    ):
        raise ImplementationPackageError("invalid_implementation_path")
    path = root / name
    _no_links(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ImplementationPackageError("invalid_implementation_path")
    if (must_exist or path.exists()) and not path.is_file():
        raise ImplementationPackageError("invalid_implementation_path")
    return path


def validate_implementation_package(
    package: QualityExperimentImplementationV1, *, root: Path = ROOT
) -> str:
    """Validate the assignment and bound public sources, not future implementation."""
    try:
        original = package.model_dump(mode="json")
        package = QualityExperimentImplementationV1.model_validate_json(package.model_dump_json())
        if quality_identity_digest(original) != quality_identity_digest(
            package.model_dump(mode="json")
        ):
            raise ImplementationPackageError("invalid_implementation_package")
        _reference(root, package.plan_path)
        review_path = _reference(root, package.impact_path)
        try:
            review = load_impact_review(review_path, root=root)
            impact_digest = validate_impact_review(review, root=root)
        except Exception:
            raise ImplementationPackageError("implementation_source_mismatch") from None
        if (
            impact_digest != package.impact_digest
            or review.plan_path != package.plan_path
            or review.plan_digest != package.plan_digest
            or review.baseline_source_sha != package.baseline_source_sha
            or review.baseline_src_tree != package.baseline_src_tree
            or review.candidate_graph_version != package.candidate_graph_version
            or review.candidate_output_contract != package.candidate_output_contract
        ):
            raise ImplementationPackageError("implementation_source_mismatch")
        if tuple(row.step for row in package.steps) != STEPS:
            raise ImplementationPackageError("implementation_coverage_mismatch")
        created: set[str] = set()
        handoffs: list[str] = []
        for index, row in enumerate(package.steps):
            if row.depends_on != ((STEPS[index - 1] if index else "E6.5"),):
                raise ImplementationPackageError("implementation_coverage_mismatch")
            for paths in (row.read_files, row.modify_files, row.create_files, row.test_files):
                if len(set(paths)) != len(paths):
                    raise ImplementationPackageError("implementation_coverage_mismatch")
            for name in row.read_files:
                _reference(root, name, must_exist=name not in created)
            for name in row.modify_files:
                _reference(root, name, must_exist=name not in created)
            for name in (*row.modify_files, *row.create_files):
                if not name.startswith(
                    ("tests/evals/", "tests/integration/db/", "evals/experiments/")
                ):
                    raise ImplementationPackageError("invalid_implementation_path")
            for name in row.create_files:
                _reference(root, name, must_exist=False)
                if name in created or name in row.modify_files:
                    raise ImplementationPackageError("implementation_coverage_mismatch")
            if not set(row.test_files).issubset(set(row.modify_files) | set(row.create_files)):
                raise ImplementationPackageError("implementation_coverage_mismatch")
            for name in row.test_files:
                if not Path(name).name.startswith("test_") or not name.endswith(".py"):
                    raise ImplementationPackageError("implementation_coverage_mismatch")
            required = ("make test-eval-contracts",)
            if any(name.startswith("tests/integration/") for name in row.test_files):
                required += ("make test-integration-core",)
            if row.acceptance_commands != required:
                raise ImplementationPackageError("implementation_coverage_mismatch")
            created.update(row.create_files)
            handoffs.extend(row.handoffs)
        if len(handoffs) != len(set(handoffs)) or set(handoffs) != set(review.e65_handoff):
            raise ImplementationPackageError("implementation_coverage_mismatch")
        for name in package.shared_modify_files:
            _reference(root, name)
        digest = quality_identity_digest(package.model_dump(mode="json"))
        # The package fixes concrete assignments; even valid-looking scope changes
        # require a newly reviewed version. Never silently rewrite this v1 digest.
        if digest != PACKAGE_DIGEST:
            raise ImplementationPackageError("implementation_policy_mismatch")
        return digest
    except ImplementationPackageError:
        raise
    except Exception:
        raise ImplementationPackageError("invalid_implementation_package") from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImplementationPackageError("invalid_implementation_package")
        result[key] = value
    return result


def load_implementation_package(
    path: Path, *, root: Path = ROOT
) -> QualityExperimentImplementationV1:
    try:
        _no_links(path)
        with path.open("rb") as stream:
            content = stream.read(MAX_PACKAGE_BYTES + 1)
        if len(content) > MAX_PACKAGE_BYTES:
            raise ImplementationPackageError("invalid_implementation_package")
        raw = json.loads(content, object_pairs_hook=_unique_object)
        package = QualityExperimentImplementationV1.model_validate_json(json.dumps(raw))
        if quality_identity_digest(raw) != quality_identity_digest(package.model_dump(mode="json")):
            raise ImplementationPackageError("invalid_implementation_package")
        validate_implementation_package(package, root=root)
        return package
    except ImplementationPackageError:
        raise
    except Exception:
        raise ImplementationPackageError("invalid_implementation_package") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ImplementationPackageError("invalid_implementation_arguments")


def main(argv=None) -> int:
    try:
        parser = _Parser(description=__doc__, allow_abbrev=False)
        commands = parser.add_subparsers(dest="command", required=True)
        validate = commands.add_parser("validate", allow_abbrev=False)
        validate.add_argument("--package", type=Path, required=True)
        args = parser.parse_args(argv)
        package = load_implementation_package(args.package)
        print(
            json.dumps(
                {
                    "category": "implementation_package_valid",
                    "digest": quality_identity_digest(package.model_dump(mode="json")),
                    "steps": len(package.steps),
                    "verification_limit": package.verification_limit,
                    "execution_readiness": package.execution_readiness,
                    "candidate_execution": package.candidate_execution,
                    "adoption": package.adoption,
                    "deployment": package.deployment,
                }
            )
        )
        return 0
    except ImplementationPackageError as error:
        category = str(error)
        if category not in CATEGORIES:
            category = "invalid_implementation_package"
        print(json.dumps({"category": category}))
        return 1
    except Exception:
        print(json.dumps({"category": "invalid_implementation_package"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
