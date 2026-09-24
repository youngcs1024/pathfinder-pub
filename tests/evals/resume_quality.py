"""R7.1 prepare/execute/report CLI. Fake/off only in this preparation increment."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from langsmith import tracing_context

from tests.evals.product_acceptance_contracts import (
    bound_files,
    new_directory,
    publish,
    read_private_json,
    require,
)
from tests.evals.product_acceptance_environment import (
    IMAGE,
    OwnedProductDatabase,
    docker,
    local_docker,
)
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import ROOT, file_inventory, git, read_json
from tests.evals.resume_quality_contracts import (
    ComparisonBudget,
    ComparisonDataset,
    review_template,
)

DATASET = ROOT / "evals/datasets/resume_quality_v1/dataset.json"


def load_dataset():
    return read_json(DATASET, ComparisonDataset)


def prepare(root):
    dataset = load_dataset()
    manifest = {
        "artifact_kind": "resume_quality_manifest_v1",
        "mode": "fake",
        "trace_mode": "off",
        "source_sha": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "files": file_inventory(ROOT, bound_files()),
        "dataset": dataset.model_dump(mode="json"),
        "dataset_digest": quality_identity_digest(dataset.model_dump(mode="json")),
        "budget": ComparisonBudget().model_dump(mode="json"),
        "live_permission": None,
    }
    root = new_directory(root)
    publish(root / "manifest.json", manifest)
    publish(root / "review-template.json", review_template(manifest["dataset_digest"]))
    return manifest


def verify(manifest):
    require(manifest["artifact_kind"] == "resume_quality_manifest_v1", "manifest_version_changed")
    require(manifest["mode"] == "fake" and manifest["trace_mode"] == "off", "live_not_authorized")
    require(manifest["live_permission"] is None, "unexpected_live_permission")
    require(
        git(ROOT, "rev-parse", "HEAD").decode().strip() == manifest["source_sha"], "source_changed"
    )
    require(
        set(manifest["files"]) == set(bound_files())
        and file_inventory(ROOT, manifest["files"]) == manifest["files"],
        "source_changed",
    )
    dataset = ComparisonDataset.model_validate_json(json.dumps(manifest["dataset"]))
    require(
        dataset.material_kind == "synthetic"
        and quality_identity_digest(dataset.model_dump(mode="json")) == manifest["dataset_digest"],
        "dataset_changed",
    )
    require(dataset == load_dataset(), "dataset_changed")
    budget = ComparisonBudget.model_validate_json(json.dumps(manifest["budget"]))
    return dataset, budget


async def execute(root):
    from tests.evals.resume_quality_runtime import execute_synthetic

    manifest = read_private_json(root / "manifest.json")
    dataset, budget = verify(manifest)
    # Create-only marker: rerunning never silently creates a second billed experiment.
    publish(root / "started.json", {"source_sha": manifest["source_sha"]})
    try:
        local_docker()
        image_id = docker("image", "inspect", IMAGE, "--format", "{{.Id}}")
        db = OwnedProductDatabase(root, image_id)
        with tracing_context(enabled=False):
            async with db as url:
                report = await execute_synthetic(
                    url, root, dataset, budget, source_check=lambda: verify(manifest)
                )
        publish(
            root / "execution.json",
            {"database_stopped": db.cleanup_ok, "report_digest": quality_identity_digest(report)},
        )
        return 0 if report["status"] == "SYNTHETIC_PASS" and db.cleanup_ok else 1
    except Exception:
        publish(
            root / "execution-failed.json", {"status": "PARTIAL", "category": "execution_failed"}
        )
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "report"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            prepare(args.root)
        elif args.action == "execute":
            return asyncio.run(execute(args.root))
        else:
            report = read_private_json(args.root / "report.json")
            # Body and identifiers remain in private files, not terminal logs.
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in (
                            "status",
                            "mode",
                            "live_evidence",
                            "manual_compile_evidence",
                            "r71_status",
                        )
                    }
                )
            )
        return 0
    except Exception:
        print("R71 invalid_configuration_or_artifact", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
