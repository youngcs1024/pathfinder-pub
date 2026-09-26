"""Opt-in staged live R7.1 acceptance; preparation never constructs a provider."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import traceback
from pathlib import Path
from uuid import uuid4

from langsmith import tracing_context

from tests.evals.product_acceptance_contracts import (
    bound_files,
    publish,
    read_private_json,
    require,
)
from tests.evals.product_acceptance_environment import IMAGE, docker
from tests.evals.quality_experiment_binding import ROOT, file_inventory, git, read_json
from tests.evals.resume_live_contracts import LiveInputs
from tests.evals.resume_live_environment import ResumableDatabase


def verify_inputs(root, inputs):
    require(root.resolve() == Path(inputs.execution_root), "allocation_root_changed")
    require(file_inventory(root / "inputs", inputs.files) == inputs.files, "inputs_changed")


def prepare(root):
    inputs = read_json(root / "inputs.json", LiveInputs)
    verify_inputs(root, inputs)
    require(not git(ROOT, "status", "--porcelain").strip(), "source_not_committed")
    manifest = {
        "artifact_kind": "resume_live_manifest_v1",
        "trace_mode": "off",
        "inputs": inputs.model_dump(mode="json"),
        "authorization_digest": inputs.digest,
        "source_sha": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "files": file_inventory(ROOT, bound_files()),
        "image_id": docker("image", "inspect", IMAGE, "--format", "{{.Id}}"),
    }
    publish(root / "manifest.json", manifest)
    return manifest


def verify(root, manifest):
    require(
        manifest["artifact_kind"] == "resume_live_manifest_v1" and manifest["trace_mode"] == "off",
        "invalid_live_manifest",
    )
    inputs = LiveInputs.model_validate_json(json.dumps(manifest["inputs"]))
    require(inputs.digest == manifest["authorization_digest"], "authorization_changed")
    require(read_json(root / "inputs.json", LiveInputs) == inputs, "authorization_changed")
    verify_inputs(root, inputs)
    require(
        git(ROOT, "rev-parse", "HEAD").decode().strip() == manifest["source_sha"], "source_changed"
    )
    require(
        set(bound_files()) == set(manifest["files"])
        and file_inventory(ROOT, manifest["files"]) == manifest["files"],
        "source_changed",
    )
    return inputs


def begin_stage(root, stage, binding):
    """A completed stage replays; an interrupted logical operation needs investigation."""
    done = root / f"{stage}-done.json"
    if done.exists():
        result = read_private_json(done)
        require(result["authorization_digest"] == binding, "stage_binding_changed")
        return result
    require(not (root / f"{stage}-started.json").exists(), "interrupted_stage_requires_review")
    publish(root / f"{stage}-started.json", {"authorization_digest": binding})
    return None


async def execute(root, stage, credentials_path):
    from tests.evals.resume_live_runtime import run_stage

    manifest = read_private_json(root / "manifest.json")
    inputs = verify(root, manifest)
    old = begin_stage(root, stage, inputs.digest)
    if old is not None:
        return old
    with tracing_context(enabled=False):
        async with ResumableDatabase(root, manifest["image_id"]) as url:
            result = await run_stage(
                url,
                root,
                inputs,
                stage,
                credentials_path,
                source_check=lambda: verify(root, manifest),
            )
    publish(root / f"{stage}-done.json", {"authorization_digest": inputs.digest, **result})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(
            args.root.is_absolute() and not args.root.resolve().is_relative_to(ROOT),
            "unsafe_directory",
        )
        if args.action == "prepare":
            prepare(args.root)
            return 0
        if args.action == "report":
            result = read_private_json(args.root / "report.json")
            print(json.dumps({k: result[k] for k in ("status", "r71_status", "human_review")}))
            return 0
        require(args.live and args.credentials is not None and args.stage, "live_opt_in_required")
        manifest = read_private_json(args.root / "manifest.json")
        inputs = verify(args.root, manifest)
        stages = {"materials", "facts", "profile", "report"} | {
            f"{c.case_id}-{s}"
            for c in inputs.cases
            for s in ("draft", "round1", "round2", "confirm")
        }
        require(args.stage in stages, "invalid_stage")
        fd = os.open(args.root / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = asyncio.run(execute(args.root, args.stage, args.credentials))
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "status": result.get("status", "COMPLETE"),
                    "usage": result.get("usage"),
                }
            )
        )
        return 0
    except Exception as error:
        # Never expose provider bodies, SQL parameters or credentials in terminal output.
        if args.root.is_dir():
            publish(
                args.root / f"failure-{uuid4().hex}.json",
                {
                    "status": "PARTIAL",
                    "exception_type": type(error).__name__,
                    "frames": [
                        {"file": f.filename, "line": f.lineno, "function": f.name}
                        for f in traceback.extract_tb(error.__traceback__)
                    ],
                    "stage": args.stage,
                },
            )
        print("R71 PARTIAL: inspect private stage evidence; no automatic replay")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
