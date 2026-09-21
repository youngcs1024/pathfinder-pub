"""Manual E8.3 CLI. Import/prepare never calls providers; decisions require exact review."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib.metadata
import json
import os
import platform
import signal
from pathlib import Path

from langsmith import tracing_context

from app.llm.qwen_adapters import create_qwen_adapters
from tests.evals.product_acceptance_contracts import (
    CASES,
    AcceptanceError,
    Manifest,
    encoded,
    new_directory,
    prepare_manifest,
    publish,
    read_private_json,
    require,
    verify_manifest,
    write_decision,
)
from tests.evals.product_acceptance_environment import (
    IMAGE,
    OwnedProductDatabase,
    docker,
    local_docker,
)
from tests.evals.product_acceptance_runtime import ProductSession, run_cases
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import no_links
from tests.evals.quality_generation_support import checked_directory
from tests.evals.quality_pilot import credentials


def software_identity():
    return {
        "python": platform.python_version(),
        "packages": sorted(
            (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
        ),
    }


def prepare(root):
    manifest = prepare_manifest()
    local_docker()
    image_id = docker("image", "inspect", IMAGE, "--format", "{{.Id}}")
    root = new_directory(root)
    publish(root / "manifest.json", manifest)
    publish(
        root / "environment.json",
        {
            "image_id": image_id,
            "software": software_identity(),
            "transport": "in_process_http",
            "worker_slots": 1,
            "production_deployment": False,
        },
    )


async def execute(root, credentials_path):
    root = no_links(root)
    checked_directory(root)
    manifest = Manifest.model_validate_json(encoded(read_private_json(root / "manifest.json")))
    environment = read_private_json(root / "environment.json")
    verify_manifest(manifest)
    require(environment["software"] == json.loads(encoded(software_identity())), "software_changed")
    require(platform.python_version() == "3.12.13", "toolchain_changed")
    # The retained lock file is never unlinked. flock prevents two controllers sharing this root.
    lock_fd = os.open(root / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AcceptanceError("execution_already_running") from None
        publish(root / "started.json", {"source_sha": manifest.source_sha, "repeat": 1})
        await _execute_once(root, credentials_path, manifest, environment)
    finally:
        os.close(lock_fd)


async def _execute_once(root, credentials_path, manifest, environment):
    bundle, product, records = None, None, []
    db = OwnedProductDatabase(root, environment["image_id"])
    category = None
    try:
        values = credentials(credentials_path)
        markers = tuple(v.get_secret_value() for v in values.values())
        bundle = create_qwen_adapters(
            api_key=values["DASHSCOPE_API_KEY"],
            workspace_id=values["PF_QWEN_WORKSPACE_ID"],
        )
        async with db as url:
            product = ProductSession(
                url,
                root,
                bundle.chat,
                bundle.embedding,
                provider="qwen",
                source_check=lambda: verify_manifest(manifest),
                markers=markers,
            )
            with tracing_context(enabled=False):
                async with product.open():
                    records = await run_cases(product)
    except asyncio.CancelledError:
        category = "cancelled"
        raise
    except Exception as error:
        category = str(error) if isinstance(error, AcceptanceError) else "execution_failed"
        raise AcceptanceError(category) from None
    finally:
        try:
            if bundle is not None:
                await bundle.aclose()
        finally:
            publish(
                root / "report.json",
                {
                    "artifact_kind": "e83_product_report_v1",
                    "source_sha": manifest.source_sha,
                    "manifest_digest": quality_identity_digest(manifest.model_dump(mode="json")),
                    "status": "PASS"
                    if len(records) == 2
                    and all(r["checks_passed"] for r in records)
                    and db.cleanup_ok
                    and category is None
                    else "PARTIAL",
                    "failure_category": category,
                    "cases": records
                    or (
                        read_private_json(root / "cases.json")["cases"]
                        if (root / "cases.json").exists()
                        else [{"case_id": c, "status": "NOT_RUN"} for c in CASES]
                    ),
                    "cleanup_ok": db.cleanup_ok,
                    "semantic_review": "REQUIRED",
                    "capacity_evidence": "PARTIAL",
                    "fault_evidence": "PARTIAL",
                    "deployment": "NOT_RUN",
                    "baseline_accepted": False,
                },
            )


async def cancellable_execute(root, credentials_path):
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await execute(root, credentials_path)
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manual E8.3 two-case product acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--root", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--credentials", type=Path, required=True)
    run.add_argument("--confirm-live", action="store_true", required=True)
    decision = commands.add_parser("decide")
    decision.add_argument("--root", type=Path, required=True)
    decision.add_argument("--case", choices=CASES, required=True)
    decision.add_argument("--decision", choices=("approve", "reject"), required=True)
    decision.add_argument("--review-digest", required=True)
    decision.add_argument("--confirm-exact-review", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.root)
        elif args.command == "run":
            asyncio.run(cancellable_execute(args.root, args.credentials))
        else:
            write_decision(args.root, args.case, args.decision, args.review_digest)
        print(json.dumps({"stage": args.command, "status": "finished"}))
        return 0
    except (asyncio.CancelledError, KeyboardInterrupt):
        print(json.dumps({"stage": args.command, "category": "cancelled"}))
        return 130
    except Exception:
        print(json.dumps({"stage": args.command, "category": "acceptance_failed"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
