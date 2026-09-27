"""Opt-in A-D public preparation. Private body artifacts never enter public CI."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4

from app.domain.resume_profile import ResumeContentV1, check_model_input_privacy
from tests.evals.product_acceptance_contracts import (
    bound_files,
    publish,
    read_private_json,
    require,
)
from tests.evals.product_acceptance_environment import IMAGE, docker
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import ROOT, file_inventory, git, no_links
from tests.evals.resume_experiment_contracts import MINIMUM, PLANNED, Inputs
from tests.evals.resume_experiment_scoring import PROMPT_DIGESTS


def preserve(path, value):
    if path.exists():
        require(read_private_json(path) == value, "existing_artifact_changed")
    else:
        publish(path, value)


def load_inputs(root):
    raw = read_private_json(root / "inputs.json")
    inputs = Inputs.model_validate_json(json.dumps(raw))
    require(root == Path(inputs.execution_root), "allocation_root_changed")
    require(file_inventory(root / "inputs", inputs.files) == inputs.files, "inputs_changed")
    for case in inputs.cases:
        require((root / "inputs" / case.body_file).read_text() == case.jd, "jd_body_changed")
    facts = read_private_json(root / "inputs" / inputs.facts_file)["facts"]
    require(bool(facts) and len({f["version_id"] for f in facts}) == len(facts), "invalid_facts")
    profile = ResumeContentV1.model_validate_json(
        json.dumps(read_private_json(root / "inputs" / inputs.profile_file)["content"])
    )
    for fact in facts:
        require(fact["review_status"] == "confirmed" and fact["evidence"], "unconfirmed_fact")
        for evidence in fact["evidence"]:
            path = evidence["path"]
            require(path in inputs.files, "evidence_file_missing")
            lines = (root / "inputs" / path).read_text().splitlines(keepends=True)
            start, end = evidence["start_line"], evidence["end_line"]
            require(1 <= start <= end <= len(lines), "evidence_lines_invalid")
            require(evidence["quote"] in "".join(lines[start - 1 : end]), "fact_quote_changed")
    check_model_input_privacy(profile, {"facts": facts, "jobs": [c.jd for c in inputs.cases]})
    return inputs


def prepare(root):
    inputs = load_inputs(root)
    require(not git(ROOT, "status", "--porcelain").strip(), "source_not_committed")
    manifest = {
        "artifact_kind": "resume_experiment_manifest_v1",
        "input_digest": inputs.digest,
        "source_sha": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "files": file_inventory(ROOT, bound_files()),
        "prompts": PROMPT_DIGESTS,
        "planned": PLANNED,
        "minimum_usable": MINIMUM,
        "image_id": docker("image", "inspect", IMAGE, "--format", "{{.Id}}"),
        "trace_mode": "off",
        "human_review": "NOT_RUN",
    }
    preserve(root / "manifest.json", manifest)
    return manifest


def verify(root):
    inputs = load_inputs(root)
    manifest = read_private_json(root / "manifest.json")
    require(
        manifest["artifact_kind"] == "resume_experiment_manifest_v1"
        and manifest["input_digest"] == inputs.digest
        and manifest["prompts"] == PROMPT_DIGESTS
        and manifest["trace_mode"] == "off"
        and manifest["planned"] == PLANNED
        and manifest["minimum_usable"] == MINIMUM,
        "manifest_changed",
    )
    require(
        git(ROOT, "rev-parse", "HEAD").decode().strip() == manifest["source_sha"], "source_changed"
    )
    require(
        set(bound_files()) == set(manifest["files"])
        and file_inventory(ROOT, manifest["files"]) == manifest["files"],
        "source_changed",
    )
    return manifest, inputs


def freeze(root, usage):
    from tests.evals.resume_experiment_runtime import validate_reviews

    manifest, inputs = verify(root)
    require(
        read_private_json(root / "preparation-complete.json")["binding"] == inputs.digest,
        "preparation_incomplete",
    )
    require(
        not usage["unknown_cost"] and not usage["unknown_usage"] and not usage["unfinished"],
        "accounting_incomplete",
    )
    facts = read_private_json(root / "inputs" / inputs.facts_file)["facts"]
    annotations = validate_reviews(root, inputs, facts)
    result = {
        "status": "PREPARATION_FROZEN",
        "input_digest": inputs.digest,
        "manifest_digest": quality_identity_digest(manifest),
        "annotations": annotations,
        "usage": usage,
        "planned": PLANNED,
        "minimum_usable": MINIMUM,
        "formal_samples_executed": 0,
        "review_kind": "AGENT_ASSESSED",
        "human_review": "NOT_RUN",
        "next_step": "D_recovery_requires_separate_user_request",
    }
    preserve(root / "frozen.json", result)
    return result


def report(root):
    manifest, inputs = verify(root)
    frozen = root / "frozen.json"
    result = read_private_json(frozen) if frozen.exists() else None
    if result:
        require(
            result["input_digest"] == inputs.digest
            and result["manifest_digest"] == quality_identity_digest(manifest),
            "freeze_changed",
        )
    return {
        "status": result["status"] if result else "PARTIAL",
        "formal_samples_executed": 0,
        "human_review": "NOT_RUN",
        "usage": result["usage"] if result else None,
        "annotations_recorded": sum(
            (root / f"annotation-{c.case_id}.json").exists() for c in inputs.cases
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "resume", "freeze", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = no_links(args.root)
        require(args.root.is_absolute() and not root.is_relative_to(ROOT), "unsafe_directory")
        fd = os.open(root / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == "prepare":
                prepare(root)
            elif args.action == "report":
                print(json.dumps(report(root)))
            else:
                from tests.evals.resume_experiment_runtime import execute

                manifest, inputs = verify(root)
                if args.action != "freeze":
                    require(args.live and args.credentials is not None, "live_opt_in_required")
                    require(not (root / "frozen.json").exists(), "preparation_already_frozen")
                usage = asyncio.run(
                    execute(
                        root, manifest, inputs, args.credentials, audit_only=args.action == "freeze"
                    )
                )
                if args.action == "freeze":
                    freeze(root, usage)
                print(json.dumps({"status": "COMPLETE", "usage": usage}))
        return 0
    except Exception:
        # Never emit source bodies, DSNs, credentials, or exception payloads.
        print(
            json.dumps(
                {"status": "PARTIAL", "error": "preparation_failed", "diagnostic_id": uuid4().hex}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
