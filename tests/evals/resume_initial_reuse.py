"""Verified cross-source generation reuse; never imports scores or resets accounting."""

import ast
import hashlib
import subprocess
from pathlib import Path
from uuid import UUID

from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiments import preserve
from tests.evals.resume_initial_baselines import shared_input
from tests.evals.resume_quality_runtime import snapshot
from tests.evals.resume_recovery import validate_ci


def git(*args):
    return subprocess.check_output(["git", *args])


def generation_fingerprint(ref="HEAD"):
    paths = git("ls-tree", "-r", "--name-only", ref).decode().splitlines()
    included = [
        p
        for p in paths
        if p.startswith("src/")
        or p
        in (
            "uv.lock",
            "pyproject.toml",
            "tests/evals/resume_initial_baselines.py",
            "tests/evals/resume_initial_fixture.py",
            "tests/evals/resume_initial_recording.py",
            "tests/evals/resume_quality_runtime.py",
        )
    ]
    digests = {p: hashlib.sha256(git("show", f"{ref}:{p}")).hexdigest() for p in included}
    tree = ast.parse(git("show", f"{ref}:tests/evals/resume_initial.py").decode())
    for name in ("generate_sample", "run_phase"):
        function = next(
            n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name
        )
        digests[name] = quality_identity_digest(ast.dump(function, include_attributes=False))
    return quality_identity_digest(digests)


def verify_origin(root, source_sha, manifest):
    require(
        len(source_sha) == 40 and all(c in "0123456789abcdef" for c in source_sha),
        "invalid_reuse_source",
    )
    origin = root / "a" / source_sha
    require(
        not origin.is_symlink() and origin.resolve().parent == (root / "a").resolve(),
        "unsafe_reuse_source",
    )
    old = read_private_json(origin / "manifest.json")
    validate_ci(old["ci"], source_sha)
    require(old["source"]["source_sha"] == source_sha, "reuse_source_changed")
    for key in ("input_digest", "preparation_digest", "d_evidence", "seed", "budget", "pilot"):
        require(old[key] == manifest[key], "reuse_configuration_changed")
    for key in ("one_shot", "selection", "pathfinder", "annotate"):
        require(old["prompts"][key] == manifest["prompts"][key], "reuse_generation_prompt_changed")
    fingerprint = generation_fingerprint(source_sha)
    require(fingerprint == generation_fingerprint(), "reuse_generation_code_changed")
    phase = origin / "pilot"
    audit = read_private_json(phase / "audit.json")
    for name, digest in audit["artifact_sha256"].items():
        path = phase / name
        require(
            path.resolve().is_relative_to(phase.resolve()) and not path.is_symlink(),
            "unsafe_reuse_artifact",
        )
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "reuse_artifact_changed")
    for s in manifest["pilot"]:
        require(
            f"{s['sample_id']}/generation.json" in audit["artifact_sha256"],
            "reuse_generation_missing",
        )
    return origin, fingerprint, audit


async def reuse_pilot(rig, root, directory, source_sha, manifest, usage_for):
    origin, fingerprint, audit = verify_origin(root, source_sha, manifest)
    phase = directory / "pilot"
    phase.mkdir(mode=0o700, exist_ok=True)
    old_phase = origin / "pilot"
    copied = {}

    def copy(name):
        require(name in audit["artifact_sha256"], "reuse_unaudited_file")
        target = phase / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        preserve(target, read_private_json(old_phase / name))
        copied[name] = audit["artifact_sha256"][name]

    before = await rig.recorder.check_admission(after=True)
    for sample in manifest["pilot"]:
        sid = sample["sample_id"]
        generated = read_private_json(old_phase / sid / "generation.json")
        require(generated["sample"] == sample, "reuse_sample_changed")
        block = f"{sample['case_id']}-r{sample['repeat']}"
        business = read_private_json(old_phase / f"{block}-session.json")
        inputs = await rig.store.execution_inputs(rig.tenant, UUID(business["session_id"]))
        from dataclasses import replace

        inputs = replace(inputs, facts=tuple(sorted(inputs.facts, key=lambda f: str(f.version_id))))
        require(str(inputs.run_id) == business["run_id"], "reuse_business_changed")
        common = shared_input(inputs, rig.identity.source_sha256)
        require(
            quality_identity_digest(common) == generated["common_input_digest"],
            "reuse_input_changed",
        )
        require(
            await usage_for(rig.recorder, generated["usage"]["invocation_ids"])
            == generated["usage"],
            "reuse_ledger_changed",
        )
        if sample["arm"] == "pathfinder" and generated["output"]:
            current = await snapshot(rig, inputs.session_id)
            require(
                all(
                    current.get(k) == v
                    for k, v in generated["output"].items()
                    if k != "automatic_repairs"
                ),
                "reuse_business_changed",
            )
        for name in audit["artifact_sha256"]:
            parts = Path(name).parts
            if (
                len(parts) == 2 and parts[0] == sid and parts[1] not in ("result.json",)
            ) or name in (f"{block}-session.json", f"{block}-command.json", "annotations.json"):
                copy(name)
    require(await rig.recorder.check_admission(after=True) == before, "reuse_added_calls")
    receipt = directory / "reuse.json"
    initial_usage = read_private_json(receipt)["usage_at_reuse"] if receipt.exists() else before
    preserve(
        receipt,
        {
            "source_sha": source_sha,
            "origin": str(origin),
            "generation_fingerprint": fingerprint,
            "artifact_sha256": copied,
            "usage_at_reuse": initial_usage,
        },
    )
