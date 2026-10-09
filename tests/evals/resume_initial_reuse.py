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


# Explicitly authorized auth-only security upgrade. Exact whole-lock digests keep
# every other package and all lock metadata protected; no general dependency bypass.
PYJWT_LOCK_COMPATIBILITY = {
    "46792d0e0554a696cbb39461d85b2cbd9cccbc4525aae1ab0cf07e63c79fca4e": "2.13.0",
    "cdae3d0317206e403e5d4cd671247f949deb96f42f4a9463ca0530eec75d3391": "2.14.0",
}


# 2026-10-09 explicitly approved security maintenance. Reviewed wheel changes do
# not affect A's local graph, HTTPX provider transport, prompts or renderer.
# Whole-lock identity limits this exception to these exact four package updates.
SECURITY_LOCK_COMPATIBILITY = {
    "3f95d8444efbb99bca12e6c8b3ea3958137ee462bd8a5ea02195503d0536e532": {
        "langgraph-sdk": "0.4.4",
        "mako": "1.4.2",
        "pyjwt": "2.15.0",
        "urllib3": "2.8.0",
    },
}


def compatible_lock_digest(digest):
    approved = digest in PYJWT_LOCK_COMPATIBILITY or digest in SECURITY_LOCK_COMPATIBILITY
    return next(iter(PYJWT_LOCK_COMPATIBILITY)) if approved else digest


def generation_lock_evidence(ref="HEAD"):
    digest = hashlib.sha256(git("show", f"{ref}:uv.lock")).hexdigest()
    return {
        "sha256": digest,
        "compatible_sha256": compatible_lock_digest(digest),
        "auth_only_pyjwt_version": PYJWT_LOCK_COMPATIBILITY.get(digest),
        **(
            {"security_upgrade_versions": SECURITY_LOCK_COMPATIBILITY[digest]}
            if digest in SECURITY_LOCK_COMPATIBILITY
            else {}
        ),
    }


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
    digests["uv.lock"] = compatible_lock_digest(digests["uv.lock"])
    tree = ast.parse(git("show", f"{ref}:tests/evals/resume_initial.py").decode())
    for name in ("generate_sample", "run_phase"):
        function = next(
            n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name
        )
        # Scoring arguments can evolve without changing generation control flow.
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "score_block"
            ):
                node.args, node.keywords = [], []
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
            "source_lock": generation_lock_evidence(source_sha),
            "current_lock": generation_lock_evidence(),
            "artifact_sha256": copied,
            "usage_at_reuse": initial_usage,
        },
    )


def score_origin(root, source_sha, manifest):
    require(
        len(source_sha) == 40 and all(c in "0123456789abcdef" for c in source_sha),
        "invalid_reuse_source",
    )
    origin = root / "a" / source_sha
    old = read_private_json(origin / "manifest.json")
    require(old["source"]["source_sha"] == source_sha, "reuse_source_changed")
    validate_ci(old["ci"], source_sha)
    for key in ("input_digest", "preparation_digest", "d_evidence", "seed", "budget", "pilot"):
        require(old[key] == manifest[key], "reuse_configuration_changed")
    version = old["scoring_version"]
    require(
        version in ("a-score-v2", "a-score-v3", "a-score-v4", "a-score-v5"),
        "reuse_score_version_unknown",
    )
    return {
        "phase": origin / "pilot",
        "version": version,
        "coverage_prompt": old["scoring_prompt"].get("coverage")
        if isinstance(old["scoring_prompt"], dict)
        else None,
    }


def formal_evidence_path(origin):
    for name in ("formal/audit.json", "closeout-recovery-v5.json", "closeout-partial-v4.json"):
        path = origin / name
        if path.exists():
            return path
    require(False, "reuse_evidence_missing")


def compatible_scores(old, manifest):
    return all(
        key in old and key in manifest and old[key] == manifest[key]
        for key in ("scoring_version", "scoring_prompt", "scoring_transport")
    )


def verify_formal_origin(root, source_sha, manifest, ancestors=()):
    require(source_sha not in ancestors and len(ancestors) < 16, "reuse_lineage_cycle")
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
    for key in (
        "input_digest",
        "preparation_digest",
        "d_evidence",
        "seed",
        "budget",
        "planned",
        "pilot",
    ):
        require(old[key] == manifest[key], "reuse_configuration_changed")
    for key in ("one_shot", "selection", "pathfinder", "annotate"):
        require(old["prompts"][key] == manifest["prompts"][key], "reuse_generation_prompt_changed")
    require(
        generation_fingerprint(source_sha) == generation_fingerprint(),
        "reuse_generation_code_changed",
    )
    if (origin / "frozen.json").exists():
        frozen = read_private_json(origin / "frozen.json")
        require(frozen["manifest_digest"] == quality_identity_digest(old), "reuse_freeze_changed")
    else:
        # A recovery may finish with terminal unresolved scores before a new freeze.
        # Its generated samples must still trace to the original frozen formal source.
        parent = old.get("reuse_formal_source")
        require(bool(parent), "reuse_freeze_missing")
        verify_formal_origin(root, parent, old, (*ancestors, source_sha))
        recovery = read_private_json(origin / "recovery-validation.json")
        require(recovery["status"] in ("PASS", "PARTIAL"), "reuse_recovery_incomplete")
        require(recovery["usage"]["unfinished"] == 0, "reuse_recovery_unfinished")
    if (origin / "formal/audit.json").exists():
        inventory = {
            "formal/" + k: v
            for k, v in read_private_json(origin / "formal/audit.json")["artifact_sha256"].items()
        }
    else:
        closeout = read_private_json(formal_evidence_path(origin))
        require(closeout["source_sha"] == source_sha, "reuse_closeout_changed")
        inventory = closeout["artifact_sha256"]
    evidence = (
        read_private_json(origin / "formal/audit.json")
        if (origin / "formal/audit.json").exists()
        else closeout
    )
    require(
        manifest["reuse_formal_evidence_digest"] == quality_identity_digest(evidence),
        "reuse_evidence_changed",
    )
    for name, digest in inventory.items():
        path = origin / name
        require(
            path.resolve().is_relative_to(origin.resolve()) and not path.is_symlink(),
            "unsafe_reuse_artifact",
        )
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "reuse_artifact_changed")
    return origin, inventory


async def reuse_formal(rig, root, directory, source_sha, manifest, usage_for):
    origin, inventory = verify_formal_origin(root, source_sha, manifest)
    target = directory / "formal"
    target.mkdir(mode=0o700, exist_ok=True)
    before = await rig.recorder.check_admission(after=True)
    copied = {}
    same_scores = compatible_scores(read_private_json(origin / "manifest.json"), manifest)
    for sample in manifest["planned"]:
        sid = sample["sample_id"]
        name = f"formal/{sid}/generation.json"
        if not (origin / name).exists():
            continue
        require(name in inventory, "reuse_unaudited_file")
        generated = read_private_json(origin / name)
        require(generated["sample"] == sample, "reuse_sample_changed")
        block = f"{sample['case_id']}-r{sample['repeat']}"
        business_name = f"formal/{block}-session.json"
        require(business_name in inventory, "reuse_unaudited_file")
        business = read_private_json(origin / business_name)
        inputs = await rig.store.execution_inputs(rig.tenant, UUID(business["session_id"]))
        from dataclasses import replace

        inputs = replace(inputs, facts=tuple(sorted(inputs.facts, key=lambda f: str(f.version_id))))
        require(str(inputs.run_id) == business["run_id"], "reuse_business_changed")
        require(
            quality_identity_digest(shared_input(inputs, rig.identity.source_sha256))
            == generated["common_input_digest"],
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
        result_name = f"formal/{sid}/result.json"
        if same_scores and result_name in inventory:
            saved = read_private_json(origin / result_name)
            require(saved["sample"] == sample, "reuse_score_sample_changed")
            require(
                saved["scoring_version"] == manifest["scoring_version"]
                if generated["output"]
                else saved["score_status"] == "NOT_APPLICABLE",
                "reuse_score_version_changed",
            )
            require(all(saved[k] == v for k, v in generated.items()), "reuse_generation_changed")
            if "score_usage" in saved:
                require(
                    await usage_for(rig.recorder, saved["score_usage"]["invocation_ids"])
                    == saved["score_usage"],
                    "reuse_score_ledger_changed",
                )
        for entry in inventory:
            parts = Path(entry).parts
            if (
                parts[:2] == ("formal", sid)
                and (same_scores or (len(parts) == 3 and parts[2] != "result.json"))
            ) or entry in (
                business_name,
                f"formal/{block}-command.json",
                "formal/annotations.json",
                f"formal/{block}-blind.json",
            ):
                dst = directory / entry
                dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                preserve(dst, read_private_json(origin / entry))
                copied[entry] = inventory[entry]
    require(await rig.recorder.check_admission(after=True) == before, "reuse_added_calls")
    preserve(directory / "formal-reuse.json", {"source_sha": source_sha, "artifact_sha256": copied})
    value = score_origin(root, source_sha, manifest)
    value["phase"] = origin / "formal"
    return value
