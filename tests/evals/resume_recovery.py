"""Explicit full D experiment; original preparation and paid ledger stay bound."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import SecretStr
from sqlalchemy import select

from app.db.models import RunJob
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.tenancy import TenantService
from scripts.ci_evidence import FULL_JOBS
from tests.evals.product_acceptance_contracts import (
    bound_files,
    publish,
    read_private_json,
    require,
)
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import ROOT, file_inventory, git
from tests.evals.resume_experiment_budget import ExperimentRecorder
from tests.evals.resume_experiments import effective_source, load_inputs, preserve
from tests.evals.resume_live_environment import ResumableDatabase
from tests.evals.resume_recovery_contracts import SEED, SETTINGS, VERSION, cases, summarize
from tests.evals.resume_recovery_faults import ledger, run_case
from tests.evals.resume_recovery_fixture import prepare, snapshot


def preparation(root):
    inputs = load_inputs(root)
    frozen = read_private_json(root / "frozen.json")
    original = read_private_json(root / "manifest.json")
    report = read_private_json(root / "preparation-report-final.json")
    require(
        frozen["status"] == "PREPARATION_FROZEN"
        and frozen["input_digest"] == inputs.digest
        and frozen["manifest_digest"] == quality_identity_digest(original),
        "preparation_changed",
    )
    require(
        frozen["effective_source"] == effective_source(root, original), "preparation_source_changed"
    )
    require(
        report["next_step_allowed"] is True
        and report["input_digest"] == inputs.digest
        and report["completed_annotations"] == 20,
        "preparation_incomplete",
    )
    for name, digest in report["artifact_sha256"].items():
        path = root / name
        require(
            path.resolve().is_relative_to(root) and not path.is_symlink(),
            "unsafe_preparation_artifact",
        )
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest, "preparation_artifact_changed"
        )
    return original, inputs, frozen


def validate_ci(evidence, sha):
    require(
        evidence["repo"] == "youngcs1024/pathfinder-pub"
        and evidence["sha"] == sha
        and evidence["category"] == "full_success"
        and evidence["attempt"] >= 1,
        "full_ci_required",
    )
    require(
        len(evidence["jobs"]) == len(FULL_JOBS)
        and {j["name"] for j in evidence["jobs"]} == FULL_JOBS
        and all(
            j["status"] == "completed" and j["conclusion"] == "success" for j in evidence["jobs"]
        ),
        "full_ci_required",
    )


def source():
    require(not git(ROOT, "status", "--porcelain").strip(), "source_not_committed")
    return {
        "source_sha": git(ROOT, "rev-parse", "HEAD").decode().strip(),
        "files": file_inventory(ROOT, bound_files()),
    }


def execution_root(root, identity):
    return root / "d" / identity["source_sha"]


async def paid_audit(sessions, root, inputs, frozen):
    binding = read_private_json(root / "ledger-binding.json")
    tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
        workspace_id=UUID(binding["workspace_id"]), actor_user_id=UUID(binding["actor_user_id"])
    )
    recorder = ExperimentRecorder(
        sessions,
        tenant,
        root=root,
        inputs=inputs,
        database_identity=quality_identity_digest(read_private_json(root / "database-owner.json")),
    )
    await recorder.initialize()
    usage = await recorder.check_admission(after=True)
    require(usage == frozen["usage"], "paid_ledger_changed_during_fake_experiment")
    return usage


def report(root):
    identity = source()
    directory = execution_root(root, identity)
    manifest = read_private_json(directory / "manifest.json")
    require(manifest["source"] == identity and manifest["cases"] == cases(), "d_manifest_changed")
    results = []
    receipts = {}
    for case in cases():
        path = directory / case["case_id"] / "result.json"
        if path.exists():
            results.append(read_private_json(path))
            receipts[case["case_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = summarize(results)
    summary.update(
        source_sha=identity["source_sha"],
        manifest_digest=quality_identity_digest(manifest),
        result_sha256=receipts,
        settings=SETTINGS,
        ci_evidence=manifest["ci"],
        paid_usage=manifest["paid_usage"],
        new_actual_provider_attempts=0,
    )
    # Completion requires a final original-ledger audit and retained database identity.
    final = directory / "final-audit.json"
    if summary["status"] == "PASS":
        require(
            final.exists() and read_private_json(final)["usage"] == manifest["paid_usage"],
            "final_audit_missing",
        )
    summary["next_step_allowed"] = summary["status"] == "PASS"
    summary["next_step"] = (
        "A_initial_draft_factual_support_and_job_coverage" if summary["next_step_allowed"] else None
    )
    return summary


async def execute(root, *, resume, ci_path):
    original, inputs, frozen = preparation(root)
    identity = source()
    directory = execution_root(root, identity)
    if resume:
        require((directory / "manifest.json").exists(), "d_resume_identity_missing")
        ci = read_private_json(directory / "manifest.json")["ci"]
    else:
        require(ci_path is not None, "d_ci_receipt_required")
        ci = read_private_json(ci_path)
    validate_ci(ci, identity["source_sha"])
    # Never create a replacement for the original allocation database.
    require(
        (root / "database-owner.json").exists() and (root / "ledger-binding.json").exists(),
        "original_database_missing",
    )
    (root / "d").mkdir(mode=0o700, exist_ok=True)
    directory.mkdir(mode=0o700, exist_ok=True)
    async with ResumableDatabase(root, original["image_id"]) as url:
        engine = create_database_engine(SecretStr(url))
        sessions = create_session_factory(engine)
        try:
            usage = await paid_audit(sessions, root, inputs, frozen)
            manifest = {
                "version": VERSION,
                "source": identity,
                "preparation_digest": quality_identity_digest(frozen),
                "database_identity": quality_identity_digest(
                    read_private_json(root / "database-owner.json")
                ),
                "cases": cases(),
                "seed": SEED,
                "settings": SETTINGS,
                "ci": ci,
                "paid_usage": usage,
            }
            preserve(directory / "manifest.json", manifest)
            for index, case in enumerate(cases()):
                path = directory / case["case_id"]
                path.mkdir(mode=0o700, exist_ok=True)
                if (path / "result.json").exists():
                    continue
                state = None
                try:
                    async with sessions() as db:
                        pending = list(
                            await db.scalars(
                                select(RunJob).where(RunJob.status.in_(("queued", "leased")))
                            )
                        )
                    if pending:
                        require((path / "state.json").exists(), "unrelated_pending_work")
                        saved = read_private_json(path / "state.json")
                        require(
                            all(
                                str(j.run_id) == saved["run_id"]
                                and str(j.workspace_id) == saved["workspace_id"]
                                for j in pending
                            ),
                            "unrelated_pending_work",
                        )
                    state = await prepare(sessions, path, case)
                    result = await run_case(url, sessions, path, state)
                except Exception as error:
                    # Fixed categories only; never publish exception messages or DSNs.
                    result = dict(
                        case=case,
                        status="PARTIAL",
                        verified=False,
                        injected=(path / "injected.json").exists(),
                        error_type=type(error).__name__,
                    )
                    if state:
                        result["observation"] = await snapshot(sessions, state)
                        result["ledger"] = await ledger(sessions, state)
                    publish(path / f"partial-{uuid4().hex}.json", result)
                    raise
                result.update(
                    source_sha=identity["source_sha"], case_digest=quality_identity_digest(case)
                )
                preserve(path / "result.json", result)
                print(f"D {index + 1}/1060 {case['case_id']} {result['status']}", flush=True)
                if result["status"] != "PASS":
                    require(
                        result.get("observation", {}).get("run_status")
                        in ("completed", "failed", "cancelled"),
                        "nonterminal_case_stopped",
                    )
            require(source() == identity, "source_changed")
            usage = await paid_audit(sessions, root, inputs, frozen)
            preserve(
                directory / "final-audit.json",
                {"usage": usage, "database_identity": manifest["database_identity"]},
            )
        except BaseException:
            publish(
                directory / f"partial-{uuid4().hex}.json",
                {
                    "status": "PARTIAL",
                    "created_at": datetime.now(UTC).isoformat(),
                    "source_sha": identity["source_sha"],
                },
            )
            raise
        finally:
            await engine.dispose()
    result = report(root)
    preserve(directory / "report.json", result)
    return result
