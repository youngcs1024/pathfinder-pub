"""Explicit A controller on the original preparation database and cumulative budget."""

import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from uuid import UUID, uuid4

from langsmith import tracing_context
from pydantic import SecretStr
from sqlalchemy import select

from app.agents.resume_generation import PROMPT_VERSION as SYSTEM_PROMPT
from app.db.models import ResumeSession, RunJob
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatMessage
from app.llm.qwen_adapters import create_qwen_adapters
from app.resume.template_render import TemplateIdentity
from tests.evals import resume_initial_scoring as scoring
from tests.evals.product_acceptance_budget import summarize as summarize_usage
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_pilot import credentials
from tests.evals.resume_experiment_budget import ExperimentRecorder
from tests.evals.resume_experiment_contracts import (
    checked_annotation,
    locate_annotation,
    normalize_json,
)
from tests.evals.resume_experiment_scoring import (
    PROMPT_DIGESTS,
    annotation_payload,
    blind_packets,
    profile_evidence,
    score_metrics,
)
from tests.evals.resume_experiment_scoring import (
    PROMPTS as REVIEW_PROMPTS,
)
from tests.evals.resume_experiments import load_inputs, preserve
from tests.evals.resume_initial_baselines import PROMPTS, candidate_text, generate, shared_input
from tests.evals.resume_initial_contracts import VERSION, samples, summarize
from tests.evals.resume_initial_fixture import seed
from tests.evals.resume_initial_recording import JournalFactory
from tests.evals.resume_initial_reuse import generation_fingerprint, reuse_pilot, score_origin
from tests.evals.resume_live_environment import ResumableDatabase
from tests.evals.resume_quality_runtime import snapshot, worker
from tests.evals.resume_recovery import preparation, source, validate_ci


def cases_for(root, inputs, pilot):
    if pilot:
        return [
            SimpleNamespace(**v)
            for v in read_private_json(root / "inputs/r71-inputs.json")["cases"]
        ]
    return list(inputs.cases)


def directory_for(root, identity):
    return root / "a" / identity["source_sha"]


def prerequisites(root):
    original, inputs, frozen = preparation(root)
    d_reports = sorted((root / "d").glob("*/report.json"))
    require(bool(d_reports), "d_evidence_missing")
    valid = []
    for path in d_reports:
        report = read_private_json(path)
        if report.get("status") != "PASS" or not report.get("next_step_allowed"):
            continue
        manifest = read_private_json(path.parent / "manifest.json")
        require(
            manifest["preparation_digest"] == quality_identity_digest(frozen), "d_binding_changed"
        )
        validate_ci(report["ci_evidence"], report["source_sha"])
        require(len(report["result_sha256"]) == 1060, "d_incomplete")
        for name, digest in report["result_sha256"].items():
            require(Path(name).name == name, "unsafe_d_case")
            require(
                hashlib.sha256((path.parent / name / "result.json").read_bytes()).hexdigest()
                == digest,
                "d_result_changed",
            )
        audit = read_private_json(path.parent / "final-audit.json")
        require(audit["usage"] == report["paid_usage"], "d_audit_changed")
        valid.append(quality_identity_digest(report))
    require(bool(valid), "d_not_passed")
    return original, inputs, frozen, valid


def config(root, inputs, frozen, identity, ci, d_evidence):
    return {
        "version": VERSION,
        "scoring_version": scoring.VERSION,
        "scoring_prompt": {
            "claims": scoring.PROMPT_DIGEST,
            "coverage": scoring.COVERAGE_PROMPT_DIGEST,
        },
        "generation_fingerprint": generation_fingerprint(),
        "source": identity,
        "ci": ci,
        "input_digest": inputs.digest,
        "preparation_digest": quality_identity_digest(frozen),
        "d_evidence": d_evidence,
        "seed": inputs.blind_seed,
        "prompts": {
            **{k: quality_identity_digest(v) for k, v in PROMPTS.items()},
            "pathfinder": SYSTEM_PROMPT,
            **PROMPT_DIGESTS,
        },
        "planned": samples([c.case_id for c in inputs.cases]),
        "pilot": samples([c.case_id for c in cases_for(root, inputs, True)], pilot=True),
        "budget": inputs.budget.model_dump(mode="json"),
        "trace_mode": "off",
        "authorization": "user_approved_A_live_materials_budget_commit_push_20260928",
    }


def report(root, *, pilot=False):
    identity = source()
    directory = directory_for(root, identity)
    manifest = read_private_json(directory / "manifest.json")
    require(manifest["source"] == identity, "a_source_changed")
    phase = directory / ("pilot" if pilot else "formal")
    planned = manifest["pilot" if pilot else "planned"]
    if (phase / "annotations.json").exists():
        annotations = read_private_json(phase / "annotations.json")
    elif not pilot:
        frozen = read_private_json(root / "frozen.json")
        annotations = {k: v["annotation"] for k, v in frozen["annotations"].items()}
    else:
        annotations = {s["case_id"]: {"requirements": []} for s in planned}
    results, digests = {}, {}
    for case in planned:
        path = phase / case["sample_id"] / "result.json"
        if path.exists():
            row = read_private_json(path)
            require(row["sample"] == case, "sample_changed")
            results[case["sample_id"]] = row
            digests[case["sample_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif (path.parent / "generation.json").exists():
            row = read_private_json(path.parent / "generation.json")
            require(row["sample"] == case, "sample_changed")
            results[case["sample_id"]] = {**row, "score_status": "UNRESOLVED", "incomplete": True}
    summary = summarize(planned, results, annotations)
    audit = phase / "audit.json"
    if not audit.exists():
        summary["status"] = "PARTIAL"
    else:
        for name, digest in read_private_json(audit)["artifact_sha256"].items():
            path = phase / name
            require(
                path.resolve().is_relative_to(phase.resolve()) and not path.is_symlink(),
                "unsafe_a_artifact",
            )
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "a_artifact_changed")
    current_usage = read_private_json(audit)["usage"] if audit.exists() else None
    if current_usage is None:
        partials = sorted(directory.glob("partial-*.json"), key=lambda p: p.stat().st_mtime_ns)
        if partials:
            current_usage = read_private_json(partials[-1]).get("usage")
    phase_usage = None
    if current_usage is not None and (phase / "started.json").exists():
        before = read_private_json(phase / "started.json")["usage"]
        phase_usage = {
            k: current_usage[k] - before[k]
            for k in (
                "attempts",
                "input_tokens",
                "output_tokens",
                "unknown_cost",
                "unknown_usage",
                "unfinished",
            )
        }
        phase_usage["known_cost_cny"] = str(
            Decimal(current_usage["known_cost_cny"]) - Decimal(before["known_cost_cny"])
        )
        phase_usage["includes_annotation_scoring_failures_and_retries"] = True
    a_total = None
    if current_usage is not None:
        prior = [read_private_json(p) for p in sorted((root / "d").glob("*/report.json"))]
        baseline = next(r["paid_usage"] for r in reversed(prior) if r["status"] == "PASS")
        a_total = {
            k: current_usage[k] - baseline[k]
            for k in ("attempts", "input_tokens", "output_tokens", "unknown_cost", "unknown_usage")
        }
        a_total["known_cost_cny"] = str(
            Decimal(current_usage["known_cost_cny"]) - Decimal(baseline["known_cost_cny"])
        )
        a_total["invocation_ids"] = sorted(
            set(current_usage["invocation_ids"]) - set(baseline["invocation_ids"])
        )
    summary.update(
        a_total_usage=a_total,
        scoring_version=manifest.get("scoring_version"),
        cumulative_usage=current_usage,
        phase_usage=phase_usage,
        source_sha=identity["source_sha"],
        ci_evidence=manifest["ci"],
        pilot=pilot,
        result_sha256=digests,
        usage=read_private_json(audit) if audit.exists() else None,
        samples=results,
        next_step_allowed=not pilot and summary["status"] == "PASS",
        limitations=[
            "Single-company convenience sample, not independent human evaluation.",
            "No PDF compilation, deployment or causal attribution to multi-stage reasoning.",
        ],
    )
    return summary


async def usage_for(recorder, ids):
    rows = [r for r in await recorder.rows() if str(r.id) in set(ids)]
    require(len(rows) == len(set(ids)), "sample_ledger_missing")
    return {
        **summarize_usage(rows, provider=recorder.provider),
        "invocation_ids": sorted(ids),
        "latency_ms": sum(r.latency_ms or 0 for r in rows),
    }


async def generate_sample(rig, sample, inputs, path):
    path.mkdir(mode=0o700, exist_ok=True)
    result = path / "generation.json"
    common = shared_input(inputs, rig.identity.source_sha256)
    preserve(path / "input.json", {"digest": quality_identity_digest(common), "input": common})
    if result.exists():
        return read_private_json(result)
    started = path / "generation-started.json"
    if not started.exists():
        preserve(started, {"usage": await rig.recorder.check_admission(), "sample": sample})
    before = read_private_json(started)["usage"]
    began = monotonic()
    journal = JournalFactory(rig.factory, path, rig.recorder)
    output, error = None, None
    if sample["arm"] == "pathfinder":
        detail = await rig.store.get_session(rig.tenant, inputs.session_id)
        if detail["run_status"] not in ("completed", "failed", "cancelled"):
            # An interrupted worker with any prior calls is never restarted automatically.
            require(
                not list(path.glob("call-*-started.json")), "interrupted_worker_requires_review"
            )
            async with rig.sessions() as db:
                pending = list(
                    await db.scalars(select(RunJob).where(RunJob.status.in_(("queued", "leased"))))
                )
            require(
                len(pending) == 1 and pending[0].run_id == inputs.run_id, "unrelated_pending_work"
            )
            require(await worker(rig, journal).run_once(asyncio.Event()), "worker_idle")
            detail = await rig.store.get_session(rig.tenant, inputs.session_id)
        await rig.recorder.check_admission(after=True)
        require(detail["run_status"] in ("completed", "failed", "cancelled"), "worker_not_terminal")
        if detail.get("current_version_id"):
            output = await snapshot(rig, inputs.session_id)
            async with rig.sessions() as db:
                session = await db.get(ResumeSession, inputs.session_id)
                output["automatic_repairs"] = session.repair_count
        else:
            error = "no_published_draft"
        preserve(path / "business.json", json.loads(json.dumps(detail, default=str)))
    else:
        try:
            output = await generate(
                sample["arm"],
                journal.create_chat_model(
                    LLMInvocationContext(rig.tenant.workspace_id, rig.tenant.actor_user_id)
                ),
                inputs,
                rig.source_bytes,
                rig.identity,
            )
        except Exception as exc:
            await rig.recorder.check_admission(after=True)
            # Only a persisted completed response can become an ordinary failed sample.
            require(bool(list(path.glob("call-*-response.json"))), "missing_model_response")
            error = type(exc).__name__
    after = await rig.recorder.check_admission(after=True)
    ids = sorted(set(after["invocation_ids"]) - set(before["invocation_ids"]))
    value = {
        "sample": sample,
        "generation_status": "COMPLETE" if output else "FAILED",
        "output": output,
        "error": error,
        "usage": await usage_for(rig.recorder, ids),
        "elapsed_seconds": monotonic() - began,
        "common_input_digest": quality_identity_digest(common),
    }
    preserve(result, value)
    return value


async def review_call(rig, path, task, payload):
    path.mkdir(mode=0o700, exist_ok=True)
    model = JournalFactory(rig.factory, path, rig.recorder).create_chat_model(
        LLMInvocationContext(rig.tenant.workspace_id, rig.tenant.actor_user_id)
    )
    response = await model.invoke(
        (
            ChatMessage(role="system", content=REVIEW_PROMPTS[task]),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ),
        (),
        {
            "task": "resume_initial_review",
            "graph_node": task,
            "prompt_version": PROMPT_DIGESTS[task],
        },
    )
    require(response.finish_status == "completed" and not response.tool_calls, "review_incomplete")
    return normalize_json(response.content or "")


async def score_block(rig, phase, block, case, annotation, reuse_phase=None):
    candidates = []
    for sample in block:
        generated = read_private_json(phase / sample["sample_id"] / "generation.json")
        if generated["output"]:
            candidates.append(
                {
                    "sample_id": sample["sample_id"],
                    "text": candidate_text(generated["output"]["content"]),
                }
            )
    from tests.evals.resume_experiment_contracts import Annotation

    rubric = Annotation.model_validate_json(json.dumps(annotation))
    packets, mapping = blind_packets(
        candidates,
        case=case,
        facts=rig.facts,
        annotation=rubric,
        seed=rig.seed + block[0]["repeat"],
        profile=rig.profile_evidence,
    )
    block_id = f"{case.case_id}-r{block[0]['repeat']}"
    preserve(phase / f"{block_id}-blind.json", {"mapping": mapping, "packets": packets})
    by_id = {mapping[p["blind_id"]]: p for p in packets}
    by_sample = {s["sample_id"]: s for s in block}
    ordered = [by_sample[mapping[p["blind_id"]]] for p in packets]
    ordered += [s for s in block if s["sample_id"] not in by_id]
    for sample in ordered:
        path = phase / sample["sample_id"]
        if (path / "result.json").exists():
            continue
        generated = read_private_json(path / "generation.json")
        result = {**generated, "score_status": "NOT_APPLICABLE", "metrics": None}
        if sample["sample_id"] in by_id:
            score_path = path / scoring.VERSION

            async def call(directory, payload, stage):
                model = JournalFactory(rig.factory, directory, rig.recorder).create_chat_model(
                    LLMInvocationContext(rig.tenant.workspace_id, rig.tenant.actor_user_id)
                )
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                return await model.invoke(
                    (
                        ChatMessage(role="system", content=scoring.prompt_for(payload)),
                        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
                    ),
                    (),
                    {
                        "task": "resume_initial_review",
                        "graph_node": f"score_{stage}",
                        "prompt_version": quality_identity_digest(scoring.prompt_for(payload)),
                    },
                )

            assessed, outcomes = await scoring.assess(
                generated["output"]["content"],
                rig.facts,
                rig.profile_evidence,
                annotation,
                case.jd,
                score_path,
                call,
                reuse_path=(reuse_phase["phase"] / sample["sample_id"] / reuse_phase["version"])
                if reuse_phase
                and (
                    reuse_phase["phase"]
                    / sample["sample_id"]
                    / reuse_phase["version"]
                    / "protocol.json"
                ).exists()
                else None,
            )
            result.update(
                scoring_version=scoring.VERSION,
                scoring_batches=outcomes,
                score_status="ASSESSED" if assessed else "UNRESOLVED",
            )
            if assessed:
                result.update(
                    assessment=assessed.model_dump(mode="json"),
                    metrics=score_metrics(assessed, content_ok=True),
                )
            all_ids = set()
            result["score_stages"] = {}
            for stage in scoring.STAGES:
                ids = set()
                for response in score_path.glob(f"batch-*/{stage}/call-*-response.json"):
                    ids.update(read_private_json(response)["invocation_ids"])
                result["score_stages"][stage] = await usage_for(rig.recorder, sorted(ids))
                all_ids.update(ids)
            result["score_usage"] = await usage_for(rig.recorder, sorted(all_ids))
        preserve(path / "result.json", result)


def pilot_review(review, proposal, *, binding, case, facts, profile):
    require(
        review["binding"] == binding
        and review["proposal_digest"] == quality_identity_digest(proposal)
        and review["review_kind"] == "AGENT_ASSESSED"
        and review["approved"] is True
        and bool(review["rationale"]),
        "pilot_review_binding_changed",
    )
    result = checked_annotation(review["final_annotation"], case, facts, profile)
    require(
        set(review["reviewed_requirement_ids"]) == {r.requirement_id for r in result.requirements},
        "pilot_review_incomplete",
    )
    return result.model_dump(mode="json")


async def run_phase(rig, root, directory, inputs, frozen, *, pilot):
    phase = directory / ("pilot" if pilot else "formal")
    phase.mkdir(mode=0o700, exist_ok=True)
    if not (phase / "started.json").exists():
        preserve(phase / "started.json", {"usage": await rig.recorder.check_admission(after=True)})
    if (phase / "audit.json").exists():
        return report(root, pilot=pilot)
    cases = cases_for(root, inputs, pilot)
    annotations, pending_reviews = {}, []
    cache = root / "a" / "pilot-annotations"
    if pilot:
        cache.mkdir(mode=0o700, exist_ok=True)
    for case in cases:
        if pilot:
            path = cache / case.case_id
            value = await review_call(
                rig, path, "annotate", annotation_payload(case, rig.facts, rig.profile_evidence)
            )
            proposal = locate_annotation(value, case)
            preserve(path / "proposal.json", proposal)
            review_path = path / "review.json"
            if not review_path.exists():
                pending_reviews.append(case.case_id)
                continue
            annotations[case.case_id] = pilot_review(
                read_private_json(review_path),
                proposal,
                binding=inputs.digest,
                case=case,
                facts=rig.facts,
                profile=rig.profile_evidence,
            )
        else:
            annotations[case.case_id] = frozen["annotations"][case.case_id]["annotation"]
    require(not pending_reviews, "pilot_annotations_require_agent_review")
    preserve(phase / "annotations.json", annotations)
    planned = samples([c.case_id for c in cases], pilot=pilot)
    for start in range(0, len(planned), 3):
        block = planned[start : start + 3]
        case = next(c for c in cases if c.case_id == block[0]["case_id"])
        block_id = f"{case.case_id}-r{block[0]['repeat']}"
        session_path = phase / f"{block_id}-session.json"
        request = SessionCreateV1(
            profile_version_id=UUID(rig.fixture["profile_version_id"]),
            preference_version=rig.fixture["preference_version"],
            project_ids=tuple(UUID(v) for v in rig.fixture["project_ids"]),
            job=JobInputV1(source="paste", text=case.jd),
            budget=GenerationBudgetV1(
                max_model_calls=12, max_tool_calls=0, max_cost_cny=Decimal("100")
            ),
        )
        command_path = phase / f"{block_id}-command.json"
        if not command_path.exists():
            preserve(command_path, {"id": str(uuid4())})
        created = await rig.store.create(
            rig.tenant, request, UUID(read_private_json(command_path)["id"])
        )
        sid = created.receipt.resource_id
        preserve(session_path, {"session_id": str(sid), "run_id": str(created.receipt.run_id)})
        generation_inputs = await rig.store.execution_inputs(rig.tenant, sid)
        generation_inputs = replace(
            generation_inputs,
            facts=tuple(sorted(generation_inputs.facts, key=lambda f: str(f.version_id))),
        )
        require(
            {str(f.version_id): (f.claim, f.kind, f.conditions) for f in generation_inputs.facts}
            == {f["version_id"]: (f["claim"], f["kind"], f["conditions"]) for f in rig.facts},
            "runtime_facts_changed",
        )
        for sample in block:
            await generate_sample(rig, sample, generation_inputs, phase / sample["sample_id"])
        await score_block(
            rig,
            phase,
            block,
            case,
            annotations[case.case_id],
            getattr(rig, "score_reuse_phase", None) if pilot else None,
        )
        print(
            json.dumps(
                {
                    "phase": "pilot" if pilot else "formal",
                    "block": block_id,
                    "recorded": start + 3,
                    "planned": len(planned),
                }
            ),
            flush=True,
        )
    preserve(
        phase / "audit.json",
        {
            "usage": await rig.recorder.check_admission(after=True),
            "artifact_sha256": {
                str(p.relative_to(phase)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(phase.rglob("*.json"))
            },
        },
    )
    summary = report(root, pilot=pilot)
    preserve(phase / "report.json", summary)
    return summary


async def execute(
    root, *, action, credentials_path, ci_path, reuse_source=None, reuse_score_source=None
):
    original, inputs, frozen, d_evidence = prerequisites(root)
    identity = source()
    directory = directory_for(root, identity)
    if (directory / "manifest.json").exists():
        ci = read_private_json(directory / "manifest.json")["ci"]
    else:
        require(action == "a-pilot" and ci_path is not None, "a_pilot_required")
        ci = read_private_json(ci_path)
    validate_ci(ci, identity["source_sha"])
    manifest = config(root, inputs, frozen, identity, ci, d_evidence)
    existing = directory / "manifest.json"
    bound_reuse = read_private_json(existing).get("reuse_source") if existing.exists() else None
    require(reuse_source is None or bound_reuse in (None, reuse_source), "reuse_binding_changed")
    manifest["reuse_source"] = reuse_source or bound_reuse
    bound_score = (
        read_private_json(existing).get("reuse_score_source") if existing.exists() else None
    )
    require(
        reuse_score_source is None or bound_score in (None, reuse_score_source),
        "reuse_score_binding_changed",
    )
    manifest["reuse_score_source"] = reuse_score_source or bound_score
    (root / "a").mkdir(mode=0o700, exist_ok=True)
    directory.mkdir(mode=0o700, exist_ok=True)
    preserve(directory / "manifest.json", manifest)
    require(
        (root / "database-owner.json").exists() and (root / "ledger-binding.json").exists(),
        "original_database_missing",
    )
    if action in ("a-run", "a-resume"):
        locked = read_private_json(directory / "frozen.json")
        require(locked["manifest_digest"] == quality_identity_digest(manifest), "a_freeze_changed")
    with tracing_context(enabled=False):
        async with ResumableDatabase(root, original["image_id"]) as url:
            engine = create_database_engine(SecretStr(url))
            sessions = create_session_factory(engine)
            bundle = recorder = None
            try:
                binding = read_private_json(root / "ledger-binding.json")
                tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
                    workspace_id=UUID(binding["workspace_id"]),
                    actor_user_id=UUID(binding["actor_user_id"]),
                )

                def source_check():
                    require(source() == identity, "a_source_changed")
                    require(load_inputs(root).digest == inputs.digest, "a_inputs_changed")

                recorder = ExperimentRecorder(
                    sessions,
                    tenant,
                    root=root,
                    inputs=inputs,
                    database_identity=quality_identity_digest(
                        read_private_json(root / "database-owner.json")
                    ),
                    source_check=source_check,
                )
                await recorder.initialize()
                await recorder.check_admission(after=True)
                if action == "a-freeze":
                    pilot = report(root, pilot=True)
                    require(pilot["status"] == "PASS", "pilot_incomplete")
                    for arm in ("one_shot", "selection", "pathfinder"):
                        require(
                            any(
                                r["sample"]["arm"] == arm and r["score_status"] == "ASSESSED"
                                for r in pilot["samples"].values()
                            ),
                            "pilot_arm_unverified",
                        )
                    value = {
                        "manifest_digest": quality_identity_digest(manifest),
                        "pilot_digest": quality_identity_digest(pilot),
                        "usage": await recorder.check_admission(after=True),
                    }
                    preserve(directory / "frozen.json", value)
                    return {"status": "PASS", "phase": "A_FROZEN"}
                values = credentials(credentials_path)
                bundle = create_qwen_adapters(
                    api_key=values["DASHSCOPE_API_KEY"], workspace_id=values["PF_QWEN_WORKSPACE_ID"]
                )
                factory = LLMFactory(
                    recorder=recorder,
                    chat_adapter=bundle.chat,
                    embedding_adapter=bundle.embedding,
                    provider="qwen",
                )
                fixture = await seed(sessions, tenant, root, directory)
                raw = (root / "inputs/resume.tex").read_bytes()
                template = TemplateIdentity(
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    preamble_sha256=hashlib.sha256(raw.split(b"\\begin{document}")[0]).hexdigest(),
                )
                rig = SimpleNamespace(
                    sessions=sessions,
                    tenant=tenant,
                    fixture=fixture,
                    store=SqlAlchemyResumeGenerationStore(sessions),
                    revisions=SqlAlchemyResumeRevisionStore(sessions),
                    reader=SqlAlchemyRunExecutionReader(sessions),
                    artifacts=SqlAlchemyResumeArtifactStore(
                        sessions,
                        expected_source_sha256=template.source_sha256,
                        expected_preamble_sha256=template.preamble_sha256,
                    ),
                    identity=template,
                    source_bytes=raw,
                    recorder=recorder,
                    factory=factory,
                    facts=read_private_json(root / "inputs/facts.json")["facts"],
                    profile_evidence=profile_evidence(
                        read_private_json(root / "inputs/profile.json")
                    ),
                    score_reuse_phase=score_origin(root, manifest["reuse_score_source"], manifest)
                    if manifest["reuse_score_source"]
                    else None,
                    seed=inputs.blind_seed,
                )
                if action == "a-pilot" and manifest["reuse_source"]:
                    await reuse_pilot(
                        rig, root, directory, manifest["reuse_source"], manifest, usage_for
                    )
                return await run_phase(
                    rig, root, directory, inputs, frozen, pilot=action == "a-pilot"
                )
            except BaseException:
                usage = await recorder.measurement() if recorder is not None else None
                publish(
                    directory / f"partial-{uuid4().hex}.json",
                    {"status": "PARTIAL", "usage": usage, "action": action},
                )
                raise
            finally:
                if bundle is not None:
                    await bundle.aclose()
                await engine.dispose()
