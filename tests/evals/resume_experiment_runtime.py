"""Only preparation runs here. Formal A/B/C/D execution is deliberately absent."""

import json
from types import SimpleNamespace
from uuid import uuid4

from langsmith import tracing_context
from pydantic import SecretStr, ValidationError

from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.provisioning import ProvisioningService
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatMessage
from app.llm.qwen_adapters import create_qwen_adapters
from tests.evals.product_acceptance_contracts import (
    AcceptanceError,
    publish,
    read_private_json,
    require,
)
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
    PROMPTS,
    annotation_payload,
    blind_packets,
    checked_assessment,
    profile_evidence,
    score_metrics,
)
from tests.evals.resume_live_environment import ResumableDatabase


def smoke_inputs():
    case = SimpleNamespace(jd="要求:实现分页查询。")
    facts = [
        {
            "version_id": "synthetic_fact",
            "claim": "实现分页查询。",
            "conditions": {"scope": "仅测试代码,不证明部署或性能。"},
        }
    ]
    annotation = checked_annotation(
        {
            "requirements": [
                {
                    "requirement_id": "pagination",
                    "quote": "实现分页查询",
                    "start": 3,
                    "end": 9,
                    "kind": "explicit",
                    "applicable": True,
                    "support": "full",
                    "fact_version_ids": ["synthetic_fact"],
                    "necessary_conditions": ["仅测试代码"],
                    "rationale": "测试代码明确实现分页。",
                }
            ]
        },
        case,
        facts,
    )
    candidates = [
        {"sample_id": "supported", "text": "测试代码实现分页查询,未验证部署或性能。"},
        {"sample_id": "fabricated", "text": "已上线服务百万用户,响应时间降低90%。"},
    ]
    packets, mapping = blind_packets(
        candidates, case=case, facts=facts, annotation=annotation, seed=20260927
    )
    return facts, annotation, packets, mapping


async def invoke_stage(root, *, stage, task, payload, model, recorder, binding):
    done = root / f"{stage}-response.json"
    started = root / f"{stage}-started.json"
    payload_digest = quality_identity_digest(payload)
    if done.exists():
        result = read_private_json(done)
        require(
            result["binding"] == binding
            and result["payload_digest"] == payload_digest
            and result["prompt_digest"] == PROMPT_DIGESTS[task],
            "stage_binding_changed",
        )
        await recorder.check_admission(after=True)
        require(
            result["finish_status"] == "completed" and not result["tool_calls"],
            "incomplete_model_response",
        )
        return normalize_json(result["content"])
    require(not started.exists(), "interrupted_stage_requires_reconciliation")
    before = await recorder.check_admission()
    publish(
        started,
        {
            "binding": binding,
            "payload_digest": payload_digest,
            "prompt_digest": PROMPT_DIGESTS[task],
            "before": before,
        },
    )
    response = await model.invoke(
        (
            ChatMessage(role="system", content=PROMPTS[task]),
            ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ),
        (),
        {
            "task": "resume_experiment_preparation",
            "graph_node": task,
            "prompt_version": PROMPT_DIGESTS[task],
        },
    )
    after = await recorder.snapshot()
    publish(
        done,
        {
            "binding": binding,
            "payload_digest": payload_digest,
            "prompt_digest": PROMPT_DIGESTS[task],
            "content": response.content,
            "finish_status": response.finish_status,
            "tool_calls": bool(response.tool_calls),
            "invocation_ids": sorted(set(after["invocation_ids"]) - set(before["invocation_ids"])),
            "after": after,
        },
    )
    require(
        response.finish_status == "completed" and not response.tool_calls,
        "incomplete_model_response",
    )
    return normalize_json(response.content or "")


async def prepare_annotations(root, inputs, model, recorder, facts):
    from tests.evals.resume_experiments import preserve

    profile = profile_evidence(read_private_json(root / "inputs" / inputs.profile_file))
    for case in inputs.cases:
        stage = f"annotation-{case.case_id}"
        value = await invoke_stage(
            root,
            stage=stage,
            task="annotate",
            payload=annotation_payload(case, facts, profile),
            model=model,
            recorder=recorder,
            binding=inputs.digest,
        )
        proposal = locate_annotation(value, case)
        preserve(root / f"{stage}-proposal.json", proposal)
        try:
            annotation = checked_annotation(proposal, case, facts, profile)
        except (AcceptanceError, ValidationError):
            # Preparation annotations are proposals, never accepted scoring answers.
            # Keep invalid references for explicit agent review; no model repair or resampling.
            preserve(
                root / f"{stage}-validation.json",
                {
                    "status": "REQUIRES_AGENT_REVIEW",
                    "proposal_digest": quality_identity_digest(proposal),
                },
            )
        else:
            preserve(root / f"{stage}.json", annotation.model_dump(mode="json"))
    smoke_facts, annotation, packets, mapping = smoke_inputs()
    scores = {}
    for packet in packets:
        stage = f"smoke-{packet['blind_id']}"
        value = await invoke_stage(
            root,
            stage=stage,
            task="score",
            payload=packet,
            model=model,
            recorder=recorder,
            binding=inputs.digest,
        )
        checked = checked_assessment(
            value, text=packet["candidate"], facts=smoke_facts, annotation=annotation
        )
        metrics = score_metrics(checked, content_ok=True)
        scores[mapping[packet["blind_id"]]] = metrics
        preserve(
            root / f"{stage}.json",
            {"assessment": checked.model_dump(mode="json"), "metrics": metrics},
        )
    require(
        scores["supported"]["usable"] and not scores["fabricated"]["usable"],
        "semantic_smoke_failed",
    )
    preserve(
        root / "preparation-complete.json",
        {
            "binding": inputs.digest,
            "case_ids": [c.case_id for c in inputs.cases],
            "smoke": scores,
            "formal_samples_executed": 0,
        },
    )


async def execute(root, manifest, inputs, credentials_path, *, audit_only=False):
    with tracing_context(enabled=False):
        async with ResumableDatabase(root, manifest["image_id"]) as url:
            return await execute_database(
                root, inputs, url, credentials_path, audit_only=audit_only
            )


async def execute_database(root, inputs, url, credentials_path, *, audit_only=False):
    from tests.evals.resume_experiments import verify

    facts = read_private_json(root / "inputs" / inputs.facts_file)["facts"]
    engine = create_database_engine(SecretStr(url))
    recorder = bundle = None
    try:
        sessions = create_session_factory(engine)
        owner = await ProvisioningService(
            SqlAlchemyProvisioningStore(sessions)
        ).provision_personal_workspace(f"resume-experiments-{inputs.allocation_id}")
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        recorder = ExperimentRecorder(
            sessions,
            tenant,
            root=root,
            inputs=inputs,
            database_identity=quality_identity_digest(
                read_private_json(root / "database-owner.json")
            ),
            source_check=lambda: verify(root),
        )
        await recorder.initialize()
        if not audit_only:
            await recorder.check_admission(after=True)
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
            context = LLMInvocationContext(tenant.workspace_id, tenant.actor_user_id)
            await prepare_annotations(
                root, inputs, factory.create_chat_model(context), recorder, facts
            )
        return await recorder.snapshot()
    except BaseException:
        usage = None
        if recorder is not None:
            try:
                usage = await recorder.snapshot()
            except Exception:
                pass
        publish(
            root / f"partial-{uuid4().hex}.json",
            {
                "status": "PARTIAL",
                "binding": inputs.digest,
                "usage": usage,
                "accounting_verified": usage is not None,
                "formal_samples_executed": 0,
            },
        )
        raise
    finally:
        if bundle is not None:
            await bundle.aclose()
        await engine.dispose()


def validate_reviews(root, inputs, facts, profile=None):
    frozen = {}
    for case in inputs.cases:
        proposal = root / f"annotation-{case.case_id}-proposal.json"
        original = read_private_json(
            proposal if proposal.exists() else root / f"annotation-{case.case_id}.json"
        )
        review = read_private_json(root / f"review-{case.case_id}.json")
        require(
            review["binding"] == inputs.digest
            and review["annotation_digest"] == quality_identity_digest(original),
            "review_binding_changed",
        )
        require(
            review["review_kind"] == "AGENT_ASSESSED"
            and review["approved"] is True
            and bool(review["rationale"]),
            "review_required",
        )
        final = checked_annotation(review.get("final_annotation", original), case, facts, profile)
        require(
            set(review["reviewed_requirement_ids"])
            == {r.requirement_id for r in final.requirements},
            "incomplete_requirement_review",
        )
        frozen[case.case_id] = {"annotation": final.model_dump(mode="json"), "review": review}
    return frozen
