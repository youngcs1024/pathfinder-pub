"""Offline inputs for E4.6 tests; never suitable as live provenance."""

import json

from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.quality_contracts import QualityGenerationPolicyV1, QualityRunManifestV1
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_generation import (
    GENERATION_SUITE_VERSION,
    generation_configuration_digest,
    generation_prompt_digest,
)
from tests.evals.quality_retrieval_support import no_sleep, retrieval_inputs


def generation_factory(chat=None, embedding=None):
    return LLMFactory(
        _StrictMemoryInvocationRecorder(),
        chat or DeterministicResearchFakeChatAdapter(),
        embedding or FakeEmbeddingModel(),
        sleeper=no_sleep,
    )


def generation_inputs(
    output_dir, private_root, *, selected=None, factory=None, arm="baseline", **changes
):
    factory = factory or generation_factory(E7AFakeChatAdapter() if arm == "candidate" else None)
    args = retrieval_inputs(output_dir, selected=selected, factory=factory)
    policy = QualityGenerationPolicyV1(retrieval=args["policy"])
    manifest = {
        **args["manifest"].model_dump(mode="json"),
        "suite_version": GENERATION_SUITE_VERSION,
        "model": factory.chat_model,
        "prompt_digest": generation_prompt_digest(arm=arm),
        "configuration_digest": generation_configuration_digest(policy, factory, arm=arm),
        "graph_version": "pathfinder-research-e7a-exp-v1"
        if arm == "candidate"
        else args["manifest"].graph_version,
        "retrieval_policy_digest": quality_identity_digest(
            policy.retrieval.model_dump(mode="json")
        ),
        "measurement_scope": "contract" if factory.provider == "fake" else "generation",
        "web_mode": "frozen_fixture",
        "provider_attempt_cap": 1000,
        "input_token_cap": 1000000,
        "output_token_cap": 1000000,
        **changes,
    }
    args.update(
        policy=policy,
        arm=arm,
        manifest=QualityRunManifestV1.model_validate_json(json.dumps(manifest)),
        private_root=private_root,
    )
    return args


class E7AFakeChatAdapter(DeterministicResearchFakeChatAdapter):
    """Scripted integration responses from delivered evidence; not a semantic assessor."""

    def __init__(self, outcome="sufficient", on_call=None):
        self.outcome = outcome
        self.on_call = on_call
        self.calls = []

    async def invoke(self, messages, tools, metadata, *, attempt):
        self.calls.append((messages, tools, metadata, attempt))
        if self.on_call is not None:
            await self.on_call(messages, tools, metadata, attempt)
        node = metadata.get("graph_node")
        if node == "evidence_assessment":
            payload = json.loads(next(m.content for m in messages if m.role == "user"))
            evidence = payload["evidence"]
            outcome = self.outcome if evidence else "insufficient"
            result = {
                "outcome": outcome,
                "evidence_ids": [evidence[0]["evidence_id"]] if evidence else [],
                "gaps": []
                if outcome == "sufficient"
                else [
                    {
                        "code": "source_conflict"
                        if outcome == "conflicting"
                        else "missing_task_fact",
                        "topic": "Additional requested information",
                    }
                ],
            }
            return self._result(content=json.dumps(result))
        if node == "write_report":
            payload = json.loads(next(m.content for m in messages if m.role == "user"))
            outcome = payload["assessment"]["outcome"]
            evidence = {e["evidence_id"]: e for e in payload["evidence"]}
            output = dict(summary=[], findings=[], limitations=[], application_draft=None)
            if outcome != "insufficient":
                for index, key in enumerate(payload["assessment"]["evidence_ids"]):
                    item = evidence[key]
                    output["summary"].append(
                        {
                            "claim_id": f"summary-{index}",
                            "text": item["text"],
                            "citations": [{"source_id": item["source_id"], "evidence_id": key}],
                        }
                    )
            if outcome != "sufficient":
                output["limitations"] = [
                    {
                        "code": "conflicting_evidence"
                        if outcome == "conflicting"
                        else "insufficient_evidence",
                        "detail": "Requested information remains unresolved",
                    }
                ]
            elif payload["request"]["include_application_draft"]:
                key = payload["resume_evidence_ids"][0]
                item = evidence[key]
                output["application_draft"] = {
                    "paragraphs": [
                        {
                            "claim_id": "draft",
                            "text": item["text"],
                            "citations": [{"source_id": item["source_id"], "evidence_id": key}],
                        }
                    ]
                }
            return self._result(content=json.dumps(output))
        if node == "research_agent" and not any(m.role == "tool" for m in messages):
            content = messages[-1].content or ""
            if "<untrusted_gap_queries>" in content:
                from app.llm.ports import ModelToolCall

                # The gap block is a JSON array, not the original plan query.
                raw = content.split("<untrusted_gap_queries>\n", 1)[1].split(
                    "\n</untrusted_gap_queries>", 1
                )[0]
                query = json.loads(raw)[0]
                args = {"query": query["query"]}
                if query["tool_name"] == "search_web":
                    args["max_results"] = 8
                return self._result(
                    tool_calls=(
                        ModelToolCall(
                            call_id="gap-call",
                            name=query["tool_name"],
                            arguments=args,
                        ),
                    )
                )
        return await super().invoke(messages, tools, metadata, attempt=attempt)
