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


def generation_inputs(output_dir, private_root, *, selected=None, factory=None, **changes):
    factory = factory or generation_factory()
    args = retrieval_inputs(output_dir, selected=selected, factory=factory)
    policy = QualityGenerationPolicyV1(retrieval=args["policy"])
    manifest = {
        **args["manifest"].model_dump(mode="json"),
        "suite_version": GENERATION_SUITE_VERSION,
        "model": factory.chat_model,
        "prompt_digest": generation_prompt_digest(),
        "configuration_digest": generation_configuration_digest(policy, factory),
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
        manifest=QualityRunManifestV1.model_validate_json(json.dumps(manifest)),
        private_root=private_root,
    )
    return args
