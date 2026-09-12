"""Shared E4.5 synthetic fixtures; does not execute providers or experiments on import."""

import json
from decimal import Decimal
from pathlib import Path

from app.domain.runs import CURRENT_GRAPH_VERSION
from app.llm.factory import LLMFactory
from app.llm.fake import FakeChatModel, FakeEmbeddingModel
from app.llm.ports import LOCKED_EMBEDDING_MODEL
from tests.evals.harness import _StrictMemoryInvocationRecorder
from tests.evals.quality_contracts import (
    QualityMappingV1,
    QualityRetrievalPolicyV1,
    QualityRunManifestV1,
)
from tests.evals.quality_dataset import (
    load_quality_dataset,
    quality_digest,
    quality_identity_digest,
)
from tests.evals.quality_run import (
    NO_CHAT_PROMPT_DIGEST,
    SUITE_VERSION,
    retrieval_configuration_digest,
)

PILOT = Path(__file__).resolve().parents[2] / "evals/datasets/quality_v1"
MAPPING = PILOT.parent / "quality_mappings/pilot_v1_heading_v1"
SOURCE_SHA = "a" * 40


async def no_sleep(_seconds):
    pass


def fake_factory(embedding=None):
    return LLMFactory(
        recorder=_StrictMemoryInvocationRecorder(),
        chat_adapter=FakeChatModel(),
        embedding_adapter=embedding or FakeEmbeddingModel(),
        sleeper=no_sleep,
    )


def retrieval_inputs(output_dir, *, factory=None, selected=None, **changes):
    dataset = load_quality_dataset(PILOT)
    factory = factory or fake_factory()
    policy = QualityRetrievalPolicyV1(unknown_attempt_reserve_cny=Decimal("0.001"))
    selected = selected if selected is not None else [case.case_id for case in dataset.cases]
    payload = dict(
        experiment_id="quality_test",
        execution_source_sha=SOURCE_SHA,
        suite_version=SUITE_VERSION,
        dataset_version=dataset.manifest.dataset_version,
        dataset_digest=dataset.manifest_digest,
        case_set_digest=quality_identity_digest(list(selected)),
        split_digest=quality_identity_digest(
            [[f.family_id, f.split] for f in dataset.manifest.families]
        ),
        rubric_version=dataset.rubric.rubric_version,
        rubric_digest=next(f.digest for f in dataset.manifest.files if f.role == "rubric"),
        model=LOCKED_EMBEDDING_MODEL,
        prompt_digest=NO_CHAT_PROMPT_DIGEST,
        graph_version=CURRENT_GRAPH_VERSION,
        embedding_profile=policy.embedding_profile,
        retrieval_policy_digest=quality_identity_digest(policy.model_dump(mode="json")),
        configuration_digest=retrieval_configuration_digest(policy, factory),
        measurement_scope="contract" if factory.provider == "fake" else "retrieval",
        llm_mode=factory.provider,
        web_mode="none",
        document_mode="fake_embedding_db" if factory.provider == "fake" else "real_embedding_db",
        selected_case_ids=list(selected),
        repeat_count=1,
        execution_order=[dict(case_id=case_id, repeat_index=0) for case_id in selected],
        cost_admission_budget_cny="10",
        provider_attempt_cap=100,
        input_token_cap=100_000,
        output_token_cap=100_000,
    )
    payload.update(changes)
    return dict(
        dataset_root=PILOT,
        dataset=dataset,
        mapping=QualityMappingV1.model_validate_json((MAPPING / "mapping.json").read_bytes()),
        mapping_rules_digest=quality_digest((MAPPING / "rules.md").read_bytes()),
        manifest=QualityRunManifestV1.model_validate_json(json.dumps(payload)),
        policy=policy,
        output_dir=output_dir,
        confirm_disposable_database=True,
        factory=factory,
        git_probe=lambda: SOURCE_SHA,
    )
