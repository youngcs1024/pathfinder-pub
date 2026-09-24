"""R7.1 contracts run in the existing eval CI branch without Docker or providers."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatModelResult, ProviderAdapterError
from tests.evals.product_acceptance_budget import admit, summarize
from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.quality_experiment_binding import ExperimentError
from tests.evals.resume_quality import load_dataset, prepare, verify
from tests.evals.resume_quality_baseline import common_input, one_shot
from tests.evals.resume_quality_contracts import (
    ComparisonBudget,
    LivePermission,
    coverage,
    input_change,
    review_template,
    validate_live_permission,
)
from tests.unit.agents.test_resume_generation import _inputs, _step
from tests.unit.llm.test_factory import _RecordingRecorder


def test_dataset_and_fixed_denominator():
    dataset = load_dataset()
    assert [c.relationship for c in dataset.cases] == ["close", "close", "different"]
    requirements = dataset.cases[0].supported_requirements
    assert coverage(requirements)["ratio"] == 0
    assert coverage(requirements, partially=("supported_core",))["ratio"] == 0
    assert coverage(requirements, fully=("supported_core",))["ratio"] == 1
    assert coverage(())["status"] == "NOT_APPLICABLE"
    assert coverage(())["ratio"] is None
    with pytest.raises(AcceptanceError):
        coverage(requirements, fully=("inferred",))
    with pytest.raises(AcceptanceError):
        coverage(requirements, fully=("supported_core",), partially=("supported_core",))


def test_review_missing_data_is_not_success_or_zero():
    review = review_template("sha256:" + "a" * 64)
    assert review["finalization_cost"]["human_minutes"] is None
    assert review["final_quality"]["pages"] is None
    assert review["final_quality"]["acceptable_for_use"] is None
    assert review["initial_quality"]["status"] == "NOT_REVIEWED"
    assert (
        input_change({"facts": [1]}, {"facts": [1, 2]})["same_input_advantage_claim_allowed"]
        is False
    )


def test_live_permission_requires_exact_material_environment_budget():
    budget = ComparisonBudget()
    digest = "sha256:" + "a" * 64
    kwargs = dict(material_digest=digest, environment_id="isolated-local", budget=budget)
    with pytest.raises(AcceptanceError):
        validate_live_permission(None, **kwargs)
    permit = LivePermission(
        **kwargs, approved_by="synthetic-reviewer", materials_may_leave_machine=True
    )
    validate_live_permission(permit, **kwargs)
    with pytest.raises(AcceptanceError):
        validate_live_permission(permit, **{**kwargs, "environment_id": "different"})
    with pytest.raises(AcceptanceError):
        validate_live_permission(
            permit, **{**kwargs, "budget": ComparisonBudget(provider_attempt_cap=101)}
        )
    with pytest.raises(ValueError):
        ComparisonBudget(cost_admission_budget_cny=Decimal("0"))


@pytest.mark.parametrize("problem", ["unknown", "unfinished", "attempts", "cost"])
def test_budget_is_fail_closed(problem):
    row = SimpleNamespace(
        status="succeeded",
        estimated_cost=Decimal("0.01"),
        token_usage={"input_tokens": 2, "output_tokens": 1},
    )
    if problem == "unknown":
        row.estimated_cost = None
    if problem == "unfinished":
        row.status = "started"
    budget = ComparisonBudget(provider_attempt_cap=1 if problem == "attempts" else 100)
    if problem == "cost":
        row.estimated_cost = Decimal("20")
    with pytest.raises(AcceptanceError):
        admit(summarize([row], provider="qwen"), budget)


async def test_one_shot_has_one_logical_call_with_counted_retry_and_private_fill():
    source, identity, inputs, _, _ = _inputs()
    content = inputs.profile_content.model_dump(mode="json")
    result = _step({key: content[key] for key in ("education", "projects", "skills")})
    recorder = _RecordingRecorder()

    class RetryAdapter(ScriptedFakeChatModel):
        count = 0

        async def invoke(self, messages, tools, metadata, *, attempt=None):
            self.count += 1
            assert "canary@example.test" not in str(messages)
            assert "Synthetic Candidate" not in str(messages)
            if self.count == 1:
                raise ProviderAdapterError(category="provider_timeout", retryable=True)
            return result

    adapter = RetryAdapter([result])

    async def no_sleep(_):
        pass

    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=adapter,
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
        sleeper=no_sleep,
    )
    model = factory.create_chat_model(LLMInvocationContext(uuid4(), uuid4()))
    output = await one_shot(model, inputs, source, identity)
    assert adapter.count == 2
    assert [e[0] for e in recorder.events] == ["prepare", "finalize"] * 2
    assert output["automatic_repairs"] == 0 and output["logical_generations"] == 1
    assert "canary@example.test" in output["tex"]
    assert output["common_input_digest"] == quality_identity_digest(
        common_input(inputs, identity.source_sha256)
    )
    assert common_input(inputs, identity.source_sha256) != common_input(
        replace(inputs, job_text="other"), identity.source_sha256
    )


async def test_b1_parse_failure_never_invokes_a_repair():
    source, identity, inputs, _, _ = _inputs()
    recorder = _RecordingRecorder()
    factory = LLMFactory(
        recorder=recorder,
        chat_adapter=ScriptedFakeChatModel([ChatModelResult(content="invalid JSON")]),
        embedding_adapter=FakeEmbeddingModel(),
        provider="fake",
    )
    with pytest.raises(ValueError):
        await one_shot(
            factory.create_chat_model(LLMInvocationContext(uuid4(), uuid4())),
            inputs,
            source,
            identity,
        )
    assert [e[0] for e in recorder.events] == ["prepare", "finalize"]


def test_prepare_does_not_call_provider_and_artifacts_are_create_only(tmp_path):
    root = tmp_path / "private"
    manifest = prepare(root)
    dataset, _ = verify(manifest)
    assert dataset == load_dataset()
    assert manifest["live_permission"] is None
    assert not (root / "started.json").exists()
    with pytest.raises(ExperimentError):
        publish(root / "manifest.json", manifest)
    with pytest.raises(AcceptanceError):
        verify({**manifest, "mode": "live"})
    with pytest.raises(AcceptanceError):
        verify({**manifest, "source_sha": "0" * 40})
