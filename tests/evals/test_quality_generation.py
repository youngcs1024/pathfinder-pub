"""E4.6 artifact, privacy, mode and graph-observation regressions (offline)."""

import asyncio
import json
import os
from time import monotonic

import pytest
from pydantic import ValidationError

from app.llm.ports import ModelToolCall
from tests.evals.live_chat import SECRET_CANARY
from tests.evals.quality_contracts import (
    QualityGenerationPolicyV1,
    QualityModelPayloadV1,
    QualityPrivateCaseV1,
    QualityPrivateOutputV1,
    QualityPrivateSourcesV1,
    QualityPrivateSourceV1,
)
from tests.evals.quality_dataset import project_model_payload, quality_digest
from tests.evals.quality_generation import FrozenQualityWeb, _ObservedTools
from tests.evals.quality_generation_fixtures import generation_inputs
from tests.evals.quality_generation_support import (
    GenerationArtifacts,
    GenerationGuard,
    GenerationSafetyError,
    QualityGenerationError,
    artifact_bytes,
)
from tests.evals.quality_run import run_quality_generation


def bundle(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    return GenerationArtifacts.reserve(
        tmp_path / "public",
        root,
        "experiment",
        GenerationGuard((SECRET_CANARY,)),
        tmp_path / "repository",
    )


def private_case():
    return QualityPrivateCaseV1(
        case_id="example",
        repeat_index=0,
        input=QualityModelPayloadV1(mode="research", query="private body canary"),
        resume_alias=None,
        web_scenario_alias=None,
        tools=(),
        output_digest=None,
        failure_type=None,
    )


def test_private_bytes_digest_permissions_and_create_only(tmp_path):
    artifacts = bundle(tmp_path)
    payload = private_case()
    digest = artifacts.write("case-0000.json", payload, private=True)
    path = artifacts.private_dir / "case-0000.json"
    assert quality_digest(path.read_bytes()) == digest == artifacts.files[0].digest
    assert artifacts.files[0].byte_count == path.stat().st_size
    assert path.stat().st_mode & 0o777 == 0o600
    assert artifacts.private_dir.stat().st_mode & 0o777 == 0o700
    assert artifacts.public_dir.stat().st_mode & 0o777 == 0o700
    assert "private body canary" in path.read_text()
    with pytest.raises(QualityGenerationError, match="artifact_publication_failed"):
        artifacts.write("case-0000.json", payload, private=True)
    assert quality_digest(path.read_bytes()) == digest
    assert len(artifacts.files) == 1


@pytest.mark.parametrize("kind", ["inside_repo", "mode", "symlink", "reuse", "overlap"])
def test_unsafe_bundle_directories_rejected(tmp_path, kind):
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    public = tmp_path / "public"
    if kind == "inside_repo":
        root = repo / "private"
        root.mkdir(mode=0o700)
    elif kind == "mode":
        root.chmod(0o755)
    elif kind == "symlink":
        link = tmp_path / "linked"
        link.symlink_to(root, target_is_directory=True)
        root = link
    elif kind == "reuse":
        (root / "experiment").mkdir(mode=0o700)
    elif kind == "overlap":
        public = root / "public"
    with pytest.raises(QualityGenerationError, match="artifact_reservation_failed"):
        GenerationArtifacts.reserve(public, root, "experiment", GenerationGuard(), repo)
    assert not public.exists()


def test_symlink_file_and_body_publication_rejected(tmp_path):
    artifacts = bundle(tmp_path)
    target = tmp_path / "target"
    target.write_text("unchanged")
    (artifacts.private_dir / "case-0000.json").symlink_to(target)
    with pytest.raises(QualityGenerationError, match="artifact_publication_failed"):
        artifacts.write("case-0000.json", private_case(), private=True)
    assert target.read_text() == "unchanged"
    with pytest.raises(QualityGenerationError, match="artifact_publication_failed"):
        artifacts.write("case-0001.json", private_case())
    with pytest.raises(QualityGenerationError, match="unsupported_artifact_contract"):
        artifact_bytes({"query": "private body canary"}, private=False)
    assert not tuple(artifacts.public_dir.iterdir())


def test_secret_output_never_written_even_to_private_bundle(tmp_path):
    artifacts = bundle(tmp_path)
    output = QualityPrivateOutputV1(
        case_id="example", repeat_index=0, output={"text": SECRET_CANARY}
    )
    with pytest.raises(GenerationSafetyError):
        artifacts.write("output-0000.json", output, private=True)
    assert artifacts.guard.secret_leak == 1
    assert not tuple(artifacts.private_dir.iterdir())


def test_failed_write_retains_partial_identity_without_registering_complete_file(
    tmp_path, monkeypatch
):
    artifacts = bundle(tmp_path)

    def fail(_fd):
        raise OSError("private body canary")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(QualityGenerationError, match=r"^artifact_publication_failed$"):
        artifacts.write("case-0000.json", private_case(), private=True)
    assert (artifacts.private_dir / "case-0000.json").exists()
    assert not artifacts.files


async def test_frozen_web_is_case_bound_and_has_no_gold_or_query_matching():
    source = QualityPrivateSourceV1(
        source_alias="role",
        kind="web",
        text="synthetic evidence",
        digest=quality_digest(b"synthetic evidence"),
    )
    port = FrozenQualityWeb(source)
    one = await port.search("first wording", 1, monotonic() + 5)
    two = await port.search("unrelated rewrite", 8, monotonic() + 5)
    assert one == two and one[0].snippet == source.text
    assert str(one[0].url) == "https://quality.invalid/role"
    assert await FrozenQualityWeb(None).search("anything", 1, monotonic() + 5) == ()


def test_model_projection_excludes_gold_and_rejected_scope():
    args = generation_inputs(None, None)
    case = next(c for c in args["dataset"].cases if c.case_id == "mixed_alpha")
    assert set(project_model_payload(case).model_dump()) == {"mode", "query"}
    for field in ("required_unit_ids", "expected_behavior", "rubric", "scope_expectation"):
        assert field not in project_model_payload(case).model_dump_json()
    rejected = next(c for c in args["dataset"].cases if c.case_id == "scope_rejected")
    with pytest.raises(ValueError, match="scope_rejected"):
        project_model_payload(rejected)


@pytest.mark.parametrize(
    "change",
    [
        {"provider_mode": "qwen"},
        {"confirm_disposable_database": False},
        {"git_probe": lambda: "b" * 40},
    ],
)
async def test_preflight_stops_before_database_or_publication(tmp_path, change):
    args = generation_inputs(tmp_path / "public", tmp_path / "private", selected=["mixed_alpha"])
    args.update(change)

    def no_database():
        pytest.fail("preflight must not open a database")

    with pytest.raises(QualityGenerationError, match=r"^generation_preflight_failed$"):
        await run_quality_generation(no_database, **args)
    assert not (tmp_path / "public").exists()


def test_generation_policy_rejects_unbounded_and_new_provider_parameters():
    policy = generation_inputs(None, None)["policy"]
    for change in ({"case_timeout_seconds": float("nan")}, {"max_tool_calls": 9}, {"seed": 1}):
        with pytest.raises(ValidationError):
            QualityGenerationPolicyV1.model_validate_json(
                json.dumps({**policy.model_dump(mode="json"), **change})
            )


async def test_tool_budget_stop_prevents_dispatch_and_failed_delivery_is_retained():
    class Runtime:
        calls = 0

        def validate_call(self, call):
            pass

        async def execute(self, call):
            self.calls += 1
            raise asyncio.CancelledError

    class Recorder:
        stop_reason = "budget_exhausted"

    delegate, recorder = Runtime(), Recorder()
    runtime = _ObservedTools(delegate, GenerationGuard(), "case.0.", recorder)
    call = ModelToolCall(call_id="one", name="search_web", arguments={"query": "synthetic"})
    with pytest.raises(QualityGenerationError, match="generation_stopped"):
        await runtime.execute(call)
    assert delegate.calls == 0
    recorder.stop_reason = None
    with pytest.raises(asyncio.CancelledError):
        await runtime.execute(call)
    assert delegate.calls == 1
    assert runtime.records[0].result is None and not runtime.records[0].delivered


def test_private_source_contract_never_serializes_as_public():
    with pytest.raises(QualityGenerationError, match="unsupported_artifact_contract"):
        artifact_bytes(QualityPrivateSourcesV1(sources=()), private=False)


async def test_generation_identity_drift_fails_before_any_database_access(tmp_path):
    args = generation_inputs(tmp_path / "public", tmp_path / "private", selected=["mixed_alpha"])
    for field, value in (
        ("prompt_digest", "sha256:" + "f" * 64),
        ("configuration_digest", "sha256:" + "e" * 64),
        ("graph_version", "unsupported-graph"),
    ):
        changed = {**args, "manifest": args["manifest"].model_copy(update={field: value})}

        def forbidden():
            pytest.fail("identity drift reached database")

        with pytest.raises(QualityGenerationError, match="generation_preflight_failed"):
            await run_quality_generation(forbidden, **changed)
    assert not (tmp_path / "public").exists()


def test_same_experiment_cannot_be_reserved_twice(tmp_path):
    artifacts = bundle(tmp_path)
    with pytest.raises(QualityGenerationError, match="artifact_reservation_failed"):
        GenerationArtifacts.reserve(
            tmp_path / "public2",
            artifacts.private_dir.parent,
            "experiment",
            GenerationGuard(),
            tmp_path / "repository",
        )
    assert not (tmp_path / "public2").exists()


async def test_observed_tool_integrity_failure_blocks_later_model_call():
    from tests.evals.quality_generation import _ObservedChat

    class Model:
        async def invoke(self, *args):
            pytest.fail("a model call followed a tool accounting failure")

    class Recorder:
        stop_reason = None

    guard = GenerationGuard(tool_accounting_complete=False)
    model = _ObservedChat(Model(), guard, "case.0.", Recorder())
    with pytest.raises(QualityGenerationError, match="generation_integrity_stopped"):
        await model.invoke((), (), {"graph_node": "plan"})


@pytest.mark.parametrize("value", [None, True, "postgresql://localhost/test"])
async def test_candidate_requires_owned_handle_before_publication(tmp_path, value):
    args = generation_inputs(tmp_path / "public", tmp_path / "private", arm="candidate")
    args["owned_database"] = value
    with pytest.raises(Exception) as error:
        await run_quality_generation(None, **args)
    assert type(error.value).__name__ in {"QualityGenerationError", "ExperimentDatabaseError"}
    assert not (tmp_path / "public").exists()


def test_candidate_identity_distinct_legacy_configuration_unchanged():
    from tests.evals.quality_generation import (
        generation_configuration_digest,
        generation_prompt_digest,
    )
    from tests.evals.quality_generation_fixtures import generation_factory

    factory = generation_factory()
    baseline = generation_inputs(None, None, factory=factory)
    candidate = generation_inputs(None, None, factory=factory, arm="candidate")
    assert baseline["manifest"].graph_version == "pathfinder-research-v6"
    assert candidate["manifest"].graph_version == "pathfinder-research-e7a-exp-v1"
    assert generation_prompt_digest() == generation_prompt_digest(arm="baseline")
    assert (
        generation_configuration_digest(baseline["policy"], factory)
        == baseline["manifest"].configuration_digest
    )
    assert candidate["manifest"].configuration_digest != baseline["manifest"].configuration_digest
    assert candidate["manifest"].prompt_digest != baseline["manifest"].prompt_digest


@pytest.mark.parametrize(
    "index,changes",
    [
        (0, {"closed": True}),
        (0, {"active": True}),
        (0, {"stop": "integrity"}),
        (True, {}),
        (1, {}),
        (-1, {}),
    ],
)
async def test_slot_admission_rejects_before_execution(index, changes):
    from types import SimpleNamespace

    from tests.evals.quality_generation import execute_generation_slot

    state = SimpleNamespace(
        closed=False,
        active=False,
        stop=None,
        complete_representation=True,
        results=[],
        manifest=SimpleNamespace(execution_order=[object()]),
    )
    for key, value in changes.items():
        setattr(state, key, value)
    with pytest.raises(QualityGenerationError, match="generation_slot_unavailable"):
        await execute_generation_slot(state, index)


async def test_factory_observation_spans_nodes_and_preserves_retry_grouping():
    from dataclasses import fields
    from types import SimpleNamespace
    from uuid import uuid4

    from app.llm.factory import LLMFactory
    from app.llm.invocations import LLMInvocationContext
    from app.llm.ports import ChatModelResult, ProviderAdapterError
    from tests.evals.live_chat import CURRENT_LOGICAL_CALL
    from tests.evals.quality_generation import _GenerationFactory, _ObservedChat
    from tests.evals.quality_generation_fixtures import generation_factory
    from tests.evals.test_quality_e7a_assessment import Adapter
    from tests.unit.llm.test_factory import CHAT_MESSAGES, CHAT_METADATA

    seen = []

    class RetryAdapter(Adapter):
        async def invoke(self, messages, tools, metadata, *, attempt):
            seen.append(CURRENT_LOGICAL_CALL.get())
            return await super().invoke(messages, tools, metadata, attempt=attempt)

    adapter = RetryAdapter(
        ProviderAdapterError(category="provider_timeout", retryable=True),
        ChatModelResult(content="ok"),
        ChatModelResult(content="ok"),
    )
    factory = generation_factory(adapter)
    context = LLMInvocationContext(uuid4(), uuid4(), run_id=uuid4())
    observed = _ObservedChat(
        factory.create_chat_model(context),
        GenerationGuard(),
        "case.0.",
        SimpleNamespace(stop_reason=None),
    )
    wrapper = _GenerationFactory(
        **{f.name: getattr(factory, f.name) for f in fields(LLMFactory)}, observed_chat=observed
    )
    for node in ("evidence_assessment", "write_report"):
        await wrapper.create_chat_model(context).invoke(
            CHAT_MESSAGES, (), {**CHAT_METADATA, "graph_node": node}
        )
    assert seen == ["case.0.chat.1", "case.0.chat.1", "case.0.chat.2"]
    assert observed.count == 2
