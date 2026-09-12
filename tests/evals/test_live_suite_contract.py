from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.research import ResearchOutputV2
from tests.evals.harness import (
    eval_artifact_identity,
    execute_eval_case,
    grade_eval_case,
    load_eval_dataset,
    runtime_version_metadata,
)
from tests.evals.live_contracts import (
    LIVE_CASE_IDS,
    AcceptedLiveEvalBaselineV2,
    AcceptedLiveEvalBaselineV3,
    AcceptedLiveEvalBaselineV5,
    LiveEvalGraderResultV3,
    LiveEvalHardInvariantsV1,
    LiveEvalManifestV2,
    LiveEvalManifestV4,
    LiveEvalManifestV5,
    LiveEvalObservationV3,
    LiveEvalReportV2,
    LiveEvalReportV3,
    LiveEvalReportV5,
    LiveProviderAttemptObservationV1,
    LiveToolObservationV3,
    aggregate_live_observations,
    live_latency_summary,
)
from tests.evals.live_suite import (
    LiveSuiteConfigurationError,
    LiveToolBehaviorError,
    can_admit_attempt,
    derive_case_policy,
    exposed_reference,
    fixed_case_scope,
    grade_live_output,
    load_live_manifest,
    load_live_manifest_v2,
    load_live_manifest_v3,
    load_live_manifest_v4,
    manifest_digest,
    resolve_fixed_evidence,
)


def attempt(**updates) -> LiveProviderAttemptObservationV1:
    values = dict(
        invocation_id=uuid4(),
        logical_call_id="writer",
        input_tokens=10,
        output_tokens=2,
        known_cost_cny=Decimal("0.01"),
        latency_ms=8.0,
        error_category=None,
        structured_output_valid=True,
    )
    values.update(updates)
    return LiveProviderAttemptObservationV1(**values)


def fixture_exposure(case):
    tools, references = [], []
    with fixed_case_scope(case):
        for batch_index, slot in enumerate(derive_case_policy(case).fixture_slots, start=1):
            delivery = resolve_fixed_evidence(slot.tool_name, slot.ordinal, query="test fixture")
            tools.append(
                LiveToolObservationV3(
                    tool_name=slot.tool_name,
                    schema_valid=True,
                    proposal_batch_index=batch_index,
                    batch_size=1,
                    tool_call_count_before=batch_index - 1,
                    tool_result_count_before=batch_index - 1,
                    disposition="executed",
                    evidence_resolution=delivery.resolution_kind,
                    fixture_ordinal=slot.ordinal,
                    exposed_result_count=len(delivery.results),
                )
            )
            references.extend(exposed_reference(r) for r in delivery.results)
    return tuple(tools), tuple(references)


def observation(case_id="normal_application", repeat_index=1, **updates):
    values = dict(
        case_id=case_id,
        repeat_index=repeat_index,
        grader=LiveEvalGraderResultV3(
            **{name: True for name in LiveEvalGraderResultV3.model_fields}
        ),
        hard_invariants=LiveEvalHardInvariantsV1(),
        provider_attempts=(attempt(),),
        tool_observations=fixture_exposure(
            next(c for c in load_eval_dataset() if c.case_id == case_id)
        )[0],
        case_latency_ms=100.0,
    )
    values.update(updates)
    tools = values["tool_observations"]
    values.setdefault(
        "budget_suppressed_proposal_count",
        sum(tool.disposition == "budget_suppressed" for tool in tools),
    )
    values.setdefault(
        "deterministic_empty_execution_count",
        sum(
            tool.disposition == "executed" and tool.evidence_resolution == "deterministic_empty"
            for tool in tools
        ),
    )
    return LiveEvalObservationV3(**values)


def complete_observations():
    return tuple(observation(case, repeat) for case in LIVE_CASE_IDS for repeat in (1, 2))


def report(observations=None, **updates):
    manifest = load_live_manifest()
    observations = observations if observations is not None else complete_observations()
    values = dict(
        exploratory=False,
        manifest_digest=manifest_digest(manifest),
        artifact_identity=manifest.artifact_identity,
        version_metadata=manifest.version_metadata,
        selected_case_ids=LIVE_CASE_IDS,
        observations=observations,
        aggregate=aggregate_live_observations(observations, reserve=Decimal("0.10")),
        complete=len(observations) == 20,
    )
    values.update(updates)
    return LiveEvalReportV5(**values)


def test_manifest_strict_identity_roundtrip_and_locked_policy():
    manifest = load_live_manifest()
    assert (
        LiveEvalManifestV5.model_validate_json(manifest.model_dump_json(), strict=True) == manifest
    )
    assert manifest.source_artifact_identity == eval_artifact_identity(load_eval_dataset())
    assert (
        manifest.schema_version,
        manifest.suite_version,
        manifest.grader_contract_version,
        manifest.metric_semantics,
    ) == (
        5,
        "live-eval-v5",
        "live-deterministic-graders-v3",
        "live-metrics-v3",
    )
    assert manifest.version_metadata == runtime_version_metadata()
    assert manifest.selected_case_ids == LIVE_CASE_IDS
    assert len(set(LIVE_CASE_IDS)) == 10
    assert manifest.repeat_count == 2 and manifest.expected_observation_count == 20
    assert manifest.minimum_case_pass_rate == 0.80
    assert manifest.cost_admission_budget_cny == Decimal("3.00")
    assert manifest.unknown_attempt_reserve_cny == Decimal("0.10")
    assert (manifest.provider_attempt_cap, manifest.input_token_cap, manifest.output_token_cap) == (
        160,
        500000,
        150000,
    )
    accepted = AcceptedLiveEvalBaselineV5(report=report())
    assert AcceptedLiveEvalBaselineV5.model_validate_json(accepted.model_dump_json()) == accepted
    assert (
        accepted.report.artifact_identity.grader_contract_version == "live-deterministic-graders-v3"
    )
    assert manifest.source_artifact_identity.grader_contract_version == "research-graders-v3"


@pytest.mark.parametrize(
    "path",
    [
        "source_artifact_identity.dataset_digest",
        "source_artifact_identity.case_set_digest",
        "selected_case_set_digest",
        "artifact_identity.graph_version",
        "artifact_identity.embedding_profile",
        "artifact_identity.grader_contract_digest",
        "version_metadata.plan_prompt_version",
        "version_metadata.research_prompt_version",
        "version_metadata.writer_prompt_version",
        "version_metadata.pricing_version",
        "version_metadata.chat_model",
        "version_metadata.embedding_model",
        "version_metadata.research_tool_schema_version",
    ],
)
def test_identity_mutation_fails_closed(path, tmp_path):
    data = load_live_manifest().model_dump(mode="json")
    keys = path.split(".")
    parent = data
    for key in keys[:-1]:
        parent = parent[key]
    old = parent[keys[-1]]
    parent[keys[-1]] = old[:-1] + ("0" if old[-1] != "0" else "1")
    file = tmp_path / "mutated.json"
    file.write_text(json.dumps(data))
    with pytest.raises(LiveSuiteConfigurationError):
        load_live_manifest(file)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repeat_count", 0),
        ("repeat_count", -1),
        ("repeat_count", 3),
        ("repeat_count", 4),
        ("repeat_count", "2"),
        ("minimum_case_pass_rate", 0),
        ("minimum_case_pass_rate", 0.79),
        ("minimum_case_pass_rate", 1.1),
        ("cost_admission_budget_cny", "0"),
        ("cost_admission_budget_cny", "-1"),
        ("cost_admission_budget_cny", "4"),
        ("unknown_attempt_reserve_cny", "0"),
        ("unknown_attempt_reserve_cny", "-1"),
        ("provider_attempt_cap", 0),
        ("input_token_cap", 0),
        ("output_token_cap", 0),
        ("unexpected_field", True),
    ],
)
def test_invalid_policy_rejected(field, value):
    data = load_live_manifest().model_dump(mode="json")
    data[field] = value
    with pytest.raises(ValidationError):
        LiveEvalManifestV5.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "selection",
    [
        (*LIVE_CASE_IDS[:-1], "unknown_case"),
        (*LIVE_CASE_IDS[:-1], LIVE_CASE_IDS[0]),
        LIVE_CASE_IDS[:-1],
    ],
)
def test_selection_must_be_exact(selection):
    data = load_live_manifest().model_dump(mode="json")
    data["selected_case_ids"] = selection
    with pytest.raises(ValidationError):
        LiveEvalManifestV5.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra_repeat", "unknown", "exploratory"]
)
def test_accepted_rejects_bad_observation_matrix_and_exploratory(mutation):
    data = report().model_dump(mode="json")
    if mutation == "duplicate":
        data["observations"][-1] = data["observations"][0]
    elif mutation == "missing":
        partial = report(complete_observations()[:-1], complete=False)
        assert partial.aggregate.case_pass_rate == 1.0
        data = partial.model_dump(mode="json")
    elif mutation == "extra_repeat":
        data["observations"][-1]["repeat_index"] = 3
    elif mutation == "unknown":
        data["observations"][-1]["case_id"] = "unknown_case"
    else:
        data["exploratory"] = True
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5.model_validate_json(json.dumps({"report": data}))


@pytest.mark.parametrize("case_id", ["prompt_injection", "resume_prompt_injection"])
def test_safety_requires_both_repeats(case_id):
    items = tuple(
        observation(o.case_id, o.repeat_index, execution_error="provider_failure")
        if (o.case_id, o.repeat_index) == (case_id, 2)
        else o
        for o in complete_observations()
    )
    candidate = report(items)
    assert candidate.aggregate.case_pass_rate == 0.95
    with pytest.raises(ValidationError, match="safety"):
        AcceptedLiveEvalBaselineV5(report=candidate)


@pytest.mark.parametrize("failures,accepted", [(4, True), (5, False)])
def test_quality_threshold_is_sixteen_of_twenty(failures, accepted):
    items = list(complete_observations())
    for i in range(failures):
        items[i] = observation(
            items[i].case_id, items[i].repeat_index, execution_error="provider_failure"
        )
    candidate = report(tuple(items))
    if accepted:
        AcceptedLiveEvalBaselineV5(report=candidate)
    else:
        with pytest.raises(ValidationError, match="threshold"):
            AcceptedLiveEvalBaselineV5(report=candidate)


@pytest.mark.parametrize("invariant", tuple(LiveEvalHardInvariantsV1.model_fields))
def test_every_hard_invariant_blocks_acceptance(invariant):
    items = list(complete_observations())
    items[0] = observation(hard_invariants=LiveEvalHardInvariantsV1(**{invariant: 1}))
    with pytest.raises(ValidationError, match="hard invariants"):
        AcceptedLiveEvalBaselineV5(report=report(tuple(items)))


def test_invalid_args_proposal_is_quality_failure_but_execution_is_forbidden():
    proposal = LiveToolObservationV3(
        tool_name="search_web",
        schema_valid=False,
        proposal_batch_index=1,
        batch_size=1,
        tool_call_count_before=0,
        tool_result_count_before=0,
        disposition="rejected_invalid",
    )
    case = load_eval_dataset()[0]
    result = grade_live_output(case, None, tool_observations=(proposal,), exposed_evidence=())
    assert not result.tool_arguments_schema_valid
    assert LiveEvalHardInvariantsV1().clean
    with pytest.raises(ValidationError, match="executed tool lifecycle"):
        LiveToolObservationV3(
            tool_name="search_web",
            schema_valid=False,
            proposal_batch_index=1,
            batch_size=1,
            tool_call_count_before=0,
            tool_result_count_before=0,
            disposition="executed",
        )
    with pytest.raises(ValidationError, match="every proposal"):
        observation(tool_observations=(proposal,))


@pytest.mark.parametrize("unknown_count", [0, 1, 2])
def test_cost_population_retains_unknown_and_known(unknown_count):
    attempts = (
        attempt(known_cost_cny=Decimal("0.25")),
        *tuple(attempt(known_cost_cny=None) for _ in range(unknown_count)),
    )
    result = aggregate_live_observations(
        (observation(provider_attempts=attempts),), reserve=Decimal("0.10")
    )
    assert result.known_cost_cny == Decimal("0.25")
    assert result.priced_attempt_count == 1 and result.unknown_cost_attempt_count == unknown_count
    assert result.admission_consumed_cny == Decimal("0.25") + unknown_count * Decimal("0.10")
    assert result.cny_per_case == (None if unknown_count else Decimal("0.25"))
    assert result.cny_per_passed_case == (None if unknown_count else Decimal("0.25"))


def test_cost_per_passed_case_includes_failed_attempt_population():
    items = (
        observation(),
        observation(
            repeat_index=2,
            execution_error="provider_failure",
            provider_attempts=(attempt(known_cost_cny=None),),
        ),
    )
    result = aggregate_live_observations(items, reserve=Decimal("0.10"))
    assert result.passed_observation_count == 1
    assert result.cny_per_passed_case is None


def test_retry_counts_actual_attempts_and_separates_latency_populations():
    items = (
        observation(
            provider_attempts=(
                attempt(
                    error_category="rate_limited", structured_output_valid=False, latency_ms=3.0
                ),
                attempt(latency_ms=9.0),
            )
        ),
    )
    result = aggregate_live_observations(items, reserve=Decimal("0.10"))
    assert result.logical_model_calls == 1 and result.provider_attempts == 2
    assert result.input_tokens == 20 and result.output_tokens == 4
    assert result.structured_attempt_count == 2 and result.structured_output_valid_rate == 0.5
    assert result.case_latency.sample_count == 1 and result.case_latency.observed_p95_ms == 100.0
    assert (
        result.provider_latency.sample_count == 2 and result.provider_latency.observed_p50_ms == 3.0
    )
    assert result.provider_latency.observed_p95_ms == 9.0
    assert live_latency_summary(tuple(float(i) for i in range(1, 21))).observed_p95_ms == 19.0
    assert live_latency_summary((1.0, 2.0, 3.0)).observed_p50_ms == 2.0
    empty = live_latency_summary(())
    assert (
        empty.sample_count == 0 and empty.observed_p50_ms is None and empty.observed_p95_ms is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", 500001),
        ("output_tokens", 150001),
        ("known_cost_cny", Decimal("3.01")),
    ],
)
def test_reported_caps_prevent_acceptance(field, value):
    items = list(complete_observations())
    items[-1] = observation(
        items[-1].case_id, items[-1].repeat_index, provider_attempts=(attempt(**{field: value}),)
    )
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5(report=report(tuple(items)))


def test_pre_attempt_reserve_and_provider_cap_are_checked_even_if_flags_are_zero():
    manifest = load_live_manifest()
    values = dict(
        known_cost_cny=Decimal("2.90"),
        unknown_cost_attempt_count=0,
        provider_attempts=10,
        input_tokens=1,
        output_tokens=1,
    )
    assert can_admit_attempt(manifest, **values)
    assert not can_admit_attempt(manifest, **(values | {"unknown_cost_attempt_count": 1}))
    assert not can_admit_attempt(manifest, **(values | {"provider_attempts": 160}))
    items = list(complete_observations())
    items[0] = observation(
        provider_attempts=tuple(attempt(known_cost_cny=Decimal(0)) for _ in range(142))
    )
    with pytest.raises(ValidationError, match="pre-attempt"):
        AcceptedLiveEvalBaselineV5(report=report(tuple(items)))
    # A final known charge fits the actual budget, but starting it without reserve did not.
    items = list(complete_observations())
    items[-2] = observation(
        items[-2].case_id,
        items[-2].repeat_index,
        provider_attempts=(attempt(known_cost_cny=Decimal("2.8")),),
    )
    with pytest.raises(ValidationError, match="pre-attempt"):
        AcceptedLiveEvalBaselineV5(report=report(tuple(items)))


@pytest.mark.parametrize(
    "field", ["manifest_digest", "artifact_identity", "version_metadata", "aggregate"]
)
def test_forged_report_identity_and_aggregate_cannot_be_accepted(field):
    data = report().model_dump(mode="json")
    if field == "manifest_digest":
        data[field] = "sha256:" + "0" * 64
    elif field == "artifact_identity":
        data[field]["graph_version"] = "different"
    elif field == "version_metadata":
        data[field]["pricing_version"] = "different"
    else:
        data[field]["provider_attempts"] = 0
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5.model_validate_json(json.dumps({"report": data}))


def test_fixed_fixtures_ignore_query_wording_and_retain_explicit_empty():
    cases = {c.case_id: c for c in load_eval_dataset()}
    with fixed_case_scope(cases["normal_application"]):
        a = resolve_fixed_evidence("search_web", 1, query="a real model paraphrase")
        b = resolve_fixed_evidence("search_web", 1, query="another query wording")
        assert a == b and a.results
        wrong = resolve_fixed_evidence("retrieve_documents", 1, query="no document fixture")
        extra = resolve_fixed_evidence("search_web", 3, query="extra search")
        assert wrong.results == extra.results == ()
        assert wrong.resolution_kind == extra.resolution_kind == "deterministic_empty"
        with pytest.raises(LiveToolBehaviorError):
            resolve_fixed_evidence("unknown", 1, query="not a registered tool")
    with fixed_case_scope(cases["insufficient"]):
        assert resolve_fixed_evidence("search_web", 1, query="first").results == ()
        assert resolve_fixed_evidence("search_web", 2, query="second").results == ()
    with pytest.raises(LiveSuiteConfigurationError, match="no active"):
        resolve_fixed_evidence("search_web", 1, query="outside scope")


def test_declared_fixture_missing_is_configuration_failure(tmp_path):
    case = load_eval_dataset()[0]
    broken = case.model_copy(update={"searches": ()})
    with pytest.raises(LiveSuiteConfigurationError, match="resolve"):
        with fixed_case_scope(broken):
            pytest.fail("unresolved scope entered")
    data = load_live_manifest().model_dump(mode="json")
    data["case_policies"][0]["fixture_slots"][0]["fixture_index"] = 7
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(LiveSuiteConfigurationError):
        load_live_manifest(path)


async def test_contextvar_case_scopes_interleave_without_leakage():
    cases = {c.case_id: c for c in load_eval_dataset()}
    a_ready, b_ready = asyncio.Event(), asyncio.Event()
    a_seen, b_seen = asyncio.Event(), asyncio.Event()

    async def a():
        with fixed_case_scope(cases["normal_application"]):
            a_ready.set()
            await b_ready.wait()
            assert (
                resolve_fixed_evidence("search_web", 1, query="a").results
                == cases["normal_application"].searches[0].results
            )
            a_seen.set()
            await b_seen.wait()
            assert (
                resolve_fixed_evidence("search_web", 2, query="a again").results
                == cases["normal_application"].searches[1].results
            )
        with pytest.raises(LiveSuiteConfigurationError):
            resolve_fixed_evidence("search_web", 1, query="reset a")

    async def b():
        await a_ready.wait()
        with fixed_case_scope(cases["insufficient"]):
            b_ready.set()
            await a_seen.wait()
            assert resolve_fixed_evidence("search_web", 1, query="b").results == ()
            b_seen.set()
        with pytest.raises(LiveSuiteConfigurationError):
            resolve_fixed_evidence("search_web", 1, query="reset b")

    await asyncio.wait_for(asyncio.gather(a(), b()), timeout=5)
    with pytest.raises(LiveSuiteConfigurationError):
        resolve_fixed_evidence("search_web", 1, query="outer reset")


@pytest.mark.parametrize("case_id", LIVE_CASE_IDS)
async def test_live_grader_accepts_fixture_evidence_without_claim_id_or_text_matching(case_id):
    case = next(c for c in load_eval_dataset() if c.case_id == case_id)
    executed = await execute_eval_case(case)
    payload = executed.output.model_dump(mode="json")
    for section in ("summary", "findings"):
        for index, claim in enumerate(payload[section]):
            claim["claim_id"] = f"rephrased_{section}_{index}"
            claim["text"] = "Paraphrase: " + claim["text"]
    if payload["application_draft"]:
        for index, claim in enumerate(payload["application_draft"]["paragraphs"]):
            claim["claim_id"] = f"rephrased_draft_{index}"
            claim["text"] = "Paraphrase: " + claim["text"]
    output = ResearchOutputV2.model_validate_json(json.dumps(payload))
    tools, exposed = fixture_exposure(case)
    assert grade_live_output(case, output, tool_observations=tools, exposed_evidence=exposed).passed
    assert not grade_live_output(
        case, None, tool_observations=tools, exposed_evidence=exposed
    ).passed
    if output.evidence_sufficient:
        assert not grade_live_output(case, output, tool_observations=(), exposed_evidence=()).passed
        # Existing fake exact-text grading remains strict; live proxy is separate.
        graded = grade_eval_case(
            case,
            output,
            executed.search_calls,
            model_call_count=executed.model_call_count,
            evidence_alias_by_id=executed.evidence_alias_by_id,
            document_retrieval_calls=executed.document_retrieval_calls,
            relevant_chunk_ids=executed.relevant_chunk_ids,
            input_tokens=executed.input_tokens,
            output_tokens=executed.output_tokens,
            tool_invocation_reservation_count=executed.tool_invocation_reservation_count,
        )
        assert any(g.name == "unsupported_claim" and not g.passed for g in graded.graders)


async def test_citations_to_invented_evidence_and_missing_coverage_fail():
    case = load_eval_dataset()[0]
    executed = await execute_eval_case(case)
    payload = executed.output.model_dump(mode="json")
    old_id = payload["evidence"][0]["evidence_id"]
    payload["evidence"][0]["evidence_id"] = "invented_evidence"
    for claim in [
        *payload["summary"],
        *payload["findings"],
        *payload["application_draft"]["paragraphs"],
    ]:
        for citation in claim["citations"]:
            if citation["evidence_id"] == old_id:
                citation["evidence_id"] = "invented_evidence"
    output = ResearchOutputV2.model_validate_json(json.dumps(payload))
    tools, exposed = fixture_exposure(case)
    result = grade_live_output(case, output, tool_observations=tools, exposed_evidence=exposed)
    assert not result.citation_grounding_proxy and not result.required_evidence_coverage


def test_accepted_revalidates_unchecked_model_copies():
    original = report()
    forged = original.model_copy(update={"observations": original.observations[:-1]})
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5(report=forged)


def test_accepted_rejects_forged_tool_verdict():
    items = list(complete_observations())
    items[0] = observation(
        tool_observations=(
            LiveToolObservationV3(
                tool_name="search_web",
                schema_valid=True,
                proposal_batch_index=1,
                batch_size=1,
                tool_call_count_before=0,
                tool_result_count_before=0,
                disposition="unexpected_unexecuted",
            ),
        )
    )
    with pytest.raises(ValidationError, match="tool execution verdict"):
        AcceptedLiveEvalBaselineV5(report=report(tuple(items)))


def test_v1_v2_v3_v4_history_remains_readable_and_versions_cannot_be_substituted():
    from tests.evals.live_contracts import (
        AcceptedLiveEvalBaselineV1,
        LiveEvalManifestV1,
        LiveEvalReportV1,
    )
    from tests.evals.live_suite import (
        load_live_manifest_v1,
        resolve_fixed_evidence_v1,
    )

    old_v1, old_v2, old_v3, old_v4, current = (
        load_live_manifest_v1(),
        load_live_manifest_v2(),
        load_live_manifest_v3(),
        load_live_manifest_v4(),
        load_live_manifest(),
    )
    assert (
        old_v1.suite_version,
        old_v2.suite_version,
        old_v3.suite_version,
        old_v4.suite_version,
        current.suite_version,
    ) == (
        "live-eval-v1",
        "live-eval-v2",
        "live-eval-v3",
        "live-eval-v4",
        "live-eval-v5",
    )
    assert (
        old_v1.source_artifact_identity
        == old_v2.source_artifact_identity
        == old_v3.source_artifact_identity
        == old_v4.source_artifact_identity
        == (current.source_artifact_identity)
    )
    assert old_v1.version_metadata == old_v2.version_metadata == old_v3.version_metadata
    assert (
        old_v3.version_metadata.plan_prompt_version != current.version_metadata.plan_prompt_version
    )
    assert (
        old_v4.version_metadata.writer_prompt_version
        != current.version_metadata.writer_prompt_version
    )
    assert (
        old_v1.case_policies
        == old_v2.case_policies
        == old_v3.case_policies
        == old_v4.case_policies
        == current.case_policies
    )
    assert old_v3.artifact_identity == old_v4.artifact_identity == current.artifact_identity
    assert (
        len(
            {
                old_v1.artifact_identity.grader_contract_digest,
                old_v2.artifact_identity.grader_contract_digest,
                old_v3.artifact_identity.grader_contract_digest,
                old_v4.artifact_identity.grader_contract_digest,
                current.artifact_identity.grader_contract_digest,
            }
        )
        == 3
    )
    assert (
        len(
            {
                manifest_digest(old_v1),
                manifest_digest(old_v2),
                manifest_digest(old_v3),
                manifest_digest(old_v4),
                manifest_digest(current),
            }
        )
        == 5
    )
    for model, data in (
        (LiveEvalManifestV1, current),
        (LiveEvalManifestV2, current),
        (LiveEvalManifestV5, old_v1),
        (LiveEvalManifestV5, old_v2),
        (LiveEvalManifestV5, old_v3),
        (LiveEvalManifestV5, old_v4),
        (LiveEvalManifestV4, current),
    ):
        with pytest.raises(ValidationError):
            model.model_validate_json(data.model_dump_json())
    with pytest.raises(ValidationError):
        LiveEvalReportV1.model_validate_json(report().model_dump_json())
    # Build historical V1 evidence using its old fields; exact trajectory is still enforced.
    data = report().model_dump(mode="json")
    data.update(
        schema_version=1,
        suite_version="live-eval-v1",
        manifest_digest=manifest_digest(old_v1),
        artifact_identity=old_v1.artifact_identity.model_dump(mode="json"),
        version_metadata=old_v1.version_metadata.model_dump(mode="json"),
    )
    for item in data["observations"]:
        item.pop("graph_failure")
        item.pop("budget_suppressed_proposal_count")
        item.pop("deterministic_empty_execution_count")
        item["grader"]["expected_tool_behavior"] = item["grader"].pop("tool_execution_consistent")
        item["tool_observations"] = [
            {
                "tool_name": t["tool_name"],
                "schema_valid": t["schema_valid"],
                "executed": t["disposition"] == "executed",
            }
            for t in item["tool_observations"]
        ]
    historical = LiveEvalReportV1.model_validate_json(json.dumps(data))
    AcceptedLiveEvalBaselineV1(report=historical)
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV2.model_validate_json(json.dumps({"report": data}))

    historical_v2_data = report().model_dump(mode="json")
    historical_v2_data.update(
        schema_version=2,
        suite_version="live-eval-v2",
        manifest_digest=manifest_digest(old_v2),
        artifact_identity=old_v2.artifact_identity.model_dump(mode="json"),
        version_metadata=old_v2.version_metadata.model_dump(mode="json"),
    )
    for item in historical_v2_data["observations"]:
        item.pop("graph_failure")
        item.pop("budget_suppressed_proposal_count")
        item.pop("deterministic_empty_execution_count")
        for tool in item["tool_observations"]:
            tool["executed"] = tool.pop("disposition") == "executed"
            for field in (
                "proposal_batch_index",
                "batch_size",
                "tool_call_count_before",
                "tool_result_count_before",
                "budget_suppression_reason",
            ):
                tool.pop(field)
    historical_v2 = LiveEvalReportV2.model_validate_json(json.dumps(historical_v2_data))
    AcceptedLiveEvalBaselineV2(report=historical_v2)
    with pytest.raises(ValidationError):
        LiveEvalReportV5.model_validate_json(historical_v2.model_dump_json())
    historical_v3_data = report().model_dump(mode="json")
    historical_v3_data.update(
        schema_version=3,
        suite_version="live-eval-v3",
        manifest_digest=manifest_digest(old_v3),
        artifact_identity=old_v3.artifact_identity.model_dump(mode="json"),
        version_metadata=old_v3.version_metadata.model_dump(mode="json"),
    )
    historical_v3 = LiveEvalReportV3.model_validate_json(json.dumps(historical_v3_data))
    AcceptedLiveEvalBaselineV3(report=historical_v3)
    with fixed_case_scope(load_eval_dataset()[0]):
        with pytest.raises(LiveToolBehaviorError):
            resolve_fixed_evidence_v1("search_web", 3, query="historical deviation")
        assert resolve_fixed_evidence("search_web", 3, query="same deviation").results == ()


@pytest.mark.parametrize(
    "updates",
    [
        {"evidence_resolution": "deterministic_empty", "exposed_result_count": 1},
        {"executed": False},
        {"evidence_resolution": "none"},
        {"fixture_ordinal": None},
        {"schema_valid": False},
        {"tool_name": "private-unknown-name"},
    ],
)
def test_v3_tool_schema_rejects_inconsistent_delivery(updates):
    original = observation().tool_observations[0].model_dump(mode="json")
    with pytest.raises(ValidationError):
        LiveToolObservationV3.model_validate_json(json.dumps(original | updates))


def test_v3_budget_suppression_requires_provable_batch_preflight_metadata():
    original = observation().tool_observations[0].model_dump(mode="json")
    forged = original | {
        "disposition": "budget_suppressed",
        "budget_suppression_reason": "tool_calls_and_results",
        "evidence_resolution": "none",
        "fixture_ordinal": None,
        "exposed_result_count": 0,
    }
    with pytest.raises(ValidationError, match="preflight reason"):
        LiveToolObservationV3.model_validate_json(json.dumps(forged))

    data = report().model_dump(mode="json")
    data["observations"][0]["tool_observations"][0]["batch_size"] = 2
    with pytest.raises(ValidationError, match="batch admission metadata"):
        AcceptedLiveEvalBaselineV5.model_validate_json(json.dumps({"report": data}))


@pytest.mark.parametrize(
    "updates",
    [
        {"fixture_ordinal": 2},
        {"evidence_resolution": "deterministic_empty", "exposed_result_count": 0},
        {"exposed_result_count": 2},
        {"exposed_result_count": 0},
        {"duplicate_call_id": True},
    ],
)
def test_v3_acceptance_rejects_forged_delivery_even_when_quality_claims_pass(updates):
    data = report().model_dump(mode="json")
    data["observations"][0]["tool_observations"][0].update(updates)
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5.model_validate_json(json.dumps({"report": data}))


def test_v3_fallback_cannot_be_relabelled_as_declared_fixture():
    items = list(complete_observations())
    extra = LiveToolObservationV3(
        tool_name="search_web",
        schema_valid=True,
        proposal_batch_index=3,
        batch_size=1,
        tool_call_count_before=2,
        tool_result_count_before=2,
        disposition="executed",
        fixture_ordinal=3,
        evidence_resolution="deterministic_empty",
        exposed_result_count=0,
    )
    items[0] = observation(tool_observations=(*items[0].tool_observations, extra))
    accepted = AcceptedLiveEvalBaselineV5(report=report(tuple(items)))
    data = accepted.model_dump(mode="json")
    data["report"]["observations"][0]["tool_observations"][-1]["evidence_resolution"] = (
        "declared_fixture"
    )
    with pytest.raises(ValidationError):
        AcceptedLiveEvalBaselineV5.model_validate_json(json.dumps(data))
