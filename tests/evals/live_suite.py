"""Versioned Gate 11 measurement helpers. No provider clients or live execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from app.agents.research_nodes import web_evidence_id
from app.domain.research import ResearchOutputV2
from app.tools.search import normalize_search_result
from tests.evals.contracts import (
    EvalArtifactIdentityV1,
    EvalDocumentResultFixtureV3,
    EvalSearchResultFixtureV1,
    ResearchEvalCaseV2,
)
from tests.evals.harness import (
    PROJECT_ROOT,
    _canonical_digest,
    eval_artifact_identity,
    eval_dataset_digests,
    load_eval_dataset,
    load_eval_manifest,
)
from tests.evals.live_contracts import (
    LIVE_CASE_IDS,
    LiveCasePolicyV1,
    LiveEvalGraderResultV1,
    LiveEvalGraderResultV2,
    LiveEvalGraderResultV3,
    LiveEvalManifestV1,
    LiveEvalManifestV2,
    LiveEvalManifestV3,
    LiveEvalManifestV4,
    LiveEvalManifestV5,
    LiveEvalObservationV1,
    LiveEvalObservationV2,
    LiveEvalObservationV3,
    LiveFixtureSlotV1,
    LiveToolObservationV1,
    LiveToolObservationV2,
    LiveToolObservationV3,
)
from tests.evals.regression import load_accepted_baseline

V1_LIVE_MANIFEST = PROJECT_ROOT / "evals/datasets/live_eval_v1_manifest.json"
V2_LIVE_MANIFEST = PROJECT_ROOT / "evals/datasets/live_eval_v2_manifest.json"
V3_LIVE_MANIFEST = PROJECT_ROOT / "evals/datasets/live_eval_v3_manifest.json"
V4_LIVE_MANIFEST = PROJECT_ROOT / "evals/datasets/live_eval_v4_manifest.json"

DEFAULT_LIVE_MANIFEST = PROJECT_ROOT / "evals/datasets/live_eval_v5_manifest.json"

# This payload versions the rubric, not exact claim IDs/text or semantic entailment.
LIVE_GRADER_SEMANTICS_V1 = {
    "version": "live-deterministic-graders-v1",
    "checks": tuple(LiveEvalGraderResultV1.model_fields),
    "citations": "source_type/source_id/evidence_id must resolve to actually exposed fixture",
    "coverage": "union of support-rule evidence aliases; no claim ID/text match",
    "tools": "exact declared tool sequence; every proposal valid and executed",
    "limitations": "exact expected code set",
    "semantic_entailment": False,
}


class LiveSuiteConfigurationError(ValueError):
    """Invalid declared evidence or identity; never a quality result or fallback."""


class LiveToolBehaviorError(ValueError):
    """Unsupported tool, or a V1-only undeclared ordinal; never empty fallback."""


def canonical_live_digest(domain: str, value: object) -> str:
    return _canonical_digest(domain.encode("ascii"), value)


def manifest_digest(
    manifest: (
        LiveEvalManifestV1
        | LiveEvalManifestV2
        | LiveEvalManifestV3
        | LiveEvalManifestV4
        | LiveEvalManifestV5
    ),
) -> str:
    return canonical_live_digest(
        f"pathfinder-live-manifest-v{manifest.schema_version}", manifest.model_dump(mode="json")
    )


def derive_case_policy(case: ResearchEvalCaseV2) -> LiveCasePolicyV1:
    slots = []
    ordinals: dict[str, int] = {}
    for research_pass in case.research_passes:
        calls = [("search_web", c.query) for c in research_pass.calls] + [
            (c.tool_name, c.query) for c in research_pass.ordered_tool_calls
        ]
        for tool, fixture_query in calls:
            if tool not in ("search_web", "retrieve_documents"):
                raise LiveSuiteConfigurationError("unsupported declared fixture tool")
            fixtures = case.searches if tool == "search_web" else case.document_retrievals
            matches = [i for i, fixture in enumerate(fixtures) if fixture.query == fixture_query]
            if len(matches) != 1:
                raise LiveSuiteConfigurationError("declared fixture cannot resolve uniquely")
            ordinals[tool] = ordinals.get(tool, 0) + 1
            slots.append(
                LiveFixtureSlotV1(tool_name=tool, ordinal=ordinals[tool], fixture_index=matches[0])
            )
    return LiveCasePolicyV1(
        case_id=case.case_id,
        required_evidence_aliases=tuple(
            sorted({a for rule in case.support_rules for a in rule.allowed_evidence_aliases})
        ),
        fixture_slots=tuple(slots),
    )


def live_artifact_identity(
    cases: tuple[ResearchEvalCaseV2, ...], *, grader_version: int = 3
) -> EvalArtifactIdentityV1:
    source = eval_artifact_identity(cases)
    return EvalArtifactIdentityV1(
        dataset_digest=source.dataset_digest,
        case_set_digest=source.case_set_digest,
        graph_version=source.graph_version,
        embedding_profile=source.embedding_profile,
        grader_contract_version=f"live-deterministic-graders-v{grader_version}",
        grader_contract_digest=canonical_live_digest(
            f"pathfinder-live-grader-v{grader_version}",
            {
                1: LIVE_GRADER_SEMANTICS_V1,
                2: LIVE_GRADER_SEMANTICS_V2,
                3: LIVE_GRADER_SEMANTICS_V3,
            }[grader_version],
        ),
    )


def validate_live_manifest(
    manifest: (
        LiveEvalManifestV1
        | LiveEvalManifestV2
        | LiveEvalManifestV3
        | LiveEvalManifestV4
        | LiveEvalManifestV5
    ),
    *,
    require_current_runtime: bool = True,
) -> None:
    cases = load_eval_dataset()
    source_identity = eval_artifact_identity(cases)
    baseline = load_accepted_baseline().report
    if (
        manifest.source_artifact_identity != source_identity
        or source_identity != baseline.artifact_identity
    ):
        raise LiveSuiteConfigurationError("source dataset/Gate 9 accepted identity mismatch")
    if require_current_runtime and (
        manifest.version_metadata != load_eval_manifest()
        or manifest.version_metadata != baseline.version_metadata
    ):
        raise LiveSuiteConfigurationError("production/Gate 9 version identity mismatch")
    selected = {c.case_id: c for c in cases}
    if not set(LIVE_CASE_IDS) <= selected.keys():
        raise LiveSuiteConfigurationError("selected case missing")
    grader_version = (
        3
        if isinstance(manifest, LiveEvalManifestV4 | LiveEvalManifestV5)
        else manifest.schema_version
    )
    if manifest.artifact_identity != live_artifact_identity(cases, grader_version=grader_version):
        raise LiveSuiteConfigurationError("live artifact identity mismatch")
    if (
        manifest.selected_case_set_digest
        != eval_dataset_digests(tuple(selected[c] for c in LIVE_CASE_IDS))[1]
    ):
        raise LiveSuiteConfigurationError("selected case-set digest mismatch")
    if manifest.case_policies != tuple(derive_case_policy(selected[c]) for c in LIVE_CASE_IDS):
        raise LiveSuiteConfigurationError("fixture/coverage policy mismatch")


def load_live_manifest_v1(path: Path = V1_LIVE_MANIFEST) -> LiveEvalManifestV1:
    try:
        manifest = LiveEvalManifestV1.model_validate_json(path.read_bytes(), strict=True)
        validate_live_manifest(manifest, require_current_runtime=False)
        return manifest
    except (OSError, ValueError) as error:
        raise LiveSuiteConfigurationError("invalid live suite configuration") from error


# Immutable bindings avoid shared mutable ordinal counters between inherited coroutine contexts.
_ACTIVE_CASE: ContextVar[tuple[ResearchEvalCaseV2, LiveCasePolicyV1] | None] = ContextVar(
    "pathfinder_live_case", default=None
)


@contextmanager
def fixed_case_scope(case: ResearchEvalCaseV2) -> Iterator[None]:
    policy = derive_case_policy(case)
    token = _ACTIVE_CASE.set((case, policy))
    try:
        yield
    finally:
        _ACTIVE_CASE.reset(token)


def resolve_fixed_evidence_v1(
    tool_name: str, ordinal: int, *, query: str
) -> tuple[EvalSearchResultFixtureV1, ...] | tuple[EvalDocumentResultFixtureV3, ...]:
    """Query wording is intentionally irrelevant; callers retain it only in controlled evidence."""
    binding = _ACTIVE_CASE.get()
    if binding is None:
        raise LiveSuiteConfigurationError("no active live case")
    case, policy = binding
    slot = next(
        (s for s in policy.fixture_slots if (s.tool_name, s.ordinal) == (tool_name, ordinal)), None
    )
    if slot is None:
        raise LiveToolBehaviorError("tool/ordinal is outside declared case behavior")
    fixtures = case.searches if tool_name == "search_web" else case.document_retrievals
    if slot.fixture_index >= len(fixtures):
        raise LiveSuiteConfigurationError("declared fixture is missing")
    return fixtures[slot.fixture_index].results


def grade_live_output_v1(
    case: ResearchEvalCaseV2,
    output: ResearchOutputV2 | None,
    *,
    tool_observations: tuple[LiveToolObservationV1, ...],
) -> LiveEvalGraderResultV1:
    """Citation/coverage proxy only. Does not prove natural-language entailment.

    Each executed tool exposes the complete declared slot. Future adapters must honor
    that contract (including fixed result count), not silently truncate by model query.
    """
    policy = derive_case_policy(case)
    references: dict[tuple[str, str, str], str] = {}
    expected_tools = tuple(slot.tool_name for slot in policy.fixture_slots)
    executed_tools = tuple(t.tool_name for t in tool_observations if t.executed)
    ordinals: dict[str, int] = {}
    with fixed_case_scope(case):
        for tool in tool_observations:
            if not tool.executed:
                continue
            ordinals[tool.tool_name] = ordinals.get(tool.tool_name, 0) + 1
            try:
                results = resolve_fixed_evidence_v1(
                    tool.tool_name, ordinals[tool.tool_name], query=""
                )
            except LiveToolBehaviorError:
                continue
            for result in results:
                if isinstance(result, EvalSearchResultFixtureV1):
                    normalized = normalize_search_result(
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        published_at=result.published_at,
                    )
                    if result.evidence_alias is not None:
                        references[
                            (
                                "web",
                                normalized.source_id,
                                web_evidence_id(
                                    source_id=normalized.source_id, snippet=normalized.snippet
                                ),
                            )
                        ] = result.evidence_alias
                else:
                    references[
                        (
                            "workspace_document",
                            f"workspace-document-v1:{result.document_id}",
                            f"workspace-chunk-v1:{result.chunk_id}",
                        )
                    ] = result.evidence_alias
    if output is not None:
        # Revalidate even a caller-provided model_construct/model_copy object.
        try:
            output = ResearchOutputV2.model_validate_json(output.model_dump_json(), strict=True)
        except ValidationError:
            output = None
    claims = (
        ()
        if output is None
        else (
            *output.summary,
            *output.findings,
            *(output.application_draft.paragraphs if output.application_draft else ()),
        )
    )
    cited = {
        (c.source_type, c.source_id, c.evidence_id) for claim in claims for c in claim.citations
    }
    aliases = {references[c] for c in cited if c in references}
    expected = case.expectations
    return LiveEvalGraderResultV1(
        structured_output_valid=output is not None,
        evidence_sufficiency_matches=output is not None
        and output.evidence_sufficient == expected.evidence_sufficient,
        application_draft_presence_matches=output is not None
        and (output.application_draft is not None) == expected.application_draft_present,
        limitation_codes_match=output is not None
        and {limitation.code for limitation in output.limitations}
        == set(expected.limitation_codes),
        citation_grounding_proxy=output is not None and cited <= references.keys(),
        required_evidence_coverage=output is not None
        and set(policy.required_evidence_aliases) <= aliases,
        minimum_cited_sources_met=output is not None
        and len({c[1] for c in cited if c in references}) >= expected.minimum_cited_sources,
        expected_tool_behavior=executed_tools == expected_tools
        and tuple(t.tool_name for t in tool_observations) == expected_tools,
        tool_arguments_schema_valid=all(t.schema_valid for t in tool_observations),
    )


def can_admit_attempt(
    manifest: (
        LiveEvalManifestV1
        | LiveEvalManifestV2
        | LiveEvalManifestV3
        | LiveEvalManifestV4
        | LiveEvalManifestV5
    ),
    *,
    known_cost_cny: Decimal,
    unknown_cost_attempt_count: int,
    provider_attempts: int,
    input_tokens: int,
    output_tokens: int,
) -> bool:
    """Conservative admission, not a forecast or exact cap on provider billing."""
    if (
        not known_cost_cny.is_finite()
        or min(
            known_cost_cny,
            unknown_cost_attempt_count,
            provider_attempts,
            input_tokens,
            output_tokens,
        )
        < 0
    ):
        raise LiveSuiteConfigurationError("invalid admission accounting")
    return (
        known_cost_cny + (unknown_cost_attempt_count + 1) * manifest.unknown_attempt_reserve_cny
        <= manifest.cost_admission_budget_cny
        and provider_attempts < manifest.provider_attempt_cap
        and input_tokens < manifest.input_token_cap
        and output_tokens < manifest.output_token_cap
    )


def validate_attempt_caps(
    manifest: (
        LiveEvalManifestV1
        | LiveEvalManifestV2
        | LiveEvalManifestV3
        | LiveEvalManifestV4
        | LiveEvalManifestV5
    ),
    observations: (
        tuple[LiveEvalObservationV1, ...]
        | tuple[LiveEvalObservationV2, ...]
        | tuple[LiveEvalObservationV3, ...]
    ),
) -> None:
    # Sequential suite evidence is stored in execution order, attempts included.
    known, unknown, count, input_tokens, output_tokens = Decimal(0), 0, 0, 0, 0
    for observation in observations:
        if observation.execution_error in {
            "budget_exhausted",
            "token_cap_exceeded",
            "provider_cap_exceeded",
        }:
            raise ValueError("stopped/capped suite cannot be accepted")
        for attempt in observation.provider_attempts:
            if not can_admit_attempt(
                manifest,
                known_cost_cny=known,
                unknown_cost_attempt_count=unknown,
                provider_attempts=count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ):
                raise ValueError("provider attempt violated pre-attempt admission")
            count += 1
            input_tokens += attempt.input_tokens
            output_tokens += attempt.output_tokens
            if attempt.known_cost_cny is None:
                unknown += 1
            else:
                known += attempt.known_cost_cny
            if input_tokens > manifest.input_token_cap or output_tokens > manifest.output_token_cap:
                raise ValueError("reported token cap exceeded; suite must stop")
    if known + unknown * manifest.unknown_attempt_reserve_cny > manifest.cost_admission_budget_cny:
        raise ValueError("admission budget exceeded")


LIVE_GRADER_SEMANTICS_V2 = {
    "version": "live-deterministic-graders-v2",
    "metrics": "live-metrics-v2",
    "checks": tuple(LiveEvalGraderResultV2.model_fields),
    "fixture_lookup": "case_tool_ordinal_declared_then_empty",
    "slots": "per-tool ordinal evidence opportunities, not required trajectory",
    "search": "declared results sliced to schema-valid max_results before exposure",
    "citations": "source_type/source_id/evidence_id must resolve to actually delivered evidence",
    "coverage": "union of support-rule evidence aliases cited from actual exposed evidence",
    "tools": "valid allowed proposals execute; consistent resolution/count/ordinal; no duplicates",
    "limitations": "exact expected code set",
    "semantic_entailment": False,
}


def load_live_manifest_v2(path: Path = V2_LIVE_MANIFEST) -> LiveEvalManifestV2:
    try:
        manifest = LiveEvalManifestV2.model_validate_json(path.read_bytes(), strict=True)
        validate_live_manifest(manifest, require_current_runtime=False)
        return manifest
    except (OSError, ValueError) as error:
        raise LiveSuiteConfigurationError("invalid live suite configuration") from error


@dataclass(frozen=True, repr=False)
class ResolvedFixedEvidenceV2:
    results: tuple[EvalSearchResultFixtureV1, ...] | tuple[EvalDocumentResultFixtureV3, ...]
    resolution_kind: Literal["declared_fixture", "deterministic_empty"]
    fixture_ordinal: int


def resolve_fixed_evidence(tool_name: str, ordinal: int, *, query: str) -> ResolvedFixedEvidenceV2:
    """Only allowed/schema-valid calls reach this adapter; query never selects evidence."""
    binding = _ACTIVE_CASE.get()
    if binding is None:
        raise LiveSuiteConfigurationError("no active live case")
    if tool_name not in {"search_web", "retrieve_documents"}:
        raise LiveToolBehaviorError("unsupported tool")
    if type(ordinal) is not int or not 1 <= ordinal <= 8:
        raise LiveSuiteConfigurationError("invalid bounded tool ordinal")
    case, policy = binding
    slot = next(
        (s for s in policy.fixture_slots if (s.tool_name, s.ordinal) == (tool_name, ordinal)), None
    )
    if slot is None:
        return ResolvedFixedEvidenceV2((), "deterministic_empty", ordinal)
    fixtures = case.searches if tool_name == "search_web" else case.document_retrievals
    if slot.fixture_index >= len(fixtures):
        raise LiveSuiteConfigurationError("declared fixture is missing")
    return ResolvedFixedEvidenceV2(
        fixtures[slot.fixture_index].results, "declared_fixture", ordinal
    )


@dataclass(frozen=True)
class ExposedEvidenceReference:
    """Invocation-local reference only: no query, fixture text, or trusted context."""

    source_type: Literal["web", "workspace_document"]
    source_id: str
    evidence_id: str
    evidence_alias: str | None


def exposed_reference(result: EvalSearchResultFixtureV1 | EvalDocumentResultFixtureV3):
    if isinstance(result, EvalSearchResultFixtureV1):
        normalized = normalize_search_result(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
            published_at=result.published_at,
        )
        return ExposedEvidenceReference(
            "web",
            normalized.source_id,
            web_evidence_id(source_id=normalized.source_id, snippet=normalized.snippet),
            result.evidence_alias,
        )
    return ExposedEvidenceReference(
        "workspace_document",
        f"workspace-document-v1:{result.document_id}",
        f"workspace-chunk-v1:{result.chunk_id}",
        result.evidence_alias,
    )


def validate_tool_observations_v2(
    case: ResearchEvalCaseV2, tools: tuple[LiveToolObservationV2, ...]
) -> bool:
    """Reject corrupt delivery metadata; quality does not require consuming every slot.

    Acceptance can check metadata against fixture bounds, not re-run output grading.
    """
    ordinals: dict[str, int] = {}
    with fixed_case_scope(case):
        for tool in tools:
            LiveToolObservationV2.model_validate_json(tool.model_dump_json(), strict=True)
            if not tool.executed:
                continue
            ordinals[tool.tool_name] = ordinals.get(tool.tool_name, 0) + 1
            if tool.fixture_ordinal != ordinals[tool.tool_name]:
                raise ValueError("tool delivery ordinal must match execution order")
            resolution = resolve_fixed_evidence(tool.tool_name, tool.fixture_ordinal, query="")
            if tool.evidence_resolution != resolution.resolution_kind:
                raise ValueError("tool delivery resolution must match declared opportunity")
            if tool.exposed_result_count > len(resolution.results):
                raise ValueError("tool delivery count exceeds declared fixture")
            if tool.tool_name == "retrieve_documents":
                if tool.exposed_result_count != len(resolution.results):
                    raise ValueError("retrieval delivery count must match fixed chunks")
            elif resolution.results and tool.exposed_result_count == 0:
                raise ValueError("nonempty search requires a positive result bound")
    return all(t.schema_valid and t.executed and not t.duplicate_call_id for t in tools)


def grade_live_output_v2(
    case: ResearchEvalCaseV2,
    output: ResearchOutputV2 | None,
    *,
    tool_observations: tuple[LiveToolObservationV2, ...],
    exposed_evidence: tuple[ExposedEvidenceReference, ...],
) -> LiveEvalGraderResultV2:
    """Grade actual delivered references; never reconstruct exposure from fixture slots."""
    policy = derive_case_policy(case)
    references = {
        (e.source_type, e.source_id, e.evidence_id): e.evidence_alias for e in exposed_evidence
    }
    consistent = validate_tool_observations_v2(case, tool_observations)
    if output is not None:
        # Revalidate even a caller-provided model_construct/model_copy object.
        try:
            output = ResearchOutputV2.model_validate_json(output.model_dump_json(), strict=True)
        except ValidationError:
            output = None
    claims = (
        ()
        if output is None
        else (
            *output.summary,
            *output.findings,
            *(output.application_draft.paragraphs if output.application_draft else ()),
        )
    )
    cited = {
        (c.source_type, c.source_id, c.evidence_id) for claim in claims for c in claim.citations
    }
    aliases = {references[c] for c in cited if c in references}
    expected = case.expectations
    return LiveEvalGraderResultV2(
        structured_output_valid=output is not None,
        evidence_sufficiency_matches=output is not None
        and output.evidence_sufficient == expected.evidence_sufficient,
        application_draft_presence_matches=output is not None
        and (output.application_draft is not None) == expected.application_draft_present,
        limitation_codes_match=output is not None
        and {limitation.code for limitation in output.limitations}
        == set(expected.limitation_codes),
        citation_grounding_proxy=output is not None and cited <= references.keys(),
        required_evidence_coverage=output is not None
        and set(policy.required_evidence_aliases) <= aliases,
        minimum_cited_sources_met=output is not None
        and len({c[1] for c in cited if c in references}) >= expected.minimum_cited_sources,
        tool_execution_consistent=consistent,
        tool_arguments_schema_valid=all(t.schema_valid for t in tool_observations),
    )


LIVE_GRADER_SEMANTICS_V3 = {
    "version": "live-deterministic-graders-v3",
    "metrics": "live-metrics-v3",
    "checks": tuple(LiveEvalGraderResultV3.model_fields),
    "fixture_lookup": "case_tool_ordinal_declared_then_empty",
    "slots": "per-tool ordinal evidence opportunities, not required trajectory",
    "search": "declared results sliced to schema-valid max_results before exposure",
    "citations": "source_type/source_id/evidence_id must resolve to actually delivered evidence",
    "coverage": "union of support-rule evidence aliases cited from actual exposed evidence",
    "tools": (
        "proposal lifecycle must be executed successfully, or be a valid unique allowed "
        "research batch suppressed by production tool-call/tool-result admission; batch index, "
        "batch size, and pre-batch call/result counts prove all-or-nothing suppression"
    ),
    "error_precedence": "primary graph outcome is preserved independently of tool diagnostics",
    "graph_diagnostics": "bounded content-free ResearchGraphProtocolError fields only",
    "limitations": "exact expected code set",
    "semantic_entailment": False,
}


def load_live_manifest_v3(path: Path = V3_LIVE_MANIFEST) -> LiveEvalManifestV3:
    try:
        manifest = LiveEvalManifestV3.model_validate_json(path.read_bytes(), strict=True)
        validate_live_manifest(manifest, require_current_runtime=False)
        return manifest
    except (OSError, ValueError) as error:
        raise LiveSuiteConfigurationError("invalid live suite configuration") from error


def load_live_manifest_v4(path: Path = V4_LIVE_MANIFEST) -> LiveEvalManifestV4:
    try:
        manifest = LiveEvalManifestV4.model_validate_json(path.read_bytes(), strict=True)
        validate_live_manifest(manifest, require_current_runtime=False)
        return manifest
    except (OSError, ValueError) as error:
        raise LiveSuiteConfigurationError("invalid live suite configuration") from error


def load_live_manifest(path: Path = DEFAULT_LIVE_MANIFEST) -> LiveEvalManifestV5:
    try:
        manifest = LiveEvalManifestV5.model_validate_json(path.read_bytes(), strict=True)
        validate_live_manifest(manifest, require_current_runtime=True)
        return manifest
    except (OSError, ValueError) as error:
        raise LiveSuiteConfigurationError("invalid live suite configuration") from error


def validate_tool_observations(
    case: ResearchEvalCaseV2, tools: tuple[LiveToolObservationV3, ...]
) -> bool:
    """Validate V3 execution delivery and accept only explained lifecycle dispositions."""
    ordinals: dict[str, int] = {}
    prior_calls = 0
    prior_results = 0
    batches: dict[int, list[LiveToolObservationV3]] = {}
    for tool in tools:
        batches.setdefault(tool.proposal_batch_index, []).append(tool)
    if tuple(batches) != tuple(range(1, len(batches) + 1)):
        raise ValueError("proposal batch indexes must be contiguous")
    with fixed_case_scope(case):
        for _batch_index, batch in batches.items():
            if any(
                tool.batch_size != len(batch)
                or tool.tool_call_count_before != prior_calls
                or tool.tool_result_count_before != prior_results
                for tool in batch
            ):
                raise ValueError("proposal batch admission metadata is inconsistent")
            suppressed = [tool.disposition == "budget_suppressed" for tool in batch]
            if any(suppressed) and not all(suppressed):
                raise ValueError("production budget suppression must reject the whole batch")
            for tool in batch:
                LiveToolObservationV3.model_validate_json(tool.model_dump_json(), strict=True)
                if tool.disposition != "executed":
                    prior_calls += int(tool.disposition == "execution_failed")
                    continue
                ordinals[tool.tool_name] = ordinals.get(tool.tool_name, 0) + 1
                if tool.fixture_ordinal != ordinals[tool.tool_name]:
                    raise ValueError("tool delivery ordinal must match execution order")
                resolution = resolve_fixed_evidence(tool.tool_name, tool.fixture_ordinal, query="")
                if tool.evidence_resolution != resolution.resolution_kind:
                    raise ValueError("tool delivery resolution must match declared opportunity")
                if tool.exposed_result_count > len(resolution.results):
                    raise ValueError("tool delivery count exceeds declared fixture")
                if tool.tool_name == "retrieve_documents":
                    if tool.exposed_result_count != len(resolution.results):
                        raise ValueError("retrieval delivery count must match fixed chunks")
                elif resolution.results and tool.exposed_result_count == 0:
                    raise ValueError("nonempty search requires a positive result bound")
                prior_calls += 1
                prior_results += 1
    return all(
        tool.disposition in {"executed", "budget_suppressed"}
        and tool.schema_valid
        and not tool.duplicate_call_id
        for tool in tools
    )


def grade_live_output(
    case: ResearchEvalCaseV2,
    output: ResearchOutputV2 | None,
    *,
    tool_observations: tuple[LiveToolObservationV3, ...],
    exposed_evidence: tuple[ExposedEvidenceReference, ...],
) -> LiveEvalGraderResultV3:
    """Grade delivered evidence and V3 proposal lifecycles without scripted trajectories."""
    policy = derive_case_policy(case)
    references = {
        (e.source_type, e.source_id, e.evidence_id): e.evidence_alias for e in exposed_evidence
    }
    consistent = validate_tool_observations(case, tool_observations)
    if output is not None:
        try:
            output = ResearchOutputV2.model_validate_json(output.model_dump_json(), strict=True)
        except ValidationError:
            output = None
    claims = (
        ()
        if output is None
        else (
            *output.summary,
            *output.findings,
            *(output.application_draft.paragraphs if output.application_draft else ()),
        )
    )
    cited = {
        (c.source_type, c.source_id, c.evidence_id) for claim in claims for c in claim.citations
    }
    aliases = {references[c] for c in cited if c in references}
    expected = case.expectations
    return LiveEvalGraderResultV3(
        structured_output_valid=output is not None,
        evidence_sufficiency_matches=output is not None
        and output.evidence_sufficient == expected.evidence_sufficient,
        application_draft_presence_matches=output is not None
        and (output.application_draft is not None) == expected.application_draft_present,
        limitation_codes_match=output is not None
        and {limitation.code for limitation in output.limitations}
        == set(expected.limitation_codes),
        citation_grounding_proxy=output is not None and cited <= references.keys(),
        required_evidence_coverage=output is not None
        and set(policy.required_evidence_aliases) <= aliases,
        minimum_cited_sources_met=output is not None
        and len({c[1] for c in cited if c in references}) >= expected.minimum_cited_sources,
        tool_execution_consistent=consistent,
        tool_arguments_schema_valid=all(t.schema_valid for t in tool_observations),
    )
