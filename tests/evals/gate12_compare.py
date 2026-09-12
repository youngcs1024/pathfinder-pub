"""Opt-in, offline Native/MCP comparison; sanitized stdout, no persisted baseline."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from time import perf_counter
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.mcp_experiment.client import MCP_EXPERIMENT_POLICY, open_mcp_experiment
from app.tools.mcp_experiment.server import PROTOCOL_TARGET, SERVER_NAME, SERVER_VERSION
from app.tools.registry import ToolRegistry
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    MANUAL_LEARNING_POLICY,
    LookupSyntheticRecordOutput,
)
from tests.evals import gate12_native as native
from tests.tracing import CollectingTraceSink

NonnegativeInt = Annotated[int, Field(ge=0)]
Milliseconds = Annotated[float, Field(ge=0)]


class ComparisonError(ValueError):
    """The observations cannot support a comparison."""


class _StrictReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class SpanSamples(_StrictReport):
    sample_count: Literal[9] = 9
    samples_ms: tuple[NonnegativeInt, ...]
    p50_ms: NonnegativeInt
    p95_ms: NonnegativeInt

    @model_validator(mode="after")
    def check_samples(self) -> SpanSamples:
        if len(self.samples_ms) != self.sample_count or (
            self.p50_ms,
            self.p95_ms,
        ) != native.latency_percentiles(self.samples_ms):
            raise ValueError("comparison sample aggregate mismatch")
        return self


class MCPSamples(_StrictReport):
    functional_cases: tuple[native.NativeCase, ...]
    warmup_call_count: Literal[3] = 3
    repeats: Literal[3] = 3
    steady_state_session_count: Literal[1] = 1
    registry_tool: SpanSamples
    mcp_transport: SpanSamples
    lifecycle_sample_count: Literal[3] = 3
    process_startup_plus_protocol_discovery_ms: tuple[Milliseconds, ...]
    cleanup_ms: tuple[Milliseconds, ...]
    startup_boundary: Literal["before_context_enter_to_registry_yield"] = (
        "before_context_enter_to_registry_yield"
    )
    cleanup_boundary: Literal["before_context_exit_to_cleanup_complete"] = (
        "before_context_exit_to_cleanup_complete"
    )
    lifecycle_clock: Literal["perf_counter"] = "perf_counter"

    @model_validator(mode="after")
    def check_lifecycle(self) -> MCPSamples:
        if (
            len(self.process_startup_plus_protocol_discovery_ms) != self.lifecycle_sample_count
            or len(self.cleanup_ms) != self.lifecycle_sample_count
        ):
            raise ValueError("comparison lifecycle sample count mismatch")
        return self


class Gate12ComparisonReportV1(_StrictReport):
    schema_version: Literal[1] = 1
    experiment_id: Literal["gate12-native-mcp-comparison"] = "gate12-native-mcp-comparison"
    tool_name: Literal["lookup_synthetic_record"] = "lookup_synthetic_record"
    tool_contract: native.NativeContract
    input_model: Literal["LookupSyntheticRecordInput"] = "LookupSyntheticRecordInput"
    output_model: Literal["LookupSyntheticRecordOutput"] = "LookupSyntheticRecordOutput"
    sdk: Literal["mcp==2.1.1"] = "mcp==2.1.1"
    protocol: Literal["2026-07-28"] = "2026-07-28"
    server_name: Literal["pathfinder-gate12-teaching"] = "pathfinder-gate12-teaching"
    server_version: Literal["0.1.0"] = "0.1.0"
    native: native.NativeReport
    mcp: MCPSamples
    contract_equivalent: Literal[True] = True
    results_equivalent: Literal[True] = True
    measurement_boundary: Literal["gate10_registry_tool_span"] = "gate10_registry_tool_span"
    transport_boundary: Literal["mcp_call_tool_inside_registry_handler"] = (
        "mcp_call_tool_inside_registry_handler"
    )
    latency_source: Literal["TraceSpanFinish.latency_ms"] = "TraceSpanFinish.latency_ms"
    latency_resolution_ms: Literal[1] = 1
    percentile_method: Literal["nearest_rank"] = "nearest_rank"
    native_setup_sample_count: Literal[1] = 1
    native_setup_clock: Literal["perf_counter"] = "perf_counter"
    external_network_used: Literal[False] = False
    external_credentials_used: Literal[False] = False
    production_dependency: Literal[False] = False
    raw_pre_parse_cap_satisfied: Literal[False] = False
    os_sandbox: Literal[False] = False

    @model_validator(mode="after")
    def check_equivalence(self) -> Gate12ComparisonReportV1:
        if self.native.tool_contract != self.tool_contract:
            raise ValueError("comparison contract mismatch")
        if self.native.functional_cases != self.mcp.functional_cases:
            raise ValueError("comparison functional mismatch")
        return self


def validate_local_contract(registry: ToolRegistry) -> None:
    # Test harness inspection, not a new runtime interface. Every field except transport
    # handler must be the original ToolSpec; discovery never supplies local authority.
    spec = LOOKUP_SYNTHETIC_RECORD_SPEC
    if set(registry._specs) != {spec.name}:
        raise ComparisonError("comparison tool surface mismatch")
    candidate = registry._specs[spec.name]
    if replace(candidate, handler=spec.handler) != spec:
        raise ComparisonError("comparison tool contract mismatch")
    policy = registry._policies.get(MCP_EXPERIMENT_POLICY.name)
    if (
        policy is None
        or policy.allowed_tool_names != MANUAL_LEARNING_POLICY.allowed_tool_names
        or policy.allowed_effects != MANUAL_LEARNING_POLICY.allowed_effects
        or set(registry._policies) != {MCP_EXPERIMENT_POLICY.name}
    ):
        raise ComparisonError("comparison local policy mismatch")


async def measure_mcp_case(registry: ToolRegistry, record_id: str, sequence: int):
    runtime = registry.bind(
        policy_name=MCP_EXPERIMENT_POLICY.name,
        context=native.native_context(sequence),
    )
    sink = CollectingTraceSink()
    with native.native_trace(sink, sequence):
        serialized = await runtime.execute(native.native_call(record_id))
    # Validate exact business results in memory; report only fixed ID and found outcome.
    result = LookupSyntheticRecordOutput.model_validate_json(serialized, strict=True)
    expected = native.EXPECTED_SUMMARIES[native.CASES.index(record_id)]
    if (
        result.record_id != record_id
        or result.summary != expected
        or result.found != (expected is not None)
    ):
        raise ComparisonError("comparison result mismatch")
    tool_finish = native.tool_observation(sink)
    if tool_finish.metadata.get("output_bytes") != len(serialized.encode("utf-8")):
        raise ComparisonError("comparison output byte mismatch")
    # The reused Native harness has a synthetic node without runtime occurrence ordinals.
    # Check this exact four-span tree instead of claiming a production graph executed.
    parents = {"graph_node": "execution_segment", "tool": "graph_node", "mcp_transport": "tool"}
    if (
        sink.starts.keys() != sink.finishes.keys()
        or len(sink.starts) != 4
        or {span.span_kind for _, span in sink.starts.values()} != {"execution_segment", *parents}
    ):
        raise ComparisonError("comparison trace tree incomplete")
    for context, span in sink.starts.values():
        if span.span_kind == "execution_segment":
            if span.parent is not None:
                raise ComparisonError("comparison trace root invalid")
        elif (
            span.parent is None
            or span.parent.span_kind != parents[span.span_kind]
            or span.parent.context_id not in sink.starts
            or sink.starts[span.parent.context_id][0] != span.parent
            or span.parent.trace_identity != context.trace_identity
            or span.parent.segment_identity != context.segment_identity
        ):
            raise ComparisonError("comparison trace parent invalid")
    transports = [
        (context, span)
        for context, span in sink.starts.values()
        if span.span_kind == "mcp_transport"
    ]
    if len(transports) != 1:
        raise ComparisonError("comparison transport span missing or duplicated")
    context, span = transports[0]
    finish = sink.finishes[context.context_id]
    if (
        dict(span.metadata) != {"transport": "stdio"}
        or finish.status != "succeeded"
        or finish.error_category is not None
        or finish.metadata
    ):
        raise ComparisonError("comparison transport observation invalid")
    return native.NativeCase(record_id=record_id, found=result.found), (
        tool_finish.latency_ms,
        finish.latency_ms,
    )


def span_samples(values: list[int]) -> SpanSamples:
    p50, p95 = native.latency_percentiles(values)
    return SpanSamples(samples_ms=tuple(values), p50_ms=p50, p95_ms=p95)


async def run_comparison() -> Gate12ComparisonReportV1:
    native_report = await native.run_native_baseline()
    startups, cleanups, functional, tools, transports = [], [], [], [], []
    for session in range(3):
        started_at = perf_counter()
        async with open_mcp_experiment() as registry:
            startups.append((perf_counter() - started_at) * 1000)
            validate_local_contract(registry)
            if session == 0:
                for sequence, record_id in enumerate(native.CASES):
                    case, _ = await measure_mcp_case(registry, record_id, sequence)
                    functional.append(case)
                for sequence, record_id in enumerate(
                    native.CASES * native.REPEATS, start=len(native.CASES)
                ):
                    _, (tool_ms, transport_ms) = await measure_mcp_case(
                        registry, record_id, sequence
                    )
                    tools.append(tool_ms)
                    transports.append(transport_ms)
            cleanup_started_at = perf_counter()
        cleanups.append((perf_counter() - cleanup_started_at) * 1000)
    return Gate12ComparisonReportV1(
        tool_contract=native_report.tool_contract,
        protocol=PROTOCOL_TARGET,
        server_name=SERVER_NAME,
        server_version=SERVER_VERSION,
        native=native_report,
        mcp=MCPSamples(
            functional_cases=tuple(functional),
            registry_tool=span_samples(tools),
            mcp_transport=span_samples(transports),
            process_startup_plus_protocol_discovery_ms=tuple(startups),
            cleanup_ms=tuple(cleanups),
        ),
    )


def serialize_report(report: Gate12ComparisonReportV1) -> str:
    # Revalidate nested instances too: model_copy/model_construct bypass validators.
    validated = Gate12ComparisonReportV1.model_validate(
        report.model_dump(mode="python", warnings=False)
    )
    return json.dumps(validated.model_dump(mode="json"), allow_nan=False, sort_keys=True)


def main() -> int:
    try:
        serialized = serialize_report(asyncio.run(run_comparison()))
    except Exception:
        print('{"error_category":"gate12_comparison_failed"}', file=sys.stderr)
        return 1
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
