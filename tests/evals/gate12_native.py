"""Offline Gate 12.1 native experiment; stdout only, never an accepted baseline."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from itertools import count
from time import monotonic, perf_counter
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.tracing import (
    TraceIdentity,
    TraceSpanFinish,
    bind_trace_scope,
    finish_trace_span,
    start_trace_span,
)
from app.llm.ports import ModelToolCall
from app.tools.contracts import ToolRunContext, ToolRuntime
from app.tools.registry import ToolRegistry
from app.tools.teaching import (
    LOOKUP_SYNTHETIC_RECORD_SPEC,
    LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
    MANUAL_LEARNING_POLICY_NAME,
    LookupSyntheticRecordOutput,
    create_teaching_tool_registry,
)
from tests.tracing import CollectingTraceSink, collecting_segment

CASES = ("record-1", "missing-record", "record-2")
EXPECTED_SUMMARIES = (
    "Synthetic backend role record.",
    None,
    "Synthetic platform engineering record.",
)
REPEATS = 3
NonnegativeInt = Annotated[int, Field(ge=0)]


class NativeBaselineError(ValueError):
    """An internal result or observation cannot support the native report."""


class _StrictReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class NativeCase(_StrictReport):
    record_id: Literal["record-1", "missing-record", "record-2"]
    outcome: Literal["succeeded"] = "succeeded"
    found: bool


class NativeContract(_StrictReport):
    effect: Literal["read_only"]
    credential_source: Literal["none"]
    timeout_seconds: Literal[1.0]
    max_attempts: Literal[1]
    per_run_call_limit: Literal[3]
    max_output_bytes: Literal[4096]


class NativeReport(_StrictReport):
    schema_version: Literal[1] = 1
    experiment_id: Literal["gate12-native-baseline"] = "gate12-native-baseline"
    tool_name: Literal["lookup_synthetic_record"] = "lookup_synthetic_record"
    tool_contract: NativeContract
    functional_cases: tuple[NativeCase, ...]
    measurement_boundary: Literal["gate10_registry_tool_span"] = "gate10_registry_tool_span"
    latency_source: Literal["TraceSpanFinish.latency_ms"] = "TraceSpanFinish.latency_ms"
    latency_resolution_ms: Literal[1] = 1
    warmup_call_count: Literal[3] = 3
    repeats: Literal[3] = REPEATS
    sample_count: Literal[9] = 9
    steady_state_samples_ms: tuple[NonnegativeInt, ...]
    p50_ms: NonnegativeInt
    p95_ms: NonnegativeInt
    setup_boundary: Literal["registry_construction_and_first_bind"] = (
        "registry_construction_and_first_bind"
    )
    native_setup_ms: Annotated[float, Field(ge=0)]
    external_network_used: Literal[False] = False
    external_credentials_used: Literal[False] = False

    @model_validator(mode="after")
    def evidence_must_agree(self) -> NativeReport:
        if tuple(case.record_id for case in self.functional_cases) != CASES:
            raise ValueError("native cases do not match the fixed suite")
        if tuple(case.found for case in self.functional_cases) != (True, False, True):
            raise ValueError("native outcomes do not match the fixed suite")
        if len(self.steady_state_samples_ms) != self.sample_count:
            raise ValueError("native sample count mismatch")
        if (self.p50_ms, self.p95_ms) != latency_percentiles(self.steady_state_samples_ms):
            raise ValueError("native latency aggregate mismatch")
        return self


def latency_percentiles(samples: Sequence[int]) -> tuple[int, int]:
    """Nearest-rank p50/p95 over nonnegative integer Gate 10 milliseconds."""
    if not samples or any(type(value) is not int or value < 0 for value in samples):
        raise NativeBaselineError("invalid latency samples")
    ordered = sorted(samples)
    return tuple(ordered[(len(ordered) * percentile + 99) // 100 - 1] for percentile in (50, 95))


class NotCancelled:
    def is_cancelled(self) -> bool:
        return False


def native_context(sequence: int) -> ToolRunContext:
    return ToolRunContext(
        workspace_id=UUID(int=1),
        actor_user_id=UUID(int=2),
        run_id=UUID(int=100 + sequence),
        action_intent_id=None,
        approval_request_id=None,
        trusted_target=None,
        deadline=monotonic() + 30.0,
        cancellation=NotCancelled(),
    )


def bind_native(registry: ToolRegistry, sequence: int) -> ToolRuntime:
    ids = count(10000 + sequence * 100)
    return registry.bind(
        policy_name=MANUAL_LEARNING_POLICY_NAME,
        context=native_context(sequence),
        invocation_id_factory=lambda: UUID(int=next(ids)),
    )


def native_call(record_id: str) -> ModelToolCall:
    return ModelToolCall(
        call_id="native-call",
        name=LOOKUP_SYNTHETIC_RECORD_TOOL_NAME,
        arguments={"record_id": record_id},
    )


def tool_observation(sink: CollectingTraceSink) -> TraceSpanFinish:
    tools = [(context, span) for context, span in sink.starts.values() if span.span_kind == "tool"]
    if len(tools) != 1:
        raise NativeBaselineError("native tool span missing or duplicated")
    context, span = tools[0]
    finish = sink.finishes.get(context.context_id)
    if (
        span.metadata.get("tool_name") != LOOKUP_SYNTHETIC_RECORD_TOOL_NAME
        or span.metadata.get("tool_effect") != "read_only"
        or finish is None
        or finish.status != "succeeded"
        or finish.metadata.get("attempt_number") != 1
        or finish.metadata.get("retry_count") != 0
    ):
        raise NativeBaselineError("native tool span invalid")
    return finish


@contextmanager
def native_trace(sink: CollectingTraceSink, sequence: int):
    identity = TraceIdentity(workspace_id=UUID(int=1), run_id=UUID(int=100 + sequence))
    with collecting_segment(sink, trace_identity=identity) as scope:
        started_at = monotonic()
        parent = start_trace_span(
            scope, span_kind="graph_node", metadata={"node_name": "research_agent"}
        )
        with bind_trace_scope(replace(scope, parent=parent)):
            try:
                yield
            finally:
                finish_trace_span(scope, parent, started_at=started_at, status="succeeded")


async def measure_case(
    runtime: ToolRuntime, record_id: str, sequence: int = 0
) -> tuple[NativeCase, int]:
    sink = CollectingTraceSink()
    with native_trace(sink, sequence):
        serialized = await runtime.execute(native_call(record_id))
    # Revalidate the runtime boundary; never copy arbitrary output into the report.
    try:
        result = LookupSyntheticRecordOutput.model_validate_json(serialized, strict=True)
        expected_summary = EXPECTED_SUMMARIES[CASES.index(record_id)]
        if result.record_id != record_id or result.summary != expected_summary:
            raise ValueError
    except (ValueError, TypeError):
        raise NativeBaselineError("native result invalid") from None
    finish = tool_observation(sink)
    if finish.metadata.get("output_bytes") != len(serialized.encode("utf-8")):
        raise NativeBaselineError("native output byte observation invalid")
    return NativeCase(record_id=record_id, found=result.found), finish.latency_ms


async def run_native_baseline() -> NativeReport:
    started_at = perf_counter()
    registry = create_teaching_tool_registry()
    first_runtime = bind_native(registry, 0)
    setup_ms = (perf_counter() - started_at) * 1000
    functional = []
    for sequence, record_id in enumerate(CASES):
        runtime = first_runtime if sequence == 0 else bind_native(registry, sequence)
        case, _ = await measure_case(runtime, record_id, sequence)
        functional.append(case)
    samples = []
    for sequence, record_id in enumerate(CASES * REPEATS, start=len(CASES)):
        _, latency_ms = await measure_case(bind_native(registry, sequence), record_id, sequence)
        samples.append(latency_ms)
    p50, p95 = latency_percentiles(samples)
    spec = LOOKUP_SYNTHETIC_RECORD_SPEC
    return NativeReport(
        tool_contract=NativeContract(
            effect=spec.effect.value,
            credential_source=spec.credential_source.value,
            timeout_seconds=spec.timeout_seconds,
            max_attempts=spec.max_attempts,
            per_run_call_limit=spec.per_run_call_limit,
            max_output_bytes=spec.max_output_bytes,
        ),
        functional_cases=tuple(functional),
        steady_state_samples_ms=tuple(samples),
        p50_ms=p50,
        p95_ms=p95,
        native_setup_ms=setup_ms,
    )


def serialize_report(report: NativeReport) -> str:
    # model_copy/model_construct can bypass validators; validate again before emission.
    validated = NativeReport.model_validate(report.model_dump(mode="python"))
    return json.dumps(validated.model_dump(mode="json"), allow_nan=False, sort_keys=True)


def main() -> int:
    try:
        serialized = serialize_report(asyncio.run(run_native_baseline()))
    except Exception:
        # CLI boundary: unexpected internal failures must not expose arbitrary exception text.
        print('{"error_category":"native_baseline_failed"}', file=sys.stderr)
        return 1
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
