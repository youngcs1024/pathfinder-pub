"""Focused comparison evidence; existing suites own the failure/security matrix."""

import json
import socket
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from pydantic import ValidationError

from app.tools.registry import ToolRegistry
from app.tools.teaching import LOOKUP_SYNTHETIC_RECORD_SPEC
from tests.evals import gate12_compare as compare
from tests.tracing import CollectingTraceSink

CANARY = "comparison-private-secret-canary"


@pytest.fixture(scope="module")
async def report():
    return await compare.run_comparison()


def test_report_exact_results_and_sanitized_roundtrip(report):
    serialized = compare.serialize_report(report)
    assert compare.Gate12ComparisonReportV1.model_validate_json(serialized) == report
    assert [case.found for case in report.mcp.functional_cases] == [True, False, True]
    assert report.mcp.functional_cases == report.native.functional_cases
    assert report.contract_equivalent is report.results_equivalent is True
    for forbidden in (
        "workspace_id",
        "actor_user_id",
        "run_id",
        "invocation_id",
        "summary",
        "trusted_target",
        "hostname",
        "stderr",
        "Synthetic backend role record.",
        "Synthetic platform engineering record.",
        CANARY,
    ):
        assert forbidden not in serialized
    assert report.raw_pre_parse_cap_satisfied is report.os_sandbox is False
    assert report.production_dependency is False


async def test_real_session_measurements_use_tool_and_child_boundaries(monkeypatch, capfd):
    counts = {"native": 0, "mcp": 0, "transport": 0}
    active = False
    session_calls = []
    original_finish = CollectingTraceSink.finish
    original_open = compare.open_mcp_experiment
    seen_sinks = []

    def finish(sink, context, outcome):
        if outcome.span_kind == "tool":
            side = "mcp" if active else "native"
            counts[side] += 1
            value = 101 if counts[side] <= 3 else (11 if active else 7)
            outcome = replace(outcome, latency_ms=value)
        elif outcome.span_kind == "mcp_transport":
            counts["transport"] += 1
            outcome = replace(outcome, latency_ms=99 if counts["transport"] <= 3 else 3)
            seen_sinks.append(sink)
        original_finish(sink, context, outcome)

    @asynccontextmanager
    async def measured_session():
        nonlocal active
        before = counts["mcp"]
        async with original_open() as registry:
            active = True
            yield registry
            active = False
        session_calls.append(counts["mcp"] - before)

    def deny_external_connect(sock, address):
        raise AssertionError("comparison attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny_external_connect)
    for key in (
        "DASHSCOPE_API_KEY",
        "TAVILY_API_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "LANGFUSE_SECRET_KEY",
        "PF_GATE12_SECRET_CANARY",
    ):
        monkeypatch.setenv(key, CANARY)
    ticks = iter([0.0, 0.25, 1.0, 1.5] * 3)
    monkeypatch.setattr(compare, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(compare, "open_mcp_experiment", measured_session)
    monkeypatch.setattr(CollectingTraceSink, "finish", finish)
    result = await compare.run_comparison()
    assert counts == {"native": 12, "mcp": 12, "transport": 12}
    assert session_calls == [12, 0, 0]
    assert result.native.steady_state_samples_ms == (7,) * 9
    assert result.mcp.registry_tool.samples_ms == (11,) * 9
    assert result.mcp.mcp_transport.samples_ms == (3,) * 9
    assert result.mcp.registry_tool.p50_ms == result.mcp.registry_tool.p95_ms == 11
    assert result.mcp.process_startup_plus_protocol_discovery_ms == (250.0,) * 3
    assert result.mcp.cleanup_ms == (500.0,) * 3
    captured = capfd.readouterr()
    assert captured.out == captured.err == ""
    observed = compare.serialize_report(result) + "".join(sink.safe_json() for sink in seen_sinks)
    assert CANARY not in observed


@pytest.mark.parametrize(
    "change",
    [
        {"sample_count": 8},
        {"samples_ms": (0,)},
        {"samples_ms": (True,) * 9},
        {"samples_ms": (float("nan"),) * 9},
        {"p95_ms": 999999},
    ],
)
def test_nested_span_corruption_fails_serialization(report, change):
    bad_span = report.mcp.registry_tool.model_copy(update=change)
    bad_mcp = report.mcp.model_copy(update={"registry_tool": bad_span})
    with pytest.raises(ValidationError):
        compare.serialize_report(report.model_copy(update={"mcp": bad_mcp}))


@pytest.mark.parametrize(
    "change",
    [
        {"process_startup_plus_protocol_discovery_ms": (float("inf"),) * 3},
        {"cleanup_ms": (-1.0,) * 3},
        {"cleanup_ms": ()},
        {"functional_cases": ()},
    ],
)
def test_lifecycle_and_results_cannot_be_fabricated(report, change):
    with pytest.raises(ValidationError):
        compare.serialize_report(
            report.model_copy(update={"mcp": report.mcp.model_copy(update=change)})
        )


@pytest.mark.parametrize(
    "change",
    [
        {"max_attempts": 2},
        {"max_output_bytes": 4095},
        {"timeout_seconds": 2.0},
    ],
)
def test_local_contract_drift_is_rejected(change):
    registry = ToolRegistry(
        specs=(replace(LOOKUP_SYNTHETIC_RECORD_SPEC, **change),),
        policies=(compare.MCP_EXPERIMENT_POLICY,),
    )
    with pytest.raises(compare.ComparisonError):
        compare.validate_local_contract(registry)


@pytest.mark.parametrize("wrong", ["id", "summary", "found", "extra"])
async def test_runtime_output_is_strictly_revalidated(wrong):
    class BrokenRuntime:
        async def execute(self, call):
            output = {
                "record_id": "record-1",
                "found": True,
                "summary": compare.native.EXPECTED_SUMMARIES[0],
            }
            output[{"id": "record_id", "extra": "role"}.get(wrong, wrong)] = CANARY
            return json.dumps(output)

    class BrokenRegistry:
        def bind(self, **kwargs):
            return BrokenRuntime()

    with pytest.raises((compare.ComparisonError, ValidationError)):
        await compare.measure_mcp_case(BrokenRegistry(), "record-1", 0)


async def test_missing_transport_is_not_reported_as_zero(monkeypatch):
    original_start = CollectingTraceSink.start

    def drop_transport(sink, span):
        return None if span.span_kind == "mcp_transport" else original_start(sink, span)

    monkeypatch.setattr(CollectingTraceSink, "start", drop_transport)
    with pytest.raises(compare.ComparisonError):
        await compare.run_comparison()


def test_cli_success_strict_json(report, monkeypatch, capsys):
    async def measured():
        return report

    monkeypatch.setattr(compare, "run_comparison", measured)
    assert compare.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert compare.Gate12ComparisonReportV1.model_validate_json(captured.out) == report


def test_corrupt_report_cannot_leak_serializer_warning(report, monkeypatch, capsys, recwarn):
    bad_span = report.mcp.registry_tool.model_copy(update={"samples_ms": (CANARY,) * 9})
    bad_mcp = report.mcp.model_copy(update={"registry_tool": bad_span})

    async def corrupt():
        return report.model_copy(update={"mcp": bad_mcp})

    monkeypatch.setattr(compare, "run_comparison", corrupt)
    assert compare.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error_category": "gate12_comparison_failed"}
    assert not recwarn


@pytest.mark.parametrize("error", [compare.ComparisonError, RuntimeError])
def test_cli_failure_is_fixed_sanitized_category(error, monkeypatch, capsys):
    async def broken():
        raise error(CANARY)

    monkeypatch.setattr(compare, "run_comparison", broken)
    assert compare.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error_category": "gate12_comparison_failed"}
