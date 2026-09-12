from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest

from app.llm.invocations import LLMInvocationOutcome
from app.llm.ports import ModelUsage
from tests.evals.live_contracts import LiveSmokeReportV1
from tests.evals.live_smoke import (
    _check_cost_budget,
    _LiveBudgetError,
    _TrackingRecorder,
    run_live_smoke,
)


class _RecorderDouble:
    async def prepare(self, attempt: object) -> None:
        return None

    async def finalize(self, attempt: object, outcome: object) -> None:
        return None


def test_cached_usage_fails_closed_as_cost_unavailable() -> None:
    recorder = _TrackingRecorder(_RecorderDouble())  # type: ignore[arg-type]
    recorder.outcomes[UUID(int=1)] = LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(
            input_tokens=10,
            output_tokens=2,
            cached_input_tokens=1,
        ),
        latency_ms=1,
    )

    with pytest.raises(_LiveBudgetError) as captured:
        _check_cost_budget(recorder)

    assert captured.value.category == "cost_unavailable"


def test_known_cost_at_limit_stops_before_another_logical_call() -> None:
    recorder = _TrackingRecorder(_RecorderDouble())  # type: ignore[arg-type]
    recorder.outcomes[UUID(int=1)] = LLMInvocationOutcome(
        status="succeeded",
        token_usage=ModelUsage(input_tokens=1, output_tokens=1),
        latency_ms=1,
        pricing_version="qwen-cn-beijing-cny-test-v1",
        currency="CNY",
        estimated_cost=Decimal("1.00"),
    )

    with pytest.raises(_LiveBudgetError) as captured:
        _check_cost_budget(recorder)

    assert captured.value.category == "budget_exceeded"


def test_live_report_forbids_provider_bodies_and_secret_extension_fields() -> None:
    payload = LiveSmokeReportV1(
        passed=False,
        error_category="provider_failure",
    ).model_dump(mode="json")
    payload["provider_body"] = "live-smoke-secret-body-canary"

    with pytest.raises(ValueError):
        LiveSmokeReportV1.model_validate(payload, strict=True)


async def test_database_revision_preflight_fails_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        "PF_LLM_MODE": "qwen",
        "PF_QWEN_WORKSPACE_ID": "workspace-test",
        "DASHSCOPE_API_KEY": "qwen-test",
        "PF_SEARCH_MODE": "tavily",
        "TAVILY_API_KEY": "tavily-test",
        "PF_TRACE_MODE": "langfuse",
        "LANGFUSE_PUBLIC_KEY": "public-test",
        "LANGFUSE_SECRET_KEY": "secret-test",
        "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
        "LANGFUSE_SAMPLE_RATE": "1.0",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    async def not_ready(_probe: object) -> bool:
        return False

    monkeypatch.setattr("tests.evals.live_smoke.DatabaseReadinessProbe.is_ready", not_ready)
    report, exit_code, missing = await run_live_smoke()

    assert exit_code == 2
    assert missing == ()
    assert report.error_category == "invalid_database_revision"
    assert report.provider_attempts == 0
