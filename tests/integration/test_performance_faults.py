"""E5.9 one tiny repetition per window in CI; fixed full matrix is manual only."""

import json

import pytest

from tests.performance.fault_contracts import CRASH, SCENARIOS, Point, load_profile
from tests.performance.faults import check_facts, run_point

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_fault_window_converges_with_owned_resources(tmp_path, monkeypatch, scenario):
    canary = "E59_INHERITED_SECRET_CANARY"
    for key in ("PF_DATABASE_URL", "PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "HTTPS_PROXY"):
        monkeypatch.setenv(key, canary)
    result = run_point(
        tmp_path / "point",
        point=Point(scenario=scenario, repetition=1),
        profile=load_profile("faults-instant-ci-v1"),
        authorization="ci_instant",
    )
    if result.status != "PASS":
        pytest.fail(f"e59_{scenario}_{result.stop}_{result.stage}", pytrace=False)
    assert result.resources_released and result.pool_remaining == 0
    assert result.complete and result.expected_state_correct
    assert result.automatic_business_complete == (result.point.expected_run == "completed")
    assert result.manual_verification_required == (scenario == "lookup_unavailable")
    assert (result.restart_to_convergence is not None) == (scenario in CRASH)
    check_facts(result.point, result.before, result.final, result.lifecycle, result.calls)
    for path in (tmp_path / "point").iterdir():
        raw = path.read_text()
        if canary in raw or "Synthetic Resume" in raw or "synthetic candidate" in raw:
            pytest.fail("e59_artifact_leak", pytrace=False)
        json.loads(raw)


def test_fault_collector_failure_retains_partial_and_cleans_up(tmp_path, monkeypatch):
    from tests.performance import faults

    async def failed(*args):
        raise ValueError("E59_COLLECTOR_PRIVATE_CANARY")

    monkeypatch.setattr(faults, "snapshot", failed)
    result = run_point(
        tmp_path / "point",
        point=Point(scenario="claim", repetition=1),
        profile=load_profile("faults-instant-ci-v1"),
        authorization="ci_instant",
    )
    assert result.status == "IN_PROGRESS" and result.stop == "correctness_failed"
    assert result.resources_released and result.pool_remaining == 0
    assert not result.complete and not result.expected_state_correct
    assert (tmp_path / "point" / "fault-result.json").exists()
    assert "E59_COLLECTOR_PRIVATE_CANARY" not in result.model_dump_json()
