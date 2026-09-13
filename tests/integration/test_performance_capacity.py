"""Single-connection instant E5.6 regression, not a capacity measurement."""

import json

import pytest

from tests.performance.capacity import run_point
from tests.performance.capacity_contracts import Point, load_profile

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("scenario", ["api", "mixed", "idle", "active", "reconnect", "slow"])
def test_capacity_instant_owned_real_paths(tmp_path, monkeypatch, scenario):
    canary = "E56_PROVIDER_BODY_CANARY"
    for key in ("PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "PF_DATABASE_URL", "HTTP_PROXY"):
        monkeypatch.setenv(key, canary)
    result = run_point(
        tmp_path / "point",
        point=Point(scenario=scenario, level=1),
        profile=load_profile("capacity-instant-ci-v1"),
        authorization="ci_instant",
    )
    if result.status != "PASS":
        pytest.fail(f"e56_{scenario}_{result.stop}_{'_'.join(result.diagnostics)}", pytrace=False)
    assert result.resources_released and result.completeness and result.correctness
    assert result.http_requests <= 512
    assert result.counts["pool_remaining"] == 0
    if scenario in {"idle", "api"}:
        assert all(r.status == "queued" and not r.model_attempts for r in result.runs)
    else:
        assert all(r.status == "completed" and r.job_status == "done" for r in result.runs)
    if scenario == "mixed":
        assert len(result.runs) == 4
        assert sum(r.mock_effects for r in result.runs) == 2
    if scenario in {"reconnect", "slow"}:
        assert len(result.connections) == 2
        first, second = result.connections
        assert second.cursor == first.processed[-1] and second.terminal
    for path in (tmp_path / "point").iterdir():
        raw = path.read_text()
        if canary in raw:
            pytest.fail("e56_artifact_leak", pytrace=False)
        json.loads(raw)
