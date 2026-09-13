"""Tiny instant E5.7 checks; manual delayed profiles never run through collection."""

import json

import httpx
import pytest

from tests.performance.queue import run_point
from tests.performance.queue_contracts import Point, load_profile

pytestmark = pytest.mark.integration


def checked(result):
    if result.status != "PASS":
        pytest.fail(
            f"e57_{result.point.kind}_{result.stop}_{'_'.join(result.diagnostics)}", pytrace=False
        )
    assert result.completeness and result.correctness and result.resources_released
    assert result.counts["pool_remaining"] == 0
    assert [s.stage for s in result.snapshots] == [
        "submission_stopped",
        "drain_cutoff",
        "worker_stopped",
    ]


@pytest.mark.parametrize("mode", ["research", "application"])
def test_queue_instant_baseline_and_bounded_arrivals(tmp_path, monkeypatch, mode):
    canary = "E57_PROVIDER_BODY_CANARY"
    for key in ("PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "PF_DATABASE_URL", "HTTP_PROXY"):
        monkeypatch.setenv(key, canary)
    profile = load_profile("queue-instant-ci-v1")
    base = run_point(
        tmp_path / "baseline",
        point=Point(kind="baseline", mode=mode),
        profile=profile,
        authorization="ci_instant",
    )
    checked(base)
    assert base.baseline and len(base.baseline.service_seconds) == 2
    if mode == "research":

        class LostOnceClient(httpx.AsyncClient):
            lost = False

            async def post(self, *args, **kwargs):
                response = await super().post(*args, **kwargs)
                if not self.lost and response.status_code == 202:
                    self.lost = True
                    await response.aclose()
                    raise httpx.ReadTimeout("E57_RESPONSE_DELIVERY_CANARY")
                return response

        monkeypatch.setattr(httpx, "AsyncClient", LostOnceClient)
    result = run_point(
        tmp_path / "load",
        point=Point(kind="load", mode=mode, factor=0.8),
        profile=profile,
        authorization="ci_instant",
        baseline=base.baseline,
    )
    checked(result)
    assert result.load and result.counts["new_runs"] >= 1
    assert result.counts["new_runs"] == len(result.requests)
    if mode == "research":
        assert result.counts["http_unknown"] == 1
        assert result.counts["new_runs"] == result.counts["http_accepted"] + 1
    assert result.load.measurement.offered >= result.counts["new_runs"]
    for directory in (tmp_path / "baseline", tmp_path / "load"):
        for path in directory.iterdir():
            raw = path.read_text()
            if canary in raw:
                pytest.fail("e57_artifact_leak", pytrace=False)
            json.loads(raw)


def test_queue_instant_approval_release_and_control_under_backlog(tmp_path):
    result = run_point(
        tmp_path / "control",
        point=Point(kind="control", mode="application"),
        profile=load_profile("queue-instant-ci-v1"),
        authorization="ci_instant",
    )
    checked(result)
    assert result.control.worker_released and result.control.backlog > 0
    assert result.control.approval_http_accepted and result.control.cancel_http_accepted
    assert result.control.converged
    assert result.counts["new_runs"] == 5
    assert result.counts["cancelled_at_cutoff"] == 1
    assert sum(f.mock_effects for f in result.snapshots[-1].runs) == 1
