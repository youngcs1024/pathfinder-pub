"""E5.3 instant CI smoke: one owned worker and one unique Run per environment."""

import json

import pytest

from tests.performance.smoke import run_smoke

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["research", "application"])
def test_instant_real_api_worker_graph_database_sse_and_synthetic_approval(
    tmp_path, mode, monkeypatch
):
    canary = "E53_INHERITED_PROVIDER_SECRET_CANARY"
    for key in (
        "PF_QWEN_API_KEY",
        "PF_TAVILY_API_KEY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "PF_DATABASE_URL",
    ):
        monkeypatch.setenv(key, canary)
    result = run_smoke(tmp_path / "smoke", mode=mode)
    # Fixed safe category on failure, never HTTP responses, model output or a DSN.
    if result.status != "PASS":
        pytest.fail(f"e53_smoke_{result.category or 'incomplete'}", pytrace=False)
    assert result.resources_released and result.unfinished_calls == 0
    assert result.model_attempts > 0 and result.event_count > 0
    assert result.mock_effects == (1 if mode == "application" else 0)
    assert result.approval_mode == ("synthetic_driver" if mode == "application" else "none")
    assert sum(result.call_counts.values()) <= 64 and result.http_requests <= 128
    assert result.elapsed_seconds <= 45  # 40 seconds business plus up to 5 seconds DB disposal.
    for path in (tmp_path / "smoke").iterdir():
        body = path.read_text()
        if canary in body:
            pytest.fail("e53_artifact_leak", pytrace=False)
        json.loads(body)
