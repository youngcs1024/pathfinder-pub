"""E5.3 instant CI smoke: one owned worker and one unique Run per environment."""

import json

import pytest

from tests.performance.__main__ import main
from tests.performance.metrics import Report, distribution, utc_difference
from tests.performance.smoke import SmokeRecord

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["research", "application"])
def test_instant_real_api_worker_graph_database_sse_and_synthetic_approval(
    tmp_path, mode, monkeypatch, request
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
    code = main(
        [
            "smoke",
            "--profile",
            "instant-v1",
            "--authorization",
            "e510_local_smoke_user_approved_v1",
            "--mode",
            mode,
            "--output",
            str(tmp_path / "smoke"),
        ]
    )
    result = SmokeRecord.model_validate_json(
        (tmp_path / "smoke" / "smoke-result.json").read_bytes()
    )
    from tests.ci_reports import smoke_projection

    request.node.user_properties.append(("smoke", smoke_projection(result)))
    # Fixed safe category on failure, never HTTP responses, model output or a DSN.
    if code != 0 or result.status != "PASS":
        pytest.fail(
            f"e53_smoke_{result.category or 'incomplete'}_stage_{result.failure_stage}"
            f"_missing_{','.join(result.missing_roles)}",
            pytrace=False,
        )
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

    assert result.metrics_status == "PASS"
    metrics = Report.model_validate_json((tmp_path / "smoke" / "metrics-result.json").read_bytes())
    assert metrics.status == "PASS" and not metrics.missing_roles
    assert metrics.source_commit == result.source_commit
    assert len({p.process_id for p in metrics.packets}) == 4
    assert all(not p.dropped and not p.write_failed for p in metrics.packets)
    by_role = {p.role: p for p in metrics.packets}
    worker_samples = by_role["worker"].samples
    timing = metrics.runs[0]
    segments = [s for s in worker_samples if s.metric == "segment"]
    assert len(segments) == timing.segments == (2 if mode == "application" else 1)
    assert timing.unfinished_segments == 0
    assert [s.claim_ordinal for s in segments] == list(range(1, len(segments) + 1))
    assert timing.worker_completed_segments_seconds.value == sum(
        s.finished - s.started for s in segments
    )
    assert timing.initial_queue_approx_seconds.value is not None
    assert timing.run_total_seconds.value is not None
    if mode == "application":
        assert timing.resume_queue_basis == ("decision_recorded_at",)
        assert timing.resume_queue_seconds[0].value is not None
        repository = [s for s in worker_samples if s.metric == "repository"]
        retrieval = [s for s in worker_samples if s.metric == "retrieval"]
        assert len(repository) == len(retrieval) == 1
        assert repository[0].sql_started > 0
        assert repository[0].sql_started == repository[0].sql_finished
        assert retrieval[0].started <= repository[0].started
        assert retrieval[0].finished >= repository[0].finished
    reads = [s for s in by_role["api"].samples if s.metric == "read_after"]
    assert reads and all(s.sql_started > 0 and s.sql_started == s.sql_finished for s in reads)
    assert sum(s.sql_failed for s in reads) == 0
    events = [s for s in by_role["driver"].samples if s.metric == "sse_event"]
    initial = [s for s in events if s.cursor == 0]
    assert len(initial) == result.event_count
    assert [s.seq for s in initial] == list(range(1, result.event_count + 1))
    assert all(s.run_id == result.run_id for s in events)
    assert metrics.sse_latency_seconds == distribution(
        [utc_difference(s.event_recorded_at, s.recorded_at) for s in events]
    )
    assert metrics.sse_latency_basis == "recorded_at_to_receive_not_commit"
    assert metrics.http_accounting.http_202 == 2
    assert metrics.http_accounting.replayed == 1
    assert (
        metrics.http_accounting.observed_unique_runs == metrics.http_accounting.persisted_runs == 1
    )
    assert metrics.recovery.status == "not_run" and metrics.recovery.cases == 0
    assert any(s.metric == "resource_db" for s in by_role["driver"].samples)
    assert any(s.metric == "resource_container" for s in by_role["supervisor"].samples)
    for packet in metrics.packets:
        for sample in packet.samples:
            if sample.resources is not None:
                for value in sample.resources.model_dump().values():
                    assert (value["value"] is None) == (value["reason"] is not None)


@pytest.mark.parametrize("failure", ["sse", "mock", "collector"])
def test_smoke_cli_failure_is_nonzero_and_retains_owned_evidence(tmp_path, monkeypatch, failure):
    from tests.performance import environment, metrics, smoke
    from tests.performance.workload import Fault

    observed = []
    canary = "E510_PRIVATE_FAILURE_CANARY"
    if failure == "sse":

        async def fail_stream(self, path, *, cursor=0):
            observed.append("sse")
            raise smoke.SmokeFailure("http_failed")

        monkeypatch.setattr(smoke.HTTP, "events", fail_stream)
    elif failure == "mock":

        class RejectingEnvironment(environment.IsolatedEnvironment):
            def start(self):
                # Test-only injection below Registry: use the existing worker call adapter.
                # The persisted call-profile records the actual fault, not a success profile.
                self.call_profile = {
                    **self.call_profile,
                    "faults": [
                        Fault(call="mock_submit", ordinal=1, kind="permanent").model_dump(
                            mode="json"
                        )
                    ],
                }
                observed.append("mock")
                return super().start()

        monkeypatch.setattr(smoke, "IsolatedEnvironment", RejectingEnvironment)
    else:
        publish = metrics.publish

        def fail_collector(directory, name, record):
            if name == "metrics-driver.json":
                observed.append("collector")
                raise environment.EnvironmentError("report_failed")
            publish(directory, name, record)

        monkeypatch.setattr(metrics, "publish", fail_collector)
    for key in ("PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "PF_DATABASE_URL", "HTTPS_PROXY"):
        monkeypatch.setenv(key, canary)
    output = tmp_path / "failed-smoke"
    assert (
        main(
            [
                "smoke",
                "--profile",
                "instant-v1",
                "--authorization",
                "e510_local_smoke_user_approved_v1",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    result = SmokeRecord.model_validate_json((output / "smoke-result.json").read_bytes())
    assert observed and result.status == "IN_PROGRESS"
    assert result.resources_released
    assert result.profile == "instant-v1" and result.mode == "application"
    assert result.http_requests <= 128 and sum(result.call_counts.values()) <= 64
    assert (
        result.category
        == {"sse": "http_failed", "mock": "business_failed", "collector": "evidence_mismatch"}[
            failure
        ]
    )
    assert result.failure_stage == ("metrics" if failure == "collector" else "drive")
    if failure == "mock":
        records = [json.loads(p.read_text()) for p in output.glob("calls-worker-*-finished.json")]
        failed = [r for r in records if r["call"] == "mock_submit"]
        assert len(failed) == 1 and failed[0]["outcome"] == "failed"
    if failure == "collector":
        assert result.mock_effects == 1  # Business completed, but its measurement cannot pass.
        assert result.metrics_status == "IN_PROGRESS"
        assert "metrics_incomplete" in result.diagnostic_errors
        assert "write_failed" in result.metrics_reasons
    for path in output.iterdir():
        body = path.read_text()
        if canary in body:
            pytest.fail("e510_artifact_leak", pytrace=False)
        json.loads(body)
