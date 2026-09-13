"""E5.4 artifact and accounting contracts, without environment or load execution."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from tests.performance.contracts import (
    Manifest,
    Receipt,
    RequestRecord,
    Result,
    Slot,
    count,
    digest,
    load_profile,
)
from tests.performance.environment import EnvironmentError
from tests.performance.workload import profile, publish


def manifest(name="scheduler-smoke-v1", **changes):
    base = load_profile(name)
    policy = type(base).model_validate_json(json.dumps({**base.model_dump(mode="json"), **changes}))
    calls = profile("instant-v1" if name == "scheduler-smoke-v1" else "delayed-v1")
    return Manifest(
        experiment_id=uuid4(),
        source_sha="a" * 40,
        lock_digest="b" * 64,
        database_image="pgvector/pgvector:0.8.5-pg16",
        database_version="16.10",
        environment_profile="environment-v1",
        profile=policy,
        profile_digest=digest(policy),
        call_profile=calls,
        call_profile_digest=digest(calls),
    )


def test_fixed_profiles_and_existing_environment_bounds():
    smoke, low = manifest(), manifest("low-load-v1")
    assert (smoke.profile.arrival_rate, smoke.profile.max_requests) == (2, 2)
    assert (
        low.profile.warmup_seconds,
        low.profile.measurement_seconds,
        low.profile.drain_seconds,
    ) == (5, 20, 10)
    for item in (smoke, low):
        assert item.profile.max_run_seconds <= 180
        assert item.call_profile.max_calls <= 63
        assert item.profile.resources.workers == 1
        assert item.profile.resources.database_cpu == 1
        assert item.profile.resources.database_memory_bytes == 1024**3
        assert item.profile.resources.database_tmpfs_bytes == 256 * 1024**2


@pytest.mark.parametrize(
    "changes",
    [
        {"arrival_rate": 0},
        {"arrival_rate": -1},
        {"arrival_rate": float("nan")},
        {"arrival_rate": float("inf")},
        {"arrival_rate": True},
        {"max_requests": 129},
        {"max_requests": 0},
        {"max_inflight": 2},
        {"max_run_seconds": 1.0},
        {"max_run_seconds": 241.0},
        {"warmup_seconds": -1.0},
        {"drain_seconds": 0.0},
        {"max_queue_depth": 0},
        {"connections": 200},
        {"saturation": "delay"},
        {"llm_mode": "qwen"},
        {"search_mode": "tavily"},
        {"auth_mode": "supabase"},
        {"trace_mode": "langfuse"},
        {"database_url": "PRIVATE_CANARY"},
        {"resources": {"workers": 2}},
    ],
)
def test_invalid_profiles_rejected(changes):
    with pytest.raises(ValidationError):
        manifest(**changes)


@pytest.mark.parametrize("name", ["../PRIVATE_CANARY", "unknown", "", "/tmp/policy"])
def test_profile_loader_rejects_arbitrary_paths(name):
    with pytest.raises(EnvironmentError, match=r"^invalid_profile$") as error:
        load_profile(name)
    assert "PRIVATE_CANARY" not in str(error.value)


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "source_sha",
        "lock_digest",
        "database_image",
        "database_version",
        "environment_profile",
        "profile_digest",
        "call_profile_digest",
    ],
)
def test_actual_identity_required(field):
    data = manifest().model_dump(mode="json")
    del data[field]
    with pytest.raises(ValidationError):
        Manifest.model_validate_json(json.dumps(data))


def test_profile_identity_cannot_silently_change():
    data = manifest().model_dump(mode="json")
    data["profile"]["seed"] += 1
    with pytest.raises(ValidationError, match="identity_mismatch"):
        Manifest.model_validate_json(json.dumps(data))
    data = manifest().model_dump(mode="json")
    data["call_profile"]["seed"] += 1
    with pytest.raises(ValidationError, match="identity_mismatch"):
        Manifest.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "changes",
    [
        {"source_sha": "PRIVATE_CANARY"},
        {"database_version": "16.10 PRIVATE_CANARY"},
        {"experiment_id": "00000000-0000-1000-8000-000000000000"},
    ],
)
def test_invalid_identity_values(changes):
    with pytest.raises(ValidationError):
        Manifest.model_validate_json(json.dumps({**manifest().model_dump(mode="json"), **changes}))


@pytest.mark.parametrize(
    "data",
    [
        {"sent": False, "http_status": 202},
        {"sent": None, "http_status": 500},
        {"sent": True, "http_status": 500, "replayed": True},
        {"sent": True, "http_status": 200, "run_id": str(uuid4())},
        {"sent": True, "body": "PRIVATE_CANARY"},
        {"sent": True, "http_status": 600},
    ],
)
def test_receipt_refuses_impossible_or_private_fields(data):
    with pytest.raises(ValidationError):
        Receipt.model_validate_json(json.dumps(data))


def record(ordinal, outcome="finished", **receipt):
    return RequestRecord(
        slot=Slot(ordinal=ordinal, fixture_seed=54, phase="measurement", scheduled_at=0.0),
        outcome=outcome,
        started_at=0.0 if outcome in {"finished", "failed", "cancelled"} else None,
        finished_at=0.0 if outcome in {"finished", "failed", "cancelled"} else None,
        receipt=Receipt(**receipt),
    )


def test_accounting_preserves_replays_failures_and_unknown_denominators():
    run_id = uuid4()
    rows = (
        record(0, sent=True, http_status=202, replayed=False, run_id=run_id),
        record(1, sent=True, http_status=202, replayed=True, run_id=run_id),
        record(2, sent=True, http_status=202),
        record(3, "failed", sent=True, http_status=503),
        record(4, "failed", sent=None),
        record(5, "generator_dropped", sent=False),
        record(6, "cancelled", sent=None),
        record(7, "unfinished", sent=None),
    )
    result = count(rows)
    assert result.offered == 8
    assert result.sent == 4
    assert result.sent_unknown == result.http_outcome_unknown == 3
    assert result.http_accepted == 3
    assert result.replays == result.observed_unique_runs == 1
    assert result.replay_unknown == result.accepted_run_unknown == 1
    assert result.generator_dropped == result.cancelled == result.unfinished == 1
    assert result.failed == 2
    assert result.http_failed == result.not_sent == 1
    assert result.offered == result.sent + result.sent_unknown + result.not_sent


def test_result_rejects_forged_counts_and_duplicate_slots():
    rows = (record(0, sent=False),)
    identity = manifest()
    data = dict(
        experiment_id=identity.experiment_id,
        manifest_digest=digest(identity),
        stop_reason="request_limit",
        drain_timed_out=False,
        tasks_released=True,
        elapsed_seconds=1.0,
        submission_stopped_at=0.5,
        records=rows,
        warmup=count(()),
        measurement=count(rows),
    )
    Result(**data)
    with pytest.raises(ValidationError, match="invalid_counts"):
        Result(**{**data, "measurement": count(())})
    with pytest.raises(ValidationError, match="invalid_ordinals"):
        Result(**{**data, "records": rows + rows})


def test_artifacts_are_create_only_and_revalidate_nested_models(tmp_path):
    item = manifest()
    publish(tmp_path, "load-manifest.json", item)
    saved = (tmp_path / "load-manifest.json").read_bytes()
    assert (tmp_path / "load-manifest.json").stat().st_mode & 0o777 == 0o600
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "load-manifest.json", item)
    assert (tmp_path / "load-manifest.json").read_bytes() == saved
    forged = item.model_copy(update={"database_version": "PRIVATE_CANARY"})
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "load-result.json", forged)
    assert not (tmp_path / "load-result.json").exists()
    assert b"PRIVATE_CANARY" not in saved


@pytest.mark.parametrize("name", ["load-body.json", "load-results.json", "../load-result.json"])
def test_publisher_does_not_accept_arbitrary_load_artifacts(tmp_path, name):
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, name, manifest())


@pytest.mark.parametrize(
    "changes",
    [
        {"started_at": -1.0},
        {"finished_at": -1.0},
        {"started_at": 1.0, "finished_at": 0.0},
        {"started_at": None},
        {"finished_at": None},
        {"outcome": "unfinished", "finished_at": 1.0},
        {"outcome": "generator_dropped", "started_at": 1.0},
    ],
)
def test_request_record_time_and_outcome_consistency(changes):
    row = record(0, sent=False).model_dump(mode="json")
    with pytest.raises(ValidationError):
        RequestRecord.model_validate_json(json.dumps({**row, **changes}))


def test_filename_cannot_mislabel_valid_manifest_as_result(tmp_path):
    with pytest.raises(EnvironmentError, match="report_failed"):
        publish(tmp_path, "load-result.json", manifest())
    assert not (tmp_path / "load-result.json").exists()
