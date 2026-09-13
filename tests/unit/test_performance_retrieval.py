"""Fast E5.8 contracts and failure propagation; no capacity/database execution."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.performance import __main__ as cli
from tests.performance import retrieval_scale as runtime
from tests.performance.capacity_contracts import Health
from tests.performance.environment import EnvironmentError
from tests.performance.retrieval_contracts import (
    AUTHORIZATION,
    PlanNode,
    Profile,
    Result,
    load_profile,
    points,
)
from tests.performance.retrieval_data import (
    QUERIES,
    QueryEmbedding,
    allocations,
    prepared_document,
    vector,
)
from tests.performance.retrieval_metrics import (
    QueryObserver,
    allowed_indexes,
    sample_schedule,
    sanitize_plan,
    summaries,
)
from tests.performance.workload import publish


@pytest.mark.parametrize("point", points(), ids=lambda p: p.kind)
def test_fixed_allocation_keeps_counts_and_scope(point):
    rows = allocations(point)
    assert len(rows) == point.documents and sum(count for _, count in rows) == point.chunks
    assert rows[0] == (0, point.candidates)
    assert [r for r in rows[1:] if r[0] == 0] == [(0, 10), (0, 10)]
    assert {workspace for workspace, _ in rows} == set(range(point.workspaces))
    assert rows == allocations(point)
    assert len(prepared_document(0, point.candidates).chunks) == point.candidates


def test_seeded_float32_vectors_and_normalized_documents():
    assert vector("query/0") == vector("query/0") != vector("query/1")
    assert len(vector("query/0")) == 1536
    assert all(-1 <= value <= 1 for value in vector("query/0"))
    source = prepared_document(0, 10)
    assert source == prepared_document(0, 10)
    assert source.content_hash != prepared_document(1, 10).content_hash
    assert all(c.token_count == len(c.text.encode()) <= 800 for c in source.chunks)


@pytest.mark.parametrize(
    "updates",
    [
        {"seed": 59},
        {"repeats": 11},
        {"query_count": 6},
        {"name": "retrieval-e58-v1", "warmup": 0},
        {"arbitrary_dsn": "CANARY"},
    ],
)
def test_invalid_profiles_rejected(updates):
    payload = load_profile("retrieval-e58-v1").model_dump(mode="json")
    payload.update(updates)
    with pytest.raises(ValueError):
        Profile.model_validate_json(json.dumps(payload))


def test_samples_keep_failed_unfinished_and_first_read_separate():
    profile = load_profile("retrieval-e58-v1")
    rows = sample_schedule(profile)
    assert len(rows) == profile.max_attempts == 65
    rows[3] = rows[3].model_copy(update={"outcome": "failed", "total_seconds": 0.75})
    rows[4] = rows[4].model_copy(update={"outcome": "succeeded", "total_seconds": 0.25})
    result = next(
        s
        for s in summaries(rows)
        if (s.query, s.phase, s.metric) == (0, "measurement", "total_seconds")
    )
    assert (result.succeeded, result.failed, result.not_run) == (1, 1, 8)
    assert result.latency.count == 10 and result.latency.observed == 2
    assert result.latency.p95.value == 0.75
    assert result.latency.excluded == {"not_run": 8}
    assert len(summaries(rows)) == 45


def raw_plan():
    return [
        {
            "Planning Time": 0.1,
            "Execution Time": 0.2,
            "Plan": {
                "Node Type": "Limit",
                "Actual Rows": 5,
                "Actual Loops": 1,
                "Output": ["RAW_BODY_CANARY"],
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Relation Name": "document_chunks",
                        "Index Name": "ix_document_chunks_workspace_id",
                        "Actual Rows": 5,
                        "Index Cond": "PARAMETER_CANARY",
                        "Filter": "VECTOR_CANARY",
                        "Shared Hit Blocks": 3,
                    }
                ],
            },
        }
    ]


def test_plan_projection_drops_expressions_and_parameters():
    assert "ix_document_chunks_workspace_id" in allowed_indexes()
    result = sanitize_plan(raw_plan(), query=0, sql_digest="a" * 64)
    assert result.root.children[0].values["Shared Hit Blocks"] == 3.0
    assert "CANARY" not in result.model_dump_json()
    assert PlanNode.model_validate_json(result.root.model_dump_json()) == result.root


@pytest.mark.parametrize(
    "field,value",
    [
        ("Index Name", "ix_DOCUMENT_BODY_CANARY"),
        ("Relation Name", "DOCUMENT_BODY_CANARY"),
        ("Node Type", "DOCUMENT_BODY_CANARY"),
        ("Actual Rows", -1),
        ("Actual Rows", float("nan")),
        ("Actual Rows", True),
    ],
)
def test_plan_rejects_untrusted_names_and_invalid_numbers(field, value):
    raw = raw_plan()
    raw[0]["Plan"]["Plans"][0][field] = value
    with pytest.raises(ValueError):
        sanitize_plan(raw, query=0, sql_digest="a" * 64)


def test_plan_recursion_is_bounded():
    raw = raw_plan()
    node = raw[0]["Plan"]
    for _ in range(18):
        node["Plans"] = [{"Node Type": "Limit"}]
        node = node["Plans"][0]
    with pytest.raises(ValueError, match="invalid_plan"):
        sanitize_plan(raw, query=0, sql_digest="a" * 64)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("tmpfs", 224 * 1024**2, "resource_limit"),
        ("memory", 896 * 1024**2, "resource_limit"),
        ("rss", 2 * 1024**3, "resource_limit"),
        ("connections", 33, "resource_limit"),
        ("available", 2 * 1024**3 - 1, "resource_limit"),
        ("disk_free", 2 * 1024**3 - 1, "resource_limit"),
        ("memory", None, "guard_failed"),
    ],
)
def test_fixed_resource_protection(field, value, reason):
    values = dict(
        at=0.0,
        memory=1,
        tmpfs=1,
        rss=1,
        connections=1,
        available=4 * 1024**3,
        disk_free=4 * 1024**3,
        queue=0,
    )
    values[field] = value
    assert Health(**values).stop(load_profile("retrieval-e58-v1")) == reason


async def test_embedding_call_cap_and_cancellation_propagation():
    adapter = QueryEmbedding(1)
    assert len((await adapter.embed((QUERIES[0],), {}, attempt=object())).vectors[0]) == 1536
    with pytest.raises(ValueError):
        await adapter.embed((QUERIES[0],), {}, attempt=object())
    observer = QueryObserver()

    class CancelledRepository:
        async def search(self, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await observer.search(CancelledRepository())
    assert not observer.active and observer.repository_seconds is not None
    assert observer.sql_seconds is None


def test_publication_is_create_only_and_rejects_false_pass(tmp_path):
    result = Result(point=points()[0], source_sha="a" * 40, status="NOT_RUN", stop="previous_stop")
    publish(tmp_path, "retrieval-result.json", result)
    with pytest.raises(EnvironmentError):
        publish(tmp_path, "retrieval-result.json", result)
    with pytest.raises(EnvironmentError):
        publish(tmp_path, "retrieval-result-other.json", result)
    invalid = result.model_copy(update={"status": "PASS", "completeness": True})
    with pytest.raises(EnvironmentError):
        publish(tmp_path, "retrieval-result.json", invalid)


def test_authorization_mismatch_rejected_before_environment(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("environment_must_not_start")

    monkeypatch.setattr(runtime, "IsolatedEnvironment", forbidden)
    with pytest.raises(runtime.StopMeasurement):
        runtime.run_point(
            tmp_path / "point",
            point=points()[0],
            profile=load_profile("retrieval-e58-v1"),
            authorization="ci_instant",
        )


def test_suite_preserves_unrun_points_after_resource_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "source_sha", lambda: "a" * 40)
    calls = []

    def stopped(directory, *, point, **kwargs):
        calls.append(point.kind)
        return Result(
            point=point, source_sha="a" * 40, stop="resource_limit", resources_released=True
        )

    monkeypatch.setattr(runtime, "run_point", stopped)
    result = runtime.run_suite(
        tmp_path / "suite", authorization=AUTHORIZATION, selected_profile="retrieval-e58-v1"
    )
    assert calls == ["baseline"]
    assert [r.status for r in result.results] == ["IN_PROGRESS"] + ["NOT_RUN"] * 4
    assert len(json.loads((tmp_path / "suite/retrieval-suite.json").read_bytes())["results"]) == 5


@pytest.mark.parametrize("failure", [False, True])
def test_cli_propagates_incomplete_or_exception(failure, monkeypatch):
    def execute(*args, **kwargs):
        if failure:
            raise ValueError("EXCEPTION_BODY_CANARY")
        return SimpleNamespace(results=[SimpleNamespace(status="IN_PROGRESS")])

    monkeypatch.setattr(runtime, "run_suite", execute)
    assert (
        cli.main(
            [
                "retrieval-scale",
                "--profile",
                "retrieval-e58-v1",
                "--authorization",
                AUTHORIZATION,
                "--output",
                "/tmp/unused/output",
            ]
        )
        == 1
    )


def test_work_deadline_prevents_guard_io():
    experiment = object.__new__(runtime.Experiment)
    experiment.deadline = 0
    with pytest.raises(runtime.StopMeasurement, match="deadline"):
        asyncio.run(experiment.guard())


def test_cancelled_suite_retains_all_planned_points(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "source_sha", lambda: "a" * 40)

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "run_point", cancelled)
    with pytest.raises(KeyboardInterrupt):
        runtime.run_suite(
            tmp_path / "suite", authorization=AUTHORIZATION, selected_profile="retrieval-e58-v1"
        )
    evidence = json.loads((tmp_path / "suite/retrieval-suite.json").read_bytes())
    assert evidence["results"][0]["stop"] == "cancelled"
    assert [r["status"] for r in evidence["results"]][1:] == ["NOT_RUN"] * 4


async def test_timeout_keeps_failed_and_not_run_samples(tmp_path):
    manifest = SimpleNamespace(profile=load_profile("retrieval-instant-ci-v1"))
    experiment = runtime.Experiment(SimpleNamespace(output_dir=tmp_path), manifest, 1000)
    experiment.dataset = SimpleNamespace(tenants=(None,), documents=(None,))

    async def guard():
        return None

    class TimedOut:
        async def retrieve(self, **kwargs):
            raise TimeoutError("TIMEOUT_BODY_CANARY")

    experiment.guard = guard
    with pytest.raises(TimeoutError):
        await experiment.measure(TimedOut())
    assert [s.outcome for s in experiment.samples] == ["timeout"] + ["not_run"] * 5
    assert experiment.samples[0].total_seconds is not None
    assert experiment.samples[0].sql_seconds is None
    assert "CANARY" not in experiment.samples[0].model_dump_json()


@pytest.mark.parametrize("released", [False, True])
def test_cleanup_failure_does_not_hide_original_failure(tmp_path, monkeypatch, released):
    class Broken:
        output_created = True

        def __init__(self, profile, directory, **kwargs):
            self.directory = directory

        def start(self):
            self.directory.mkdir(mode=0o700)
            raise EnvironmentError("startup_failed")

        def close(self):
            return {
                "resources_released": released,
                "category": "startup_failed" if released else "cleanup_failed",
            }

    monkeypatch.setattr(runtime, "source_sha", lambda: "a" * 40)
    monkeypatch.setattr(runtime, "IsolatedEnvironment", Broken)
    result = runtime.run_point(
        tmp_path / "point",
        point=points(instant=True)[0],
        profile=load_profile("retrieval-instant-ci-v1"),
        authorization="ci_instant",
    )
    assert result.status == "IN_PROGRESS" and result.stop == "environment_failed"
    assert result.diagnostics == (() if released else ("cleanup_failed",))
    assert result.resources_released == released
