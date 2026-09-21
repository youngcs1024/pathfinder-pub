"""Resource boundary/failure oracles, with no Docker or provider calls."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.evals.quality_experiment_binding import ExperimentError
from tests.evals.quality_experiment_resources import (
    LIMIT,
    ResourceObserver,
    ResourceReportV1,
    ResourceSampleV1,
    process_memory,
    resource_complete,
)


def sample(t, runner=100, database=200):
    return ResourceSampleV1(
        elapsed_seconds=float(t),
        runner_rss_bytes=runner,
        database_memory_bytes=database,
        combined_bytes=runner + database,
    )


def test_combined_peak_is_same_time_not_sum_of_independent_peaks():
    rows = (sample(0, 1000, 10), sample(1, 10, 1000))
    report = ResourceReportV1(
        arm="candidate",
        samples=rows,
        elapsed_seconds=1.1,
        complete=True,
        failure=None,
        peak_combined_bytes=1010,
    )
    assert report.peak_combined_bytes == 1010
    with pytest.raises(ValidationError):
        ResourceReportV1(
            arm="candidate",
            samples=rows,
            elapsed_seconds=1.1,
            complete=True,
            failure=None,
            peak_combined_bytes=2000,
        )


@pytest.mark.parametrize(
    "rows,elapsed",
    [
        ([], 0),
        ([sample(0)], 1),
        ([sample(0), sample(4)], 4),
        ([sample(0), sample(1)], 5),
        ([sample(0), sample(0)], 1),
        ([sample(0), sample(1, LIMIT + 1)], 1),
    ],
)
def test_missing_gaps_duplicates_and_limits_are_not_complete(rows, elapsed):
    assert not resource_complete(rows, elapsed, None)


def test_proc_tree_includes_descendants_and_same_supervisor_only(tmp_path):
    for pid, ppid, rss in [(10, 1, 3), (11, 10, 5), (12, 11, 7), (20, 1, 100), (1, 0, 2)]:
        path = tmp_path / str(pid)
        path.mkdir()
        fields = ["0"] * 22
        fields[0], fields[1], fields[19], fields[21] = "S", str(ppid), "50", str(rss)
        (path / "stat").write_text(f"{pid} (name with ) spaces) " + " ".join(fields))
    assert process_memory(10, 1, proc=tmp_path, page_size=4096) == (17 * 4096, 50)
    with pytest.raises(ExperimentError, match="resource_measurement_missing"):
        process_memory(99, proc=tmp_path)


async def test_sampler_writes_and_checks_every_observation(tmp_path):
    tmp_path.chmod(0o700)
    observer = ResourceObserver(None, tmp_path, "baseline", probe=lambda: (100, 200))
    await observer.start()
    report = await observer.finish()
    assert report.complete and report.peak_combined_bytes == 300
    assert (tmp_path / "sample-00000.json").exists() and (tmp_path / "report.json").exists()
    with pytest.raises(ExperimentError):
        await observer.start()


@pytest.mark.parametrize("probe", [lambda: (LIMIT + 1, 0), lambda: (1, LIMIT + 1), lambda: (0, 1)])
async def test_unsafe_first_sample_blocks_work(tmp_path, probe):
    tmp_path.chmod(0o700)
    observer = ResourceObserver(None, tmp_path, "candidate", probe=probe)
    with pytest.raises(ExperimentError):
        await observer.start()
    assert observer.failed.is_set()


async def test_sampler_failure_cancels_inflight_work_and_preserves_failure(tmp_path):
    tmp_path.chmod(0o700)
    observer = ResourceObserver(None, tmp_path, "candidate", probe=lambda: (1, 1))
    await observer.start()
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(observer.protect(work()))
    await asyncio.sleep(0)
    observer.failure = "resource_measurement_missing"
    observer.failed.set()
    with pytest.raises(ExperimentError):
        await task
    report = await observer.finish()
    assert cancelled.is_set() and not report.complete
    assert report.failure == "resource_measurement_missing"


def test_database_identity_and_missing_usage_cannot_be_zero():
    wrapped = SimpleNamespace(
        id="expected", stats=lambda **_: {"id": "wrong", "memory_stats": {"usage": 0}}
    )
    owner = SimpleNamespace(
        _verify_container=lambda: None,
        _container_id="expected",
        _container=SimpleNamespace(get_wrapped_container=lambda: wrapped),
    )
    observer = ResourceObserver(owner, None, "baseline")
    with pytest.raises(ExperimentError, match="identity_drift"):
        observer._probe()


@pytest.mark.parametrize(
    "mode", ["omitted", "matching", "foreign", "null", "changed_owner", "missing_usage"]
)
def test_one_shot_stats_identity_is_bound_to_owned_request(tmp_path, monkeypatch, mode):
    import tests.evals.quality_experiment_resources as module
    from tests.evals.quality_experiment_binding import ExperimentError

    identity = "a" * 64
    checks = []
    owner = SimpleNamespace(
        _container_id=identity, _verify_container=lambda: checks.append("inspect")
    )

    def stats(**kwargs):
        assert kwargs == {"stream": False, "one_shot": True}
        value = {"memory_stats": {"usage": 200}}
        if mode == "matching":
            value["id"] = identity
        if mode == "foreign":
            value["id"] = "b" * 64
        if mode == "null":
            value["id"] = None
        if mode == "changed_owner":
            owner._container_id = "c" * 64
        if mode == "missing_usage":
            value["memory_stats"] = {}
        return value

    container = SimpleNamespace(id=identity, stats=stats)
    owner._container = SimpleNamespace(get_wrapped_container=lambda: container)
    monkeypatch.setattr(module, "process_memory", lambda *a, **kw: (100, 50))
    observer = ResourceObserver(owner, tmp_path, "baseline")
    if mode in {"omitted", "matching"}:
        assert observer._probe() == (100, 200)
    else:
        with pytest.raises(
            ExperimentError,
            match="resource_measurement_missing" if mode == "missing_usage" else "identity_drift",
        ):
            observer._probe()
    assert checks == ["inspect", "inspect"]
