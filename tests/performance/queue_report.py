"""Separate completeness, correctness and finite-window queue observations."""

from __future__ import annotations

from tests.performance.capacity_contracts import safe_read
from tests.performance.contracts import digest
from tests.performance.metrics import (
    DecisionFact,
    RunFact,
    Value,
    distribution,
    run_timing,
)
from tests.performance.queue_contracts import SAFE_STOPS, TERMINAL, Baseline, Result
from tests.performance.queue_metrics import QueuePacket


def baseline_from(manifest, facts, timings):
    chosen = [f for f in facts if f.phase == "measurement"]
    by_run = {r.run_id: r for r in timings}
    if len(chosen) != manifest.profile.baseline_samples or any(
        f.status != "completed" or f.job_status != "done" or f.run_id not in by_run for f in chosen
    ):
        return None
    selected = [by_run[f.run_id] for f in chosen]
    if any(
        t.unfinished_segments
        or t.worker_completed_segments_seconds.value is None
        or t.worker_completed_segments_seconds.value <= 0
        for t in selected
    ):
        return None
    values = tuple(t.worker_completed_segments_seconds.value for t in selected)
    return Baseline(
        experiment_id=manifest.experiment_id,
        manifest_digest=digest(manifest),
        mode=manifest.point.mode,
        source_sha=manifest.source_sha,
        lock_digest=manifest.lock_digest,
        call_profile_digest=manifest.call_profile_digest,
        database_version=manifest.database_version,
        service_seconds=values,
        mean_seconds=sum(values) / len(values),
    )


def window_counts(requests, facts, start, end):
    if start is None or end is None or end <= start:
        return Value(reason="empty"), Value(reason="empty"), 0, 0
    duration = (end - start).total_seconds()
    accepted = sum(
        r.http_status == 202 and r.finished_at is not None and start <= r.finished_at < end
        for r in requests
    )
    finished = [f for f in facts if f.finished_at is not None and start <= f.finished_at < end]
    return (
        Value(value=accepted / duration),
        Value(value=sum(f.status == "completed" for f in finished) / duration),
        sum(f.status == "failed" for f in finished),
        sum(f.status == "cancelled" for f in finished),
    )


def summarize(experiment, reason, cleanup):
    p = experiment
    diagnostics = list(p.diagnostics)
    packets = []
    for role in ("api", "worker", "driver", "supervisor"):
        try:
            packet = safe_read(p.env.output_dir / f"metrics-{role}.json", QueuePacket)
            if (
                packet.role != role
                or packet.dropped
                or packet.write_failed
                or packet.pool_remaining
            ):
                raise ValueError("incomplete")
            packets.append(packet)
        except (OSError, ValueError):
            diagnostics.append("metrics_incomplete")
    snapshots = {s.stage: s for s in p.snapshots}
    complete = len(packets) == 4 and set(snapshots) == {
        "submission_stopped",
        "drain_cutoff",
        "worker_stopped",
    }
    if not complete:
        diagnostics.append("metrics_incomplete")
    if not cleanup.get("resources_released") or cleanup.get("category"):
        diagnostics.append("cleanup_failed")
    cut = snapshots.get("drain_cutoff")
    final = snapshots.get("worker_stopped")
    facts = final.runs if final else ()
    samples = [s for packet in packets for s in packet.samples]
    timings = []
    for f in cut.runs if cut else ():
        observed = []
        for s in samples:
            if s.started > cut.at:
                continue
            if s.finished is not None and s.finished > cut.at:
                s = s.model_copy(update={"finished": None, "outcome": "pending"})
            observed.append(s)
        decisions = [
            DecisionFact(
                run_id=x.run_id, request_id=x.approval_request_id, decided_at=x.decision_at
            )
            for x in cut.runs
            if x.decision_at is not None
        ]
        timings.append(
            run_timing(
                RunFact(
                    run_id=f.run_id,
                    created_at=f.created_at,
                    started_at=f.started_at,
                    finished_at=f.finished_at,
                    approval_mode="synthetic_driver" if f.mode == "application" else "none",
                ),
                observed,
                [],
                decisions,
            )
        )
    baseline = baseline_from(p.manifest, facts, timings) if p.point.kind == "baseline" else None
    if p.point.kind == "baseline" and baseline is None:
        diagnostics.append("baseline_incomplete")
    if p.point.kind == "control" and (not p.control_result or not p.control_result.converged):
        diagnostics.append("correctness_failed")
    if p.point.kind == "load" and p.load_result is None:
        diagnostics.append("metrics_incomplete")
    accepted, completed, failed, cancelled = window_counts(
        p.requests, facts, p.measurement_start_utc, p.actual_end_utc
    )
    final_by_id = {f.run_id: f for f in facts}
    cutoff = cut.runs if cut else ()
    counts = dict(
        new_runs=len(facts),
        http_accepted=sum(r.http_status == 202 for r in p.requests),
        http_unknown=sum(r.http_status is None for r in p.requests),
        http_failed=sum(r.http_status is not None and r.http_status != 202 for r in p.requests),
        replays=sum(r.replayed is True for r in p.requests),
        unfinished_at_cutoff=sum(f.status not in TERMINAL for f in cutoff),
        completed_at_cutoff=sum(f.status == "completed" for f in cutoff),
        failed_at_cutoff=sum(f.status == "failed" for f in cutoff),
        cancelled_at_cutoff=sum(f.status == "cancelled" for f in cutoff),
        changed_during_shutdown=sum(f != final_by_id.get(f.run_id) for f in cutoff),
        pool_remaining=sum(p.pool_remaining for p in packets),
        unfinished_calls=p.unfinished_calls,
    )
    good = complete and not diagnostics and reason in SAFE_STOPS and p.worker_stopped
    return Result(
        point=p.point,
        source_sha=p.manifest.source_sha,
        stop=reason,
        status="PASS" if good else "IN_PROGRESS",
        diagnostics=tuple(dict.fromkeys(diagnostics)),
        completeness=complete and "metrics_incomplete" not in diagnostics,
        correctness=True if good else None,
        resources_released=cleanup.get("resources_released") is True,
        experiment_id=p.manifest.experiment_id,
        manifest_digest=digest(p.manifest),
        baseline=baseline if good else None,
        measurement_start=p.measurement_start_utc,
        measurement_end=p.measurement_end_utc,
        actual_measurement_end=p.actual_end_utc,
        requests=tuple(p.requests),
        snapshots=tuple(p.snapshots),
        queue=tuple(p.queue),
        health=tuple(p.health),
        load=p.load_result,
        control=p.control_result,
        timings=tuple(timings),
        accepted_rate=accepted,
        completed_rate=completed,
        failed_in_window=failed,
        cancelled_in_window=cancelled,
        initial_queue=distribution([t.initial_queue_approx_seconds for t in timings]),
        counts=counts,
    )
