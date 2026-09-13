"""Short, read-only snapshots from the owned database; no checkpoint product reads."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from time import monotonic

from sqlalchemy import func, select, text

from app.db.models import (
    ActionIntent,
    ApprovalDecision,
    ApprovalRequest,
    LLMInvocation,
    MockSubmission,
    Run,
    RunEvent,
    RunJob,
    ToolInvocation,
)
from tests.performance.capacity_contracts import safe_read
from tests.performance.metrics import Value
from tests.performance.queue_contracts import Fact, QueueSample, Snapshot
from tests.performance.smoke import require
from tests.performance.workload import QueueCallRecord, publish


async def state_rows(sessions, tenant):
    async with asyncio.timeout(2), sessions() as session, session.begin():
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        rows = (
            (
                await session.execute(
                    select(
                        Run.id,
                        Run.mode,
                        Run.status,
                        RunJob.status.label("job_status"),
                        RunJob.available_at,
                        RunJob.resume_approval_request_id,
                    )
                    .join(
                        RunJob,
                        (RunJob.run_id == Run.id) & (RunJob.workspace_id == Run.workspace_id),
                    )
                    .where(Run.workspace_id == tenant.workspace_id)
                )
            )
            .mappings()
            .all()
        )
        now = await session.scalar(select(func.clock_timestamp()))
    return rows, now


def sample(rows, now, at):
    due = [
        row["available_at"]
        for row in rows
        if row["job_status"] == "queued" and row["available_at"] <= now
    ]
    return QueueSample(
        at=at,
        recorded_at=now,
        queued=sum(r["job_status"] == "queued" for r in rows),
        leased=sum(r["job_status"] == "leased" for r in rows),
        waiting_approval=sum(r["status"] == "waiting_approval" for r in rows),
        completed=sum(r["status"] == "completed" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows),
        cancelled=sum(r["status"] == "cancelled" for r in rows),
        oldest_due_seconds=Value(value=(now - min(due)).total_seconds())
        if due
        else Value(reason="empty"),
    )


async def snapshot(experiment, stage):
    start = monotonic()
    tenant = experiment.tenant
    by_key = {r.request_id: r for r in experiment.requests}
    async with asyncio.timeout(10), experiment.sessions() as session, session.begin():
        await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))

        async def rows(cls):
            return list(
                await session.scalars(select(cls).where(cls.workspace_id == tenant.workspace_id))
            )

        runs = await rows(Run)
        jobs = await rows(RunJob)
        actions = await rows(ActionIntent)
        approvals = await rows(ApprovalRequest)
        decisions = await rows(ApprovalDecision)
        invocations = await rows(LLMInvocation)
        effects = await rows(MockSubmission)
        events = (
            await session.execute(
                select(RunEvent.run_id, RunEvent.seq)
                .where(RunEvent.workspace_id == tenant.workspace_id)
                .order_by(RunEvent.seq)
            )
        ).all()
        all_ids = set(await session.scalars(select(Run.id)))
    require(all_ids == {r.id for r in runs})
    require(len(runs) == len(jobs) and len(runs) <= 128)
    require({r.client_request_id for r in runs} <= set(by_key))
    require(len({r.client_request_id for r in runs}) == len(runs))
    require({j.run_id for j in jobs} == all_ids)
    event_ids = defaultdict(list)
    for run_id, seq in events:
        event_ids[run_id].append(seq)
    result = []
    for run in runs:
        request = by_key[run.client_request_id]
        job = next(j for j in jobs if j.run_id == run.id)
        require(run.created_by_user_id == tenant.actor_user_id and run.mode == request.mode)
        require(request.run_id in {None, run.id})
        sequences = event_ids[run.id]
        require(
            sequences == list(range(1, len(sequences) + 1))
            and run.next_event_seq == len(sequences) + 1
        )
        run_actions = [a for a in actions if a.run_id == run.id]
        run_approvals = [a for a in approvals if a.run_id == run.id]
        run_effects = [e for e in effects if e.run_id == run.id]
        require(len(run_effects) <= 1)
        if run.mode == "research":
            require(not run_effects and not run_actions and not run_approvals)
        decision = None
        if run_approvals:
            require(len(run_approvals) == len(run_actions) == 1)
            approval, action = run_approvals[0], run_actions[0]
            for field in (
                "args_digest",
                "target_digest",
                "approval_binding_digest",
                "approval_binding_version",
            ):
                require(getattr(action, field) == getattr(approval, field))
            choices = [d for d in decisions if d.approval_request_id == approval.id]
            require(len(choices) <= 1)
            decision = choices[0] if choices else None
            if run_effects:
                require(
                    decision is not None
                    and decision.decision == "approve"
                    and approval.status == "consumed"
                )
                require(run_effects[0].idempotency_key == action.idempotency_key)
        if run.status == "completed":
            if stage == "worker_stopped":
                require(job.status == "done")
            if run.mode == "application":
                require(len(run_effects) == 1 and run_actions[0].status == "succeeded")
        result.append(
            Fact(
                run_id=run.id,
                request_id=run.client_request_id,
                mode=run.mode,
                phase=request.phase,
                status=run.status,
                job_status=job.status,
                created_at=run.created_at,
                started_at=run.started_at,
                finished_at=run.finished_at,
                available_at=job.available_at,
                decision_at=decision.decided_at if decision else None,
                approval_request_id=decision.approval_request_id if decision else None,
                event_count=len(sequences),
                model_attempts=sum(i.run_id == run.id for i in invocations),
                mock_effects=len(run_effects),
            )
        )
    for request in experiment.requests:
        if request.run_id is not None:
            require(request.run_id in all_ids)
    snap = Snapshot(
        at=monotonic(),
        recorded_at=datetime.now(UTC),
        stage=stage,
        runs=tuple(sorted(result, key=lambda r: str(r.run_id))),
    )
    experiment.snapshots.append(snap)
    publish(experiment.env.output_dir, f"queue-snapshot-{stage}.json", snap)
    experiment.metrics.observer_seconds += monotonic() - start
    return snap


def read_calls(directory):
    records = []
    unfinished = 0
    for process, limit in (("worker", 2047), ("ingest", 1)):
        gap = False
        for sequence in range(1, limit + 1):
            start = directory / f"calls-{process}-{sequence:03}-started.json"
            finish = directory / f"calls-{process}-{sequence:03}-finished.json"
            if not start.exists():
                require(not finish.exists())
                gap = True
                continue
            require(not gap)
            beginning = safe_read(start, QueueCallRecord, 4096)
            require(
                beginning.process == process
                and beginning.sequence == sequence
                and beginning.phase == "started"
            )
            if finish.exists():
                ending = safe_read(finish, QueueCallRecord, 4096)
                require(
                    ending.phase == "finished"
                    and ending.model_dump(exclude={"phase", "outcome", "elapsed_seconds"})
                    == beginning.model_dump(exclude={"phase", "outcome", "elapsed_seconds"})
                )
                records.append(ending)
            else:
                records.append(beginning)
                unfinished += 1
    return records, unfinished


async def reconcile_calls(experiment):
    records, unfinished = read_calls(experiment.env.output_dir)
    tenant = experiment.tenant
    async with asyncio.timeout(5), experiment.sessions() as session:
        models = list(
            await session.scalars(
                select(LLMInvocation).where(LLMInvocation.workspace_id == tenant.workspace_id)
            )
        )
        tools = list(
            await session.scalars(
                select(ToolInvocation).where(ToolInvocation.workspace_id == tenant.workspace_id)
            )
        )
    providers = [c for c in records if c.call in {"chat", "embedding"}]
    require(
        len(providers) == len(models)
        and {c.invocation_id for c in providers} == {r.id for r in models}
    )
    require(all(r.provider == "fake" for r in models))
    for kind, name in (("search", "search_web"), ("mock_submit", "submit_mock_application")):
        require(sum(c.call == kind for c in records) == sum(t.tool_name == name for t in tools))
    return unfinished
