"""D fault barriers around the real single Worker and atomic publication path."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import UUID

import psycopg
from pydantic import SecretStr
from sqlalchemy import func, select, text, update

from app.db.models import (
    LLMInvocation,
    ResumeFeedback,
    ResumeSession,
    ResumeVersion,
    WorkspaceMembership,
)
from app.db.session import create_database_engine, create_session_factory
from app.domain.errors import DomainNotFoundError, DomainUnavailableError
from app.domain.jobs import ClaimedJob
from app.llm.invocations import LLMInvocationContext
from app.llm.ports import ChatMessage, ChatModelResult, ModelUsage
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.resume_quality_runtime import factory, worker
from tests.evals.resume_recovery_contracts import WINDOW_SECONDS
from tests.evals.resume_recovery_fixture import replay, restore, script, snapshot


class InjectedExit(BaseException):
    """Crash simulation bypasses ordinary executor error handling."""


class Clock:
    def __init__(self):
        self.offset = timedelta()

    def __call__(self):
        return datetime.now(UTC) + self.offset


def saved_job(value):
    fields = dict(value)
    for key in ("job_id", "workspace_id", "run_id", "originating_actor_user_id", "owner_token"):
        fields[key] = UUID(fields[key])
    fields["lease_expires_at"] = datetime.fromisoformat(fields["lease_expires_at"])
    if fields["resume_approval_request_id"]:
        fields["resume_approval_request_id"] = UUID(fields["resume_approval_request_id"])
    return ClaimedJob(**fields)


async def ledger(sessions, state):
    async with sessions() as db:
        rows = list(
            await db.scalars(
                select(LLMInvocation)
                .where(
                    LLMInvocation.workspace_id == UUID(state["workspace_id"]),
                    LLMInvocation.run_id == UUID(state["run_id"]),
                )
                .order_by(LLMInvocation.created_at, LLMInvocation.id)
            )
        )
    require(
        all(
            r.provider == "fake" and r.status == "succeeded" and r.token_usage is not None
            for r in rows
        ),
        "fake_accounting_incomplete",
    )
    return [
        dict(
            id=str(r.id),
            request_hash=r.request_hash,
            status=r.status,
            token_usage=r.token_usage,
            provider=r.provider,
            graph_node=r.graph_node,
        )
        for r in rows
    ]


async def apply_block(rig, state):
    fault = state["case"]["fault"]
    if fault == "cancelled":
        await rig.store.cancel(rig.tenant, UUID(state["session_id"]))
    elif fault == "revoked":
        async with rig.sessions.begin() as db:
            await db.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == rig.tenant.workspace_id,
                    WorkspaceMembership.user_id == rig.tenant.actor_user_id,
                )
                .values(revoked_at=func.now())
            )
    elif fault == "budget":
        limit = (
            state["feedback_request"]["max_model_calls"]
            if state["case"]["mode"] == "revision"
            else state["request"]["budget"]["max_model_calls"]
        )
        current = await ledger(rig.sessions, state)
        response = ChatModelResult(
            content="budget fixture", usage=ModelUsage(input_tokens=1, output_tokens=1)
        )
        model = factory(rig, [response] * limit).create_chat_model(
            LLMInvocationContext(
                rig.tenant.workspace_id, rig.tenant.actor_user_id, run_id=UUID(state["run_id"])
            )
        )
        for _ in range(limit - len(current)):
            await model.invoke(
                (ChatMessage(role="user", content="synthetic budget fixture"),),
                (),
                {"task": "d_budget_fixture"},
            )


async def inject(url, sessions, root, state):
    rig = restore(sessions, state)
    runner = worker(rig, factory(rig, script(state)))
    jobs = runner._store
    fault, kind = state["case"]["fault"], state["case"]["kind"]
    blocking = kind == "blocking"
    original_claim, original_complete = jobs.claim_due_job, jobs.complete
    publisher = (
        jobs._revision_publisher
        if state["case"]["mode"] == "revision"
        else jobs._generation_publisher
    )
    original_publish = publisher.publish

    async def barrier(extra=None):
        before = await ledger(sessions, state)
        publish(
            root / "injected.json",
            {
                "fault": fault,
                "kind": kind,
                "pid": os.getpid(),
                "monotonic": monotonic(),
                "utc": datetime.now(UTC).isoformat(),
                "ledger": before,
                **(extra or {}),
            },
        )
        if kind == "real" and fault != "connection":
            os.kill(os.getpid(), signal.SIGKILL)
        if not blocking:
            raise InjectedExit()

    async def claim(**kwargs):
        job = await original_claim(**kwargs)
        require(
            job is not None
            and str(job.run_id) == state["run_id"]
            and str(job.workspace_id) == state["workspace_id"],
            "unexpected_claim",
        )
        publish(root / "claim.json", json.loads(json.dumps(asdict(job), default=str)))
        if fault == "after_claim":
            await barrier()
        if blocking and (fault == "budget" or state["case"]["ordinal"] % 2 == 0):
            await apply_block(rig, state)
            await barrier()
        return job

    async def complete(**kwargs):
        if fault == "before_publish":
            await barrier()
        if blocking and not (root / "injected.json").exists():
            await apply_block(rig, state)
            await barrier()
        result = await original_complete(**kwargs)
        if fault == "after_commit":
            require(result, "commit_not_completed")
            await barrier()
        return result

    async def publication(db, *args):
        if fault == "connection":
            if kind == "real":
                pid = await db.scalar(text("SELECT pg_backend_pid()"))
                # Exact backend of this transaction; no server restart or broad termination.
                with psycopg.connect(
                    url.replace("postgresql+psycopg:", "postgresql:"), connect_timeout=5
                ) as admin:
                    terminated = admin.execute(
                        "SELECT pg_terminate_backend(%s)", (pid,)
                    ).fetchone()[0]
                require(terminated, "connection_not_terminated")
                publish(
                    root / "injected.json",
                    {
                        "fault": fault,
                        "kind": kind,
                        "pid": os.getpid(),
                        "backend_pid": pid,
                        "monotonic": monotonic(),
                        "utc": datetime.now(UTC).isoformat(),
                        "ledger": await ledger(sessions, state),
                    },
                )
                await db.execute(text("SELECT 1"))
                require(False, "terminated_connection_survived")
            await barrier()
        result = await original_publish(db, *args)
        if fault == "in_transaction":
            await db.flush()
            await barrier()
        return result

    jobs.claim_due_job, jobs.complete, publisher.publish = claim, complete, publication
    try:
        await runner.run_once(asyncio.Event())
    except InjectedExit:
        pass
    except DomainUnavailableError:
        require(
            fault == "connection" and (root / "injected.json").exists(),
            "unexpected_database_failure",
        )
    require((root / "injected.json").exists(), "fault_not_injected")


async def real_inject(url, root):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.evals.resume_recovery_faults",
        str(root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(process.communicate(url.encode()), 60)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    state = read_private_json(root / "state.json")
    expected = 0 if state["case"]["fault"] == "connection" else -signal.SIGKILL
    require(
        process.returncode == expected and (root / "injected.json").exists(),
        "real_fault_process_failed",
    )
    publish(
        root / "process-exit.json",
        {"pid": process.pid, "returncode": process.returncode, "old_process_exited": True},
    )


async def check_result(rig, state, before, after):
    blocking = state["case"]["kind"] == "blocking"
    increment = 0 if blocking else 1
    baseline = state["baseline"]
    for key in ("versions", "artifacts"):
        require(
            len(after[key]) == len(baseline[key]) + increment, "duplicate_or_missing_publication"
        )
        require(all(after[key].get(k) == v for k, v in baseline[key].items()), "history_changed")
    require(after["completed_events"] == increment, "incorrect_completion_events")
    if blocking:
        expected = "failed" if state["case"]["fault"] == "budget" else "cancelled"
        require(after["run_status"] == expected, "incorrect_block_status")
        require(after["feedback"] == baseline["feedback"], "blocked_feedback_changed")
        if state["case"]["fault"] == "budget":
            require(
                after["error_category"]
                in ("generation_budget_exhausted", "invalid_revision_input"),
                "incorrect_budget_failure",
            )
            require(len(before) == len(await ledger(rig.sessions, state)), "blocked_model_called")
    else:
        require(
            after["run_status"] == "completed" and after["job_status"] == "done",
            "incorrect_terminal_state",
        )
        if state["case"]["mode"] == "revision":
            require(
                after["feedback"][state["feedback_id"]] in after["versions"], "feedback_not_applied"
            )
    if not blocking:
        new_ids = set(after["versions"]) - set(baseline["versions"])
        require(len(new_ids) == 1, "new_version_identity_invalid")
        async with rig.sessions() as db:
            version = await db.get(ResumeVersion, UUID(next(iter(new_ids))))
            bullets = [b["text"] for p in version.content_json["projects"] for b in p["bullets"]]
            expected_text = (
                state["patch"]["text"]
                if state["case"]["mode"] == "revision"
                else state["fact_text"]
            )
            require(expected_text in bullets, "published_content_incorrect")
    if state["case"]["fault"] == "revoked":
        try:
            await replay(rig, state)
        except DomainNotFoundError:
            pass
        else:
            require(False, "revoked_replay_accepted")
    else:
        await replay(rig, state)
    async with rig.sessions() as db:
        obj = await db.get(
            ResumeFeedback if state["case"]["mode"] == "revision" else ResumeSession,
            UUID(state.get("feedback_id", state["session_id"])),
        )
        require(obj.repair_count == int(state["case"]["ordinal"] % 3 == 0), "repair_budget_reset")
    foreign = type(rig.tenant)(UUID(int=1), rig.tenant.actor_user_id, rig.tenant.role)
    try:
        await rig.store.get_session(foreign, UUID(state["session_id"]))
    except DomainNotFoundError:
        pass
    else:
        require(False, "cross_tenant_read_accepted")


async def run_case(url, sessions, root, state):
    case = state["case"]
    rig = restore(sessions, state)
    if not (root / "injected.json").exists():
        require(not (root / "claim.json").exists(), "interrupted_injection_requires_review")
        if case["kind"] == "real":
            await real_inject(url, root)
        else:
            await inject(url, sessions, root, state)
    marker = read_private_json(root / "injected.json")
    if case["kind"] == "real":
        require((root / "process-exit.json").exists(), "process_exit_receipt_missing")
    before = await ledger(sessions, state)
    intermediate = await snapshot(sessions, state)
    # Only check the first observation; a controller may resume after recovery committed.
    observed = root / "rollback-verified.json"
    if not observed.exists():
        if case["kind"] != "blocking" and case["fault"] != "after_commit":
            require(
                intermediate["versions"] == state["baseline"]["versions"]
                and intermediate["artifacts"] == state["baseline"]["artifacts"]
                and intermediate["completed_events"] == 0,
                "uncommitted_publication_survived",
            )
        publish(observed, intermediate)
    clock = Clock()
    if case["kind"] != "real":
        clock.offset = timedelta(seconds=65)
    runner = worker(rig, factory(rig, script(state)))
    runner._clock = clock
    old = saved_job(read_private_json(root / "claim.json"))
    original_prepare = runner._store.prepare_claimed_job

    async def prepare_reclaimed(**kwargs):
        prepared = await original_prepare(**kwargs)
        require(kwargs["job"].owner_token != old.owner_token, "owner_token_reused")
        require(
            not await runner._store.complete(job=old, result=None, now=clock()),
            "old_owner_committed_during_new_lease",
        )
        if not (root / "old-owner-rejected.json").exists():
            publish(
                root / "old-owner-rejected.json",
                {"old_owner": str(old.owner_token), "new_owner": str(kwargs["job"].owner_token)},
            )
        return prepared

    runner._store.prepare_claimed_job = prepare_reclaimed
    start = marker["monotonic"] if case["kind"] == "real" else monotonic()
    terminal = {"completed", "failed", "cancelled"}
    while intermediate["run_status"] not in terminal and monotonic() - start <= WINDOW_SECONDS:
        await runner.run_once(asyncio.Event())
        intermediate = await snapshot(sessions, state)
        if intermediate["run_status"] not in terminal:
            if case["kind"] == "real":
                await asyncio.sleep(0.5)
            else:
                clock.offset += timedelta(seconds=1)
                await asyncio.sleep(0)
                if (
                    clock() - datetime.fromisoformat(marker["utc"])
                ).total_seconds() > WINDOW_SECONDS:
                    break
    elapsed = monotonic() - start
    logical_elapsed = (clock() - datetime.fromisoformat(marker["utc"])).total_seconds()
    terminal_receipt = root / "terminal-observation.json"
    if terminal_receipt.exists():
        saved = read_private_json(terminal_receipt)
        require(saved["snapshot"] == intermediate, "terminal_observation_changed")
        elapsed = saved["recovery_seconds"]
        logical_elapsed = saved.get("logical_seconds", elapsed)
    timed_out = elapsed > WINDOW_SECONDS or (
        case["kind"] != "real" and logical_elapsed > WINDOW_SECONDS
    )
    if timed_out:
        return dict(
            case=case,
            status="FAIL",
            verified=False,
            injected=True,
            timed_out=True,
            recovery_seconds=elapsed if case["kind"] == "real" else None,
            clock="monotonic" if case["kind"] == "real" else "injected",
            error="recovery_timeout",
            observation=intermediate,
        )
    observation_path = root / "terminal-observation.json"
    if observation_path.exists():
        observation = read_private_json(observation_path)
        require(observation["snapshot"] == intermediate, "terminal_observation_changed")
        elapsed = observation["recovery_seconds"]
    else:
        publish(
            observation_path,
            {
                "snapshot": intermediate,
                "recovery_seconds": elapsed,
                "logical_seconds": logical_elapsed,
            },
        )
    await check_result(rig, state, before, intermediate)
    if case["kind"] != "blocking" and case["fault"] != "after_commit":
        require((root / "old-owner-rejected.json").exists(), "old_owner_not_checked_during_reclaim")
    require(
        not await runner._store.heartbeat(
            job=old, now=clock(), lease_duration=timedelta(seconds=30)
        ),
        "old_owner_heartbeat_accepted",
    )
    require(
        not await runner._store.complete(job=old, result=None, now=clock()),
        "old_owner_commit_accepted",
    )
    after = await ledger(sessions, state)
    require(after[: len(marker["ledger"])] == marker["ledger"], "attempt_history_changed")
    if case["fault"] == "after_commit":
        require(after == marker["ledger"], "committed_result_reexecuted")
    final = await snapshot(sessions, state)
    require(final == intermediate, "replay_changed_result")
    result = dict(
        case=case,
        status="PASS",
        verified=True,
        injected=True,
        timed_out=False,
        recovery_seconds=elapsed if case["kind"] == "real" else None,
        clock="monotonic" if case["kind"] == "real" else "injected",
        observation=final,
        model_attempts=len(after),
        replayed_model_attempts=max(0, len(after) - len(marker["ledger"]))
        if marker["ledger"]
        else 0,
        ledger=after,
        duplicate_side_effects=0,
        old_owner_rejected=True,
    )
    return result


async def child(root, url):
    engine = create_database_engine(SecretStr(url))
    try:
        await inject(
            url, create_session_factory(engine), root, read_private_json(root / "state.json")
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(child(Path(sys.argv[1]), sys.stdin.read()))
    except Exception:
        sys.exit(2)
