"""One explicitly authorized interrupted generation; preserve the unknown provider outcome."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.db.jobs import SqlAlchemyWorkerJobStore
from app.db.models import RunJob
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.runs import SqlAlchemyRunStore
from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiment_budget import CHAT_HEADROOM_CNY
from tests.evals.resume_experiments import preserve
from tests.evals.resume_live_budget import ledger_identity

AUTHORIZATION = "a-generation-interruption-authorization.json"
RECEIPT = "a-generation-interruption-receipt.json"


def validate_authorization(root, binding, rows):
    value = read_private_json(root / AUTHORIZATION)
    require(
        value["policy"] == "single_interrupted_generation_no_replay_v1"
        and value["binding"] == binding
        and value["authorization"]
        and value["reserved_cost_cny"] == str(CHAT_HEADROOM_CNY),
        "interruption_authorization_changed",
    )
    row = next((r for r in rows if str(r.id) == value["invocation_id"]), None)
    require(
        row is not None
        and ledger_identity(row) == value["ledger"]
        and row.run_id is not None
        and row.status == "started"
        and row.provider == "qwen"
        and row.invocation_kind == "chat"
        and row.graph_node == "analyze_job"
        and row.token_usage is None
        and row.estimated_cost is None
        and str(row.workspace_id) == binding["workspace_id"]
        and str(row.actor_user_id) == binding["actor_user_id"]
        and str(row.run_id) == value["run_id"],
        "interruption_ledger_changed",
    )
    relative = Path(value["started_path"])
    require(
        not relative.is_absolute()
        and len(relative.parts) == 5
        and relative.parts[0] == "a"
        and relative.parts[1] == value["source_sha"]
        and relative.parts[2] == "formal"
        and relative.parts[3] == value["sample_id"]
        and relative.name == "call-00-started.json"
        and ".." not in relative.parts,
        "interruption_path_changed",
    )
    path = root / relative
    require(not path.is_symlink(), "interruption_path_changed")
    started = read_private_json(path)
    require(
        quality_identity_digest(started) == value["started_digest"]
        and str(row.id) not in started["before"]["invocation_ids"]
        and not path.with_name("call-00-response.json").exists(),
        "interruption_journal_changed",
    )
    require(
        [str(r.id) for r in rows if r.run_id == row.run_id] == [str(row.id)],
        "interruption_call_ambiguous",
    )
    return value


def reservation(root, binding, rows, usage):
    receipt_path = root / RECEIPT
    if not receipt_path.exists():
        return usage
    value = validate_authorization(root, binding, rows)
    receipt = read_private_json(receipt_path)
    require(
        receipt["authorization_digest"] == quality_identity_digest(value)
        and receipt["provider_outcome"] == "UNKNOWN"
        and receipt["business_status"] == "cancelled"
        and receipt["replayed"] is False
        and receipt["reserved_cost_cny"] == str(CHAT_HEADROOM_CNY),
        "interruption_receipt_changed",
    )
    amount = CHAT_HEADROOM_CNY
    return {
        **usage,
        "generation_interruption": receipt,
        "reserved_unfinished_attempts": 1,
        "a_reserved_unknown_attempts": usage.get("a_reserved_unknown_attempts", 0) + 1,
        "reserved_unknown_attempts": usage["reserved_unknown_attempts"] + 1,
        "reserved_cost_cny": str(Decimal(usage["reserved_cost_cny"]) + amount),
        "budget_occupied_cny": str(Decimal(usage["budget_occupied_cny"]) + amount),
        "remaining_admission_cny": str(Decimal(usage["remaining_admission_cny"]) - amount),
    }


async def reconcile(recorder, sessions, tenant, origin, directory, inventory, usage_for):
    if not (recorder.root / AUTHORIZATION).exists():
        require(not (recorder.root / RECEIPT).exists(), "interruption_authorization_missing")
        return
    value = validate_authorization(recorder.root, recorder.binding, await recorder.audit())
    require(origin.name == value["source_sha"], "interruption_source_changed")
    relative = Path(value["started_path"]).relative_to(Path("a") / origin.name)
    require(str(relative) in inventory, "reuse_unaudited_file")
    sample_path = relative.parent
    generated = read_private_json(origin / sample_path / "generation-started.json")
    sample = generated["sample"]
    require(
        sample["sample_id"] == value["sample_id"] and sample["arm"] == "pathfinder",
        "interruption_sample_changed",
    )
    block = f"{sample['case_id']}-r{sample['repeat']}"
    business = read_private_json(origin / "formal" / f"{block}-session.json")
    require(business["run_id"] == value["run_id"], "interruption_business_changed")
    store = SqlAlchemyResumeGenerationStore(sessions)
    detail = await store.get_session(tenant, UUID(business["session_id"]))
    require(not detail.get("current_version_id"), "interruption_output_exists")
    async with sessions() as db:
        pending = list(
            await db.scalars(select(RunJob).where(RunJob.status.in_(("queued", "leased"))))
        )
    require(all(str(j.run_id) == value["run_id"] for j in pending), "unrelated_pending_work")
    if detail["run_status"] != "cancelled":
        require(
            len(pending) == 1 and pending[0].lease_expires_at <= datetime.now(UTC),
            "interruption_worker_may_be_active",
        )
        await SqlAlchemyRunStore(sessions).cancel_run(
            tenant=tenant, run_id=UUID(value["run_id"]), allow_other_creator=False
        )
        await SqlAlchemyWorkerJobStore(sessions, lambda attempt: timedelta(0)).reclaim_stale_leases(
            now=datetime.now(UTC), limit=1
        )
    detail = await store.get_session(tenant, UUID(business["session_id"]))
    require(
        detail["run_status"] == "cancelled" and not detail.get("current_version_id"),
        "interruption_business_not_terminal",
    )
    preserve(
        recorder.root / RECEIPT,
        {
            "authorization_digest": quality_identity_digest(value),
            "invocation_id": value["invocation_id"],
            "provider_outcome": "UNKNOWN",
            "business_status": "cancelled",
            "replayed": False,
            "reserved_cost_cny": str(CHAT_HEADROOM_CNY),
        },
    )
    for entry in inventory:
        if Path(entry).parent == sample_path or entry in (
            f"formal/{block}-session.json",
            f"formal/{block}-command.json",
            "formal/annotations.json",
        ):
            target = directory / entry
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            preserve(target, read_private_json(origin / entry))
    preserve(
        directory / sample_path / "generation.json",
        {
            "sample": sample,
            "generation_status": "FAILED",
            "output": None,
            "error": "controller_interrupted_provider_outcome_unknown",
            "usage": await usage_for(recorder, [value["invocation_id"]]),
            "elapsed_seconds": None,
            "common_input_digest": read_private_json(origin / sample_path / "input.json")["digest"],
            "interruption_receipt_digest": quality_identity_digest(
                read_private_json(recorder.root / RECEIPT)
            ),
        },
    )
