"""E8.4 fixtures reuse the E8.3 production-path test assembly, with fake providers only."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from alembic import command
from sqlalchemy import func, select, update

from app.api.dependencies import actor_context
from app.auth.contracts import ActorContext
from app.db.models import LLMInvocation, RunJob, WorkspaceMembership
from app.llm.fake import FakeEmbeddingModel
from app.tools.adapters.mock_portal import MockPortalHTTPAdapter, MockPortalTransportError
from tests.evals import product_acceptance_runtime
from tests.evals.product_acceptance_runtime import AcceptanceFakeChat, ProductSession
from tests.evals.quality_dataset import project_model_payload
from tests.integration.db.test_migrations import _seed_gate6_migration_parent
from tests.integration.restore_support import (
    HEAD,
    OLD_HEAD,
    canonical,
    digest,
    require,
)
from tests.integration.support import alembic_config
from tests.legacy_runtime import SqlAlchemyWorkerJobStore

CASES = ("success", "approve", "reject", "unknown")
LEGACY_RESULT = {
    "schema_version": 1,
    "evidence_sufficient": False,
    "summary": [],
    "findings": [],
    "evidence": [],
    "sources": [],
    "limitations": [{"code": "insufficient_evidence", "detail": "E84 historical fixture."}],
    "application_draft": None,
}


def seed_legacy(database):
    database.upgrade(OLD_HEAD)
    engine = sa.create_engine(database.maintenance_url)
    try:
        with engine.begin() as connection:
            actor, workspace, run, _ = _seed_gate6_migration_parent(
                connection, subject="e84-legacy", graph_version="pathfinder-research-v1"
            )
            connection.execute(
                sa.text(
                    "UPDATE runs SET status='completed', finished_at=now(), "
                    "result_json=CAST(:result AS jsonb) WHERE id=:run"
                ),
                {"result": json.dumps(LEGACY_RESULT), "run": run},
            )
            before = connection.scalar(sa.text("SELECT to_jsonb(r)::text FROM runs r"))
        database.upgrade()
        with engine.connect() as connection:
            after = connection.scalar(
                sa.text(
                    "SELECT (to_jsonb(r) - 'client_request_id' - 'create_request_digest' "
                    "- 'create_request_version')::text FROM runs r"
                )
            )
            identity = connection.execute(
                sa.text(
                    "SELECT client_request_id, create_request_digest, create_request_version "
                    "FROM runs"
                )
            ).one()
        require(before == after and tuple(identity) == (None, None, None), "legacy_changed")
        return actor, workspace, run
    finally:
        engine.dispose()


def product(database, directory):
    directory.mkdir(mode=0o700)
    instance = ProductSession(
        database.url, directory, AcceptanceFakeChat(), FakeEmbeddingModel(), provider="fake"
    )
    for alias in CASES:
        instance.cases[alias] = instance.cases["mixed_alpha"]
    return instance


def payload(instance, alias):
    value = project_model_payload(instance.cases[alias]).model_dump(mode="json")
    value["resume_document_id"] = str(instance.document_id)
    return value


async def decide(instance, alias, choice):
    review = await instance.review(alias)
    require(review is not None, "missing_approval")
    endpoint = f"{instance.base}/action-intents/{review['action_intent_id']}/decision"
    body = {
        "decision": choice,
        "expected_version": review["approval_request"]["version"],
        "reason": "Synthetic E8.4 automated test decision; not human acceptance",
    }
    response = await instance.client.post(endpoint, json=body)
    repeated = await instance.client.post(endpoint, json=body)
    require(
        response.status_code == repeated.status_code == 200 and response.json() == repeated.json(),
        "decision_failed",
    )


class LostMockResponse(MockPortalHTTPAdapter):
    posts = 0
    gets = 0

    async def submit(self, **kwargs):
        type(self).posts += 1
        await super().submit(**kwargs)
        raise MockPortalTransportError

    async def get_by_idempotency_key(self, **kwargs):
        type(self).gets += 1
        raise MockPortalTransportError


async def seed_current(instance, monkeypatch, rehearsal):
    async with instance.open():
        await instance.ingest()
        for alias in CASES:
            await instance.create(alias)
            await instance.work(alias)
            facts = await instance.facts(alias)
            require(
                facts["run_status"] == "waiting_approval" and facts["checkpoint_count"] > 0,
                "fixture_failed",
            )
            if alias == "success":
                await decide(instance, alias, "approve")
                await instance.work(alias)
                require((await instance.facts(alias))["mock_count"] == 1, "fixture_failed")
            if alias == "unknown":
                await decide(instance, alias, "approve")
                LostMockResponse.posts = LostMockResponse.gets = 0
                with monkeypatch.context() as scoped:
                    scoped.setattr(
                        product_acceptance_runtime, "MockPortalHTTPAdapter", LostMockResponse
                    )
                    await instance.work(alias)
                    require(
                        (await instance.facts(alias))["action_statuses"] == ["executing"],
                        "fixture_failed",
                    )
                    while True:
                        rehearsal.remaining()
                        async with instance.sessions() as session:
                            due = await session.scalar(
                                select(RunJob.available_at).where(
                                    RunJob.run_id == instance.runs[alias]
                                )
                            )
                        if due <= datetime.now(UTC):
                            break
                        await asyncio.sleep(0.05)
                    await instance.work(alias)
                facts = await instance.facts(alias)
                require(
                    facts["run_status"] == "failed"
                    and facts["action_statuses"] == ["outcome_unknown"]
                    and facts["job_statuses"] == ["done"]
                    and facts["mock_count"] == 1
                    and LostMockResponse.posts == LostMockResponse.gets == 1,
                    "unknown_fixture_failed",
                )


def reject_downgrade(database):
    before = database.snapshot()
    rejected = False
    try:
        command.downgrade(alembic_config(database.maintenance_url), OLD_HEAD)
    except RuntimeError as error:
        rejected = str(error) == "E3 request identity data cannot be safely downgraded"
    require(rejected and database.snapshot() == before, "downgrade_not_safe")
    with database.connect() as connection:
        require(
            connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == HEAD,
            "schema_mismatch",
        )


async def check_history(instance, legacy):
    actor, workspace, run = legacy

    async def legacy_actor():
        return ActorContext(user_id=actor, subject="e84-legacy")

    instance.app.dependency_overrides[actor_context] = legacy_actor
    try:
        response = await instance.client.get(f"/api/v1/workspaces/{workspace}/runs/{run}")
        require(response.status_code == 200, "history_unreadable")
        result = response.json()
        require(
            result["graph_version"] == "pathfinder-research-v1"
            and result["result"] == LEGACY_RESULT,
            "history_changed",
        )
    finally:
        instance.app.dependency_overrides.pop(actor_context)


async def check_replay(instance, original):
    for alias in CASES:
        saved = json.loads((original.root / f"{alias}-created.json").read_text())
        response = await instance.client.post(
            f"{instance.base}/runs",
            json=payload(instance, alias),
            headers={"Idempotency-Key": saved["key"]},
        )
        require(
            response.status_code == 202
            and response.json() == saved["receipt"]
            and response.headers.get("Idempotency-Replayed") == "true",
            "replay_failed",
        )
        conflict = await instance.client.post(
            f"{instance.base}/runs",
            json={**payload(instance, alias), "query": "Different request"},
            headers={"Idempotency-Key": saved["key"]},
        )
        require(conflict.status_code == 409, "conflict_not_rejected")


async def check_authorization(instance, original, legacy):
    saved = json.loads((original.root / "success-created.json").read_text())

    async def other_actor():
        return ActorContext(user_id=legacy[0], subject="e84-legacy")

    async def replay():
        return await instance.client.post(
            f"{instance.base}/runs",
            json=payload(instance, "success"),
            headers={"Idempotency-Key": saved["key"]},
        )

    instance.app.dependency_overrides[actor_context] = other_actor
    try:
        require((await replay()).status_code == 404, "foreign_replay_allowed")
    finally:
        instance.app.dependency_overrides.pop(actor_context)
    async with instance.sessions.begin() as session:
        await session.execute(
            update(WorkspaceMembership)
            .where(
                WorkspaceMembership.workspace_id == instance.tenant.workspace_id,
                WorkspaceMembership.user_id == instance.tenant.actor_user_id,
            )
            .values(revoked_at=datetime.now(UTC))
        )

    # Avoid fake provisioning reactivating/rediscovering identity: fix only identity, not tenancy.
    async def revoked_actor():
        return ActorContext(user_id=instance.tenant.actor_user_id, subject="e84-revoked")

    instance.app.dependency_overrides[actor_context] = revoked_actor
    try:
        require((await replay()).status_code == 404, "revoked_replay_allowed")
    finally:
        instance.app.dependency_overrides.pop(actor_context)


async def verify_restored(instance, original, legacy, report, database):
    instance.document_id, instance.runs = original.document_id, dict(original.runs)
    async with instance.open():
        require((await instance.client.get("/readyz")).status_code == 200, "readiness_failed")
        await check_history(instance, legacy)
        report["checks"]["historical_output"] = True
        before = database.snapshot()
        await check_replay(instance, original)
        require(before == database.snapshot(), "replay_wrote_facts")
        report["checks"]["request_replay"] = True
        terminal_before = {alias: await instance.facts(alias) for alias in ("success", "unknown")}
        async with instance.sessions() as session:
            model_count = await session.scalar(select(func.count()).select_from(LLMInvocation))
        for alias, choice, expected in (("approve", "approve", 1), ("reject", "reject", 0)):
            initial = await instance.facts(alias)
            await decide(instance, alias, choice)
            await instance.work(alias)
            facts = await instance.facts(alias)
            require(
                facts["run_status"] == "completed"
                and facts["mock_count"] == expected
                and facts["job_statuses"] == ["done"]
                and facts["decisions"] == [choice]
                and facts["request_statuses"] == ["consumed" if expected else "rejected"]
                and facts["action_statuses"] == ["succeeded" if expected else "cancelled"]
                and facts["checkpoint_count"] > initial["checkpoint_count"],
                "resume_failed",
            )
        async with instance.sessions() as session:
            require(
                await session.scalar(select(func.count()).select_from(LLMInvocation))
                == model_count,
                "checkpoint_repeated_models",
            )
        report["checks"]["approval_checkpoint_resume"] = True
        jobs = SqlAlchemyWorkerJobStore(instance.sessions, lambda _: timedelta(0))
        claimed = await jobs.claim_due_job(
            worker_id="e84-idle-probe",
            now=datetime.now(UTC),
            lease_duration=timedelta(seconds=30),
        )
        require(claimed is None, "terminal_job_requeued")
        require(
            terminal_before == {alias: await instance.facts(alias) for alias in terminal_before},
            "terminal_changed",
        )
        report["checks"]["terminal_no_resubmit"] = True
        before_auth = database.snapshot()
        await check_authorization(instance, original, legacy)
        after_auth = database.snapshot()
        # The deliberate membership revocation is the only authorized mutation in this check.
        before_auth.pop("public.workspace_memberships")
        after_auth.pop("public.workspace_memberships")
        require(before_auth == after_auth, "denied_replay_wrote_facts")
        report["checks"]["replay_authorization"] = True
        report["business_digest"] = digest(canonical(after_auth))
