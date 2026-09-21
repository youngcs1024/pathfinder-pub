from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, update

from app.db.models import LLMInvocation, WorkspaceMembership
from app.llm.factory import LLMAccountingError
from app.llm.fake import FakeEmbeddingModel
from app.llm.invocations import LLMInvocationContext
from tests.evals.product_acceptance_budget import DatabaseBudgetRecorder
from tests.evals.product_acceptance_contracts import (
    AcceptanceError,
    Decision,
    read_private_json,
    review_digest,
)
from tests.evals.product_acceptance_runtime import AcceptanceFakeChat, ProductSession, run_cases

pytestmark = pytest.mark.integration


def product(database_url, tmp_path, **kwargs):
    root = tmp_path / "product"
    root.mkdir(mode=0o700)
    return ProductSession(
        database_url, root, AcceptanceFakeChat(), FakeEmbeddingModel(), provider="fake", **kwargs
    )


def decision(case, review, choice="approve"):
    return Decision(
        case_id=case,
        review_digest=review_digest(review),
        decision=choice,
        reviewed_exact_content=True,
        acknowledged_at=datetime.now(UTC),
    )


async def test_two_product_paths_replay_and_new_checkpoint_connections(
    migrated_database_url, tmp_path
):
    instance = product(migrated_database_url, tmp_path)

    async def reviewed(root, case, review):
        assert read_private_json(root / f"{case}-review.json") == review
        assert (await instance.facts(case))["mock_count"] == 0
        return decision(case, review, "approve" if case == "mixed_alpha" else "reject")

    async with instance.open():
        records = await run_cases(instance, decision_reader=reviewed)
        assert [r["status"] for r in records] == ["PASS", "PASS"]
        assert records[0]["facts"]["mock_count"] == 1
        assert records[1]["facts"]["mock_count"] == 0
        assert (await instance.recorder.usage())["pricing_applicable"] is False
    assert read_private_json(instance.root / "invocations.json")["usage"]["attempts"] > 0


@pytest.mark.parametrize("change", ["digest", "case", "revoked", "cancelled"])
async def test_stale_review_cannot_create_mock(migrated_database_url, tmp_path, change):
    instance = product(migrated_database_url, tmp_path)
    async with instance.open():
        await instance.ingest()
        await instance.create("mixed_alpha")
        await instance.work("mixed_alpha")
        review = await instance.review("mixed_alpha")
        command = decision("mixed_alpha", review)
        if change == "digest":
            command = command.model_copy(update={"review_digest": "sha256:" + "b" * 64})
        elif change == "case":
            command = command.model_copy(update={"case_id": "gap_alpha"})
        elif change == "revoked":
            async with instance.sessions.begin() as session:
                await session.execute(
                    update(WorkspaceMembership)
                    .where(
                        WorkspaceMembership.workspace_id == instance.tenant.workspace_id,
                        WorkspaceMembership.user_id == instance.tenant.actor_user_id,
                    )
                    .values(revoked_at=datetime.now(UTC))
                )
        else:
            response = await instance.client.post(
                f"{instance.base}/runs/{instance.runs['mixed_alpha']}/cancel"
            )
            assert response.status_code == 200
        with pytest.raises(AcceptanceError):
            await instance.decide("mixed_alpha", command)
        assert (await instance.facts("mixed_alpha"))["mock_count"] == 0


async def test_budget_reload_sees_preexisting_unknown_and_refuses_provider(
    migrated_database_url,
    tmp_path,
):
    instance = product(migrated_database_url, tmp_path)
    async with instance.open():
        await instance.ingest()
        async with instance.sessions.begin() as session:
            row = await session.scalar(select(LLMInvocation))
            # An interruption after durable prepare but before a known outcome.
            await session.execute(
                update(LLMInvocation)
                .where(LLMInvocation.id == row.id)
                .values(
                    status="started",
                    token_usage=None,
                    estimated_cost=None,
                    latency_ms=None,
                    provider_response_id=None,
                    error_category=None,
                    trace_ids=None,
                )
            )
        reloaded = DatabaseBudgetRecorder(instance.sessions, instance.tenant, provider="fake")
        assert (await reloaded.usage())["unfinished"] == 1
        factory = instance.factory()
        factory = replace(factory, recorder=reloaded)
        with pytest.raises(LLMAccountingError):
            await factory.create_embedding_model(
                LLMInvocationContext(
                    instance.tenant.workspace_id,
                    instance.tenant.actor_user_id,
                )
            ).embed(["Must not reach a provider"], {"graph_node": "ingest"})
        assert reloaded.stopped
        assert (await reloaded.usage())["attempts"] == 1


async def test_cancel_during_human_wait_keeps_partial_slot_and_ledger(
    migrated_database_url, tmp_path
):
    instance = product(migrated_database_url, tmp_path)

    async def cancelled(*args):
        raise asyncio.CancelledError

    async with instance.open():
        with pytest.raises(asyncio.CancelledError):
            await run_cases(instance, decision_reader=cancelled)
        assert (await instance.facts("mixed_alpha"))["mock_count"] == 0
    saved = read_private_json(instance.root / "cases.json")
    assert saved["cases"][0]["status"] == "IN_PROGRESS"
    assert saved["cases"][1]["status"] == "NOT_RUN"
    assert (instance.root / "invocations.json").exists()


async def test_source_drift_blocks_embedding_before_attempt(migrated_database_url, tmp_path):
    def drift():
        raise AcceptanceError("source_changed")

    instance = product(migrated_database_url, tmp_path, source_check=drift)
    async with instance.open():
        with pytest.raises(LLMAccountingError):
            await instance.ingest()
        assert (await instance.recorder.usage())["attempts"] == 0
