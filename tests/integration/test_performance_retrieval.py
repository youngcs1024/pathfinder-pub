"""Tiny E5.8 real repository/EXPLAIN smoke; never the manual scale profile."""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import update

from app.db.documents import SqlAlchemyDocumentRepository
from app.db.models import WorkspaceMembership
from app.retrieval.documents import EMBEDDING_PROFILE, DocumentRetrievalAuthorizationError
from tests.performance import retrieval_scale as runtime
from tests.performance.retrieval_contracts import load_profile, points
from tests.performance.retrieval_data import vector

pytestmark = pytest.mark.integration


def checked(result):
    if result.status != "PASS":
        pytest.fail(f"e58_{result.stop}_{'_'.join(result.diagnostics)}", pytrace=False)
    assert result.completeness and result.correctness and result.resources_released
    assert result.pool_remaining == 0
    assert result.attempts == result.adapter_calls == len(result.samples) == 6
    assert len(result.plans) == 2
    assert result.layout.candidates == 5
    assert sum(result.layout.document_chunk_counts) == 30
    assert {r.name: r.rows for r in result.final_layout.relations}["llm_invocations"] == 6


def test_instant_real_query_scope_plan_accounting_and_cleanup(tmp_path, monkeypatch):
    original = runtime.create_dataset
    checks = []

    async def verified(sessions, point, check, on_created):
        dataset = await original(sessions, point, check, on_created)
        tenant = dataset.tenants[0]
        repository = SqlAlchemyDocumentRepository(sessions)

        async def search(ids):
            return await repository.search(
                tenant=tenant,
                allowed_document_ids=ids,
                embedding_model=EMBEDDING_PROFILE,
                query_embedding=vector("query/0"),
                limit=5,
            )

        hits = await search((dataset.documents[0],))
        assert len(hits) == 5 and {h.document_id for h in hits} == {dataset.documents[0]}
        assert await search((dataset.documents[2],)) == ()  # Same tenant, wrong profile.
        assert await search((dataset.documents[3],)) == ()  # Different tenant.
        mixed = await search((dataset.documents[0], dataset.documents[2], dataset.documents[3]))
        assert {h.document_id for h in mixed} == {dataset.documents[0]}
        async with sessions() as session, session.begin():
            await session.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=datetime.now(UTC))
            )
        with pytest.raises(DocumentRetrievalAuthorizationError):
            await search((dataset.documents[0],))
        async with sessions() as session, session.begin():
            await session.execute(
                update(WorkspaceMembership)
                .where(
                    WorkspaceMembership.workspace_id == tenant.workspace_id,
                    WorkspaceMembership.user_id == tenant.actor_user_id,
                )
                .values(revoked_at=None)
            )
        checks.append(True)
        return dataset

    monkeypatch.setattr(runtime, "create_dataset", verified)
    canary = "E58_EXTERNAL_SECRET_CANARY"
    for name in ("PF_QWEN_API_KEY", "PF_TAVILY_API_KEY", "PF_DATABASE_URL", "HTTP_PROXY"):
        monkeypatch.setenv(name, canary)
    result = runtime.run_point(
        tmp_path / "point",
        point=points(instant=True)[0],
        profile=load_profile("retrieval-instant-ci-v1"),
        authorization="ci_instant",
    )
    checked(result)
    assert checks == [True]
    for path in (tmp_path / "point").iterdir():
        raw = path.read_text()
        if any(
            marker in raw
            for marker in (canary, "E58_SYNTHETIC_BODY_CANARY", "E58_SYNTHETIC_QUERY_CANARY")
        ):
            pytest.fail("e58_artifact_leak", pytrace=False)
        json.loads(raw)


def test_instant_collector_failure_retains_partial_and_releases_resources(tmp_path, monkeypatch):
    async def failed(*args, **kwargs):
        raise ValueError("E58_PLAN_ERROR_CANARY")

    monkeypatch.setattr(runtime, "explain", failed)
    result = runtime.run_point(
        tmp_path / "point",
        point=points(instant=True)[0],
        profile=load_profile("retrieval-instant-ci-v1"),
        authorization="ci_instant",
    )
    assert result.status == "IN_PROGRESS" and result.stop == "correctness_failed"
    assert not result.completeness and result.resources_released and result.pool_remaining == 0
    assert len(result.samples) == 6 and all(s.outcome == "succeeded" for s in result.samples)
    assert result.plans == ()
    assert "E58_PLAN_ERROR_CANARY" not in result.model_dump_json()
