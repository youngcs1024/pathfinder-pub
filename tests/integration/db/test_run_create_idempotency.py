"""E3.6 HTTP concurrency and post-commit transport loss, using real PostgreSQL."""

import asyncio
from uuid import UUID

import httpx
import pytest
from sqlalchemy import select

from app.db.models import Message, Run, RunEvent, RunJob
from app.domain.runs import RunMode, create_request_digest_v1
from tests.integration.db import test_run_api_store as support
from tests.integration.db import test_run_create_http as http
from tests.integration.db.test_run_api_store import runtime as runtime

pytestmark = pytest.mark.integration


async def _concurrent_posts(application, tenant, payloads):
    # Release before acquiring DB connections: a ten-party in-transaction barrier
    # would exhaust the production-sized pool. E3.3 forces unique-index contention.
    start = asyncio.Event()

    async def post(payload):
        await start.wait()
        return await http._request(
            application, tenant, headers={"Idempotency-Key": http.KEY}, payload=payload
        )

    tasks = [asyncio.create_task(post(payload)) for payload in payloads]
    try:
        async with asyncio.timeout(20):
            start.set()
            return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _assert_single_creation(runtime, receipt, query):
    assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)
    async with runtime.session_factory() as session:
        run = (await session.scalars(select(Run))).one()
        message = (await session.scalars(select(Message))).one()
        job = (await session.scalars(select(RunJob))).one()
        event = (await session.scalars(select(RunEvent))).one()
        assert run.id == UUID(receipt["run_id"])
        assert run.request_message_id == message.id
        assert run.conversation_id == message.conversation_id
        assert run.input_json["query"] == message.content == query
        assert run.create_request_digest == create_request_digest_v1(
            mode=RunMode.RESEARCH, query=query
        )
        assert run.client_request_id == UUID(http.KEY)
        assert run.create_request_version == 1
        assert run.status == job.status == "queued"
        assert job.run_id == event.run_id == run.id
        assert event.type == "run.created"
        assert event.seq == 1
        assert run.next_event_seq == 2


async def test_ten_same_key_http_requests_converge(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e36-ten-same")
    payload = {"mode": "research", "query": "E36 synthetic same query"}
    responses = await _concurrent_posts(http._app(runtime, tenant), tenant, [payload] * 10)
    assert all(response.status_code == 202 for response in responses)
    receipt = responses[0].json()
    assert receipt["status"] == "queued"
    assert all(response.json() == receipt for response in responses)
    replayed = [response.headers["Idempotency-Replayed"] for response in responses]
    assert replayed.count("false") == 1
    assert replayed.count("true") == 9
    await _assert_single_creation(runtime, receipt, payload["query"])


async def test_two_payloads_race_without_mixing_business_facts(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e36-two-payloads")
    payloads = [{"mode": "research", "query": f"E36-PRIVATE-{i % 2}"} for i in range(10)]
    responses = await _concurrent_posts(http._app(runtime, tenant), tenant, payloads)
    accepted = [response for response in responses if response.status_code == 202]
    assert len(accepted) == 5
    receipt = accepted[0].json()
    winner_queries = {
        payload["query"]
        for payload, response in zip(payloads, responses, strict=True)
        if response.status_code == 202
    }
    assert len(winner_queries) == 1
    winner = winner_queries.pop()
    for payload, response in zip(payloads, responses, strict=True):
        if payload["query"] == winner:
            assert response.status_code == 202
            assert response.json() == receipt
        else:
            assert response.status_code == 409
            assert response.headers["content-type"] == "application/problem+json"
            assert response.json()["detail"] == (
                "The request conflicts with the current resource state."
            )
            for private in (payload["query"], winner, http.KEY, receipt["run_id"]):
                assert private not in response.text
            assert "Idempotency-Replayed" not in response.headers
    replayed = [response.headers["Idempotency-Replayed"] for response in accepted]
    assert replayed.count("false") == 1
    assert replayed.count("true") == 4
    await _assert_single_creation(runtime, receipt, winner)


class _LoseFirstPostResponse(httpx.AsyncBaseTransport):
    """Discard an application-produced response; this is not a socket disconnect."""

    def __init__(self, application):
        self.inner = httpx.ASGITransport(app=application)
        self.lost = False

    async def handle_async_request(self, request):
        response = await self.inner.handle_async_request(request)
        if request.method == "POST" and not self.lost:
            assert response.status_code == 202
            await response.aread()
            await response.aclose()
            self.lost = True
            raise httpx.ReadError("Synthetic response loss after commit", request=request)
        return response

    async def aclose(self):
        await self.inner.aclose()


async def test_committed_http_response_loss_retries_original_run(runtime: support._Runtime) -> None:
    tenant = await support._personal_tenant(runtime, "e36-http-loss")
    transport = _LoseFirstPostResponse(http._app(runtime, tenant))
    path = f"/api/v1/workspaces/{tenant.workspace_id}/runs"
    payload = {"mode": "research", "query": "E36 synthetic lost response"}
    async with asyncio.timeout(20):
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            with pytest.raises(httpx.ReadError, match="Synthetic response loss after commit"):
                await client.post(path, headers={"Idempotency-Key": http.KEY}, json=payload)
            # A fresh session sees committed facts before the client retries.
            async with runtime.session_factory() as session:
                persisted = (await session.scalars(select(Run))).one()
                run_id = persisted.id
                rows = await session.execute(select(RunJob.__table__))
                job_before = dict(rows.mappings().one())
            assert transport.lost
            assert await support._business_counts(runtime.session_factory) == (1, 1, 1, 1, 1)
            replay = await client.post(path, headers={"Idempotency-Key": http.KEY}, json=payload)
    assert replay.status_code == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay.json() == {
        "run_id": str(run_id),
        "status": "queued",
        "events_url": f"{path}/{run_id}/events",
    }
    await _assert_single_creation(runtime, replay.json(), payload["query"])
    async with runtime.session_factory() as session:
        job_after = dict((await session.execute(select(RunJob.__table__))).mappings().one())
    assert job_after == job_before
