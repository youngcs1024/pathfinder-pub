from __future__ import annotations

import asyncio
import json
from uuid import UUID

import httpx
from fastapi import FastAPI

from app.domain.errors import DomainConflictError
from app.main import create_app
from app.obs.logging import configure_logging, get_logging_context

CANARY = "PATHFINDER-REQUEST-CANARY"


class _FailingStream:
    def write(self, _value: str) -> int:
        raise RuntimeError(CANARY)

    def flush(self) -> None:
        raise RuntimeError(CANARY)


async def _send_request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, headers=headers)


def _request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    return asyncio.run(
        _send_request(
            application,
            method,
            path,
            headers=headers,
            raise_app_exceptions=raise_app_exceptions,
        )
    )


def _last_log(captured: str) -> dict[str, object]:
    return json.loads(captured.strip().splitlines()[-1])


def test_server_request_id_is_shared_by_response_and_log_and_context_is_cleared(capsys) -> None:
    application = create_app()

    response = _request(
        application,
        "GET",
        f"/healthz?token={CANARY}",
        headers={"X-Request-ID": CANARY},
    )

    request_id = response.headers["x-request-id"]
    UUID(request_id)
    assert request_id != CANARY
    event = _last_log(capsys.readouterr().out)
    assert event["event"] == "http.request.completed"
    assert event["request_id"] == request_id
    assert event["method"] == "GET"
    assert event["route"] == "/healthz"
    assert event["status_code"] == 200
    assert CANARY not in json.dumps(event)
    assert get_logging_context() == {}


def test_concurrent_requests_have_distinct_ids_and_no_context_leak() -> None:
    application = create_app()

    async def run_requests() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(*(client.get("/healthz") for _ in range(8)))

    responses = asyncio.run(run_requests())
    request_ids = {response.headers["x-request-id"] for response in responses}

    assert len(request_ids) == len(responses)
    assert all(response.status_code == 200 for response in responses)
    assert get_logging_context() == {}


def test_http_problems_share_request_id_with_completed_log(capsys) -> None:
    application = create_app()

    for method, path, expected_status in (
        ("GET", "/missing", 404),
        ("POST", "/healthz", 405),
    ):
        response = _request(application, method, path)
        event = _last_log(capsys.readouterr().out)

        assert response.status_code == expected_status
        assert response.json()["request_id"] == response.headers["x-request-id"]
        assert event["request_id"] == response.headers["x-request-id"]
        assert event["status_code"] == expected_status


def test_validation_and_domain_problems_share_request_id_with_log(capsys) -> None:
    application = create_app()

    @application.get("/items/{item_id}")
    async def get_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    @application.get("/domain-conflict")
    async def domain_conflict() -> None:
        raise DomainConflictError(CANARY)

    for path, expected_status in (
        (f"/items/{CANARY}", 422),
        ("/domain-conflict", 409),
    ):
        response = _request(application, "GET", path)
        event = _last_log(capsys.readouterr().out)

        assert response.status_code == expected_status
        assert response.json()["request_id"] == response.headers["x-request-id"]
        assert event["request_id"] == response.headers["x-request-id"]
        assert event["status_code"] == expected_status
        assert CANARY not in json.dumps(event)


def test_unexpected_exception_log_and_problem_share_request_id_without_message(capsys) -> None:
    application = create_app()

    @application.get("/explode")
    async def explode() -> None:
        raise RuntimeError(CANARY)

    response = _request(
        application,
        "GET",
        "/explode",
        raise_app_exceptions=False,
    )

    event = _last_log(capsys.readouterr().out)
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert event["event"] == "http.request.failed"
    assert event["request_id"] == response.headers["x-request-id"]
    assert event["status_code"] == 500
    assert event["error_type"] == "RuntimeError"
    assert CANARY not in response.text
    assert CANARY not in json.dumps(event)


def test_logging_sink_failure_does_not_break_health_or_echo_secret(capsys) -> None:
    application = create_app()
    configure_logging(log_level="INFO", stream=_FailingStream())

    response = _request(application, "GET", "/healthz")

    assert response.status_code == 200
    captured = capsys.readouterr()
    assert CANARY not in captured.out
    assert CANARY not in captured.err
