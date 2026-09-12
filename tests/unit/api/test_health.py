from __future__ import annotations

import asyncio
import socket
from typing import NoReturn
from uuid import UUID

import httpx
from fastapi import FastAPI

from app.main import create_app


class _ReadinessProbe:
    def __init__(self, result: bool) -> None:
        self._result = result

    async def is_ready(self) -> bool:
        return self._result


class _BrokenReadinessProbe:
    async def is_ready(self) -> bool:
        raise RuntimeError("PATHFINDER-READINESS-CANARY")


async def _send_request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path)


def _request(
    application: FastAPI,
    method: str,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    return asyncio.run(
        _send_request(
            application,
            method,
            path,
            raise_app_exceptions=raise_app_exceptions,
        )
    )


def test_healthz_contract_and_openapi_schema() -> None:
    application = create_app()

    response = _request(application, "GET", "/healthz")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    UUID(response.headers["x-request-id"])
    assert response.json() == {"status": "ok"}

    operation = application.openapi()["paths"]["/healthz"]["get"]
    assert operation.get("parameters", []) == []
    assert "requestBody" not in operation
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/HealthResponse"
    )


def test_healthz_does_not_open_network_connections(monkeypatch) -> None:
    def fail_connect(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("/healthz must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    response = _request(create_app(), "GET", "/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_rejects_post() -> None:
    response = _request(create_app(), "POST", "/healthz")

    assert response.status_code == 405
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["allow"] == "GET"
    assert response.json() == {
        "type": "about:blank",
        "title": "Method Not Allowed",
        "status": 405,
        "detail": "The requested method is not allowed for this resource.",
        "request_id": response.headers["x-request-id"],
    }


def test_readyz_fails_closed_before_lifespan() -> None:
    response = _request(create_app(), "GET", "/readyz")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": "about:blank",
        "title": "Service Unavailable",
        "status": 503,
        "detail": "The service is not ready.",
        "request_id": response.headers["x-request-id"],
    }


def test_readyz_reports_probe_result() -> None:
    application = create_app()
    application.state.readiness_probe = _ReadinessProbe(True)

    ready_response = _request(application, "GET", "/readyz")
    application.state.readiness_probe = _ReadinessProbe(False)
    unavailable_response = _request(application, "GET", "/readyz")

    assert ready_response.status_code == 200
    assert ready_response.headers["content-type"] == "application/json"
    UUID(ready_response.headers["x-request-id"])
    assert ready_response.json() == {"status": "ok"}
    assert unavailable_response.status_code == 503
    assert unavailable_response.headers["content-type"] == "application/problem+json"
    assert unavailable_response.json()["request_id"] == unavailable_response.headers["x-request-id"]


def test_readyz_unknown_probe_error_uses_safe_internal_problem() -> None:
    application = create_app()
    application.state.readiness_probe = _BrokenReadinessProbe()

    response = _request(
        application,
        "GET",
        "/readyz",
        raise_app_exceptions=False,
    )

    assert response.status_code == 500
    assert response.json() == {
        "type": "urn:pathfinder:problem:internal-error",
        "title": "Internal Server Error",
        "status": 500,
        "detail": "An unexpected error occurred.",
        "request_id": response.headers["x-request-id"],
    }
    assert "PATHFINDER-READINESS-CANARY" not in response.text


def test_readyz_rejects_post() -> None:
    response = _request(create_app(), "POST", "/readyz")

    assert response.status_code == 405
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["allow"] == "GET"
    assert response.json() == {
        "type": "about:blank",
        "title": "Method Not Allowed",
        "status": 405,
        "detail": "The requested method is not allowed for this resource.",
        "request_id": response.headers["x-request-id"],
    }


def test_readyz_openapi_documents_success_and_problem_media_types() -> None:
    operation = create_app().openapi()["paths"]["/readyz"]["get"]

    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/HealthResponse"
    )
    problem_schema = operation["responses"]["503"]["content"]["application/problem+json"]["schema"]
    assert problem_schema["title"] == "ProblemDetail"
    assert set(problem_schema["required"]) == {
        "type",
        "title",
        "status",
        "detail",
        "request_id",
    }
