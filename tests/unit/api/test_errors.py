from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    DomainConflictError,
    DomainError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainValidationError,
)
from app.main import create_app

CANARY = "PATHFINDER-ERROR-CANARY"


async def _send_request(
    application: FastAPI,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def _request(
    application: FastAPI,
    path: str,
    *,
    raise_app_exceptions: bool = True,
) -> httpx.Response:
    return asyncio.run(
        _send_request(
            application,
            path,
            raise_app_exceptions=raise_app_exceptions,
        )
    )


def _app_raising(path: str, error_factory: Callable[[], Exception]) -> FastAPI:
    application = create_app()

    @application.get(path)
    async def raise_error() -> None:
        raise error_factory()

    return application


@pytest.mark.parametrize(
    ("error_type", "status", "problem_type", "title", "detail"),
    [
        (
            DomainNotFoundError,
            404,
            "urn:pathfinder:problem:not-found",
            "Not Found",
            "The requested resource was not found.",
        ),
        (
            DomainConflictError,
            409,
            "urn:pathfinder:problem:conflict",
            "Conflict",
            "The request conflicts with the current resource state.",
        ),
        (
            DomainValidationError,
            422,
            "urn:pathfinder:problem:domain-validation",
            "Unprocessable Content",
            "The request violates a domain rule.",
        ),
        (
            DomainInvariantError,
            500,
            "urn:pathfinder:problem:internal-error",
            "Internal Server Error",
            "An unexpected error occurred.",
        ),
    ],
)
def test_typed_domain_errors_map_to_safe_problem_details(
    error_type: type[DomainError],
    status: int,
    problem_type: str,
    title: str,
    detail: str,
) -> None:
    application = _app_raising("/domain-error", lambda: error_type(CANARY))

    response = _request(application, "/domain-error")

    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json() == {
        "type": problem_type,
        "title": title,
        "status": status,
        "detail": detail,
        "request_id": response.headers["x-request-id"],
    }
    UUID(response.headers["x-request-id"])
    assert CANARY not in response.text


def test_request_validation_does_not_echo_invalid_value() -> None:
    application = create_app()

    @application.get("/items/{item_id}")
    async def get_item(item_id: int) -> dict[str, int]:
        return {"item_id": item_id}

    response = _request(application, f"/items/{CANARY}")

    assert response.status_code == 422
    assert response.json() == {
        "type": "urn:pathfinder:problem:request-validation",
        "title": "Unprocessable Content",
        "status": 422,
        "detail": "The request did not satisfy the required schema.",
        "request_id": response.headers["x-request-id"],
    }
    assert CANARY not in response.text


def test_http_exception_uses_safe_detail_and_preserves_headers() -> None:
    application = _app_raising(
        "/http-error",
        lambda: StarletteHTTPException(
            status_code=409,
            detail=CANARY,
            headers={"X-Safe-Test": "preserved"},
        ),
    )

    response = _request(application, "/http-error")

    assert response.status_code == 409
    assert response.headers["x-safe-test"] == "preserved"
    assert response.json() == {
        "type": "about:blank",
        "title": "Conflict",
        "status": 409,
        "detail": "The request could not be completed.",
        "request_id": response.headers["x-request-id"],
    }
    assert CANARY not in response.text


def test_unexpected_exception_returns_generic_problem() -> None:
    application = _app_raising("/explode", lambda: RuntimeError(CANARY))

    response = _request(
        application,
        "/explode",
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
    assert CANARY not in response.text


@pytest.mark.parametrize("commit_unknown", [False, True])
def test_database_unavailable_reuses_503_without_retry_or_commit_claim(commit_unknown):
    from app.domain.errors import DomainUnavailableError

    application = _app_raising(
        "/storage-error",
        lambda: DomainUnavailableError(commit_outcome_unknown=commit_unknown),
    )
    response = _request(application, "/storage-error")
    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "about:blank"
    assert response.json()["detail"] == "The service is not ready."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert "Retry-After" not in response.headers
    assert "commit" not in response.text
