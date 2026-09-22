from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Final
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.schemas.problem import ProblemDetail
from app.domain.errors import (
    DomainConflictError,
    DomainError,
    DomainForbiddenError,
    DomainInvariantError,
    DomainNotFoundError,
    DomainUnavailableError,
    DomainValidationError,
    InvalidEventCursorError,
)

PROBLEM_MEDIA_TYPE: Final = "application/problem+json"
REQUEST_ID_HEADER: Final = "X-Request-ID"


class InvalidIdempotencyKeyError(StarletteHTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=400, detail="Idempotency-Key must be a single UUID4.")


@dataclass(frozen=True)
class _ProblemSpec:
    problem_type: str
    title: str
    status: int
    detail: str


_DOMAIN_PROBLEMS: Final = {
    InvalidEventCursorError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:invalid-event-cursor",
        title="Invalid Event Cursor",
        status=400,
        detail="Last-Event-ID must identify a committed event position for this run.",
    ),
    DomainForbiddenError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:forbidden",
        title="Forbidden",
        status=403,
        detail="Access to the requested workspace has been revoked.",
    ),
    DomainNotFoundError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:not-found",
        title="Not Found",
        status=404,
        detail="The requested resource was not found.",
    ),
    DomainConflictError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:conflict",
        title="Conflict",
        status=409,
        detail="The request conflicts with the current resource state.",
    ),
    DomainValidationError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:domain-validation",
        title="Unprocessable Content",
        status=422,
        detail="The request violates a domain rule.",
    ),
    DomainInvariantError: _ProblemSpec(
        problem_type="urn:pathfinder:problem:internal-error",
        title="Internal Server Error",
        status=500,
        detail="An unexpected error occurred.",
    ),
}
_REQUEST_VALIDATION_PROBLEM: Final = _ProblemSpec(
    problem_type="urn:pathfinder:problem:request-validation",
    title="Unprocessable Content",
    status=422,
    detail="The request did not satisfy the required schema.",
)
_INTERNAL_PROBLEM: Final = _ProblemSpec(
    problem_type="urn:pathfinder:problem:internal-error",
    title="Internal Server Error",
    status=500,
    detail="An unexpected error occurred.",
)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


def _problem_response(
    request: Request,
    spec: _ProblemSpec,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    response_headers = dict(headers or {})
    response_headers[REQUEST_ID_HEADER] = request_id
    problem = ProblemDetail(
        type=spec.problem_type,
        title=spec.title,
        status=spec.status,
        detail=spec.detail,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=spec.status,
        content=problem.model_dump(mode="json"),
        headers=response_headers,
        media_type=PROBLEM_MEDIA_TYPE,
    )


def _http_problem(status_code: int) -> _ProblemSpec:
    try:
        title = HTTPStatus(status_code).phrase
    except ValueError:
        title = "HTTP Error"

    if status_code == 404:
        detail = "The requested resource was not found."
    elif status_code == 410:
        detail = (
            "The research and application workflow is retired; historical records are read-only."
        )
    elif status_code == 405:
        detail = "The requested method is not allowed for this resource."
    elif status_code == 503:
        detail = "The service is not ready."
    elif status_code >= 500:
        detail = "An unexpected error occurred."
    else:
        detail = "The request could not be completed."
    return _ProblemSpec(
        problem_type="about:blank",
        title=title,
        status=status_code,
        detail=detail,
    )


async def domain_exception_handler(request: Request, error: DomainError) -> JSONResponse:
    if isinstance(error, DomainUnavailableError):
        return _problem_response(request, _http_problem(503))
    spec = next(
        (
            candidate
            for error_type, candidate in _DOMAIN_PROBLEMS.items()
            if isinstance(error, error_type)
        ),
        _INTERNAL_PROBLEM,
    )
    return _problem_response(request, spec)


async def request_validation_exception_handler(
    request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    return _problem_response(request, _REQUEST_VALIDATION_PROBLEM)


async def http_exception_handler(
    request: Request,
    error: StarletteHTTPException,
) -> JSONResponse:
    return _problem_response(
        request,
        (
            _ProblemSpec(
                problem_type="urn:pathfinder:problem:invalid-idempotency-key",
                title="Invalid Idempotency Key",
                status=400,
                detail="Idempotency-Key must be a single UUID4.",
            )
            if isinstance(error, InvalidIdempotencyKeyError)
            else _http_problem(error.status_code)
        ),
        headers=dict(error.headers or {}),
    )


async def unexpected_exception_handler(
    request: Request,
    _error: Exception,
) -> JSONResponse:
    return _problem_response(request, _INTERNAL_PROBLEM)


def install_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(DomainError, domain_exception_handler)
    application.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    application.add_exception_handler(StarletteHTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unexpected_exception_handler)
