from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from pydantic import ValidationError

from app.mock_portal.contracts import MockSubmissionRequestV1, MockSubmissionResponseV1
from app.mock_portal.service import MockPortalService, MockSubmissionConflictError

router = APIRouter(prefix="/internal/mock-portal", tags=["internal-mock-portal"])


def _service(request: Request) -> MockPortalService:
    service = getattr(request.app.state, "mock_portal_service", None)
    if not isinstance(service, MockPortalService):
        raise RuntimeError("mock portal service is unavailable")
    return service


@router.post(
    "/submissions",
    response_model=MockSubmissionResponseV1,
    status_code=status.HTTP_201_CREATED,
)
async def submit_mock_application(
    request: Request,
    response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> MockSubmissionResponseV1:
    try:
        body = MockSubmissionRequestV1.model_validate_json(
            await request.body(),
            strict=True,
        )
    except (TypeError, ValueError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="invalid submission",
        ) from None
    try:
        result = await _service(request).submit(body, idempotency_key=idempotency_key)
    except MockSubmissionConflictError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="idempotency conflict"
        ) from None
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="invalid key"
        ) from None
    response.status_code = status.HTTP_201_CREATED if result.created else status.HTTP_200_OK
    return result


@router.get(
    "/submissions/by-idempotency-key/{idempotency_key}",
    response_model=MockSubmissionResponseV1,
)
async def get_mock_application(
    idempotency_key: str,
    request: Request,
) -> MockSubmissionResponseV1:
    try:
        result = await _service(request).get(idempotency_key)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="invalid key"
        ) from None
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="submission not found")
    return result
