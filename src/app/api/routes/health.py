from typing import Protocol

from fastapi import APIRouter, HTTPException, Request, status

from app.api.errors import PROBLEM_MEDIA_TYPE
from app.api.schemas.health import HealthResponse
from app.api.schemas.problem import ProblemDetail

router = APIRouter(tags=["health"])


class ReadinessProbe(Protocol):
    async def is_ready(self) -> bool: ...


@router.get("/healthz", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get(
    "/readyz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "The service is not ready.",
            "content": {
                PROBLEM_MEDIA_TYPE: {
                    "schema": ProblemDetail.model_json_schema(),
                }
            },
        }
    },
)
async def readyz(request: Request) -> HealthResponse:
    probe: ReadinessProbe | None = getattr(request.app.state, "readiness_probe", None)
    if probe is None or not await probe.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    return HealthResponse(status="ok")
