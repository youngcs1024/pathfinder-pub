from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import FileResponse

from app.api.schemas.ui import UIConfigResponse

router = APIRouter()
_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@router.get("/", include_in_schema=False, response_class=FileResponse)
async def get_ui() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


@router.get(
    "/api/v1/ui-config",
    response_model=UIConfigResponse,
    status_code=status.HTTP_200_OK,
    tags=["ui"],
)
async def get_ui_config(request: Request) -> UIConfigResponse:
    settings = request.app.state.settings
    if settings.auth_mode == "fake":
        return UIConfigResponse(
            auth_mode="fake",
            supabase_url=None,
            supabase_publishable_key=None,
        )
    return UIConfigResponse(
        auth_mode="supabase",
        supabase_url=settings.supabase_url,
        supabase_publishable_key=settings.supabase_publishable_key,
    )
