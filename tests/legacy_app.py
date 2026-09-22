"""Historical application assembly. Production has no switch that selects this app."""

from contextlib import asynccontextmanager

from fastapi import APIRouter

from app.api.router import api_router
from app.api.routes.action_intents import router as action_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.me import router as me_router
from app.api.routes.runs import router as runs_router
from app.api.routes.ui import router as ui_router
from app.db.mock_submissions import SqlAlchemyMockSubmissionRepository
from app.domain.runs import RunService
from app.main import application_lifespan
from app.main import create_app as _create_app
from app.mock_portal.router import router as mock_router
from app.mock_portal.service import MockPortalService
from tests.legacy_routes import router as write_router
from tests.legacy_runtime import SqlAlchemyRunStore


@asynccontextmanager
async def _legacy_lifespan(application):
    async with application_lifespan(application):
        sessions = application.state.database_session_factory
        application.state.run_service = RunService(SqlAlchemyRunStore(sessions))
        application.state.mock_portal_service = MockPortalService(
            SqlAlchemyMockSubmissionRepository(sessions)
        )
        try:
            yield
        finally:
            application.state.mock_portal_service = None


def create_app(settings=None):
    application = _create_app(settings)
    application.router.lifespan_context = _legacy_lifespan
    # FastAPI includes nested routers without flattening their routes. Replace the
    # included router itself; never mutate the globally shared production router.
    application.router.routes = [
        route
        for route in application.router.routes
        if getattr(route, "original_router", None) is not api_router
    ]
    reads = APIRouter()
    for router in (runs_router, action_router):
        reads.routes.extend(route for route in router.routes if "GET" in route.methods)
    for router in (ui_router, health_router, me_router, events_router, reads):
        application.include_router(router)
    application.include_router(write_router)
    application.include_router(mock_router)
    return application
