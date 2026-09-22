"""Historical application assembly. Production has no switch that selects this app."""

from contextlib import asynccontextmanager

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
    application.router.routes = [
        route
        for route in application.router.routes
        if not (
            getattr(route, "methods", set()) == {"POST"}
            and getattr(route, "path", "").startswith("/api/v1/workspaces/")
        )
    ]
    application.include_router(write_router)
    application.include_router(mock_router)
    return application
