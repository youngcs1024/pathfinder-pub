from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.errors import install_exception_handlers
from app.api.middleware.browser_security import BrowserSecurityHeadersMiddleware
from app.api.middleware.request_context import RequestContextMiddleware
from app.api.router import api_router
from app.auth.fake import FakeActorProvider
from app.auth.supabase import SupabaseActorProvider, SupabaseJwtVerifier
from app.config import Settings
from app.db.approvals import SqlAlchemyApprovalStore
from app.db.events import SqlAlchemyRunEventReader
from app.db.material import SqlAlchemyMaterialStore
from app.db.project_facts import SqlAlchemyProjectFactStore
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.readiness import DatabaseReadinessProbe
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_confirmation import SqlAlchemyResumeConfirmationStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_profiles import SqlAlchemyResumeProfileStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.runs import SqlAlchemyRunStore
from app.db.runtime_policy import DatabaseComponent, DatabasePoolPolicy
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.approvals import ApprovalService
from app.domain.material import MaterialService
from app.domain.project_facts import ProjectFactService
from app.domain.provisioning import ProvisioningService
from app.domain.resume_artifacts import ResumeArtifactService
from app.domain.resume_confirmation import ResumeConfirmationService
from app.domain.resume_generation import ResumeGenerationService
from app.domain.resume_profiles import ResumeProfileService
from app.domain.resume_revision import ResumeRevisionService
from app.domain.runs import RunService
from app.domain.tenancy import TenantService
from app.material.aliases import load_aliases
from app.obs.logging import configure_logging

_STATIC_DIR = Path(__file__).resolve().parent / "api" / "static"


@asynccontextmanager
async def application_lifespan(application: FastAPI) -> AsyncIterator[None]:
    engine = create_database_engine(
        application.state.settings.database_url,
        policy=DatabasePoolPolicy.from_settings(application.state.settings, DatabaseComponent.API),
    )
    readiness_probe: DatabaseReadinessProbe | None = None
    try:
        session_factory = create_session_factory(engine)
        readiness_probe = DatabaseReadinessProbe(engine)
        provisioning_service = ProvisioningService(SqlAlchemyProvisioningStore(session_factory))
        actor_provider = (
            FakeActorProvider(provisioning_service)
            if application.state.settings.auth_mode == "fake"
            else SupabaseActorProvider(
                SupabaseJwtVerifier(application.state.settings),
                provisioning_service,
            )
        )
        tenant_service = TenantService(SqlAlchemyTenantResolver(session_factory))
        run_service = RunService(SqlAlchemyRunStore(session_factory))
        run_event_reader = SqlAlchemyRunEventReader(session_factory)
        approval_service = ApprovalService(SqlAlchemyApprovalStore(session_factory))
        material_aliases = load_aliases(application.state.settings.material_aliases_file)
        material_service = MaterialService(
            SqlAlchemyMaterialStore(session_factory, material_aliases)
        )
        project_fact_service = ProjectFactService(SqlAlchemyProjectFactStore(session_factory))
        resume_profile_service = ResumeProfileService(SqlAlchemyResumeProfileStore(session_factory))
        resume_artifact_service = ResumeArtifactService(
            SqlAlchemyResumeArtifactStore(session_factory)
        )
        resume_confirmation_service = ResumeConfirmationService(
            SqlAlchemyResumeConfirmationStore(session_factory)
        )
        resume_generation_service = ResumeGenerationService(
            SqlAlchemyResumeGenerationStore(session_factory)
        )
        resume_revision_service = ResumeRevisionService(
            SqlAlchemyResumeRevisionStore(session_factory)
        )
        application.state.database_engine = engine
        application.state.database_session_factory = session_factory
        application.state.readiness_probe = readiness_probe
        application.state.actor_provider = actor_provider
        application.state.tenant_service = tenant_service
        application.state.run_service = run_service
        application.state.run_event_reader = run_event_reader
        application.state.approval_service = approval_service
        application.state.material_aliases = material_aliases
        application.state.material_service = material_service
        application.state.project_fact_service = project_fact_service
        application.state.resume_profile_service = resume_profile_service
        application.state.resume_artifact_service = resume_artifact_service
        application.state.resume_generation_service = resume_generation_service
        application.state.resume_confirmation_service = resume_confirmation_service
        application.state.resume_revision_service = resume_revision_service
        yield
    finally:
        try:
            if readiness_probe is not None:
                readiness_probe.mark_not_ready()
        finally:
            try:
                await engine.dispose()
            finally:
                application.state.readiness_probe = None
                application.state.actor_provider = None
                application.state.tenant_service = None
                application.state.run_service = None
                application.state.run_event_reader = None
                application.state.approval_service = None
                application.state.material_aliases = None
                application.state.material_service = None
                application.state.project_fact_service = None
                application.state.resume_profile_service = None
                application.state.resume_artifact_service = None
                application.state.resume_generation_service = None
                application.state.resume_confirmation_service = None
                application.state.resume_revision_service = None
                application.state.database_session_factory = None
                application.state.database_engine = None


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings()
    configure_logging(log_level=resolved_settings.log_level)

    application = FastAPI(lifespan=application_lifespan)
    application.state.settings = resolved_settings
    application.add_middleware(RequestContextMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "Idempotency-Key"],
    )
    application.add_middleware(
        BrowserSecurityHeadersMiddleware,
        browser_connect_origin=resolved_settings.supabase_url,
    )
    install_exception_handlers(application)
    application.include_router(api_router)
    application.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    return application


app = create_app()


def run() -> None:
    uvicorn.run(
        app,
        host=app.state.settings.api_host,
        port=8000,
        lifespan="on",
        access_log=False,
        log_config=None,
        workers=1,
    )


if __name__ == "__main__":
    run()
