from fastapi import APIRouter

from app.api.routes.action_intents import router as action_intents_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.material import router as material_router
from app.api.routes.me import router as me_router
from app.api.routes.project_facts import router as project_facts_router
from app.api.routes.resume_artifacts import router as resume_artifacts_router
from app.api.routes.resume_generation import router as resume_generation_router
from app.api.routes.resume_profiles import router as resume_profiles_router
from app.api.routes.runs import router as runs_router
from app.api.routes.ui import router as ui_router

api_router = APIRouter()
api_router.include_router(ui_router)
api_router.include_router(health_router)
api_router.include_router(me_router)
api_router.include_router(material_router)
api_router.include_router(project_facts_router)
api_router.include_router(resume_profiles_router)
api_router.include_router(resume_artifacts_router)
api_router.include_router(resume_generation_router)
api_router.include_router(runs_router)
api_router.include_router(events_router)
api_router.include_router(action_intents_router)
