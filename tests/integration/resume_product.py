"""Synthetic R7.2 API/worker assembly; production services, no legacy executor."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr
from sqlalchemy import select

from app.config import Settings
from app.db.llm_invocations import SqlAlchemyInvocationRecorder
from app.db.material import SqlAlchemyMaterialStore
from app.db.models import MaterialSnapshotFile
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_confirmation import SqlAlchemyResumeConfirmationStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_profiles import SqlAlchemyResumeProfileStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.domain.material import MaterialService
from app.domain.resume_confirmation import ResumeConfirmationService
from app.domain.resume_profiles import ResumeProfileService
from app.llm.ports import ChatModelResult
from app.main import create_app
from app.material.aliases import MaterialAlias, MaterialAliasRegistry
from tests.evals.resume_quality_runtime import factory, worker
from tests.integration.db.test_material_snapshots import _runner
from tests.unit.agents.test_resume_generation import _requirements, _selection

SOURCE = Path(__file__).resolve().parents[1] / "fixtures/resume/synthetic_main.tex"


@asynccontextmanager
async def product(database_url, root):
    root.mkdir(mode=0o700)
    app = create_app(
        Settings(
            _env_file=None,
            database_url=SecretStr(database_url),
            log_level="ERROR",
            auth_mode="fake",
            llm_mode="fake",
            search_mode="fake",
            trace_mode="off",
        )
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/api/v1/me")
            assert response.status_code == 200
            me = response.json()
            workspace = me["workspaces"][0]["workspace_id"]
            tenant = await app.state.tenant_service.resolve_tenant(
                workspace_id=UUID(workspace), actor_user_id=UUID(me["user_id"])
            )
            sessions = app.state.database_session_factory
            (root / "facts.txt").write_text("Built a synthetic service\n")
            aliases = MaterialAliasRegistry(
                (MaterialAlias("demo", "file", root, ("facts.txt",), (tenant.workspace_id,)),)
            )
            materials = SqlAlchemyMaterialStore(sessions, aliases)
            app.state.material_aliases = aliases
            app.state.material_service = MaterialService(materials)
            raw = SOURCE.read_bytes()
            template = dict(
                expected_source_sha256=hashlib.sha256(raw).hexdigest(),
                expected_preamble_sha256=hashlib.sha256(
                    raw.split(b"\\begin{document}")[0]
                ).hexdigest(),
            )
            artifacts = SqlAlchemyResumeArtifactStore(sessions, **template)
            app.state.resume_profile_service = ResumeProfileService(
                SqlAlchemyResumeProfileStore(sessions, **template)
            )
            app.state.resume_confirmation_service = ResumeConfirmationService(
                SqlAlchemyResumeConfirmationStore(sessions, artifacts)
            )
            yield SimpleNamespace(
                app=app,
                client=client,
                sessions=sessions,
                tenant=tenant,
                materials=materials,
                base=f"/api/v2/workspaces/{workspace}",
                artifacts=artifacts,
                store=SqlAlchemyResumeGenerationStore(sessions),
                revisions=SqlAlchemyResumeRevisionStore(sessions),
                reader=SqlAlchemyRunExecutionReader(sessions),
                recorder=SqlAlchemyInvocationRecorder(sessions),
            )


async def get(rig, path):
    response = await rig.client.get(path)
    assert response.status_code == 200, "product read failed"
    return response.json()


async def post(rig, path, body, *, key=None):
    key = str(key or uuid4())
    response = await rig.client.post(path, json=body, headers={"Idempotency-Key": key})
    assert response.status_code in (200, 201, 202), (
        "product command failed",
        response.status_code,
        [
            (e.get("type"), e.get("loc"))
            for e in response.json().get("detail", [])
            if isinstance(e, dict)
        ],
    )
    return response.json(), key


async def seed(rig):
    project, _ = await post(rig, rig.base + "/projects", {"name": "R72 synthetic"})
    project_path = rig.base + "/projects/" + project["id"]
    source, _ = await post(rig, project_path + "/material-sources", {"alias": "demo"})
    imported, _ = await post(rig, project_path + "/imports", {"source_ids": [source["id"]]})
    assert await _runner(rig.sessions, rig.materials).run_once(asyncio.Event())
    progress = await get(rig, project_path + "/imports/" + imported["import_id"])
    assert progress["status"] == "completed"
    async with rig.sessions() as db:
        file_id = await db.scalar(
            select(MaterialSnapshotFile.id).where(
                MaterialSnapshotFile.workspace_id == rig.tenant.workspace_id,
                MaterialSnapshotFile.path == "facts.txt",
            )
        )
    added, _ = await post(
        rig,
        project_path + "/facts",
        {
            "import_id": imported["import_id"],
            "candidate": {
                "claim": "Built a synthetic service",
                "kind": "personal_statement",
                "evidence": [
                    {
                        "snapshot_file_id": str(file_id),
                        "start_line": 1,
                        "end_line": 1,
                        "quote": "Built a synthetic service",
                    }
                ],
            },
        },
    )
    await post(
        rig,
        project_path + "/facts/" + added["resource_id"] + "/reviews",
        {
            "import_id": imported["import_id"],
            "expected_version": 1,
            "decision": "confirm",
            "attested": True,
        },
    )
    catalog = await get(rig, project_path + "/facts")
    fact = next(f for f in catalog["facts"] if f["id"] == added["resource_id"])
    profile_receipt, _ = await post(
        rig, rig.base + "/profiles/imports", {"source_tex": SOURCE.read_text()}
    )
    profile_path = rig.base + "/profiles/" + profile_receipt["resource_id"]
    profile = await get(rig, profile_path)
    item = profile["content"]["projects"][0]["id"]
    await post(rig, profile_path + "/item-reviews", {"expected_version": 1, "item_id": item})
    profile = await get(rig, profile_path)
    claim = next(c for c in profile["claims"] if c["project_item_id"] == item)
    await post(
        rig,
        profile_path + "/claim-reviews",
        {
            "claim_id": claim["id"],
            "expected_review_version": 0,
            "decision": "linked",
            "project_id": project["id"],
            "fact_version_ids": [fact["version_id"]],
        },
    )
    body = {
        "profile_version_id": profile["version_id"],
        "preference_version": 1,
        "project_ids": [project["id"]],
        "job": {"source": "paste", "text": "Build a synthetic service"},
        "budget": {"max_model_calls": 6, "max_tool_calls": 0, "max_cost_cny": "1"},
    }
    path = rig.base + "/resume-sessions"
    created, key = await post(rig, path, body)
    replay, _ = await post(rig, path, body, key=key)
    assert replay["replayed"] and replay["run_id"] == created["run_id"]
    assert await worker(
        rig, factory(rig, [_requirements(), _selection(item, fact["version_id"])])
    ).run_once(asyncio.Event())
    state = {
        "session_id": created["session_id"],
        "create_body": body,
        "create_key": key,
        "run_id": created["run_id"],
        "fact_id": fact["version_id"],
        "versions": [],
    }
    state["versions"].append(await snapshot(rig, state))
    return state


def session_path(rig, state):
    return rig.base + "/resume-sessions/" + state["session_id"]


async def snapshot(rig, state):
    path = session_path(rig, state)
    detail = await get(rig, path)
    assert detail["run_status"] == "completed", "worker did not complete"
    version = await get(rig, path + "/versions/" + detail["current_version_id"])
    downloaded = await rig.client.get(path + "/versions/" + version["version_id"] + "/download")
    assert downloaded.status_code == 200
    digest = hashlib.sha256(downloaded.content).hexdigest()
    assert digest == downloaded.headers["x-content-sha256"]
    return {**version, "tex": downloaded.content, "tex_sha256": digest}


async def revise(rig, state, *, lock=False):
    path = session_path(rig, state)
    current = state["versions"][-1]
    project = current["content"]["projects"][0]
    detail = await get(rig, path)
    if lock:
        await post(
            rig,
            path + "/locks",
            {
                "expected_session_revision": detail["revision"],
                "base_version_id": current["version_id"],
                "item_id": project["bullet_ids"][0],
                "locked": True,
            },
        )
        detail = await get(rig, path)
    summary = bool(detail["locked_item_ids"])
    target = project["id"] if summary else project["bullet_ids"][0]
    patch = {
        "operation": "replace_text",
        "item_id": target,
        "field": "summary" if summary else "bullet",
        "text": (project["summary"]["text"] if summary else project["bullets"][0]["text"]) + ".",
        "fact_version_ids": [state["fact_id"]],
    }
    body = {
        "kind": "content",
        "expected_session_revision": detail["revision"],
        "base_version_id": current["version_id"],
        "target_item_ids": [target],
        "instruction": "Clarify the existing synthetic statement",
    }
    feedback, key = await post(rig, path + "/feedback", body)
    replay, _ = await post(rig, path + "/feedback", body, key=key)
    assert replay["replayed"] and feedback["run_id"] == replay["run_id"]
    script = [ChatModelResult(content=json.dumps({"patches": [patch]}))]
    assert await worker(rig, factory(rig, script)).run_once(asyncio.Event())
    revised = await snapshot(rig, state)
    assert revised["version_id"] != current["version_id"]
    assert revised["content"]["display_name"] == current["content"]["display_name"]
    if summary:
        assert revised["content"]["projects"][0]["bullets"] == project["bullets"]
    state["versions"].append(revised)


async def confirm(rig, state):
    path = session_path(rig, state)
    current = state["versions"][-1]
    detail = await get(rig, path)
    body = {
        "version_id": current["version_id"],
        "expected_session_revision": detail["revision"],
        "expected_current_version_id": current["version_id"],
        "artifact_id": current["artifact_id"],
        "tex_sha256": current["tex_sha256"],
        "attested": True,
    }
    endpoint = path + "/versions/" + current["version_id"] + "/confirm"
    receipt, key = await post(rig, endpoint, body)
    replay, _ = await post(rig, endpoint, body, key=key)
    assert replay["replayed"] and replay["confirmation_id"] == receipt["confirmation_id"]
    state.update(confirm_body=body, confirm_key=key, confirmation=receipt)
    await verify_history(rig, state)


async def verify_history(rig, state):
    path = session_path(rig, state)
    versions = await get(rig, path + "/versions")
    assert len(versions) == len(state["versions"])
    for version in state["versions"]:
        endpoint = path + "/versions/" + version["version_id"] + "/download"
        for _ in range(2):
            response = await rig.client.get(endpoint)
            assert response.status_code == 200 and response.content == version["tex"]
            assert response.headers["x-content-sha256"] == version["tex_sha256"]
        foreign = endpoint.replace(str(rig.tenant.workspace_id), str(uuid4()))
        assert (await rig.client.get(foreign)).status_code == 404
    confirmed_id = state["confirmation"]["version_id"]
    matching = next(v for v in versions if v["version_id"] == confirmed_id)
    assert matching["confirmation"]["tex_sha256"] == state["confirmation"]["tex_sha256"]
    replay, _ = await post(
        rig, rig.base + "/resume-sessions", state["create_body"], key=state["create_key"]
    )
    assert replay["replayed"] and replay["session_id"] == state["session_id"]


async def check_sse(rig, state):
    path = f"/api/v1/workspaces/{rig.tenant.workspace_id}/runs/{state['run_id']}/events"
    response = await rig.client.get(path)
    assert response.status_code == 200
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    assert ids and ids == sorted(set(ids))
    replay = await rig.client.get(path, headers={"Last-Event-ID": str(ids[-1])})
    assert replay.status_code == 200 and "id: " not in replay.text
