"""R7.1 shared-accounting comparison against the actual PostgreSQL/worker boundary."""

import hashlib
from uuid import UUID

import pytest
from pydantic import SecretStr

from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.session import create_database_engine, create_session_factory
from app.domain.errors import DomainNotFoundError, DomainValidationError
from app.domain.resume_profile import ResumeContentV1, ResumePreferencesV1
from tests.evals.product_acceptance_contracts import read_private_json
from tests.evals.resume_quality import load_dataset
from tests.evals.resume_quality_contracts import ComparisonBudget
from tests.evals.resume_quality_runtime import SOURCE, execute_synthetic

pytestmark = pytest.mark.integration


async def test_three_comparisons_use_worker_and_preserve_versions(migrated_database_url, tmp_path):
    root = tmp_path / "comparison"
    root.mkdir(mode=0o700)
    report = await execute_synthetic(
        migrated_database_url, root, load_dataset(), ComparisonBudget(), raise_on_error=True
    )
    assert report["status"] == "SYNTHETIC_PASS", report
    assert report["r71_status"] == "IN_PROGRESS"
    assert report["manual_compile_evidence"] == "NOT_RUN"
    assert report["usage"]["attempts"] >= 15
    assert report["usage"]["cost_status"] == "NOT_APPLICABLE"
    all_ids = []
    for case in report["cases"]:
        arms = case["arms"]
        assert len({arm["common_input_digest"] for arm in arms.values()}) == 1
        system = arms["system"]
        assert len(system["versions"]) == 3
        assert len({v["version_id"] for v in system["versions"]}) == 3
        assert system["revision_rounds"] == 2
        assert system["usage"]["attempts"] == 4
        assert arms["b1"]["usage"]["attempts"] == 1
        assert arms["b1"]["review"]["final_quality"]["compiled"] == "NOT_RUN"
        for version in system["versions"]:
            assert hashlib.sha256(version["tex"].encode()).hexdigest() == version["tex_sha256"]
        all_ids.extend(system["usage"]["invocation_ids"])
        all_ids.extend(arms["b1"]["usage"]["invocation_ids"])
        assert read_private_json(root / f"{case['case_id']}-result.json")["status"] == "COMPLETE"
    assert len(all_ids) == len(set(all_ids))
    assert len({c["session_id"] for c in report["cases"]}) == 3
    # No session's versions can be read through a sibling handle.
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        from sqlalchemy import select

        from app.db.models import ResumeSession
        from app.db.tenancy import SqlAlchemyTenantResolver
        from app.domain.tenancy import TenantService

        sessions = create_session_factory(engine)
        a, b = report["cases"][:2]
        async with sessions() as db:
            row = await db.scalar(
                select(ResumeSession).where(ResumeSession.id == UUID(a["session_id"]))
            )
            tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
                workspace_id=row.workspace_id, actor_user_id=row.owner_user_id
            )
        with pytest.raises(DomainNotFoundError):
            await SqlAlchemyResumeGenerationStore(sessions).get_version(
                tenant,
                UUID(a["session_id"]),
                UUID(b["arms"]["system"]["versions"][0]["version_id"]),
            )
        raw = SOURCE.read_bytes()
        artifacts = SqlAlchemyResumeArtifactStore(
            sessions,
            expected_source_sha256=hashlib.sha256(raw).hexdigest(),
            expected_preamble_sha256=hashlib.sha256(raw.split(b"\\begin{document}")[0]).hexdigest(),
        )
        base = ResumeContentV1.model_validate(a["arms"]["system"]["versions"][1]["content"])
        final = ResumeContentV1.model_validate(a["arms"]["system"]["versions"][2]["content"])
        preferences = ResumePreferencesV1(locked_item_ids=(base.projects[0].bullet_ids[0],))
        changed_bullet = base.projects[0].bullets[0].model_copy(update={"text": "tampered"})
        changed_project = final.projects[0].model_copy(update={"bullets": (changed_bullet,)})
        tampered = final.model_copy(update={"projects": (changed_project,)})
        async with sessions.begin() as db:
            with pytest.raises(DomainValidationError, match="locked"):
                await artifacts.stage(
                    db,
                    tenant,
                    profile_version_id=row.profile_version_id,
                    content=tampered,
                    preferences=preferences,
                    lock_base_content=base,
                )
            with pytest.raises(DomainValidationError, match="personal"):
                await artifacts.stage(
                    db,
                    tenant,
                    profile_version_id=row.profile_version_id,
                    content=final.model_copy(update={"display_name": "tampered"}),
                    preferences=preferences,
                    lock_base_content=base.model_copy(update={"display_name": "tampered"}),
                )
    finally:
        await engine.dispose()


async def test_budget_stop_preserves_partial_report(migrated_database_url, tmp_path):
    root = tmp_path / "stopped"
    root.mkdir(mode=0o700)
    report = await execute_synthetic(
        migrated_database_url, root, load_dataset(), ComparisonBudget(provider_attempt_cap=1)
    )
    assert report["status"] == "PARTIAL"
    assert report["usage"]["attempts"] == 1
    assert read_private_json(root / "report.json")["status"] == "PARTIAL"
    assert all(case["status"] != "COMPLETE" for case in report["cases"])


async def test_recreated_live_recorder_keeps_original_attempt_ledger(
    migrated_database_url, tmp_path
):
    from sqlalchemy import select

    from app.db.models import LLMInvocation
    from app.db.tenancy import SqlAlchemyTenantResolver
    from app.domain.tenancy import TenantService
    from tests.evals.product_acceptance_budget import admit
    from tests.evals.product_acceptance_contracts import AcceptanceError
    from tests.evals.resume_quality_budget import ComparisonRecorder

    root = tmp_path / "resume-ledger"
    root.mkdir(mode=0o700)
    report = await execute_synthetic(
        migrated_database_url, root, load_dataset(), ComparisonBudget(provider_attempt_cap=1)
    )
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        async with sessions() as db:
            row = await db.scalar(select(LLMInvocation))
        tenant = await TenantService(SqlAlchemyTenantResolver(sessions)).resolve_tenant(
            workspace_id=row.workspace_id, actor_user_id=row.actor_user_id
        )
        resumed = ComparisonRecorder(
            sessions, tenant, provider="fake", budget=ComparisonBudget(provider_attempt_cap=1)
        )
        usage = await resumed.measurement()
        assert usage["attempts"] == report["usage"]["attempts"] == 1
        assert usage["invocation_ids"] == report["usage"]["invocation_ids"]
        with pytest.raises(AcceptanceError, match="budget_exhausted"):
            admit(usage, resumed.budget)
        # Unknown live pricing cannot become zero by rebuilding the recorder.
        live = ComparisonRecorder(sessions, tenant, provider="qwen", budget=ComparisonBudget())
        with pytest.raises(AcceptanceError, match="unknown_usage_or_cost"):
            admit(await live.usage(), live.budget)
    finally:
        await engine.dispose()
