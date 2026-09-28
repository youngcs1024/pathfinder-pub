"""A fixture import and three-arm business execution against real PostgreSQL."""

import json
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr
from sqlalchemy import select

from app.db.models import LLMInvocation, ResumeVersion
from app.db.provisioning import SqlAlchemyProvisioningStore
from app.db.resume_artifacts import SqlAlchemyResumeArtifactStore
from app.db.resume_generation import SqlAlchemyResumeGenerationStore
from app.db.resume_revision import SqlAlchemyResumeRevisionStore
from app.db.run_execution import SqlAlchemyRunExecutionReader
from app.db.session import create_database_engine, create_session_factory
from app.db.tenancy import SqlAlchemyTenantResolver
from app.domain.errors import DomainNotFoundError
from app.domain.provisioning import ProvisioningService
from app.domain.resume_generation import GenerationBudgetV1, JobInputV1, SessionCreateV1
from app.domain.tenancy import TenantService
from app.llm.factory import LLMFactory
from app.llm.fake import FakeEmbeddingModel, ScriptedFakeChatModel
from app.llm.ports import ChatModelResult, ModelUsage
from tests.evals.product_acceptance_contracts import AcceptanceError, publish
from tests.evals.resume_experiment_budget import ExperimentRecorder
from tests.evals.resume_experiment_contracts import ExperimentBudget
from tests.evals.resume_initial import generate_sample, score_block
from tests.evals.resume_initial_fixture import seed
from tests.evals.test_resume_initial import proposal, selection
from tests.unit.agents.test_resume_generation import _inputs, _requirements, _selection

pytestmark = pytest.mark.integration


async def test_a_snapshot_three_arms_recovery_authorization(migrated_database_url, tmp_path):
    tmp_path.chmod(0o700)
    directory = tmp_path / "a"
    directory.mkdir(mode=0o700)
    material = tmp_path / "inputs"
    material.mkdir(mode=0o700)
    raw, identity, generation, item_id, fact_id = _inputs()
    pid = generation.facts[0].project_id
    profile_id, version_id = uuid4(), uuid4()
    claim = generation.facts[0].claim
    profile = {
        "profile_id": str(profile_id),
        "version_id": str(version_id),
        "version": 1,
        "preference_version": 1,
        "content": generation.profile_content.model_dump(mode="json"),
        "preferences": generation.preferences.model_dump(mode="json"),
        "claims": [
            {
                "id": str(uuid4()),
                "item_id": str(item_id),
                "project_item_id": str(item_id),
                "field": "summary",
                "text": claim,
                "source": {},
                "review_version": 1,
                "decision": "linked",
                "project_id": str(pid),
                "fact_version_ids": [str(fact_id)],
            }
        ],
    }
    fact = {
        "id": str(uuid4()),
        "version_id": str(fact_id),
        "version": 1,
        "claim": claim,
        "kind": "implementation",
        "conditions": {},
        "issues": [],
        "review_status": "confirmed",
        "evidence": [{"path": "fact.txt", "start_line": 1, "end_line": 1, "quote": claim}],
    }
    publish(material / "profile.json", profile)
    publish(material / "facts.json", {"facts": [fact]})
    publish(
        material / "r71-facts-done.json",
        {"projects": {"synthetic": {"project_id": str(pid), "fact_version_ids": [str(fact_id)]}}},
    )
    (material / "resume.tex").write_bytes(raw)
    (material / "fact.txt").write_text(claim)
    engine = create_database_engine(SecretStr(migrated_database_url))
    try:
        sessions = create_session_factory(engine)
        provision = ProvisioningService(SqlAlchemyProvisioningStore(sessions))
        tenancy = TenantService(SqlAlchemyTenantResolver(sessions))
        owner = await provision.provision_personal_workspace(f"a-{uuid4()}")
        tenant = await tenancy.resolve_tenant(
            workspace_id=owner.workspace_id, actor_user_id=owner.user_id
        )
        fixture = await seed(sessions, tenant, tmp_path, directory)
        assert await seed(sessions, tenant, tmp_path, directory) == fixture
        recorder = ExperimentRecorder(
            sessions,
            tenant,
            root=tmp_path,
            inputs=SimpleNamespace(
                allocation_id=uuid4(), digest="synthetic-a", budget=ExperimentBudget()
            ),
            database_identity="synthetic-db",
            provider="fake",
        )
        await recorder.initialize()
        rig = SimpleNamespace(
            sessions=sessions,
            tenant=tenant,
            store=SqlAlchemyResumeGenerationStore(sessions),
            revisions=SqlAlchemyResumeRevisionStore(sessions),
            reader=SqlAlchemyRunExecutionReader(sessions),
            recorder=recorder,
            source_bytes=raw,
            identity=identity,
            artifacts=SqlAlchemyResumeArtifactStore(
                sessions,
                expected_source_sha256=identity.source_sha256,
                expected_preamble_sha256=identity.preamble_sha256,
            ),
        )
        request = SessionCreateV1(
            profile_version_id=UUID(fixture["profile_version_id"]),
            preference_version=1,
            project_ids=(pid,),
            job=JobInputV1(source="paste", text=generation.job_text),
            budget=GenerationBudgetV1(
                max_model_calls=12, max_tool_calls=0, max_cost_cny=Decimal("100")
            ),
        )
        key = uuid4()
        created = await rig.store.create(tenant, request, key)
        assert (
            await rig.store.create(tenant, request, key)
        ).receipt.resource_id == created.receipt.resource_id
        inputs = await rig.store.execution_inputs(tenant, created.receipt.resource_id)
        assert inputs.facts[0].version_id == fact_id and inputs.facts[0].claim == claim
        assert inputs.profile_content == generation.profile_content
        outputs = {}
        block = []
        for arm in ("one_shot", "selection", "pathfinder"):
            script = (
                [ChatModelResult(content=json.dumps(proposal(inputs)))]
                if arm == "one_shot"
                else [ChatModelResult(content=json.dumps(selection(item_id, fact_id)))]
                if arm == "selection"
                else [_requirements(), _selection(item_id, fact_id)]
            )
            script = [
                s.model_copy(update={"usage": ModelUsage(input_tokens=10, output_tokens=10)})
                for s in script
            ]
            rig.factory = LLMFactory(
                recorder=recorder,
                chat_adapter=ScriptedFakeChatModel(script),
                embedding_adapter=FakeEmbeddingModel(),
                provider="fake",
            )
            sample = {
                "sample_id": f"test-r1-{arm}",
                "arm": arm,
                "case_id": "test",
                "repeat": 1,
                "c_start": arm != "selection",
            }
            block.append(sample)
            output = await generate_sample(rig, sample, inputs, directory / sample["sample_id"])
            assert output["generation_status"] == "COMPLETE", output
            before = await recorder.measurement()
            assert (
                await generate_sample(rig, sample, inputs, directory / sample["sample_id"])
                == output
            )
            assert await recorder.measurement() == before
            outputs[arm] = output
        assert len({o["common_input_digest"] for o in outputs.values()}) == 1
        assert (await recorder.measurement())["attempts"] == 4
        async with sessions() as db:
            versions = list(
                await db.scalars(
                    select(ResumeVersion).where(ResumeVersion.workspace_id == tenant.workspace_id)
                )
            )
            assert len(versions) == 1
            assert (
                len(
                    list(
                        await db.scalars(
                            select(LLMInvocation).where(
                                LLMInvocation.workspace_id == tenant.workspace_id
                            )
                        )
                    )
                )
                == 4
            )
        assessment = {
            "claims": [
                {
                    "section": "body",
                    "quote": claim,
                    "support": "full",
                    "fact_version_ids": [str(fact_id)],
                    "profile_item_ids": [],
                    "experimental": False,
                    "conditions_complete": True,
                    "rationale": "Exact fact",
                }
            ],
            "coverage": [{"requirement_id": "req", "status": "full", "rationale": "Exact support"}],
        }
        rig.facts, rig.profile_evidence, rig.seed = [fact], {}, 42
        rig.factory = LLMFactory(
            recorder=recorder,
            chat_adapter=ScriptedFakeChatModel(
                [
                    ChatModelResult(
                        content=json.dumps(assessment),
                        usage=ModelUsage(input_tokens=10, output_tokens=10),
                    )
                    for _ in range(3)
                ]
            ),
            embedding_adapter=FakeEmbeddingModel(),
            provider="fake",
        )
        annotation = {
            "requirements": [
                {
                    "requirement_id": "req",
                    "quote": inputs.job_text,
                    "start": 0,
                    "end": len(inputs.job_text),
                    "kind": "explicit",
                    "applicable": True,
                    "support": "full",
                    "fact_version_ids": [str(fact_id)],
                    "profile_item_ids": [],
                    "necessary_conditions": [],
                    "rationale": "Synthetic",
                }
            ]
        }
        case = SimpleNamespace(case_id="test", jd=inputs.job_text)
        await score_block(rig, directory, block, case, annotation)
        scored_usage = await recorder.measurement()
        assert scored_usage["attempts"] == 7
        await score_block(rig, directory, block, case, annotation)
        assert await recorder.measurement() == scored_usage
        other = await provision.provision_personal_workspace(f"other-{uuid4()}")
        foreign = await tenancy.resolve_tenant(
            workspace_id=other.workspace_id, actor_user_id=other.user_id
        )
        with pytest.raises(DomainNotFoundError):
            await rig.store.execution_inputs(foreign, created.receipt.resource_id)
        changed = dict(profile, content={**profile["content"], "schema_version": 2})
        # Compare bound database data directly on a changed input; no fixture deletion.
        from unittest.mock import patch

        original_read = __import__(
            "tests.evals.resume_initial_fixture", fromlist=["read_private_json"]
        ).read_private_json
        with patch(
            "tests.evals.resume_initial_fixture.read_private_json",
            side_effect=lambda p: changed if p.name == "profile.json" else original_read(p),
        ):
            with pytest.raises(AcceptanceError, match="fixture_database_changed"):
                await seed(sessions, tenant, tmp_path, directory)
    finally:
        await engine.dispose()
