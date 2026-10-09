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


def snapshot_material(tmp_path):
    tmp_path.chmod(0o700)
    directory = tmp_path / "pilot"
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
    return directory, raw, identity, generation, item_id, fact_id, pid, claim, fact, profile


async def test_a_snapshot_three_arms_recovery_authorization(
    migrated_database_url, tmp_path, monkeypatch
):
    directory, raw, identity, generation, item_id, fact_id, pid, claim, fact, profile = (
        snapshot_material(tmp_path)
    )
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
        rig.facts, rig.profile_evidence, rig.seed = [fact], {}, 42
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
        from tests.evals.resume_experiment_contracts import Annotation
        from tests.evals.resume_experiment_scoring import blind_packets
        from tests.evals.resume_initial_baselines import candidate_text
        from tests.evals.resume_initial_scoring import packets_for
        from tests.evals.test_resume_initial_scoring import synthetic_response

        packets, mapping = blind_packets(
            [
                {
                    "sample_id": s["sample_id"],
                    "text": candidate_text(outputs[s["arm"]]["output"]["content"]),
                }
                for s in block
            ],
            case=case,
            facts=[fact],
            annotation=Annotation.model_validate_json(json.dumps(annotation)),
            seed=43,
            profile={},
        )
        scripts = []
        for packet in packets:
            sample = next(s for s in block if s["sample_id"] == mapping[packet["blind_id"]])
            chunks, _ = packets_for(
                outputs[sample["arm"]]["output"]["content"], [fact], {}, annotation, case.jd
            )
            scripts.extend(
                ChatModelResult(
                    content=json.dumps(synthetic_response(c)),
                    usage=ModelUsage(input_tokens=10, output_tokens=10),
                )
                for c in chunks
            )
        rig.factory = LLMFactory(
            recorder=recorder,
            chat_adapter=ScriptedFakeChatModel(scripts),
            embedding_adapter=FakeEmbeddingModel(),
            provider="fake",
        )
        await score_block(rig, directory, block, case, annotation)
        scored_usage = await recorder.measurement()
        assert scored_usage["attempts"] == 4 + len(scripts)
        from tests.evals.product_acceptance_contracts import read_private_json

        assert all(
            read_private_json(directory / s["sample_id"] / "result.json")["score_status"]
            == "ASSESSED"
            for s in block
        )
        await score_block(rig, directory, block, case, annotation)
        assert await recorder.measurement() == scored_usage
        # Reuse actual published PG versions and ledger rows in another source directory.
        import hashlib

        from tests.evals import resume_initial_reuse
        from tests.evals.resume_initial import usage_for

        publish(
            directory / "test-r1-session.json",
            {"session_id": str(created.receipt.resource_id), "run_id": str(created.receipt.run_id)},
        )
        publish(directory / "test-r1-command.json", {"id": str(key)})
        publish(directory / "annotations.json", {"test": annotation})
        audit = {
            "artifact_sha256": {
                str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*.json")
            }
        }
        monkeypatch.setattr(
            resume_initial_reuse,
            "verify_origin",
            lambda *args: (tmp_path, "synthetic-fingerprint", audit),
        )
        target = tmp_path / "new-source"
        target.mkdir(mode=0o700)
        monkeypatch.setattr(
            resume_initial_reuse,
            "generation_lock_evidence",
            lambda ref="HEAD": {"sha256": "synthetic-lock", "compatible_sha256": "synthetic-lock"},
        )
        before_reuse = await recorder.measurement()
        await resume_initial_reuse.reuse_pilot(
            rig, tmp_path, target, "a" * 40, {"pilot": block}, usage_for
        )
        assert await recorder.measurement() == before_reuse
        assert not list((target / "pilot").glob("*/result.json"))
        await resume_initial_reuse.reuse_pilot(
            rig, tmp_path, target, "a" * 40, {"pilot": block}, usage_for
        )
        assert await recorder.measurement() == before_reuse
        for sample in block:
            assert (
                await generate_sample(rig, sample, inputs, target / "pilot" / sample["sample_id"])
                == outputs[sample["arm"]]
            )
        assert await recorder.measurement() == before_reuse
        # Preserve every compatible paid batch without any fresh calls.
        from tests.evals.resume_initial_scoring import VERSION as SCORE_VERSION

        coverage_script = []
        for packet in packets:
            sample = next(s for s in block if s["sample_id"] == mapping[packet["blind_id"]])
            chunks, _ = packets_for(
                outputs[sample["arm"]]["output"]["content"], [fact], {}, annotation, case.jd
            )
            coverage_script.extend(
                ChatModelResult(
                    content=json.dumps(synthetic_response(c)),
                    usage=ModelUsage(input_tokens=10, output_tokens=10),
                )
                for c in chunks
                if c["kind"] == "coverage"
            )
        rig.factory = LLMFactory(
            recorder=recorder,
            chat_adapter=ScriptedFakeChatModel(coverage_script),
            embedding_adapter=FakeEmbeddingModel(),
            provider="fake",
        )
        await score_block(
            rig,
            target / "pilot",
            block,
            case,
            annotation,
            {"phase": directory, "version": SCORE_VERSION},
        )
        after_score_reuse = await recorder.measurement()
        assert after_score_reuse["attempts"] == before_reuse["attempts"]
        for sample in block:
            saved = read_private_json(target / "pilot" / sample["sample_id"] / "result.json")
            assert saved["score_status"] == "ASSESSED"
            assert any(b["reused"] for b in saved["scoring_batches"])
        await score_block(
            rig,
            target / "pilot",
            block,
            case,
            annotation,
            {"phase": directory, "version": SCORE_VERSION},
        )
        assert await recorder.measurement() == after_score_reuse
        # A partial formal phase can reuse audited artifacts and real PG identities.
        formal = tmp_path / "formal"
        formal.mkdir(mode=0o700)
        for old_path in directory.rglob("*.json"):
            new_path = formal / old_path.relative_to(directory)
            new_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            publish(new_path, read_private_json(old_path))
        inventory = {
            "formal/" + str(p.relative_to(formal)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in formal.rglob("*.json")
        }
        monkeypatch.setattr(
            resume_initial_reuse, "verify_formal_origin", lambda *args: (tmp_path, inventory)
        )
        monkeypatch.setattr(
            resume_initial_reuse,
            "score_origin",
            lambda *args: {"phase": formal, "version": SCORE_VERSION},
        )
        score_manifest = {
            "planned": block,
            "scoring_version": SCORE_VERSION,
            "scoring_prompt": {"claims": "synthetic"},
            "scoring_transport": {"deadline_seconds": 300, "max_attempts": 1},
        }
        publish(tmp_path / "manifest.json", score_manifest)
        formal_target = tmp_path / "formal-target"
        formal_target.mkdir(mode=0o700)
        score_source = await resume_initial_reuse.reuse_formal(
            rig, tmp_path, formal_target, "a" * 40, score_manifest, usage_for
        )
        await score_block(rig, formal_target / "formal", block, case, annotation, score_source)
        assert await recorder.measurement() == after_score_reuse
        assert all(
            read_private_json(formal_target / "formal" / s["sample_id"] / "result.json")[
                "score_status"
            ]
            == "ASSESSED"
            for s in block
        )
        await resume_initial_reuse.reuse_formal(
            rig, tmp_path, formal_target, "a" * 40, score_manifest, usage_for
        )
        assert await recorder.measurement() == after_score_reuse
        # A terminal unresolved score is retained exactly, never paid for again.
        unresolved_path = formal / block[0]["sample_id"] / "result.json"
        unresolved = read_private_json(unresolved_path)
        unresolved["score_status"] = "UNRESOLVED"
        unresolved_path.write_text(json.dumps(unresolved))
        inventory["formal/" + str(unresolved_path.relative_to(formal))] = hashlib.sha256(
            unresolved_path.read_bytes()
        ).hexdigest()
        terminal_target = tmp_path / "terminal-target"
        terminal_target.mkdir(mode=0o700)
        await resume_initial_reuse.reuse_formal(
            rig, tmp_path, terminal_target, "a" * 40, score_manifest, usage_for
        )
        await score_block(rig, terminal_target / "formal", block, case, annotation, score_source)
        assert (
            read_private_json(terminal_target / "formal" / block[0]["sample_id"] / "result.json")
            == unresolved
        )
        assert await recorder.measurement() == after_score_reuse
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
