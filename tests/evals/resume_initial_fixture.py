"""Atomic import of reviewed snapshots into the existing isolated experiment tenant."""

from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid5

from app.db import models as m
from tests.evals.product_acceptance_contracts import read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest
from tests.evals.resume_experiments import preserve


async def seed(sessions, tenant, root, directory):
    profile = read_private_json(root / "inputs/profile.json")
    facts = read_private_json(root / "inputs/facts.json")["facts"]
    projects = read_private_json(root / "inputs/r71-facts-done.json")["projects"]
    raw = (root / "inputs/resume.tex").read_bytes()
    identity = quality_identity_digest({"profile": profile, "facts": facts, "projects": projects})
    namespace = tenant.workspace_id

    def key(name):
        return uuid5(namespace, "a-frozen-fixture:" + name)

    receipt = {
        "input_digest": identity,
        "workspace_id": str(namespace),
        "profile_version_id": profile["version_id"],
        "preference_version": profile["preference_version"],
        "project_ids": [v["project_id"] for v in projects.values()],
        "fact_version_map": {f["version_id"]: f["version_id"] for f in facts},
        "source_profile_version_id": profile["version_id"],
        "fixture_kind": "reviewed_snapshot_import_no_model_extraction",
    }
    async with sessions() as db, db.begin():

        async def add(cls, ident, **values):
            values = dict(workspace_id=namespace, **values)
            existing = await db.get(cls, ident)
            if existing is None:
                db.add(cls(id=ident, **values))
                await db.flush()
            else:
                require(
                    all(getattr(existing, k) == v for k, v in values.items()),
                    "fixture_database_changed",
                )

        for alias, project in projects.items():
            pid = UUID(project["project_id"])
            await add(m.MaterialProject, pid, created_by_user_id=tenant.actor_user_id, name=alias)
            conversation, message, run = (key(alias + suffix) for suffix in (":c", ":m", ":r"))
            await add(
                m.Conversation,
                conversation,
                created_by_user_id=tenant.actor_user_id,
                title="Frozen experiment snapshot import",
            )
            await add(
                m.Message,
                message,
                conversation_id=conversation,
                actor_user_id=tenant.actor_user_id,
                role="user",
                content="Frozen snapshot fixture",
            )
            # This is explicitly a fixture import receipt, never evidence of a model execution.
            existing_run = await db.get(m.Run, run)
            now = existing_run.started_at if existing_run else datetime.now(UTC)
            await add(
                m.Run,
                run,
                created_by_user_id=tenant.actor_user_id,
                conversation_id=conversation,
                request_message_id=message,
                mode="material_preparation",
                graph_version="pathfinder-resume-v2",
                input_json={"fixture_digest": identity},
                limits_json={},
                status="completed",
                result_json={"fixture_import": True},
                started_at=now,
                finished_at=now,
            )
            sid, iid, snapshot, factset = (key(alias + s) for s in (":s", ":i", ":snap", ":fs"))
            digest = sha256((identity + alias).encode()).hexdigest()
            await add(
                m.MaterialSource,
                sid,
                project_id=pid,
                alias_name=alias,
                alias_digest=digest,
                kind="file",
            )
            await add(
                m.MaterialImport,
                iid,
                project_id=pid,
                run_id=run,
                source_ids=[str(sid)],
                cache_digest=digest,
            )
            await add(
                m.MaterialSnapshot,
                snapshot,
                import_id=iid,
                source_id=sid,
                source_revision=identity,
                manifest_digest=digest,
                cache_digest=digest,
                inventory_json={"frozen_source": identity},
            )
            await add(
                m.MaterialFactSet,
                factset,
                project_id=pid,
                import_id=iid,
                cache_digest=digest,
                extractor_digest=digest,
                complete=True,
                issues_json=[],
            )
            selected = [f for f in facts if f["version_id"] in project["fact_version_ids"]]
            require(len(selected) == len(project["fact_version_ids"]), "fixture_facts_missing")
            for ordinal, fact in enumerate(selected):
                fid, vid = UUID(fact["id"]), UUID(fact["version_id"])
                await add(
                    m.MaterialFact,
                    fid,
                    fact_set_id=factset,
                    ordinal=ordinal,
                    current_version=fact["version"],
                )
                await add(
                    m.MaterialFactVersion,
                    vid,
                    fact_id=fid,
                    version=fact["version"],
                    claim=fact["claim"],
                    kind=fact["kind"],
                    conditions_json=fact["conditions"],
                    review_status="confirmed",
                    issues_json=fact["issues"],
                    created_by_user_id=tenant.actor_user_id,
                )
                for n, evidence in enumerate(fact["evidence"]):
                    body = (root / "inputs" / evidence["path"]).read_bytes()
                    file_id = key(alias + ":file:" + evidence["path"])
                    await add(
                        m.MaterialSnapshotFile,
                        file_id,
                        snapshot_id=snapshot,
                        path=evidence["path"],
                        content=body,
                        content_digest=sha256(body).hexdigest(),
                    )
                    await add(
                        m.MaterialFactEvidence,
                        key(f"{vid}:e:{n}"),
                        fact_version_id=vid,
                        snapshot_file_id=file_id,
                        start_line=evidence["start_line"],
                        end_line=evidence["end_line"],
                        quote=evidence["quote"],
                    )
        pid, vid, imported = (
            UUID(profile["profile_id"]),
            UUID(profile["version_id"]),
            key("profile"),
        )
        await add(
            m.ResumeProfile,
            pid,
            owner_user_id=tenant.actor_user_id,
            current_version=profile["version"],
            current_preference_version=profile["preference_version"],
        )
        from app.resume.template_import import TEMPLATE_COMMIT

        await add(
            m.ResumeProfileImport,
            imported,
            profile_id=pid,
            template_commit=TEMPLATE_COMMIT,
            source_sha256=sha256(raw).hexdigest(),
            source_bytes=raw,
        )
        await add(
            m.ResumeProfileVersion,
            vid,
            profile_id=pid,
            source_import_id=imported,
            version=profile["version"],
            content_json=profile["content"],
            created_by_user_id=tenant.actor_user_id,
        )
        await add(
            m.ResumePreferenceVersion,
            key("preferences"),
            profile_id=pid,
            version=profile["preference_version"],
            preferences_json=profile["preferences"],
            created_by_user_id=tenant.actor_user_id,
        )
        for claim in profile["claims"]:
            cid = UUID(claim["id"])
            await add(
                m.ResumeSourceClaim,
                cid,
                source_import_id=imported,
                project_item_id=UUID(claim["project_item_id"]),
                item_id=UUID(claim["item_id"]),
                field=claim["field"],
                claim_text=claim["text"],
                source_json=claim["source"],
                review_version=claim["review_version"],
            )
            if claim["review_version"]:
                rid = key(str(cid) + ":review")
                await add(
                    m.ResumeClaimReview,
                    rid,
                    claim_id=cid,
                    version=claim["review_version"],
                    decision=claim["decision"],
                    actor_user_id=tenant.actor_user_id,
                    project_id=UUID(claim["project_id"]) if claim["project_id"] else None,
                )
                for fact_id in claim["fact_version_ids"]:
                    await add(
                        m.ResumeClaimFactLink,
                        key(str(rid) + fact_id),
                        review_id=rid,
                        fact_version_id=UUID(fact_id),
                    )
    preserve(directory / "fixture.json", receipt)
    return receipt
