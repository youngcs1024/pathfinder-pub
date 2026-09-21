"""Real pg_dump/pg_restore evidence, executed by the existing CI integration shards."""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from scripts.ci_reports import RESTORE_ERRORS
from tests.integration.restore_product import (
    product,
    reject_downgrade,
    seed_current,
    seed_legacy,
    verify_restored,
)
from tests.integration.restore_support import (
    HEAD,
    IMAGE,
    OLD_HEAD,
    Rehearsal,
    RestoreError,
    canonical,
    consistency_snapshot,
    digest,
    require,
)

pytestmark = pytest.mark.integration


async def test_real_upgrade_dump_restore_and_resume(tmp_path, monkeypatch, request):
    report = {
        "status": "IN_PROGRESS",
        "stage": "environment",
        "checks": {},
        "old_revision": OLD_HEAD,
        "revision": HEAD,
        "graph": "pathfinder-research-v6",
        "postgres_image": IMAGE,
        "elapsed_ms": 0,
        "resources_stopped": False,
        "tables": {},
        "failure_category": None,
    }
    request.node.user_properties.append(("restore", report))
    rehearsal = Rehearsal(tmp_path / "e84")
    failed = False
    try:
        async with asyncio.timeout(600):
            source = rehearsal.start("source")
            target = rehearsal.start("target")
            report["postgres_image_id"] = source.image_id
            require(source.image_id == target.image_id, "image_mismatch")
            report["stage"] = "upgrade"
            legacy = seed_legacy(source)
            report["checks"]["legacy_upgrade"] = True
            report["stage"] = "fixtures"
            original = product(source, rehearsal.directory / "source-private")
            await seed_current(original, monkeypatch, rehearsal)
            report["checks"]["current_fixtures"] = True
            report["stage"] = "downgrade_guard"
            reject_downgrade(source)
            report["checks"]["keyed_downgrade_refused"] = True
            report["stage"] = "backup"
            pending = (original.runs["approve"], original.runs["reject"])
            probe = consistency_snapshot(
                source, original.runs["success"], original.document_id, pending
            )
            require(probe["fixture"]["run"]["client_request_id"] is not None, "identity_missing")
            before = source.snapshot()
            dump, checksum = source.dump()
            require(source.snapshot() == before, "source_changed")
            report["dump_sha256"], report["dump_bytes"] = checksum, dump.stat().st_size
            report["checks"]["stable_backup"] = True
            report["stage"] = "restore"
            corrupt = rehearsal.directory / "corrupt.dump"
            descriptor = os.open(corrupt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(dump.read_bytes()[:32])
            with pytest.raises(RestoreError, match=r"^command_failed$"):
                target.validate_dump(corrupt)
            with pytest.raises(RestoreError, match=r"^checksum_mismatch$"):
                target.restore(dump, "0" * 64, before)
            report["checks"]["invalid_backup_refused"] = True
            restored = target.restore(dump, checksum, before)
            report["tables"] = restored
            report["snapshot_sha256"] = digest(canonical(restored))
            report["checks"]["exact_restore"] = True
            with pytest.raises(RestoreError, match=r"^target_not_empty$"):
                target.restore(dump, checksum, before)
            require(target.snapshot() == restored, "occupied_target_changed")
            report["checks"]["occupied_target_refused"] = True
            require(
                consistency_snapshot(
                    target, original.runs["success"], original.document_id, pending
                )
                == probe,
                "probe_changed",
            )
            report["checks"]["operational_probe"] = True
            report["stage"] = "resume"
            recovered = product(target, rehearsal.directory / "target-private")
            await verify_restored(recovered, original, legacy, report, target)
            rehearsal.remaining()
            report["stage"] = "complete"
            report["status"] = "PASS"
    except asyncio.CancelledError:
        failed = True
        raise
    except Exception as error:
        # Driver, SQL, assertion, transport and validation exceptions can contain business data.
        # The stage and completed allowlisted checks are the only public diagnostic.
        failed = True
        report["failure_category"] = (
            str(error)
            if isinstance(error, RestoreError) and str(error) in RESTORE_ERRORS
            else "unexpected_error"
        )
        raise RestoreError("restore_rehearsal_failed") from None
    finally:
        stopped = rehearsal.stop()
        report["resources_stopped"] = stopped
        report["elapsed_ms"] = round((time.monotonic() - rehearsal.started) * 1000)
        if failed or not stopped:
            report["status"] = "IN_PROGRESS"
        if not stopped and not failed:
            report["stage"] = "cleanup"
            raise RestoreError("cleanup_failed")
