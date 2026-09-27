"""Read-only HTTP verification in an isolated fake/off API container; no model calls."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request


def verify(expected, fetch):
    fixture = expected["fixture"]
    workspace, session = fixture["workspace_id"], fixture["session_id"]
    me = json.loads(fetch("/api/v1/me")[0])
    if me["user_id"] != fixture["owner_user_id"] or not any(
        w["workspace_id"] == workspace for w in me["workspaces"]
    ):
        raise ValueError("fixture_identity_mismatch")
    base = f"/api/v2/workspaces/{workspace}/resume-sessions/{session}/versions"
    versions = json.loads(fetch(base)[0])
    if len(versions) != len(fixture["versions"]):
        raise ValueError("history_count_mismatch")
    for original in fixture["versions"]:
        actual = next(v for v in versions if v["version_id"] == original["version_id"])
        if any(actual[k] != original[k] for k in ("artifact_id", "tex_sha256", "version")):
            raise ValueError("history_identity_mismatch")
        if bool(actual["confirmation"]) != original["confirmed"]:
            raise ValueError("confirmation_mismatch")
        if actual["confirmation"] and any(
            actual["confirmation"][k] != original[k]
            for k in ("artifact_id", "tex_sha256", "version_id")
        ):
            raise ValueError("confirmation_mismatch")
        for _ in range(2):
            body, headers = fetch(base + "/" + original["version_id"] + "/download")
            if (
                hashlib.sha256(body).hexdigest() != original["tex_sha256"]
                or headers["x-content-sha256"] != original["tex_sha256"]
                or headers["x-resume-version-id"] != original["version_id"]
            ):
                raise ValueError("download_mismatch")
    return {"history_download": True, "confirmation": True, "versions": len(versions)}


def main():
    def fetch(path):
        with urllib.request.urlopen("http://127.0.0.1:8000" + path, timeout=5) as response:
            return response.read(), response.headers

    try:
        result = verify(json.loads(sys.argv[1]), fetch)
    except Exception:
        raise SystemExit("resume_restore_verification_failed") from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
