"""R7.2 restored HTTP bytes and confirmation identities must agree."""

import copy
import hashlib
import json

import pytest

from scripts.resume_restore_verify import verify


def fixture():
    payload = b"synthetic tex"
    sha = hashlib.sha256(payload).hexdigest()
    version = dict(version_id="v", artifact_id="a", tex_sha256=sha, version=1, confirmed=True)
    expected = {
        "fixture": {
            "workspace_id": "w",
            "session_id": "s",
            "owner_user_id": "u",
            "versions": [version],
        }
    }
    actual = {
        **version,
        "confirmation": {k: version[k] for k in ("version_id", "artifact_id", "tex_sha256")},
    }
    responses = {
        "/api/v1/me": (
            json.dumps({"user_id": "u", "workspaces": [{"workspace_id": "w"}]}).encode(),
            {},
        ),
        "/api/v2/workspaces/w/resume-sessions/s/versions": (json.dumps([actual]).encode(), {}),
        "/api/v2/workspaces/w/resume-sessions/s/versions/v/download": (
            payload,
            {"x-content-sha256": sha, "x-resume-version-id": "v"},
        ),
    }
    return expected, responses


def test_restored_downloads_are_read_twice_and_confirmed():
    expected, responses = fixture()
    calls = []

    def fetch(path):
        calls.append(path)
        return responses[path]

    assert verify(expected, fetch) == {
        "history_download": True,
        "confirmation": True,
        "versions": 1,
    }
    assert sum(p.endswith("/download") for p in calls) == 2


@pytest.mark.parametrize("failure", ["actor", "bytes", "header", "confirmation", "count"])
def test_inconsistent_restored_product_is_refused(failure):
    expected, responses = fixture()
    responses = copy.deepcopy(responses)
    listing = "/api/v2/workspaces/w/resume-sessions/s/versions"
    download = listing + "/v/download"
    if failure == "actor":
        responses["/api/v1/me"] = (b'{"user_id":"other","workspaces":[]}', {})
    elif failure == "bytes":
        responses[download] = (b"changed", responses[download][1])
    elif failure == "header":
        responses[download][1]["x-resume-version-id"] = "other"
    else:
        versions = json.loads(responses[listing][0])
        if failure == "count":
            versions = []
        else:
            versions[0]["confirmation"]["tex_sha256"] = "0" * 64
        responses[listing] = (json.dumps(versions).encode(), {})
    with pytest.raises(ValueError):
        verify(expected, responses.__getitem__)
