"""E4.7 operation safeguards; never starts Docker or contacts providers."""

from decimal import Decimal

import pytest

from tests.evals import quality_pilot as pilot
from tests.evals.quality_dataset import load_quality_dataset
from tests.evals.quality_generation_fixtures import generation_factory
from tests.evals.test_quality_score import SHA


def test_fixed_manifest_and_budget():
    manifest, policy = pilot.build_manifest(
        load_quality_dataset(pilot.DATASET), SHA, generation_factory(), "e47_test"
    )
    assert len(manifest.selected_case_ids) == 15
    assert manifest.repeat_count == 1
    assert manifest.cost_admission_budget_cny == Decimal("10")
    assert (manifest.provider_attempt_cap, manifest.input_token_cap, manifest.output_token_cap) == (
        300,
        1_000_000,
        100_000,
    )
    assert policy.retrieval.unknown_attempt_reserve_cny == Decimal("0.1")
    assert policy.case_timeout_seconds == 600


@pytest.mark.parametrize(
    "content",
    [
        "",
        "DASHSCOPE_API_KEY=\nPF_QWEN_WORKSPACE_ID=valid\n",
        "DASHSCOPE_API_KEY=private-value\nPF_QWEN_WORKSPACE_ID=bad/id\n",
        "DASHSCOPE_API_KEY=private-value\nPF_QWEN_WORKSPACE_ID=valid\nOTHER=private-value\n",
    ],
)
def test_credentials_failure_has_no_content(tmp_path, content):
    tmp_path.chmod(0o700)
    path = tmp_path / "credentials.env"
    path.write_text(content)
    path.chmod(0o600)
    with pytest.raises(pilot.PilotError, match="credentials_missing_or_invalid") as error:
        pilot.credentials(path)
    assert "private-value" not in str(error.value)


def test_preflight_missing_credentials_never_reaches_docker(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)

    def forbidden(*args, **kwargs):
        pytest.fail("Docker must not run before credential validation")

    monkeypatch.setattr(pilot, "docker", forbidden)
    with pytest.raises(pilot.PilotError, match="credentials_missing_or_invalid"):
        pilot.preflight(tmp_path)


async def test_live_requires_explicit_flag(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("preflight must not start")

    monkeypatch.setattr(pilot, "preflight", forbidden)
    with pytest.raises(pilot.PilotError, match="live_confirmation_required"):
        await pilot.run_pilot(tmp_path, confirm_live=False)


def test_database_command_has_bounded_loopback_and_no_delete(monkeypatch):
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        return "127.0.0.1:54321" if args[0] == "port" else "ok"

    monkeypatch.setattr(pilot, "docker", docker)

    class Result:
        returncode = 0

    monkeypatch.setattr(pilot.subprocess, "run", lambda *a, **kw: Result())
    url = pilot.provision_database("pf-e47-test", "pathfinder_test_" + "a" * 32, "secret")
    command = calls[0]
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == "2g"
    assert command[command.index("--publish") + 1] == "127.0.0.1::5432"
    assert "secret" not in command and "--rm" not in command
    assert "127.0.0.1:54321" in url


def test_public_error_sanitized(tmp_path, capsys, monkeypatch):
    def failure(*args):
        raise RuntimeError("private-secret-marker")

    monkeypatch.setattr(pilot, "preflight", failure)
    assert pilot.main(["preflight", "--root", str(tmp_path)]) == 2
    assert "private-secret-marker" not in capsys.readouterr().out
