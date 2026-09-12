from dataclasses import replace

import pytest

from app.worker.settings import WorkerRuntimeSettings


def test_worker_runtime_settings_use_bounded_execution_default() -> None:
    settings = WorkerRuntimeSettings()

    assert settings.execution_timeout_seconds == 300.0
    assert settings.lease_seconds == 30.0
    assert settings.heartbeat_seconds == 10.0


def test_worker_runtime_settings_accept_execution_timeout_override() -> None:
    settings = WorkerRuntimeSettings(execution_timeout_seconds=60.0)

    assert settings.execution_timeout_seconds == 60.0


@pytest.mark.parametrize(
    "field_name",
    [
        "poll_seconds",
        "lease_seconds",
        "heartbeat_seconds",
        "shutdown_grace_seconds",
        "execution_timeout_seconds",
        "retry_base_seconds",
        "retry_cap_seconds",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [0, -1, True, False, "1", float("nan"), float("inf"), float("-inf")],
)
def test_worker_timing_settings_reject_non_finite_or_non_positive_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError, match="worker timing settings must be positive numbers"):
        replace(WorkerRuntimeSettings(), **{field_name: invalid_value})
