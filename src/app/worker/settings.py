from dataclasses import dataclass
from math import isfinite
from pathlib import Path

WORKER_READY_PATH = Path("/tmp/pathfinder-worker-ready")


@dataclass(frozen=True, slots=True)
class WorkerRuntimeSettings:
    poll_seconds: float = 0.5
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    shutdown_grace_seconds: float = 15.0
    execution_timeout_seconds: float = 300.0
    stale_batch_size: int = 10
    approval_expiry_batch_size: int = 10
    action_recovery_max_attempts: int = 3
    retry_base_seconds: float = 1.0
    retry_cap_seconds: float = 30.0
    retry_jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        positive_values = (
            self.poll_seconds,
            self.lease_seconds,
            self.heartbeat_seconds,
            self.shutdown_grace_seconds,
            self.execution_timeout_seconds,
            self.retry_base_seconds,
            self.retry_cap_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
            or value <= 0
            for value in positive_values
        ):
            raise ValueError("worker timing settings must be positive numbers")
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("worker lease must exceed two heartbeat intervals")
        if self.retry_cap_seconds < self.retry_base_seconds:
            raise ValueError("worker retry cap must not be below its base")
        if (
            isinstance(self.stale_batch_size, bool)
            or not isinstance(self.stale_batch_size, int)
            or self.stale_batch_size < 1
        ):
            raise ValueError("worker stale batch size must be positive")
        if (
            isinstance(self.approval_expiry_batch_size, bool)
            or not isinstance(self.approval_expiry_batch_size, int)
            or self.approval_expiry_batch_size < 1
        ):
            raise ValueError("worker approval expiry batch size must be positive")
        if (
            isinstance(self.action_recovery_max_attempts, bool)
            or not isinstance(self.action_recovery_max_attempts, int)
            or self.action_recovery_max_attempts < 1
        ):
            raise ValueError("worker action recovery max attempts must be positive")
        if not 0 <= self.retry_jitter_ratio <= 0.25:
            raise ValueError("worker retry jitter ratio must be between zero and 0.25")
