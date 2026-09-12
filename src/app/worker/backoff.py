from dataclasses import dataclass
from datetime import timedelta
from random import Random

from app.worker.settings import WorkerRuntimeSettings


@dataclass(slots=True)
class ExponentialBackoff:
    settings: WorkerRuntimeSettings
    random_source: Random

    def __call__(self, attempt: int) -> timedelta:
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("retry attempt must be a positive integer")
        base_delay = min(
            self.settings.retry_cap_seconds,
            self.settings.retry_base_seconds * (2 ** (attempt - 1)),
        )
        jitter = base_delay * self.settings.retry_jitter_ratio * self.random_source.random()
        return timedelta(seconds=base_delay + jitter)
