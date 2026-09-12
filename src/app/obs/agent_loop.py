from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.contracts import AgentLoopObservationV1
from app.obs.logging import get_logger


@dataclass(frozen=True, slots=True)
class StructlogAgentLoopObserver:
    """Write content-free Agent loop observations through the safe JSON logger."""

    _logger: Any = field(default_factory=lambda: get_logger("app.agent_loop"), repr=False)

    def observe(self, observation: AgentLoopObservationV1) -> None:
        fields = observation.model_dump(mode="json", exclude_none=True)
        event = fields.pop("event")
        self._logger.info(event, **fields)
