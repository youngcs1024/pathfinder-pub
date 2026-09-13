"""Version 3 observations extend claim capacity without rewriting old packets."""

from typing import Literal

from pydantic import Field

from tests.performance.capacity_metrics import CapacityCollector, CapacityPacket, CapacitySample
from tests.performance.metrics import SegmentRecord


class QueueMetricSample(CapacitySample):
    claim_ordinal: int | None = Field(default=None, ge=1, le=512)


class QueuePacket(CapacityPacket):
    schema_version: Literal[3] = 3
    samples: tuple[QueueMetricSample, ...] = Field(max_length=16384)


class QueueSegment(SegmentRecord):
    schema_version: Literal[2] = 2
    sample: QueueMetricSample


class QueueCollector(CapacityCollector):
    segment_limit = 512
    sample_type = QueueMetricSample
    segment_type = QueueSegment

    def packet(self):
        return QueuePacket(
            role=self.role,
            process_id=self.process_id,
            started=self.started,
            finished=self.clock(),
            dropped=self.dropped,
            write_failed=self.write_failed,
            observer_seconds=self.observer_seconds,
            samples=tuple(self.samples),
            pool_checkouts=self.checkouts,
            pool_peak=self.peak,
            pool_remaining=self.checked_out,
        )
