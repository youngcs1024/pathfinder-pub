"""Bounded HTTP/SSE transport with distinct received and application-processed cursors."""

from __future__ import annotations

import asyncio
from datetime import datetime
from time import monotonic
from uuid import UUID

import httpx

from tests.performance.capacity_contracts import Connection
from tests.performance.smoke import HTTP, SmokeFailure, parse_events, require


class CapacityHTTP(HTTP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sse_bytes = 0

    def reserve(self):
        require(self.count < 512, "request_limit")
        self.count += 1


class Parser:
    def __init__(self):
        self.pending = bytearray()
        self.total = 0

    def feed(self, chunk):
        self.total += len(chunk)
        require(self.total <= 2 * 1024 * 1024)
        self.pending.extend(chunk)
        frames = []
        while b"\n\n" in self.pending:
            raw, _, tail = self.pending.partition(b"\n\n")
            require(len(raw) <= 65536)
            self.pending = bytearray(tail)
            frames.extend(parse_events(raw.decode()))
        require(len(self.pending) <= 65536)
        return frames

    def finish(self):
        require(not self.pending.strip())


class Consumer:
    def __init__(self, http, run_id, path, records, *, cursor=0, delay=0.0):
        self.http = http
        self.path = path
        self.delay = delay
        self.records = records
        self.index = len(records)
        self.record = Connection(ordinal=self.index, run_id=run_id, cursor=cursor)
        self.received = []
        self.processed = []
        records.append(self.record)
        self.established = asyncio.Event()

    @property
    def cursor(self):
        return self.processed[-1] if self.processed else self.record.cursor

    def update(self, **fields):
        self.record = Connection.model_validate({**self.record.model_dump(), **fields})
        self.records[self.index] = self.record

    async def run(self):
        self.http.reserve()
        parser = Parser()
        token = self.http.metrics.begin(
            "sse_connection", cursor=self.cursor, run_id=self.record.run_id
        )
        outcome = "failed"
        try:
            async with self.http.client.stream(
                "GET", self.path, headers={"Last-Event-ID": str(self.cursor)}
            ) as response:
                self.http.metrics.update(token, http_status=response.status_code)
                require(response.status_code == 200, "http_failed")
                self.update(established_at=monotonic())
                self.established.set()
                async for chunk in response.aiter_bytes():
                    self.http.sse_bytes += len(chunk)
                    require(self.http.sse_bytes <= 32 * 1024 * 1024)
                    frames = parser.feed(chunk)
                    for frame in frames:
                        seq = frame["id"]
                        require(UUID(frame["data"]["run_id"]) == self.record.run_id)
                        require(
                            seq == (self.received[-1] if self.received else self.record.cursor) + 1
                        )
                        require(len(self.received) < 256)
                        self.received.append(seq)
                        sample = self.http.metrics.begin(
                            "sse_event",
                            run_id=self.record.run_id,
                            seq=seq,
                            cursor=self.record.cursor,
                            event_recorded_at=datetime.fromisoformat(frame["data"]["occurred_at"]),
                        )
                        self.http.metrics.end(sample)
                    self.update(received=tuple(self.received))
                    for frame in frames:
                        if self.delay:
                            await asyncio.sleep(self.delay)
                        self.processed.append(frame["id"])
                        self.update(processed=tuple(self.processed))
                        if frame["event"] in {"run.completed", "run.failed", "run.cancelled"}:
                            self.update(terminal=True)
                parser.finish()
                outcome = "closed"
        except asyncio.CancelledError:
            outcome = "disconnected"
            raise
        except (httpx.HTTPError, ValueError, UnicodeError, KeyError, TypeError):
            raise SmokeFailure("http_failed") from None
        finally:
            self.update(closed_at=monotonic(), outcome=outcome)
            self.http.metrics.end(
                token,
                "succeeded"
                if outcome == "closed"
                else ("cancelled" if outcome == "disconnected" else "failed"),
            )


async def close_consumers(tasks):
    for task in tasks:
        if not task.done():
            task.cancel()
    results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result
