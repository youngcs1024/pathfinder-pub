"""Controlled I/O below the real Factory/Registry, used only in owned test processes."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path
from random import Random
from time import monotonic
from unittest.mock import patch

from app.llm.fake import FakeEmbeddingModel
from app.llm.ports import ModelToolCall, ProviderAdapterError
from app.tools.adapters.mock_portal import (
    MockPortalHTTPAdapter,
    MockPortalRejectedError,
    MockPortalTransportError,
)
from app.tools.fake_search import FakeSearch
from app.tools.search import SearchPermanentError, SearchTransientError, normalize_search_result
from app.worker.fake_research_adapter import DeterministicResearchFakeChatAdapter
from tests.performance.environment import EnvironmentError
from tests.performance.workload import (
    KINDS,
    CallProfile,
    CallRecord,
    QueueCallRecord,
    parse_profile,
    publish,
)

QUERY = "Compare the synthetic Python backend role and prepare a grounded application"
RESUME = "# Synthetic Resume\n\nThe synthetic candidate builds Python APIs and PostgreSQL services."


class CallLimitReached(asyncio.CancelledError):
    """Stop graph execution instead of allowing factory retry to exceed the harness bound."""


class Calls:
    def __init__(self, policy: CallProfile, *, directory: Path | None = None, process="worker"):
        self.policy = parse_profile(policy.model_dump(mode="json"))
        self.directory = directory
        if process not in {"worker", "ingest"}:
            raise ValueError("invalid_process")
        self.process = process
        self.counts = Counter()
        self.records: list[CallRecord] = []
        self.random = {kind: Random(f"e53-v1:{self.policy.seed}:{kind}") for kind in KINDS}

    def record(self, record: CallRecord):
        if self.directory is not None:
            publish(
                self.directory,
                f"calls-{self.process}-{record.sequence:03}-{record.phase}.json",
                record,
            )
        self.records.append(record)

    async def invoke(self, kind, operation, *, invocation_id=None, action_id=None, timeout=None):
        if kind not in KINDS:
            raise ValueError("invalid_call")
        limit = 1 if self.process == "ingest" else self.policy.max_calls
        if sum(self.counts.values()) >= limit:
            raise CallLimitReached("call_limit")
        self.counts[kind] += 1
        ordinal = self.counts[kind]
        delay = self.policy.delays[kind]
        seconds = self.random[kind].uniform(delay.minimum, delay.maximum)
        record_type = QueueCallRecord if self.policy.schema_version == 2 else CallRecord
        record = record_type(
            process=self.process,
            sequence=sum(self.counts.values()),
            call=kind,
            ordinal=ordinal,
            invocation_id=invocation_id,
            action_id=action_id,
            delay_seconds=seconds,
            phase="started",
            outcome="pending",
            elapsed_seconds=0.0,
        )
        self.record(record)
        started = monotonic()
        outcome = "failed"
        cancellation = None
        fault = next(
            (
                item.kind
                for item in self.policy.faults
                if (item.call, item.ordinal) == (kind, ordinal)
            ),
            None,
        )
        try:
            async with asyncio.timeout(timeout):
                await asyncio.sleep(seconds)
                if fault and fault != "response_lost":
                    self.fail(kind, fault)
                result = await operation()
                if fault == "response_lost":
                    outcome = "response_lost"
                    raise MockPortalTransportError
                outcome = "succeeded"
                return result
        except asyncio.CancelledError as exc:
            cancellation = exc
            outcome = "cancelled"
            raise
        except TimeoutError:
            outcome = "timeout"
            raise
        finally:
            try:
                self.record(
                    record.model_copy(
                        update={
                            "phase": "finished",
                            "outcome": outcome,
                            "elapsed_seconds": max(0.0, monotonic() - started),
                        }
                    )
                )
            except EnvironmentError:
                if cancellation is not None:
                    raise cancellation from None
                raise

    @staticmethod
    def fail(kind, fault):
        if kind in {"chat", "embedding"}:
            raise ProviderAdapterError(
                category={
                    "timeout": "provider_timeout",
                    "transient": "provider_unavailable",
                    "permanent": "provider_rejected",
                }[fault],
                retryable=fault != "permanent",
            )
        if kind == "search":
            if fault == "permanent":
                raise SearchPermanentError(category="provider_rejected")
            raise SearchTransientError(
                category="provider_timeout" if fault == "timeout" else "provider_unavailable"
            )
        if fault == "permanent":
            raise MockPortalRejectedError
        raise MockPortalTransportError


class SyntheticChat(DeterministicResearchFakeChatAdapter):
    """Gate 8 fixture pattern: both production research tools when a resume is available."""

    async def invoke(self, messages, tools, metadata, *, attempt=None):
        if metadata.get("graph_node") == "research_agent" and not any(
            message.role == "tool" for message in messages
        ):
            payload = json.loads(
                (messages[-1].content or "")
                .split("<untrusted_research_state>\n", 1)[1]
                .split("\n</untrusted_research_state>", 1)[0]
            )
            if payload.get("document_scope_available") is True:
                query = payload["plan"]["queries"][0]
                number = metadata.get("research_pass_number", "1")
                return self._result(
                    tool_calls=(
                        ModelToolCall(
                            call_id=f"e53-search-{number}",
                            name="search_web",
                            arguments={"query": query, "max_results": 8},
                        ),
                        ModelToolCall(
                            call_id=f"e53-rag-{number}",
                            name="retrieve_documents",
                            arguments={"query": query},
                        ),
                    )
                )
        return await super().invoke(messages, tools, metadata, attempt=attempt)


class Chat:
    provider = SyntheticChat.provider
    model = SyntheticChat.model

    def __init__(self, calls: Calls):
        self.calls = calls
        self.delegate = SyntheticChat()

    async def invoke(self, messages, tools, metadata, *, attempt):
        return await self.calls.invoke(
            "chat",
            lambda: self.delegate.invoke(messages, tools, metadata, attempt=attempt),
            invocation_id=attempt.invocation_id,
            timeout=attempt.timeout_seconds,
        )


class Embedding:
    provider = FakeEmbeddingModel.provider
    model = FakeEmbeddingModel.model

    def __init__(self, calls: Calls):
        self.calls = calls
        self.delegate = FakeEmbeddingModel()

    async def embed(self, texts, metadata, *, attempt):
        return await self.calls.invoke(
            "embedding",
            lambda: self.delegate.embed(texts, metadata, attempt=attempt),
            invocation_id=attempt.invocation_id,
            timeout=attempt.timeout_seconds,
        )


class Search:
    def __init__(self, calls: Calls):
        self.calls = calls
        self.delegate = FakeSearch(
            {
                QUERY: (
                    normalize_search_result(
                        title="Synthetic Python role",
                        url="https://example.test/jobs/e53",
                        snippet="The synthetic role requires Python APIs and PostgreSQL services.",
                        published_at="2026-09-12",
                    ),
                )
            }
        )

    async def search(self, query, max_results, deadline):
        return await self.calls.invoke(
            "search",
            lambda: self.delegate.search(query, max_results, deadline),
            timeout=max(0.0, deadline - monotonic()),
        )


class Mock:
    def __init__(self, client, calls: Calls):
        self.calls = calls
        self.delegate = MockPortalHTTPAdapter(client)

    async def submit(self, **kwargs):
        return await self.calls.invoke(
            "mock_submit",
            lambda: self.delegate.submit(**kwargs),
            action_id=kwargs["action_intent_id"],
            timeout=10.0,
        )

    async def get_by_idempotency_key(self, **kwargs):
        return await self.calls.invoke(
            "mock_lookup",
            lambda: self.delegate.get_by_idempotency_key(**kwargs),
            action_id=kwargs["action_intent_id"],
            timeout=10.0,
        )


@contextmanager
def worker_adapters(worker, calls: Calls):
    # Invoked only inside the dedicated child, never as a pytest-wide monkeypatch.
    with ExitStack() as stack:
        for name, factory in {
            "DeterministicResearchFakeChatAdapter": lambda: Chat(calls),
            "FakeEmbeddingModel": lambda: Embedding(calls),
            "FakeSearch": lambda _fixtures: Search(calls),
            "MockPortalHTTPAdapter": lambda client: Mock(client, calls),
        }.items():
            stack.enter_context(patch.object(worker, name, factory))
        yield
