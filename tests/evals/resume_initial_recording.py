"""Create-only model response journal: replay saved responses, never repeat an uncertain call."""

import json
from time import monotonic

from app.llm.factory import LLMProviderError
from app.llm.invocations import LOCKED_CHAT_MODEL, chat_request_hash
from app.llm.ports import ChatModelResult
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest


class EstimatedGenerationFailure(Exception):
    """A verified terminal provider failure; never replay its request."""


async def terminal_estimate(recorder, started, identity, messages, tools, metadata):
    from tests.evals.resume_initial_estimates import estimated_failure

    saved = read_private_json(started)
    require("identity" in saved and "before" in saved, "uncertain_call_requires_reconciliation")
    require(saved["identity"] == identity, "response_identity_changed")
    usage = await recorder.check_admission(after=True)
    request_hash = chat_request_hash(model=LOCKED_CHAT_MODEL, messages=messages, tools=tools)
    rows = [
        r
        for r in await recorder.audit()
        if str(r.id) not in saved["before"]["invocation_ids"]
        and r.request_hash == request_hash
        and r.graph_node == metadata["graph_node"]
    ]
    require(
        1 <= len(rows) <= 3 and all(estimated_failure(usage, row.id) for row in rows),
        "uncertain_call_requires_reconciliation",
    )
    return str(rows[0].id)


class JournalModel:
    def __init__(self, model, directory, recorder):
        self.wrapped, self.directory, self.recorder = model, directory, recorder
        self.model, self.ordinal = model.model, 0

    async def invoke(self, messages, tools, metadata):
        ordinal = self.ordinal
        self.ordinal += 1
        identity = quality_identity_digest(
            {
                "messages": [m.model_dump(mode="json") for m in messages],
                "tools": [t.model_dump(mode="json") for t in tools],
                "metadata": dict(metadata),
            }
        )
        started = self.directory / f"call-{ordinal:02}-started.json"
        response = self.directory / f"call-{ordinal:02}-response.json"
        if response.exists():
            saved = read_private_json(response)
            require(saved["identity"] == identity, "response_identity_changed")
            await self.recorder.check_admission(after=True)
            return ChatModelResult.model_validate_json(json.dumps(saved["response"]))
        if started.exists():
            await terminal_estimate(self.recorder, started, identity, messages, tools, metadata)
            raise EstimatedGenerationFailure()

        before = await self.recorder.check_admission()
        publish(started, {"identity": identity, "before": before})
        began = monotonic()
        try:
            result = await self.wrapped.invoke(messages, tools, metadata)
        except LLMProviderError:
            await terminal_estimate(self.recorder, started, identity, messages, tools, metadata)
            raise EstimatedGenerationFailure() from None
        after = await self.recorder.check_admission(after=True)
        publish(
            response,
            {
                "identity": identity,
                "response": result.model_dump(mode="json"),
                "elapsed_seconds": monotonic() - began,
                "invocation_ids": sorted(
                    set(after["invocation_ids"]) - set(before["invocation_ids"])
                ),
            },
        )
        return result


class JournalFactory:
    def __init__(self, factory, directory, recorder):
        self.factory, self.directory, self.recorder = factory, directory, recorder

    def create_chat_model(self, context):
        return JournalModel(self.factory.create_chat_model(context), self.directory, self.recorder)
