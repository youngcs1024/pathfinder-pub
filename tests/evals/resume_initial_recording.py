"""Create-only model response journal: replay saved responses, never repeat an uncertain call."""

import json
from time import monotonic

from app.llm.ports import ChatModelResult
from tests.evals.product_acceptance_contracts import publish, read_private_json, require
from tests.evals.quality_dataset import quality_identity_digest


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
        require(not started.exists(), "uncertain_call_requires_reconciliation")
        before = await self.recorder.check_admission()
        publish(started, {"identity": identity, "before": before})
        began = monotonic()
        result = await self.wrapped.invoke(messages, tools, metadata)
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
