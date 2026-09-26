"""Keep private model outputs without bypassing Factory attempt accounting."""

from uuid import uuid4

from tests.evals.product_acceptance_contracts import publish


class RecordingModel:
    def __init__(self, model, root):
        self.wrapped, self.root = model, root
        self.model = model.model

    async def invoke(self, messages, tools, metadata):
        result = await self.wrapped.invoke(messages, tools, metadata)
        publish(
            self.root / f"model-response-{uuid4().hex}.json",
            {
                "metadata": dict(metadata),
                "finish_status": result.finish_status,
                "content": result.content,
                "tool_calls_present": bool(result.tool_calls),
            },
        )
        return result


class RecordingFactory:
    def __init__(self, factory, root):
        self.factory, self.root = factory, root

    def create_chat_model(self, context):
        return RecordingModel(self.factory.create_chat_model(context), self.root)

    def create_embedding_model(self, context):
        return self.factory.create_embedding_model(context)
