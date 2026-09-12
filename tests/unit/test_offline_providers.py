import json
import socket

import pytest
from pydantic import ValidationError

from app.llm.fake import (
    FakeChatModel,
    FakeEmbeddingModel,
    ScriptedFakeChatModel,
    ScriptedFakeFailure,
    ScriptedFakeProviderError,
)
from app.llm.ports import ChatMessage, ChatModelResult, ModelToolSchema
from app.tools.fake_search import FakeSearch

PROVIDER_SECRET_NAMES = (
    "DASHSCOPE_API_KEY",
    "TAVILY_API_KEY",
    "LANGFUSE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
)


@pytest.mark.asyncio
async def test_fake_providers_do_not_use_network_or_expose_environment_secrets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "pf-step06-secret-canary"
    for name in PROVIDER_SECRET_NAMES:
        monkeypatch.setenv(name, canary)

    def deny_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline fake attempted network access")

    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)

    chat_result = await FakeChatModel().invoke(
        (ChatMessage(role="user", content="Synthetic request"),),
        (
            ModelToolSchema(
                name="synthetic_tool",
                description="A schema carrying opaque untrusted metadata",
                input_schema={"description": canary},
            ),
        ),
        {"opaque": canary},
    )
    embedding_result = await FakeEmbeddingModel().embed((canary,), {"opaque": canary})
    search_result = await FakeSearch({}, clock=lambda: 10.0).search(
        canary,
        max_results=1,
        deadline=11.0,
    )
    scripted_model = ScriptedFakeChatModel(
        (
            ChatModelResult(content="Offline scripted response."),
            ScriptedFakeFailure(kind="timeout"),
            ScriptedFakeFailure(kind="provider_error"),
        )
    )
    scripted_result = await scripted_model.invoke(
        (ChatMessage(role="user", content=canary),),
        (
            ModelToolSchema(
                name="synthetic_tool",
                description="A schema carrying opaque untrusted metadata",
                input_schema={"description": canary},
            ),
        ),
        {"opaque": canary},
    )
    with pytest.raises(TimeoutError) as scripted_timeout:
        await scripted_model.invoke(
            (ChatMessage(role="user", content=canary),),
            (),
            {"opaque": canary},
        )
    with pytest.raises(ScriptedFakeProviderError) as scripted_provider_error:
        await scripted_model.invoke(
            (ChatMessage(role="user", content=canary),),
            (),
            {"opaque": canary},
        )

    with pytest.raises(ValueError) as invalid_metadata:
        await FakeChatModel().invoke(
            (ChatMessage(role="user", content="Synthetic request"),),
            (),
            {"opaque": canary, "invalid": object()},  # type: ignore[dict-item]
        )
    with pytest.raises(ValidationError) as invalid_scripted_failure:
        ScriptedFakeFailure.model_validate({"kind": "timeout", "message": canary})

    captured = capsys.readouterr()
    observable_surface = "\n".join(
        (
            repr(chat_result),
            repr(embedding_result),
            repr(search_result),
            repr(scripted_model),
            repr(scripted_result),
            chat_result.model_dump_json(),
            embedding_result.model_dump_json(),
            scripted_result.model_dump_json(),
            json.dumps([result.model_dump(mode="json") for result in search_result]),
            str(scripted_timeout.value),
            str(scripted_provider_error.value),
            str(invalid_metadata.value),
            str(invalid_scripted_failure.value),
            caplog.text,
            captured.out,
            captured.err,
        )
    )

    assert canary not in observable_surface
