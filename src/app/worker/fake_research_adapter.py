from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from unicodedata import normalize

from app.llm.ports import (
    LOCKED_CHAT_MODEL,
    ChatMessage,
    ChatModelResult,
    ModelToolCall,
    ModelToolSchema,
    ModelUsage,
    ProviderAttemptContext,
)


def _delimited_json(content: str, start: str, end: str) -> dict[str, object]:
    prefix = content.find(start)
    suffix = content.find(end, prefix + len(start))
    if prefix < 0 or suffix < 0:
        raise ValueError("fake research input delimiter is missing")
    payload = json.loads(content[prefix + len(start) : suffix].strip())
    if not isinstance(payload, dict):
        raise ValueError("fake research input must be an object")
    return payload


class DeterministicResearchFakeChatAdapter:
    provider = "fake"
    model = LOCKED_CHAT_MODEL

    async def invoke(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ModelToolSchema],
        metadata: Mapping[str, str],
        *,
        attempt: ProviderAttemptContext | None = None,
    ) -> ChatModelResult:
        if not messages or (
            attempt is not None and not isinstance(attempt, ProviderAttemptContext)
        ):
            raise ValueError("fake research invocation is invalid")
        graph_node = metadata.get("graph_node")
        if graph_node == "plan":
            payload = _delimited_json(
                messages[-1].content or "",
                "<untrusted_research_request>",
                "</untrusted_research_request>",
            )
            query = payload.get("normalized_query")
            if not isinstance(query, str):
                raise ValueError("fake research query is invalid")
            normalized_query = " ".join(normalize("NFKC", query).split())
            content = json.dumps(
                {"queries": [normalized_query]},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            return self._result(content=content)

        if graph_node == "research_agent":
            tool_messages = [message for message in messages if message.role == "tool"]
            if tool_messages:
                return self._result(content="The bounded fake research pass is complete.")
            payload = _delimited_json(
                messages[-1].content or "",
                "<untrusted_research_state>",
                "</untrusted_research_state>",
            )
            plan = payload.get("plan")
            queries = plan.get("queries") if isinstance(plan, dict) else None
            if not isinstance(queries, list) or not queries or not isinstance(queries[0], str):
                raise ValueError("fake research plan is invalid")
            pass_number = metadata.get("research_pass_number", "1")
            if payload.get("document_scope_available") is True:
                return self._result(
                    tool_calls=(
                        ModelToolCall(
                            call_id=f"fake-retrieve-{pass_number}",
                            name="retrieve_documents",
                            arguments={"query": queries[0]},
                        ),
                    )
                )
            return self._result(
                tool_calls=(
                    ModelToolCall(
                        call_id=f"fake-search-{pass_number}",
                        name="search_web",
                        arguments={"query": queries[0], "max_results": 8},
                    ),
                )
            )

        if graph_node == "write_report":
            payload = _delimited_json(
                messages[-1].content or "",
                "<untrusted_validated_research_data>",
                "</untrusted_validated_research_data>",
            )
            output: dict[str, object] = {
                "summary": [],
                "findings": [],
                "limitations": [],
                "application_draft": None,
            }
            evidence = payload.get("evidence")
            document_evidence = payload.get("document_evidence")
            if isinstance(document_evidence, list) and document_evidence:
                evidence = document_evidence
            if (
                payload.get("evidence_sufficient") is True
                and isinstance(evidence, list)
                and evidence
            ):
                item = evidence[0]
                if isinstance(item, dict):
                    evidence_id = item.get("evidence_id")
                    source_id = item.get("source_id")
                    text = item.get("text")
                    if all(isinstance(value, str) for value in (evidence_id, source_id, text)):
                        output["summary"] = [
                            {
                                "claim_id": "fake-summary-1",
                                "text": text,
                                "citations": [{"source_id": source_id, "evidence_id": evidence_id}],
                            }
                        ]
                        request = payload.get("request")
                        if (
                            isinstance(request, dict)
                            and request.get("include_application_draft") is True
                        ):
                            output["application_draft"] = {
                                "paragraphs": [
                                    {
                                        "claim_id": "fake-draft-1",
                                        "text": (
                                            "My synthetic background matches this cited "
                                            "requirement."
                                        ),
                                        "citations": [
                                            {
                                                "source_id": source_id,
                                                "evidence_id": evidence_id,
                                            }
                                        ],
                                    }
                                ]
                            }
            return self._result(
                content=json.dumps(
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        raise ValueError("fake research graph node is unsupported")

    @staticmethod
    def _result(
        *,
        content: str | None = None,
        tool_calls: tuple[ModelToolCall, ...] = (),
    ) -> ChatModelResult:
        return ChatModelResult(
            content=content,
            tool_calls=tool_calls,
            usage=ModelUsage(input_tokens=0, output_tokens=0),
        )
