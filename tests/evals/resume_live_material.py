"""Bounded live material preparation prompt, with private failed-response evidence."""

from hashlib import sha256
from uuid import uuid4

from app.agents.material_facts import SYSTEM_PROMPT
from app.llm.ports import ChatMessage
from tests.evals.product_acceptance_contracts import publish

LIMITS = (
    "本次只提取最多两条最核心的 implementation 事实,用中文,每条 claim 不超过100个汉字,"
    "每条只引用一个最短的充分证据片段(quote不超过100个字符,必须逐字来自原文)。"
    "不要推断个人职责、上线部署或效果指标;不要在claim中加入数字、版本号或编号。"
    "conditions保留必要适用范围,不知道的字段填null。questions最多两条。"
    "输出须在有限额度内闭合为完整JSON,不追求覆盖全部材料。"
)


MATERIAL_PROMPT_VERSION = "sha256:" + sha256((SYSTEM_PROMPT + "\n" + LIMITS).encode()).hexdigest()


class LiveMaterialModel:
    """Transform the injected model prompt; accounting stays in the wrapped Factory model."""

    def __init__(self, model, output_root):
        self.model, self.output_root = model, output_root

    async def invoke(self, messages, tools, metadata):
        system = messages[0].content + "\n" + LIMITS
        version = "sha256:" + sha256(system.encode()).hexdigest()
        messages = (ChatMessage(role="system", content=system), *messages[1:])
        result = await self.model.invoke(messages, tools, {**metadata, "prompt_version": version})
        publish(
            self.output_root / f"material-response-{uuid4().hex}.json",
            {
                "prompt_version": version,
                "finish_status": result.finish_status,
                "content": result.content,
                "tool_calls_present": bool(result.tool_calls),
            },
        )
        return result
