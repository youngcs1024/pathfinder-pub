from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.agents.material_facts import FactExtractionError, MaterialFactExtractor
from app.domain.project_facts import ScopedMaterialFile
from app.llm.fake import FakeChatModel, ScriptedFakeChatModel
from app.llm.ports import ChatModelResult, ModelUsage


def _file():
    return ScopedMaterialFile(
        id=uuid4(),
        path="README.md",
        content=b"Planned launch.\nLocal test measured 12 requests.\n",
        document_id=None,
        source_revision="fixed",
    )


async def test_scripted_extraction_produces_unconfirmed_evidence_bound_candidate() -> None:
    file = _file()
    response = {
        "facts": [
            {
                "claim": "Local test measured 12 requests",
                "kind": "experiment",
                "conditions": {
                    "environment": "local",
                    "scope": "synthetic",
                    "metric_basis": "test log",
                },
                "evidence": [
                    {
                        "snapshot_file_id": str(file.id),
                        "start_line": 2,
                        "end_line": 2,
                        "quote": "Local test measured 12 requests",
                    }
                ],
            }
        ],
        "questions": ["Who performed the work?"],
    }
    model = ScriptedFakeChatModel(
        (ChatModelResult(content=json.dumps(response), usage=ModelUsage()),)
    )
    extracted = await MaterialFactExtractor(model).extract([file])
    assert len(extracted.facts) == 1
    assert extracted.facts[0].kind == "experiment"
    assert extracted.facts[0].evidence[0].snapshot_file_id == file.id
    assert extracted.questions == ("Who performed the work?",)


async def test_one_schema_correction_and_then_fail_closed() -> None:
    file = _file()
    bad = ChatModelResult(content="not json", usage=ModelUsage())
    corrected = ChatModelResult(content='{"facts":[],"questions":[]}', usage=ModelUsage())
    assert (
        await MaterialFactExtractor(ScriptedFakeChatModel((bad, corrected))).extract([file])
    ).facts == ()
    with pytest.raises(FactExtractionError, match="schema invalid"):
        await MaterialFactExtractor(ScriptedFakeChatModel((bad, bad))).extract([file])


async def test_default_fake_exposes_incomplete_semantic_extraction() -> None:
    result = await MaterialFactExtractor(FakeChatModel()).extract([_file()])
    assert result.facts == ()
    assert result.questions == ("offline_fake_no_semantic_extraction",)
