from __future__ import annotations

import json
from time import monotonic
from uuid import uuid4

import pytest

from app.domain.project_facts import MaterialRetrievalScope, ScopedMaterialFile
from app.domain.provisioning import WorkspaceRole
from app.domain.tenancy import TenantContext
from app.llm.ports import EmbeddingResult, ModelToolCall, ModelUsage
from app.retrieval.documents import DocumentRetrievalService, RetrievedDocumentChunk
from app.tools.contracts import ToolCallBudget, ToolExecutionContext, ToolRunContext
from app.tools.material_retrieval import (
    MATERIAL_POLICY,
    ReadExcerptHandler,
    ReadExcerptInputV1,
    RetrieveMaterialHandler,
    RetrieveMaterialInputV1,
    create_material_registry,
)


class _Embedding:
    async def embed(self, texts, metadata):
        return EmbeddingResult(vectors=(tuple(0.0 for _ in range(1536)),), usage=ModelUsage())


class _Repository:
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids

    async def search(self, **kwargs):
        assert kwargs["allowed_document_ids"] == self.doc_ids
        return (
            RetrievedDocumentChunk(
                document_id=self.doc_ids[-1],
                chunk_id=uuid4(),
                source_name="shared.md",
                section=None,
                ordinal=0,
                cosine_distance=0.1,
                text="Synthetic source text is untrusted.",
            ),
        )


class _Cancellation:
    def is_cancelled(self):
        return False


def _context(tenant):
    return ToolExecutionContext(
        workspace_id=tenant.workspace_id,
        actor_user_id=tenant.actor_user_id,
        run_id=uuid4(),
        invocation_id=uuid4(),
        action_intent_id=None,
        approval_request_id=None,
        trusted_target=None,
        deadline=monotonic() + 60,
        budget=ToolCallBudget(call_number=1, call_limit=8, remaining_calls=7),
        cancellation=_Cancellation(),
    )


async def test_multiple_project_documents_remain_server_scoped() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    docs = (uuid4(), uuid4())
    files = tuple(
        ScopedMaterialFile(
            id=uuid4(),
            path=f"project-{index}/shared.md",
            content=b"Synthetic source text is untrusted.\n",
            document_id=doc,
            source_revision=f"fixed-{index}",
        )
        for index, doc in enumerate(docs)
    )
    scope = MaterialRetrievalScope(files)
    service = DocumentRetrievalService(repository=_Repository(docs), embedding=_Embedding())
    result = await RetrieveMaterialHandler(scope, service, tenant)(
        RetrieveMaterialInputV1(query="source"), _context(tenant)
    )
    assert len(result.results) == 1
    assert result.results[0].file_ref == files[1].id
    assert result.results[0].source_revision == "fixed-1"
    registry = create_material_registry(scope=scope, service=service, tenant=tenant)
    runtime = registry.bind(
        policy_name=MATERIAL_POLICY,
        context=ToolRunContext(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            run_id=uuid4(),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=monotonic() + 60,
            cancellation=_Cancellation(),
        ),
    )
    assert {tool.name for tool in runtime.model_tools()} == {
        "retrieve_project_material",
        "read_material_excerpt",
    }


async def test_bounded_excerpt_rejects_other_files_and_large_ranges() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    file = ScopedMaterialFile(
        id=uuid4(),
        path="notes.md",
        content=b"first\nsecond\n",
        document_id=None,
        source_revision="fixed",
    )
    handler = ReadExcerptHandler(MaterialRetrievalScope((file,)))
    result = await handler(
        ReadExcerptInputV1(file_ref=str(file.id), start_line=2, end_line=2), _context(tenant)
    )
    assert result.untrusted_text == "second"
    for invalid in (
        ReadExcerptInputV1(file_ref=str(uuid4()), start_line=1, end_line=1),
        ReadExcerptInputV1(file_ref=str(file.id), start_line=2, end_line=1),
        ReadExcerptInputV1(file_ref=str(file.id), start_line=1, end_line=81),
    ):
        with pytest.raises(ValueError):
            await handler(invalid, _context(tenant))


async def test_shared_document_keeps_each_authorized_source_reference() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    second = uuid4()
    files = tuple(
        ScopedMaterialFile(
            id=uuid4(),
            path=f"project-{index}/shared.md",
            content=b"same\n",
            document_id=second,
            source_revision=f"fixed-{index}",
        )
        for index in range(2)
    )
    scope = MaterialRetrievalScope(files)
    service = DocumentRetrievalService(repository=_Repository((second,)), embedding=_Embedding())
    result = await RetrieveMaterialHandler(scope, service, tenant)(
        RetrieveMaterialInputV1(query="same"), _context(tenant)
    )
    assert {item.file_ref for item in result.results} == {item.id for item in files}


async def test_model_can_read_only_a_server_scoped_excerpt() -> None:
    tenant = TenantContext(uuid4(), uuid4(), WorkspaceRole.ADMIN)
    file = ScopedMaterialFile(
        id=uuid4(),
        path="notes.md",
        content=b"first\nsecond\n",
        document_id=None,
        source_revision="fixed",
    )
    runtime = create_material_registry(
        scope=MaterialRetrievalScope((file,)),
        service=DocumentRetrievalService(
            repository=_Repository((uuid4(), uuid4())), embedding=_Embedding()
        ),
        tenant=tenant,
    ).bind(
        policy_name=MATERIAL_POLICY,
        context=ToolRunContext(
            workspace_id=tenant.workspace_id,
            actor_user_id=tenant.actor_user_id,
            run_id=uuid4(),
            action_intent_id=None,
            approval_request_id=None,
            trusted_target=None,
            deadline=monotonic() + 60,
            cancellation=_Cancellation(),
        ),
    )
    output = await runtime.execute(
        ModelToolCall(
            call_id="read-1",
            name="read_material_excerpt",
            arguments={"file_ref": str(file.id), "start_line": 2, "end_line": 2},
        )
    )
    assert json.loads(output)["untrusted_text"] == "second"
