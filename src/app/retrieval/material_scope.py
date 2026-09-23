"""Deterministic document allowlist from a server-authorized material scope."""

from __future__ import annotations

from uuid import UUID

from app.domain.project_facts import MaterialRetrievalScope


def allowed_document_ids(scope: MaterialRetrievalScope) -> tuple[UUID, ...]:
    return tuple(
        dict.fromkeys(item.document_id for item in scope.files if item.document_id is not None)
    )
