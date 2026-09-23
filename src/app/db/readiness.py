from __future__ import annotations

import asyncio
from typing import Final

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

REQUIRED_DATABASE_REVISION: Final = "0019_r21_material_line_ranges"
_READINESS_TIMEOUT_SECONDS: Final = 1.0


class DatabaseReadinessProbe:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._enabled = True

    def mark_not_ready(self) -> None:
        self._enabled = False

    async def is_ready(self) -> bool:
        if not self._enabled:
            return False

        try:
            async with asyncio.timeout(_READINESS_TIMEOUT_SECONDS):
                async with self._engine.connect() as connection:
                    result = await connection.execute(
                        text(
                            """
                            SELECT
                                (SELECT version_num FROM alembic_version) AS revision,
                                (
                                    SELECT format_type(attribute.atttypid, attribute.atttypmod)
                                    FROM pg_attribute AS attribute
                                    JOIN pg_class AS relation
                                      ON relation.oid = attribute.attrelid
                                    JOIN pg_namespace AS namespace
                                      ON namespace.oid = relation.relnamespace
                                    WHERE namespace.nspname = current_schema()
                                      AND relation.relname = 'document_chunks'
                                      AND attribute.attname = 'embedding'
                                      AND attribute.attnum > 0
                                      AND NOT attribute.attisdropped
                                ) AS embedding_type
                            """
                        )
                    )
                    row = result.one()
        except (TimeoutError, SQLAlchemyError):
            return False

        return row.revision == REQUIRED_DATABASE_REVISION and row.embedding_type == "vector(1536)"
