"""R2.1 line-addressable chunks; historical Markdown chunks remain unchanged."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from app.material.reader import MaterialFile, MaterialReadError
from app.retrieval.chunking import (
    PreparedIngestionChunk,
    PreparedIngestionSource,
    normalize_document_content,
)

MATERIAL_CHUNKING_VERSION = "material-lines-utf8-800-v1"
MATERIAL_NORMALIZATION_VERSION = "nfc-lf-v1"
_CODE_EXTENSIONS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx", ".cjs", ".mjs", ".sql", ".sh"})


def prepare_material_file(item: MaterialFile) -> PreparedIngestionSource:
    content = normalize_document_content(item.content.decode("utf-8"))
    if not content.strip() or len(content.encode("utf-8")) > 1024 * 1024:
        raise MaterialReadError("empty_or_oversize_file")
    chunks: list[PreparedIngestionChunk] = []
    for line_no, line in enumerate(content.splitlines(keepends=True), start=1):
        part = ""
        for character in line:
            if len((part + character).encode("utf-8")) > 800:
                if part.strip():
                    chunks.append(_chunk(len(chunks), part, line_no))
                part = ""
            part += character
        if part.strip():
            chunks.append(_chunk(len(chunks), part, line_no))
    if not chunks:
        raise MaterialReadError("empty_file")
    suffix = PurePosixPath(item.path).suffix.lower()
    source_type = (
        "code" if suffix in _CODE_EXTENSIONS else "markdown" if suffix == ".md" else "text"
    )
    return PreparedIngestionSource(
        source_name=item.path[-255:],
        source_type=source_type,
        title=PurePosixPath(item.path).name[:255],
        content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        normalization_version=MATERIAL_NORMALIZATION_VERSION,
        chunking_version=MATERIAL_CHUNKING_VERSION,
        chunks=tuple(chunks),
    )


def _chunk(ordinal: int, text: str, line_no: int) -> PreparedIngestionChunk:
    return PreparedIngestionChunk(
        ordinal=ordinal,
        section=None,
        text=text,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_count=len(text.encode("utf-8")),
        start_line=line_no,
        end_line=line_no,
    )
