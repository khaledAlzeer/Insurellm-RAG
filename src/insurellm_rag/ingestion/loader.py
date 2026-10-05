"""Recursive Markdown discovery and safe loading."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import KnowledgeBaseError
from ..schemas import Document
from .preprocessing import derive_title, normalize_text

log = logging.getLogger(__name__)


@dataclass
class LoadResult:
    documents: list[Document] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # relative path -> reason


def discover_markdown(kb_dir: Path) -> list[Path]:
    if not kb_dir.is_dir():
        raise KnowledgeBaseError(
            "Knowledge Base folder not found. Copy the original Insurellm Knowledge Base "
            "(category folders such as company/, contracts/, employees/, products/) into "
            "'knowledge-base/' and run: uv run python scripts/ingest.py"
        )
    return sorted(p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".md")


def relative_source(path: Path, kb_dir: Path) -> str:
    return path.relative_to(kb_dir).as_posix()


def doc_id_for(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def category_for(source: str) -> str:
    parts = source.split("/")
    return parts[0] if len(parts) > 1 else "uncategorized"


def file_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_document(path: Path, kb_dir: Path, max_bytes: int) -> tuple[Document | None, str | None]:
    """Return ``(document, None)`` or ``(None, skip_reason)``. Never raises for bad files."""
    source = relative_source(path, kb_dir)
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return None, f"file larger than {max_bytes} bytes"
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "not valid UTF-8"
    except OSError as exc:
        return None, f"unreadable ({exc.__class__.__name__})"
    text = normalize_text(text)
    if not text:
        return None, "empty document"
    return (
        Document(
            doc_id=doc_id_for(source), source=source, category=category_for(source),
            title=derive_title(text, path.stem), text=text,
            content_hash=file_hash(text.encode("utf-8")),
        ),
        None,
    )


def load_all(kb_dir: Path, max_bytes: int) -> LoadResult:
    result = LoadResult()
    for path in discover_markdown(kb_dir):
        doc, reason = load_document(path, kb_dir, max_bytes)
        if doc:
            result.documents.append(doc)
        else:
            result.skipped[relative_source(path, kb_dir)] = reason or "unknown"
            log.warning("Skipped %s: %s", relative_source(path, kb_dir), reason)
    return result
