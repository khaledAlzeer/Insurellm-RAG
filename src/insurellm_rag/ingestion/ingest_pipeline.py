"""Incremental, non-destructive ingestion with a resumable manifest.

Safety ordering per changed document: (1) upsert the new chunks, (2) delete the old chunks that
are no longer present, (3) atomically write the manifest entry. A crash between steps leaves
extra chunks in Chroma but never a document without chunks; orphans are removed by a
reconciliation pass at the end of the next run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from ..config import Settings
from ..errors import VectorStoreError
from ..schemas import Document, IngestionReport
from .chunking import chunk_document
from .embeddings import EmbedderProtocol, validate_vectors
from .loader import load_all
from .vector_store import ChromaStore

log = logging.getLogger(__name__)
MANIFEST_VERSION = 1


def manifest_path(settings: Settings) -> Path:
    return settings.vector_db_dir / "manifest.json"


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != MANIFEST_VERSION or not isinstance(data.get("files"), dict):
            raise ValueError("unexpected structure")
        return data
    except (ValueError, OSError) as exc:
        raise VectorStoreError(
            f"Ingestion manifest is unreadable ({exc}). Rebuild with "
            "uv run python scripts/ingest.py --rebuild"
        ) from exc


def save_manifest(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _fingerprint(settings: Settings) -> str:
    return f"{settings.chunk_size}/{settings.chunk_overlap}/{settings.min_chunk_chars}"


def run_ingestion(
    settings: Settings, embedder: EmbedderProtocol, store: ChromaStore,
    *, dry_run: bool = False, rebuild: bool = False,
) -> IngestionReport:
    start = time.perf_counter()
    report = IngestionReport(dry_run=dry_run, rebuilt=rebuild)
    loaded = load_all(settings.knowledge_base_dir, settings.max_file_bytes)
    report.skipped = loaded.skipped
    report.discovered = len(loaded.documents) + len(loaded.skipped)
    mpath = manifest_path(settings)

    if rebuild and not dry_run:
        store.reset()
        manifest: dict = {}
    else:
        manifest = load_manifest(mpath)
    if manifest and manifest.get("embedding_model") not in (None, settings.embedding_model):
        raise VectorStoreError(
            "Manifest was created with a different embedding model. Use --rebuild."
        )
    files: dict = manifest.get("files", {}) if not (rebuild and dry_run) else {}
    fingerprint = _fingerprint(settings)
    existing_ids = store.existing_ids() if not (rebuild and dry_run) else set()

    current = {d.source: d for d in loaded.documents}
    todo: list[tuple[Document, bool]] = []  # (document, is_new)
    for source, doc in current.items():
        entry = files.get(source)
        if entry is None:
            report.new.append(source)
            todo.append((doc, True))
        elif (
            entry["hash"] != doc.content_hash or entry.get("fingerprint") != fingerprint
            or not set(entry["chunk_ids"]) <= existing_ids
        ):
            report.changed.append(source)
            todo.append((doc, False))
        else:
            report.unchanged.append(source)
    # Documents that failed to load this time are *not* treated as deleted.
    report.deleted = sorted(s for s in files if s not in current and s not in loaded.skipped)

    if dry_run:
        for doc, _ in todo:
            n = len(chunk_document(doc, settings.chunk_size, settings.chunk_overlap,
                                   settings.min_chunk_chars))
            report.chunks_created += n
        report.chunks_removed = sum(len(files[s]["chunk_ids"]) for s in report.deleted)
        report.seconds = time.perf_counter() - start
        return report

    for source in report.deleted:
        ids = files[source]["chunk_ids"]
        store.delete(ids)
        report.chunks_removed += len(ids)
        del files[source]
        _write(mpath, settings, files)

    dim: int | None = None
    for doc, _is_new in todo:
        chunks = chunk_document(
            doc, settings.chunk_size, settings.chunk_overlap, settings.min_chunk_chars
        )
        if not chunks:
            report.skipped[doc.source] = "no chunks produced"
            continue
        new_ids = {c.chunk_id for c in chunks}
        to_embed = [c for c in chunks if c.chunk_id not in existing_ids]
        report.chunks_reused += len(chunks) - len(to_embed)
        vectors: list[list[float]] = []
        for i in range(0, len(to_embed), settings.embedding_batch_size):
            batch = to_embed[i:i + settings.embedding_batch_size]
            vecs = embedder.embed([c.text for c in batch])
            dim = validate_vectors(vecs, len(batch), dim)
            vectors.extend(vecs)
        store.upsert(to_embed, vectors)
        report.embeddings_generated += len(to_embed)
        report.chunks_created += len(to_embed)
        old_ids = set(files.get(doc.source, {}).get("chunk_ids", []))
        stale = old_ids - new_ids
        if stale:
            store.delete(stale)
            report.chunks_removed += len(stale)
        existing_ids |= new_ids
        existing_ids -= stale
        files[doc.source] = {
            "hash": doc.content_hash, "doc_id": doc.doc_id, "fingerprint": fingerprint,
            "chunk_ids": sorted(new_ids),
        }
        _write(mpath, settings, files)
        log.info("Ingested %s (%d chunks, %d embedded)", doc.source, len(chunks), len(to_embed))

    # Reconcile: remove chunks Chroma holds that no manifest entry claims (interrupted runs).
    claimed = {cid for e in files.values() for cid in e["chunk_ids"]}
    orphans = store.existing_ids() - claimed
    if orphans:
        store.delete(orphans)
        report.orphans_removed = len(orphans)
    _write(mpath, settings, files)
    report.seconds = time.perf_counter() - start
    return report


def _write(path: Path, settings: Settings, files: dict) -> None:
    save_manifest(path, {
        "version": MANIFEST_VERSION, "embedding_model": settings.embedding_model,
        "files": files, "updated": time.time(),
    })
