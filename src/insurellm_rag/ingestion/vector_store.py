"""ChromaDB persistence with embedding-model compatibility checks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import chromadb

from ..errors import VectorStoreError
from ..schemas import Chunk

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


class ChromaStore:
    def __init__(self, db_dir: Path, collection_name: str, embedding_model: str) -> None:
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        db_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(db_dir))
            self._col = self._client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine", "embedding_model": embedding_model,
                    "schema_version": SCHEMA_VERSION,
                },
            )
        except Exception as exc:  # chromadb raises assorted internal types
            raise VectorStoreError(
                f"Could not open the vector database at 'vector_db/': {exc}. "
                "If it is corrupted, rebuild it with: uv run python scripts/ingest.py --rebuild"
            ) from exc
        self.check_compatible(embedding_model)

    def check_compatible(self, embedding_model: str) -> None:
        stored = (self._col.metadata or {}).get("embedding_model")
        if stored and stored != embedding_model:
            raise VectorStoreError(
                f"The vector database was built with embedding model '{stored}' but the current "
                f"setting is '{embedding_model}'. Vectors from different models are not "
                "comparable. Rebuild with: uv run python scripts/ingest.py --rebuild"
            )

    # --- reads ---
    def count(self) -> int:
        return self._col.count()

    def existing_ids(self) -> set[str]:
        return set(self._col.get(include=[])["ids"])

    def get_all_chunks(self) -> list[Chunk]:
        data = self._col.get(include=["documents", "metadatas"])
        chunks = [
            Chunk(
                chunk_id=cid, doc_id=str(m["doc_id"]), source=str(m["source"]),
                category=str(m["category"]), title=str(m["title"]),
                chunk_index=int(m["chunk_index"]), text=text,
                heading_path=str(m.get("heading_path", "")),
                content_hash=str(m.get("content_hash", "")),
            )
            for cid, text, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        return sorted(chunks, key=lambda c: (c.source, c.chunk_index))

    def query(
        self, embedding: Sequence[float], n: int, where: dict | None = None
    ) -> list[tuple[Chunk, float]]:
        total = self.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[list(embedding)], n_results=min(n, total), where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, text, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append((
                Chunk(
                    chunk_id=cid, doc_id=str(m["doc_id"]), source=str(m["source"]),
                    category=str(m["category"]), title=str(m["title"]),
                    chunk_index=int(m["chunk_index"]), text=text,
                    heading_path=str(m.get("heading_path", "")),
                    content_hash=str(m.get("content_hash", "")),
                ),
                float(dist),
            ))
        return out

    # --- writes ---
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        dim = len(embeddings[0])
        self._col.upsert(
            ids=[c.chunk_id for c in chunks], embeddings=[list(e) for e in embeddings],
            documents=[c.text for c in chunks], metadatas=[c.metadata() for c in chunks],
        )
        log.debug("Upserted %d chunks (dim=%d)", len(chunks), dim)

    def delete(self, ids: Sequence[str]) -> None:
        ids = list(ids)
        for i in range(0, len(ids), 500):
            self._col.delete(ids=ids[i:i + 500])

    def reset(self) -> None:
        """Drop and recreate the collection. Only called for explicit rebuilds."""
        self._client.delete_collection(self.collection_name)
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine", "embedding_model": self.embedding_model,
                "schema_version": SCHEMA_VERSION,
            },
        )
