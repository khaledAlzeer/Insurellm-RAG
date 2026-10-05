"""Validated data structures shared across the pipeline (no UI formatting here)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    source: str  # portable, project-relative POSIX path (relative to the KB root)
    category: str
    title: str
    text: str
    content_hash: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    category: str
    title: str
    chunk_index: int
    text: str
    heading_path: str = ""
    content_hash: str = ""  # hash of the *document*

    def metadata(self) -> dict[str, str | int]:
        return {
            "doc_id": self.doc_id, "source": self.source, "category": self.category,
            "title": self.title, "chunk_index": self.chunk_index,
            "heading_path": self.heading_path, "content_hash": self.content_hash,
        }


class RetrievedChunk(BaseModel):
    chunk: Chunk
    rank: int = 0
    methods: list[str] = Field(default_factory=list)  # e.g. ["semantic", "lexical"]
    semantic_distance: float | None = None  # raw cosine distance from Chroma (lower = closer)
    lexical_score: float | None = None  # raw BM25 score
    fusion_score: float | None = None  # RRF score (relative ordering only)
    rerank_score: float | None = None  # raw cross-encoder logit (not a probability)
    matched_subqueries: list[int] = Field(default_factory=list)
    fuzzy_terms: dict[str, str] = Field(default_factory=dict)  # query token -> KB token


class Source(BaseModel):
    label: str
    source: str
    category: str
    title: str
    cited: bool = False


class ChatTurn(BaseModel):
    user: str
    assistant: str
    sources: list[str] = Field(default_factory=list)  # sources actually used last turn
    user_entities: list[str] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    query_used: str = ""
    rewritten_query: str | None = None
    expanded_queries: list[str] = Field(default_factory=list)
    subqueries: list[str] = Field(default_factory=list)
    retrieval_metadata: dict[str, object] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)  # seconds, measured
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    status: Literal["ok", "abstained", "error"] = "ok"


class IngestionReport(BaseModel):
    discovered: int = 0
    new: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)
    chunks_created: int = 0
    chunks_removed: int = 0
    chunks_reused: int = 0
    embeddings_generated: int = 0
    seconds: float = 0.0
    dry_run: bool = False
    rebuilt: bool = False
    orphans_removed: int = 0

    def summary(self) -> str:
        mode = " (dry run - nothing written)" if self.dry_run else ""
        lines = [
            f"Ingestion summary{mode}",
            f"  Documents discovered : {self.discovered}",
            f"  New                  : {len(self.new)}",
            f"  Changed              : {len(self.changed)}",
            f"  Unchanged            : {len(self.unchanged)}",
            f"  Deleted              : {len(self.deleted)}",
            f"  Skipped              : {len(self.skipped)}",
            f"  Chunks created       : {self.chunks_created}",
            f"  Chunks removed       : {self.chunks_removed}",
            f"  Chunks reused        : {self.chunks_reused}",
            f"  Embeddings generated : {self.embeddings_generated}",
            f"  Orphan chunks removed: {self.orphans_removed}",
            f"  Time taken           : {self.seconds:.2f}s",
        ]
        for path, why in self.skipped.items():
            lines.append(f"    skipped {path}: {why}")
        return "\n".join(lines)
