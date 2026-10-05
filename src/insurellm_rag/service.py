"""Service layer used by the UI and scripts. Owns model/store construction and health checks."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

from .config import Settings
from .conversation import ConversationState
from .errors import InsurellmError, KnowledgeBaseError
from .generation.model import LLMProtocol, OllamaChat
from .ingestion.embeddings import EmbedderProtocol, SentenceTransformerEmbedder
from .ingestion.ingest_pipeline import run_ingestion
from .ingestion.loader import discover_markdown
from .ingestion.vector_store import ChromaStore
from .pipeline import RAGPipeline
from .retrieval.reranking import CrossEncoderReranker, RerankerProtocol
from .schemas import IngestionReport, RAGResponse

log = logging.getLogger(__name__)


@dataclass
class HealthCheck:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


class KnowledgeAssistant:
    """Builds components lazily so importing/starting never loads a model by itself."""

    def __init__(
        self, settings: Settings, *, embedder: EmbedderProtocol | None = None,
        llm: LLMProtocol | None = None, reranker: RerankerProtocol | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        self.settings = settings
        self._embedder, self._llm, self._reranker, self._store = embedder, llm, reranker, store
        self._pipeline: RAGPipeline | None = None

    @property
    def embedder(self) -> EmbedderProtocol:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                self.settings.embedding_model, self.settings.embedding_batch_size,
                self.settings.embedding_device)
        return self._embedder

    @property
    def llm(self) -> LLMProtocol:
        if self._llm is None:
            s = self.settings
            self._llm = OllamaChat(s.ollama_base_url, s.chat_model, s.temperature,
                                   s.max_tokens, s.num_ctx, s.request_timeout)
        return self._llm

    @property
    def store(self) -> ChromaStore:
        if self._store is None:
            self._store = ChromaStore(self.settings.vector_db_dir, self.settings.collection_name,
                                      self.settings.embedding_model)
        return self._store

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            if self._reranker is None and self.settings.rerank_enabled:
                self._reranker = CrossEncoderReranker(
                    self.settings.reranker_model, self.settings.embedding_device)
            self._pipeline = RAGPipeline(
                self.settings, self.store, self.embedder, self.llm, self._reranker)
        return self._pipeline

    def new_conversation(self) -> ConversationState:
        return ConversationState(max_turns=self.settings.history_turns)

    def ask(self, question: str, conversation: ConversationState | None = None) -> RAGResponse:
        try:
            return self.pipeline.answer(question, conversation)
        except InsurellmError as exc:  # e.g. incompatible vector DB while building the pipeline
            return RAGResponse(error=str(exc), status="error")

    def ingest(self, *, dry_run: bool = False, rebuild: bool = False) -> IngestionReport:
        return run_ingestion(self.settings, self.embedder, self.store,
                             dry_run=dry_run, rebuild=rebuild)

    # ---------- diagnostics ----------
    def health(self, *, load_models: bool = False) -> list[HealthCheck]:
        s, out = self.settings, []
        for mod in ("gradio", "chromadb", "sentence_transformers", "ollama", "pydantic"):
            ok = importlib.util.find_spec(mod) is not None
            out.append(HealthCheck(f"dependency:{mod}", "ok" if ok else "fail",
                                   "installed" if ok else "missing - run 'uv sync'"))
        try:
            n = len(discover_markdown(s.knowledge_base_dir))
            out.append(HealthCheck("knowledge_base", "ok" if n else "warn",
                                   f"{n} Markdown files" if n else
                                   "folder is empty - copy the original Knowledge Base into it"))
        except KnowledgeBaseError as exc:
            out.append(HealthCheck("knowledge_base", "warn", str(exc)))
        try:
            count = self.store.count()
            out.append(HealthCheck("vector_db", "ok" if count else "warn",
                                   f"{count} chunks indexed" if count else
                                   "empty - run: uv run python scripts/ingest.py"))
        except InsurellmError as exc:
            out.append(HealthCheck("vector_db", "fail", str(exc)))
        try:
            llm = self.llm
            llm.check()  # type: ignore[attr-defined]
            out.append(HealthCheck("ollama", "ok", f"model '{s.chat_model}' available"))
        except InsurellmError as exc:
            out.append(HealthCheck("ollama", "fail", str(exc)))
        except AttributeError:
            out.append(HealthCheck("ollama", "ok", "custom LLM in use"))
        try:
            import torch

            cuda = torch.cuda.is_available()
            out.append(HealthCheck("compute", "ok" if cuda else "warn",
                                   "CUDA available" if cuda else
                                   "CPU only (fine for the default small models)"))
        except ImportError:
            out.append(HealthCheck("compute", "warn", "PyTorch not installed"))
        if load_models:
            try:
                self.embedder.embed(["health check"])
                out.append(HealthCheck("embedding_model", "ok", s.embedding_model))
            except InsurellmError as exc:
                out.append(HealthCheck("embedding_model", "fail", str(exc)))
        return out

    def is_ready(self) -> bool:
        return all(c.status != "fail" for c in self.health()
                   if c.name in ("vector_db", "ollama")) and self.store.count() > 0
