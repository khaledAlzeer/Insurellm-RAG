"""Validated application settings (environment variables / `.env`, prefix ``INSURELLM_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INSURELLM_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- paths (relative paths are resolved against the project root) ---
    knowledge_base_dir: Path = Path("knowledge-base")
    vector_db_dir: Path = Path("vector_db")
    cache_dir: Path = Path("cache")
    collection_name: str = "insurellm_kb"

    # --- models ---
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_batch_size: int = 32
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- chunking (characters) ---
    chunk_size: int = 900
    chunk_overlap: int = 120
    min_chunk_chars: int = 60
    max_file_bytes: int = 2_000_000

    # --- retrieval ---
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    fused_top_k: int = 15
    rerank_enabled: bool = True
    rerank_top_n: int = 10
    final_context_chunks: int = 6
    max_context_chars: int = 6000
    enable_rewriting: bool = True
    enable_expansion: bool = True
    enable_multihop: bool = True
    max_expansions: int = 2
    max_subqueries: int = 3
    history_turns: int = 6

    # --- generation ---
    temperature: float = 0.1
    max_tokens: int = 512
    num_ctx: int = 4096
    request_timeout: float = 120.0

    # --- runtime / UI ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    ui_host: str = "127.0.0.1"
    ui_port: int = 7860
    show_diagnostics: bool = False

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size(cls, v: int) -> int:
        if not 200 <= v <= 4000:
            raise ValueError("must be between 200 and 4000 characters")
        return v

    @field_validator(
        "embedding_batch_size", "semantic_top_k", "lexical_top_k", "fused_top_k",
        "rerank_top_n", "final_context_chunks", "max_context_chars", "max_tokens",
        "num_ctx", "history_turns", "max_subqueries", "min_chunk_chars",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must be between 0.0 and 1.0")
        return v

    @field_validator("request_timeout")
    @classmethod
    def _timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("chunk_overlap must be >= 0 and less than half of chunk_size")
        if self.max_expansions < 0 or self.max_expansions > 5:
            raise ValueError("max_expansions must be between 0 and 5")
        self.rerank_top_n = min(self.rerank_top_n, self.fused_top_k)
        for name in ("knowledge_base_dir", "vector_db_dir", "cache_dir"):
            p = getattr(self, name)
            if not p.is_absolute():
                setattr(self, name, (PROJECT_ROOT / p).resolve())
        return self


def load_settings(**overrides: object) -> Settings:
    """Load settings; raise :class:`ConfigError` with a readable message on invalid values."""
    try:
        return Settings(**overrides)  # type: ignore[arg-type]
    except ValidationError as exc:
        lines = [
            f"  - INSURELLM_{'.'.join(str(p) for p in e['loc']).upper()}: {e['msg']}"
            for e in exc.errors()
        ]
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
