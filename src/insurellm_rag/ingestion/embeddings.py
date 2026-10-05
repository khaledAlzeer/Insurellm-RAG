"""Local embedding model wrapper (lazy-loaded, batched, validated)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Protocol

from ..errors import EmbeddingError

log = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vectors(vectors: Sequence[Sequence[float]], expected: int, dim: int | None) -> int:
    if len(vectors) != expected:
        raise EmbeddingError(f"Expected {expected} embeddings, got {len(vectors)}.")
    seen = dim if dim is not None else (len(vectors[0]) if vectors else 0)
    for v in vectors:
        if len(v) != seen or seen == 0:
            raise EmbeddingError("Embedding dimension mismatch or empty embedding.")
        if not all(math.isfinite(x) for x in v):
            raise EmbeddingError("Embedding contains NaN or infinite values.")
    return seen


class SentenceTransformerEmbedder:
    """Loads the model on first use. Downloads it from Hugging Face the first time only."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "auto") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device_pref = device
        self._model = None
        self.device = "unloaded"

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            device = self.device_pref
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"
            log.info("Loading embedding model %s on %s", self.model_name, device)
            self._model = SentenceTransformer(self.model_name, device=device)
            self.device = device
        except Exception as exc:  # third-party loaders raise many types; wrap with guidance
            raise EmbeddingError(
                f"Could not load embedding model '{self.model_name}': {exc}. "
                "The first run needs internet access to download it from Hugging Face "
                "(about 90 MB for the default model)."
            ) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        clean = [t for t in texts]
        if any(not t.strip() for t in clean):
            raise EmbeddingError("Refusing to embed empty text.")
        if not clean:
            return []
        model = self._load()
        batch = self.batch_size
        while True:
            try:
                arr = model.encode(
                    clean, batch_size=batch, normalize_embeddings=True,
                    show_progress_bar=False, convert_to_numpy=True,
                )
                break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and batch > 1:
                    batch //= 2
                    log.warning("Out of memory while embedding; retrying with batch=%d", batch)
                    continue
                raise EmbeddingError(f"Embedding failed: {exc}") from exc
        vectors = arr.tolist()
        validate_vectors(vectors, len(clean), None)
        return vectors
