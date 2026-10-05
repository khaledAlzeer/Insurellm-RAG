"""Optional cross-encoder reranking with validation and a safe fallback."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Protocol

from ..errors import InsurellmError
from ..schemas import RetrievedChunk

log = logging.getLogger(__name__)


class RerankerProtocol(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Compact cross-encoder (~80 MB), loaded on first use, run on a small candidate set."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name, self.device, self._model = model_name, device, None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                device = None if self.device == "auto" else self.device
                self._model = CrossEncoder(self.model_name, device=device)
            except Exception as exc:
                raise InsurellmError(f"Could not load reranker '{self.model_name}': {exc}") from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        model = self._load()
        return [float(s) for s in model.predict([(query, t) for t in texts], show_progress_bar=False)]


def rerank(
    reranker: RerankerProtocol | None, query: str, candidates: list[RetrievedChunk],
    top_n: int, protect_lexical: int = 2,
) -> tuple[list[RetrievedChunk], list[str]]:
    """Rerank the first ``top_n`` candidates. Returns ``(ordered, warnings)``.

    * Invalid scores or any reranker failure -> fused order is kept (with a warning).
    * The best ``protect_lexical`` lexical hits are never dropped, so exact-match evidence
      cannot be pushed out by a semantic-only reranker.
    """
    if reranker is None or len(candidates) < 2:
        return candidates, []
    head, tail = candidates[:top_n], candidates[top_n:]
    try:
        scores = reranker.score(query, [c.chunk.text for c in head])
        if len(scores) != len(head) or not all(math.isfinite(s) for s in scores):
            raise ValueError("reranker returned invalid scores")
    except (InsurellmError, ValueError, RuntimeError) as exc:
        log.warning("Reranking skipped: %s", exc)
        return candidates, [f"Reranking unavailable; using fused order ({exc})."]
    for c, s in zip(head, scores):
        c.rerank_score = s
    ordered = sorted(head, key=lambda c: -(c.rerank_score or 0.0))
    protected = [c for c in head if "lexical" in c.methods and c.lexical_score is not None]
    protected = sorted(protected, key=lambda c: -(c.lexical_score or 0.0))[:protect_lexical]
    window = max(1, len(head) // 2 + 1)  # the part of the list most likely to reach the context
    for offset, c in enumerate(protected):
        if c not in ordered[:window]:
            ordered.remove(c)
            ordered.insert(min(2 + offset, window - 1), c)
    return ordered + tail, []
