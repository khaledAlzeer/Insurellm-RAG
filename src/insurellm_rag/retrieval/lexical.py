"""Lightweight BM25 with typo-tolerant query-term matching (no extra dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..schemas import Chunk

_TOKEN = re.compile(r"[\w][\w'\-]*", re.UNICODE)
STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "why", "with", "you", "your", "me", "my", "tell", "about", "please", "can", "could", "would", "should"]
)


def tokenize(text: str) -> list[str]:
    return [t.lower().strip("'-") for t in _TOKEN.findall(text) if t.strip("'-")]


def query_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def osa_distance(a: str, b: str, limit: int = 3) -> int:
    """Optimal-string-alignment distance (counts an adjacent transposition as one edit)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


@dataclass
class LexicalHit:
    chunk_id: str
    score: float
    fuzzy_terms: dict[str, str] = field(default_factory=dict)


class BM25Index:
    FUZZY_WEIGHT = 0.5  # fuzzy matches count less than exact matches

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids = [c.chunk_id for c in chunks]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, chunk in enumerate(chunks):
            # Include title/source so name queries match even if the body omits the name.
            toks = tokenize(f"{chunk.title} {chunk.source} {chunk.text}")
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.n = len(chunks)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0

    @property
    def vocabulary(self) -> set[str]:
        return set(self.postings)

    def fuzzy_variants(self, term: str) -> list[str]:
        if term in self.postings or len(term) < 5 or term.isdigit():
            return []
        limit = 1 if len(term) < 8 else 2
        found = [
            v for v in self.postings
            if v[0] == term[0] and v != term and abs(len(v) - len(term)) <= limit
            and osa_distance(term, v, limit) <= limit
        ]
        # Prefer rarer (more specific) variants; keep the list tiny to avoid drift.
        return sorted(found, key=lambda v: len(self.postings[v]))[:2]

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[LexicalHit]:
        if not self.n:
            return []
        weights: dict[str, float] = {}
        fuzzy: dict[str, str] = {}
        for term in dict.fromkeys(query_terms(query)):
            if term in self.postings:
                weights[term] = max(weights.get(term, 0.0), 1.0)
            else:
                for variant in self.fuzzy_variants(term):
                    weights[variant] = max(weights.get(variant, 0.0), self.FUZZY_WEIGHT)
                    fuzzy[term] = variant
        scores: dict[int, float] = defaultdict(float)
        hit_fuzzy: dict[int, dict[str, str]] = defaultdict(dict)
        for term, w in weights.items():
            idf = self._idf(term)
            for i, tf in self.postings[term]:
                norm = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                )
                scores[i] += w * idf * norm
                for q, v in fuzzy.items():
                    if v == term:
                        hit_fuzzy[i][q] = v
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [LexicalHit(self.ids[i], s, hit_fuzzy.get(i, {})) for i, s in ranked]
