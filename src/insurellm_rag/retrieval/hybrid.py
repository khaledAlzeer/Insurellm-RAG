"""Reciprocal Rank Fusion over several ranked ID lists."""

from __future__ import annotations


def rrf_fuse(
    rankings: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[str, float, list[str]]]:
    """Return ``(id, fused_score, contributing_lists)`` sorted best-first.

    The fused score is only meaningful for ordering; it is not a probability.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for name, ids in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, cid in enumerate(dict.fromkeys(ids), start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            sources.setdefault(cid, []).append(name)
    return sorted(((c, s, sources[c]) for c, s in scores.items()), key=lambda t: (-t[1], t[0]))
