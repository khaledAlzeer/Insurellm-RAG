"""Context selection: dedupe, bounded size, relevance filtering, sub-query coverage."""

from __future__ import annotations

from ..schemas import RetrievedChunk
from .lexical import tokenize


# Maximum allowed score drop from the strongest reranked result
# before a chunk is considered unrelated to the current query.
RERANK_SCORE_DROP = 4.0


def _tokens(c: RetrievedChunk) -> set[str]:
    return set(tokenize(c.chunk.text))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a & b) / min(len(a), len(b))


def _has_rerank_scores(ranked: list[RetrievedChunk]) -> bool:
    """Return True when the ranked list contains usable reranker scores."""

    return any(
        c.rerank_score is not None
        for c in ranked
    )


def _is_relevant(
    c: RetrievedChunk,
    best_score: float | None,
    n_subqueries: int,
) -> bool:
    """Decide whether a chunk is relevant enough for the final context.

    Reranker scores are model logits, so absolute values are not used.
    Instead, a chunk is compared with the strongest reranked result.

    Chunks explicitly matched to a multi-hop sub-query are preserved so
    relevance filtering does not break multi-document answers.
    """

    # A chunk matched to a multi-hop sub-query is explicitly protected.
    if n_subqueries > 0 and c.matched_subqueries:
        return True

    # If there is no reranker score at all, preserve the existing
    # retrieval behavior.
    if best_score is None:
        return True

    # When reranking is active, a None score means this chunk was
    # outside the rerank window and therefore was never evaluated.
    # It should not be added to the final context merely because
    # max_chunks has not been reached.
    if c.rerank_score is None:
        return False

    score_drop = best_score - c.rerank_score

    return score_drop <= RERANK_SCORE_DROP


def select_context(
    ranked: list[RetrievedChunk],
    max_chunks: int,
    max_chars: int,
    per_doc_cap: int = 3,
    dup_threshold: float = 0.85,
    n_subqueries: int = 0,
) -> list[RetrievedChunk]:
    """Pick the smallest useful, non-redundant, relevant context.

    When reranking is available, chunks that fall far behind the strongest
    reranked evidence are excluded. Multi-hop sub-query matches are preserved.
    If reranking is unavailable, the previous ranking-based behavior is used.
    """

    chosen: list[RetrievedChunk] = []
    toks: list[set[str]] = []
    used = 0
    per_doc: dict[str, int] = {}

    reranked = _has_rerank_scores(ranked)

    best_score: float | None = None

    if reranked:
        scores = [
            c.rerank_score
            for c in ranked
            if c.rerank_score is not None
        ]

        if scores:
            best_score = max(scores)

    def try_add(c: RetrievedChunk) -> bool:
        nonlocal used

        t = _tokens(c)

        if any(
            _overlap(t, o) >= dup_threshold
            for o in toks
        ):
            return False

        if per_doc.get(c.chunk.doc_id, 0) >= per_doc_cap:
            return False

        if chosen and used + len(c.chunk.text) > max_chars:
            return False

        chosen.append(c)
        toks.append(t)
        used += len(c.chunk.text)

        per_doc[c.chunk.doc_id] = (
            per_doc.get(c.chunk.doc_id, 0) + 1
        )

        return True

    # Guarantee each sub-query contributes its best relevant chunk.
    # This happens before normal relevance selection because multi-hop
    # coverage is an explicit retrieval requirement.
    for sq in range(n_subqueries):
        for c in ranked:
            if (
                sq in c.matched_subqueries
                and c not in chosen
                and len(chosen) < max_chunks
            ):
                if try_add(c):
                    break

    # Normal relevance-ordered selection.
    for c in ranked:
        if len(chosen) >= max_chunks:
            break

        if c in chosen:
            continue

        # If reranking is active, reject chunks that are far behind
        # the strongest evidence. Without reranking, preserve the
        # previous behavior.
        if reranked and not _is_relevant(
            c,
            best_score,
            n_subqueries,
        ):
            continue

        try_add(c)

    # Restore original relevance order.
    order = {
        id(c): i
        for i, c in enumerate(ranked)
    }

    chosen.sort(
        key=lambda c: order[id(c)]
    )

    # Assign final context ranks.
    for i, c in enumerate(
        chosen,
        start=1,
    ):
        c.rank = i

    return chosen