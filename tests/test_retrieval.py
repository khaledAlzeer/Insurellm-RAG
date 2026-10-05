import math

from insurellm_rag.retrieval.context import select_context
from insurellm_rag.retrieval.hybrid import rrf_fuse
from insurellm_rag.retrieval.lexical import BM25Index, osa_distance
from insurellm_rag.retrieval.query import (
    is_multihop_candidate,
    parse_json_object,
    validate_expansions,
    validate_rewrite,
    validate_subqueries,
)
from insurellm_rag.retrieval.reranking import rerank
from insurellm_rag.schemas import Chunk, RetrievedChunk


def mk(cid, text, source="a/A.md", doc="d1", methods=("semantic",), lex=None):
    ch = Chunk(chunk_id=cid, doc_id=doc, source=source, category="a", title=source,
               chunk_index=0, text=text)
    return RetrievedChunk(chunk=ch, methods=list(methods), lexical_score=lex)


def test_osa_counts_transposition_as_one_edit():
    assert osa_distance("iioyt", "iioty") == 1
    assert osa_distance("abc", "xyz") > 2


def test_bm25_exact_and_typo_tolerant():
    chunks = [mk("1", "The IIOTY Innovator Award went to Jane").chunk,
              mk("2", "Completely unrelated text about widgets").chunk]
    idx = BM25Index(chunks)
    exact = idx.search("IIOTY award", 5)
    assert exact[0].chunk_id == "1" and not exact[0].fuzzy_terms
    typo = idx.search("who won the IIOYT award", 5)
    assert typo[0].chunk_id == "1" and typo[0].fuzzy_terms == {"iioyt": "iioty"}


def test_bm25_unknown_term_returns_no_false_match():
    idx = BM25Index([mk("1", "alpha beta gamma").chunk])
    assert idx.search("Manchester", 5) == []


def test_rrf_fuses_and_is_deterministic():
    fused = rrf_fuse({"sem": ["a", "b", "c"], "lex": ["c", "a"]})
    assert [f[0] for f in fused][:2] == ["a", "c"]
    assert fused == rrf_fuse({"sem": ["a", "b", "c"], "lex": ["c", "a"]})
    assert set(fused[0][2]) == {"sem", "lex"}


def test_rewrite_validation_fallbacks():
    q = "Did she attend Manchester University in 2019?"
    assert validate_rewrite(q, None) is None
    assert validate_rewrite(q, "") is None
    assert validate_rewrite(q, q) is None
    assert validate_rewrite(q, "Did Ann attend Leeds?") is None  # lost entities
    good = "Did Ann Lee attend Manchester University in 2019?"
    assert validate_rewrite(q, good) == good
    assert validate_rewrite(q, good + " " + "x" * 500) is None


def test_expansion_is_bounded_deduplicated_and_on_topic():
    q = "IIOTY award winner"
    items = ["IIOTY award winner", "who won the IIOTY award", "who won the IIOTY award",
             "unrelated pizza recipes", 5, "IIOTY innovator recognition", "IIOTY prize holder"]
    out = validate_expansions(q, items, 2)
    assert out == ["who won the IIOTY award", "IIOTY innovator recognition"]
    assert validate_expansions(q, "nope", 2) == [] and validate_expansions(q, items, 0) == []


def test_json_parsing_and_subquery_validation():
    assert parse_json_object('noise {"a": 1} tail') == {"a": 1}
    assert parse_json_object("not json") is None
    assert validate_subqueries("q", ["one"], 3) == []  # a single subquery is not a plan
    assert validate_subqueries("q", ["one", "two", "three", "four"], 3) == ["one", "two", "three"]


def test_multihop_gate_is_selective():
    assert not is_multihop_candidate("Who is Jane Fixture?")
    assert is_multihop_candidate(
        "Which project does the engineer who won the IIOTY award lead, and what does it deliver?")


def test_rerank_validates_scores_and_falls_back():
    cands = [mk("1", "a"), mk("2", "b"), mk("3", "c")]

    class Bad:
        def score(self, q, texts):
            return [float("nan")] * len(texts)

    out, warns = rerank(Bad(), "q", cands, 3)
    assert [c.chunk.chunk_id for c in out] == ["1", "2", "3"] and warns

    class Raises:
        def score(self, q, texts):
            raise RuntimeError("boom")

    out, warns = rerank(Raises(), "q", cands, 3)
    assert warns and len(out) == 3


def test_rerank_reorders_and_keeps_exact_match_hits():
    cands = [mk("1", "a", methods=("semantic",)), mk("2", "b", methods=("semantic",)),
             mk("3", "c", methods=("semantic",)), mk("4", "d", methods=("lexical",), lex=9.0)]

    class Prefers:
        def score(self, q, texts):
            return [0.1, 0.2, 0.9, -5.0]

    out, warns = rerank(Prefers(), "q", cands, 4)
    assert not warns and out[0].chunk.chunk_id == "3"
    assert all(math.isfinite(c.rerank_score) for c in out)
    assert "4" in [c.chunk.chunk_id for c in out[:3]]  # lexical exact hit protected


def test_context_selection_dedupes_and_limits_per_doc():
    dup = "alpha beta gamma delta epsilon zeta eta theta"
    ranked = [mk("1", dup), mk("2", dup + " iota"), mk("3", "completely different words here"),
              mk("4", "yet another distinct passage", doc="d2", source="b/B.md")]
    out = select_context(ranked, max_chunks=5, max_chars=1000)
    ids = [c.chunk.chunk_id for c in out]
    assert "1" in ids and "2" not in ids and "4" in ids
    assert [c.rank for c in out] == list(range(1, len(out) + 1))


def test_context_respects_char_budget_and_subquery_coverage():
    a, b, c = mk("1", "x " * 100), mk("2", "y " * 100), mk("3", "z " * 10, doc="d3")
    c.matched_subqueries = [1]
    a.matched_subqueries = [0]
    out = select_context([a, b, c], max_chunks=3, max_chars=250, n_subqueries=2)
    ids = {x.chunk.chunk_id for x in out}
    assert {"1", "3"} <= ids and "2" not in ids
