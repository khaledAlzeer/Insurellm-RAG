"""Retrieval metrics, answer checks and latency summaries over user-supplied test cases.

Relevance is judged at *document* level: the ranked chunk list is collapsed to unique source
documents (first occurrence wins) and compared with ``expected_sources`` (relative path or bare
filename). No ground truth is shipped; see evaluation/README.md.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from .conversation import ConversationState
from .errors import InsurellmError
from .generation.model import LLMProtocol
from .pipeline import RAGPipeline
from .retrieval.query import entity_phrases
from .schemas import ChatTurn


class TestCase(BaseModel):
    __test__ = False  # not a pytest class

    id: str
    question: str
    category: str = "general"
    expected_answer_facts: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)  # earlier user questions (follow-ups)
    is_follow_up: bool = False
    requires_multihop: bool = False
    should_abstain: bool = False
    validated: bool = False  # True only after a human confirmed the ground truth
    notes: str = ""


def load_cases(path: Path) -> list[TestCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["cases"] if isinstance(data, dict) else data
    return [TestCase.model_validate(c) for c in raw]


# ---------- metrics (binary relevance over ranked unique documents) ----------
def recall_at_k(rel: list[bool], total_relevant: int, k: int) -> float:
    return sum(rel[:k]) / total_relevant if total_relevant else 0.0


def precision_at_k(rel: list[bool], k: int) -> float:
    top = rel[:k]
    return sum(top) / len(top) if top else 0.0


def hit_rate_at_k(rel: list[bool], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def mrr(rel: list[bool]) -> float:
    return next((1.0 / i for i, r in enumerate(rel, 1) if r), 0.0)


def ndcg_at_k(rel: list[bool], total_relevant: int, k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rel[:k], 1) if r)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def _matches(source: str, expected: str) -> bool:
    return source == expected or source.rsplit("/", 1)[-1] == expected.rsplit("/", 1)[-1]


def ranked_documents(sources: list[str]) -> list[str]:
    return list(dict.fromkeys(sources))


def relevance(sources: list[str], expected: list[str]) -> list[bool]:
    return [any(_matches(s, e) for e in expected) for s in ranked_documents(sources)]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def latency_summary(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    stages = {k for r in rows for k in r}
    return {
        st: {
            "mean_s": statistics.fmean(r[st] for r in rows if st in r),
            "p50_s": _percentile([r[st] for r in rows if st in r], 0.5),
            "p95_s": _percentile([r[st] for r in rows if st in r], 0.95),
        }
        for st in sorted(stages)
    }


def _replay(pipeline: RAGPipeline, history: list[str]) -> ConversationState:
    """Rebuild conversation state from earlier questions using retrieval only (no generation)."""
    conv = ConversationState(max_turns=pipeline.s.history_turns)
    for q in history:
        res = pipeline.retrieve(q, conv, use_llm=False)
        conv.add_turn(ChatTurn(
            user=q, assistant="", user_entities=entity_phrases(q),
            sources=[res.context[0].chunk.title] if res.context else []))
    return conv


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[TestCase], k: int = 5,
                       use_llm: bool = False) -> dict:
    per_case, timing_rows = [], []
    for case in cases:
        if not case.expected_sources:
            per_case.append({"id": case.id, "skipped": "no expected_sources"})
            continue
        conv = _replay(pipeline, case.history) if case.history else None
        res = pipeline.retrieve(case.question, conv, use_llm=use_llm)
        rel = relevance([c.chunk.source for c in res.candidates], case.expected_sources)
        n_rel = len(case.expected_sources)
        per_case.append({
            "id": case.id, "category": case.category, "validated": case.validated,
            f"recall@{k}": recall_at_k(rel, n_rel, k), f"precision@{k}": precision_at_k(rel, k),
            "mrr": mrr(rel), f"ndcg@{k}": ndcg_at_k(rel, n_rel, k),
            f"hit@{k}": hit_rate_at_k(rel, k),
        })
        timing_rows.append(res.timings)
    scored = [c for c in per_case if "skipped" not in c]
    keys = [m for m in (f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", f"hit@{k}")]
    agg = {m: statistics.fmean(c[m] for c in scored) for m in keys} if scored else {}
    return {"k": k, "cases_scored": len(scored), "aggregate": agg, "per_case": per_case,
            "latency": latency_summary(timing_rows),
            "unvalidated_cases": sum(1 for c in scored if not c["validated"])}


def fact_coverage(answer: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    low = answer.lower()
    return sum(f.lower() in low for f in facts) / len(facts)


JUDGE_PROMPT = (
    "You grade an assistant answer against the context it was given. Reply with ONE JSON object: "
    '{"faithful": bool, "relevant": bool}. faithful = every claim is supported by the context. '
    "relevant = the answer addresses the question."
)


def evaluate_answers(pipeline: RAGPipeline, cases: list[TestCase],
                     judge: LLMProtocol | None = None) -> dict:
    """End-to-end answer checks. The optional LLM judge is approximate, not ground truth."""
    rows, timing_rows = [], []
    for case in cases:
        conv = None
        if case.history:
            conv = ConversationState(max_turns=pipeline.s.history_turns)
            for q in case.history:
                pipeline.answer(q, conv)
        resp = pipeline.answer(case.question, conv)
        cited = {s.source for s in resp.sources}
        row = {
            "id": case.id, "category": case.category, "validated": case.validated,
            "status": resp.status, "error": resp.error,
            "abstention_correct": (resp.status == "abstained") == case.should_abstain,
            "fact_coverage": fact_coverage(resp.answer, case.expected_answer_facts),
            "source_attribution_correct": (
                True if not case.expected_sources
                else all(any(_matches(s, e) for s in cited) for e in case.expected_sources)),
        }
        if judge is not None and resp.retrieved_chunks and resp.status == "ok":
            ctx = "\n".join(c.chunk.text for c in resp.retrieved_chunks)
            try:
                from .retrieval.query import parse_json_object

                data = parse_json_object(judge.chat([
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {case.question}"
                                                f"\n\nAnswer: {resp.answer}"}], json_mode=True))
                row["judge_approx"] = data
            except InsurellmError as exc:
                row["judge_approx"] = {"error": str(exc)}
        rows.append(row)
        timing_rows.append(resp.timings)
    return {"cases": len(rows), "per_case": rows, "latency": latency_summary(timing_rows)}


def compare_variants(make_pipeline, cases: list[TestCase], k: int,
                     variants: dict[str, dict]) -> dict:
    """Run retrieval evaluation per settings variant (e.g. rerank on/off)."""
    return {name: evaluate_retrieval(make_pipeline(overrides), cases, k)
            for name, overrides in variants.items()}
