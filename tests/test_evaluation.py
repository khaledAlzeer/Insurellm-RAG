import json
from pathlib import Path

import pytest

from conftest import FakeLLM
from insurellm_rag.evaluation import (
    TestCase,
    evaluate_answers,
    evaluate_retrieval,
    fact_coverage,
    hit_rate_at_k,
    latency_summary,
    load_cases,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    relevance,
)

REL = [False, True, False, True]


def test_metric_definitions():
    assert recall_at_k(REL, 2, 2) == 0.5 and recall_at_k(REL, 2, 4) == 1.0
    assert precision_at_k(REL, 4) == 0.5 and precision_at_k([], 3) == 0.0
    assert mrr(REL) == 0.5 and mrr([False]) == 0.0
    assert hit_rate_at_k(REL, 1) == 0.0 and hit_rate_at_k(REL, 2) == 1.0
    assert ndcg_at_k([True, True], 2, 2) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(REL, 2, 4) < 1


def test_relevance_collapses_chunks_to_documents_and_matches_filenames():
    assert relevance(["a/X.md", "a/X.md", "b/Y.md"], ["Y.md"]) == [False, True]


def test_fact_coverage_and_latency_summary():
    assert fact_coverage("Jane is an Engineer", ["jane", "engineer"]) == 1.0
    assert fact_coverage("x", []) == 1.0
    s = latency_summary([{"a": 1.0}, {"a": 3.0}])
    assert s["a"]["mean_s"] == 2.0


def test_example_template_loads_and_is_marked_unvalidated():
    cases = load_cases(Path(__file__).parents[1] / "evaluation" / "test_cases.example.json")
    assert len(cases) >= 12 and not any(c.validated for c in cases)
    ids = {c.id for c in cases}
    assert {"acronym-typo", "university", "missing-info", "multi-hop"} <= ids
    assert "IIOYT" in next(c for c in cases if c.id == "acronym-typo").question


def test_retrieval_eval_on_controlled_fixture(assistant):
    assistant.ingest()
    cases = [
        TestCase(id="t1", question="Who is Jane Fixture?", expected_sources=["employees/Jane Fixture.md"]),
        TestCase(id="t2", question="Who won the IIOYT award?", expected_sources=["Jane Fixture.md"]),
        TestCase(id="t3", question="no truth", expected_sources=[]),
    ]
    rep = evaluate_retrieval(assistant.pipeline, cases, k=3)
    assert rep["cases_scored"] == 2 and rep["aggregate"]["hit@3"] == 1.0
    assert rep["per_case"][2]["skipped"] and rep["unvalidated_cases"] == 2
    assert rep["latency"]["fusion"]["mean_s"] >= 0


def test_followup_replay_in_retrieval_eval(assistant):
    assistant.ingest()
    case = TestCase(id="f", question="Which award did she win?", history=["Who is Jane Fixture?"],
                    expected_sources=["Jane Fixture.md"], is_follow_up=True)
    assert evaluate_retrieval(assistant.pipeline, [case], k=3)["aggregate"]["mrr"] == 1.0


def test_answer_eval_checks_abstention_attribution_and_judge(assistant):
    assistant.ingest()
    assistant._llm = FakeLLM(["Jane is an engineer [Jane Fixture.md]", '{"faithful": true, "relevant": true}',
                              json.dumps({"expansions": []}), "The Knowledge Base does not contain enough information to answer this."])
    assistant._pipeline = None
    cases = [
        TestCase(id="a", question="Who is Jane Fixture?", expected_answer_facts=["engineer"],
                 expected_sources=["Jane Fixture.md"]),
        TestCase(id="b", question="Who attended Manchester University?", should_abstain=True),
    ]
    rep = evaluate_answers(assistant.pipeline, cases, judge=assistant.llm)
    a, b = rep["per_case"]
    assert a["fact_coverage"] == 1.0 and a["source_attribution_correct"] and a["judge_approx"]["faithful"]
    assert b["abstention_correct"] and b["status"] == "abstained"
