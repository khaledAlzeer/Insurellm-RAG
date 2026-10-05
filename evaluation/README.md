# Evaluation

No ground truth is shipped. `test_cases.example.json` is a **template**: every `REPLACE_*` value
must be replaced with facts you verified in the real Knowledge Base, then set `"validated": true`.
Copy it to `evaluation/test_cases.json` (git-ignored) and run:

```powershell
uv run python scripts/evaluate.py --cases evaluation/test_cases.json            # retrieval metrics
uv run python scripts/evaluate.py --cases evaluation/test_cases.json --answers  # + answer checks
uv run python scripts/evaluate.py --cases evaluation/test_cases.json --answers --judge  # + approx. LLM judge
uv run python scripts/evaluate.py --cases evaluation/test_cases.json --compare  # rerank on vs off
```

## Case fields
`id`, `question`, `category`, `expected_answer_facts` (substrings that must appear),
`expected_sources` (relative path or bare filename), `expected_chunk_ids` (reserved),
`history` (earlier user questions replayed before the question), `is_follow_up`,
`requires_multihop`, `should_abstain`, `validated`, `notes`.

## Metrics (document level, binary relevance)
The ranked chunk list is collapsed to unique source documents (first occurrence wins). With
`R` = number of expected sources and `rel[i]` = whether the i-th document is expected:
Recall@K = hits in top K / R; Precision@K = hits in top K / K; Hit Rate@K = 1 if any hit in top K;
MRR = 1 / rank of first hit; nDCG@K = DCG@K / ideal DCG@K with gain 1 and log2 discount.
Cases without `expected_sources` are skipped from retrieval metrics.

## Answer checks
Fact coverage (case-insensitive substring), source attribution (every expected source appears among
the sources the answer actually cited/used), abstention correctness (`should_abstain` vs. the
model's abstention). `--judge` adds a local LLM faithful/relevant verdict: **approximate**, not
ground truth.

## Scenario map
Template cases cover scenarios 1-12 (including `IIOYT`/`IIOTY` and Manchester University).
Scenarios 13-15 have no KB ground truth and are covered by unit tests with controlled fixtures:
empty/malformed retrieval (`test_pipeline.py::test_empty_index...`, `test_retrieval.py` rerank
fallbacks), duplicate/overlapping chunks (`test_retrieval.py::test_context_selection_dedupes...`),
unavailable Ollama/model (`test_pipeline.py::test_missing_ollama...`, `test_generation.py`).
