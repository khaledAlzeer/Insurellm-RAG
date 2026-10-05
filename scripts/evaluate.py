"""Evaluate retrieval/answers on YOUR test cases.

  uv run python scripts/evaluate.py --cases evaluation/my_cases.json
  uv run python scripts/evaluate.py --cases evaluation/my_cases.json --answers --judge
  uv run python scripts/evaluate.py --cases evaluation/my_cases.json --compare
"""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from insurellm_rag.config import load_settings
from insurellm_rag.errors import InsurellmError
from insurellm_rag.evaluation import (
    compare_variants,
    evaluate_answers,
    evaluate_retrieval,
    load_cases,
)
from insurellm_rag.pipeline import RAGPipeline
from insurellm_rag.service import KnowledgeAssistant


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cases", type=Path, default=Path("evaluation/test_cases.json"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--answers", action="store_true", help="also run end-to-end answer checks")
    ap.add_argument("--judge", action="store_true", help="approximate local LLM judge (optional)")
    ap.add_argument("--compare", action="store_true",
                    help="compare rerank on/off (retrieval metrics + latency)")
    ap.add_argument("--out", type=Path, default=Path("evaluation/results.json"))
    args = ap.parse_args()
    if not args.cases.exists():
        print(f"No test cases at {args.cases}. Copy evaluation/test_cases.example.json, "
              "replace the placeholders with cases you have verified, and re-run.")
        return 1
    try:
        assistant = KnowledgeAssistant(load_settings())
        cases = load_cases(args.cases)
        if assistant.store.count() == 0:
            print("Vector index is empty. Run scripts/ingest.py first.")
            return 1
        out: dict = {"retrieval": evaluate_retrieval(assistant.pipeline, cases, args.k)}
        if args.compare:
            def make(overrides: dict) -> RAGPipeline:
                s = assistant.settings.model_copy(update=overrides)
                return RAGPipeline(s, assistant.store, assistant.embedder, assistant.llm,
                                   assistant.pipeline.reranker)
            out["compare"] = compare_variants(
                make, cases, args.k,
                {"rerank_on": {"rerank_enabled": True}, "rerank_off": {"rerank_enabled": False}})
        if args.answers:
            out["answers"] = evaluate_answers(assistant.pipeline, cases,
                                              assistant.llm if args.judge else None)
    except InsurellmError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    agg = out["retrieval"]["aggregate"]
    print(json.dumps({"cases_scored": out["retrieval"]["cases_scored"], **agg}, indent=2))
    n_un = out["retrieval"]["unvalidated_cases"]
    if n_un:
        print(f"NOTE: {n_un} scored case(s) are marked validated=false; treat as provisional.")
    print(f"Full results written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
