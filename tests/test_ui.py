from conftest import FakeLLM
from insurellm_rag.schemas import Chunk, RAGResponse, RetrievedChunk, Source
from insurellm_rag.ui import (
    EMPTY_CONTEXT,
    NO_CONTEXT,
    build_ui,
    format_answer,
    render_context_html,
    render_diagnostics,
)


def chunk(text="<script>alert(1)</script> body"):
    return RetrievedChunk(
        chunk=Chunk(chunk_id="c", doc_id="d", source="employees/A.md", category="employees",
                    title="A", chunk_index=2, text=text),
        rank=1, methods=["semantic", "lexical"], rerank_score=1.5)


def test_format_answer_variants():
    ok = RAGResponse(answer="Hi [A.md]", sources=[Source(label="A.md", source="employees/A.md",
                     category="employees", title="A", cited=True)])
    assert "**Sources:** `A.md` (employees)" in format_answer(ok)
    fallback = ok.model_copy(update={"sources": [ok.sources[0].model_copy(update={"cited": False})]})
    assert "Retrieved from" in format_answer(fallback)
    assert format_answer(RAGResponse(error="boom", status="error")).startswith("⚠️")


def test_context_cards_escape_html_and_hide_diagnostics_by_default():
    html_ = render_context_html([chunk()])
    assert "&lt;script&gt;" in html_ and "<script>" not in html_
    assert "employees/A.md" in html_ and "rerank logit" not in html_
    assert "rerank logit 1.50" in render_context_html([chunk()], diagnostics=True)
    assert render_context_html([]) == NO_CONTEXT and "after you ask" in EMPTY_CONTEXT


def test_diagnostics_contain_only_measured_values():
    d = render_diagnostics(RAGResponse(timings={"total": 1.23456}))
    assert d["timings_seconds"]["total"] == 1.235


def test_ui_builds_without_loading_models(assistant):
    assistant._llm = FakeLLM()
    assert build_ui(assistant) is not None
    assert assistant._pipeline is None  # nothing heavy constructed at build time
