"""Gradio interface. Calls only the service layer; formatting helpers are pure and testable."""

from __future__ import annotations

import html
import inspect
import logging

import gradio as gr

from .conversation import ConversationState
from .schemas import RAGResponse, RetrievedChunk
from .service import KnowledgeAssistant

log = logging.getLogger(__name__)

CSS = """
.app-header {padding: 18px 22px; border-radius: 14px; color: #fff;
  background: linear-gradient(120deg, #1e3a8a, #0f766e);}
.app-header h1 {margin: 0; font-size: 1.6rem;} .app-header p {margin: 4px 0 0; opacity: .9;}
.status-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size:.8rem;
  background: rgba(255,255,255,.2); margin-top: 8px;}
.ctx-card {border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 8px 12px; margin-bottom: 8px;}
.ctx-card summary {cursor: pointer; font-weight: 600;}
.ctx-meta {font-size: .8rem; opacity: .75; margin: 4px 0;}
.ctx-text {white-space: pre-wrap; font-size: .88rem; margin-top: 6px;}
.empty-state {opacity: .7; padding: 16px; text-align: center;}
"""
EMPTY_CONTEXT = '<div class="empty-state">Retrieved passages will appear here after you ask a question.</div>'
NO_CONTEXT = '<div class="empty-state">No passages were retrieved for this question.</div>'


def format_answer(resp: RAGResponse) -> str:
    if resp.error:
        return f"⚠️ {resp.error}"
    text = resp.answer
    if resp.sources:
        label = "Sources" if resp.sources[0].cited else "Retrieved from"
        text += f"\n\n**{label}:** " + ", ".join(f"`{s.label}` ({s.category})" for s in resp.sources)
    if resp.warnings:
        text += "\n\n_" + " ".join(resp.warnings) + "_"
    return text


def _method_text(c: RetrievedChunk) -> str:
    bits = [" + ".join(c.methods) or "n/a"]
    if c.rerank_score is not None:
        bits.append(f"rerank logit {c.rerank_score:.2f}")
    if c.semantic_distance is not None:
        bits.append(f"cosine distance {c.semantic_distance:.3f}")
    if c.fuzzy_terms:
        bits.append("fuzzy: " + ", ".join(f"{k}→{v}" for k, v in c.fuzzy_terms.items()))
    return " · ".join(bits)


def render_context_html(chunks: list[RetrievedChunk], diagnostics: bool = False) -> str:
    if not chunks:
        return NO_CONTEXT
    cards = []
    for c in chunks:
        meta = f"{html.escape(c.chunk.category)} · chunk {c.chunk.chunk_index}"
        if diagnostics:
            meta += " · " + html.escape(_method_text(c))
        cards.append(
            f'<details class="ctx-card"><summary>{c.rank}. {html.escape(c.chunk.source)}</summary>'
            f'<div class="ctx-meta">{meta}</div>'
            f'<div class="ctx-text">{html.escape(c.chunk.text)}</div></details>'
        )
    return "".join(cards)


def render_diagnostics(resp: RAGResponse) -> dict:
    return {
        "query_used": resp.query_used, "rewritten_query": resp.rewritten_query,
        "expanded_queries": resp.expanded_queries, "subqueries": resp.subqueries,
        "retrieval": resp.retrieval_metadata,
        "timings_seconds": {k: round(v, 3) for k, v in resp.timings.items()},
        "status": resp.status,
    }


def header_html(status: str) -> str:
    return (
        '<div class="app-header"><h1>Insurellm-RAG</h1>'
        "<p>Local, source-grounded assistant for the Insurellm Knowledge Base</p>"
        f'<span class="status-pill">{html.escape(status)}</span></div>'
    )


def build_ui(assistant: KnowledgeAssistant) -> gr.Blocks:
    s = assistant.settings

    def status_text() -> str:
        try:
            checks = {c.name: c for c in assistant.health()}
        except Exception:  # health must never block the UI
            log.exception("health check failed")
            return "Status unknown"
        vdb, ol = checks.get("vector_db"), checks.get("ollama")
        if vdb and vdb.status != "ok":
            return "● Knowledge index empty - run ingestion"
        if ol and ol.status != "ok":
            return "● Ollama unavailable"
        return "● Ready"

    def respond(message: str, history: list[dict], conv: ConversationState, diag: bool):
        history = list(history or [])
        message = (message or "").strip()
        if not message:
            return history, "", conv, EMPTY_CONTEXT if not history else gr.skip(), {}
        resp = assistant.ask(message, conv)
        history += [{"role": "user", "content": message},
                    {"role": "assistant", "content": format_answer(resp)}]
        ctx = render_context_html(resp.retrieved_chunks, diag) if not resp.error else NO_CONTEXT
        return history, "", conv, ctx, (render_diagnostics(resp) if diag else {})

    def new_conversation():
        return [], "", assistant.new_conversation(), EMPTY_CONTEXT, {}

    with gr.Blocks(title="Insurellm-RAG") as demo:
        status = gr.HTML(header_html("Checking status…"))
        conv = gr.State(assistant.new_conversation())  # deep-copied per session
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversation", height=480)
                box = gr.Textbox(
                    placeholder="Ask about employees, products, contracts or the company…",
                    show_label=False, lines=1, max_lines=4, autofocus=True)
                with gr.Row():
                    ask = gr.Button("Ask", variant="primary")
                    new = gr.Button("New conversation")
                    clear_ctx = gr.Button("Clear context")
            with gr.Column(scale=2):
                gr.Markdown("### Retrieved Context")
                ctx = gr.HTML(EMPTY_CONTEXT)
                diag = gr.Checkbox(label="Show retrieval diagnostics", value=s.show_diagnostics)
                diag_out = gr.JSON(label="Diagnostics", visible=True)
        inputs = [box, chat, conv, diag]
        outputs = [chat, box, conv, ctx, diag_out]
        box.submit(respond, inputs, outputs, trigger_mode="once", concurrency_limit=1)
        ask.click(respond, inputs, outputs, trigger_mode="once", concurrency_limit=1)
        new.click(new_conversation, None, outputs)
        clear_ctx.click(lambda: (EMPTY_CONTEXT, {}), None, [ctx, diag_out])
        demo.load(lambda: header_html(status_text()), None, status)
    return demo


def launch(assistant: KnowledgeAssistant) -> None:
    demo = build_ui(assistant)
    kwargs: dict = {"server_name": assistant.settings.ui_host,
                    "server_port": assistant.settings.ui_port}
    params = inspect.signature(demo.launch).parameters
    if "css" in params:
        kwargs.update(css=CSS, theme=gr.themes.Soft())
    demo.queue().launch(**kwargs)
