"""End-to-end pipeline tests: real Chroma (temp dir) + hash embedder + scripted LLM."""

import json

from conftest import FakeLLM
from insurellm_rag.conversation import ConversationState
from insurellm_rag.errors import GenerationError, ModelUnavailableError
from insurellm_rag.generation.prompts import ABSTAIN_PHRASE
from insurellm_rag.schemas import RAGResponse


def ready(assistant, responses):
    assistant.ingest()
    assistant._llm = FakeLLM(responses)
    assistant._pipeline = None
    return assistant


def user_prompt(llm, i=-1):
    return llm.calls[i]["messages"][1]["content"]


def test_exact_lookup_answer_has_real_sources_and_timings(assistant):
    a = ready(
        assistant,
        ["Jane Fixture is a Test Engineer at Examplecorp. [Jane Fixture.md]"],
    )
    r = a.ask("Who is Jane Fixture?")

    assert isinstance(r, RAGResponse) and r.status == "ok" and r.error is None
    assert [s.label for s in r.sources] == ["Jane Fixture.md"]
    assert r.sources[0].cited
    assert r.retrieved_chunks
    assert r.retrieved_chunks[0].chunk.source == "employees/Jane Fixture.md"
    assert {"lexical"} <= set(r.retrieved_chunks[0].methods)
    assert r.timings["total"] >= r.timings["generation"] >= 0
    assert len(a.llm.calls) == 1


def test_acronym_typo_surfaces_spelling_note_and_bounded_expansion(assistant):
    a = ready(
        assistant,
        [
            json.dumps(
                {
                    "expansions": [
                        "IIOYT award winner",
                        "Innovator Award winner",
                        "IIOYT prize",
                        "IIOYT honour",
                    ]
                }
            ),
            (
                "The IIOTY Innovator Award winner is Jane Fixture "
                "[Jane Fixture.md]. Note: your question said IIOYT."
            ),
        ],
    )

    r = a.ask("Who won the IIOYT award?")

    # Temporary debug output
    print("\n========== IIOYT DEBUG ==========")
    print("STATUS:", r.status)
    print("ANSWER:", r.answer)
    print("EXPANDED:", r.expanded_queries)
    print(
        "CONTEXT:",
        [
            (c.chunk.source, c.fuzzy_terms)
            for c in r.retrieved_chunks
        ],
    )
    print("METADATA:", r.retrieval_metadata)
    print("WARNINGS:", r.warnings)
    print("LLM CALLS:", len(a.llm.calls))

    if a.llm.calls:
        print("LAST USER PROMPT:")
        print(user_prompt(a.llm))

    print("=================================\n")

    assert r.status == "ok" and len(r.expanded_queries) <= 2

    note = user_prompt(a.llm)

    assert "IIOYT" in note
    assert "closest term found is 'iioty'" in note.lower()
    assert any(
        c.fuzzy_terms == {"iioyt": "iioty"}
        for c in r.retrieved_chunks
    )
    assert r.retrieved_chunks[0].chunk.source == "employees/Jane Fixture.md"


def test_missing_entity_is_flagged_and_abstention_has_no_sources(assistant):
    a = ready(
        assistant,
        [
            json.dumps({"expansions": []}),
            ABSTAIN_PHRASE,
        ],
    )

    r = a.ask("Who attended Manchester University?")

    assert r.status == "abstained"
    assert r.sources == []
    assert "Manchester" in user_prompt(a.llm)
    assert "does not appear" in user_prompt(a.llm)
    assert r.retrieved_chunks


def test_malformed_expansion_output_falls_back_gracefully(assistant):
    a = ready(
        assistant,
        [
            "not json at all",
            "Answer [Jane Fixture.md]",
        ],
    )

    r = a.ask("Who won the IIOYT award?")

    assert r.status == "ok"
    assert r.expanded_queries == []


def test_follow_up_resolved_without_extra_llm_call(assistant):
    a = ready(
        assistant,
        [
            "Jane is a Test Engineer [Jane Fixture.md]",
            "She won the IIOTY Innovator Award [Jane Fixture.md]",
        ],
    )

    conv = ConversationState()

    a.ask("Who is Jane Fixture?", conv)
    r = a.ask("Which award did she win?", conv)

    assert r.rewritten_query
    assert "Jane Fixture" in r.rewritten_query
    assert "Which award did she win?" in user_prompt(a.llm)
    assert len(a.llm.calls) == 2
    assert r.retrieved_chunks[0].chunk.source.endswith("Jane Fixture.md")


def test_topic_shift_does_not_inherit_old_context(assistant):
    a = ready(
        assistant,
        [
            "Jane [Jane Fixture.md]",
            "Widget Pro is a product [Widget Pro.md]",
        ],
    )

    conv = ConversationState()

    a.ask("Who is Jane Fixture?", conv)
    r = a.ask("What are the features of Widget Pro?", conv)

    assert r.rewritten_query is None
    assert r.query_used == "What are the features of Widget Pro?"
    assert r.retrieved_chunks[0].chunk.source == "products/Widget Pro.md"


def test_multihop_planner_runs_only_for_complex_questions(assistant):
    q = (
        "Which product does the engineer who won the IIOTY award work on, "
        "and what does the contract say about that rollout?"
    )

    plan = json.dumps(
        {
            "subqueries": [
                "IIOTY award winner engineer",
                "Contract Alpha rollout terms",
            ]
        }
    )

    a = ready(
        assistant,
        [
            plan,
            "Partial answer [Jane Fixture.md] [Contract Alpha.md]",
        ],
    )

    r = a.ask(q)

    assert r.subqueries == [
        "IIOTY award winner engineer",
        "Contract Alpha rollout terms",
    ]

    sources = {
        c.chunk.source
        for c in r.retrieved_chunks
    }

    assert {
        "employees/Jane Fixture.md",
        "contracts/Contract Alpha.md",
    } <= sources

    assert "Retrieval subqueries used to gather the context:" in user_prompt(a.llm)

    simple = ready(
        assistant,
        ["x [Jane Fixture.md]"],
    )

    simple.ask("Who is Jane Fixture?")

    assert len(simple.llm.calls) == 1


def test_invalid_citation_is_reported_not_displayed_as_source(assistant):
    a = ready(
        assistant,
        ["Claim [Ghost.md]"],
    )

    r = a.ask("Who is Jane Fixture?")

    assert any(
        "Ghost.md" in w
        for w in r.warnings
    )

    assert all(
        s.label != "Ghost.md"
        for s in r.sources
    )

    assert all(
        not s.cited
        for s in r.sources
    )


def test_empty_index_gives_setup_message_not_fake_answer(assistant):
    r = assistant.ask("Who is anyone?")

    assert r.status == "error"
    assert "knowledge-base" in r.error
    assert r.answer == ""


def test_missing_ollama_does_not_crash_and_keeps_context(assistant):
    a = ready(
        assistant,
        [
            ModelUnavailableError(
                "Cannot reach Ollama at http://x."
            )
        ],
    )

    r = a.ask("Who is Jane Fixture?")

    assert r.status == "error"
    assert "Ollama" in r.error
    assert r.retrieved_chunks


def test_generation_failure_is_reported(assistant):
    a = ready(
        assistant,
        [
            GenerationError(
                "The model did not answer within 5s."
            )
        ],
    )

    assert "did not answer" in a.ask(
        "Who is Jane Fixture?"
    ).error


def test_input_validation(assistant):
    assert assistant.ask("   ").error == "Please enter a question."
    assert "too long" in assistant.ask(
        "x" * 2500
    ).error


def test_unexpected_exception_is_hidden_from_user(assistant):
    a = ready(assistant, [])

    a.pipeline.llm = type(
        "L",
        (),
        {
            "chat": lambda *args, **kwargs: (
                _ for _ in ()
            ).throw(
                KeyError("secret/path")
            )
        },
    )()

    r = a.ask("Who is Jane Fixture?")

    assert r.status == "error"
    assert "secret" not in r.error


def test_lexical_index_refreshes_after_new_ingestion(assistant, kb):
    a = ready(
        assistant,
        ["a [Zed.md]"],
    )

    a.ask("Who is Jane Fixture?")

    (
        kb / "company" / "Zed.md"
    ).write_text(
        "# Zed\n\nZedword is unique here.",
        "utf-8",
    )

    a.ingest()

    assert any(
        c.chunk.source == "company/Zed.md"
        for c in a.pipeline.retrieve(
            "Zedword",
            use_llm=False,
        ).candidates
    )


def test_reranker_failure_falls_back_with_warning(assistant):
    a = ready(
        assistant,
        ["ok [Jane Fixture.md]"],
    )

    a.settings.rerank_enabled = True

    def broken_score(*args, **kwargs):
        return [float("inf")] * len(args[2])

    a.pipeline.reranker = type(
        "R",
        (),
        {
            "score": broken_score,
        },
    )()

    r = a.ask("Who is Jane Fixture?")

    assert r.status == "ok"
    assert any(
        "Reranking unavailable" in w
        for w in r.warnings
    )