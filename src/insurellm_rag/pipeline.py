"""Query-time orchestration: understand -> hybrid retrieve -> (plan) -> rerank -> select -> answer.

Selective activation keeps latency low: rewriting is deterministic for most follow-ups, query
expansion runs only when the first pass looks weak (unknown/misspelled terms), and the
multi-hop planner runs only when a cheap heuristic flags a multi-part question.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .config import Settings
from .conversation import ConversationState, FollowUpDecision
from .errors import InsurellmError
from .generation.model import LLMProtocol
from .generation.prompts import (
    ABSTAIN_PHRASE,
    build_messages,
    extract_citations,
    planner_messages,
    source_labels,
    understanding_messages,
)
from .ingestion.embeddings import EmbedderProtocol
from .ingestion.ingest_pipeline import manifest_path
from .ingestion.vector_store import ChromaStore
from .retrieval.context import select_context
from .retrieval.hybrid import rrf_fuse
from .retrieval.lexical import BM25Index
from .retrieval.query import (
    entity_phrases,
    extract_entities,
    is_multihop_candidate,
    parse_json_object,
    validate_expansions,
    validate_rewrite,
    validate_subqueries,
)
from .retrieval.reranking import RerankerProtocol, rerank
from .schemas import ChatTurn, Chunk, RAGResponse, RetrievedChunk, Source

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000

_CITATION_TOKEN = re.compile(r"\[([^\[\]]{1,200})\]")


@dataclass
class RetrievalResult:
    question: str
    query_used: str
    rewritten: str | None = None
    expansions: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    candidates: list[RetrievedChunk] = field(default_factory=list)
    context: list[RetrievedChunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    decision: FollowUpDecision | None = None


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        store: ChromaStore,
        embedder: EmbedderProtocol,
        llm: LLMProtocol,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self.s, self.store, self.embedder, self.llm, self.reranker = (
            settings,
            store,
            embedder,
            llm,
            reranker,
        )
        self._index: BM25Index | None = None
        self._chunks: dict[str, Chunk] = {}
        self._signature: tuple | None = None

    # ---------- helpers ----------
    @contextmanager
    def _time(
        self,
        timings: dict[str, float],
        name: str,
    ) -> Iterator[None]:
        t0 = time.perf_counter()

        try:
            yield
        finally:
            timings[name] = (
                timings.get(name, 0.0)
                + time.perf_counter()
                - t0
            )

    def _lexical(self) -> tuple[BM25Index, dict[str, Chunk]]:
        mp = manifest_path(self.s)

        sig = (
            self.store.count(),
            mp.stat().st_mtime_ns if mp.exists() else 0,
        )

        if self._index is None or sig != self._signature:
            chunks = self.store.get_all_chunks()

            self._index, self._chunks, self._signature = (
                BM25Index(chunks),
                {c.chunk_id: c for c in chunks},
                sig,
            )

        return self._index, self._chunks

    def _llm_json(
        self,
        messages: list[dict[str, str]],
        warnings: list[str],
    ) -> dict | None:
        try:
            return parse_json_object(
                self.llm.chat(
                    messages,
                    json_mode=True,
                    max_tokens=200,
                )
            )

        except InsurellmError as exc:
            warnings.append(
                f"Query understanding skipped: {exc}"
            )
            return None

    def _run_queries(
        self,
        texts: list[str],
        pool: dict[str, RetrievedChunk],
        index: BM25Index,
        chunk_map: dict[str, Chunk],
        timings: dict[str, float],
    ) -> list[dict[str, list[str]]]:
        """Semantic + lexical search for each text; accumulates metadata in ``pool``."""

        with self._time(timings, "embed_query"):
            vectors = self.embedder.embed(texts)

        out: list[dict[str, list[str]]] = []

        for text, vec in zip(texts, vectors):
            with self._time(timings, "semantic_search"):
                sem = self.store.query(
                    vec,
                    self.s.semantic_top_k,
                )

            with self._time(timings, "lexical_search"):
                lex = index.search(
                    text,
                    self.s.lexical_top_k,
                )

            for chunk, dist in sem:
                rc = pool.setdefault(
                    chunk.chunk_id,
                    RetrievedChunk(chunk=chunk),
                )

                if "semantic" not in rc.methods:
                    rc.methods.append("semantic")

                rc.semantic_distance = (
                    dist
                    if rc.semantic_distance is None
                    else min(
                        rc.semantic_distance,
                        dist,
                    )
                )

            for hit in lex:
                rc = pool.setdefault(
                    hit.chunk_id,
                    RetrievedChunk(
                        chunk=chunk_map[hit.chunk_id]
                    ),
                )

                if "lexical" not in rc.methods:
                    rc.methods.append("lexical")

                rc.lexical_score = max(
                    rc.lexical_score or 0.0,
                    hit.score,
                )

                rc.fuzzy_terms.update(
                    hit.fuzzy_terms
                )

            out.append(
                {
                    "semantic": [
                        c.chunk_id
                        for c, _ in sem
                    ],
                    "lexical": [
                        h.chunk_id
                        for h in lex
                    ],
                }
            )

        return out

    def _evidence_notes(
        self,
        question: str,
        context: list[RetrievedChunk],
        index: BM25Index,
    ) -> list[str]:
        """Flag missing terms while allowing strong fuzzy matches supported by context."""

        text = " ".join(
            c.chunk.text
            for c in context
        ).lower()

        notes = []

        for ent in extract_entities(question):
            low = ent.lower()

            if low in text:
                continue

            variants = [
                v
                for v in index.fuzzy_variants(low)
                if v in text
            ]

            if variants:
                variant = variants[0]

                notes.append(
                    f"The question uses '{ent}', while the closest term found is "
                    f"'{variant}'. Treat '{variant}' as the evidence-backed form "
                    "only when the retrieved context directly supports the requested fact. "
                    f"Use '{variant}' exactly as written in the Knowledge Base."
                )

            else:
                notes.append(
                    f"The term '{ent}' from the question does not appear in the retrieved text. "
                    "Do not assume anything about it."
                )

        return notes

    def _has_entity_fuzzy_evidence(
        self,
        question: str,
        context: list[RetrievedChunk],
        index: BM25Index,
    ) -> bool:
        """Require entity + acronym evidence together when the question names an entity.

        For general acronym questions, fuzzy lexical evidence is sufficient.
        For entity-specific acronym questions, require the named entity and the
        exact/fuzzy acronym evidence to appear in the same retrieved chunk.
        """

        phrases = entity_phrases(question)

        # Ignore acronym-only entities here.
        # An acronym such as "IIOYT" is the term being verified, not a named
        # person/entity that must appear in the same chunk.
        acronym_only_phrases = {
            phrase
            for phrase in phrases
            if phrase in extract_entities(question)
            and len(phrase) >= 3
            and phrase.isupper()
        }

        phrases = [
            phrase
            for phrase in phrases
            if phrase not in acronym_only_phrases
        ]

        # General acronym questions do not have a specific named entity to protect.
        if not phrases:
            return True

        acronyms = [
            ent
            for ent in extract_entities(question)
            if len(ent) >= 3 and ent.isupper()
        ]

        # No acronym means this safety check is not applicable.
        if not acronyms:
            return True

        for phrase in phrases:
            phrase_low = phrase.lower()

            for acronym in acronyms:
                acronym_low = acronym.lower()

                for retrieved in context:
                    text = retrieved.chunk.text.lower()

                    # The named entity must be present in this exact chunk.
                    if phrase_low not in text:
                        continue

                    # Exact acronym evidence is valid.
                    if acronym_low in text:
                        return True

                    # Fuzzy acronym evidence is valid when the same chunk
                    # contains the closest KB vocabulary variant.
                    variants = index.fuzzy_variants(acronym_low)

                    if any(
                        variant.lower() in text
                        for variant in variants
                    ):
                        return True

        # Entity-specific acronym question, but no single chunk connects
        # the requested entity with the exact/fuzzy acronym evidence.
        return False

    # ---------- retrieval ----------
    def retrieve(
        self,
        question: str,
        conversation: ConversationState | None = None,
        *,
        use_llm: bool = True,
    ) -> RetrievalResult:
        question = question.strip()

        res = RetrievalResult(
            question=question,
            query_used=question,
        )

        t = res.timings
        index, chunk_map = self._lexical()

        decision = (
            conversation.decide(question)
            if conversation
            else FollowUpDecision(False)
        )

        res.decision = decision

        with self._time(
            t,
            "query_understanding",
        ):
            if (
                decision.is_follow_up
                and conversation
                and self.s.enable_rewriting
            ):
                det = conversation.resolve_deterministic(
                    question,
                    decision,
                )

                rewritten = validate_rewrite(
                    question,
                    det,
                )

                if rewritten is None and use_llm:
                    data = self._llm_json(
                        understanding_messages(
                            question,
                            decision.focus,
                            decision.previous_question,
                            True,
                            0,
                        ),
                        res.warnings,
                    )

                    rewritten = validate_rewrite(
                        question,
                        (data or {}).get("rewrite"),
                    )

                res.rewritten = rewritten

        res.query_used = (
            res.rewritten
            or question
        )

        # Use the rewritten standalone query for follow-ups.
        # The original follow-up may contain unresolved references such as "her" or "his",
        # so retrieving it alongside the rewritten query can inject unrelated evidence.
        texts = (
            [res.rewritten]
            if res.rewritten
            else [question]
        )

        pool: dict[str, RetrievedChunk] = {}

        runs = self._run_queries(
            texts,
            pool,
            index,
            chunk_map,
            t,
        )

        # Treat a strong fuzzy lexical match as evidence that the term may be misspelled.
        # Avoid LLM expansion when BM25 already found a close vocabulary variant.
        fuzzy_entities = {
            original.lower(): variant.lower()
            for rc in pool.values()
            for original, variant in rc.fuzzy_terms.items()
        }

        # Adaptive expansion: only when a term in the question is unknown to the index
        # and no fuzzy lexical variant has already been found.
        weak = (
            any(
                e.lower() not in index.vocabulary
                and e.lower() not in fuzzy_entities
                and not e.isdigit()
                for e in extract_entities(question)
            )
            or not any(
                r["lexical"]
                for r in runs
            )
        )

        if (
            weak
            and use_llm
            and self.s.enable_expansion
            and self.s.max_expansions > 0
        ):
            with self._time(
                t,
                "query_expansion",
            ):
                data = self._llm_json(
                    understanding_messages(
                        question,
                        [],
                        "",
                        False,
                        self.s.max_expansions,
                    ),
                    res.warnings,
                )

                res.expansions = validate_expansions(
                    question,
                    (data or {}).get("expansions"),
                    self.s.max_expansions,
                )

            if res.expansions:
                runs += self._run_queries(
                    res.expansions,
                    pool,
                    index,
                    chunk_map,
                    t,
                )

                texts += res.expansions

        rankings: dict[str, list[str]] = {}
        weights: dict[str, float] = {}

        for i, (text, run) in enumerate(
            zip(texts, runs)
        ):
            for kind, ids in run.items():
                rankings[
                    f"{kind}:{i}"
                ] = ids

                weights[
                    f"{kind}:{i}"
                ] = (
                    1.0
                    if text in (
                        question,
                        res.rewritten,
                    )
                    else 0.6
                )

        # Selective multi-hop planning.
        if (
            use_llm
            and self.s.enable_multihop
            and is_multihop_candidate(question)
        ):
            with self._time(
                t,
                "planning",
            ):
                data = self._llm_json(
                    planner_messages(
                        question,
                        self.s.max_subqueries,
                    ),
                    res.warnings,
                )

                res.subqueries = validate_subqueries(
                    question,
                    (data or {}).get("subqueries"),
                    self.s.max_subqueries,
                )

            if res.subqueries:
                sub_runs = self._run_queries(
                    res.subqueries,
                    pool,
                    index,
                    chunk_map,
                    t,
                )

                for j, run in enumerate(
                    sub_runs
                ):
                    for kind, ids in run.items():
                        rankings[
                            f"sub{j}:{kind}"
                        ] = ids

                    sub_fused = rrf_fuse(
                        {
                            k: v
                            for k, v in run.items()
                        }
                    )

                    for cid, _, _ in sub_fused[:4]:
                        pool[
                            cid
                        ].matched_subqueries.append(j)

        with self._time(
            t,
            "fusion",
        ):
            fused = rrf_fuse(
                rankings,
                weights=weights,
            )[: self.s.fused_top_k]

            candidates = []

            for cid, score, _ in fused:
                rc = pool[cid]
                rc.fusion_score = score
                candidates.append(rc)

        if (
            self.s.rerank_enabled
            and self.reranker is not None
        ):
            with self._time(
                t,
                "rerank",
            ):
                candidates, warns = rerank(
                    self.reranker,
                    (
                        res.query_used
                        if not res.subqueries
                        else question
                    ),
                    candidates,
                    self.s.rerank_top_n,
                )

                res.warnings += warns

        res.candidates = candidates

        with self._time(
            t,
            "context_selection",
        ):
            res.context = select_context(
                candidates,
                self.s.final_context_chunks,
                self.s.max_context_chars,
                n_subqueries=len(
                    res.subqueries
                ),
            )

            res.notes = self._evidence_notes(
                question,
                res.context,
                index,
            )

        return res

    # ---------- answering ----------
    def answer(
        self,
        question: str,
        conversation: ConversationState | None = None,
    ) -> RAGResponse:
        t0 = time.perf_counter()

        question = (
            question or ""
        ).strip()

        if not question:
            return RAGResponse(
                error="Please enter a question.",
                status="error",
            )

        if len(question) > MAX_QUESTION_CHARS:
            return RAGResponse(
                error=(
                    f"Question is too long "
                    f"(limit {MAX_QUESTION_CHARS} characters)."
                ),
                status="error",
            )

        resp = RAGResponse(
            query_used=question,
        )

        try:
            if self.store.count() == 0:
                resp.status, resp.error = (
                    "error",
                    (
                        "The knowledge index is empty. Copy the Insurellm Knowledge Base into "
                        "'knowledge-base/' and run: uv run python scripts/ingest.py"
                    ),
                )

                return resp

            res = self.retrieve(
                question,
                conversation,
            )

            resp.query_used = res.query_used
            resp.rewritten_query = res.rewritten
            resp.expanded_queries = res.expansions
            resp.subqueries = res.subqueries
            resp.retrieved_chunks = res.context
            resp.warnings = list(
                res.warnings
            )
            resp.timings = dict(
                res.timings
            )

            resp.retrieval_metadata = {
                "follow_up": bool(
                    res.decision
                    and res.decision.is_follow_up
                ),
                "follow_up_reason": (
                    res.decision.reason
                    if res.decision
                    else ""
                ),
                "candidates": len(
                    res.candidates
                ),
                "context_chunks": len(
                    res.context
                ),
                "evidence_notes": res.notes,
                "reranked": any(
                    c.rerank_score is not None
                    for c in res.candidates
                ),
            }

            if not res.context:
                resp.answer = ABSTAIN_PHRASE
                resp.status = "abstained"
                return resp

            index, _ = self._lexical()

            # Important safety check:
            # For entity + acronym questions, require the requested entity
            # and the exact/fuzzy acronym evidence to exist in the same chunk.
            #
            # This prevents cross-entity hallucinations such as:
            # "What award did Avery Lancaster win at IIOYT?"
            # being answered using Maxine Thompson's IIOTY evidence.
            if not self._has_entity_fuzzy_evidence(
                question,
                res.context,
                index,
            ):
                resp.answer = (
                    f"{ABSTAIN_PHRASE}\n\n"
                    "The retrieved evidence does not directly connect "
                    "the requested entity with the requested term."
                )
                resp.status = "abstained"
                return resp

            t = time.perf_counter()

            raw = self.llm.chat(
                build_messages(
                    question,
                    res.context,
                    res.notes,
                    res.subqueries,
                )
            )

            resp.timings["generation"] = (
                time.perf_counter()
                - t
            )

            self._finalize(
                resp,
                raw,
                res,
            )

            if (
                conversation is not None
                and resp.status != "error"
            ):
                conversation.add_turn(
                    ChatTurn(
                        user=question,
                        assistant=resp.answer,
                        sources=[
                            c.chunk.title
                            for c in res.context[:1]
                        ]
                        if resp.status == "ok"
                        else [],
                        user_entities=(
                            entity_phrases(question)
                            or (
                                conversation.turns[-1].user_entities
                                if conversation.turns
                                and res.decision
                                and res.decision.is_follow_up
                                else []
                            )
                        ),
                    )
                )

        except InsurellmError as exc:
            resp.status, resp.error = (
                "error",
                str(exc),
            )

        except Exception:
            # Application boundary: log details, show a safe message.
            log.exception(
                "Unexpected failure while answering"
            )

            resp.status, resp.error = (
                "error",
                "Something went wrong while answering. See the log.",
            )

        resp.timings["total"] = (
            time.perf_counter()
            - t0
        )

        return resp

    def _clean_invalid_citations(
        self,
        answer: str,
        known_labels: set[str],
    ) -> tuple[str, list[str]]:
        """Remove file-like citations that were not part of the retrieved context."""

        invalid: list[str] = []

        def replace(match: re.Match[str]) -> str:
            ref = match.group(1).strip()

            # Only treat .md bracket references as document citations.
            if (
                ref.lower().endswith(".md")
                and ref not in known_labels
            ):
                if ref not in invalid:
                    invalid.append(ref)

                return ""

            return match.group(0)

        cleaned = _CITATION_TOKEN.sub(
            replace,
            answer,
        )

        # Remove accidental whitespace left behind by deleted citations.
        cleaned = re.sub(
            r"[ \t]{2,}",
            " ",
            cleaned,
        )

        cleaned = re.sub(
            r"\n[ \t]+\n",
            "\n\n",
            cleaned,
        )

        return cleaned.strip(), invalid

    def _finalize(
        self,
        resp: RAGResponse,
        raw: str,
        res: RetrievalResult,
    ) -> None:
        labels = source_labels(
            res.context
        )

        by_label = {
            lab: src
            for src, lab in labels.items()
        }

        # Remove citations that do not belong to the retrieved context.
        # This prevents the LLM from inventing source filenames such as
        # "[Avery Lancaster.md]" when the actual evidence is from
        # "Maxine Thompson.md".
        cleaned, removed_invalid = (
            self._clean_invalid_citations(
                raw,
                set(by_label),
            )
        )

        resp.answer = cleaned

        valid, invalid = extract_citations(
            cleaned,
            set(by_label),
        )

        all_invalid = list(
            dict.fromkeys(
                removed_invalid + invalid
            )
        )

        if all_invalid:
            resp.warnings.append(
                "The answer referenced documents that were not retrieved: "
                + ", ".join(all_invalid)
            )

        if ABSTAIN_PHRASE in cleaned:
            resp.status = "abstained"
            return

        meta: dict[str, Chunk] = {
            c.chunk.source: c.chunk
            for c in res.context
        }

        chosen = (
            [by_label[v] for v in valid]
            or list(meta)
        )

        if not valid:
            resp.warnings.append(
                "The answer did not cite a source; listing retrieved documents."
            )

        resp.sources = [
            Source(
                label=labels[s],
                source=s,
                category=meta[s].category,
                title=meta[s].title,
                cited=bool(valid),
            )
            for s in dict.fromkeys(chosen)
        ]
