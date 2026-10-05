"""Bounded conversation state, follow-up and topic-shift detection.

Assistant text is never fed back as fact: only the previous *user question*, its named
entities and the titles of sources that were actually retrieved are carried forward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval.query import entity_phrases
from .schemas import ChatTurn


_REFERENCE = re.compile(
    r"\b(he|she|him|her|hers|his|they|them|their|theirs|it|its|this|that|these|those|"
    r"there|then|former|latter|same|such)\b",
    re.IGNORECASE,
)

_CONTINUATION = re.compile(
    r"^\s*(and|also|what about|how about|why|so|then|but)\b",
    re.IGNORECASE,
)

# Words that can be capitalized at the beginning of a sentence
# but are not actual entities.
_NON_ENTITY_WORDS = frozenset(
    {
        "now",
        "then",
        "also",
        "so",
        "and",
        "but",
        "what",
        "how",
        "why",
        "please",
        "tell",
        "today",
        "here",
        "there",
    }
)


@dataclass
class FollowUpDecision:
    is_follow_up: bool
    focus: list[str] = field(default_factory=list)  # entities from the previous turn
    previous_question: str = ""
    reason: str = ""


@dataclass
class ConversationState:
    max_turns: int = 6
    turns: list[ChatTurn] = field(default_factory=list)

    # Entity resolved from a follow-up query.
    # This is carried into the next stored ChatTurn so multi-turn
    # follow-ups keep the same subject.
    _pending_entities: list[str] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def add_turn(self, turn: ChatTurn) -> None:
        """Store a turn while preserving any entity resolved for this turn."""

        if self._pending_entities:
            existing = list(turn.user_entities)

            # Put the resolved entity first so it becomes the main focus
            # for the next follow-up.
            turn.user_entities = list(
                dict.fromkeys(
                    self._pending_entities + existing
                )
            )

            self._pending_entities.clear()

        self.turns.append(turn)
        del self.turns[:-self.max_turns]

    def reset(self) -> None:
        """Clear conversation history and any pending resolved entities."""

        self.turns.clear()
        self._pending_entities.clear()

    @staticmethod
    def _clean_entities(entities: list[str]) -> list[str]:
        """Remove discourse words that were incorrectly extracted as entities."""

        cleaned = []

        for entity in entities:
            value = entity.strip()

            if not value:
                continue

            if value.lower() in _NON_ENTITY_WORDS:
                continue

            if value not in cleaned:
                cleaned.append(value)

        return cleaned

    def _get_previous_focus(self, previous: ChatTurn) -> list[str]:
        """Return the most relevant entities from the previous user question."""

        # Start with the entities already extracted from the previous turn.
        entities = self._clean_entities(previous.user_entities)

        # If no usable entity was stored, extract entities directly from
        # the previous user question.
        if not entities:
            extracted = self._clean_entities(
                entity_phrases(previous.user)
            )
            entities = extracted

        # Sources remain a fallback only when no real entity exists.
        if not entities:
            entities = list(previous.sources[:1])

        return list(dict.fromkeys(entities))

    def decide(self, query: str) -> FollowUpDecision:
        """Decide whether ``query`` depends on the previous turn."""

        if not self.turns:
            return FollowUpDecision(
                False,
                reason="no history",
            )

        previous = self.turns[-1]

        own_entities = {
            entity.lower()
            for entity in entity_phrases(query)
            if entity.lower() not in _NON_ENTITY_WORDS
        }

        words = query.split()

        has_reference = (
            bool(_REFERENCE.search(query))
            or bool(_CONTINUATION.match(query))
        )

        focus = self._get_previous_focus(previous)

        focus_text = " ".join(
            entity.lower()
            for entity in focus
        )

        # A clearly different named entity means the user changed topic.
        if own_entities and not any(
            entity in focus_text
            for entity in own_entities
        ):
            # A query such as "What did she do at Foo Corp?"
            # still refers to the previous subject.
            if not re.search(
                r"\b(he|she|his|her|they|their)\b",
                query,
                re.IGNORECASE,
            ):
                return FollowUpDecision(
                    False,
                    reason="new named entity",
                )

        if has_reference or (
            len(words) <= 4 and not own_entities
        ):
            return FollowUpDecision(
                True,
                focus=focus,
                previous_question=previous.user,
                reason="reference or short query",
            )

        return FollowUpDecision(
            False,
            reason="self-contained query",
        )

    def resolve_deterministic(
        self,
        query: str,
        decision: FollowUpDecision,
    ) -> str | None:
        """Resolve follow-up references into natural standalone queries."""

        # Clear any stale pending entity before resolving a new query.
        self._pending_entities.clear()

        if not decision.is_follow_up or not decision.focus:
            return None

        # The first focus entity is the main subject of the previous turn.
        entity = decision.focus[0]

        if entity.lower() in query.lower():
            return None

        # Remember the resolved entity so add_turn() can preserve it
        # as the subject of the current conversation turn.
        self._pending_entities = [entity]

        cleaned = query.rstrip("?").strip()

        # Convert common possessive follow-ups into natural entity-first
        # queries so retrieval receives a query that resembles a direct question.
        #
        # Example:
        # "What is her role?" + "Lisa Anderson"
        # -> "What is Lisa Anderson's role?"
        if re.search(
            r"\b(her|his|their)\s+role\b",
            cleaned,
            re.IGNORECASE,
        ):
            return (
                re.sub(
                    r"\b(her|his|their)\s+role\b",
                    f"{entity}'s role",
                    cleaned,
                    count=1,
                    flags=re.IGNORECASE,
                )
                + "?"
            )

        # Generic follow-ups keep the original query and append the entity.
        return f"{cleaned} {entity}"
