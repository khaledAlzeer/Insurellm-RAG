"""Query understanding: selective rewriting/expansion, validation, multi-hop detection."""

from __future__ import annotations

import json
import re

from .lexical import query_terms

_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{2,}\b")
_NUMBER = re.compile(r"\b\d[\d,.\-/]*\b")
_CAP_WORD = re.compile(r"\b[A-Z][\w'\u2019-]+\b")
_QUESTION_WORDS = frozenset(
    ["who", "what", "which", "when", "where", "how", "why", "is", "are", "was", "were", "does", "do", "did", "tell", "list", "give", "show", "the", "a", "an", "in", "on", "at", "of", "for", "and", "or", "can", "could", "would", "should", "please", "i", "we", "you"]
)
_MULTIHOP = re.compile(
    r"\b(and then|both|compare|difference between|who (?:works|worked|leads|led|manages|managed)"
    r".*\b(?:that|which|whose)\b|which .* (?:of|did) the .* (?:who|that)|"
    r"(?:and|also) (?:what|who|which|when|where|how)\b|of the (?:project|product|contract|company)"
    r" (?:that|which)|whose\b)",
    re.IGNORECASE,
)


def extract_entities(text: str) -> list[str]:
    """Acronyms, numbers and capitalised words (excluding question words) - must be preserved."""
    ents = _ACRONYM.findall(text) + _NUMBER.findall(text)
    ents += [w for w in _CAP_WORD.findall(text) if w.lower() not in _QUESTION_WORDS]
    return list(dict.fromkeys(ents))


def entity_phrases(text: str) -> list[str]:
    """Multi-word proper-noun phrases, e.g. 'Avery Lancaster'."""
    out, cur = [], []
    for tok in re.findall(r"[\w'\u2019-]+", text):
        if tok[0].isupper() and tok.lower() not in _QUESTION_WORDS:
            cur.append(tok)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return list(dict.fromkeys(out))


def is_multihop_candidate(query: str) -> bool:
    """Cheap heuristic gate: the planner (an LLM call) runs only when this is True."""
    if len(query.split()) < 8:
        return False
    return bool(_MULTIHOP.search(query)) or query.count("?") > 1


def should_expand(query: str) -> bool:
    words = query.split()
    return bool(_ACRONYM.search(query)) or len(words) <= 4


def preserves_entities(original: str, candidate: str) -> bool:
    cand = candidate.lower()
    return all(e.lower() in cand for e in extract_entities(original))


def validate_rewrite(original: str, rewritten: str | None, max_ratio: float = 3.0) -> str | None:
    """Return the cleaned rewrite or ``None`` (caller falls back to the original query)."""
    if not rewritten:
        return None
    rewritten = rewritten.strip().strip('"').strip()
    if not rewritten or rewritten.lower() == original.strip().lower():
        return None
    if len(rewritten) > max(len(original) * max_ratio, 80) or "\n" in rewritten:
        return None
    if not preserves_entities(original, rewritten):
        return None
    return rewritten


def validate_expansions(original: str, items: object, limit: int) -> list[str]:
    """Bounded, de-duplicated expansions that keep at least one key term of the question."""
    if not isinstance(items, list) or limit <= 0:
        return []
    key = set(query_terms(original))
    ents = [e.lower() for e in extract_entities(original)]
    seen = {original.strip().lower()}
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        q = item.strip().strip('"')
        low = q.lower()
        if not q or low in seen or len(q) > max(len(original) * 3, 80) or "\n" in q:
            continue
        if ents and not any(e in low for e in ents) and not (key & set(query_terms(q))):
            continue
        seen.add(low)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def parse_json_object(raw: str) -> dict | None:
    """Tolerant JSON extraction from model output."""
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def validate_subqueries(original: str, items: object, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if isinstance(item, str) and item.strip() and item.strip() not in out and len(item) < 300:
            out.append(item.strip())
    return out[:limit] if len(out) >= 2 else []
