"""Structure-aware Markdown chunking with heading context and stable chunk IDs.

Strategy: split into heading sections; pack consecutive small sections together up to
``chunk_size``; split oversized sections on blank lines (keeps lists/tables whole), then
sentences, then a hard character split with overlap. Overlap is only used when a single
block has to be cut. Every chunk starts with a context line ``Title > Heading > Subheading``.
"""

from __future__ import annotations

import hashlib
import re

from ..schemas import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _sections(text: str) -> list[tuple[str, str]]:
    """Split into ``(heading_path, section_text)``; section_text includes its heading line."""
    sections: list[tuple[str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    current: list[str] = []
    path = ""
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            if "".join(current).strip():
                sections.append((path, current))
            level, title = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            current = [line]
        else:
            current.append(line)
    if "".join(current).strip():
        sections.append((path, current))
    return [(p, "\n".join(lines).strip()) for p, lines in sections]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def _split_block(block: str, size: int, overlap: int) -> list[str]:
    if len(block) <= size:
        return [block]
    sentences = [s for s in _SENTENCE.split(block) if s.strip()]
    if len(sentences) <= 1:
        return _hard_split(block, size, overlap)
    out: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.extend(_hard_split(s, size, overlap))
            continue
        if cur and len(cur) + 1 + len(s) > size:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if tail and " " in tail else ""
            cur = f"{tail} {s}".strip() if tail and len(tail) + len(s) + 1 <= size else s
        else:
            cur = f"{cur} {s}".strip() if cur else s
    if cur:
        out.append(cur)
    return out


def _units(sections: list[tuple[str, str]], size: int, overlap: int) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    for path, body in sections:
        if len(body) <= size:
            units.append((path, body))
            continue
        cur = ""
        for block in re.split(r"\n\s*\n", body):
            block = block.strip()
            if not block:
                continue
            if cur and len(cur) + 2 + len(block) <= size:
                cur = f"{cur}\n\n{block}"
                continue
            if cur:
                units.append((path, cur))
                cur = ""
            if len(block) > size:
                units.extend((path, p) for p in _split_block(block, size, overlap))
            else:
                cur = block
        if cur:
            units.append((path, cur))
    return units


def make_chunk_id(doc_id: str, index: int, text: str) -> str:
    return f"{doc_id}-{index:04d}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


def chunk_document(doc: Document, chunk_size: int, overlap: int, min_chars: int) -> list[Chunk]:
    sections = _sections(doc.text) or [("", doc.text)]
    units = _units(sections, chunk_size, overlap)
    # Pack consecutive units into chunks (reserve room for the context header).
    packed: list[tuple[str, str]] = []
    cur_path, cur_text = "", ""
    for path, text in units:
        if cur_text and len(cur_text) + 2 + len(text) <= chunk_size:
            cur_text = f"{cur_text}\n\n{text}"
        else:
            if cur_text:
                packed.append((cur_path, cur_text))
            cur_path, cur_text = path, text
    if cur_text:
        packed.append((cur_path, cur_text))
    # Merge a tiny trailing chunk into its predecessor instead of keeping a stub.
    if len(packed) > 1 and len(packed[-1][1]) < min_chars:
        p_path, p_text = packed[-2]
        packed[-2:] = [(p_path, f"{p_text}\n\n{packed[-1][1]}")]

    chunks: list[Chunk] = []
    for i, (path, body) in enumerate(packed):
        if not path:
            header = doc.title
        elif path.split(" > ")[0] == doc.title:
            header = path
        else:
            header = f"{doc.title} > {path}"
        text = f"{header}\n{body}"
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, i, text), doc_id=doc.doc_id,
                source=doc.source, category=doc.category, title=doc.title,
                chunk_index=i, text=text, heading_path=path, content_hash=doc.content_hash,
            )
        )
    return chunks
