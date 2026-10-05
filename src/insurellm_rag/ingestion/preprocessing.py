"""Deterministic text cleaning. Keeps names, numbers, acronyms, headings and list structure."""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = _HTML_COMMENT.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def derive_title(text: str, fallback: str) -> str:
    """First level-1 Markdown heading, otherwise the filename stem."""
    for line in text.split("\n"):
        m = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
        if m:
            return m.group(1).strip()
    return fallback
