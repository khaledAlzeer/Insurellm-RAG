"""Shared fixtures. All data here is SYNTHETIC test content, not Insurellm knowledge."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from insurellm_rag.config import load_settings
from insurellm_rag.errors import GenerationError
from insurellm_rag.ingestion.vector_store import ChromaStore
from insurellm_rag.service import KnowledgeAssistant


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    model_name = "fake-hash-embedder"

    def __init__(self, dim: int = 64) -> None:
        self.dim, self.calls, self.texts_embedded = dim, 0, 0

    def embed(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class FakeLLM:
    """Scripted LLM: returns queued responses (or raises queued exceptions)."""

    def __init__(self, responses=None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def chat(self, messages, *, json_mode=False, max_tokens=None):
        self.calls.append({"messages": messages, "json": json_mode})
        if not self.responses:
            raise GenerationError("no scripted response")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def check(self):  # health-check compatible
        return None


KB_FILES = {
    "employees/Jane Fixture.md": (
        "# Jane Fixture\n\n## Summary\nJane Fixture is a Test Engineer at Examplecorp.\n\n"
        "## Awards\n- 2023: IIOTY Innovator Award winner.\n\n## Education\nStudied at Fakeville College.\n"
    ),
    "products/Widget Pro.md": (
        "# Widget Pro\n\nWidget Pro is a synthetic product used only in tests.\n\n"
        "## Features\n- Fast\n- Reliable\n- Cheap\n"
    ),
    "company/About.md": "# About Examplecorp\n\nExamplecorp was founded for testing purposes.\n",
    "contracts/Contract Alpha.md": "# Contract Alpha\n\nThis contract covers the Widget Pro rollout.\n",
}


@pytest.fixture
def kb(tmp_path):
    root = tmp_path / "knowledge-base"
    for rel, text in KB_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path, kb):
    return load_settings(
        knowledge_base_dir=kb, vector_db_dir=tmp_path / "vdb", cache_dir=tmp_path / "cache",
        rerank_enabled=False, chunk_size=300, chunk_overlap=40,
    )


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(settings, embedder):
    return ChromaStore(settings.vector_db_dir, settings.collection_name, embedder.model_name)


@pytest.fixture
def assistant(settings, embedder, store):
    s = settings.model_copy(update={"embedding_model": embedder.model_name})
    a = KnowledgeAssistant(s, embedder=embedder, llm=FakeLLM(), store=store)
    return a
