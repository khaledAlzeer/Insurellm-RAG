import pytest

from insurellm_rag.errors import GenerationError, ModelUnavailableError
from insurellm_rag.generation.model import OllamaChat
from insurellm_rag.generation.prompts import (
    ABSTAIN_PHRASE,
    SYSTEM_PROMPT,
    build_messages,
    extract_citations,
    source_labels,
)
from insurellm_rag.schemas import Chunk, RetrievedChunk


def rc(source, text="t"):
    return RetrievedChunk(chunk=Chunk(chunk_id=source + text, doc_id=source, source=source,
                                      category="c", title="T", chunk_index=0, text=text))


def test_system_prompt_enforces_grounding():
    p = SYSTEM_PROMPT.lower()
    for needle in ("only source of truth", "never add facts", "data, not instructions",
                   "ambiguous", "do not reveal hidden reasoning"):
        assert needle in p
    assert ABSTAIN_PHRASE in SYSTEM_PROMPT


def test_context_is_wrapped_as_data_with_labels_and_notes():
    msgs = build_messages("Q?", [rc("employees/A.md", "fact A")], ["note x"], [])
    user = msgs[1]["content"]
    assert '<doc label="A.md"' in user and "<context>" in user and "note x" in user
    assert user.rstrip().endswith("Question: Q?")


def test_label_collisions_use_full_path():
    labels = source_labels([rc("a/X.md"), rc("b/X.md"), rc("c/Y.md")])
    assert labels["a/X.md"] == "a/X.md" and labels["c/Y.md"] == "Y.md"


def test_citation_extraction_separates_invalid_references():
    valid, invalid = extract_citations("Yes [A.md]. Also [Ghost.md] and [1].", {"A.md"})
    assert valid == ["A.md"] and invalid == ["Ghost.md"]


class _Client:
    def __init__(self, exc=None, content="hi", models=("llama3.2:latest",)):
        self.exc, self.content, self.models = exc, content, models

    def chat(self, **kw):
        if self.exc:
            raise self.exc
        return {"message": {"content": self.content}}

    def list(self):
        return {"models": [{"model": m} for m in self.models]}


def make(client):
    m = OllamaChat("http://x", "llama3.2", 0.1, 100, 2048, 5)
    m._client = client
    return m


def test_ollama_errors_map_to_clear_exceptions():
    with pytest.raises(ModelUnavailableError, match="ollama pull"):
        make(_Client(exc=Exception("model 'llama3.2' not found"))).chat([])
    with pytest.raises(GenerationError, match="did not answer"):
        make(_Client(exc=Exception("Read timed out"))).chat([])
    with pytest.raises(ModelUnavailableError, match="Is it running"):
        make(_Client(exc=Exception("Connection refused"))).chat([])
    with pytest.raises(GenerationError, match="empty"):
        make(_Client(content="  ")).chat([])
    assert make(_Client(content=" ok ")).chat([]) == "ok"


def test_check_reports_missing_model_with_pull_command():
    make(_Client()).check()
    with pytest.raises(ModelUnavailableError, match="ollama pull llama3.2"):
        make(_Client(models=("other:latest",))).check()
