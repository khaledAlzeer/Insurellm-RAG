"""Optional: needs live Ollama, downloaded models and the real KB. Run with: pytest -m integration"""
import pytest

from insurellm_rag.config import load_settings
from insurellm_rag.service import KnowledgeAssistant

pytestmark = pytest.mark.integration


def test_live_stack_answers_with_sources(tmp_path):
    s = load_settings(vector_db_dir=tmp_path / "vdb")  # temp DB: never touches the real index
    a = KnowledgeAssistant(s)
    if not any(s.knowledge_base_dir.rglob("*.md")):
        pytest.skip("Real Knowledge Base not present")
    a.ingest()
    r = a.ask("Who is the CEO?")
    assert r.status in {"ok", "abstained"} and r.error is None
