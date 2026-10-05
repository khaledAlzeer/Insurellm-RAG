import pytest

from insurellm_rag.config import load_settings
from insurellm_rag.errors import ConfigError


def test_defaults_are_valid_and_paths_absolute():
    s = load_settings()
    assert s.knowledge_base_dir.is_absolute() and s.vector_db_dir.name == "vector_db"
    assert s.chat_model == "llama3.2" and 0 <= s.temperature <= 1


def test_invalid_values_raise_actionable_error():
    with pytest.raises(ConfigError) as e:
        load_settings(chunk_size=50, temperature=3)
    assert "CHUNK_SIZE" in str(e.value) and "TEMPERATURE" in str(e.value)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ConfigError):
        load_settings(chunk_size=300, chunk_overlap=200)


def test_env_override(monkeypatch):
    monkeypatch.setenv("INSURELLM_CHAT_MODEL", "other:1b")
    assert load_settings().chat_model == "other:1b"


def test_rerank_top_n_is_clamped():
    assert load_settings(rerank_top_n=50, fused_top_k=10).rerank_top_n == 10
