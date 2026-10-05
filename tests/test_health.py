from insurellm_rag.service import KnowledgeAssistant


def test_health_reports_empty_state_without_pretending(assistant):
    checks = {c.name: c for c in assistant.health()}
    assert checks["vector_db"].status == "warn" and "ingest" in checks["vector_db"].detail
    assert checks["knowledge_base"].status == "ok"


def test_missing_kb_directory_is_a_warning_not_a_crash(assistant, tmp_path):
    assistant.settings = assistant.settings.model_copy(update={"knowledge_base_dir": tmp_path / "x"})
    assert {c.name: c for c in assistant.health()}["knowledge_base"].status == "warn"


def test_constructing_assistant_loads_nothing(settings):
    a = KnowledgeAssistant(settings)
    assert a._embedder is None and a._store is None and a._pipeline is None
