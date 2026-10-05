import pytest

from insurellm_rag.errors import EmbeddingError, KnowledgeBaseError, VectorStoreError
from insurellm_rag.ingestion.chunking import chunk_document
from insurellm_rag.ingestion.embeddings import validate_vectors
from insurellm_rag.ingestion.ingest_pipeline import run_ingestion
from insurellm_rag.ingestion.loader import discover_markdown, load_all, load_document
from insurellm_rag.ingestion.preprocessing import derive_title, normalize_text
from insurellm_rag.ingestion.vector_store import ChromaStore


def test_discovery_is_recursive_and_portable(kb, settings):
    docs = load_all(kb, settings.max_file_bytes).documents
    assert {d.source for d in docs} == {
        "employees/Jane Fixture.md", "products/Widget Pro.md", "company/About.md",
        "contracts/Contract Alpha.md"}
    assert all("\\" not in d.source and ":" not in d.source for d in docs)
    assert {d.category for d in docs} == {"employees", "products", "company", "contracts"}
    assert next(d for d in docs if d.source.startswith("employees")).title == "Jane Fixture"


def test_new_category_folder_needs_no_code_change(kb, settings):
    (kb / "newcat").mkdir()
    (kb / "newcat" / "X.md").write_text("# X\n\nhello there world", encoding="utf-8")
    assert "newcat" in {d.category for d in load_all(kb, settings.max_file_bytes).documents}


def test_missing_kb_raises_helpful_error(tmp_path):
    with pytest.raises(KnowledgeBaseError, match="knowledge-base"):
        discover_markdown(tmp_path / "nope")


def test_bad_files_are_skipped_not_fatal(kb, settings):
    (kb / "bad.md").write_bytes(b"\xff\xfe\x00bad")
    (kb / "empty.md").write_text("   \n", encoding="utf-8")
    res = load_all(kb, settings.max_file_bytes)
    assert res.skipped["bad.md"] and res.skipped["empty.md"] == "empty document"
    assert len(res.documents) == 4
    doc, why = load_document(kb / "company" / "About.md", kb, max_bytes=5)
    assert doc is None and "larger" in why


def test_preprocessing_is_deterministic_and_preserves_content():
    raw = "# T\r\n\r\n\r\n\r\nIIOTY  Award 2023 \t\r\n<!-- hidden -->- item\u200b"
    out = normalize_text(raw)
    assert out == normalize_text(out)
    assert "IIOTY" in out and "2023" in out and "hidden" not in out and "\r" not in out
    assert "\n\n\n" not in out
    assert derive_title("no heading", "fallback") == "fallback"


def test_chunking_keeps_heading_context_and_stable_ids(kb, settings):
    doc = next(d for d in load_all(kb, 10**6).documents if d.title == "Jane Fixture")
    a = chunk_document(doc, 300, 40, 20)
    b = chunk_document(doc, 300, 40, 20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.text.startswith("Jane Fixture") for c in a)
    assert any("IIOTY" in c.text for c in a)
    assert all(c.source == doc.source and c.category == "employees" for c in a)


def test_long_section_is_split_with_bounds_and_no_empty_chunks():
    from insurellm_rag.schemas import Document

    text = "# Big\n\n" + " ".join(f"Sentence number {i} is here." for i in range(200))
    doc = Document(doc_id="d", source="x/Big.md", category="x", title="Big", text=text,
                   content_hash="h")
    chunks = chunk_document(doc, 400, 60, 20)
    assert len(chunks) > 3
    assert all(c.text.strip() and len(c.text) <= 400 + 40 for c in chunks)


def test_doc_without_headings_and_tables_stay_together():
    from insurellm_rag.schemas import Document

    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    doc = Document(doc_id="d", source="Plain.md", category="uncategorized", title="Plain",
                   text=f"Intro paragraph.\n\n{table}", content_hash="h")
    chunks = chunk_document(doc, 300, 40, 20)
    assert len(chunks) == 1 and table in chunks[0].text


def test_validate_vectors():
    assert validate_vectors([[1.0, 0.0]], 1, None) == 2
    with pytest.raises(EmbeddingError):
        validate_vectors([[float("nan"), 0.0]], 1, None)
    with pytest.raises(EmbeddingError):
        validate_vectors([[1.0], [1.0, 2.0]], 2, None)


def test_incremental_ingestion_lifecycle(settings, kb, embedder, store):
    r1 = run_ingestion(settings, embedder, store)
    assert len(r1.new) == 4 and r1.embeddings_generated == r1.chunks_created > 0
    n_chunks = store.count()

    r2 = run_ingestion(settings, embedder, store)  # nothing changed -> nothing embedded
    assert len(r2.unchanged) == 4 and r2.embeddings_generated == 0 and store.count() == n_chunks

    (kb / "company" / "About.md").write_text("# About Examplecorp\n\nNew text now.", "utf-8")
    (kb / "contracts" / "Contract Alpha.md").unlink()
    before = embedder.texts_embedded
    r3 = run_ingestion(settings, embedder, store)
    assert r3.changed == ["company/About.md"] and r3.deleted == ["contracts/Contract Alpha.md"]
    assert embedder.texts_embedded - before == r3.embeddings_generated
    sources = {c.source for c in store.get_all_chunks()}
    assert "contracts/Contract Alpha.md" not in sources
    assert any("New text now" in c.text for c in store.get_all_chunks())


def test_dry_run_writes_nothing(settings, embedder, store):
    r = run_ingestion(settings, embedder, store, dry_run=True)
    assert r.dry_run and len(r.new) == 4 and store.count() == 0
    assert not (settings.vector_db_dir / "manifest.json").exists()


def test_rebuild_is_explicit_and_recreates(settings, embedder, store):
    run_ingestion(settings, embedder, store)
    r = run_ingestion(settings, embedder, store, rebuild=True)
    assert r.rebuilt and len(r.new) == 4 and store.count() > 0


def test_orphans_from_interrupted_run_are_reconciled(settings, embedder, store):
    run_ingestion(settings, embedder, store)
    chunk = store.get_all_chunks()[0]
    stray = chunk.model_copy(update={"chunk_id": "stray-0001-deadbeef"})
    store.upsert([stray], embedder.embed([stray.text]))
    r = run_ingestion(settings, embedder, store)
    assert r.orphans_removed == 1 and "stray-0001-deadbeef" not in store.existing_ids()


def test_lost_vectors_trigger_reprocessing(settings, embedder, store):
    run_ingestion(settings, embedder, store)
    store.delete(list(store.existing_ids())[:1])
    r = run_ingestion(settings, embedder, store)
    assert len(r.changed) == 1 and store.count() > 0


def test_embedding_model_mismatch_is_detected(settings, embedder, store):
    with pytest.raises(VectorStoreError, match="rebuild|Rebuild"):
        ChromaStore(settings.vector_db_dir, settings.collection_name, "another-model")


def test_empty_kb_ingests_cleanly(tmp_path, settings, embedder, store):
    empty = tmp_path / "empty-kb"
    empty.mkdir()
    s = settings.model_copy(update={"knowledge_base_dir": empty})
    r = run_ingestion(s, embedder, store)
    assert r.discovered == 0 and store.count() == 0
