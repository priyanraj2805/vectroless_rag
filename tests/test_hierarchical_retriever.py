import os
import tempfile
import types
import pytest

from app.storage.database import Database
from app.query.hierarchical_retriever import HierarchicalRetriever
from app.query import reranker
from app.query.query_analysis import build_plan_from_question, detect_intent


def make_settings(**overrides):
    defaults = dict(
        retrieval_top_k_candidates=50,
        retrieval_doc_score_threshold=0.4,
        retrieval_min_documents=1,
        retrieval_max_documents=5,
        retrieval_rerank_top_n=10,
        retrieval_rerank_enabled=False,  # deterministic BM25 ordering by default in tests
        retrieval_neighbor_window=1,
        retrieval_weight_bm25=0.40,
        retrieval_weight_heading=0.15,
        retrieval_weight_entity=0.20,
        retrieval_weight_metadata=0.10,
        retrieval_weight_density=0.15,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    yield database
    database.close()
    os.unlink(db_path)


def test_query_analysis_extracts_keywords_and_intent():
    plan = build_plan_from_question("What is the difference between revenue and profit?")
    assert "revenue" in plan["keywords"]
    assert plan["query_type"] == "comparison"
    assert detect_intent("Summarize this document") == "summary"
    assert detect_intent("What is machine learning?") == "definition"


def test_basic_retrieve_returns_matching_chunks(db):
    doc_id = db.insert_document("test.pdf", 1)
    db.insert_chunk(doc_id, "Acme Corp reported strong revenue growth this quarter.", 1, "Financial Results", chunk_index=0)
    db.insert_chunk(doc_id, "Unrelated paragraph about the weather in Paris.", 2, "Misc", chunk_index=1)

    retriever = HierarchicalRetriever(db, settings=make_settings())
    result = retriever.retrieve("What was Acme Corp's revenue?")

    assert len(result["chunks"]) >= 1
    assert any("Acme Corp" in c[1] for c in result["chunks"])
    assert result["doc_id_to_name"].get(doc_id) == "test.pdf"


def test_document_scoring_selects_only_relevant_document(db):
    relevant_doc = db.insert_document("finance.pdf", 1)
    db.insert_chunk(relevant_doc, "Acme Corp revenue grew by 20 percent in the fiscal year.", 1, "Acme Corp Revenue", chunk_index=0)
    db.insert_chunk(relevant_doc, "Acme Corp revenue guidance was raised for next year.", 2, "Acme Corp Revenue", chunk_index=1)

    weak_doc = db.insert_document("cooking.pdf", 1)
    db.insert_chunk(weak_doc, "Acme Corp was mentioned once in passing during dinner.", 1, "Misc", chunk_index=0)

    retriever = HierarchicalRetriever(db, settings=make_settings(retrieval_doc_score_threshold=0.6, retrieval_min_documents=1))
    result = retriever.retrieve("Tell me about Acme Corp revenue")

    doc_ids_used = {c[5] for c in result["chunks"]}
    assert relevant_doc in doc_ids_used
    assert weak_doc not in doc_ids_used


def test_document_scoring_min_documents_floor(db):
    doc_a = db.insert_document("a.pdf", 1)
    db.insert_chunk(doc_a, "Acme Corp revenue report for the fiscal year.", 1, "Revenue", chunk_index=0)
    doc_b = db.insert_document("b.pdf", 1)
    db.insert_chunk(doc_b, "Acme Corp was mentioned briefly in this unrelated passage.", 1, "Misc", chunk_index=0)

    retriever = HierarchicalRetriever(db, settings=make_settings(retrieval_doc_score_threshold=0.99, retrieval_min_documents=2))
    result = retriever.retrieve("Acme Corp revenue")

    doc_ids_used = {c[5] for c in result["chunks"]}
    assert doc_a in doc_ids_used
    assert doc_b in doc_ids_used


def test_neighbor_expansion_and_merge(db):
    doc_id = db.insert_document("doc.pdf", 1)
    for i in range(5):
        db.insert_chunk(doc_id, f"Chunk number {i} content about topic X.", 1, "Section", chunk_index=i)

    rows = db.execute("SELECT id, content, page_number, section_title, 0, document_id, chunk_index FROM chunks WHERE chunk_index = 2").fetchall()
    top_chunks = rows  # simulate a single reranked top chunk at index 2

    retriever = HierarchicalRetriever(db, settings=make_settings(retrieval_neighbor_window=1))
    merged = retriever._expand_and_merge_neighbors(top_chunks)

    merged_indices = sorted(c[6] for c in merged)
    assert merged_indices == [1, 2, 3]


def test_neighbor_expansion_dedupes_overlapping_windows(db):
    doc_id = db.insert_document("doc.pdf", 1)
    for i in range(5):
        db.insert_chunk(doc_id, f"Chunk number {i}.", 1, "Section", chunk_index=i)

    rows = db.execute(
        "SELECT id, content, page_number, section_title, 0, document_id, chunk_index FROM chunks WHERE chunk_index IN (1, 2)"
    ).fetchall()

    retriever = HierarchicalRetriever(db, settings=make_settings(retrieval_neighbor_window=1))
    merged = retriever._expand_and_merge_neighbors(rows)

    merged_indices = [c[6] for c in merged]
    assert merged_indices == sorted(merged_indices)
    assert len(merged_indices) == len(set(merged_indices))
    assert merged_indices == [0, 1, 2, 3]


def test_kg_query_expansion_finds_related_entities(db):
    doc_id = db.insert_document("kg.pdf", 1)
    acme_id = db.insert_node("Acme Corp", "organization", {}, doc_id)
    jane_id = db.insert_node("Jane Smith", "person", {}, doc_id)
    db.insert_edge(jane_id, acme_id, "authored_by")
    # nodes_fts has no insert trigger — rebuild manually to simulate the
    # per-request Database() construction that syncs it in production.
    db.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
    db.commit()

    retriever = HierarchicalRetriever(db, settings=make_settings())
    expanded_terms, matched_nodes = retriever._expand_with_graph(["acme", "corp"], None)

    matched_names = {n[1] for n in matched_nodes}
    assert "Acme Corp" in matched_names
    assert "Jane Smith" in matched_names
    assert "Jane Smith" in expanded_terms


def test_kg_expansion_gracefully_degrades_when_graph_empty(db):
    retriever = HierarchicalRetriever(db, settings=make_settings())
    expanded_terms, matched_nodes = retriever._expand_with_graph(["nothing", "here"], None)
    assert expanded_terms == []
    assert matched_nodes == []


def test_reranker_falls_back_without_model(monkeypatch):
    monkeypatch.setattr(reranker, "_get_model", lambda: None)
    items = [(1, "b content"), (2, "a content"), (3, "c content")]
    result = reranker.rerank("query", items, content_fn=lambda i: i[1], top_n=2)
    assert result == items[:2]


def test_reranker_reorders_with_mocked_cross_encoder(monkeypatch):
    class FakeModel:
        def predict(self, pairs):
            # Score higher for pairs whose content contains "match"
            return [1.0 if "match" in content else 0.0 for _, content in pairs]

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeModel())
    items = [(1, "no relevance here"), (2, "this is a match"), (3, "also irrelevant")]
    result = reranker.rerank("query", items, content_fn=lambda i: i[1], top_n=2)

    assert result[0][0] == 2


def test_reranker_empty_items_returns_empty(monkeypatch):
    assert reranker.rerank("q", [], content_fn=lambda i: i, top_n=5) == []
