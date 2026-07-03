from typing import Dict, List, Optional
from app.storage.database import Database
from app.ingestion.embedder import Embedder


class QueryExecutor:
    def __init__(self, db: Database):
        self.db = db
        self.embedder = Embedder()

    def execute(self, plan: Dict, document_ids: Optional[List[int]] = None) -> Dict[str, List]:
        nodes = self._search_nodes(plan, document_ids)
        node_ids = [n[0] for n in nodes]

        related_nodes = []
        for node_id in node_ids:
            for edge_type in plan.get("traverse_edges", []):
                doc_filter = ""
                params = (node_id, node_id, edge_type, node_id)
                if document_ids:
                    placeholders = ",".join("?" * len(document_ids))
                    doc_filter = f" AND n.document_id IN ({placeholders})"
                    params = (node_id, node_id, edge_type, node_id, *document_ids)
                related = self.db.execute(
                    f"""SELECT n.id, n.name, n.type, n.attributes
                       FROM edges e
                       JOIN nodes n ON (n.id = e.target_id OR n.id = e.source_id)
                       WHERE (e.source_id = ? OR e.target_id = ?) AND e.type = ? AND n.id != ?{doc_filter}""",
                    params,
                ).fetchall()
                related_nodes.extend(related)

        all_node_ids = list(set(node_ids + [r[0] for r in related_nodes]))

        # Run FTS keyword search and semantic search, then merge
        fts_chunks = self._search_chunks_fts(plan, document_ids)
        semantic_chunks = self._search_chunks_semantic(plan, document_ids)
        chunks = self._rrf_merge(fts_chunks, semantic_chunks, limit=plan.get("max_results", 10))

        return {
            "nodes": nodes + related_nodes,
            "chunks": chunks,
            "node_ids": all_node_ids,
        }

    def _search_nodes(self, plan: Dict, document_ids: Optional[List[int]] = None) -> List:
        results = []
        entity_types = plan.get("entity_types", [])
        search_terms = plan.get("search_terms", [])

        doc_filter = ""
        doc_params = ()
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            doc_filter = f" AND n.document_id IN ({placeholders})"
            doc_params = tuple(document_ids)

        for term in search_terms:
            for query in [f'"{term}"', term]:
                try:
                    found = self.db.execute(
                        f"SELECT n.id, n.name, n.type, n.attributes, rank FROM nodes_fts fts JOIN nodes n ON n.id = fts.rowid WHERE nodes_fts MATCH ?{doc_filter} ORDER BY rank LIMIT ?",
                        (query, *doc_params, plan.get("max_results", 10)),
                    ).fetchall()
                    if found:
                        results.extend(found)
                        break
                except Exception:
                    pass

        if entity_types and not results:
            placeholders = ",".join("?" * len(entity_types))
            if document_ids:
                doc_id_placeholders = ",".join("?" * len(document_ids))
                results = self.db.execute(
                    f"SELECT id, name, type, attributes FROM nodes WHERE type IN ({placeholders}) AND document_id IN ({doc_id_placeholders}) LIMIT ?",
                    (*entity_types, *document_ids, plan.get("max_results", 10)),
                ).fetchall()
            else:
                results = self.db.execute(
                    f"SELECT id, name, type, attributes FROM nodes WHERE type IN ({placeholders}) LIMIT ?",
                    (*entity_types, plan.get("max_results", 10)),
                ).fetchall()

        return results

    def _search_chunks_fts(self, plan: Dict, document_ids: Optional[List[int]] = None) -> List:
        """BM25 keyword search over chunk text."""
        chunks = []
        search_terms = plan.get("search_terms", [])
        limit = plan.get("max_results", 10)

        doc_filter = ""
        doc_params = ()
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            doc_filter = f" AND c.document_id IN ({placeholders})"
            doc_params = tuple(document_ids)

        for term in search_terms:
            for query in [f'"{term}"', term]:
                try:
                    found = self.db.execute(
                        f"SELECT c.id, c.content, c.page_number, c.section_title, rank FROM chunks_fts fts JOIN chunks c ON c.id = fts.rowid WHERE chunks_fts MATCH ?{doc_filter} ORDER BY rank LIMIT ?",
                        (query, *doc_params, limit),
                    ).fetchall()
                    if found:
                        chunks.extend(found)
                        break
                except Exception:
                    pass

        if not chunks:
            if document_ids:
                placeholders = ",".join("?" * len(document_ids))
                chunks = self.db.execute(
                    f"SELECT id, content, page_number, section_title, 0 FROM chunks WHERE document_id IN ({placeholders}) ORDER BY id LIMIT ?",
                    (*document_ids, limit),
                ).fetchall()
            else:
                chunks = self.db.execute(
                    "SELECT id, content, page_number, section_title, 0 FROM chunks ORDER BY id LIMIT ?",
                    (limit,),
                ).fetchall()

        return chunks

    def _search_chunks_semantic(self, plan: Dict, document_ids: Optional[List[int]] = None) -> List:
        """Cosine similarity search over stored embeddings."""
        search_terms = plan.get("search_terms", [])
        limit = plan.get("max_results", 10)

        if not search_terms:
            return []

        # Embed the combined search query
        query_text = " ".join(search_terms)
        query_vec = self.embedder.embed_query(query_text)

        # Load all stored embeddings (filtered by document if needed)
        rows = self.db.get_all_embeddings(document_ids)
        if not rows:
            return []

        # Score each chunk by cosine similarity
        scored = []
        for chunk_id, vector_blob, content, page_number, section_title in rows:
            chunk_vec = self.embedder.from_bytes(vector_blob)
            score = self.embedder.cosine_similarity(query_vec, chunk_vec)
            scored.append((chunk_id, content, page_number, section_title, score))

        # Return top-N sorted by similarity score descending
        scored.sort(key=lambda x: x[4], reverse=True)
        return scored[:limit]

    def _rrf_merge(self, fts_chunks: List, semantic_chunks: List, limit: int = 10, k: int = 60) -> List:
        """
        Reciprocal Rank Fusion — merges two ranked lists into one.
        Score = 1/(k + rank_fts) + 1/(k + rank_semantic)
        Higher score = chunk appeared high in both lists.
        """
        scores: Dict[int, float] = {}
        chunk_data: Dict[int, tuple] = {}

        for rank, chunk in enumerate(fts_chunks):
            chunk_id = chunk[0]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk_data[chunk_id] = chunk

        for rank, chunk in enumerate(semantic_chunks):
            chunk_id = chunk[0]
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            chunk_data[chunk_id] = chunk

        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:limit]
        return [chunk_data[cid] for cid in ranked_ids]
