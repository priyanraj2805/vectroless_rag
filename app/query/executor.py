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

        # Run FTS keyword search and semantic search, then merge with RRF
        limit = plan.get("max_results", 10)
        fts_chunks = self._search_chunks_fts(plan, document_ids)
        semantic_chunks = self._search_chunks_semantic(plan, document_ids)
        chunks = self._rrf_merge(fts_chunks, semantic_chunks, limit=limit)

        # Guarantee every selected document contributes chunks (critical for multi-doc summaries)
        if document_ids and len(document_ids) > 1:
            min_per_doc = max(3, limit // len(document_ids))
            chunks = self._ensure_per_doc_coverage(chunks, document_ids, min_per_doc=min_per_doc)

        return {
            "nodes": nodes + related_nodes,
            "chunks": chunks,
            "node_ids": all_node_ids,
        }

    def _ensure_per_doc_coverage(self, chunks: List, document_ids: List[int], min_per_doc: int = 3) -> List:
        """
        Ensures each selected document has at least min_per_doc chunks in the result.
        If a document is under-represented (e.g. crowded out by a dominant doc in BM25),
        fetch its top chunks directly by rowid and append them.
        """
        # Count how many chunks each doc already has in the merged result
        # We need to look up document_id for each chunk_id
        chunk_ids = [c[0] for c in chunks]
        doc_chunk_count: Dict[int, int] = {doc_id: 0 for doc_id in document_ids}

        if chunk_ids:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = self.db.execute(
                f"SELECT id, document_id FROM chunks WHERE id IN ({placeholders})",
                tuple(chunk_ids),
            ).fetchall()
            for row in rows:
                chunk_id, doc_id = row
                if doc_id in doc_chunk_count:
                    doc_chunk_count[doc_id] = doc_chunk_count.get(doc_id, 0) + 1

        existing_chunk_ids = set(chunk_ids)
        extra_chunks = []

        for doc_id in document_ids:
            have = doc_chunk_count.get(doc_id, 0)
            need = min_per_doc - have
            if need <= 0:
                continue

            # Fetch the top chunks from this doc that aren't already in the result
            rows = self.db.execute(
                "SELECT id, content, page_number, section_title FROM chunks WHERE document_id = ? ORDER BY id LIMIT ?",
                (doc_id, need + len(existing_chunk_ids)),  # over-fetch to account for dedup
            ).fetchall()

            added = 0
            for row in rows:
                if added >= need:
                    break
                if row[0] not in existing_chunk_ids:
                    extra_chunks.append((row[0], row[1], row[2], row[3], 0))
                    existing_chunk_ids.add(row[0])
                    added += 1

        return chunks + extra_chunks

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

        query_text = " ".join(search_terms)
        query_vec = self.embedder.embed_query(query_text)

        rows = self.db.get_all_embeddings(document_ids)
        if not rows:
            return []

        scored = []
        for chunk_id, vector_blob, content, page_number, section_title in rows:
            chunk_vec = self.embedder.from_bytes(vector_blob)
            score = self.embedder.cosine_similarity(query_vec, chunk_vec)
            scored.append((chunk_id, content, page_number, section_title, score))

        scored.sort(key=lambda x: x[4], reverse=True)
        return scored[:limit]

    def _rrf_merge(self, fts_chunks: List, semantic_chunks: List, limit: int = 10, k: int = 60) -> List:
        """
        Reciprocal Rank Fusion — merges two ranked lists into one.
        Score = 1/(k + rank_fts) + 1/(k + rank_semantic)
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
