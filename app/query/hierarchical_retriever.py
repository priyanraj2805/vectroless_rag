"""Multi-stage hierarchical retrieval.

Stages: query analysis -> knowledge-graph query expansion -> a single global
FTS5 BM25 candidate search -> document-level scoring & dynamic selection ->
cross-encoder rerank -> neighbor-window expansion & merge -> structured chunks.

Entirely vectorless: retrieval is driven by SQLite FTS5's inverted index end
to end (one global MATCH query, not a scan of every document). The output
dict matches the contract expected by app.query.synthesizer.AnswerSynthesizer
(chunk tuples shaped (id, content, page_number, section_title, rank,
document_id[, chunk_index])).
"""
from typing import Dict, List, Optional, Tuple
from app.storage.database import Database
from app.query.query_analysis import build_plan_from_question
from app.query import reranker

JUNK_PREFIXES = (
    "title suppressed", "figure ", "fig.", "table ",
    "copyright", "arxiv:", "published in", "proceedings of",
)


def _is_quality_chunk(content: str) -> bool:
    if not content:
        return False
    lower = content.lower().strip()
    if any(lower.startswith(p) for p in JUNK_PREFIXES):
        return False
    return len(content.strip()) >= 30


def _sanitize_term(term: str) -> str:
    return term.replace('"', " ").strip()


class HierarchicalRetriever:
    def __init__(self, db: Database, settings=None):
        self.db = db
        if settings is None:
            from app.config import settings as default_settings
            settings = default_settings
        self.settings = settings

    def retrieve(self, question: str, document_ids: Optional[List[int]] = None) -> Dict[str, List]:
        plan = build_plan_from_question(question, num_docs=len(document_ids) if document_ids else 1)
        keywords = plan["keywords"]

        expanded_terms, matched_nodes = self._expand_with_graph(keywords, document_ids)
        all_terms = plan["search_terms"] + expanded_terms

        candidates = self._global_candidate_search(all_terms, document_ids, self.settings.retrieval_top_k_candidates)
        candidates = [c for c in candidates if _is_quality_chunk(c[1])]

        if not candidates:
            return {
                "nodes": matched_nodes,
                "chunks": [],
                "node_ids": [n[0] for n in matched_nodes],
                "doc_id_to_name": self._get_doc_id_to_name(document_ids),
            }

        doc_filenames = self._get_doc_id_to_name(list({c[5] for c in candidates}))
        doc_scores = self._score_documents(candidates, keywords, matched_nodes, doc_filenames)
        min_docs = self.settings.retrieval_min_documents
        if plan["query_type"] == "comparison":
            min_docs = max(min_docs, 2)
        selected_doc_ids = self._select_documents(doc_scores, min_docs)

        filtered = [c for c in candidates if c[5] in selected_doc_ids]
        filtered = self._deduplicate(filtered) or candidates

        if self.settings.retrieval_rerank_enabled:
            top_chunks = reranker.rerank(
                question, filtered, content_fn=lambda r: r[1], top_n=self.settings.retrieval_rerank_top_n
            )
        else:
            top_chunks = filtered[: self.settings.retrieval_rerank_top_n]

        final_chunks = self._expand_and_merge_neighbors(top_chunks)

        return {
            "nodes": matched_nodes,
            "chunks": final_chunks,
            "node_ids": [n[0] for n in matched_nodes],
            "doc_id_to_name": self._get_doc_id_to_name(selected_doc_ids or document_ids),
            "doc_scores": doc_scores,
        }

    # ---- Stage 1+2: KG-based query expansion ----
    def _expand_with_graph(self, keywords: List[str], document_ids: Optional[List[int]]) -> Tuple[List[str], List]:
        """Look up graph nodes matching the query's keywords, then one-hop
        traverse edges to pull in related entity names as extra search terms.
        No-ops gracefully when the graph is empty for these documents (lazy
        entity extraction may not have run for them yet)."""
        doc_filter = ""
        doc_params: tuple = ()
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            doc_filter = f" AND n.document_id IN ({placeholders})"
            doc_params = tuple(document_ids)

        matched_nodes = []
        seen_ids = set()
        for term in keywords[:8]:
            try:
                rows = self.db.execute(
                    f"""SELECT n.id, n.name, n.type, n.attributes, n.document_id
                        FROM nodes_fts fts JOIN nodes n ON n.id = fts.rowid
                        WHERE nodes_fts MATCH ?{doc_filter} LIMIT 5""",
                    (f'"{_sanitize_term(term)}"', *doc_params),
                ).fetchall()
            except Exception:
                rows = []
            for row in rows:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    matched_nodes.append(row)

        expanded_terms: List[str] = []
        for node in matched_nodes[:10]:
            node_id = node[0]
            try:
                related = self.db.execute(
                    """SELECT n2.id, n2.name, n2.type, n2.attributes, n2.document_id
                       FROM edges e JOIN nodes n2 ON (n2.id = e.target_id OR n2.id = e.source_id)
                       WHERE (e.source_id = ? OR e.target_id = ?) AND n2.id != ? LIMIT 5""",
                    (node_id, node_id, node_id),
                ).fetchall()
            except Exception:
                related = []
            for row in related:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    matched_nodes.append(row)
                    if len(expanded_terms) < 5:
                        expanded_terms.append(row[1])

        return expanded_terms, matched_nodes

    # ---- Stage 3: single global FTS5 BM25 candidate search ----
    def _global_candidate_search(self, terms: List[str], document_ids: Optional[List[int]], top_k: int) -> List[Tuple]:
        clean_terms = [_sanitize_term(t) for t in terms if _sanitize_term(t)]
        if not clean_terms:
            return []
        match_query = " OR ".join(f'"{t}"' for t in clean_terms)

        doc_filter = ""
        doc_params: tuple = ()
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            doc_filter = f" AND c.document_id IN ({placeholders})"
            doc_params = tuple(document_ids)

        try:
            rows = self.db.execute(
                f"""SELECT c.id, c.content, c.page_number, c.section_title,
                           bm25(chunks_fts) AS score, c.document_id, c.chunk_index
                    FROM chunks_fts fts JOIN chunks c ON c.id = fts.rowid
                    WHERE chunks_fts MATCH ?{doc_filter}
                    ORDER BY score
                    LIMIT ?""",
                (match_query, *doc_params, top_k),
            ).fetchall()
        except Exception:
            return []
        return rows

    # ---- Stage 4: document-level scoring + dynamic selection ----
    def _score_documents(self, candidates: List[Tuple], keywords: List[str],
                          matched_nodes: List, doc_filenames: Dict[int, str]) -> Dict[int, float]:
        by_doc: Dict[int, List[Tuple]] = {}
        for row in candidates:
            by_doc.setdefault(row[5], []).append(row)

        raw_bm25_best = {doc_id: min(r[4] for r in rows) for doc_id, rows in by_doc.items()}
        all_raw = list(raw_bm25_best.values())
        min_raw, max_raw = min(all_raw), max(all_raw)
        raw_span = max_raw - min_raw

        entity_counts: Dict[int, int] = {}
        for node in matched_nodes:
            doc_id = node[4] if len(node) > 4 else None
            if doc_id is not None:
                entity_counts[doc_id] = entity_counts.get(doc_id, 0) + 1
        max_entity = max(entity_counts.values()) if entity_counts else 0

        max_density = max(len(rows) for rows in by_doc.values())

        s = self.settings
        scores = {}
        for doc_id, rows in by_doc.items():
            bm25_component = ((max_raw - raw_bm25_best[doc_id]) / raw_span) if raw_span > 0 else 1.0

            heading_text = " ".join((r[3] or "").lower() for r in rows)
            heading_component = (
                sum(1 for kw in keywords if kw in heading_text) / len(keywords) if keywords else 0.0
            )

            entity_component = (entity_counts.get(doc_id, 0) / max_entity) if max_entity else 0.0

            filename = doc_filenames.get(doc_id, "").lower()
            metadata_component = (
                sum(1 for kw in keywords if kw in filename) / len(keywords) if keywords else 0.0
            )

            density_component = len(rows) / max_density if max_density else 0.0

            scores[doc_id] = (
                s.retrieval_weight_bm25 * bm25_component
                + s.retrieval_weight_heading * heading_component
                + s.retrieval_weight_entity * entity_component
                + s.retrieval_weight_metadata * metadata_component
                + s.retrieval_weight_density * density_component
            )
        return scores

    def _select_documents(self, doc_scores: Dict[int, float], min_documents: int) -> List[int]:
        if not doc_scores:
            return []
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_score = ranked[0][1]
        threshold = best_score * self.settings.retrieval_doc_score_threshold
        selected = [doc_id for doc_id, score in ranked if score >= threshold]
        if len(selected) < min_documents:
            selected = [doc_id for doc_id, _ in ranked[:min_documents]]
        if len(selected) > self.settings.retrieval_max_documents:
            selected = selected[: self.settings.retrieval_max_documents]
        return selected

    def _deduplicate(self, chunks: List[Tuple]) -> List[Tuple]:
        seen = set()
        unique = []
        for chunk in chunks:
            key = (chunk[1] or "")[:100].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique

    # ---- Stage 6: neighbor expansion (+/- window) and overlap merge ----
    def _expand_and_merge_neighbors(self, top_chunks: List[Tuple]) -> List[Tuple]:
        window = self.settings.retrieval_neighbor_window
        merged: Dict[int, Tuple] = {row[0]: row for row in top_chunks}

        indices_by_doc: Dict[int, set] = {}
        for row in top_chunks:
            doc_id = row[5]
            idx = row[6] if len(row) > 6 else None
            if idx is None:
                continue
            wanted = indices_by_doc.setdefault(doc_id, set())
            for offset in range(-window, window + 1):
                candidate_idx = idx + offset
                if candidate_idx >= 0:
                    wanted.add(candidate_idx)

        for doc_id, indices in indices_by_doc.items():
            if not indices:
                continue
            placeholders = ",".join("?" * len(indices))
            rows = self.db.execute(
                f"""SELECT id, content, page_number, section_title, 0 AS rank, document_id, chunk_index
                    FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders})""",
                (doc_id, *indices),
            ).fetchall()
            for row in rows:
                merged.setdefault(row[0], row)

        def sort_key(row):
            idx = row[6] if len(row) > 6 and row[6] is not None else float("inf")
            return (row[5], idx)

        return sorted(merged.values(), key=sort_key)

    def _get_doc_id_to_name(self, document_ids: Optional[List[int]]) -> Dict[int, str]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" * len(document_ids))
        rows = self.db.execute(
            f"SELECT id, filename FROM documents WHERE id IN ({placeholders})",
            tuple(document_ids),
        ).fetchall()
        return {row[0]: row[1] for row in rows}
