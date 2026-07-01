from typing import Dict, List, Optional
from app.storage.database import Database


class QueryExecutor:
    def __init__(self, db: Database):
        self.db = db

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
        chunks = self._search_chunks(plan, all_node_ids, document_ids)

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

    def _search_chunks(self, plan: Dict, node_ids: List[int], document_ids: Optional[List[int]] = None) -> List:
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
