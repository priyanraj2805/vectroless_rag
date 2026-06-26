from typing import Dict, List
from app.storage.database import Database


class QueryExecutor:
    def __init__(self, db: Database):
        self.db = db

    def execute(self, plan: Dict) -> Dict[str, List]:
        nodes = self._search_nodes(plan)
        node_ids = [n[0] for n in nodes]

        related_nodes = []
        for node_id in node_ids:
            for edge_type in plan.get("traverse_edges", []):
                related = self.db.execute(
                    """SELECT n.id, n.name, n.type, n.attributes
                       FROM edges e
                       JOIN nodes n ON (n.id = e.target_id OR n.id = e.source_id)
                       WHERE (e.source_id = ? OR e.target_id = ?) AND e.type = ? AND n.id != ?""",
                    (node_id, node_id, edge_type, node_id),
                ).fetchall()
                related_nodes.extend(related)

        all_node_ids = list(set(node_ids + [r[0] for r in related_nodes]))
        chunks = self._search_chunks(plan, all_node_ids)

        return {
            "nodes": nodes + related_nodes,
            "chunks": chunks,
            "node_ids": all_node_ids,
        }

    def _search_nodes(self, plan: Dict) -> List:
        results = []
        entity_types = plan.get("entity_types", [])
        search_terms = plan.get("search_terms", [])

        for term in search_terms:
            try:
                found = self.db.search_nodes(f'"{term}"', limit=plan.get("max_results", 10))
                results.extend(found)
            except Exception:
                pass

        if entity_types and not results:
            placeholders = ",".join("?" * len(entity_types))
            results = self.db.execute(
                f"SELECT id, name, type, attributes FROM nodes WHERE type IN ({placeholders}) LIMIT ?",
                (*entity_types, plan.get("max_results", 10)),
            ).fetchall()

        return results

    def _search_chunks(self, plan: Dict, node_ids: List[int]) -> List:
        chunks = []
        search_terms = plan.get("search_terms", [])

        for term in search_terms:
            try:
                found = self.db.search_chunks(f'"{term}"', limit=plan.get("max_results", 10))
                chunks.extend(found)
            except Exception:
                pass

        if not chunks and node_ids:
            placeholders = ",".join("?" * len(node_ids))
            chunks = self.db.execute(
                f"""SELECT c.id, c.content, c.page_number, c.section_title, 0
                    FROM chunks c
                    JOIN edges e ON e.source_id = c.id OR e.target_id = c.id
                    WHERE e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders})
                    LIMIT ?""",
                (*node_ids, *node_ids, plan.get("max_results", 10)),
            ).fetchall()

        return chunks
