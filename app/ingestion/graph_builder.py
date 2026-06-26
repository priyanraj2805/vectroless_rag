from typing import Dict, List
from app.storage.database import Database


class GraphBuilder:
    def __init__(self, db: Database):
        self.db = db

    def build_graph(self, extraction: Dict[str, List], document_id: int) -> List[int]:
        name_to_id = {}

        for entity in extraction.get("entities", []):
            node_id = self.db.insert_node(
                name=entity["name"],
                node_type=entity["type"],
                attributes=entity.get("attributes", {}),
                document_id=document_id,
            )
            name_to_id[entity["name"]] = node_id

        for rel in extraction.get("relationships", []):
            source_id = name_to_id.get(rel["source"])
            target_id = name_to_id.get(rel["target"])
            if source_id and target_id:
                self.db.insert_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=rel["type"],
                    attributes=rel.get("attributes", {}),
                )

        return list(name_to_id.values())
