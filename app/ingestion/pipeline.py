from pathlib import Path
from app.storage.database import Database
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.graph_builder import GraphBuilder


class IngestionPipeline:
    def __init__(self, db: Database, groq_key: str, model: str = "llama3-8b-8192"):
        self.db = db
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.extractor = EntityExtractor(api_key=groq_key, model=model)
        self.graph_builder = GraphBuilder(db)

    def ingest(self, pdf_path: str) -> int:
        filename = Path(pdf_path).name
        doc_id = self.db.insert_document(filename)

        try:
            parsed = self.parser.extract_text(pdf_path)
            self.db.execute(
                "UPDATE documents SET page_count = ? WHERE id = ?",
                (parsed["page_count"], doc_id),
            )
            self.db.commit()

            chunks = self.chunker.chunk_from_pages(parsed["pages"], document_id=doc_id)

            for chunk in chunks:
                chunk_id = self.db.insert_chunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title", ""),
                )

                extraction = self.extractor.extract(chunk["content"], document_id=doc_id)
                node_ids = self.graph_builder.build_graph(extraction, document_id=doc_id)

                for node_id in node_ids:
                    self.db.insert_edge(
                        source_id=chunk_id,
                        target_id=node_id,
                        edge_type="mentions",
                    )

            self.db.update_document_status(doc_id, "ready")
            return doc_id

        except Exception as e:
            self.db.update_document_status(doc_id, "error")
            raise e
