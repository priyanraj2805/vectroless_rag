from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.storage.database import Database
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.embedder import Embedder
from app.ingestion.graph_builder import GraphBuilder


class IngestionPipeline:
    def __init__(self, db: Database, groq_key: str = "", openrouter_key: str = "", opencode_key: str = "", redis_url: str = ""):
        self.db = db
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.extractor = EntityExtractor(groq_key=groq_key, openrouter_key=openrouter_key, opencode_key=opencode_key, redis_url=redis_url)
        self.embedder = Embedder()
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

            # Step 1: Insert all chunks into DB in one transaction (no per-row commit)
            chunk_ids = []
            for chunk in chunks:
                chunk_id = self.db.insert_chunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title", ""),
                    auto_commit=False,
                )
                chunk_ids.append(chunk_id)
            self.db.commit()

            # Step 2: Batch-embed all chunks in a single model forward pass
            texts = [chunk["content"] for chunk in chunks]
            vectors = self.embedder.embed_batch(texts)

            # Step 3: Insert all embeddings in one transaction
            for chunk_id, vector in zip(chunk_ids, vectors):
                self.db.insert_embedding(chunk_id, vector, auto_commit=False)
            self.db.commit()

            # Step 4: Extract entities in parallel (5 concurrent LLM calls)
            extractions = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures = {
                    pool.submit(self.extractor.extract, chunk["content"], doc_id): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    extractions[futures[future]] = future.result()

            # Step 5: Build graph from extractions
            for extraction in extractions:
                if extraction:
                    self.graph_builder.build_graph(extraction, document_id=doc_id)

            self.db.update_document_status(doc_id, "ready")
            return doc_id

        except Exception:
            self.db.rollback()
            self.db.update_document_status(doc_id, "error")
            raise
