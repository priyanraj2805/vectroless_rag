from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.storage.database import Database
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.graph_builder import GraphBuilder


class IngestionPipeline:
    def __init__(self, db: Database, groq_api_key: str = "", groq_base_url: str = "", groq_model: str = "",
                 ollama_base_url: str = "", ollama_model: str = "", ollama_api_key: str = "", redis_url: str = ""):
        self.db = db
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.extractor = EntityExtractor(
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
            redis_url=redis_url,
        )
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

            # Step 1: Insert all chunks — BM25/FTS5 search works as soon as this commits.
            for i, chunk in enumerate(chunks):
                self.db.insert_chunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title", ""),
                    chunk_index=i,
                    auto_commit=False,
                )
            self.db.commit()

            # Mark ready now — users can start querying immediately via BM25.
            # Entity extraction below improves graph-based retrieval but is not required for Q&A.
            self.db.update_document_status(doc_id, "ready")

            # Step 2: Extract entities in batches of 10 chunks per LLM call, 2 concurrent batches.
            # Batch size 10 halves LLM calls vs 5; 2 workers avoids Groq RPM exhaustion.
            try:
                BATCH_SIZE = 10
                batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
                batch_texts = [[c["content"] for c in batch] for batch in batches]

                extractions = []
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(self.extractor.extract_batch, bt, doc_id) for bt in batch_texts]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            extractions.append(result)

                # Step 3: Build graph from extractions
                for extraction in extractions:
                    if extraction:
                        self.graph_builder.build_graph(extraction, document_id=doc_id)
            except Exception as e:
                print(f"[pipeline] entity extraction failed for doc {doc_id}, Q&A still works: {e}")

            return doc_id

        except Exception:
            self.db.rollback()
            self.db.update_document_status(doc_id, "error")
            raise
