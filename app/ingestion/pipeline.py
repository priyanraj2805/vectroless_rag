from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.storage.database import Database
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import TextChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.graph_builder import GraphBuilder
from app.ingestion.tika_detector import TikaDetector
from app.ingestion.document_router import DocumentRouter
from app.ingestion.docling_parser import DoclingParser


class IngestionPipeline:
    def __init__(self, db: Database, groq_api_key: str = "", groq_base_url: str = "", groq_model: str = "",
                 ollama_base_url: str = "", ollama_model: str = "", ollama_api_key: str = "", redis_url: str = ""):
        self.db = db
        # Legacy parser — kept as fallback when Docling is unavailable
        self.legacy_parser = PDFParser()
        self.chunker = TextChunker()
        self.extractor = EntityExtractor(
            groq_api_key=groq_api_key, groq_base_url=groq_base_url, groq_model=groq_model,
            ollama_base_url=ollama_base_url, ollama_model=ollama_model, ollama_api_key=ollama_api_key,
            redis_url=redis_url,
        )
        self.graph_builder = GraphBuilder(db)

        # New document intelligence layer
        self.tika = TikaDetector()
        self.router = DocumentRouter()
        self.docling = DoclingParser()

    def ingest(self, file_path: str, doc_id: int = None) -> int:
        filename = Path(file_path).name

        # ── Stage 1: Tika — detect file type and extract metadata ───────────
        detection = self.tika.detect(file_path)
        mime_type = detection["mime_type"]
        meta = detection["metadata"]
        routing = self.router.route(mime_type)

        print(f"[pipeline] {filename} → {mime_type} ({routing['category']})")

        if not routing["supported"]:
            if doc_id:
                self.db.update_document_status(doc_id, "error")
            raise ValueError(f"Unsupported file type: {mime_type} for '{filename}'")

        if doc_id is None:
            # No pre-created row — insert fresh with all metadata
            doc_id = self.db.insert_document(
                filename=filename,
                mime_type=mime_type,
                content_type=routing["category"],
                author=meta.get("author"),
                document_title=meta.get("title"),
                language=meta.get("language"),
            )
        else:
            # Pre-created as 'queued' in upload.py — update with Tika metadata and mark active
            self.db.execute(
                "UPDATE documents SET mime_type=?, content_type=?, author=?, document_title=?, language=?, status='processing' WHERE id=?",
                (mime_type, routing["category"], meta.get("author"), meta.get("title"), meta.get("language"), doc_id),
            )
            self.db.commit()

        try:
            chunks = self._extract_chunks(file_path, routing, doc_id, meta)

            if not chunks:
                raise ValueError(f"No text could be extracted from '{filename}'")

            # ── Stage 3: Insert all chunks — BM25 search works immediately ──
            for i, chunk in enumerate(chunks):
                self.db.insert_chunk(
                    document_id=doc_id,
                    content=chunk["content"],
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title", ""),
                    chunk_index=i,
                    chunk_type=chunk.get("chunk_type", "text"),
                    auto_commit=False,
                )
            self.db.commit()

            # Mark ready now — users can query immediately via BM25
            self.db.update_document_status(doc_id, "ready")

            # ── Stage 4: Entity extraction in background (optional) ──────────
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

    def _extract_chunks(self, file_path: str, routing: dict, doc_id: int, meta: dict) -> list:
        """
        Stage 2: Extract structured chunks.

        Path A — Docling available and file type is supported:
          Docling parses the document into structured elements (headings,
          paragraphs, tables, figures), then the structured chunker converts
          them into high-quality, context-aware chunks.

        Path B — Docling not available (or plain text file):
          Falls back to the legacy PyMuPDF parser + regex-based chunker.
          Only works for PDFs; non-PDF formats will raise an error.
        """
        category = routing["category"]

        if self.docling.available and routing["use_docling"]:
            print(f"[pipeline] Using Docling structured parser for {category}")
            elements = self.docling.parse(file_path)
            chunks = self.chunker.chunk_from_structured(elements, doc_id)
            print(f"[pipeline] Docling produced {len(chunks)} chunks "
                  f"({sum(1 for c in chunks if c.get('chunk_type') == 'table')} tables, "
                  f"{sum(1 for c in chunks if c.get('chunk_type') == 'figure')} figures)")
            return chunks

        # Fallback: legacy PDF path
        if category == "pdf":
            print(f"[pipeline] Falling back to PyMuPDF parser (Docling unavailable)")
            parsed = self.legacy_parser.extract_text(file_path)
            if parsed.get("page_count"):
                self.db.execute(
                    "UPDATE documents SET page_count = ? WHERE id = ?",
                    (parsed["page_count"], doc_id),
                )
                self.db.commit()
            return self.chunker.chunk_from_pages(parsed["pages"], doc_id)

        raise ValueError(
            f"Cannot extract text from '{category}' file: "
            f"install Docling (pip install docling) to enable support for this format."
        )
