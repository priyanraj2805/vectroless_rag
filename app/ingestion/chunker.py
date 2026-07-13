import re
from typing import List, Dict

MAX_CHUNK_CHARS = 1500


class TextChunker:
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    NUMBERED_PATTERN = re.compile(r"^(\d+\.?\d*\.?\d*)\s+(.+)$", re.MULTILINE)

    def chunk(self, text: str, document_id: int) -> List[Dict]:
        sections = self._split_by_headings(text)
        if len(sections) <= 1:
            sections = self._split_by_paragraphs(text)

        chunks = []
        for section in sections:
            if section["content"].strip():
                chunks.append({
                    "document_id": document_id,
                    "content": section["content"].strip(),
                    "section_title": section.get("title", ""),
                    "page_number": None,
                })
        return chunks

    def chunk_from_structured(self, elements: List[Dict], document_id: int) -> List[Dict]:
        """
        Converts Docling structured elements into database-ready chunks.

        Strategy:
        - Paragraphs accumulate until MAX_CHUNK_CHARS, then flush as one chunk.
        - Each table is always its own chunk — never merged into text.
        - Each figure caption is always its own chunk.
        - Every chunk carries the section_title from the nearest heading above it.
        - Page number comes from the first element that contributed to the chunk.
        """
        chunks: List[Dict] = []
        current_section = ""
        buffer_text = ""
        buffer_page = 1

        def flush_buffer():
            nonlocal buffer_text, buffer_page
            if buffer_text.strip():
                chunks.append({
                    "document_id": document_id,
                    "content": buffer_text.strip(),
                    "section_title": current_section,
                    "page_number": buffer_page,
                    "chunk_type": "text",
                })
            buffer_text = ""

        for el in elements:
            el_type = el.get("type")
            text = el.get("text", "").strip()
            page = el.get("page", 1)

            if el_type == "heading":
                flush_buffer()
                current_section = text
                buffer_page = page

            elif el_type == "paragraph":
                if not buffer_text:
                    buffer_page = page
                if len(buffer_text) + len(text) > MAX_CHUNK_CHARS:
                    flush_buffer()
                    buffer_page = page
                buffer_text += ("\n\n" + text if buffer_text else text)

            elif el_type == "table":
                flush_buffer()
                chunks.append({
                    "document_id": document_id,
                    "content": text,
                    "section_title": current_section,
                    "page_number": page,
                    "chunk_type": "table",
                })

            elif el_type == "figure":
                flush_buffer()
                chunks.append({
                    "document_id": document_id,
                    "content": text,
                    "section_title": current_section,
                    "page_number": page,
                    "chunk_type": "figure",
                })

        flush_buffer()
        return chunks

    def chunk_from_pages(self, pages: List[Dict], document_id: int) -> List[Dict]:
        chunks = []
        for page in pages:
            page_chunks = self.chunk(page["text"], document_id)
            for chunk in page_chunks:
                chunk["page_number"] = page["page_num"]
            chunks.extend(page_chunks)
        return chunks

    def _split_by_headings(self, text: str) -> List[Dict]:
        headings = []
        for match in self.HEADING_PATTERN.finditer(text):
            headings.append({"title": match.group(2).strip(), "start": match.start(), "end": match.end()})

        if not headings:
            for match in self.NUMBERED_PATTERN.finditer(text):
                headings.append({"title": match.group(2).strip(), "start": match.start(), "end": match.end()})

        if not headings:
            return [{"content": text, "title": ""}]

        sections = []
        if headings[0]["start"] > 0:
            pre_content = text[:headings[0]["start"]].strip()
            if pre_content:
                sections.append({"content": pre_content, "title": ""})

        for i, heading in enumerate(headings):
            content_start = heading["end"]
            content_end = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
            content = text[content_start:content_end].strip()
            sections.append({"content": content, "title": heading["title"]})

        return sections

    def _split_by_paragraphs(self, text: str) -> List[Dict]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return [{"content": text, "title": ""}]

        merged = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > 2000:
                if current:
                    merged.append({"content": current, "title": ""})
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current:
            merged.append({"content": current, "title": ""})

        return merged
