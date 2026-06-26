import re
from typing import List, Dict


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
