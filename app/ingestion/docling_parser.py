from pathlib import Path
from typing import List, Dict


class DoclingParser:
    """
    Wraps IBM Docling's DocumentConverter to produce a structured list of
    document elements (headings, paragraphs, tables, figures) from any
    supported file format.

    Falls back gracefully if Docling is not installed.
    """

    def __init__(self):
        self._available = False
        self._converter = None
        try:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            self._available = True
            print("[docling] Document converter ready")
        except Exception as e:
            print(f"[docling] Not available ({e}) — structured parsing disabled")

    @property
    def available(self) -> bool:
        return self._available

    def parse(self, file_path: str) -> List[Dict]:
        """
        Returns a flat list of structured elements, in document reading order:

        [
          {"type": "heading",   "level": 1, "text": "Introduction", "page": 1, "section": ""},
          {"type": "paragraph", "text": "...",                       "page": 1, "section": "Introduction"},
          {"type": "table",     "text": "| Col1 | Col2 |\\n...",     "page": 2, "section": "Results"},
          {"type": "figure",    "text": "Figure 3: Revenue chart",   "page": 3, "section": "Results"},
        ]

        The 'section' field on each element is the title of the most recent
        heading seen before that element — enables chunker to tag every chunk
        with its correct section context without re-parsing heading structure.
        """
        if not self._available:
            raise RuntimeError("Docling is not installed. Run: pip install docling")

        result = self._converter.convert(str(file_path))
        doc = result.document

        elements: List[Dict] = []
        current_section = ""

        # ── Text items (headings + paragraphs) ──────────────────────────────
        try:
            from docling.datamodel.base_models import ItemAndImagesSelf
        except ImportError:
            pass

        for text_item in doc.texts:
            label = str(text_item.label).upper()
            text = (text_item.text or "").strip()
            page = self._page(text_item)

            if not text:
                continue

            if "SECTION_HEADER" in label or "TITLE" in label or "HEADING" in label:
                level = getattr(text_item, "level", 1) or 1
                current_section = text
                elements.append({
                    "type": "heading",
                    "level": int(level),
                    "text": text,
                    "page": page,
                    "section": current_section,
                })

            elif "LIST" in label:
                elements.append({
                    "type": "paragraph",
                    "text": f"• {text}",
                    "page": page,
                    "section": current_section,
                })

            else:
                if len(text) >= 20:
                    elements.append({
                        "type": "paragraph",
                        "text": text,
                        "page": page,
                        "section": current_section,
                    })

        # ── Tables ──────────────────────────────────────────────────────────
        for table_item in doc.tables:
            page = self._page(table_item)
            md = self._table_to_markdown(table_item)
            if md:
                elements.append({
                    "type": "table",
                    "text": md,
                    "page": page,
                    "section": current_section,
                })

        # ── Figures / pictures ───────────────────────────────────────────────
        for picture_item in doc.pictures:
            page = self._page(picture_item)
            caption = self._caption(picture_item)
            if caption:
                elements.append({
                    "type": "figure",
                    "text": f"Figure: {caption}",
                    "page": page,
                    "section": current_section,
                })

        return elements

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _page(self, item) -> int:
        try:
            prov = getattr(item, "prov", None)
            if prov:
                return prov[0].page_no
        except Exception:
            pass
        return 1

    def _caption(self, item) -> str:
        try:
            captions = getattr(item, "captions", None)
            if captions:
                return captions[0].text.strip()
        except Exception:
            pass
        return ""

    def _table_to_markdown(self, table_item) -> str:
        try:
            df = table_item.export_to_dataframe()
            if df is not None and not df.empty:
                headers = list(df.columns)
                lines = [
                    "| " + " | ".join(str(h) for h in headers) + " |",
                    "|" + "|".join(" --- " for _ in headers) + "|",
                ]
                for _, row in df.iterrows():
                    lines.append("| " + " | ".join(str(v) for v in row) + " |")
                return "\n".join(lines)
        except Exception:
            pass

        try:
            return table_item.export_to_markdown()
        except Exception:
            return ""
