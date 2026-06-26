import fitz
from pathlib import Path


class PDFParser:
    def extract_text(self, pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        pages = []
        full_text = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({"page_num": page_num + 1, "text": text})
            full_text.append(text)

        doc.close()

        return {
            "text": "\n\n".join(full_text),
            "pages": pages,
            "page_count": len(pages),
        }
