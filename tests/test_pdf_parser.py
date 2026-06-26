import tempfile
import os
from app.ingestion.pdf_parser import PDFParser


def test_extract_text_from_simple_pdf():
    parser = PDFParser()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>>>endobj\n4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000266 00000 n \ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n340\n%%EOF")
        f.flush()
        result = parser.extract_text(f.name)
        assert "pages" in result
        assert result["page_count"] >= 1
        assert len(result["text"]) >= 0
    os.unlink(f.name)
