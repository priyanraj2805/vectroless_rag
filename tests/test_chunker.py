from app.ingestion.chunker import TextChunker


def test_chunk_by_headings():
    chunker = TextChunker()
    text = """# Introduction
This is the intro section.

# Methods
This is the methods section.

# Results
This is the results section."""

    chunks = chunker.chunk(text, document_id=1)
    assert len(chunks) == 3
    assert chunks[0]["section_title"] == "Introduction"
    assert chunks[1]["section_title"] == "Methods"
    assert chunks[2]["section_title"] == "Results"


def test_chunk_preserves_page_number():
    chunker = TextChunker()
    pages = [
        {"page_num": 1, "text": "# Chapter 1\nFirst chapter content."},
        {"page_num": 2, "text": "# Chapter 2\nSecond chapter content."},
    ]

    chunks = chunker.chunk_from_pages(pages, document_id=1)
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2


def test_fallback_to_paragraphs():
    chunker = TextChunker()
    text = """This is paragraph one with some content.

This is paragraph two with more content.

This is paragraph three with even more content."""

    chunks = chunker.chunk(text, document_id=1)
    assert len(chunks) >= 1
