"""Shared query analysis: keyword/phrase extraction and lightweight intent detection.

Used by both the legacy plan builder (app/api/chat.py) and the hierarchical
retriever (app/query/hierarchical_retriever.py) so the two paths agree on what
counts as a "meaningful term" in a question.
"""
from typing import Dict, List

STOP_WORDS = {
    'what', 'are', 'is', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at',
    'to', 'for', 'of', 'with', 'by', 'from', 'and', 'or', 'but', 'not',
    'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will',
    'would', 'should', 'may', 'might', 'shall', 'this', 'that', 'these',
    'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
    'about', 'tell', 'me', 'list', 'all', 'some', 'which', 'who', 'when',
    'where', 'how', 'why', 'give', 'show', 'describe', 'explain',
}

COMPARISON_MARKERS = {'difference', 'compare', 'comparison', 'versus', 'vs', 'contrast', 'better', 'worse'}
DEFINITION_MARKERS = {'define', 'definition', 'meaning', 'what is', 'what are'}
SUMMARY_MARKERS = {
    'summary', 'summarize', 'summarise', 'overview', 'brief',
    'tell me about all', 'give me all', 'list all', 'describe all',
    'what are these', 'what is in', 'what do these',
}


def extract_terms(question: str) -> Dict[str, List[str]]:
    """Extract meaningful keywords and 2/3-word phrases from a question."""
    words = question.lower().replace('?', '').replace('.', '').replace(',', '').split()
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]

    phrases = []
    for i in range(len(meaningful) - 1):
        phrases.append(f'{meaningful[i]} {meaningful[i + 1]}')
    for i in range(len(meaningful) - 2):
        phrases.append(f'{meaningful[i]} {meaningful[i + 1]} {meaningful[i + 2]}')

    return {"keywords": meaningful, "phrases": phrases, "search_terms": phrases + meaningful}


def detect_intent(question: str) -> str:
    """Cheap heuristic intent classification — no LLM call, keeps retrieval fast."""
    q = question.lower().strip()
    if any(kw in q for kw in SUMMARY_MARKERS):
        return "summary"
    if any(kw in q for kw in COMPARISON_MARKERS):
        return "comparison"
    if any(kw in q for kw in DEFINITION_MARKERS):
        return "definition"
    return "factual"


def build_plan_from_question(question: str, num_docs: int = 1) -> Dict:
    """Build a search plan directly from the question — no LLM call needed.

    Shared by the legacy executor path and the hierarchical retriever so both
    produce the same search_terms/entity_types/traverse_edges shape.
    """
    terms = extract_terms(question)
    max_results = min(30, max(10, num_docs * 5))

    return {
        "search_terms": terms["search_terms"],
        "keywords": terms["keywords"],
        "entity_types": ["person", "organization", "concept", "technology", "project", "date", "metric"],
        "traverse_edges": ["relates_to", "mentions"],
        "max_results": max_results,
        "num_docs": num_docs,
        "query_type": detect_intent(question),
    }
