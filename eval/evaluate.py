"""
RAG evaluation with custom scoring — Hallucination, Answer Relevance, Context Precision.
Uses Groq API directly (no OpenAI key needed). Displays results as percentages.
"""
import json
import time
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.storage.database import Database
from app.query.executor import QueryExecutor
from app.query.synthesizer import AnswerSynthesizer
from app.llm_client import OllamaClient

TEST_FILE = Path(__file__).parent / "test_cases.jsonl"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"


def load_test_cases() -> list:
    cases = []
    if not TEST_FILE.exists():
        print(f"[eval] No test file at {TEST_FILE}")
        return cases
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_plan(question: str, num_docs: int = 1) -> dict:
    stop_words = {
        'what', 'are', 'is', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'and', 'or', 'but', 'not',
        'do', 'does', 'did', 'have', 'has', 'had', 'can', 'could', 'will',
        'would', 'should', 'may', 'might', 'shall', 'this', 'that', 'these',
        'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
        'about', 'tell', 'me', 'list', 'all', 'some', 'which', 'who', 'when',
        'where', 'how', 'why', 'give', 'show', 'describe', 'explain',
    }
    words = question.lower().replace('?', '').replace('.', '').replace(',', '').split()
    search_terms = [w for w in words if w not in stop_words and len(w) > 2]
    return {
        "search_terms": search_terms,
        "entity_types": ["person", "organization", "concept", "technology", "project", "date", "metric"],
        "traverse_edges": ["relates_to", "mentions"],
        "max_results": max(10, num_docs * 5),
        "num_docs": num_docs,
        "query_type": "factual",
    }


def query_rag_fast(db, executor, synthesizer, question, document_ids=None):
    plan = build_plan(question, num_docs=len(document_ids) if document_ids else 1)
    context = executor.execute(plan, document_ids=document_ids, question=question)
    context["nodes"] = []
    context["doc_groups"] = {}
    result = synthesizer.synthesize(question, context, has_documents=True)
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "entities_found": 0,
        "chunks_used": result.get("chunk_count", 0),
    }


class LLMScorer:
    """Scores RAG outputs using Groq API with simple LLM prompts."""

    def __init__(self):
        self.client = OllamaClient(
            base_url=settings.groq_base_url, model=settings.groq_model, api_key=settings.groq_api_key,
            fallback_base_url=settings.opencode_base_url, fallback_model=settings.opencode_model, fallback_api_key=settings.opencode_api_key,
        )

    def _ask(self, system: str, user: str, max_tokens: int = 150) -> str:
        try:
            resp = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  [llm error] {e}")
            return ""

    def _parse_score(self, text: str) -> float | None:
        """Extract a 0-100 number from LLM response — the LAST number found, since
        judge models sometimes reason out loud before landing on a verdict (see
        app.api.chat.LLMScorer._parse_score for the full rationale)."""
        import re
        nums = re.findall(r'(\d+(?:\.\d+)?)', text)
        if nums:
            val = float(nums[-1])
            if val <= 1.0:
                return val  # Already 0-1
            if val <= 100:
                return val / 100.0
        return None

    def _build_judge_context(self, context: list[str], max_chars: int = 10000) -> str:
        """Join retrieved chunks by character budget, not a fixed chunk count — see
        app.api.chat.LLMScorer._build_judge_context for why (chunk order != relevance
        order for hierarchical retrieval, so a naive [:5] slice can miss the chunk that
        actually supports the answer and cause false 100% hallucination scores)."""
        parts = []
        total = 0
        for text in context:
            if total >= max_chars:
                break
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    def hallucination(self, question: str, answer: str, context: list[str]) -> float | None:
        """Score: 0 = no hallucination, 1 = fully hallucinated."""
        ctx = self._build_judge_context(context)
        prompt = f"""You are a hallucination detector. Rate how much of the answer is supported by the context.

Context:
{ctx}

Question: {question}
Answer: {answer}

Rate from 0 to 100:
- 0 = Answer is fully supported by context
- 50 = Partially supported
- 100 = Completely made up / hallucinated

Reply with ONLY a number (0-100)."""
        resp = self._ask("You are a hallucination detector. Reply with only a number.", prompt)
        return self._parse_score(resp)

    def answer_relevance(self, question: str, answer: str) -> float | None:
        """Score: 0 = irrelevant, 1 = perfectly relevant."""
        prompt = f"""Rate how well this answer addresses the question. Be strict.

Question: {question}
Answer: {answer}

Rate from 0 to 100:
- 0 = Completely irrelevant
- 50 = Partially relevant
- 100 = Perfectly answers the question

Reply with ONLY a number (0-100)."""
        resp = self._ask("You answer relevance judge. Reply with only a number.", prompt)
        return self._parse_score(resp)

    def context_precision(self, question: str, context: list[str]) -> float | None:
        """Score: 0 = irrelevant context, 1 = all context is relevant."""
        ctx = self._build_judge_context(context)
        prompt = f"""Rate how relevant the retrieved context is to the question.

Question: {question}
Retrieved Context:
{ctx}

Rate from 0 to 100:
- 0 = Context is completely irrelevant
- 50 = Some relevant parts
- 100 = All context is directly relevant

Reply with ONLY a number (0-100)."""
        resp = self._ask("You are a context relevance judge. Reply with only a number.", prompt)
        return self._parse_score(resp)


def run_evaluation():
    cases = load_test_cases()
    if not cases:
        print("[eval] No test cases. Exiting.")
        return

    print(f"\n[eval] Running evaluation on {len(cases)} test cases...\n")

    db = Database(settings.database_path)
    executor = QueryExecutor(db)
    synthesizer = AnswerSynthesizer(
        groq_base_url=settings.groq_base_url, groq_model=settings.groq_model, groq_api_key=settings.groq_api_key,
        fallback_base_url=settings.opencode_base_url, fallback_model=settings.opencode_model, fallback_api_key=settings.opencode_api_key,
    )

    print("[eval] Starting evaluation...")
    t0 = time.time()

    scorer = LLMScorer()
    all_results = []
    metric_totals = {"hallucination": [], "answer_relevance": [], "context_precision": []}

    for i, case in enumerate(cases):
        question = case["question"]
        document_ids = case.get("document_ids")
        expected = case.get("expected_answer", "")

        print(f"[eval] {i+1}/{len(cases)}: {question}")
        t0 = time.time()

        response = query_rag_fast(db, executor, synthesizer, question, document_ids)
        elapsed = time.time() - t0
        answer = response["answer"]

        if not answer:
            print(f"  -> No answer. Skipping.\n")
            continue

        safe = answer[:100].encode('ascii', 'replace').decode()
        print(f"  -> Answer ({elapsed:.1f}s): {safe}...")

        # Get chunks for context scoring
        plan = build_plan(question)
        chunks = executor.execute(plan, document_ids=document_ids, question=question).get("chunks", [])
        context_texts = [c[1] for c in chunks if len(c) > 1]

        # Score with LLM
        h = scorer.hallucination(question, answer, context_texts)
        ar = scorer.answer_relevance(question, answer)
        cp = scorer.context_precision(question, context_texts)

        h_pct = f"{h*100:.0f}%" if h is not None else "N/A"
        ar_pct = f"{ar*100:.0f}%" if ar is not None else "N/A"
        cp_pct = f"{cp*100:.0f}%" if cp is not None else "N/A"
        print(f"  -> Hallucination: {h_pct}  |  Relevance: {ar_pct}  |  Precision: {cp_pct}\n")

        if h is not None: metric_totals["hallucination"].append(h)
        if ar is not None: metric_totals["answer_relevance"].append(ar)
        if cp is not None: metric_totals["context_precision"].append(cp)

        all_results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "sources_count": response["chunks_used"],
            "scores": {
                "hallucination": round(h, 4) if h is not None else None,
                "answer_relevance": round(ar, 4) if ar is not None else None,
                "context_precision": round(cp, 4) if cp is not None else None,
            },
        })

    db.close()

    # ── Summary Report ──
    print("=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print(f"  Test cases: {len(all_results)} / {len(cases)}")
    print()

    for metric in ["hallucination", "answer_relevance", "context_precision"]:
        vals = metric_totals[metric]
        label = metric.replace("_", " ").title()
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {label:.<35} {avg*100:.1f}%")
        else:
            print(f"  {label:.<35} No scores")

    print()
    print("-" * 65)
    print(f"  {'Question':<40} {'Halluc':>7} {'Relev':>7} {'Prec':>7}")
    print("-" * 65)
    for r in all_results:
        s = r["scores"]
        h = f"{s['hallucination']*100:.0f}%" if s.get("hallucination") is not None else "-"
        ar = f"{s['answer_relevance']*100:.0f}%" if s.get("answer_relevance") is not None else "-"
        cp = f"{s['context_precision']*100:.0f}%" if s.get("context_precision") is not None else "-"
        q = r["question"][:38]
        print(f"  {q:<40} {h:>7} {ar:>7} {cp:>7}")
    print("=" * 65)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_evaluation()
