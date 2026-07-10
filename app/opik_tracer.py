import os
from dotenv import load_dotenv

load_dotenv()

OPIK_API_KEY = os.environ.get("OPIK_API_KEY", "")
OPIK_WORKSPACE = os.environ.get("OPIK_WORKSPACE", "")
OPIK_PROJECT = os.environ.get("OPIK_PROJECT", "vectorless_rag")

if OPIK_API_KEY:
    import opik
    from opik import opik_context

    opik.configure(
        api_key=OPIK_API_KEY,
        workspace=OPIK_WORKSPACE or None,
        project_name=OPIK_PROJECT,
        use_local=False,
        force=True,   # suppress "already configured" warning
    )

    def track(func):
        """Decorator that applies @opik.track with project name."""
        return opik.track(func, project_name=OPIK_PROJECT)

    def get_current_trace_id():
        """Return the trace id of the currently-executing @track'd function, or None.

        Must be called from inside the dynamic extent of a @track-decorated
        function (e.g. AnswerSynthesizer.synthesize) — outside of that, there
        is no active trace and this safely returns None.
        """
        try:
            trace_data = opik_context.get_current_trace_data()
            return trace_data.id if trace_data else None
        except Exception:
            return None

    _opik_client = None

    def _get_opik_client():
        """Lazily create (and cache) a single Opik() client for feedback-score logging."""
        global _opik_client
        if _opik_client is None:
            try:
                _opik_client = opik.Opik(project_name=OPIK_PROJECT)
            except Exception as e:
                print(f"[opik] Failed to create Opik client: {e}")
                _opik_client = False  # sentinel: don't retry every call
        return _opik_client or None

    def log_feedback_scores(trace_id, scores: dict):
        """Attach LLM-judge scores (hallucination, answer_relevance, context_precision, ...)
        to a specific trace, after the fact — i.e. once synthesize()'s own trace has
        already closed and scoring has happened in the calling route.

        Never raises: any failure (missing trace_id, no scores, SDK mismatch, network
        error) is logged and swallowed so a broken Opik integration can never break
        the actual chat/eval response.
        """
        if not trace_id or not scores:
            return
        feedback_scores = [
            {"id": trace_id, "name": name, "value": float(value)}
            for name, value in scores.items()
            if value is not None
        ]
        if not feedback_scores:
            return
        try:
            client = _get_opik_client()
            if client is None:
                return
            # SDK versions differ slightly on the batch method's parameter name —
            # try the documented forms before falling back to per-score logging.
            try:
                client.log_traces_feedback_scores(feedback_scores)
            except TypeError:
                try:
                    client.log_traces_feedback_scores(traces_feedback_scores=feedback_scores)
                except (TypeError, AttributeError):
                    for fs in feedback_scores:
                        client.log_trace_feedback_score(
                            id=fs["id"], name=fs["name"], value=fs["value"]
                        )
        except Exception as e:
            print(f"[opik] Failed to log feedback scores: {e}")
else:
    def track(func):
        """No-op when Opik isn't configured (no OPIK_API_KEY set)."""
        return func

    def get_current_trace_id():
        """No-op when Opik isn't configured — no trace ever exists."""
        return None

    def log_feedback_scores(trace_id, scores: dict):
        """No-op when Opik isn't configured (no OPIK_API_KEY set)."""
        pass
