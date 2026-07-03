import openai


def _is_fallback_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ["429", "402", "rate_limit", "insufficient", "credits", "quota"])


class _Completions:
    def __init__(self, providers):
        self._providers = providers

    def create(self, **kwargs):
        last_err = None
        for provider in self._providers:
            try:
                return provider["client"].chat.completions.create(
                    **{**kwargs, "model": provider["model"]}
                )
            except Exception as e:
                if _is_fallback_error(e):
                    print(f"[{provider['name']}] limit hit, switching to next provider...")
                    last_err = e
                else:
                    raise
        raise last_err or RuntimeError("All LLM providers exhausted")


class _Chat:
    def __init__(self, providers):
        self.completions = _Completions(providers)


class FallbackLLMClient:
    """Tries Groq → OpenRouter → OpenCode Zen in order, falling back on rate limit or credit errors."""

    def __init__(self, groq_api_key: str = "", openrouter_api_key: str = "", opencode_api_key: str = ""):
        providers = []
        if groq_api_key:
            providers.append({
                "name": "groq",
                "client": openai.OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1"),
                "model": "llama-3.1-8b-instant",
            })
        if openrouter_api_key:
            providers.append({
                "name": "openrouter",
                "client": openai.OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
                "model": "meta-llama/llama-3.1-8b-instruct",
            })
        if opencode_api_key:
            providers.append({
                "name": "opencode",
                "client": openai.OpenAI(api_key=opencode_api_key, base_url="https://opencode.ai/zen/v1"),
                "model": "nemotron-3-ultra-free",
            })
        if not providers:
            raise ValueError("At least one API key must be set")
        self.chat = _Chat(providers)
