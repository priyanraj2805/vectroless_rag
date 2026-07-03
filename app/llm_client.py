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
                resp = provider["client"].chat.completions.create(
                    **{**kwargs, "model": provider["model"]}
                )
                # Treat None/empty content as provider failure and try next
                if not resp.choices or not resp.choices[0].message.content:
                    print(f"[{provider['name']}] empty response, switching to next provider...")
                    last_err = RuntimeError(f"{provider['name']} returned empty response")
                    continue
                return resp
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
    """Tries OpenRouter first, falls back to OpenCode Zen on rate limit or credit errors."""

    def __init__(self, openrouter_api_key: str = "", opencode_api_key: str = ""):
        providers = []
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
            raise ValueError("At least one of openrouter_api_key or opencode_api_key must be set")
        self.chat = _Chat(providers)
