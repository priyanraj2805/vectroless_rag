import openai


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
                # Always try the next provider — any error (rate limit, quota, network, etc.)
                print(f"[{provider['name']}] error: {e!s:.120}, trying next provider...")
                last_err = e
        raise last_err or RuntimeError("All LLM providers exhausted")


class _Chat:
    def __init__(self, providers):
        self.completions = _Completions(providers)


class FallbackLLMClient:
    """Answer LLMs: Groq (primary) → Ollama (fallback)."""

    def __init__(self, groq_api_key: str = "", groq_base_url: str = "", groq_model: str = "",
                 ollama_base_url: str = "", ollama_model: str = "", ollama_api_key: str = ""):
        providers = []
        if groq_api_key:
            providers.append({
                "name": "groq",
                "client": openai.OpenAI(
                    api_key=groq_api_key,
                    base_url=groq_base_url or "https://api.groq.com/openai/v1",
                ),
                "model": groq_model or "llama-3.1-8b-instant",
            })
        if ollama_base_url:
            providers.append({
                "name": "ollama",
                "client": openai.OpenAI(
                    api_key=ollama_api_key or "ollama",
                    base_url=ollama_base_url,
                ),
                "model": ollama_model or "gemma4:31b",
            })
        if not providers:
            raise ValueError("At least one of groq_api_key or ollama_base_url must be set")
        self.chat = _Chat(providers)


class OpenCodeClient:
    """OpenCode — used only for eval/judge scoring."""

    def __init__(self, api_key: str = "", base_url: str = "", model: str = ""):
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://opencode.ai/zen/v1",
        )
        self.chat = _Chat([{
            "name": "opencode",
            "client": client,
            "model": model or "nemotron-3-ultra-free",
        }])
