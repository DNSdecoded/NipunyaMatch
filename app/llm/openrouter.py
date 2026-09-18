from typing import Any

import httpx

from app.llm.types import LLMResponse, ProviderError


class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str, http: httpx.AsyncClient | None = None) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._http = http or httpx.AsyncClient(timeout=60)

    async def generate(self, model: str, prompt: str, json_schema: dict[str, Any]) -> LLMResponse:
        body = {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            },
        }
        try:
            resp = await self._http.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json=body,
            )
        except httpx.HTTPError as e:
            raise ProviderError(503, str(e)) from e
        if resp.status_code != 200:
            ra = resp.headers.get("retry-after")
            raise ProviderError(resp.status_code, resp.text[:200], float(ra) if ra else None)
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResponse(
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
